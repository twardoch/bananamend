# this_file: bananamendy/src/bananamendy/reference.py
"""A forward pass in numpy, for calibration only.

The Rust engine is the fast path, and it is the only path for a user. This file
exists because good quantization needs to know what each matrix multiplication
actually receives. The engine does not give those numbers out, and a browser
does not need them, so the calibration pass lives here instead.

The steps follow `crates/bananamendr/src/model.rs` exactly:

1. Multiply the embedding of the token by the square root of the hidden size.
2. For each block: RMSNorm, then the projections for query, key and value; then
   per-head RMSNorm on query and key; then interleaved rotary embedding; then
   attention with a causal mask; then the output projection and a residual add.
3. RMSNorm again, then a gate projection and an up projection, a SwiGLU, a down
   projection, and a residual add.
4. RMSNorm at the end, and then the output matrix.

`collect_inputs` returns, for each matrix, the rows that it received. A
quantizer uses those rows to make the error of the *output* small, instead of the
error of the weights.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

# The names of the matrices in one block, and the activation that each one reads.
BLOCK_MATRICES = (
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.o_proj",
    "mlp.w_gate",
    "mlp.w_up",
    "mlp.w_down",
)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """RMSNorm over the last axis, in 32-bit floats, as the engine does it."""
    mean_square = np.mean(np.square(x), axis=-1, keepdims=True)
    return (x / np.sqrt(mean_square + eps)) * weight


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def rope_tables(head_dim: int, positions: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """The cosine and sine tables of the interleaved rotary embedding."""
    half = head_dim // 2
    index = np.arange(half, dtype=np.float64)
    freq = 1.0 / np.power(theta, (2.0 * index) / head_dim)
    angle = np.arange(positions, dtype=np.float64)[:, None] * freq[None, :]
    return np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)


def apply_rope(vectors: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Rotates pairs `(2i, 2i + 1)`. The pairing is interleaved, not split."""
    even = vectors[..., 0::2]
    odd = vectors[..., 1::2]
    rotated = np.empty_like(vectors)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    return rotated


@dataclass
class Reference:
    """A checkpoint that this file can run."""

    config: dict
    tensors: dict[str, np.ndarray]

    @classmethod
    def load(cls, path: Path | str) -> Reference:
        path = Path(path)
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        tensors = load_file(path / "model.safetensors")
        return cls(config=config, tensors={k: v.astype(np.float32) for k, v in tensors.items()})

    # ---- shapes ---------------------------------------------------------

    @property
    def layers(self) -> int:
        return int(self.config["num_hidden_layers"])

    @property
    def hidden(self) -> int:
        return int(self.config["hidden_size"])

    @property
    def head_dim(self) -> int:
        return int(self.config["head_dim"])

    @property
    def heads(self) -> int:
        return int(self.config["num_attention_heads"])

    @property
    def kv_heads(self) -> int:
        return int(self.config["num_key_value_heads"])

    @property
    def eps(self) -> float:
        return float(self.config["rms_norm_eps"])

    def matrix(self, name: str) -> np.ndarray:
        return self.tensors[name]

    def output_matrix(self) -> np.ndarray:
        return self.tensors.get("lm_head.weight", self.tensors["transformer.wte.weight"])

    # ---- the forward pass ----------------------------------------------

    def forward(
        self, tokens: list[int], *, collect: dict[str, list[np.ndarray]] | None = None
    ) -> np.ndarray:
        """Runs the tokens and returns the logits of every position.

        `collect` receives the input rows of each matrix, if it is given.
        """
        cfg = self.config
        length = len(tokens)
        embedding = self.tensors["transformer.wte.weight"]
        x = embedding[np.asarray(tokens, dtype=np.int64)] * np.float32(np.sqrt(self.hidden))

        cos, sin = rope_tables(self.head_dim, length, float(cfg["rope_theta"]))
        # The mask keeps a position from reading a later position.
        mask = np.triu(np.full((length, length), -np.inf, dtype=np.float32), k=1)
        repeats = self.heads // self.kv_heads
        scale = 1.0 / np.sqrt(self.head_dim)

        def record(name: str, rows: np.ndarray) -> None:
            if collect is not None:
                collect.setdefault(name, []).append(rows.astype(np.float32, copy=True))

        for layer in range(self.layers):
            prefix = f"transformer.h.{layer}."
            normed = rms_norm(x, self.matrix(f"{prefix}ln_1.weight"), self.eps)

            record(f"{prefix}attn.q_proj.weight", normed)
            record(f"{prefix}attn.k_proj.weight", normed)
            record(f"{prefix}attn.v_proj.weight", normed)

            q = normed @ self.matrix(f"{prefix}attn.q_proj.weight").T
            k = normed @ self.matrix(f"{prefix}attn.k_proj.weight").T
            v = normed @ self.matrix(f"{prefix}attn.v_proj.weight").T

            q = q.reshape(length, self.heads, self.head_dim)
            k = k.reshape(length, self.kv_heads, self.head_dim)
            v = v.reshape(length, self.kv_heads, self.head_dim)

            # Per-head RMSNorm, and only then the rotation. The order matters.
            q = rms_norm(q, self.matrix(f"{prefix}attn.q_norm.weight"), self.eps)
            k = rms_norm(k, self.matrix(f"{prefix}attn.k_norm.weight"), self.eps)
            q = apply_rope(q, cos[:, None, :], sin[:, None, :])
            k = apply_rope(k, cos[:, None, :], sin[:, None, :])

            # Grouped-query attention: each key and value head serves several
            # query heads.
            k_full = np.repeat(k, repeats, axis=1)
            v_full = np.repeat(v, repeats, axis=1)
            scores = np.einsum("qhd,khd->hqk", q, k_full) * scale + mask[None, :, :]
            scores = scores - scores.max(axis=-1, keepdims=True)
            weights = np.exp(scores)
            weights /= weights.sum(axis=-1, keepdims=True)
            attention = np.einsum("hqk,khd->qhd", weights, v_full)
            attention = attention.reshape(length, self.heads * self.head_dim)

            record(f"{prefix}attn.o_proj.weight", attention)
            x = x + attention @ self.matrix(f"{prefix}attn.o_proj.weight").T

            normed = rms_norm(x, self.matrix(f"{prefix}ln_2.weight"), self.eps)
            record(f"{prefix}mlp.w_gate.weight", normed)
            record(f"{prefix}mlp.w_up.weight", normed)
            gate = normed @ self.matrix(f"{prefix}mlp.w_gate.weight").T
            up = normed @ self.matrix(f"{prefix}mlp.w_up.weight").T
            hidden = silu(gate) * up
            record(f"{prefix}mlp.w_down.weight", hidden)
            x = x + hidden @ self.matrix(f"{prefix}mlp.w_down.weight").T

        x = rms_norm(x, self.matrix("transformer.ln_f.weight"), self.eps)
        return x @ self.output_matrix().T

    def matrix_names(self) -> list[str]:
        """The names of the matrices that a quantizer can work on."""
        names = []
        for layer in range(self.layers):
            for part in BLOCK_MATRICES:
                names.append(f"transformer.h.{layer}.{part}.weight")
        return names


def collect_inputs(
    path: Path | str, texts: list[list[int]], *, max_rows: int = 4096
) -> dict[str, np.ndarray]:
    """Runs the calibration texts and returns the input rows of each matrix.

    `max_rows` limits the memory: the function keeps that many rows for each
    matrix, taken evenly from all of the texts.
    """
    model = Reference.load(path)
    collected: dict[str, list[np.ndarray]] = {}
    for tokens in texts:
        model.forward(tokens, collect=collected)

    result: dict[str, np.ndarray] = {}
    for name, blocks in collected.items():
        rows = np.concatenate(blocks, axis=0)
        if rows.shape[0] > max_rows:
            step = rows.shape[0] // max_rows
            rows = rows[::step][:max_rows]
        result[name] = rows
    return result
