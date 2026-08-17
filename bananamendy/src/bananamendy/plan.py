# this_file: bananamendy/src/bananamendy/plan.py
"""Which tensor becomes ternary, and which tensor becomes 8-bit.

Ternary weights are small and they lose much. Eight-bit weights are four times
larger and they lose almost nothing. The best checkpoint is therefore mixed: as
many ternary tensors as the model can carry, and 8-bit for the rest.

This file finds that mixture. The method is a measurement, and not a rule:

1. Quantize every candidate tensor to 8 bits. This is the base, and it is close
   to the float model.
2. For each candidate, replace only that tensor with its ternary form, and
   measure how much the answers of the model change. The measure is the
   Kullback-Leibler divergence on a text that the quantizer does not use.
3. Sort the candidates from the smallest change to the largest.
4. Take the candidates in that order, and keep each one ternary while the total
   change stays inside the budget. Give the rest 8 bits.

Step 2 needs a complete forward pass for each candidate, so the work is
proportional to the number of matrices. The pass is the numpy pass in
`reference.py`, which agrees with the engine to five decimal places.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .calibration import EVALUATION_TEXT
from .quantize import (
    DEFAULT_GROUP_SIZE,
    quantize_matrix,
    quantize_matrix_calibrated,
    quantize_matrix_int8,
)
from .reference import Reference, collect_inputs


@dataclass
class Candidate:
    """One matrix, and what the two grids do to it."""

    name: str
    ternary_kl: float
    ternary_error: float
    int8_error: float
    ternary_bytes: int
    int8_bytes: int
    float_bytes: int


def _log_softmax(scores: np.ndarray) -> np.ndarray:
    top = scores.max(axis=-1, keepdims=True)
    shifted = scores - top
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def kl_divergence(reference_logits: np.ndarray, candidate_logits: np.ndarray) -> float:
    """The mean Kullback-Leibler divergence over the positions of a text."""
    log_p = _log_softmax(reference_logits.astype(np.float64))
    log_q = _log_softmax(candidate_logits.astype(np.float64))
    p = np.exp(log_p)
    return float((p * (log_p - log_q)).sum(axis=-1).mean())


def _grids(
    model: Reference,
    names: list[str],
    calibration: dict[str, np.ndarray] | None,
    group_size: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, int, int]]:
    """Builds the ternary and the 8-bit form of each candidate.

    Returns, for each name, the ternary values, the 8-bit values, the number of
    bytes that the ternary form needs, and the number that the 8-bit form needs.
    """
    result: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for name in names:
        matrix = model.matrix(name)
        rows = None if calibration is None else calibration.get(name)
        if rows is not None and rows.shape[1] == matrix.shape[1] and rows.shape[0] >= 8:
            codes, scale_positive, scale_negative, ternary = quantize_matrix_calibrated(
                matrix, rows, group_size
            )
        else:
            codes, scale_positive, scale_negative, ternary = quantize_matrix(matrix, group_size)
        ternary_bytes = (codes.size + 3) // 4 + scale_positive.nbytes + scale_negative.nbytes
        int8_codes, int8_scale, int8 = quantize_matrix_int8(matrix, group_size)
        result[name] = (ternary, int8, ternary_bytes, int8_codes.nbytes + int8_scale.nbytes)
    return result


def measure(
    path: Path | str,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    calibration: dict[str, np.ndarray] | None = None,
    tokens: list[int] | None = None,
) -> tuple[list[Candidate], float]:
    """Measures every candidate. Returns the candidates and the base divergence.

    The base divergence is the divergence of the all-8-bit checkpoint. A mixture
    cannot be better than that number.
    """
    model = Reference.load(path)
    names = model.matrix_names()
    if tokens is None:
        raise ValueError("tokens are necessary; tokenize EVALUATION_TEXT with the engine")

    reference_logits = model.forward(tokens)
    grids = _grids(model, names, calibration, group_size)

    # The base: every candidate in 8 bits.
    originals = {name: model.matrix(name) for name in names}
    for name in names:
        model.tensors[name] = grids[name][1]
    base_logits = model.forward(tokens)
    base_kl = kl_divergence(reference_logits, base_logits)

    candidates: list[Candidate] = []
    for name in names:
        ternary, int8, ternary_bytes, int8_bytes = grids[name]
        model.tensors[name] = ternary
        logits = model.forward(tokens)
        model.tensors[name] = int8
        original = originals[name]
        norm = float(np.linalg.norm(original)) or 1.0
        candidates.append(
            Candidate(
                name=name,
                ternary_kl=kl_divergence(reference_logits, logits),
                ternary_error=float(np.linalg.norm(original - ternary) / norm),
                int8_error=float(np.linalg.norm(original - int8) / norm),
                ternary_bytes=ternary_bytes,
                int8_bytes=int8_bytes,
                float_bytes=original.size * 4,
            )
        )

    for name, original in originals.items():
        model.tensors[name] = original
    return candidates, base_kl


def choose(
    path: Path | str,
    *,
    kl_budget: float = 0.05,
    group_size: int = DEFAULT_GROUP_SIZE,
    calibration: dict[str, np.ndarray] | None = None,
    tokens: list[int] | None = None,
    candidates: list[Candidate] | None = None,
) -> tuple[dict[str, str], dict]:
    """Selects a grid for each candidate. Returns the plan and a report.

    `kl_budget` is the largest divergence that the result may reach on the
    measurement text. A smaller budget keeps more tensors in 8 bits.
    """
    model = Reference.load(path)
    names = model.matrix_names()
    if tokens is None:
        raise ValueError("tokens are necessary")
    if candidates is None:
        candidates, _ = measure(
            path, group_size=group_size, calibration=calibration, tokens=tokens
        )

    reference_logits = model.forward(tokens)
    grids = _grids(model, names, calibration, group_size)
    for name in names:
        model.tensors[name] = grids[name][1]

    plan = {name: "int8" for name in names}
    accepted: list[str] = []
    current = kl_divergence(reference_logits, model.forward(tokens))

    # The easiest candidates first. Each acceptance is verified, because the
    # effects of two ternary tensors are not simply the sum of the two effects.
    for candidate in sorted(candidates, key=lambda c: c.ternary_kl):
        ternary, int8, _, _ = grids[candidate.name]
        model.tensors[candidate.name] = ternary
        trial = kl_divergence(reference_logits, model.forward(tokens))
        if trial <= kl_budget:
            plan[candidate.name] = "ternary"
            accepted.append(candidate.name)
            current = trial
        else:
            model.tensors[candidate.name] = int8

    ternary_bytes = sum(
        grids[name][2] if plan[name] == "ternary" else grids[name][3] for name in names
    )
    float_bytes = sum(model.matrix(name).size * 4 for name in names)
    report = {
        "kl_budget": kl_budget,
        "kl_result": round(current, 5),
        "candidates": len(names),
        "ternary": len(accepted),
        "int8": len(names) - len(accepted),
        "matrix_float_mb": round(float_bytes / 1e6, 2),
        "matrix_result_mb": round(ternary_bytes / 1e6, 2),
        "matrix_ratio": round(float_bytes / ternary_bytes, 2) if ternary_bytes else 0.0,
        "ternary_tensors": accepted,
    }
    return plan, report


def tokenize_evaluation(engine, limit: int = 96) -> list[int]:
    """The tokens of the measurement text, with the start token in front."""
    tokens = [int(engine.config["bos_token_id"])] + list(engine.tokenize(EVALUATION_TEXT))
    return tokens[:limit]


def collect_for_layer(
    path: Path | str, texts: list[list[int]], names: set[str], max_rows: int
) -> dict[str, np.ndarray]:
    """Collects the inputs of a few matrices only.

    A complete collection over a large checkpoint needs more memory than a small
    computer has, so the quantizer works on one block at a time.
    """
    collected = collect_inputs(path, texts, max_rows=max_rows)
    return {name: rows for name, rows in collected.items() if name in names}


def summarize(candidates: list[Candidate], base_kl: float, limit: int = 8) -> str:
    """A short table for a human. The worst candidates come first."""
    lines = [f"base divergence with 8 bits everywhere: {base_kl:.5f}", "worst ternary tensors:"]
    for candidate in sorted(candidates, key=lambda c: -c.ternary_kl)[:limit]:
        lines.append(
            f"  kl={candidate.ternary_kl:8.4f}  error={candidate.ternary_error:.3f}  "
            f"{candidate.name}"
        )
    lines.append("best ternary tensors:")
    for candidate in sorted(candidates, key=lambda c: c.ternary_kl)[:limit]:
        lines.append(
            f"  kl={candidate.ternary_kl:8.4f}  error={candidate.ternary_error:.3f}  "
            f"{candidate.name}"
        )
    total = sum(1 for c in candidates if math.isfinite(c.ternary_kl))
    lines.append(f"{total} candidates measured")
    return "\n".join(lines)
