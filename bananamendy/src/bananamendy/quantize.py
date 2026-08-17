# this_file: bananamendy/src/bananamendy/quantize.py
"""Ternary quantization of a checkpoint, after training.

Each weight becomes one of three values: minus one, zero, or plus one. A scale
for a small group of weights gives the size back. The result needs 2 bits for
each weight, and 2 numbers for each group, so a matrix becomes approximately
10 times smaller than the same matrix in 32-bit floats.

The method comes from Ternary Weight Networks (Li and others, 2016), with two
additions that cost no training:

* The threshold is not the usual `0.7 * mean(|w|)`. For each group the
  quantizer tries many thresholds and keeps the one with the smallest square
  error. The 0.7 rule comes from an assumption about the distribution of the
  weights, and a real group often disagrees with that assumption.
* The positive weights and the negative weights get separate scales. This is
  the asymmetric grid of PT2-LLM. It cannot increase the error, and it usually
  makes it smaller.

Some tensors stay in 32-bit floats, because a ternary form of them destroys the
model: the token embedding, the output projection when it is not tied to the
embedding, and every normalisation weight. The quantizer also reports the error
of each tensor, so a caller can keep the worst tensors in floats as well.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from safetensors.numpy import load_file, save_file

METHOD = "ternary-twn-v1"
DEFAULT_GROUP_SIZE = 64
# The search runs over these multiples of the mean absolute value of a group.
# The classic Ternary Weight Networks value is 0.7, and it is inside the range.
# The range starts low, because a group can hold a few large weights and many
# small ones. A threshold of 0.2 would then remove every small weight, and the
# search must be able to keep them.
THRESHOLD_STEPS: tuple[float, ...] = tuple(round(0.05 + 0.05 * i, 2) for i in range(22))

# A ternary form of these tensors makes the model useless, so they stay in floats.
KEEP_FLOAT_SUFFIXES = (
    "wte.weight",  # the token embedding
    "lm_head.weight",  # the output projection
    "ln_f.weight",  # the last normalisation
)
# Every normalisation weight is a vector, and the quantizer only takes matrices.
# This suffix list is therefore a statement of intent, not the only guard.


@dataclass
class TensorReport:
    """What happened to one tensor."""

    name: str
    shape: tuple[int, ...]
    quantized: bool
    reason: str = ""
    relative_error: float = 0.0
    zero_fraction: float = 0.0
    float_bytes: int = 0
    quantized_bytes: int = 0

    @property
    def saved_bytes(self) -> int:
        return max(0, self.float_bytes - self.quantized_bytes)


@dataclass
class QuantizationReport:
    """What happened to the complete checkpoint."""

    group_size: int
    tensors: list[TensorReport] = field(default_factory=list)

    @property
    def float_bytes(self) -> int:
        return sum(t.float_bytes for t in self.tensors)

    @property
    def result_bytes(self) -> int:
        return sum(t.quantized_bytes if t.quantized else t.float_bytes for t in self.tensors)

    @property
    def ratio(self) -> float:
        return self.float_bytes / self.result_bytes if self.result_bytes else 1.0

    def summary(self) -> dict:
        quantized = [t for t in self.tensors if t.quantized]
        errors = [t.relative_error for t in quantized]
        ternary = [t for t in quantized if t.reason != "8-bit"]
        return {
            "group_size": self.group_size,
            "tensors_ternary": len(ternary),
            "tensors_int8": len(quantized) - len(ternary),
            "tensors_total": len(self.tensors),
            "tensors_quantized": len(quantized),
            "float_mb": round(self.float_bytes / 1e6, 2),
            "result_mb": round(self.result_bytes / 1e6, 2),
            "ratio": round(self.ratio, 2),
            "worst_relative_error": round(max(errors), 4) if errors else 0.0,
            "mean_relative_error": round(float(np.mean(errors)), 4) if errors else 0.0,
            "mean_zero_fraction": round(
                float(np.mean([t.zero_fraction for t in quantized])), 4
            )
            if quantized
            else 0.0,
        }


def _grouped(matrix: np.ndarray, group_size: int) -> tuple[np.ndarray, int, int]:
    """Reshapes `[rows, cols]` to `[rows, groups, group_size]`, with padding.

    The padding holds zeros. A zero never passes the threshold, so it takes the
    code for zero and it does not change a scale.
    """
    rows, cols = matrix.shape
    size = min(group_size, cols)
    groups = (cols + size - 1) // size
    padded = groups * size
    if padded != cols:
        matrix = np.concatenate(
            [matrix, np.zeros((rows, padded - cols), dtype=matrix.dtype)], axis=1
        )
    return matrix.reshape(rows, groups, size), groups, size


def quantize_matrix(
    matrix: np.ndarray, group_size: int = DEFAULT_GROUP_SIZE
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quantizes one matrix.

    Returns the codes (0 for minus one, 1 for zero, 2 for plus one), the scale of
    the positive weights, the scale of the negative weights, and the values that
    the codes and the scales rebuild.
    """
    original = matrix.astype(np.float32, copy=False)
    blocks, groups, size = _grouped(original, group_size)

    mean_absolute = np.abs(blocks).mean(axis=2, keepdims=True)
    best_error = np.full((blocks.shape[0], groups, 1), np.inf, dtype=np.float32)
    best_positive = np.zeros((blocks.shape[0], groups, 1), dtype=np.float32)
    best_negative = np.zeros((blocks.shape[0], groups, 1), dtype=np.float32)
    best_threshold = np.zeros((blocks.shape[0], groups, 1), dtype=np.float32)

    for step in THRESHOLD_STEPS:
        threshold = mean_absolute * step
        positive = blocks > threshold
        negative = blocks < -threshold

        # Separate scales for the two signs. An empty side gets a scale of zero,
        # and then every weight of that side takes the code for zero.
        positive_count = positive.sum(axis=2, keepdims=True)
        negative_count = negative.sum(axis=2, keepdims=True)
        positive_sum = np.where(positive, blocks, 0.0).sum(axis=2, keepdims=True)
        negative_sum = np.where(negative, -blocks, 0.0).sum(axis=2, keepdims=True)
        scale_positive = np.divide(
            positive_sum,
            positive_count,
            out=np.zeros_like(positive_sum),
            where=positive_count > 0,
        )
        scale_negative = np.divide(
            negative_sum,
            negative_count,
            out=np.zeros_like(negative_sum),
            where=negative_count > 0,
        )

        rebuilt = np.where(positive, scale_positive, 0.0) - np.where(
            negative, scale_negative, 0.0
        )
        error = ((blocks - rebuilt) ** 2).sum(axis=2, keepdims=True)

        better = error < best_error
        best_error = np.where(better, error, best_error)
        best_positive = np.where(better, scale_positive, best_positive)
        best_negative = np.where(better, scale_negative, best_negative)
        best_threshold = np.where(better, threshold, best_threshold)

    positive = blocks > best_threshold
    negative = blocks < -best_threshold
    # A side with a scale of zero carries no information, so it becomes zero.
    positive &= best_positive > 0
    negative &= best_negative > 0

    codes = np.ones(blocks.shape, dtype=np.uint8)
    codes[positive] = 2
    codes[negative] = 0
    rebuilt = np.where(positive, best_positive, 0.0) - np.where(negative, best_negative, 0.0)

    rows, cols = original.shape
    flat_codes = codes.reshape(rows, groups * size)[:, :cols]
    flat_rebuilt = rebuilt.reshape(rows, groups * size)[:, :cols].astype(np.float32)
    return (
        flat_codes,
        best_positive.reshape(rows, groups).astype(np.float32),
        best_negative.reshape(rows, groups).astype(np.float32),
        flat_rebuilt,
    )


def pack_codes(codes: np.ndarray) -> np.ndarray:
    """Packs 2-bit codes, four in each byte, from the lowest bits upwards."""
    flat = codes.reshape(-1)
    padded = (-len(flat)) % 4
    if padded:
        # The code for zero fills the end, so the extra values change nothing.
        flat = np.concatenate([flat, np.ones(padded, dtype=np.uint8)])
    quads = flat.reshape(-1, 4).astype(np.uint8)
    return (quads[:, 0] | (quads[:, 1] << 2) | (quads[:, 2] << 4) | (quads[:, 3] << 6)).astype(
        np.uint8
    )


def unpack_codes(packed: np.ndarray, count: int) -> np.ndarray:
    """The opposite of `pack_codes`. This function exists for the tests."""
    bytes_ = packed.astype(np.uint8)
    quads = np.stack(
        [bytes_ & 0b11, (bytes_ >> 2) & 0b11, (bytes_ >> 4) & 0b11, (bytes_ >> 6) & 0b11],
        axis=1,
    )
    return quads.reshape(-1)[:count]


def dequantize(
    packed: np.ndarray,
    scale_positive: np.ndarray,
    scale_negative: np.ndarray,
    shape: tuple[int, int],
    group_size: int,
) -> np.ndarray:
    """Rebuilds a matrix of floats. The Rust engine does the same work."""
    rows, cols = shape
    codes = unpack_codes(packed, rows * cols).reshape(rows, cols)
    size = min(group_size, cols)
    index = np.minimum(np.arange(cols) // size, scale_positive.shape[1] - 1)
    positive = scale_positive[:, index]
    negative = scale_negative[:, index]
    out = np.zeros((rows, cols), dtype=np.float32)
    out[codes == 2] = positive[codes == 2]
    out[codes == 0] = -negative[codes == 0]
    return out


def _keeps_floats(name: str, array: np.ndarray) -> str:
    """Gives the reason to keep a tensor in floats, or an empty string."""
    if array.ndim != 2:
        return "not a matrix"
    if array.dtype != np.float32:
        return f"dtype {array.dtype}"
    if any(name.endswith(suffix) for suffix in KEEP_FLOAT_SUFFIXES):
        return "in the list of tensors that stay in floats"
    if min(array.shape) < 8:
        return "too small to be worth a scale"
    return ""


def quantize_checkpoint(
    source: Path,
    destination: Path,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    keep_float: Iterable[str] = (),
    error_limit: float | None = None,
    calibration: dict[str, np.ndarray] | None = None,
    plan: dict[str, str] | None = None,
) -> QuantizationReport:
    """Writes a ternary copy of a checkpoint.

    `keep_float` names tensors that must stay in floats. `error_limit` keeps a
    tensor in floats when its relative error is above the limit, which gives a
    checkpoint of mixed precision.

    `calibration` holds the recorded input rows of each matrix, from
    `reference.collect_inputs`. With it, the quantizer makes the error of the
    output small instead of the error of the weights, which is a large
    improvement. Without it, the quantizer uses the simple method.
    """
    source = Path(source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    keep = set(keep_float)

    tensors = load_file(source / "model.safetensors")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))

    report = QuantizationReport(group_size=group_size)
    output: dict[str, np.ndarray] = {}
    per_tensor: dict[str, dict] = {}

    for name in sorted(tensors):
        array = tensors[name]
        float_bytes = array.size * 4
        reason = "named by the caller" if name in keep else _keeps_floats(name, array)
        planned = (plan or {}).get(name)
        # A plan that names a tensor overrules the default rule, but a tensor
        # that is not a matrix of 32-bit floats can never be quantized.
        if (
            planned in {"int8", "ternary"}
            and name not in keep
            and array.ndim == 2
            and array.dtype == np.float32
        ):
            reason = ""
        if reason:
            output[name] = array
            report.tensors.append(
                TensorReport(
                    name=name,
                    shape=tuple(array.shape),
                    quantized=False,
                    reason=reason,
                    float_bytes=float_bytes,
                    quantized_bytes=float_bytes,
                )
            )
            continue

        choice = planned or "ternary"
        if choice == "float":
            output[name] = array
            report.tensors.append(
                TensorReport(
                    name=name,
                    shape=tuple(array.shape),
                    quantized=False,
                    reason="the plan keeps this tensor in floats",
                    float_bytes=float_bytes,
                    quantized_bytes=float_bytes,
                )
            )
            continue

        if choice == "int8":
            codes8, scale8, rebuilt8 = quantize_matrix_int8(array, group_size)
            norm8 = float(np.linalg.norm(array))
            relative8 = float(np.linalg.norm(array - rebuilt8) / norm8) if norm8 else 0.0
            output[f"{name}.int8.codes"] = codes8
            output[f"{name}.int8.scale"] = scale8
            per_tensor[name] = {
                "method": INT8_METHOD,
                "group_size": min(group_size, array.shape[1]),
                "shape": list(array.shape),
                "groups": int(scale8.shape[1]),
                "relative_error": round(relative8, 5),
            }
            report.tensors.append(
                TensorReport(
                    name=name,
                    shape=tuple(array.shape),
                    quantized=True,
                    reason="8-bit",
                    relative_error=relative8,
                    zero_fraction=float((codes8 == 0).mean()),
                    float_bytes=float_bytes,
                    quantized_bytes=codes8.nbytes + scale8.nbytes,
                )
            )
            continue

        rows = None if calibration is None else calibration.get(name)
        if rows is not None and rows.shape[1] == array.shape[1] and rows.shape[0] >= 8:
            codes, scale_positive, scale_negative, rebuilt = quantize_matrix_calibrated(
                array, rows, group_size
            )
            method = GPTQ_METHOD
        else:
            codes, scale_positive, scale_negative, rebuilt = quantize_matrix(array, group_size)
            method = METHOD
        norm = float(np.linalg.norm(array))
        relative = float(np.linalg.norm(array - rebuilt) / norm) if norm else 0.0

        if error_limit is not None and relative > error_limit:
            output[name] = array
            report.tensors.append(
                TensorReport(
                    name=name,
                    shape=tuple(array.shape),
                    quantized=False,
                    reason=f"relative error {relative:.3f} is above the limit {error_limit}",
                    relative_error=relative,
                    float_bytes=float_bytes,
                    quantized_bytes=float_bytes,
                )
            )
            continue

        packed = pack_codes(codes)
        output[f"{name}.ternary.codes"] = packed
        output[f"{name}.ternary.scale_pos"] = scale_positive
        output[f"{name}.ternary.scale_neg"] = scale_negative
        quantized_bytes = packed.nbytes + scale_positive.nbytes + scale_negative.nbytes
        per_tensor[name] = {
            "method": method,
            "group_size": min(group_size, array.shape[1]),
            "shape": list(array.shape),
            "groups": int(scale_positive.shape[1]),
            "relative_error": round(relative, 5),
        }
        report.tensors.append(
            TensorReport(
                name=name,
                shape=tuple(array.shape),
                quantized=True,
                relative_error=relative,
                zero_fraction=float((codes == 1).mean()),
                float_bytes=float_bytes,
                quantized_bytes=quantized_bytes,
            )
        )

    methods = sorted({info["method"] for info in per_tensor.values()}) or [METHOD]
    config["quantization"] = {
        "method": methods[0] if len(methods) == 1 else "mixed",
        "methods": methods,
        "group_size": group_size,
        "packing": "2bit-4per-byte-low-first",
        "codes": {"0": -1, "1": 0, "2": 1},
        "producer": "bananamendy",
        "tensors": per_tensor,
    }
    save_file(output, str(destination / "model.safetensors"))
    (destination / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # The tokenizer and the chat format must travel with the weights.
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
    ):
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, destination / name)

    (destination / "quantization_report.json").write_text(
        json.dumps(
            {
                "summary": report.summary(),
                "tensors": [
                    {
                        "name": t.name,
                        "shape": list(t.shape),
                        "quantized": t.quantized,
                        "reason": t.reason,
                        "relative_error": round(t.relative_error, 5),
                        "zero_fraction": round(t.zero_fraction, 5),
                    }
                    for t in report.tensors
                ],
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def dequantize_checkpoint(source: Path, destination: Path) -> int:
    """Writes a float copy of a ternary checkpoint.

    The result is numerically identical to what the engine computes from the
    ternary form, so it is the fast way to measure the loss in quality.
    Returns the number of tensors that it rebuilt.
    """
    source = Path(source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    tensors = load_file(source / "model.safetensors")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    quantization = config.get("quantization") or {}
    per_tensor = quantization.get("tensors") or {}
    group_size = int(quantization.get("group_size", DEFAULT_GROUP_SIZE))

    output: dict[str, np.ndarray] = {}
    rebuilt = 0
    quantized_suffixes = (
        ".ternary.codes",
        ".ternary.scale_pos",
        ".ternary.scale_neg",
        ".int8.codes",
        ".int8.scale",
    )
    for name, array in tensors.items():
        if name.endswith(quantized_suffixes):
            continue
        output[name] = array
    for name, info in per_tensor.items():
        shape = tuple(int(v) for v in info["shape"])
        size = int(info.get("group_size", group_size))
        method = info.get("method", METHOD)
        if method == INT8_METHOD:
            output[name] = dequantize_int8(
                tensors[f"{name}.int8.codes"],
                tensors[f"{name}.int8.scale"],
                (shape[0], shape[1]),
                size,
            )
        else:
            output[name] = dequantize(
                tensors[f"{name}.ternary.codes"],
                tensors[f"{name}.ternary.scale_pos"],
                tensors[f"{name}.ternary.scale_neg"],
                (shape[0], shape[1]),
                size,
            )
        rebuilt += 1

    config.pop("quantization", None)
    config["quantization_source"] = {"method": METHOD, "rebuilt_tensors": rebuilt}
    save_file(output, str(destination / "model.safetensors"))
    (destination / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
    ):
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, destination / name)
    return rebuilt

# ---------------------------------------------------------------------------
# Calibrated quantization
# ---------------------------------------------------------------------------
#
# The simple method above makes the error of the *weights* small. That is the
# wrong target: a weight only matters as much as the activation that multiplies
# it. The method below makes the error of the *output* small instead. It is the
# GPTQ procedure (Frantar and others, 2022) with a ternary grid, which is also
# what PT2-LLM builds on.
#
# For each matrix the quantizer:
#
# 1. builds the second-moment matrix H = X^T X of the recorded inputs;
# 2. takes the columns in order, quantizes one column, and then moves the error
#    of that column into the columns that are still to come;
# 3. finds the scales of a group from the weights that arrive at that group,
#    and not from the original weights.
#
# Step 2 is the part that the simple method cannot do. The error of an early
# column becomes a correction to a later column, so the output of the matrix
# stays near the output of the float matrix.

GPTQ_METHOD = "ternary-gptq-v1"
# The damping keeps the Hessian invertible, and it also keeps the corrections
# small. GPTQ uses one percent of the mean of the diagonal. A ternary grid makes
# each correction large, so this quantizer uses more.
DAMPING = 0.1


def _group_grid(
    block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finds the threshold and the two scales for each row of one group.

    `block` is `[rows, group_size]`. The search is the same as in
    `quantize_matrix`: try many thresholds, and keep the one with the smallest
    square error.
    """
    mean_absolute = np.abs(block).mean(axis=1, keepdims=True)
    best_error = np.full((block.shape[0], 1), np.inf, dtype=np.float64)
    best_positive = np.zeros((block.shape[0], 1), dtype=np.float64)
    best_negative = np.zeros((block.shape[0], 1), dtype=np.float64)
    best_threshold = np.zeros((block.shape[0], 1), dtype=np.float64)

    for step in THRESHOLD_STEPS:
        threshold = mean_absolute * step
        positive = block > threshold
        negative = block < -threshold
        positive_count = positive.sum(axis=1, keepdims=True)
        negative_count = negative.sum(axis=1, keepdims=True)
        positive_sum = np.where(positive, block, 0.0).sum(axis=1, keepdims=True)
        negative_sum = np.where(negative, -block, 0.0).sum(axis=1, keepdims=True)
        scale_positive = np.divide(
            positive_sum, positive_count, out=np.zeros_like(positive_sum), where=positive_count > 0
        )
        scale_negative = np.divide(
            negative_sum, negative_count, out=np.zeros_like(negative_sum), where=negative_count > 0
        )
        rebuilt = np.where(positive, scale_positive, 0.0) - np.where(negative, scale_negative, 0.0)
        error = ((block - rebuilt) ** 2).sum(axis=1, keepdims=True)
        better = error < best_error
        best_error = np.where(better, error, best_error)
        best_positive = np.where(better, scale_positive, best_positive)
        best_negative = np.where(better, scale_negative, best_negative)
        best_threshold = np.where(better, threshold, best_threshold)

    return best_threshold, best_positive, best_negative


def _quantize_column(
    column: np.ndarray,
    threshold: np.ndarray,
    scale_positive: np.ndarray,
    scale_negative: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gives the codes and the rebuilt values of one column."""
    positive = (column > threshold[:, 0]) & (scale_positive[:, 0] > 0)
    negative = (column < -threshold[:, 0]) & (scale_negative[:, 0] > 0)
    codes = np.ones(column.shape, dtype=np.uint8)
    codes[positive] = 2
    codes[negative] = 0
    rebuilt = np.where(positive, scale_positive[:, 0], 0.0) - np.where(
        negative, scale_negative[:, 0], 0.0
    )
    return codes, rebuilt


def quantize_matrix_calibrated(
    matrix: np.ndarray,
    inputs: np.ndarray,
    group_size: int = DEFAULT_GROUP_SIZE,
    damping: float = DAMPING,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quantizes one matrix with the recorded inputs of that matrix.

    `matrix` is `[rows, cols]`, and `inputs` is `[samples, cols]`. The result has
    the same shape as the result of `quantize_matrix`.
    """
    weights = matrix.astype(np.float64, copy=True)
    rows, cols = weights.shape
    size = min(group_size, cols)
    groups = (cols + size - 1) // size

    samples = inputs.astype(np.float64, copy=False)
    hessian = samples.T @ samples
    diagonal = np.diag(hessian).copy()
    # A column that no sample used carries no information. A small value on the
    # diagonal keeps the matrix invertible, and the column then behaves as it
    # does in the simple method.
    dead = diagonal == 0.0
    mean_diagonal = float(diagonal[~dead].mean()) if (~dead).any() else 1.0
    diagonal[dead] = mean_diagonal
    hessian[np.arange(cols), np.arange(cols)] = diagonal + damping * mean_diagonal

    # The upper Cholesky factor of the inverse holds the correction weights.
    inverse = np.linalg.inv(hessian)
    factor = np.linalg.cholesky(inverse).T

    codes = np.ones((rows, cols), dtype=np.uint8)
    rebuilt = np.zeros((rows, cols), dtype=np.float64)
    scale_positive = np.zeros((rows, groups), dtype=np.float32)
    scale_negative = np.zeros((rows, groups), dtype=np.float32)

    for group in range(groups):
        start = group * size
        end = min(start + size, cols)
        threshold, positive, negative = _group_grid(weights[:, start:end])
        scale_positive[:, group] = positive[:, 0]
        scale_negative[:, group] = negative[:, 0]

        for column in range(start, end):
            code, value = _quantize_column(weights[:, column], threshold, positive, negative)
            codes[:, column] = code
            rebuilt[:, column] = value
            # Move the error of this column into the columns that follow.
            error = (weights[:, column] - value) / factor[column, column]
            if column + 1 < cols:
                weights[:, column + 1 :] -= np.outer(error, factor[column, column + 1 :])

    return (
        codes,
        scale_positive,
        scale_negative,
        rebuilt.astype(np.float32),
    )

# ---------------------------------------------------------------------------
# 8-bit quantization
# ---------------------------------------------------------------------------
#
# Ternary weights are very small and they lose a lot. Eight-bit weights are four
# times larger than ternary weights, and they lose almost nothing. A checkpoint
# that mixes the two keeps the quality of the model and still becomes much
# smaller. The selection is in `plan.py`.

INT8_METHOD = "int8-sym-v1"


def quantize_matrix_int8(
    matrix: np.ndarray, group_size: int = DEFAULT_GROUP_SIZE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantizes one matrix to 8-bit codes with one scale for each group.

    Returns the codes, the scales, and the values that they rebuild. The grid is
    symmetric: a code of zero is a weight of zero, and the largest weight of a
    group takes the code 127.
    """
    original = matrix.astype(np.float32, copy=False)
    blocks, groups, size = _grouped(original, group_size)
    largest = np.abs(blocks).max(axis=2, keepdims=True)
    scale = np.where(largest > 0, largest / 127.0, 1.0).astype(np.float32)
    codes = np.clip(np.rint(blocks / scale), -127, 127).astype(np.int8)
    rebuilt = codes.astype(np.float32) * scale

    rows, cols = original.shape
    flat_codes = codes.reshape(rows, groups * size)[:, :cols]
    flat_rebuilt = rebuilt.reshape(rows, groups * size)[:, :cols].astype(np.float32)
    return flat_codes, scale.reshape(rows, groups).astype(np.float32), flat_rebuilt


def dequantize_int8(
    codes: np.ndarray, scale: np.ndarray, shape: tuple[int, int], group_size: int
) -> np.ndarray:
    """Rebuilds a matrix from 8-bit codes. The Rust engine does the same work."""
    rows, cols = shape
    values = codes.astype(np.float32).reshape(rows, cols)
    size = min(group_size, cols)
    index = np.minimum(np.arange(cols) // size, scale.shape[1] - 1)
    return values * scale[:, index]

