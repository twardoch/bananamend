# this_file: bananamendy/tests/test_quantize.py
"""The quantizer must lose nothing that it does not have to lose.

These tests use small matrices with known numbers, so a failure names the step
that broke. The tests need no model weights.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bananamendy.quantize import (
    DEFAULT_GROUP_SIZE,
    GPTQ_METHOD,
    INT8_METHOD,
    METHOD,
    THRESHOLD_STEPS,
    dequantize,
    dequantize_int8,
    pack_codes,
    quantize_checkpoint,
    quantize_matrix,
    quantize_matrix_calibrated,
    quantize_matrix_int8,
    unpack_codes,
)

CODES = {"minus one": 0, "zero": 1, "plus one": 2}


def gaussian(rows: int, cols: int, seed: int = 0, scale: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((rows, cols)) * scale).astype(np.float32)


# ---- the packing ----------------------------------------------------------


def test_pack_and_unpack_return_the_same_codes():
    codes = np.array([0, 1, 2, 2, 1, 0, 2, 1, 0], dtype=np.uint8)
    packed = pack_codes(codes)
    assert packed.nbytes == 3, "nine codes need three bytes"
    assert unpack_codes(packed, codes.size).tolist() == codes.tolist()


def test_packing_uses_two_bits_for_each_code():
    codes = np.array([0, 1, 2, 1], dtype=np.uint8)
    # 0b01_10_01_00, from the lowest bits upwards.
    assert pack_codes(codes).tolist() == [0b01100100]


def test_packing_needs_a_quarter_of_a_byte_for_each_weight():
    codes = np.ones(1024, dtype=np.uint8)
    assert pack_codes(codes).nbytes == 256


# ---- the ternary grid -----------------------------------------------------


def test_codes_name_only_the_three_permitted_values():
    codes, _, _, _ = quantize_matrix(gaussian(8, 64), 64)
    assert set(np.unique(codes)).issubset(set(CODES.values()))


def test_the_rebuilt_matrix_holds_at_most_three_values_in_a_group():
    codes, positive, negative, rebuilt = quantize_matrix(gaussian(4, 32), 32)
    for row in range(4):
        values = set(np.unique(rebuilt[row]))
        assert len(values) <= 3, values


def test_dequantize_agrees_with_the_quantizer():
    matrix = gaussian(16, 96, seed=2)
    codes, positive, negative, rebuilt = quantize_matrix(matrix, 32)
    packed = pack_codes(codes)
    assert np.array_equal(dequantize(packed, positive, negative, matrix.shape, 32), rebuilt)


def test_the_searched_threshold_is_not_worse_than_the_classic_rule():
    matrix = gaussian(32, 128, seed=3)
    _, _, _, searched = quantize_matrix(matrix, 64)

    # The classic rule of Ternary Weight Networks: one threshold at 0.7 times the
    # mean absolute value, and one scale for both signs.
    blocks = matrix.reshape(32, -1, 64)
    threshold = np.abs(blocks).mean(axis=2, keepdims=True) * 0.7
    chosen = np.abs(blocks) > threshold
    count = chosen.sum(axis=2, keepdims=True).astype(np.float32)
    alpha = np.divide(
        np.where(chosen, np.abs(blocks), 0).sum(axis=2, keepdims=True),
        count,
        out=np.zeros_like(count),
        where=count > 0,
    )
    classic = (np.sign(blocks) * chosen * alpha).reshape(matrix.shape)

    assert np.linalg.norm(matrix - searched) <= np.linalg.norm(matrix - classic)


def test_the_two_scales_can_differ_in_a_group():
    # A group with small positive weights and large negative weights must not use
    # one scale for both sides.
    matrix = np.array([[0.1, 0.1, -1.0, -1.0]], dtype=np.float32)
    _, positive, negative, rebuilt = quantize_matrix(matrix, 4)
    assert positive[0, 0] == pytest.approx(0.1, abs=1e-6)
    assert negative[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(rebuilt, [[0.1, 0.1, -1.0, -1.0]], atol=1e-6)


def test_a_group_of_zeros_gives_only_the_code_for_zero():
    codes, positive, negative, rebuilt = quantize_matrix(np.zeros((2, 8), np.float32), 8)
    assert (codes == CODES["zero"]).all()
    assert not rebuilt.any()
    assert not positive.any() and not negative.any()


def test_a_group_size_above_the_width_becomes_one_group():
    _, positive, _, _ = quantize_matrix(gaussian(4, 10), 64)
    assert positive.shape == (4, 1)


def test_a_width_that_the_group_does_not_divide_still_works():
    matrix = gaussian(3, 70, seed=5)
    codes, positive, negative, rebuilt = quantize_matrix(matrix, 64)
    assert codes.shape == matrix.shape
    assert positive.shape == (3, 2)
    packed = pack_codes(codes)
    assert np.array_equal(dequantize(packed, positive, negative, matrix.shape, 64), rebuilt)


def test_the_threshold_search_includes_the_classic_value():
    assert 0.7 in THRESHOLD_STEPS
    assert min(THRESHOLD_STEPS) <= 0.05, "a group of few large and many small weights needs this"


# ---- the 8-bit grid -------------------------------------------------------


def test_int8_loses_very_little():
    matrix = gaussian(16, 64, seed=6)
    _, _, rebuilt = quantize_matrix_int8(matrix, 64)
    error = np.linalg.norm(matrix - rebuilt) / np.linalg.norm(matrix)
    assert error < 0.01, error


def test_int8_dequantize_agrees_with_the_quantizer():
    matrix = gaussian(8, 48, seed=7)
    codes, scale, rebuilt = quantize_matrix_int8(matrix, 16)
    assert np.array_equal(dequantize_int8(codes, scale, matrix.shape, 16), rebuilt)


def test_int8_keeps_the_sign_of_every_weight():
    matrix = gaussian(8, 32, seed=8)
    _, _, rebuilt = quantize_matrix_int8(matrix, 32)
    same = np.sign(matrix) == np.sign(rebuilt)
    assert same.mean() > 0.99


def test_int8_is_almost_three_times_larger_than_ternary():
    # With a group of 64: ternary needs 0.25 bytes for each weight and two scales
    # for each group, which is 0.375 bytes. 8 bits need 1 byte and one scale,
    # which is 1.0625 bytes. The ratio is therefore near 2.8, and not 4.
    matrix = gaussian(64, 256, seed=9)
    ternary_codes, positive, negative, _ = quantize_matrix(matrix, 64)
    int8_codes, int8_scale, _ = quantize_matrix_int8(matrix, 64)
    ternary_bytes = pack_codes(ternary_codes).nbytes + positive.nbytes + negative.nbytes
    int8_bytes = int8_codes.nbytes + int8_scale.nbytes
    assert 2.5 < int8_bytes / ternary_bytes < 3.0
    assert ternary_bytes / matrix.size == pytest.approx(0.375, abs=0.01)


# ---- the calibrated grid --------------------------------------------------


def test_calibration_reduces_the_error_of_the_output():
    matrix = gaussian(64, 128, seed=10)
    rng = np.random.default_rng(11)
    # Channels of very different size, which is what a real layer receives.
    channel = np.exp(rng.normal(0.0, 1.0, size=128)).astype(np.float32)
    inputs = (rng.standard_normal((512, 128)) * channel).astype(np.float32)

    _, _, _, plain = quantize_matrix(matrix, 64)
    _, _, _, calibrated = quantize_matrix_calibrated(matrix, inputs, 64)

    def output_error(candidate: np.ndarray) -> float:
        return float(
            np.linalg.norm(inputs @ (matrix - candidate).T) / np.linalg.norm(inputs @ matrix.T)
        )

    assert output_error(calibrated) < output_error(plain)


def test_calibration_gives_the_shapes_that_the_format_needs():
    matrix = gaussian(16, 64, seed=12)
    inputs = gaussian(128, 64, seed=13)
    codes, positive, negative, rebuilt = quantize_matrix_calibrated(matrix, inputs, 32)
    assert codes.shape == matrix.shape
    assert positive.shape == (16, 2)
    assert negative.shape == (16, 2)
    assert rebuilt.shape == matrix.shape
    assert set(np.unique(codes)).issubset(set(CODES.values()))


def test_a_dead_channel_does_not_stop_the_quantizer():
    matrix = gaussian(8, 32, seed=14)
    inputs = gaussian(64, 32, seed=15)
    inputs[:, 5] = 0.0  # no sample ever uses this column
    codes, _, _, rebuilt = quantize_matrix_calibrated(matrix, inputs, 32)
    assert np.isfinite(rebuilt).all()
    assert set(np.unique(codes)).issubset(set(CODES.values()))


# ---- a complete checkpoint ------------------------------------------------


def write_checkpoint(directory, tensors: dict[str, np.ndarray], config: dict) -> None:
    from safetensors.numpy import save_file

    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")


def small_checkpoint(tmp_path):
    tensors = {
        "transformer.wte.weight": gaussian(32, 16, seed=20),
        "transformer.ln_f.weight": np.ones(16, dtype=np.float32),
        "transformer.h.0.attn.q_proj.weight": gaussian(16, 16, seed=21),
        "transformer.h.0.mlp.w_down.weight": gaussian(16, 32, seed=22),
    }
    source = tmp_path / "float"
    write_checkpoint(source, tensors, {"hidden_size": 16, "model_type": "test"})
    return source, tensors


def test_a_checkpoint_keeps_the_tensors_that_must_stay_in_floats(tmp_path):
    source, _ = small_checkpoint(tmp_path)
    report = quantize_checkpoint(source, tmp_path / "small", group_size=8)
    kept = {t.name for t in report.tensors if not t.quantized}
    assert "transformer.wte.weight" in kept
    assert "transformer.ln_f.weight" in kept


def test_a_checkpoint_writes_the_block_that_the_engine_reads(tmp_path):
    source, _ = small_checkpoint(tmp_path)
    destination = tmp_path / "small"
    quantize_checkpoint(source, destination, group_size=8)
    config = json.loads((destination / "config.json").read_text("utf-8"))
    block = config["quantization"]
    assert block["packing"] == "2bit-4per-byte-low-first"
    assert block["codes"] == {"0": -1, "1": 0, "2": 1}
    assert block["method"] == METHOD
    name = "transformer.h.0.attn.q_proj.weight"
    assert block["tensors"][name]["shape"] == [16, 16]
    assert block["tensors"][name]["group_size"] == 8


def test_a_plan_decides_the_grid_of_each_tensor(tmp_path):
    source, _ = small_checkpoint(tmp_path)
    destination = tmp_path / "mixed"
    plan = {
        "transformer.h.0.attn.q_proj.weight": "ternary",
        "transformer.h.0.mlp.w_down.weight": "int8",
        "transformer.wte.weight": "int8",
    }
    quantize_checkpoint(source, destination, group_size=8, plan=plan)
    block = json.loads((destination / "config.json").read_text("utf-8"))["quantization"]
    assert block["tensors"]["transformer.h.0.attn.q_proj.weight"]["method"] == METHOD
    assert block["tensors"]["transformer.h.0.mlp.w_down.weight"]["method"] == INT8_METHOD
    # The plan overrules the rule that keeps the embedding in floats.
    assert block["tensors"]["transformer.wte.weight"]["method"] == INT8_METHOD
    assert block["method"] == "mixed"


def test_a_plan_can_keep_a_tensor_in_floats(tmp_path):
    source, _ = small_checkpoint(tmp_path)
    destination = tmp_path / "float-kept"
    quantize_checkpoint(
        source,
        destination,
        group_size=8,
        plan={"transformer.h.0.attn.q_proj.weight": "float"},
    )
    block = json.loads((destination / "config.json").read_text("utf-8"))["quantization"]
    assert "transformer.h.0.attn.q_proj.weight" not in block["tensors"]


def test_calibration_marks_the_method_in_the_block(tmp_path):
    source, tensors = small_checkpoint(tmp_path)
    name = "transformer.h.0.attn.q_proj.weight"
    destination = tmp_path / "calibrated"
    quantize_checkpoint(
        source,
        destination,
        group_size=8,
        calibration={name: gaussian(64, 16, seed=23)},
    )
    block = json.loads((destination / "config.json").read_text("utf-8"))["quantization"]
    assert block["tensors"][name]["method"] == GPTQ_METHOD


def test_the_error_limit_keeps_a_bad_tensor_in_floats(tmp_path):
    source, _ = small_checkpoint(tmp_path)
    report = quantize_checkpoint(source, tmp_path / "limited", group_size=8, error_limit=0.01)
    # A ternary grid cannot reach an error of one percent, so nothing is quantized.
    assert all(not t.quantized for t in report.tensors)


def test_the_report_counts_the_bytes(tmp_path):
    source, _ = small_checkpoint(tmp_path)
    report = quantize_checkpoint(source, tmp_path / "counted", group_size=8)
    summary = report.summary()
    assert summary["float_mb"] > summary["result_mb"]
    assert summary["ratio"] > 1.0
    assert summary["tensors_quantized"] == 2


def test_the_tokenizer_travels_with_the_weights(tmp_path):
    source, _ = small_checkpoint(tmp_path)
    destination = tmp_path / "copied"
    quantize_checkpoint(source, destination, group_size=8)
    assert (destination / "tokenizer.json").is_file()
    assert (destination / "quantization_report.json").is_file()


def test_the_default_group_size_is_the_documented_one():
    assert DEFAULT_GROUP_SIZE == 64


# ---- the engine reads what the quantizer writes ---------------------------
#
# These tests build a very small but complete model, quantize it, and then load
# the result with the engine. A difference here means that the Rust side and the
# Python side disagree about the format.


def complete_checkpoint(tmp_path, name: str = "float"):
    """A small model that the engine can load. The numbers are arbitrary."""
    from safetensors.numpy import save_file

    hidden, heads, kv_heads, head_dim, ffn, vocab, layers = 16, 2, 1, 8, 32, 24, 2
    rng = np.random.default_rng(99)

    def matrix(rows: int, cols: int) -> np.ndarray:
        return (rng.standard_normal((rows, cols)) * 0.08).astype(np.float32)

    tensors = {
        "transformer.wte.weight": matrix(vocab, hidden),
        "transformer.ln_f.weight": np.ones(hidden, dtype=np.float32),
    }
    for layer in range(layers):
        prefix = f"transformer.h.{layer}."
        tensors[f"{prefix}ln_1.weight"] = np.ones(hidden, dtype=np.float32)
        tensors[f"{prefix}ln_2.weight"] = np.ones(hidden, dtype=np.float32)
        tensors[f"{prefix}attn.q_norm.weight"] = np.ones(head_dim, dtype=np.float32)
        tensors[f"{prefix}attn.k_norm.weight"] = np.ones(head_dim, dtype=np.float32)
        tensors[f"{prefix}attn.q_proj.weight"] = matrix(heads * head_dim, hidden)
        tensors[f"{prefix}attn.k_proj.weight"] = matrix(kv_heads * head_dim, hidden)
        tensors[f"{prefix}attn.v_proj.weight"] = matrix(kv_heads * head_dim, hidden)
        tensors[f"{prefix}attn.o_proj.weight"] = matrix(hidden, heads * head_dim)
        tensors[f"{prefix}mlp.w_gate.weight"] = matrix(ffn, hidden)
        tensors[f"{prefix}mlp.w_up.weight"] = matrix(ffn, hidden)
        tensors[f"{prefix}mlp.w_down.weight"] = matrix(hidden, ffn)

    config = {
        "hidden_size": hidden,
        "intermediate_size": ffn,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "vocab_size": vocab,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "model_type": "tiny",
    }
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    # The engine always reads a tokenizer, so the test writes a very small one.
    words = ["[UNK]", "<|bos|>", "<|eos|>"] + [f"w{i}" for i in range(vocab - 3)]
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": {word: index for index, word in enumerate(words)},
            "unk_token": "[UNK]",
        },
    }
    (directory / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
    return directory


def logits_of(directory, tokens: list[int]) -> np.ndarray:
    """The scores of the engine after `tokens`."""
    import bananamendr

    return np.asarray(bananamendr.Model(str(directory)).logits(tokens=tokens), dtype=np.float32)


TOKENS = [1, 5, 9, 13, 7, 3, 11, 4]


@pytest.mark.parametrize("grid", ["int8", "ternary"])
def test_the_engine_reads_a_quantized_checkpoint(tmp_path, grid):
    source = complete_checkpoint(tmp_path)
    destination = tmp_path / grid
    names = [n for n in _matrix_names() if True]
    quantize_checkpoint(source, destination, group_size=8, plan={n: grid for n in names})
    scores = logits_of(destination, TOKENS)
    assert scores.shape == (24,)
    assert np.isfinite(scores).all()


def _matrix_names() -> list[str]:
    names = []
    for layer in range(2):
        for part in (
            "attn.q_proj",
            "attn.k_proj",
            "attn.v_proj",
            "attn.o_proj",
            "mlp.w_gate",
            "mlp.w_up",
            "mlp.w_down",
        ):
            names.append(f"transformer.h.{layer}.{part}.weight")
    return names


def test_the_engine_and_the_quantizer_rebuild_the_same_values(tmp_path):
    """The strongest check: the engine must compute what the floats compute.

    The test quantizes a model, writes a float copy of the *quantized* weights,
    and compares the scores of the two. The engine rebuilds each value inside the
    multiplication, so a difference means that the Rust side reads the codes or
    the scales differently.
    """
    from bananamendy.quantize import dequantize_checkpoint

    source = complete_checkpoint(tmp_path)
    quantized = tmp_path / "quantized"
    rebuilt = tmp_path / "rebuilt"
    plan = {n: ("ternary" if "attn" in n else "int8") for n in _matrix_names()}
    plan["transformer.wte.weight"] = "int8"
    quantize_checkpoint(source, quantized, group_size=8, plan=plan)
    dequantize_checkpoint(quantized, rebuilt)

    from_codes = logits_of(quantized, TOKENS)
    from_floats = logits_of(rebuilt, TOKENS)
    assert np.allclose(from_codes, from_floats, atol=2e-4), (
        np.abs(from_codes - from_floats).max()
    )


def test_a_quantized_checkpoint_is_not_the_float_checkpoint(tmp_path):
    # A guard against a test that would pass because nothing was quantized.
    source = complete_checkpoint(tmp_path)
    quantized = tmp_path / "ternary-all"
    quantize_checkpoint(
        source, quantized, group_size=8, plan={n: "ternary" for n in _matrix_names()}
    )
    assert not np.allclose(logits_of(source, TOKENS), logits_of(quantized, TOKENS), atol=1e-3)


def test_the_engine_refuses_a_method_that_it_does_not_know(tmp_path):
    import bananamendr

    source = complete_checkpoint(tmp_path)
    broken = tmp_path / "broken"
    quantize_checkpoint(
        source, broken, group_size=8, plan={n: "int8" for n in _matrix_names()}
    )
    config = json.loads((broken / "config.json").read_text("utf-8"))
    config["quantization"]["tensors"]["transformer.h.0.attn.q_proj.weight"]["method"] = "future-v9"
    (broken / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Exception, match="future-v9"):
        bananamendr.Model(str(broken))
