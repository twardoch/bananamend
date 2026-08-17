# this_file: bananamendy/tests/test_plan.py
"""The planner and the calibration pass.

These tests build a very small model with known numbers, so they need no model
weights and no network. The forward pass of `reference.py` is compared with the
engine in the WebAssembly parity work; here the tests check the parts that the
planner needs: the divergence, the order of the candidates, and the budget.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bananamendy.calibration import (
    CALIBRATION_TEXTS,
    EVALUATION_CHATS,
    EVALUATION_PROMPTS,
    EVALUATION_TEXT,
)
from bananamendy.plan import Candidate, choose, kl_divergence, measure, summarize
from bananamendy.reference import Reference, apply_rope, collect_inputs, rms_norm, rope_tables

HIDDEN = 16
HEADS = 2
KV_HEADS = 1
HEAD_DIM = 8
FFN = 32
VOCAB = 24
LAYERS = 2


def tiny_config() -> dict:
    return {
        "hidden_size": HIDDEN,
        "intermediate_size": FFN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "model_type": "tiny",
    }


def tiny_checkpoint(tmp_path):
    """Writes a small model that the numpy pass can run."""
    from safetensors.numpy import save_file

    rng = np.random.default_rng(42)

    def matrix(rows: int, cols: int) -> np.ndarray:
        return (rng.standard_normal((rows, cols)) * 0.08).astype(np.float32)

    tensors: dict[str, np.ndarray] = {
        "transformer.wte.weight": matrix(VOCAB, HIDDEN),
        "transformer.ln_f.weight": np.ones(HIDDEN, dtype=np.float32),
    }
    for layer in range(LAYERS):
        prefix = f"transformer.h.{layer}."
        tensors[f"{prefix}ln_1.weight"] = np.ones(HIDDEN, dtype=np.float32)
        tensors[f"{prefix}ln_2.weight"] = np.ones(HIDDEN, dtype=np.float32)
        tensors[f"{prefix}attn.q_norm.weight"] = np.ones(HEAD_DIM, dtype=np.float32)
        tensors[f"{prefix}attn.k_norm.weight"] = np.ones(HEAD_DIM, dtype=np.float32)
        tensors[f"{prefix}attn.q_proj.weight"] = matrix(HEADS * HEAD_DIM, HIDDEN)
        tensors[f"{prefix}attn.k_proj.weight"] = matrix(KV_HEADS * HEAD_DIM, HIDDEN)
        tensors[f"{prefix}attn.v_proj.weight"] = matrix(KV_HEADS * HEAD_DIM, HIDDEN)
        tensors[f"{prefix}attn.o_proj.weight"] = matrix(HIDDEN, HEADS * HEAD_DIM)
        tensors[f"{prefix}mlp.w_gate.weight"] = matrix(FFN, HIDDEN)
        tensors[f"{prefix}mlp.w_up.weight"] = matrix(FFN, HIDDEN)
        tensors[f"{prefix}mlp.w_down.weight"] = matrix(HIDDEN, FFN)

    directory = tmp_path / "tiny"
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(tiny_config()), encoding="utf-8")
    return directory


TOKENS = [1, 5, 9, 13, 7, 3, 11, 4, 8, 2]


# ---- the parts of the forward pass ---------------------------------------


def test_rms_norm_gives_a_unit_mean_square():
    x = np.array([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32)
    out = rms_norm(x, np.ones(4, dtype=np.float32), 0.0)
    assert np.mean(np.square(out)) == pytest.approx(1.0, abs=1e-5)


def test_rms_norm_multiplies_by_the_weight():
    x = np.ones((1, 4), dtype=np.float32)
    weight = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert np.allclose(rms_norm(x, weight, 0.0), weight[None, :])


def test_rope_at_position_zero_changes_nothing():
    cos, sin = rope_tables(8, 4, 10000.0)
    vector = np.arange(8, dtype=np.float32)[None, :]
    assert np.allclose(apply_rope(vector, cos[0][None, :], sin[0][None, :]), vector)


def test_rope_rotates_the_interleaved_pairs():
    # A rotation keeps the length of each pair.
    cos, sin = rope_tables(4, 3, 10000.0)
    vector = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    rotated = apply_rope(vector, cos[2][None, :], sin[2][None, :])
    first = np.hypot(rotated[0, 0], rotated[0, 1])
    second = np.hypot(rotated[0, 2], rotated[0, 3])
    assert first == pytest.approx(1.0, abs=1e-5)
    assert second == pytest.approx(1.0, abs=1e-5)
    assert not np.allclose(rotated, vector)


def test_the_forward_pass_gives_one_row_of_scores_for_each_token(tmp_path):
    model = Reference.load(tiny_checkpoint(tmp_path))
    logits = model.forward(TOKENS)
    assert logits.shape == (len(TOKENS), VOCAB)
    assert np.isfinite(logits).all()


def test_a_later_token_cannot_change_an_earlier_score(tmp_path):
    """The mask must be causal, or the calibration records the wrong inputs."""
    model = Reference.load(tiny_checkpoint(tmp_path))
    short = model.forward(TOKENS[:5])
    long = model.forward(TOKENS)
    assert np.allclose(short, long[:5], atol=1e-5)


def test_the_collection_records_every_matrix(tmp_path):
    directory = tiny_checkpoint(tmp_path)
    collected = collect_inputs(directory, [TOKENS], max_rows=64)
    names = Reference.load(directory).matrix_names()
    assert set(collected) == set(names)
    assert len(names) == LAYERS * 7


def test_the_recorded_rows_have_the_width_of_the_matrix(tmp_path):
    directory = tiny_checkpoint(tmp_path)
    model = Reference.load(directory)
    collected = collect_inputs(directory, [TOKENS, TOKENS[:6]], max_rows=64)
    for name, rows in collected.items():
        assert rows.shape[1] == model.matrix(name).shape[1], name


def test_the_collection_respects_the_row_limit(tmp_path):
    directory = tiny_checkpoint(tmp_path)
    collected = collect_inputs(directory, [TOKENS] * 8, max_rows=16)
    for rows in collected.values():
        assert rows.shape[0] <= 16


# ---- the divergence ------------------------------------------------------


def test_the_divergence_of_a_model_with_itself_is_zero():
    scores = np.array([[1.0, 2.0, 3.0], [0.5, 0.25, 0.1]], dtype=np.float32)
    assert kl_divergence(scores, scores) == pytest.approx(0.0, abs=1e-9)


def test_the_divergence_grows_with_the_difference():
    reference = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    near = np.array([[1.1, 2.0, 3.0]], dtype=np.float32)
    far = np.array([[3.0, 2.0, 1.0]], dtype=np.float32)
    assert kl_divergence(reference, near) < kl_divergence(reference, far)


def test_the_divergence_is_never_negative():
    rng = np.random.default_rng(7)
    for _ in range(5):
        a = rng.standard_normal((4, 10)).astype(np.float32)
        b = rng.standard_normal((4, 10)).astype(np.float32)
        assert kl_divergence(a, b) >= 0.0


# ---- the plan ------------------------------------------------------------


def test_the_measurement_covers_every_candidate(tmp_path):
    directory = tiny_checkpoint(tmp_path)
    candidates, base = measure(directory, group_size=8, tokens=TOKENS)
    assert len(candidates) == LAYERS * 7
    assert base >= 0.0
    for candidate in candidates:
        assert candidate.ternary_kl >= 0.0
        assert candidate.int8_error < candidate.ternary_error, candidate.name


def test_a_small_budget_keeps_almost_everything_in_eight_bits(tmp_path):
    directory = tiny_checkpoint(tmp_path)
    plan, report = choose(directory, kl_budget=0.0, group_size=8, tokens=TOKENS)
    assert report["ternary"] == 0
    assert set(plan.values()) == {"int8"}


def test_a_large_budget_accepts_ternary_tensors(tmp_path):
    directory = tiny_checkpoint(tmp_path)
    plan, report = choose(directory, kl_budget=100.0, group_size=8, tokens=TOKENS)
    assert report["ternary"] == len(plan)
    assert set(plan.values()) == {"ternary"}


def test_the_plan_never_passes_the_budget(tmp_path):
    directory = tiny_checkpoint(tmp_path)
    budget = 0.05
    _, report = choose(directory, kl_budget=budget, group_size=8, tokens=TOKENS)
    assert report["kl_result"] <= budget + 1e-9


def test_a_larger_budget_gives_a_smaller_file(tmp_path):
    directory = tiny_checkpoint(tmp_path)
    _, small = choose(directory, kl_budget=0.0, group_size=8, tokens=TOKENS)
    _, large = choose(directory, kl_budget=100.0, group_size=8, tokens=TOKENS)
    assert large["matrix_result_mb"] <= small["matrix_result_mb"]
    assert large["matrix_ratio"] >= small["matrix_ratio"]


def test_the_summary_names_the_worst_and_the_best(tmp_path):
    directory = tiny_checkpoint(tmp_path)
    candidates, base = measure(directory, group_size=8, tokens=TOKENS)
    text = summarize(candidates, base, limit=2)
    assert "worst ternary tensors" in text
    assert "best ternary tensors" in text
    assert f"{len(candidates)} candidates measured" in text


def test_a_measurement_without_tokens_is_refused(tmp_path):
    directory = tiny_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="tokens"):
        measure(directory, group_size=8)


def test_a_candidate_holds_the_sizes_of_both_grids():
    candidate = Candidate(
        name="x",
        ternary_kl=0.1,
        ternary_error=0.4,
        int8_error=0.01,
        ternary_bytes=375,
        int8_bytes=1062,
        float_bytes=4000,
    )
    assert candidate.ternary_bytes < candidate.int8_bytes < candidate.float_bytes


# ---- the text ------------------------------------------------------------


def test_the_calibration_text_and_the_measurement_text_are_different():
    for text in CALIBRATION_TEXTS:
        assert text not in EVALUATION_TEXT
        assert EVALUATION_TEXT not in text


def test_there_is_enough_calibration_text():
    words = sum(len(text.split()) for text in CALIBRATION_TEXTS)
    assert len(CALIBRATION_TEXTS) >= 20
    assert words >= 1000, words


def test_the_measurement_has_prompts_and_conversations():
    assert len(EVALUATION_PROMPTS) >= 5
    assert len(EVALUATION_CHATS) >= 3
    for messages in EVALUATION_CHATS:
        assert messages[-1]["role"] == "user"
