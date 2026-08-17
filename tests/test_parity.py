# this_file: tests/test_parity.py
"""Parity tests: the Rust runtime against Hugging Face `transformers`.

These are the tests that actually discriminate. Output that "reads like English"
survives a wrong RoPE pairing or a missing embedding scale; token-exact greedy
decoding does not.

Requires `torch` and `transformers` (reference), and the `bananamendr` extension
module (`maturin develop --release -m crates/bananamendr-py/Cargo.toml`).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
bananamendr = pytest.importorskip("bananamendr")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = {
    "nano": REPO_ROOT / "ref" / "BananaMind-2-Nano-Chat",
    "mini": REPO_ROOT / "ref" / "BananaMind-2-Mini-Chat",
    "pro": REPO_ROOT / "ref" / "BananaMind-2-Pro-Preview-Chat",
}
# Pro is ~556 MB and its reference forward pass is slow; opt in explicitly.
DEFAULT_MODELS = ["nano", "mini"]
SELECTED = os.environ.get("BANANAMIND_TEST_MODELS", ",".join(DEFAULT_MODELS)).split(",")

PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Wie geht es dir? Antworte auf Deutsch, bitte. 🍌",
]

_reference_cache: dict[str, tuple] = {}


def available(name: str) -> Path:
    path = CHECKPOINTS[name]
    if not (path / "model.safetensors").is_file():
        pytest.skip(f"checkpoint not present: {path}")
    return path


def reference(name: str):
    """Loads the HF model/tokenizer once per session."""
    if name not in _reference_cache:
        path = available(name)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, dtype=torch.float32
        )
        model.eval()
        _reference_cache[name] = (model, tokenizer)
    return _reference_cache[name]


@pytest.fixture(scope="session")
def rust_model_factory():
    cache: dict[str, bananamendr.Model] = {}

    def get(name: str) -> bananamendr.Model:
        if name not in cache:
            cache[name] = bananamendr.Model(str(available(name)))
        return cache[name]

    return get


def ids_of(prompt_or_messages, tokenizer) -> list[int]:
    messages = (
        prompt_or_messages
        if isinstance(prompt_or_messages, list)
        else [{"role": "user", "content": prompt_or_messages}]
    )
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    # transformers 5.x returns a BatchEncoding (a UserDict, not a dict);
    # older versions return a plain list of ids.
    if hasattr(encoded, "keys") and "input_ids" in encoded.keys():
        ids = encoded["input_ids"]
        # Batched encodings nest one list per input.
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(i) for i in ids]
    return [int(i) for i in encoded]


@pytest.mark.parametrize("name", SELECTED)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_tokenizer_when_encoding_text_then_matches_transformers(
    name, prompt, rust_model_factory
):
    _, tokenizer = reference(name)
    rust = rust_model_factory(name)
    expected = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    assert rust.tokenize(prompt) == expected, "Rust tokenizer diverged from transformers"


@pytest.mark.parametrize("name", SELECTED)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_chat_template_when_rendered_then_matches_transformers(
    name, prompt, rust_model_factory
):
    _, tokenizer = reference(name)
    rust = rust_model_factory(name)
    messages = [{"role": "user", "content": prompt}]
    expected_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    assert rust.apply_chat_template(messages) == expected_text
    assert rust.tokenize(expected_text) == ids_of(prompt, tokenizer)


@pytest.mark.parametrize("name", SELECTED)
def test_chat_template_when_multi_turn_then_matches_transformers(name, rust_model_factory):
    """Exercises the system, user and assistant branches, not just user."""
    _, tokenizer = reference(name)
    rust = rust_model_factory(name)
    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Name one ocean."},
        {"role": "assistant", "content": "The Pacific."},
        {"role": "user", "content": "And a river? 🍌"},
    ]
    for add_generation_prompt in (True, False):
        expected = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        assert rust.apply_chat_template(messages, add_generation_prompt) == expected
    assert rust.tokenize(
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    ) == ids_of(messages, tokenizer)


@pytest.mark.parametrize("name", SELECTED)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_logits_when_prompt_prefilled_then_close_to_transformers(
    name, prompt, rust_model_factory
):
    model, tokenizer = reference(name)
    rust = rust_model_factory(name)
    ids = ids_of(prompt, tokenizer)

    with torch.no_grad():
        expected = model(torch.tensor([ids])).logits[0, -1].numpy()
    actual = rust.logits(tokens=ids)

    assert len(actual) == expected.shape[0]
    diff = max(abs(float(a) - float(b)) for a, b in zip(actual, expected))
    scale = float(abs(expected).max())
    assert diff < 2e-3 * max(scale, 1.0), (
        f"{name}: max logit deviation {diff:.6f} against reference scale {scale:.3f}"
    )
    assert int(expected.argmax()) == max(
        range(len(actual)), key=actual.__getitem__
    ), "argmax token differs from the reference"


@pytest.mark.parametrize("name", SELECTED)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_generation_when_32_tokens_then_token_exact(
    name, prompt, rust_model_factory
):
    model, tokenizer = reference(name)
    rust = rust_model_factory(name)
    ids = ids_of(prompt, tokenizer)

    with torch.no_grad():
        expected = model.generate(
            torch.tensor([ids]),
            do_sample=False,
            max_new_tokens=32,
            pad_token_id=model.config.pad_token_id,
        )[0].tolist()[len(ids) :]

    actual = rust.generate_tokens(ids, max_new_tokens=32, temperature=0.0).tokens
    assert actual == expected, (
        f"{name}: greedy token streams diverge\n  rust: {actual}\n  hf:   {expected}"
    )


@pytest.mark.parametrize("name", SELECTED)
def test_decode_through_cache_matches_fresh_prefill(name, rust_model_factory):
    """Tokens decoded through the KV cache must equal a fresh prefill of the same
    prefix — the check that catches a position or cache-indexing bug."""
    _, tokenizer = reference(name)
    rust = rust_model_factory(name)
    ids = ids_of(PROMPTS[0], tokenizer)

    cached = rust.generate_tokens(ids, max_new_tokens=16, temperature=0.0).tokens
    assert len(cached) > 4

    split = 4
    restarted = rust.generate_tokens(
        ids + cached[:split],
        max_new_tokens=len(cached) - split,
        temperature=0.0,
    ).tokens
    assert cached[split:] == restarted, (
        "decoding through the cache diverged from a fresh prefill"
    )

    # Growing the prompt must change the logits: a runtime that ignored the tail
    # of the prompt would pass everything above.
    assert rust.logits(tokens=ids) != rust.logits(tokens=ids[:-1])


@pytest.mark.parametrize("name", SELECTED)
def test_sampling_when_seeded_then_reproducible(name, rust_model_factory):
    rust = rust_model_factory(name)
    messages = [{"role": "user", "content": PROMPTS[0]}]
    kwargs = dict(max_new_tokens=16, temperature=0.8, top_k=40, top_p=0.95, seed=1234)
    first = rust.chat(messages, **kwargs).tokens
    second = rust.chat(messages, **kwargs).tokens
    assert first == second, "same seed must reproduce the same tokens"


@pytest.mark.parametrize("name", SELECTED)
def test_streaming_callback_reassembles_full_text(name, rust_model_factory):
    rust = rust_model_factory(name)
    chunks: list[str] = []
    generation = rust.chat(
        [{"role": "user", "content": PROMPTS[0]}],
        max_new_tokens=24,
        temperature=0.0,
        on_token=lambda text, token: chunks.append(text),
    )
    assert "".join(chunks) == generation.text


def test_context_overflow_raises(rust_model_factory):
    rust = rust_model_factory(SELECTED[0])
    with pytest.raises(RuntimeError, match="context length"):
        rust.generate_tokens([1, 2, 3], max_new_tokens=64, max_seq_len=2)
