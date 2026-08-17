# this_file: bananamendy/tests/test_server.py
"""OpenAI-compatible surface, driven against a fake engine (no weights needed)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from bananamendy.config import Config
from bananamendy.models import ModelError
from bananamendy.server import create_app


@dataclass
class FakeGeneration:
    text: str = "hello there"
    tokens: tuple[int, ...] = (1, 2, 3)
    prompt_tokens: int = 7
    prefill_seconds: float = 0.1
    decode_seconds: float = 0.2
    tokens_per_second: float = 15.0
    finished_by_eos: bool = True


class FakeEngine:
    """Records what the server asked for and returns canned output."""

    def __init__(self, *, fail: Exception | None = None) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.fail = fail

    def loaded_names(self) -> list[str]:
        return ["nano"]

    def load(self, name):  # pragma: no cover - unused by these tests
        raise AssertionError("load() should not be reached")

    def chat(self, name, messages, **sampling):
        if self.fail:
            raise self.fail
        self.calls.append(("chat", name, sampling))
        return FakeGeneration()

    def generate(self, name, prompt, **sampling):
        if self.fail:
            raise self.fail
        self.calls.append(("generate", name, sampling))
        return FakeGeneration()

    def stream(self, name, *, prompt=None, messages=None, **sampling):
        if self.fail:
            raise self.fail
        self.calls.append(("stream", name, sampling))
        yield "hel"
        yield "lo"


@pytest.fixture
def client_and_engine():
    engine = FakeEngine()
    return TestClient(create_app(Config(model="nano"), engine)), engine


def test_health_when_called_then_reports_loaded(client_and_engine):
    client, _ = client_and_engine
    body = client.get("/health").json()
    assert body == {"status": "ok", "loaded": ["nano"]}


def test_list_models_when_called_then_includes_registry(client_and_engine):
    client, _ = client_and_engine
    ids = [entry["id"] for entry in client.get("/v1/models").json()["data"]]
    assert ids[0] == "nano"
    assert {"nano", "mini", "pro"} <= set(ids)
    assert len(ids) == len(set(ids)), "model ids must not repeat"


def test_chat_completion_when_called_then_openai_shape(client_and_engine):
    client, _ = client_and_engine
    body = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    ).json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "nano"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "hello there"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


def test_chat_completion_when_max_tokens_given_then_forwarded(client_and_engine):
    client, engine = client_and_engine
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
    )
    assert engine.calls[0][2]["max_new_tokens"] == 5


def test_chat_completion_when_no_sampling_given_then_config_defaults(client_and_engine):
    client, engine = client_and_engine
    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert engine.calls[0][2] == Config(model="nano").sampling()


def test_chat_completion_when_streaming_then_sse_with_done(client_and_engine):
    client, _ = client_and_engine
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        text = "".join(response.iter_text())
    assert '"delta":{"role":"assistant"}' in text
    assert '"content":"hel"' in text and '"content":"lo"' in text
    assert '"finish_reason":"stop"' in text
    assert text.rstrip().endswith("data: [DONE]")


def test_completion_when_called_then_text_choice(client_and_engine):
    client, _ = client_and_engine
    body = client.post("/v1/completions", json={"prompt": "once upon"}).json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == "hello there"


def test_completion_when_streaming_then_text_deltas(client_and_engine):
    client, _ = client_and_engine
    with client.stream(
        "POST", "/v1/completions", json={"prompt": "once upon", "stream": True}
    ) as response:
        text = "".join(response.iter_text())
    assert '"text":"hel"' in text
    assert text.rstrip().endswith("data: [DONE]")


def test_request_when_model_named_then_used_instead_of_default(client_and_engine):
    client, engine = client_and_engine
    client.post(
        "/v1/chat/completions",
        json={"model": "mini", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert engine.calls[0][1] == "mini"


def test_chat_completion_when_model_missing_then_404():
    engine = FakeEngine(fail=ModelError("nope is not in the local cache"))
    client = TestClient(create_app(Config(), engine))
    response = client.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    assert "local cache" in response.json()["detail"]


def test_chat_completion_when_engine_fails_then_400():
    engine = FakeEngine(fail=RuntimeError("context overflow"))
    client = TestClient(create_app(Config(), engine))
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 400


def test_chat_completion_when_messages_missing_then_422(client_and_engine):
    client, _ = client_and_engine
    assert client.post("/v1/chat/completions", json={}).status_code == 422
