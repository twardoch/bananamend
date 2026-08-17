# this_file: bananamendy/tests/test_engine.py
"""Engine caching and streaming, with a stub in place of a real checkpoint."""

from __future__ import annotations

from pathlib import Path

import pytest

from bananamendy.engine import Engine, LoadedModel
from bananamendy.models import Checkpoint


class StubModel:
    """Calls `on_token` per delta, like `bananamendr.Model` does."""

    def __init__(self, deltas=("a", "b", "c"), error: Exception | None = None) -> None:
        self.deltas = deltas
        self.error = error
        self.loads = 0

    def _run(self, on_token):
        for delta in self.deltas:
            if on_token is not None:
                on_token(delta, 0)
        if self.error:
            raise self.error
        return "".join(self.deltas)

    def chat(self, messages, *, on_token=None, **_sampling):
        return self._run(on_token)

    def generate(self, prompt, *, on_token=None, **_sampling):
        return self._run(on_token)


@pytest.fixture
def engine_with_stub(monkeypatch, tmp_path):
    stub = StubModel()
    loads = {"count": 0}

    def fake_load(name):
        loads["count"] += 1
        return LoadedModel(Checkpoint(name=name, path=Path(tmp_path)), stub)

    engine = Engine()
    monkeypatch.setattr(engine, "load", fake_load)
    return engine, stub, loads


def test_stream_when_messages_given_then_yields_deltas(engine_with_stub):
    engine, _, _ = engine_with_stub
    out = list(engine.stream("nano", messages=[{"role": "user", "content": "hi"}]))
    assert out == ["a", "b", "c"]


def test_stream_when_prompt_given_then_yields_deltas(engine_with_stub):
    engine, _, _ = engine_with_stub
    assert "".join(engine.stream("nano", prompt="once")) == "abc"


def test_stream_when_both_inputs_given_then_rejected(engine_with_stub):
    engine, _, _ = engine_with_stub
    with pytest.raises(ValueError, match="exactly one"):
        list(engine.stream("nano", prompt="x", messages=[]))


def test_stream_when_neither_input_given_then_rejected(engine_with_stub):
    engine, _, _ = engine_with_stub
    with pytest.raises(ValueError, match="exactly one"):
        list(engine.stream("nano"))


def test_stream_when_worker_raises_then_reraised_after_deltas(monkeypatch, tmp_path):
    stub = StubModel(deltas=("a",), error=RuntimeError("decode blew up"))
    engine = Engine()
    monkeypatch.setattr(
        engine, "load", lambda name: LoadedModel(Checkpoint(name=name, path=tmp_path), stub)
    )
    seen = []
    with pytest.raises(RuntimeError, match="decode blew up"):
        for delta in engine.stream("nano", prompt="x"):
            seen.append(delta)
    assert seen == ["a"], "deltas produced before the failure must still reach the caller"


def test_load_when_called_twice_then_cached(monkeypatch, tmp_path):
    from bananamendy import engine as engine_module

    calls = {"resolve": 0, "model": 0}

    def fake_resolve(name, download=True):
        calls["resolve"] += 1
        return Checkpoint(name=name, path=tmp_path)

    class Counter:
        def __init__(self, path):
            calls["model"] += 1

    monkeypatch.setattr(engine_module, "resolve", fake_resolve)
    monkeypatch.setattr(engine_module.bananamendr, "Model", Counter)
    engine = Engine()
    engine.load("nano")
    engine.load("nano")
    assert calls == {"resolve": 2, "model": 1}
