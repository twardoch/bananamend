# this_file: bananamendy/src/bananamendy/engine.py
"""Loaded-model cache and streaming over `bananamendr`.

`bananamendr.Model` holds the GIL for the whole of a generation except while it
runs the `on_token` callback, so a generation cannot be parallelised inside one
process. Every entry point here therefore serialises on a single lock, and
streaming works by running the generation on a worker thread whose callback
pushes deltas into a queue.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bananamendr

from .models import Checkpoint, resolve

Message = dict[str, str]

_SENTINEL = object()


@dataclass
class LoadedModel:
    checkpoint: Checkpoint
    model: Any

    @property
    def config(self) -> dict[str, Any]:
        return dict(self.model.config)


class Engine:
    """Caches loaded checkpoints and serialises generation."""

    def __init__(self, *, download: bool = True) -> None:
        self._download = download
        self._models: dict[Path, LoadedModel] = {}
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()

    def load(self, name: str) -> LoadedModel:
        checkpoint = resolve(name, download=self._download)
        with self._load_lock:
            loaded = self._models.get(checkpoint.path)
            if loaded is None:
                loaded = LoadedModel(checkpoint, bananamendr.Model(str(checkpoint.path)))
                self._models[checkpoint.path] = loaded
            return loaded

    def loaded_names(self) -> list[str]:
        return sorted(m.checkpoint.name for m in self._models.values())

    def info(self, name: str) -> dict[str, Any]:
        loaded = self.load(name)
        return {
            "name": loaded.checkpoint.name,
            "path": str(loaded.checkpoint.path),
            "repo_id": loaded.checkpoint.repo_id,
            **loaded.config,
        }

    def generate(self, name: str, prompt: str, **sampling: Any) -> Any:
        loaded = self.load(name)
        with self._run_lock:
            return loaded.model.generate(prompt, **sampling)

    def chat(self, name: str, messages: list[Message], **sampling: Any) -> Any:
        loaded = self.load(name)
        with self._run_lock:
            return loaded.model.chat(list(messages), **sampling)

    def stream(
        self,
        name: str,
        *,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        **sampling: Any,
    ) -> Iterator[str]:
        """Yield decoded deltas as they are produced, then return.

        Exactly one of `prompt` or `messages` must be given. Errors raised on the
        worker thread are re-raised here, after whatever was already streamed.
        """
        if (prompt is None) == (messages is None):
            raise ValueError("stream() takes exactly one of prompt or messages")

        loaded = self.load(name)
        channel: queue.Queue[Any] = queue.Queue()

        def on_token(text: str, _token_id: int) -> None:
            # bananamendr calls back with (decoded delta, token id) per step.
            channel.put(text)

        def work() -> None:
            try:
                with self._run_lock:
                    if messages is not None:
                        loaded.model.chat(list(messages), on_token=on_token, **sampling)
                    else:
                        loaded.model.generate(prompt, on_token=on_token, **sampling)
            except BaseException as error:  # surfaced to the consumer below
                channel.put(error)
            finally:
                channel.put(_SENTINEL)

        worker = threading.Thread(target=work, name="bananamendy-generate", daemon=True)
        worker.start()
        try:
            while True:
                item = channel.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            worker.join()
