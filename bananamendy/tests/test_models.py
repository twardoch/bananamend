# this_file: bananamendy/tests/test_models.py
"""Checkpoint resolution: local paths, cache hits, and refusal to guess."""

from __future__ import annotations

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from bananamendy import models


def make_checkpoint(path):
    path.mkdir(parents=True, exist_ok=True)
    for name in models.REQUIRED_FILES:
        (path / name).write_text("x", encoding="utf-8")
    return path


def test_repo_id_for_when_alias_then_expanded():
    assert models.repo_id_for("nano") == "BananaMind/BananaMind-2-Nano-Chat"


def test_repo_id_for_when_repo_id_then_unchanged():
    assert models.repo_id_for("acme/Model-7B") == "acme/Model-7B"


def test_resolve_when_local_dir_then_no_network(tmp_path, monkeypatch):
    def explode(*_args, **_kwargs):
        raise AssertionError("resolve() must not touch the hub for a local path")

    monkeypatch.setattr(models, "snapshot_download", explode)
    checkpoint = models.resolve(str(make_checkpoint(tmp_path / "nano")))
    assert checkpoint.path == tmp_path / "nano"
    assert checkpoint.repo_id is None


def test_resolve_when_local_dir_incomplete_then_error(tmp_path):
    incomplete = tmp_path / "half"
    incomplete.mkdir()
    (incomplete / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(models.ModelError, match="model.safetensors"):
        models.resolve(str(incomplete))


def test_resolve_when_cached_then_uses_cache(tmp_path, monkeypatch):
    cached = make_checkpoint(tmp_path / "cached")
    calls = []

    def fake_download(repo_id, **kwargs):
        calls.append(kwargs.get("local_files_only", False))
        return str(cached)

    monkeypatch.setattr(models, "snapshot_download", fake_download)
    checkpoint = models.resolve("nano")
    assert checkpoint.path == cached
    assert calls == [True], "a cache hit must not fall through to a download"


def test_resolve_when_absent_and_download_disabled_then_error(monkeypatch):
    def fake_download(repo_id, **kwargs):
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(models, "snapshot_download", fake_download)
    with pytest.raises(models.ModelError, match="bananamendy pull"):
        models.resolve("nano", download=False)


def test_resolve_when_absent_then_downloads(tmp_path, monkeypatch):
    fetched = make_checkpoint(tmp_path / "fetched")
    seen = []

    def fake_download(repo_id, **kwargs):
        seen.append(kwargs.get("local_files_only", False))
        if kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError("not cached")
        return str(fetched)

    monkeypatch.setattr(models, "snapshot_download", fake_download)
    assert models.resolve("nano").path == fetched
    assert seen == [True, False]


def test_pull_when_hub_fails_then_model_error(monkeypatch):
    def fake_download(repo_id, **kwargs):
        raise OSError("no route to host")

    monkeypatch.setattr(models, "snapshot_download", fake_download)
    with pytest.raises(models.ModelError, match="cannot download"):
        models.pull("nano")


def test_list_local_when_partially_cached_then_only_present(tmp_path, monkeypatch):
    cached = make_checkpoint(tmp_path / "nano")

    def fake_download(repo_id, **kwargs):
        if repo_id == models.REGISTRY["nano"]:
            return str(cached)
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(models, "snapshot_download", fake_download)
    assert [c.name for c in models.list_local()] == ["nano"]
