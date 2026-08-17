# this_file: bananamendy/src/bananamendy/models.py
"""Checkpoint resolution and download.

Weights live in the ordinary Hugging Face cache (`HF_HOME` / `HF_HUB_CACHE`, or
`~/.cache/huggingface/hub`), not in a bananamend-specific directory: the
checkpoints are plain HF repos, and anyone who already has them should not pay
for a second copy. `bananamendr` itself only ever accepts a directory path —
resolution to a path happens here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

# Short alias -> Hugging Face repo id.
REGISTRY: dict[str, str] = {
    "nano": "BananaMind/BananaMind-2-Nano-Chat",
    "mini": "BananaMind/BananaMind-2-Mini-Chat",
    "pro": "BananaMind/BananaMind-2-Pro-Preview-Chat",
}

# `bananamendr` reads config.json, model.safetensors, tokenizer.json and
# (optionally) tokenizer_config.json. The rest is fetched because it is small
# and makes the local copy a faithful checkpoint.
ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "*.md",
]

REQUIRED_FILES = ("config.json", "model.safetensors", "tokenizer.json")


class ModelError(RuntimeError):
    """A checkpoint could not be resolved, downloaded, or is incomplete."""


@dataclass(frozen=True)
class Checkpoint:
    """A checkpoint on disk, ready to hand to `bananamendr.Model`."""

    name: str
    path: Path
    repo_id: str | None = None

    @property
    def size_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.path.glob("*") if f.is_file())


def repo_id_for(name: str) -> str:
    """Alias, or a repo id passed through unchanged."""
    return REGISTRY.get(name, name)


def _verify(path: Path) -> Path:
    missing = [f for f in REQUIRED_FILES if not (path / f).is_file()]
    if missing:
        raise ModelError(f"{path} is not a usable checkpoint; missing: {', '.join(missing)}")
    return path


def pull(name: str, *, revision: str | None = None) -> Checkpoint:
    """Download a checkpoint into the Hugging Face cache and return its path."""
    repo_id = repo_id_for(name)
    try:
        path = snapshot_download(
            repo_id, revision=revision, allow_patterns=ALLOW_PATTERNS
        )
    except Exception as error:  # network, auth, unknown repo
        raise ModelError(f"cannot download {repo_id}: {error}") from error
    return Checkpoint(name=name, path=_verify(Path(path)), repo_id=repo_id)


def resolve(name: str, *, revision: str | None = None, download: bool = True) -> Checkpoint:
    """Resolve an alias, repo id, or local directory to a checkpoint on disk.

    A path that exists wins over the alias table, so a local checkout is always
    usable without touching the network. Otherwise the cache is consulted
    offline first; only then is a download attempted.
    """
    candidate = Path(name).expanduser()
    if candidate.is_dir():
        return Checkpoint(name=candidate.name, path=_verify(candidate))

    repo_id = repo_id_for(name)
    try:
        cached = snapshot_download(
            repo_id,
            revision=revision,
            allow_patterns=ALLOW_PATTERNS,
            local_files_only=True,
        )
        return Checkpoint(name=name, path=_verify(Path(cached)), repo_id=repo_id)
    except (LocalEntryNotFoundError, OSError, ModelError):
        pass

    if not download:
        raise ModelError(
            f"{name} is not in the local cache; run `bananamendy pull {name}` first"
        )
    return pull(name, revision=revision)


def list_local() -> list[Checkpoint]:
    """Registry checkpoints already present in the cache, smallest first."""
    found: list[Checkpoint] = []
    for alias in REGISTRY:
        try:
            found.append(resolve(alias, download=False))
        except ModelError:
            continue
    return sorted(found, key=lambda c: (c.size_bytes, c.name))
