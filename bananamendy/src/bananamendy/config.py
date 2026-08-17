# this_file: bananamendy/src/bananamendy/config.py
"""TOML configuration in the platformdirs location.

Precedence: explicit argument -> TOML config -> environment -> built-in default.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, replace
from typing import Any
from pathlib import Path

import tomli_w
from platformdirs import user_config_path

APP_NAME = "bananamendy"
CONFIG_NAME = "config.toml"


def config_path() -> Path:
    return user_config_path(APP_NAME, appauthor=False) / CONFIG_NAME


@dataclass(frozen=True)
class Config:
    """Defaults for the CLI and the server."""

    model: str = "nano"
    host: str = "127.0.0.1"
    port: int = 8377
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    seed: int = 0
    max_seq_len: int | None = None

    def merged(self, **overrides: Any) -> Config:
        """Copy with only the non-None overrides applied."""
        given = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **given)

    def sampling(self) -> dict[str, Any]:
        """Keyword arguments accepted by `bananamendr.Model.chat`/`generate`."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
            "max_seq_len": self.max_seq_len,
        }


ENV_PREFIX = "BANANAMENDY_"


def _from_env() -> dict[str, Any]:
    """Read BANANAMENDY_* overrides, coerced to the field types."""
    out: dict[str, Any] = {}
    for field, default in asdict(Config()).items():
        raw = os.environ.get(f"{ENV_PREFIX}{field.upper()}")
        if raw is None or raw == "":
            continue
        if isinstance(default, bool):
            out[field] = raw.strip().lower() in {"1", "true", "yes", "on"}
        elif isinstance(default, int) or field in {"port", "max_new_tokens", "top_k", "seed", "max_seq_len"}:
            out[field] = int(raw)
        elif isinstance(default, float):
            out[field] = float(raw)
        else:
            out[field] = raw
    return out


def load_config(path: Path | None = None) -> Config:
    """Config from TOML if present, then environment overrides."""
    target = path or config_path()
    data: dict[str, Any] = {}
    if target.is_file():
        parsed = tomllib.loads(target.read_text(encoding="utf-8"))
        known = set(asdict(Config()))
        data = {k: v for k, v in parsed.items() if k in known}
    data.update(_from_env())
    return Config(**data)


def write_default_config(path: Path | None = None, *, force: bool = False) -> Path:
    """Write the built-in defaults to the config location."""
    target = path or config_path()
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in asdict(Config()).items() if v is not None}
    target.write_text(tomli_w.dumps(payload), encoding="utf-8")
    return target
