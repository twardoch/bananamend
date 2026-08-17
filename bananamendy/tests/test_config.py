# this_file: bananamendy/tests/test_config.py
"""Configuration precedence and round-tripping."""

from __future__ import annotations

from bananamendy.config import Config, load_config, write_default_config


def test_defaults_when_no_file_then_builtin(tmp_path):
    config = load_config(tmp_path / "missing.toml")
    assert config == Config()


def test_written_config_when_reloaded_then_identical(tmp_path):
    target = tmp_path / "config.toml"
    write_default_config(target)
    assert load_config(target) == Config()


def test_write_default_config_when_present_then_preserved(tmp_path):
    target = tmp_path / "config.toml"
    target.write_text('model = "pro"\n', encoding="utf-8")
    write_default_config(target)
    assert load_config(target).model == "pro"


def test_write_default_config_when_forced_then_overwritten(tmp_path):
    target = tmp_path / "config.toml"
    target.write_text('model = "pro"\n', encoding="utf-8")
    write_default_config(target, force=True)
    assert load_config(target).model == Config().model


def test_unknown_keys_when_loaded_then_ignored(tmp_path):
    target = tmp_path / "config.toml"
    target.write_text('model = "mini"\nnonsense = 1\n', encoding="utf-8")
    assert load_config(target).model == "mini"


def test_environment_when_set_then_overrides_file(tmp_path, monkeypatch):
    target = tmp_path / "config.toml"
    target.write_text('model = "mini"\nport = 1111\n', encoding="utf-8")
    monkeypatch.setenv("BANANAMENDY_MODEL", "pro")
    monkeypatch.setenv("BANANAMENDY_PORT", "2222")
    config = load_config(target)
    assert (config.model, config.port) == ("pro", 2222)


def test_merged_when_value_is_none_then_default_kept():
    config = Config().merged(model=None, temperature=0.1)
    assert config.model == Config().model
    assert config.temperature == 0.1


def test_sampling_when_called_then_matches_bananamendr_keywords():
    keys = set(Config().sampling())
    assert keys == {
        "max_new_tokens",
        "temperature",
        "top_k",
        "top_p",
        "repetition_penalty",
        "seed",
        "max_seq_len",
    }
