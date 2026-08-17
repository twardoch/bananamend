<!--
this_file: CHANGELOG.md
-->

# Changelog

## Unreleased

First public release, extracted from the private `bananamind` prototype.

### Added

- `bananamendr` — Rust crate with the CPU inference runtime (RMSNorm pre-norm
  blocks, grouped-query attention, memory-mapped safetensors) plus the
  `bananamendr` CLI (`info`, `chat`, `generate`, `logits`, `bench`). Published to
  crates.io, and to PyPI as an abi3 extension module.
- `bananamendy` — Python package: checkpoint resolution and download via
  `huggingface_hub` into the ordinary HF cache, TOML config under
  `platformdirs`, a Fire CLI, and a persistent OpenAI-compatible server
  (`/v1/models`, `/v1/chat/completions`, `/v1/completions`, all with SSE
  streaming, plus `/health`).
- `build.sh`, `install.sh`, `publish.sh` and `scripts/release-manifests.py`: one
  version number across every manifest, tag-driven releases via `gitnextver`, and
  a build that needs no model weights and fails if it modifies the checkout.

### Changed from the prototype

- Renamed: crates `bananamind-core` + `bananamind-cli` merged into `bananamendr`
  (library and binary in one crate, so `cargo install bananamendr` ships the
  CLI); `bananamind-py` became `crates/bananamendr-py`, published as
  `bananamendr`. The upstream model identifiers (`BananaMind2NanoForCausalLM`,
  `model_type = "bananamind2_nano"`, the `BananaMind/*` repo ids) are unchanged —
  they must match the published `config.json`.
- `crates/bananamendr-py` is `publish = false` for crates.io: a `cdylib`
  extension module is not usable as a Rust dependency. It ships as a wheel only.
- `pyo3/extension-module` is no longer a default Cargo feature; maturin enables
  it per build, which keeps `cargo build`, `cargo test` and `cargo publish`
  working.
- MSRV recorded as 1.87 (the runtime uses `u32::is_multiple_of`).
- Model download moved out of the Rust side entirely: `bananamendr` takes
  explicit paths, `bananamendy pull` fetches.
- Reference checkpoints and prototype notes stay in the private `_priv/` area.
