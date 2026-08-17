<!--
this_file: CHANGELOG.md
-->

# Changelog

## Unreleased

### Added

- The demonstration page now holds a list of checkpoints, and a field for a
  repository of your own. It frees the previous model before it loads the next
  one, and it reports the form of the weights and the memory that they need.
- `Model.info()` in the WebAssembly module reports `storage` and `weight_bytes`,
  which is what the page shows.
- `fontlab/BananaMind-2-Mini-Chat-int8`: the third model in 8 bits.

- `bananamendy info` now reports the form of each part of the model and the number
  of bytes that the weights need in memory, so the claim that a quantized
  checkpoint stays small while it runs is checkable. `bananamendr.Model` gained
  `storage` and `weight_bytes` for the same reason.
- A model card can hold an example of what the checkpoint writes
  (`bananamendy push --sample`). The two research checkpoints use it.

### Fixed

- The demonstration page used `requestAnimationFrame` before a generation. A
  browser stops those calls in a tab that nobody looks at, and the generation
  then never started. A short timer replaces it.
- The sizes of the three models in the documentation were wrong: Mini is 101 MB
  and Pro is 556 MB, and not 230 MB and 1.1 GB.

### Changed

- The default of `bananamendy quantize` is now `int8`, and not `mixed`. `mixed` is
  only a little smaller and it changes more answers, so it must be a choice.
- `Matrix::matvec` spreads the rows of a quantized projection over the Rayon pool,
  as the float path already did. Before this change a quantized checkpoint ran on
  one thread.
- The tags of a model card describe the file: an 8-bit checkpoint is no longer
  tagged `ternary`.
- The threshold search of the ternary grid starts at 0.05 instead of 0.20 times
  the mean absolute value. A group of a few large weights and many small ones
  needs that.
- Measured numbers for every variant, including speed, are now in the
  documentation.

### Added

- A WebAssembly build. `bananamendr-wasm` runs the engine in a browser. The crate
  is not published, because a WebAssembly module has no use as a Rust dependency.
- `Pipeline::from_parts`, `Weights::from_bytes` and `Tokenizer::from_json`. These
  take a checkpoint that is already in memory, which is what a browser needs.
- Features in the engine: `std-fs` for the file loaders, `parallel` for rayon,
  `cli` for the program, and `onig` or `wasm` for the tokenizer. The defaults are
  unchanged for a computer; a browser build gives
  `--no-default-features --features wasm`.
- A documentation site in `docs/`, for GitHub Pages with the Just the Docs theme.
  The demonstration page downloads the Nano model from Hugging Face and then runs
  the engine in the browser. It uses daisyUI from a CDN for its components.
- `./wasm.sh` builds the module and compares it with the Python module: the model
  facts, the token ids for five texts, the chat format, three greedy answers, two
  chat answers, the top scores, and the streaming callback. `./build.sh` runs the
  same test, and it also builds the engine in the browser form.

- Quantization. `bananamendy quantize` writes a checkpoint of 8-bit or ternary
  weights, and the engine reads it directly. `bananamendy compare` measures a
  checkpoint against the float checkpoint, and `bananamendy push` sends it to
  Hugging Face with a model card that carries the numbers.
  - The ternary grid is Ternary Weight Networks with a searched threshold and
    separate scales for the two signs, and with the GPTQ error feedback over the
    columns. The calibration uses a forward pass in numpy that agrees with the
    engine to five decimal places.
  - `--method mixed` measures each matrix on its own and gives ternary weights
    only where the answers barely move.
  - Measured: 8 bits give 98% to 100% next-token agreement at 3.8 times smaller.
    Ternary weights everywhere give 22% on Nano at 7.7 times smaller, which is
    not usable. The documentation states this plainly.
- `Matrix` in the engine: a projection is now 32-bit floats, 8-bit codes or
  ternary codes, and the multiplication of ternary codes needs no multiplication
  for each weight.

### Fixed

- `std::time::Instant` panics in WebAssembly. The engine now uses `web_time` for
  that target.

## 1.0.0 — 2026-08-17

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
