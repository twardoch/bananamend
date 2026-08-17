# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

CPU inference for the BananaMind-2 chat checkpoints (Nano, Mini, Pro Preview) in
Rust, callable from Python, with no PyTorch at runtime. It reads the published
`model.safetensors` and `tokenizer.json` directly.

| Package | Language | Registries | Role |
| --- | --- | --- | --- |
| `bananamendr` | Rust (lib + bin, PyO3 abi3-py39) | crates.io + PyPI | runtime, `bananamendr` CLI, extension module |
| `bananamendy` | Python (Fire CLI, FastAPI) | PyPI | downloads, config, engine, OpenAI-compatible server |

**The division of labour is the main design decision here: `bananamendr` takes
explicit paths and never touches the network. Downloading, caching, config files
and the server live in `bananamendy`.** Don't add HTTP or a model registry to the
Rust side.

## Names: what may be renamed and what may not

The packages are `bananamend*`. The **model identifiers are upstream and must not
be renamed**: `BananaMind2NanoForCausalLM`, `model_type = "bananamind2_nano"`,
and the `BananaMind/BananaMind-2-*-Chat` Hugging Face repo ids all have to match
the published `config.json` or nothing loads. `crates/bananamendr/src/config.rs`
and `crates/bananamendr/tests/checkpoint.rs` both depend on that.

## Commands

```bash
./build.sh                  # the gate: fmt, clippy, cargo test + doctests, wheels, pytest, smoke
./install.sh                # CLI into ~/.local/bin, wheels into uv's system Python
./publish.sh                # dry run: preflight + all gates + predicted version
./publish.sh --real         # irreversible: tag, push, upload to crates.io + PyPI
scripts/fetch_models.sh nano  # checkpoints into ref/ for the Rust tests (needs git-lfs)
```

Narrower loops:

```bash
cargo test --workspace
cargo test -p bananamendr --doc
cargo clippy --all-targets --no-deps -- -D warnings
cargo run --release -p bananamendr -- info -m ref/BananaMind-2-Nano-Chat

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ./crates/bananamendr-py -e ./bananamendy pytest httpx
.venv/bin/python -m pytest bananamendy/tests -q
.venv/bin/python -m pytest bananamendy/tests/test_server.py::test_chat_completion_when_streaming_then_sse_with_done -q
```

`build.sh` needs no weights, and it fails if it leaves the checkout modified —
the release flow depends on the build being a pure function of the tree.

## Architecture

**Rust side** (`crates/bananamendr/src/`): `weights.rs` memory-maps the
safetensors (all published BananaMind-2 tensors are F32), `config.rs` parses
`config.json`, `model.rs` is the decoder (RMSNorm pre-norm blocks, grouped-query
attention, RoPE), `ops.rs` holds the rayon-parallel kernels, `sample.rs` the
sampling and RNG, `tokenizer.rs` wraps `tokenizers` plus the chat template, and
`lib.rs` composes them into `Pipeline`. `main.rs` is the CLI in the same crate so
`cargo install bananamendr` ships the binary.

**Extension module** (`crates/bananamendr-py/`): `Model` wraps `Pipeline`.
Without an `on_token` callback the GIL is released for the whole decode; with one
the GIL is held and the callback is invoked as `(decoded_delta, token_id)` per
step. `publish = false` for crates.io — a `cdylib` is not usable as a Rust
dependency; it ships as a wheel only.

**Python side** (`bananamendy/src/bananamendy/`):

- `models.py` — alias table, `resolve()` and `pull()`. A path that exists wins
  over the alias table, then the HF cache is consulted **offline first**, and only
  then is a download attempted. Weights go in the ordinary HF cache, not a
  bananamend-specific directory.
- `config.py` — frozen dataclass; TOML under `platformdirs`, then `BANANAMENDY_*`
  environment overrides. `Config.sampling()` is the bridge to `bananamendr`'s
  keyword names; if you add a knob, add it there.
- `engine.py` — caches loaded checkpoints by resolved path and **serialises every
  generation on one lock**. Streaming runs the generation on a worker thread whose
  callback pushes deltas into a `queue.Queue`; errors raised on the worker are
  re-raised in the consumer after the deltas already produced.
- `server.py` — the OpenAI-compatible surface, built by `create_app(config,
  engine)` so tests inject a fake engine and need no weights. Request parameters
  override config defaults, not OpenAI's.
- `cli.py` — Fire; every command takes an optional model name and falls back to
  config.

Single-flight generation is a property of CPU inference in one process, not an
oversight. Do not add a worker pool expecting parallel decodes.

## Release mechanics

The **git tag is the single source of truth** for the version.
`scripts/release-manifests.py` is the only thing allowed to move version numbers
in tracked files; it keeps the workspace version, the `bananamendr` path-dependency
pin, `bananamendy/pyproject.toml` (its own version and the `bananamendr==` pin)
and `bananamendy/__init__.py` identical. `publish.sh` enforces:

- branch `main`, upstream `origin/main`, remote not ahead, no unresolved conflicts
- next version mirrors `gitnextver`: highest `v*` tag plus one patch, or `1.0.0`
  when no tags exist
- gates run at the *current* version, then manifests sync to the *target*
  version, then `build.sh` runs again; a diff outside the manifest allowlist aborts
- `uvx gitnextver@1.0.1` makes the single commit + tag + push. It only tags when
  the tree is dirty, which the manifest sync guarantees — do not commit the sync
  yourself, or nothing gets tagged
- uploads last, in dependency order: crates.io `bananamendr`, PyPI `bananamendr`,
  PyPI `bananamendy`, each waited on until the registry serves it

`CARGO_REGISTRY_TOKEN` and `UV_PUBLISH_TOKEN` are required for `--real`.

## Conventions

- Every source file carries a `this_file:` path record near the top. Update it
  when moving files.
- MSRV is 1.87 (`u32::is_multiple_of`), edition 2024. Clippy runs with
  `-D warnings`, and `clippy::incompatible_msrv` will catch a newer API.
- `pyo3/extension-module` must stay out of the default Cargo features; maturin
  enables it per build.
- Tests must not need weights or a network. Rust checkpoint tests skip when
  `ref/` is empty; `tests/test_parity.py` skips without `transformers`;
  `bananamendy` tests stub the model and the hub.

## `_priv/` is private and stays private

`_priv/` holds the original `bananamind` prototype and, under `_priv/**/ref/`,
downloaded checkpoints and upstream repositories with their own licenses. It is
git-ignored; do not copy weights, `ref/`, or prototype notes (`WORK.md`,
`IDEA.md`, `ANALYSIS.md`, `PLAN.md`, `TODO.md`) into the public tree. Use
`bananamendy pull` or `scripts/fetch_models.sh` to get checkpoints instead.
