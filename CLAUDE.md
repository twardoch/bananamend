# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Local inference for the BananaMind-2 chat checkpoints (Nano, Mini, Pro Preview).
The engine is Rust, and it reads the published `model.safetensors` and
`tokenizer.json` directly. PyTorch is not necessary at any time.

| Product | Language | Registry | Function |
|:--------|:---------|:---------|:---------|
| `bananamendr` | Rust (library and program, plus a PyO3 abi3-py39 module) | crates.io and PyPI | the engine and the command line |
| `bananamendr-wasm` | Rust (WebAssembly) | none; `publish = false` | the same engine in a browser |
| `bananamendy` | Python (Fire, FastAPI) | PyPI | downloads, configuration, and the server |

**The division of work is the main design decision here: `bananamendr` takes a
path and nothing else, and it makes no network request.** Downloads, a cache, a
configuration file and the server are in `bananamendy`. Do not add HTTP or a model
registry to the Rust side.

## Names: what you may rename, and what you may not

The packages are `bananamend*`. The **model identifiers are from upstream, and you
must not rename them**: `BananaMind2NanoForCausalLM`,
`model_type = "bananamind2_nano"`, and the `BananaMind/BananaMind-2-*-Chat`
repository names must all match the published `config.json`, or nothing loads.
`crates/bananamendr/src/config.rs`, `crates/bananamendr/tests/checkpoint.rs`,
`bananamendy/src/bananamendy/models.py` and the demonstration page all depend on
that.

## Commands

```bash
./build.sh                        # the gate: fmt, clippy, tests, wheels, pytest, WASM parity
BANANAMEND_SKIP_WASM=1 ./build.sh # the same, without the WebAssembly step
./wasm.sh                         # build the WebAssembly module and compare it with Python
./wasm.sh --refresh               # the same, and write the module into docs/assets/wasm/
./install.sh                      # the program into ~/.local/bin, the wheels into system Python
./publish.sh                      # a test run of a release
./publish.sh --real               # a release; it cannot be undone
scripts/fetch_models.sh nano      # checkpoints into ref/ for the Rust tests (needs git-lfs)
```

Narrow loops:

```bash
cargo test --workspace
cargo test -p bananamendr --doc
cargo build -p bananamendr --no-default-features --features wasm   # the browser form
cargo clippy --all-targets --no-deps -- -D warnings
cargo run --release -p bananamendr -- info -m "$(bananamendy where nano)"

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ./crates/bananamendr-py -e ./bananamendy pytest httpx
.venv/bin/python -m pytest bananamendy/tests -q
.venv/bin/python -m pytest bananamendy/tests/test_server.py::test_chat_completion_when_streaming_then_sse_with_done -q
BANANAMEND_CHECKPOINT=/path/to/checkpoint ./wasm.sh
```

`build.sh` needs no weights, and it fails if it leaves the checkout modified. The
release runs the build two times and compares the tree, so that property is
load-bearing.

## Architecture

**The engine** (`crates/bananamendr/src/`): `weights.rs` reads the safetensors
(all published tensors are F32; `load` maps a file, `from_bytes` takes bytes),
`config.rs` parses `config.json`, `model.rs` is the decoder (RMSNorm pre-norm
blocks, grouped-query attention, RoPE), `ops.rs` holds the kernels, `sample.rs`
the sampling and the random numbers, `tokenizer.rs` wraps `tokenizers` and renders
the chat format, and `lib.rs` composes them into `Pipeline`. `main.rs` is the
command line in the same crate, so `cargo install bananamendr` gives the program.

**The features are what make three targets possible.** `std-fs` gives the file
loaders. `parallel` gives rayon. `cli` gives the program. The tokenizer needs
exactly one library for regular expressions: `onig` (C, the default) or `wasm`
(Rust, for a browser). A browser build is
`--no-default-features --features wasm`, and it uses `Pipeline::from_parts`, which
takes the four parts of a checkpoint in memory.

Two more browser rules, both learned from a failure:

- `std::time::Instant` panics in WebAssembly. `lib.rs` uses `web_time::Instant`
  for `target_arch = "wasm32"`. Do not put a bare `std::time` call in the engine.
- `getrandom` needs a backend in a browser. `.cargo/config.toml` gives
  `--cfg getrandom_backend="wasm_js"`, and the WebAssembly crate depends on
  `getrandom` with the `wasm_js` feature for that target.

**The extension module** (`crates/bananamendr-py/`): `Model` wraps `Pipeline`.
Without an `on_token` callback the interpreter lock is released for the complete
decode; with one the lock is held, and the callback receives
`(decoded_delta, token_id)` for each step. `publish = false` for crates.io,
because a `cdylib` has no use as a Rust dependency.

**The WebAssembly bindings** (`crates/bananamendr-wasm/`): the same shape as the
Python module, and deliberately so. `logits` must not add a token that the Python
module does not add, and `applyChatTemplate` must give the text while `chatTokens`
gives the ids. The parity test compares both sides, and a difference in this file
is what it finds.

**The Python side** (`bananamendy/src/bananamendy/`):

- `models.py` — the alias table, `resolve()` and `pull()`. A path that exists wins
  over the alias table, then the Hugging Face cache is read **offline first**, and
  only then is a download attempted. Weights go in the ordinary cache.
- `config.py` — a frozen dataclass; TOML under `platformdirs`, then
  `BANANAMENDY_*` from the environment. `Config.sampling()` is the bridge to the
  keyword names of `bananamendr`. A new control goes there.
- `engine.py` — it keeps each loaded checkpoint by resolved path, and it lets
  **one generation run at a time**. Streaming runs the generation on a worker
  thread whose callback puts the pieces into a `queue.Queue`; an error on the
  worker is raised again in the consumer after the pieces that it already gave.
- `server.py` — the OpenAI interface. `create_app(config, engine)` takes an
  engine, so a test injects a false one and needs no weights. A parameter that a
  request does not give comes from the configuration.
- `cli.py` — Fire. Each command takes an optional model name.

One generation at a time is a property of the interpreter lock, and not an
oversight. Do not add a pool of workers and expect more speed.

## Quantization

A quantized checkpoint keeps its codes in memory and rebuilds each value inside
the multiplication. `crates/bananamendr/src/matrix.rs` holds that: `Matrix` is
`Dense`, `Int8` or `Ternary`, and `model.rs` stores a `Matrix` for every
projection and for the embedding. The form comes from the `quantization` block of
`config.json`, which `config.rs` parses and `weights.rs` acts on. A tensor that
the block does not name must be present as 32-bit floats.

Ternary codes are two bits, four in a byte, from the lowest bits upwards: 0 is
minus one, 1 is zero, 2 is plus one. Each group of columns carries two scales, one
for each sign. The ternary multiplication adds the inputs of the positive codes,
subtracts the inputs of the negative codes, and multiplies twice for each group;
it never multiplies for a single weight.

The Python side is four files:

- `quantize.py` — the grids. `quantize_matrix` is Ternary Weight Networks with a
  searched threshold and two scales; `quantize_matrix_calibrated` adds the GPTQ
  error feedback; `quantize_matrix_int8` is symmetric 8-bit. `quantize_checkpoint`
  writes the checkpoint and the report, and a `plan` decides the grid of each
  tensor.
- `reference.py` — a forward pass in numpy, **for calibration only**. It agrees
  with the engine to five decimal places, and `test_rust_agreement` in the wasm
  parity work keeps it honest. It exists because the engine does not report the
  input of each matrix.
- `plan.py` — the mixture. It measures each matrix on its own, sorts by the
  change of the answers, and accepts ternary forms while the total change stays
  inside a budget. It measures after each acceptance, because two ternary
  matrices are worse than the sum of the two alone.
- `evaluate.py` and `calibration.py` — the measurement, and the two texts. The
  calibration text and the measurement text must stay different.

**The finding, so that nobody repeats the work:** ternary weights everywhere give
22% next-token agreement on Nano and a perplexity ten times higher. Eight bits
give 98% to 100% and a perplexity within 1%. Do not present a fully ternary
checkpoint of these models as usable.

## The WebAssembly module and the site

`docs/` is a Jekyll site for GitHub Pages. It loads the Just the Docs theme with
`remote_theme`, and the demonstration page loads daisyUI from a CDN. The repository
holds no copy of either one. daisyUI gives the components only; the few Tailwind
utility classes that the markup uses are written in `docs/assets/css/demo.css`.

`docs/assets/wasm/` holds the prebuilt module, and `docs/assets/wasm/VERSION` holds
its version. `build.sh` compares that version with the manifests and fails if they
differ. `./wasm.sh --refresh` writes a new module; commit the result.

The demonstration page downloads the Nano checkpoint from
`https://huggingface.co/BananaMind/BananaMind-2-Nano-Chat/resolve/main`. Hugging
Face answers those requests with permissive CORS headers, which is what makes the
page possible.

## Release mechanics

The **git tag is the only source of the version number**.
`scripts/release-manifests.py` is the only program that may change a version in a
tracked file. It keeps the workspace version and its pin, the WebAssembly pin,
`bananamendy/pyproject.toml` (its own version and the `bananamendr==` pin) and
`bananamendy/__init__.py` identical. Each version site must match exactly one
line.

`publish.sh` holds these rules:

- branch `main`, upstream `origin/main`, a remote that is not in front, no conflicts
- the next version follows `gitnextver`: the highest `v` tag plus one patch, or
  `1.0.0` when the repository has no tag
- the gates run at the current version; then the manifests move to the target
  version; then the WebAssembly module is built again; then `build.sh` runs. A
  change to a file outside the allowlist stops the release. The allowlist is a
  limit and not an exact list, because a rebuild can give identical bytes.
- `uvx gitnextver@1.0.1` makes the one commit, the one tag and the one push. It
  makes a tag only when the tree is dirty, and the version change is that change.
  Do not commit the version change yourself.
- the uploads are last and in order: crates.io `bananamendr`, PyPI `bananamendr`,
  PyPI `bananamendy`. The script waits for each registry.

`CARGO_REGISTRY_TOKEN` and `UV_PUBLISH_TOKEN` are necessary for `--real`.

Two traps that cost time before:

- `comm` needs `LC_ALL=C` when the input came from `LC_ALL=C sort`. Without it the
  allowlist check reports a file that is in the list.
- cargo does not always rebuild a WebAssembly artifact after a version change.
  `wasm.sh` removes the previous artifact of that crate first.

Before a real release, test the target version in a clone: tag it, run `sync`,
commit, and run `build.sh`. The normal test run only builds the current version.

## Style rules for this repository

- Prose in Markdown files uses ASD-STE100 Simplified Technical English: short
  sentences, the active voice, one instruction in one sentence, and no idiom.
- Every source file holds a `this_file:` line near the top. Update it when you
  move a file.
- Tests must not need a network. Tests that need weights skip themselves.
- MSRV is 1.87, edition 2024. Clippy runs with `-D warnings`, and
  `clippy::incompatible_msrv` catches a newer API.

## `_priv/` is private

`_priv/` holds the first `bananamind` prototype and, under `_priv/**/ref/`,
downloaded checkpoints and other people's repositories with their own licences. Git
ignores it. Do not copy weights, `ref/`, or the prototype notes (`WORK.md`,
`IDEA.md`, `ANALYSIS.md`, `PLAN.md`, `TODO.md`) into the public tree. Use
`bananamendy pull` or `scripts/fetch_models.sh` to get checkpoints.
