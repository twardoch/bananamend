<!--
this_file: README.md
-->

# bananamend

Local CPU inference for the **BananaMind-2** chat checkpoints — Nano, Mini and
Pro (Preview) — written in Rust, callable from Python, with no PyTorch at
runtime. It reads the published `model.safetensors` and `tokenizer.json`
directly: no conversion step, no second weight format.

Greedy decoding is **token-exact against Hugging Face `transformers`** for all
three checkpoints; that equivalence is a test, not a claim
(`tests/test_parity.py`).

| Package | Language | Registries | Role |
| --- | --- | --- | --- |
| `bananamendr` | Rust | [crates.io](https://crates.io/crates/bananamendr) · [PyPI](https://pypi.org/project/bananamendr/) | the runtime, the `bananamendr` CLI, and the Python extension module |
| `bananamendy` | Python | [PyPI](https://pypi.org/project/bananamendy/) | checkpoint fetching, config, Fire CLI, OpenAI-compatible server |

The split is deliberate: **`bananamendr` takes explicit paths and never touches
the network**; everything convenient — downloads, caching, config files, a
server — lives in `bananamendy`.

## Install

```bash
uv pip install bananamendy          # CLI + server + the extension module
cargo install bananamendr           # just the Rust CLI
```

From this checkout:

```bash
./install.sh                        # CLI into ~/.local/bin, wheels into system Python
```

## Python

```bash
bananamendy pull nano                      # into the Hugging Face cache
bananamendy models                         # what is cached locally
bananamendy info                           # architecture facts
bananamendy chat --prompt "Why is the sky blue?"
bananamendy chat                           # REPL
bananamendy generate --prompt "Once upon a time"
bananamendy logits --text "The capital of France is"
bananamendy bench
bananamendy serve                          # OpenAI-compatible on 127.0.0.1:8377
```

Aliases `nano`, `mini`, `pro` expand to the `BananaMind/BananaMind-2-*-Chat`
repos; any repo id or local directory works too. Weights live in the ordinary
Hugging Face cache (`HF_HOME` / `HF_HUB_CACHE` are respected), so a checkpoint
you already have is not downloaded twice.

Configuration is TOML in the platformdirs location — `bananamendy init_config`
writes it, `bananamendy config` shows the effective values, and `BANANAMENDY_*`
environment variables override it.

### The server

```bash
curl http://127.0.0.1:8377/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "nano", "messages": [{"role": "user", "content": "Hi"}], "stream": true}'
```

`GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions` (both with
SSE streaming) and `GET /health`. Sampling parameters absent from a request fall
back to your config rather than to OpenAI's defaults, so the server and the CLI
behave the same.

Requests are served **one at a time**: a CPU decode holds the GIL, so a worker
pool would not decode in parallel. Streaming still works while a generation is in
flight because deltas are handed to the response through a queue.

### Library

```python
import bananamendr

model = bananamendr.Model("/path/to/BananaMind-2-Nano-Chat")
generation = model.chat([{"role": "user", "content": "Hi"}], max_new_tokens=64)
print(generation.text, generation.tokens_per_second)

model.generate("Once upon a time", on_token=lambda text, token: print(text, end=""))
```

The Python API defaults to greedy decoding; the CLIs sample
(`--temperature 0.8 --top-k 40 --top-p 0.95 --repetition-penalty 1.1`), because
interactive output should not loop.

## Rust

```sh
scripts/fetch_models.sh nano        # into ref/ (needs git-lfs), or use bananamendy pull
cargo run --release -p bananamendr -- info -m ref/BananaMind-2-Nano-Chat
```

```rust
use bananamendr::{GenerateOptions, Message, Pipeline};

let pipeline = Pipeline::from_dir("ref/BananaMind-2-Nano-Chat".as_ref())?;
let generation = pipeline.chat(
    &[Message::new("user", "Why is the sky blue?")],
    &GenerateOptions::default(),
)?;
```

## Layout

```
crates/bananamendr/      runtime library + `bananamendr` CLI binary
crates/bananamendr-py/   PyO3 extension module, published to PyPI as `bananamendr`
bananamendy/             Python package: models, config, engine, server, CLI
tests/test_parity.py     token-exactness against transformers (skips if absent)
ref/                     checkpoints, git-ignored
scripts/                 release version graph, artifact inspection, model fetch
```

## Development

```bash
./build.sh      # fmt, clippy, cargo test + doctests, both wheels, pytest, smoke
./install.sh    # install the CLI and the built wheels
./publish.sh    # dry run: full gates + predicted version
./publish.sh --real
```

`build.sh` needs no weights: checkpoint-dependent Rust tests skip themselves when
`ref/` is empty, and the parity suite skips without `transformers`. It fails if it
modifies the checkout. `publish.sh` requires a clean, pushed `main`; it syncs the
version across manifests, tags once via `gitnextver`, then uploads —
`bananamendr` to crates.io and PyPI, then `bananamendy` to PyPI.

## License

Apache-2.0
