<!--
this_file: README.md
-->

# bananamend

**Local inference for the BananaMind-2 chat models. Rust engine, Python command
line, a server with the OpenAI interface, and a build that runs in a browser.**

The models are small: Nano is 40 MB, Mini is 230 MB, and Pro (Preview) is 1.1 GB.
The engine reads the published `model.safetensors` and `tokenizer.json` directly,
so there is no conversion step and no second copy of the weights. PyTorch is not
necessary at any time.

Greedy decoding gives the same tokens as Hugging Face `transformers` for all three
models. That agreement is a test in this repository (`tests/test_parity.py`), and
not only a statement.

Documentation and a browser demonstration: <https://code.twardoch.com/bananamend/>

## Three products

| Product | Language | Registry | Function |
|:--------|:---------|:---------|:---------|
| [`bananamendr`](https://crates.io/crates/bananamendr) | Rust | crates.io and PyPI | the engine, the `bananamendr` program, and the Python module |
| [`bananamendy`](https://pypi.org/project/bananamendy/) | Python | PyPI | downloads, configuration, a Fire command line, and the server |
| the browser build | WebAssembly | none | the demonstration page in `docs/` |

The division of work is deliberate: **`bananamendr` takes a path and nothing else.
It makes no network request.** All of the convenient parts — downloads, a cache, a
configuration file, a server — are in `bananamendy`.

## Install

```bash
uv pip install bananamendy          # the command line, the server, and the engine
cargo install bananamendr           # only the Rust program
```

From this repository:

```bash
./install.sh                        # the program into ~/.local/bin, the wheels into system Python
```

## Use

```bash
bananamendy pull nano                      # about 40 MB into the Hugging Face cache
bananamendy models                         # what is in the cache
bananamendy info                           # the architecture facts
bananamendy chat --prompt "Why is the sky blue?"
bananamendy chat                           # a conversation; an empty line ends it
bananamendy generate --prompt "Once upon a time"
bananamendy logits --text "The capital of France is"
bananamendy bench
bananamendy serve                          # OpenAI interface on 127.0.0.1:8377
```

The aliases `nano`, `mini` and `pro` become the `BananaMind/BananaMind-2-*-Chat`
repositories. Any repository name works, and so does a path to a directory. The
weights go into the usual Hugging Face cache, so `HF_HOME` and `HF_HUB_CACHE`
control the location, and a model that you already have is not downloaded again.

Configuration is a TOML file in the `platformdirs` location.
`bananamendy init_config` writes it, `bananamendy config` shows the values in use,
and each value has a `BANANAMENDY_*` environment variable.

### The server

```bash
curl http://127.0.0.1:8377/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "nano", "messages": [{"role": "user", "content": "Hi"}], "stream": true}'
```

The server answers `GET /v1/models`, `POST /v1/chat/completions`,
`POST /v1/completions` and `GET /health`. Both `POST` paths accept
`"stream": true`, and the answer is then a sequence of server-sent events.

A parameter that a request does not give comes from your configuration, and not
from the defaults of OpenAI. The server and the command line then behave in the
same way.

The server answers **one request at a time**. A generation holds the Python
interpreter lock for the complete decode, so two generations in one process cannot
run together. A pool of workers would give no more speed and would use more
memory. Streaming still works, because the generation runs on a worker thread and
its callback puts each piece of text into a queue.

### Python

```python
import bananamendr

model = bananamendr.Model("/path/to/BananaMind-2-Nano-Chat")
generation = model.chat([{"role": "user", "content": "Hi"}], max_new_tokens=64)
print(generation.text, generation.tokens_per_second)

model.generate("Once upon a time", on_token=lambda text, token: print(text, end=""))
```

The Python interface is greedy by default. The two command line programs sample by
default (`--temperature 0.8 --top-k 40 --top-p 0.95 --repetition-penalty 1.1`),
because an interactive answer must not repeat itself.

### Rust

```sh
scripts/fetch_models.sh nano        # into ref/, or use bananamendy pull
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

`Pipeline::from_parts` takes the four parts of a checkpoint in memory. The browser
build uses that function, because a browser has no files.

### The browser

The [demonstration page](https://code.twardoch.com/bananamend/demo/) downloads the
Nano model from Hugging Face and then runs the engine in your browser. Nothing goes
to a server of that site.

The features of the engine make this possible:

| Feature | Default | Function |
|:--------|:--------|:---------|
| `std-fs` | yes | Read a checkpoint from a directory. |
| `parallel` | yes | Use all of the cores, through rayon. |
| `cli` | yes | The `bananamendr` program. |
| `onig` | yes | The C library for regular expressions in the tokenizer. |
| `wasm` | no | The Rust library for regular expressions, for a browser. |

```toml
[dependencies]
bananamendr = { version = "1", default-features = false, features = ["wasm"] }
```

## Small checkpoints

`bananamendy quantize` writes a copy of a checkpoint that needs 1 byte, or a
quarter of a byte, for each weight. The engine reads that copy directly, and it
keeps the codes in memory, so the memory in use is as small as the file.

```bash
bananamendy quantize /tmp/nano-int8 --name nano --method int8
bananamendy chat --name /tmp/nano-int8 --prompt "Name one ocean."
bananamendy compare /tmp/nano-int8 --name nano
```

Ready-made checkpoints are on Hugging Face under
[fontlab](https://huggingface.co/fontlab):

```bash
bananamendy chat --name fontlab/BananaMind-2-Pro-Preview-Chat-int8 --prompt "Hi"
```

| Checkpoint | Method | File | Weights in memory | Speed | Same next token | Perplexity | Identical greedy answers |
|:-----------|:-------|:-----|:------------------|:------|:----------------|:-----------|:-------------------------|
| Nano (10 M) | floats | 39.9 MB | 39.9 MB | 611 tokens/s | — | 66.3 | — |
| Nano | 8 bits | 10.6 MB (3.8×) | 10.6 MB | 302 tokens/s | 97.9% | 67.2 | 4 of 8 |
| Nano | mixed (3 ternary) | 10.5 MB (3.8×) | 10.5 MB | 261 tokens/s | 96.8% | 67.5 | 3 of 8 |
| Nano | ternary (70) | 5.2 MB (7.7×) | 5.2 MB | 34 tokens/s | 22.1% | 709.9 | 0 of 8 |
| Pro (139 M) | floats | 555.9 MB | 555.9 MB | 71 tokens/s | — | 33.3 | — |
| Pro | 8 bits | 147.8 MB (3.8×) | 147.8 MB | 54 tokens/s | 100.0% | 33.3 | 6 of 8 |
| Pro | mixed (12 ternary) | 144.9 MB (3.8×) | 144.9 MB | 46 tokens/s | 90.5% | 33.3 | 3 of 8 |
| Pro | ternary (168) | 66.7 MB (8.3×) | 66.7 MB | 11 tokens/s | 41.3% | 131.1 | 0 of 8 |

Speed is one thread group on an Apple M4 Max, greedy, 32 new tokens. The two
ternary rows for Pro use 64 tokens of the measurement text, and the other rows
use 96, so compare a perplexity only inside one model.

`bananamendy info --name <checkpoint>` reports the form and the memory, so those
two columns are checkable:

```
$ bananamendy info --name fontlab/BananaMind-2-Nano-Chat-int8
storage:    {"embedding": "int8", "matrices_int8": "70"}
weight_mb:  10.61
```

**Eight bits are nearly free. Ternary weights everywhere are not usable at these
sizes.** The ternary grid holds three values, minus one, zero and plus one, with
two scales for each group of 64 weights. The quantizer searches the threshold of
each group for the smallest error, gives the two signs separate scales, and moves
the error of each column into the columns that follow (GPTQ). Even so, a model of
10 to 139 million weights cannot carry ternary weights everywhere: the published
work either trains with the ternary grid from the start, or works above one
billion parameters.

`--method mixed` therefore measures each matrix on its own and gives ternary
weights only where the answers barely move. See
[Small checkpoints](https://code.twardoch.com/bananamend/quantization/) for the
format, the method and the complete numbers.

## Development

```bash
./build.sh      # the gate: format, lint, test, wheels, Python tests, WASM parity
./wasm.sh       # build the WebAssembly module and compare it with Python
./install.sh    # install the program and the wheels
./publish.sh    # a test run of a release
./publish.sh --real
```

The gate needs no model weights: the Rust tests that need a checkpoint skip
themselves when `ref/` is empty, and the parity test against `transformers` skips
itself without `torch`. `build.sh` fails if it changes any file in the repository.

The WebAssembly parity test does need the Nano checkpoint. It compares the browser
build with the Python module on the model facts, the token ids, the chat format,
the greedy answers and the top scores. Run `./wasm.sh --refresh` after each change
to the Rust code, and commit the new module in `docs/assets/wasm/`.

See [Build and release](https://code.twardoch.com/bananamend/develop/) for the
complete description.

## Files

```
crates/bananamendr/       the engine and the `bananamendr` program
crates/bananamendr-py/    the Python extension module, published as `bananamendr`
crates/bananamendr-wasm/  the WebAssembly bindings
bananamendy/              the Python package: models, config, engine, server, CLI
docs/                     the documentation site and the prebuilt WebAssembly module
tests/                    parity against transformers, and parity against the browser build
scripts/                  versions, artifact inspection, and model download
ref/                      checkpoints; git ignores them
```

## Licence

Apache-2.0
