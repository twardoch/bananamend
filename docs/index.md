---
this_file: docs/index.md
title: Home
layout: default
nav_order: 1
---

# bananamend

bananamend runs the **BananaMind-2** chat models on your own computer. The models
are small: Nano is 40 MB, Mini is 230 MB, and Pro (Preview) is 1.1 GB.

The engine is Rust. It reads the published `model.safetensors` and
`tokenizer.json` directly, so there is no conversion step and no second copy of
the weights. PyTorch is not necessary at any time.

Greedy decoding gives the same tokens as Hugging Face `transformers`. That
agreement is a test in this repository, and not only a statement
(`tests/test_parity.py`).

## Three ways to use it

| Product | Language | Installation | Use it for |
|:--------|:---------|:-------------|:-----------|
| [`bananamendr`](https://crates.io/crates/bananamendr) | Rust | `cargo install bananamendr` | a command line program, and a library for your own Rust code |
| [`bananamendy`](https://pypi.org/project/bananamendy/) | Python | `uv pip install bananamendy` | model downloads, a command line program, and a server with the OpenAI interface |
| the browser build | WebAssembly | nothing to install | the [demonstration page](demo/) |

The division of work is deliberate: **`bananamendr` takes a path and nothing
else. It makes no network request.** All of the convenient parts — downloads, a
cache, a configuration file, a server — are in `bananamendy`.

## Start here

```bash
uv pip install bananamendy
bananamendy pull nano                        # about 40 MB into the Hugging Face cache
bananamendy chat --prompt "Name one ocean."
bananamendy serve                            # OpenAI interface on 127.0.0.1:8377
```

Then point any OpenAI client at the server:

```bash
curl http://127.0.0.1:8377/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "nano", "messages": [{"role": "user", "content": "Hi"}]}'
```

## The demonstration page

The [demonstration page](demo/) runs the same engine in your browser. The page
downloads the Nano model from Hugging Face, and then all of the work stays on
your computer. No text goes to a server.

## Speed

Nano writes approximately 500 tokens each second on an Apple M4 Max, with all of
the cores. The browser build has one thread, and it is therefore slower. The
larger models are slower in proportion to their size.

Measure your own computer:

```bash
bananamendy bench
bananamendy bench --name mini
```

## Next steps

* [Install bananamend](install/)
* [Use the command line](cli/)
* [Use the server](server/)
* [Try the browser demonstration](demo/)
* [Build and release bananamend](develop/)
