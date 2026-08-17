---
this_file: docs/install.md
title: Installation
layout: default
nav_order: 3
permalink: /install/
---

# Installation

## Python: the complete set

```bash
uv pip install bananamendy
```

`pip install bananamendy` also works. The command installs `bananamendr`, which
is the engine. The wheel needs no compiler, because one wheel serves all CPython
versions from 3.9.

Make sure that it works:

```bash
bananamendy registry
python -c "import bananamendr; print(hasattr(bananamendr, 'Model'))"
```

## Rust: the command line program only

```bash
cargo install bananamendr
bananamendr --version
```

This program takes a path to a checkpoint. It downloads nothing. Get a checkpoint
with `bananamendy pull`, or with `scripts/fetch_models.sh` in the repository.

For your own Rust code:

```toml
[dependencies]
bananamendr = "1"
```

The default features read files and use all of the cores. A build for a browser
switches both off:

```toml
[dependencies]
bananamendr = { version = "1", default-features = false, features = ["wasm"] }
```

## From this repository

```bash
git clone https://github.com/twardoch/bananamend
cd bananamend
./install.sh
```

The script installs the `bananamendr` program into `~/.local/bin`, and it installs
both wheels into the interpreter that `uv` finds as the system Python. It tests
the wheels in a temporary environment first, and it makes sure that it changes no
file in the repository.

Other locations:

```bash
./install.sh --prefix /usr/local --python /opt/homebrew/bin/python3.12
```

## For development

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ./crates/bananamendr-py -e ./bananamendy pytest httpx
python -m pytest bananamendy/tests -q
```

## Get the models

| Alias | Hugging Face repository | Size |
|:------|:------------------------|:-----|
| `nano` | `BananaMind/BananaMind-2-Nano-Chat` | 40 MB |
| `mini` | `BananaMind/BananaMind-2-Mini-Chat` | 101 MB |
| `pro` | `BananaMind/BananaMind-2-Pro-Preview-Chat` | 556 MB |

```bash
bananamendy pull nano
bananamendy models          # what is in the cache now
bananamendy where nano      # the directory of one model
```

The weights go into the usual Hugging Face cache. `HF_HOME` and `HF_HUB_CACHE`
control the location, and a model that you already have is not downloaded again.
Any repository name works, and so does a path to a directory:

```bash
bananamendy info --name /path/to/my-checkpoint
bananamendy chat --name SomeOrg/Some-Chat-Model --prompt "Hello"
```

The engine reads `config.json`, `model.safetensors`, `tokenizer.json` and, if it
is present, `tokenizer_config.json`. All of the tensors must be F32. The engine
refuses a different type; it does not convert it.

## Configuration

```bash
bananamendy init_config     # write the TOML file
bananamendy config          # show the values in use, and the location of the file
```

`platformdirs` decides the location. On macOS it is
`~/Library/Application Support/bananamendy/config.toml`. On Linux it is
`~/.config/bananamendy/config.toml`.

Each value also has an environment variable. The name is the field in capital
letters with the prefix `BANANAMENDY_`:

```bash
BANANAMENDY_MODEL=mini BANANAMENDY_PORT=9000 bananamendy serve
```

The order is: the value that you give in the command, then the environment, then
the TOML file, then the value in the program.
