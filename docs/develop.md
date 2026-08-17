---
this_file: docs/develop.md
title: Build and release
layout: default
nav_order: 6
permalink: /develop/
---

# Build and release

Four scripts do all of the work. Each script stops at the first error, and each
script tells you why it stopped.

| Script | Function |
|:-------|:---------|
| `./build.sh` | The gate. It builds, tests and inspects everything. |
| `./wasm.sh` | It builds the WebAssembly module and compares it with Python. |
| `./install.sh` | It installs the program and the wheels. |
| `./publish.sh` | It makes a release. |

## The gate

```bash
./build.sh
```

The script does these steps:

1. It confirms that all of the necessary files are present.
2. It confirms that all of the manifests agree about the version.
3. It runs `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test` and the
   documentation tests.
4. It builds the engine again without the default features, which is the form
   that the browser build uses.
5. It builds both wheels, and it inspects the metadata of each one.
6. It installs the wheels in a new temporary environment, and then it runs the
   Python tests there.
7. It runs `./wasm.sh`, which compares the WebAssembly module with Python.
8. It confirms that the repository did not change.

The gate needs no model weights. The Rust tests that need a checkpoint skip
themselves when `ref/` is empty, and `tests/test_parity.py` skips itself without
`transformers`.

Step 8 is important. A release depends on a build that writes into no tracked
file. If the check fails, something wrote into the repository.

The log goes to `build.log.txt`.

To skip the WebAssembly step, for example on a machine without Node:

```bash
BANANAMEND_SKIP_WASM=1 ./build.sh
```

## The WebAssembly module

```bash
./wasm.sh              # build and compare; docs/ does not change
./wasm.sh --refresh    # build, compare, and write the new module into docs/
```

The comparison is the interesting part. The Python extension module writes a
reference file with the model facts, the token ids, the chat format, the greedy
answers and the top scores. Node then gives the same checkpoint to the WebAssembly
module and compares every value. Greedy decoding must give the same tokens, or the
browser and your computer do different work.

The test needs the Nano checkpoint. It finds it with `bananamendy where nano`, or
you can name one:

```bash
BANANAMEND_CHECKPOINT=/path/to/checkpoint ./wasm.sh
```

Without a checkpoint the script builds the module and tests only that it loads.

`docs/assets/wasm/` holds the module that the demonstration page uses, and
`docs/assets/wasm/VERSION` holds its version. `./build.sh` compares that version
with the manifests. Run `./wasm.sh --refresh` after each change to the Rust code,
and then commit the result.

You need these tools: `wasm-pack`, `node`, and the `wasm32-unknown-unknown`
target.

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

## The features of the engine

| Feature | Default | Function |
|:--------|:--------|:---------|
| `std-fs` | yes | Read a checkpoint from a directory. |
| `parallel` | yes | Use all of the cores for the projections, through rayon. |
| `cli` | yes | The `bananamendr` program. |
| `onig` | yes | The C library for regular expressions in the tokenizer. |
| `wasm` | no | The Rust library for regular expressions. A browser needs this one. |

The tokenizer needs exactly one library for regular expressions. Give `onig` for a
computer, or `wasm` for a browser. A browser also has no files and one thread, so
a browser build gives `default-features = false, features = ["wasm"]`.

## Versions

The git tag is the only source of the version number.
`scripts/release-manifests.py` is the only program that may change a version in a
tracked file. It keeps these places identical:

* `Cargo.toml`, and its pin on the path dependency
* `crates/bananamendr-wasm/Cargo.toml`, and its pin
* `bananamendy/pyproject.toml`, and its pin on `bananamendr`
* `bananamendy/src/bananamendy/__init__.py`

```bash
python3 scripts/release-manifests.py check      # print the agreed version
python3 scripts/release-manifests.py sync 1.2.3 # write one version everywhere
```

## A release

```bash
./publish.sh            # a test run: it changes nothing
./publish.sh --real     # a release: it cannot be undone
```

The test run is safe. Do it first, and read the predicted version.

A real release does these steps in this order:

1. It confirms the state: branch `main`, upstream `origin/main`, no conflicts, and
   a remote that is not in front of your work.
2. It predicts the next version. The rule is the same as the rule in
   `gitnextver`: the highest `v` tag plus one patch, or `1.0.0` when the
   repository has no tag.
3. It commits your work, so that the release points to a commit.
4. It runs all of the gates at the version that is in the manifests now.
5. It writes the new version into the manifests, it builds the WebAssembly module
   again, and then it runs `./build.sh`. A change to a file that is not in the
   allowlist stops the release.
6. It calls `uvx gitnextver@1.0.1`. That program makes one commit, one tag, and
   one push.
7. It confirms that the local tag, the remote branch and the remote tag point to
   the same commit.
8. It uploads: first the crate to crates.io, then `bananamendr` to PyPI, then
   `bananamendy` to PyPI. After each upload it waits until the registry serves the
   new version.

{: .warning }
> Do not commit the version change yourself. `gitnextver` makes a tag only when
> the repository has a change to commit. The version change is that change.

A real release needs two tokens in the environment:

```bash
export CARGO_REGISTRY_TOKEN=...
export UV_PUBLISH_TOKEN=...
```

`crates/bananamendr-wasm` and `crates/bananamendr-py` have `publish = false` for
crates.io. A WebAssembly module and a Python extension module have no use as a
Rust dependency; both go to PyPI or to the site instead.

If an upload fails, the script prints the artifacts that it published before the
failure. You cannot remove them from the registries, so the next release must use
a new version.

## The documentation site

The site is the `docs/` directory. GitHub Pages builds it with Jekyll, and it
loads the [Just the Docs](https://just-the-docs.com/) theme from GitHub. The
demonstration page loads [daisyUI](https://daisyui.com/) from a CDN. The
repository holds no copy of either one.

To see the site on your computer:

```bash
cd docs
bundle init
bundle add jekyll jekyll-remote-theme jekyll-seo-tag
bundle exec jekyll serve --baseurl ""
```

The demonstration page needs a server, because a browser refuses to load a
WebAssembly module from a `file:` address. The Jekyll server above gives the
complete page. A plain static server also serves the module and the scripts:

```bash
python3 -m http.server --directory docs 8000
```

That server does not run Jekyll, so it shows the Markdown files as text.
