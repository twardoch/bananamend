---
this_file: docs/cli.md
title: Command line
layout: default
nav_order: 4
permalink: /cli/
---

# The command line programs

There are two programs. `bananamendy` is the Python program, and it does the
convenient work: it downloads models, it reads the configuration file, and it runs
the server. `bananamendr` is the Rust program, and it takes a path.

## bananamendy

`bananamendy` uses [Fire](https://github.com/google/python-fire). Give `-- --help`
to any command to see all of its flags:

```bash
bananamendy chat -- --help
```

Two flags work for all commands:

| Flag | Default | Function |
|:-----|:--------|:---------|
| `--model` | `nano`, or the value in your configuration | Which model to use. |
| `--offline` | `False` | With `True`, a model that is not in the cache gives an error instead of a download. |

### Models

```bash
bananamendy registry            # the aliases and the repositories behind them
bananamendy pull nano           # download a model
bananamendy pull nano --revision 6def7ab  # download one exact revision
bananamendy models              # the models in the cache, smallest first
bananamendy where nano          # the directory of one model
bananamendy info                # the architecture facts of a model
```

### Text

```bash
bananamendy chat --prompt "Name one ocean."
bananamendy chat                                 # a conversation; an empty line ends it
bananamendy chat --system "You are concise." --prompt "Why is the sky blue?"
bananamendy generate --prompt "Once upon a time"
bananamendy generate --prompt "Once upon a time" --stream=False
```

`chat` applies the chat format of the checkpoint. `generate` does not: it
continues your text.

Both commands accept these flags:

| Flag | Default | Function |
|:-----|:--------|:---------|
| `--max_new_tokens` | 512 | How many tokens to write. |
| `--temperature` | 0.8 | 0 always selects the most probable token. |
| `--top_k` | 40 | Consider only the k most probable tokens. 0 means all of them. |
| `--top_p` | 0.95 | Consider the most probable tokens up to this total probability. |
| `--repetition_penalty` | 1.1 | Above 1.0 the model repeats itself less. |
| `--seed` | 0 | The start value for the random numbers. |
| `--name` | your configuration | Which model to use for this command. |

{: .note }
> The command line samples by default, because an interactive answer must not
> repeat itself. The Python interface and the WebAssembly module are greedy by
> default, because a program wants the same answer each time. Give
> `--temperature 0` for a greedy answer on the command line.

### Measurement

```bash
bananamendy bench                        # tokens each second on a fixed prompt
bananamendy bench --max_new_tokens 128
bananamendy logits --text "The capital of France is"
bananamendy logits --text "Water freezes at" --top 10
```

### The server

```bash
bananamendy serve                        # 127.0.0.1:8377
bananamendy serve --port 9000 --name mini
bananamendy serve --preload=False        # load the model at the first request
```

See [The server](../server/) for the interface.

## bananamendr

```bash
bananamendr info     -m ref/BananaMind-2-Nano-Chat
bananamendr chat     -m ref/BananaMind-2-Nano-Chat                  # a conversation
bananamendr chat     -m ref/BananaMind-2-Nano-Chat --prompt "Hi" --temperature 0
bananamendr generate -m ref/BananaMind-2-Mini-Chat --prompt "Once upon a time"
bananamendr logits   -m ref/BananaMind-2-Nano-Chat --text "The capital of France is"
bananamendr bench    -m ref/BananaMind-2-Pro-Preview-Chat
```

`-m` is necessary, because this program downloads nothing. Get the path from
`bananamendy where nano`:

```bash
bananamendr info -m "$(bananamendy where nano)"
```

The knobs are the same as above: `--max-new-tokens`, `--temperature`, `--top-k`,
`--top-p`, `--repetition-penalty`, `--seed` and `--max-seq-len`. The Rust program
uses a minus sign in a flag name, and the Python program uses an underscore.

## From Python

```python
import bananamendy

engine = bananamendy.Engine()
print(engine.info("nano")["model_type"])

generation = engine.chat("nano", [{"role": "user", "content": "Name one ocean."}],
                         max_new_tokens=32, temperature=0.0)
print(generation.text, generation.tokens_per_second)

for delta in engine.stream("nano", prompt="The capital of France is", max_new_tokens=8):
    print(delta, end="")
```

The engine keeps each loaded model, and it lets one generation run at a time. See
[The server](../server/) for the reason.

The callback of a stream does not receive the end token: the engine stops at it,
and a reader must not see the marker. The count of the pieces is therefore one
less than the count of the tokens when the answer ended by itself.

The lower level is `bananamendr`, which takes a path:

```python
import bananamendr

model = bananamendr.Model("/path/to/BananaMind-2-Nano-Chat")
print(model.generate("Hello", max_new_tokens=16, temperature=0.0).text)
model.generate("Hello", on_token=lambda text, token: print(text, end=""))
```

## From Rust

```rust
use bananamendr::{GenerateOptions, Message, Pipeline};

let pipeline = Pipeline::from_dir("ref/BananaMind-2-Nano-Chat".as_ref())?;
let generation = pipeline.chat(
    &[Message::new("user", "Why is the sky blue?")],
    &GenerateOptions::default(),
)?;
println!("{}", generation.text);
```

`GenerateOptions::default()` is greedy. `Pipeline::from_parts` takes the four
parts of a checkpoint in memory, which is what the browser build uses.
