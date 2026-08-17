# bananamendr

CPU inference for the **BananaMind-2** chat checkpoints — Nano, Mini and Pro
(Preview) — in Rust, with no PyTorch anywhere. It reads the published
`model.safetensors` and `tokenizer.json` directly: no conversion step, no second
weight format.

Greedy decoding is token-exact against Hugging Face `transformers` for all three
checkpoints; that equivalence is a test, not a claim (see
[`tests/test_parity.py`](https://github.com/twardoch/bananamend/blob/main/tests/test_parity.py)).

Paths are always explicit — this crate never downloads anything. Use
[`bananamendy`](https://pypi.org/project/bananamendy/) if you want checkpoint
fetching, a config file and an OpenAI-compatible server.

## Library

```rust
use bananamendr::{GenerateOptions, Message, Pipeline};

let pipeline = Pipeline::from_dir("ref/BananaMind-2-Nano-Chat".as_ref())?;
let generation = pipeline.chat(
    &[Message::new("user", "Why is the sky blue?")],
    &GenerateOptions::default(),
)?;
println!("{}", generation.text);
```

## CLI

```sh
cargo install bananamendr

bananamendr info     -m ref/BananaMind-2-Nano-Chat
bananamendr chat     -m ref/BananaMind-2-Nano-Chat                 # REPL
bananamendr chat     -m ref/BananaMind-2-Nano-Chat --prompt "Hi" --temperature 0
bananamendr generate -m ref/BananaMind-2-Mini-Chat --prompt "Once upon a time"
bananamendr logits   -m ref/BananaMind-2-Nano-Chat --text "The capital of France is"
bananamendr bench    -m ref/BananaMind-2-Pro-Preview-Chat
```

`chat` applies the checkpoint's chat template; `generate` does not.
`--temperature 0` is greedy. Other knobs: `--top-k`, `--top-p`,
`--repetition-penalty`, `--seed`, `--max-new-tokens`, `--max-seq-len`.

The CLI samples by default (`--temperature 0.8 --top-k 40 --top-p 0.95
--repetition-penalty 1.1`) because interactive output should not loop.

## License

Apache-2.0
