# bananamendr (Python)

Local CPU inference for the **BananaMind-2** chat checkpoints — Nano, Mini and
Pro (Preview) — with no PyTorch at runtime. The Rust core reads the published
`model.safetensors` and `tokenizer.json` directly. One abi3 wheel serves every
CPython ≥ 3.9.

Greedy decoding is token-exact against Hugging Face `transformers`.

This module takes **paths only** — it never downloads anything. For checkpoint
fetching, a config file and an OpenAI-compatible server, install
[`bananamendy`](https://pypi.org/project/bananamendy/).

```bash
pip install bananamendr
```

```python
import bananamendr

model = bananamendr.Model("ref/BananaMind-2-Nano-Chat")
print(model.config["model_type"], model.config["num_hidden_layers"])

generation = model.chat(
    [{"role": "user", "content": "Why is the sky blue?"}],
    max_new_tokens=64,
    temperature=0.0,               # greedy; the Python API defaults to greedy
)
print(generation.text, generation.tokens_per_second)

# Streaming: the callback receives (decoded delta, token id) per step.
model.generate("Once upon a time", on_token=lambda text, token: print(text, end=""))
```

Also available: `tokenize`, `detokenize`, `apply_chat_template`, `logits`,
`generate_tokens`.

## License

Apache-2.0
