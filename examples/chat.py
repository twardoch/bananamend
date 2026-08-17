#!/usr/bin/env python3
# this_file: examples/chat.py
"""Minimal chat turn through the Rust runtime.

    python examples/chat.py ref/BananaMind-2-Nano-Chat "Why is the sky blue?"
"""

from __future__ import annotations

import sys

import bananamendr


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    path, question = argv[0], " ".join(argv[1:])

    model = bananamendr.Model(path)
    print(f"loaded {model.config['model_type']}: {model.config['num_hidden_layers']} layers")

    result = model.chat(
        [{"role": "user", "content": question}],
        max_new_tokens=128,
        temperature=0.7,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.1,
        seed=7,
    )
    print(result.text)
    print(
        f"\n[{result.prompt_tokens} prompt tokens, {len(result.tokens)} generated, "
        f"{result.tokens_per_second:.1f} tok/s]",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
