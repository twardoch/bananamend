#!/usr/bin/env python3
# this_file: examples/stream_and_inspect.py
"""Streaming generation, multi-turn state, and raw logit inspection.

    python examples/stream_and_inspect.py ref/BananaMind-2-Nano-Chat
"""

from __future__ import annotations

import sys

import bananamendr


def stream(model: bananamendr.Model, messages: list[dict]) -> str:
    """Prints tokens as they arrive and returns the full reply."""
    result = model.chat(
        messages,
        max_new_tokens=96,
        temperature=0.7,
        top_k=40,
        seed=11,
        on_token=lambda text, token_id: (sys.stdout.write(text), sys.stdout.flush()),
    )
    print()
    return result.text


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    model = bananamendr.Model(argv[0])

    # 1. Multi-turn: keep appending to the message list.
    messages: list[dict] = [{"role": "system", "content": "You are concise."}]
    for question in ["Name one ocean.", "Now name a river."]:
        messages.append({"role": "user", "content": question})
        print(f"\n--- user: {question}")
        reply = stream(model, messages)
        messages.append({"role": "assistant", "content": reply})

    # 2. What the model actually sees.
    print("\n--- rendered prompt")
    print(repr(model.apply_chat_template(messages, add_generation_prompt=True)))

    # 3. Next-token distribution for an arbitrary prefix.
    print("\n--- top next tokens after '<|bos|>The capital of France is'")
    ids = model.tokenize("<|bos|>The capital of France is")
    logits = model.logits(tokens=ids)
    top = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)[:5]
    for token_id in top:
        print(f"  {token_id:>6} {logits[token_id]:>10.4f}  {model.detokenize([token_id], False)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
