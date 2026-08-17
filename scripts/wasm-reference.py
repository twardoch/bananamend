#!/usr/bin/env python3
# this_file: scripts/wasm-reference.py
"""Writes the reference output that the WebAssembly parity test compares against.

The values come from the Python extension module, which uses the same Rust code
as the command line program. The parity test then gives the same checkpoint to the
WebAssembly module. Greedy decoding must give the same tokens on both sides.

    wasm-reference.py OUTPUT.json CHECKPOINT_DIR VERSION
"""

from __future__ import annotations

import json
import sys

import bananamendr

# Greedy decoding only. A temperature above zero uses the random number
# generator, and the two builds then have no reason to agree.
GREEDY = {
    "max_new_tokens": 24,
    "temperature": 0.0,
    "top_k": 0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "seed": 0,
    "stop_on_eos": True,
}

PROMPTS = [
    "The capital of France is",
    "Water freezes at",
    "One, two, three,",
]

CHATS = [
    [{"role": "user", "content": "Name one ocean."}],
    [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Why is the sky blue?"},
    ],
]

TOKENIZE = [
    "Hello, world!",
    "  leading and trailing space  ",
    "Zażółć gęślą jaźń",
    "الأنهار والآلات",
    "emoji: 🍌 and numbers: 1234",
]


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    destination, checkpoint, version = argv

    model = bananamendr.Model(checkpoint)
    config = dict(model.config)

    payload = {
        "version": version,
        "checkpoint": checkpoint,
        "options": GREEDY,
        "info": {
            "model_type": config["model_type"],
            "hidden_size": config["hidden_size"],
            "num_hidden_layers": config["num_hidden_layers"],
            "num_attention_heads": config["num_attention_heads"],
            "num_key_value_heads": config["num_key_value_heads"],
            "head_dim": config["head_dim"],
            "vocab_size": config["vocab_size"],
            "max_position_embeddings": config["max_position_embeddings"],
            "bos_token_id": config["bos_token_id"],
            "eos_token_id": config["eos_token_id"],
        },
        "tokenize": [
            {
                "text": text,
                "tokens": model.tokenize(text),
                "roundtrip": model.detokenize(model.tokenize(text), True),
            }
            for text in TOKENIZE
        ],
        "chat_template": [
            {
                "messages": messages,
                "rendered": model.apply_chat_template(messages),
                "tokens": model.tokenize(model.apply_chat_template(messages)),
            }
            for messages in CHATS
        ],
        "generate": [],
        "chat": [],
        "logits": [],
    }

    for prompt in PROMPTS:
        generation = model.generate(prompt, **GREEDY)
        payload["generate"].append(
            {
                "prompt": prompt,
                "text": generation.text,
                "tokens": list(generation.tokens),
                "prompt_tokens": generation.prompt_tokens,
                "finished_by_eos": generation.finished_by_eos,
            }
        )
        scores = model.logits(text=prompt)
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)[:5]
        payload["logits"].append(
            {
                "text": prompt,
                "length": len(scores),
                "top": [[int(i), round(float(s), 3)] for i, s in ranked],
            }
        )

    for messages in CHATS:
        generation = model.chat(messages, **GREEDY)
        payload["chat"].append(
            {
                "messages": messages,
                "text": generation.text,
                "tokens": list(generation.tokens),
                "prompt_tokens": generation.prompt_tokens,
                "finished_by_eos": generation.finished_by_eos,
            }
        )

    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
    print(
        f"reference written: {destination} "
        f"({len(PROMPTS)} prompts, {len(CHATS)} chats, {len(TOKENIZE)} texts)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
