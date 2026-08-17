# this_file: bananamendy/src/bananamendy/evaluate.py
"""Measurement of the difference between two checkpoints.

The numbers here decide if a quantized checkpoint is good enough to publish.
Three of them matter:

* **Agreement.** How often do the two models select the same next token? A
  greedy answer only stays the same while the agreement holds.
* **Perplexity.** How surprised is each model by a text? A large increase means
  a large loss of quality.
* **Same first tokens.** How many tokens of a greedy answer are identical? This
  is what a reader sees.

All of the work uses the engine, so the numbers describe the real behaviour and
not a model of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import bananamendr

from .calibration import EVALUATION_CHATS, EVALUATION_PROMPTS, EVALUATION_TEXT

SAMPLE_TEXT = EVALUATION_TEXT
PROMPTS = EVALUATION_PROMPTS
CHATS = EVALUATION_CHATS


@dataclass
class Comparison:
    """The result of a comparison of two checkpoints."""

    tokens_compared: int
    top1_agreement: float
    top5_agreement: float
    mean_kl: float
    reference_perplexity: float
    candidate_perplexity: float
    greedy_prefix_tokens: list[int]
    greedy_identical: int
    greedy_total: int

    def as_dict(self) -> dict:
        return {
            "tokens_compared": self.tokens_compared,
            "top1_agreement": round(self.top1_agreement, 4),
            "top5_agreement": round(self.top5_agreement, 4),
            "mean_kl_divergence": round(self.mean_kl, 4),
            "reference_perplexity": round(self.reference_perplexity, 3),
            "candidate_perplexity": round(self.candidate_perplexity, 3),
            "perplexity_ratio": round(
                self.candidate_perplexity / self.reference_perplexity, 3
            )
            if self.reference_perplexity
            else 0.0,
            "greedy_identical_prefix": self.greedy_prefix_tokens,
            "greedy_identical_answers": f"{self.greedy_identical}/{self.greedy_total}",
        }


def _softmax(scores: list[float]) -> list[float]:
    top = max(scores)
    exponents = [math.exp(s - top) for s in scores]
    total = sum(exponents)
    return [e / total for e in exponents]


def _log_softmax_at(scores: list[float], index: int) -> float:
    top = max(scores)
    total = sum(math.exp(s - top) for s in scores)
    return (scores[index] - top) - math.log(total)


def _top_k(scores: list[float], k: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]


def compare(
    reference_path: Path | str,
    candidate_path: Path | str,
    *,
    max_tokens: int = 256,
    greedy_tokens: int = 24,
) -> Comparison:
    """Compares two checkpoints on the text and the prompts in this file."""
    reference = bananamendr.Model(str(reference_path))
    candidate = bananamendr.Model(str(candidate_path))

    tokens = reference.tokenize(SAMPLE_TEXT)[:max_tokens]
    if candidate.tokenize(SAMPLE_TEXT)[: len(tokens)] != tokens:
        raise ValueError("the two checkpoints do not have the same tokenizer")

    same_top1 = 0
    same_top5 = 0
    kl_total = 0.0
    reference_logprob = 0.0
    candidate_logprob = 0.0
    compared = 0

    # One step for each position: the models see the same prefix, and the next
    # token of the text is the answer that both must score.
    for position in range(1, len(tokens)):
        prefix = tokens[:position]
        target = tokens[position]
        reference_scores = reference.logits(tokens=prefix)
        candidate_scores = candidate.logits(tokens=prefix)

        reference_top = _top_k(reference_scores, 5)
        candidate_top = _top_k(candidate_scores, 5)
        same_top1 += int(reference_top[0] == candidate_top[0])
        same_top5 += int(candidate_top[0] in reference_top)

        reference_probabilities = _softmax(reference_scores)
        candidate_probabilities = _softmax(candidate_scores)
        kl_total += sum(
            p * math.log(p / q)
            for p, q in zip(reference_probabilities, candidate_probabilities)
            if p > 1e-9 and q > 1e-12
        )

        reference_logprob += _log_softmax_at(reference_scores, target)
        candidate_logprob += _log_softmax_at(candidate_scores, target)
        compared += 1

    greedy_prefix: list[int] = []
    identical = 0
    for prompt in PROMPTS:
        options = {"max_new_tokens": greedy_tokens, "temperature": 0.0}
        first = list(reference.generate(prompt, **options).tokens)
        second = list(candidate.generate(prompt, **options).tokens)
        shared = 0
        for a, b in zip(first, second):
            if a != b:
                break
            shared += 1
        greedy_prefix.append(shared)
        identical += int(first == second)
    for messages in CHATS:
        options = {"max_new_tokens": greedy_tokens, "temperature": 0.0}
        first = reference.chat(list(messages), **options).text
        second = candidate.chat(list(messages), **options).text
        identical += int(first == second)

    total_answers = len(PROMPTS) + len(CHATS)
    return Comparison(
        tokens_compared=compared,
        top1_agreement=same_top1 / compared if compared else 0.0,
        top5_agreement=same_top5 / compared if compared else 0.0,
        mean_kl=kl_total / compared if compared else 0.0,
        reference_perplexity=math.exp(-reference_logprob / compared) if compared else 0.0,
        candidate_perplexity=math.exp(-candidate_logprob / compared) if compared else 0.0,
        greedy_prefix_tokens=greedy_prefix,
        greedy_identical=identical,
        greedy_total=total_answers,
    )


def perplexity(path: Path | str, *, max_tokens: int = 256) -> float:
    """The perplexity of one checkpoint on the text in this file."""
    model = bananamendr.Model(str(path))
    tokens = model.tokenize(SAMPLE_TEXT)[:max_tokens]
    total = 0.0
    for position in range(1, len(tokens)):
        scores = model.logits(tokens=tokens[:position])
        total += _log_softmax_at(scores, tokens[position])
    steps = len(tokens) - 1
    return math.exp(-total / steps) if steps else 0.0
