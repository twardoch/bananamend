---
this_file: docs/quantization.md
title: Small checkpoints
layout: default
nav_order: 6
permalink: /quantization/
---

# Small checkpoints

A checkpoint of 32-bit floats needs 4 bytes for each weight. `bananamendy
quantize` writes a copy that needs 1 byte, or a quarter of a byte, for most of
the weights. The engine reads that copy directly.

| Grid | Bytes for each weight | What it costs |
|:-----|:----------------------|:--------------|
| 32-bit floats | 4 | nothing; this is the original |
| 8 bits | 1 + 4 for each group of 64 = **1.0625** | almost nothing |
| ternary | 0.25 + 8 for each group of 64 = **0.375** | much, and see the measurements below |

Ternary means three values: minus one, zero and plus one. A scale for each group
of weights gives the size back.

## Use it

```bash
bananamendy quantize /tmp/nano-int8  --name nano --method int8
bananamendy quantize /tmp/nano-mixed --name nano --method mixed --kl_budget 0.01
bananamendy chat --name /tmp/nano-int8 --prompt "Name one ocean."
bananamendy compare /tmp/nano-int8 --name nano
```

The command writes the checkpoint, `quantization_report.json` with the result for
each tensor, and `quality_report.json` with the measurement.

Ready-made checkpoints are on Hugging Face under
[fontlab](https://huggingface.co/fontlab), for all three models:

| Model | 8 bits | mixed | ternary (does not work) |
|:------|:-------|:------|:------------------------|
| Nano | `fontlab/BananaMind-2-Nano-Chat-int8` | `-mixed` | `-ternary` |
| Mini | `fontlab/BananaMind-2-Mini-Chat-int8` | — | — |
| Pro | `fontlab/BananaMind-2-Pro-Preview-Chat-int8` | `-mixed` | `-ternary` |

```bash
bananamendy chat --name fontlab/BananaMind-2-Pro-Preview-Chat-int8 --prompt "Hi"
```

The [demonstration page](../demo/) loads any of them in a browser, and it also
takes the name of a repository of your own.

## The three methods

`--method int8` gives 8-bit weights to every matrix. **Use this one.** It is
close to the float model, and the file is approximately 3.8 times smaller.

`--method mixed` gives ternary weights to the matrices that a measurement shows
can carry them, and 8-bit weights to the rest. The file is a little smaller than
`int8`, and the answers change more. `--kl_budget` controls the trade.

`--method ternary` gives ternary weights to every matrix. The file is 7.7 to 8.3
times smaller, and the answers of these small models are not useful. The method is
here for measurement, and the results are below.

## Measured results

The measurement compares the quantized checkpoint with the float checkpoint on a
text that the quantizer never sees. Both sides run in the engine.

| Checkpoint | Method | File | Weights in memory | Speed | Same next token | Perplexity | Identical greedy answers |
|:-----------|:-------|:-----|:------------------|:------|:----------------|:-----------|:-------------------------|
| Nano (10 M) | floats | 39.9 MB | 39.9 MB | 611 tokens/s | this is the reference | 66.3 | this is the reference |
| Nano | 8 bits | 10.6 MB (3.8×) | 10.6 MB | 302 tokens/s | 97.9% | 67.2 against 66.3 (1.01×) | 4 of 8 |
| Nano | mixed (3 ternary) | 10.5 MB (3.8×) | 10.5 MB | 261 tokens/s | 96.8% | 67.5 against 66.3 (1.02×) | 3 of 8 |
| Nano | ternary (70) | 5.2 MB (7.7×) | 5.2 MB | 34 tokens/s | 22.1% | 709.9 against 66.3 (10.7×) | 0 of 8 |
| Mini (25 M) | floats | 100.7 MB | 100.7 MB | 234 tokens/s | this is the reference | 49.1 | this is the reference |
| Mini | 8 bits | 26.8 MB (3.8×) | 26.8 MB | 153 tokens/s | 96.8% | 49.5 against 49.1 (1.01×) | 7 of 8 |
| Pro (139 M) | floats | 555.9 MB | 555.9 MB | 71 tokens/s | this is the reference | 33.3 | this is the reference |
| Pro | 8 bits | 147.8 MB (3.8×) | 147.8 MB | 54 tokens/s | 100.0% | 33.3 against 33.3 (1.00×) | 6 of 8 |
| Pro | mixed (12 ternary) | 144.9 MB (3.8×) | 144.9 MB | 46 tokens/s | 90.5% | 33.3 against 33.3 (1.00×) | 3 of 8 |
| Pro | ternary (168) | 66.7 MB (8.3×) | 66.7 MB | 11 tokens/s | 41.3% | 131.1 against 38.3 (3.4×) | 0 of 8 |

Each perplexity is against the reference of the same measurement, and the number
in brackets is the ratio. The Pro ternary row uses 64 tokens of the measurement
text, and every other row uses 96; a longer text gives a different reference,
which is why each row carries its own.

Speed is one Apple M4 Max, greedy, 32 new tokens, with all of the cores.

`bananamendy info --name <checkpoint>` reports the form and the memory, so those
two columns are checkable:

```
$ bananamendy info --name fontlab/BananaMind-2-Nano-Chat-int8
storage:    {"embedding": "int8", "matrices_int8": "70"}
weight_mb:  10.61
```

The numbers say three things clearly.

**Eight bits are nearly free in quality.** Pro in 8 bits selects the same next
token every time in the measurement, and its perplexity moves by 0.1%.

**Eight bits are not free in speed.** A code becomes a value inside the
multiplication, and that work costs time: Pro loses a quarter of its speed, and
Nano loses half. A ternary checkpoint loses much more, because the two bits of
each weight must be unpacked one at a time.

**Ternary weights everywhere do not work at this size.** Nano writes
`arararararong the the the hear hear hear`, and Pro writes
`Engineererererererererer`. Both still produce English pieces, and neither answers
the question. This is not a fault of the quantizer. The published work on ternary language models either trains the model
with the ternary grid from the beginning (BitNet b1.58), or works on models above
one billion parameters (PT-BitNet, PT²-LLM). Nano holds 10 million weights, and
Pro holds 139 million.

## How the ternary grid is chosen

For each group of 64 weights in one row:

1. Try 22 thresholds, from 0.05 to 1.10 times the mean absolute value of the
   group.
2. For each threshold, give the weights above it the code for plus one, and the
   weights below minus it the code for minus one. Everything else takes the code
   for zero.
3. Give the two signs separate scales: the mean of the positive weights, and the
   mean of the negative weights.
4. Keep the threshold with the smallest square error.

Steps 1 and 4 replace the classic rule of Ternary Weight Networks, which uses one
fixed threshold of 0.7 times the mean absolute value. That rule comes from an
assumption about the distribution of the weights, and a real group often
disagrees. Step 3 is the asymmetric grid of PT²-LLM. Neither addition can make
the error larger.

## How the calibration works

A weight only matters as much as the activation that multiplies it. The quantizer
therefore does not minimise the error of the weights; it minimises the error of
the output.

1. Twenty calibration texts, which are inside the package, run through the model.
   The quantizer records what each matrix receives. The calibration text and the
   measurement text are different, so a good number cannot come from the
   quantizer learning the measurement.
2. The quantizer builds the second-moment matrix of those inputs.
3. It takes the columns in order. After each column it moves the error of that
   column into the columns that are still to come (GPTQ).

The calibration pass is a forward pass in numpy, in
`bananamendy/src/bananamendy/reference.py`. It agrees with the engine to five
decimal places, and a test holds that agreement.

## How the mixture is chosen

`--method mixed` measures instead of guessing:

1. Give every matrix 8-bit weights. This is the base.
2. For each matrix on its own, replace it with its ternary form, and measure the
   divergence of the answers from the float model.
3. Sort the matrices from the smallest change to the largest.
4. Take them in that order. Keep each ternary form while the total divergence
   stays inside `--kl_budget`. Give the rest 8-bit weights.

Step 4 measures after each acceptance, because two ternary matrices together are
worse than the sum of the two alone.

The attention projections for the query and the key accept ternary weights most
easily. The three matrices of the MLP, and everything in the last block, accept
them least.

## What stays in floats

Every normalisation weight stays in 32-bit floats. They are vectors, they are
small, and they control the size of everything after them.

The token embedding becomes 8 bits, because it is a large part of a small
checkpoint: 8.4 MB of Nano's 39.9 MB. In these checkpoints the output matrix is
the same table, so one decision covers both.

## The format

`config.json` holds a `quantization` block:

```json
{
  "quantization": {
    "method": "mixed",
    "group_size": 64,
    "packing": "2bit-4per-byte-low-first",
    "codes": {"0": -1, "1": 0, "2": 1},
    "tensors": {
      "transformer.h.0.attn.q_proj.weight": {
        "method": "ternary-gptq-v1",
        "group_size": 64,
        "shape": [640, 640],
        "groups": 10
      }
    }
  }
}
```

`model.safetensors` then holds, for a ternary tensor, three arrays:
`<name>.ternary.codes` as bytes, and `<name>.ternary.scale_pos` and
`<name>.ternary.scale_neg` as floats of shape `[rows, groups]`. An 8-bit tensor
holds `<name>.int8.codes` and `<name>.int8.scale`.

The engine keeps the codes in memory and rebuilds each value inside the
multiplication, so the memory in use is as small as the file. A ternary
multiplication needs no multiplication for each weight: it adds the inputs of the
positive codes, subtracts the inputs of the negative codes, and multiplies twice
at the end of the group.

## Publish a checkpoint

```bash
bananamendy push /tmp/nano-int8 fontlab/BananaMind-2-Nano-Chat-int8
bananamendy push /tmp/nano-int8 fontlab/BananaMind-2-Nano-Chat-int8 --card_only
```

The command writes a model card with the measured numbers and sends the
directory. `--card_only` writes the card and sends nothing. The token comes from
`HF_TOKEN`, as `huggingface_hub` expects.
