# this_file: bananamendy/src/bananamendy/publish.py
"""Upload of a checkpoint to the Hugging Face hub, with a model card.

The card carries the measured numbers. A quantized checkpoint without numbers is
not useful to anybody: a reader cannot know if the file is a small copy of the
model or a small copy of the noise.
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi

CARD_NAME = "README.md"

# The files that a checkpoint needs, and the extra files that we also send.
UPLOAD_PATTERNS = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
    "quantization_report.json",
    "quality_report.json",
    CARD_NAME,
]


def build_card(
    *,
    repo_id: str,
    base_model: str,
    quantization: dict,
    quality: dict | None,
    licence: str = "apache-2.0",
    producer_version: str = "",
    research_only: bool = False,
    sample: str = "",
) -> str:
    """Writes the model card. The numbers come from the two reports.

    `research_only` puts a warning at the top. Use it for a checkpoint that shows
    what a method does, and not for a checkpoint that a person should use.
    """
    summary = quantization.get("summary", {})
    ternary = summary.get("tensors_ternary")
    int8 = summary.get("tensors_int8")
    lines: list[str] = []
    lines.append("---")
    lines.append(f"license: {licence}")
    lines.append(f"base_model: {base_model}")
    lines.append("library_name: bananamendr")
    lines.append("pipeline_tag: text-generation")
    lines.append("tags:")
    lines.append("- quantized")
    # The tags must describe this file, and not the tool that made it.
    if ternary:
        lines.append("- ternary")
    if int8 or not ternary:
        lines.append("- int8")
    lines.append("- bananamend")
    if research_only:
        lines.append("- research-only")
    lines.append("---")
    lines.append("")
    lines.append(f"# {repo_id.split('/')[-1]}")
    lines.append("")
    if research_only:
        lines.append("> **This checkpoint does not work as a chat model.**")
        lines.append(">")
        lines.append(
            "> Every matrix holds ternary weights, and a model of this size cannot "
            "carry that. The answers have no relation to the question. The numbers "
            "below say how bad it is."
        )
        lines.append(">")
        lines.append(
            "> The file is here so that the measurement can be repeated, and so that "
            "the size of a fully ternary checkpoint is visible. For work, use the "
            "`-int8` or the `-mixed` checkpoint of the same model."
        )
        lines.append("")
    if sample:
        lines.append("What it writes for `Name one ocean.`, with a temperature of zero:")
        lines.append("")
        lines.append("```")
        lines.append(sample.strip())
        lines.append("```")
        lines.append("")
    lines.append(
        f"A quantized copy of [{base_model}](https://huggingface.co/{base_model}), "
        "for the [bananamend](https://github.com/twardoch/bananamend) engine."
    )
    lines.append("")
    lines.append("## How to use it")
    lines.append("")
    lines.append("```bash")
    lines.append("uv pip install bananamendy")
    lines.append(f"bananamendy chat --name {repo_id} --prompt \"Name one ocean.\"")
    lines.append("```")
    lines.append("")
    lines.append("```python")
    lines.append("import bananamendr")
    lines.append("from huggingface_hub import snapshot_download")
    lines.append("")
    lines.append(f'model = bananamendr.Model(snapshot_download("{repo_id}"))')
    lines.append('print(model.chat([{"role": "user", "content": "Hi"}]).text)')
    lines.append("```")
    lines.append("")
    lines.append(
        "The file needs the bananamend engine. `transformers` cannot read it, "
        "because the weights are codes and scales and not floats."
    )
    lines.append("")
    lines.append("## What is inside")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|:-----|:------|")
    lines.append(f"| Method | `{summary.get('method', 'mixed')}` |")
    lines.append(f"| Group size | {summary.get('group_size', '')} |")
    if ternary is not None:
        lines.append(f"| Ternary matrices | {ternary} |")
    if int8 is not None:
        lines.append(f"| 8-bit matrices | {int8} |")
    lines.append(f"| Float size | {summary.get('float_mb', '')} MB |")
    lines.append(f"| This file | {summary.get('result_mb', '')} MB |")
    lines.append(f"| Smaller by | {summary.get('ratio', '')} times |")
    if producer_version:
        lines.append(f"| Made by | `bananamendy {producer_version}` |")
    lines.append("")

    if quality:
        lines.append("## Measured quality")
        lines.append("")
        lines.append(
            "The numbers compare this checkpoint with the float checkpoint on a text "
            "that the quantizer never saw. The engine produced both sides."
        )
        lines.append("")
        lines.append("| Measure | Value |")
        lines.append("|:--------|:------|")
        lines.append(f"| Same next token | {quality.get('top1_agreement', 0) * 100:.1f}% |")
        lines.append(
            f"| Next token inside the first five | {quality.get('top5_agreement', 0) * 100:.1f}% |"
        )
        lines.append(f"| Divergence (KL) | {quality.get('mean_kl_divergence', 0):.4f} |")
        lines.append(
            f"| Perplexity | {quality.get('candidate_perplexity', 0):.1f} "
            f"against {quality.get('reference_perplexity', 0):.1f} |"
        )
        lines.append(
            f"| Identical greedy answers | {quality.get('greedy_identical_answers', '')} |"
        )
        lines.append("")

    lines.append("## Why the mixture")
    lines.append("")
    lines.append(
        "Ternary weights hold three values: minus one, zero and plus one. They are "
        "very small, and they lose much. Eight-bit weights are four times larger, and "
        "they lose almost nothing."
    )
    lines.append("")
    lines.append(
        "A model of this size cannot carry ternary weights everywhere. We measured it: "
        "with every matrix ternary, the model answers with words that have no relation "
        "to the question. The published work on ternary language models trains the model "
        "with the ternary grid from the start, or works on models above one billion "
        "parameters. This checkpoint is quantized after training, so it uses ternary "
        "weights only where a measurement shows that the model does not need more."
    )
    lines.append("")
    lines.append("## How it was made")
    lines.append("")
    lines.append("```bash")
    lines.append("bananamendy quantize --name <base model> --out <directory>")
    lines.append("```")
    lines.append("")
    lines.append("The steps are:")
    lines.append("")
    lines.append(
        "1. Run a calibration text through the model, and record what each matrix "
        "receives."
    )
    lines.append(
        "2. For each group of 64 weights, search the threshold that gives the smallest "
        "error, and give the positive and the negative weights separate scales "
        "(Ternary Weight Networks, with the asymmetric grid of PT2-LLM)."
    )
    lines.append(
        "3. Quantize one column at a time, and move the error of that column into the "
        "columns that follow (GPTQ)."
    )
    lines.append(
        "4. Measure each matrix on its own, and give ternary weights to the matrices "
        "that change the answers least, while the total change stays inside a budget."
    )
    lines.append("5. Give every other matrix 8-bit weights.")
    lines.append("")
    lines.append("`quantization_report.json` in this repository holds the result per tensor.")
    lines.append("")
    lines.append("## Licence")
    lines.append("")
    lines.append(f"{licence}, the same as the base model.")
    lines.append("")
    return "\n".join(lines)


def write_card(directory: Path | str, text: str) -> Path:
    directory = Path(directory)
    target = directory / CARD_NAME
    target.write_text(text, encoding="utf-8")
    return target


def upload(
    directory: Path | str,
    repo_id: str,
    *,
    private: bool = False,
    message: str = "Add a quantized checkpoint",
) -> str:
    """Sends a directory to the hub, and returns the address of the repository."""
    directory = Path(directory)
    if not (directory / "model.safetensors").is_file():
        raise FileNotFoundError(f"{directory} holds no model.safetensors")
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(directory),
        repo_id=repo_id,
        repo_type="model",
        commit_message=message,
        allow_patterns=UPLOAD_PATTERNS,
    )
    return f"https://huggingface.co/{repo_id}"


def read_reports(directory: Path | str) -> tuple[dict, dict | None]:
    """Reads the two reports that `quantize` and `compare` wrote."""
    directory = Path(directory)
    quantization = json.loads((directory / "quantization_report.json").read_text("utf-8"))
    quality_path = directory / "quality_report.json"
    quality = json.loads(quality_path.read_text("utf-8")) if quality_path.is_file() else None
    return quantization, quality
