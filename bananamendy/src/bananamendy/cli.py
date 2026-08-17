# this_file: bananamendy/src/bananamendy/cli.py
"""Fire CLI for bananamendy."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from dataclasses import asdict

from .config import Config, config_path, load_config, write_default_config
from .engine import Engine
from .models import REGISTRY, ModelError, list_local, pull, resolve
from .server import serve as _serve


class Cli:
    """Local BananaMind-2 inference: fetch checkpoints, generate, chat, serve."""

    def __init__(self, *, model: str | None = None, offline: bool = False) -> None:
        self._settings: Config = load_config().merged(model=model)
        self._offline = offline
        self._engine = Engine(download=not offline)

    # ---- configuration -------------------------------------------------

    def init_config(self, force: bool = False) -> str:
        """Write the default TOML config and print its path."""
        return str(write_default_config(force=force))

    def config(self) -> dict[str, Any]:
        """Show the effective configuration and where it came from."""
        return {"path": str(config_path()), **asdict(self._settings)}

    # ---- checkpoints ---------------------------------------------------

    def registry(self) -> dict[str, str]:
        """Known aliases and the Hugging Face repos behind them."""
        return dict(REGISTRY)

    def models(self) -> list[dict[str, Any]]:
        """Checkpoints already in the local Hugging Face cache."""
        return [
            {
                "name": c.name,
                "repo_id": c.repo_id,
                "path": str(c.path),
                "size_mb": round(c.size_bytes / 1e6, 1),
            }
            for c in list_local()
        ]

    def pull(self, name: str | None = None, revision: str | None = None) -> str:
        """Download a checkpoint into the Hugging Face cache."""
        checkpoint = pull(name or self._settings.model, revision=revision)
        return str(checkpoint.path)

    def where(self, name: str | None = None) -> str:
        """Resolve an alias, repo id, or path to a checkpoint directory."""
        return str(resolve(name or self._settings.model, download=False).path)

    def info(self, name: str | None = None) -> dict[str, Any]:
        """Architecture and tokenizer facts, straight from the checkpoint."""
        return self._engine.info(name or self._settings.model)

    # ---- inference -----------------------------------------------------

    def generate(
        self,
        prompt: str,
        name: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        seed: int | None = None,
        stream: bool = True,
    ) -> str | None:
        """Continue `prompt` without applying the chat template."""
        settings = self._settings.merged(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        model = name or settings.model
        if not stream:
            return str(self._engine.generate(model, prompt, **settings.sampling()))
        for delta in self._engine.stream(model, prompt=prompt, **settings.sampling()):
            sys.stdout.write(delta)
            sys.stdout.flush()
        sys.stdout.write("\n")
        return None

    def chat(
        self,
        prompt: str | None = None,
        name: str | None = None,
        system: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        seed: int | None = None,
    ) -> None:
        """One turn with `--prompt`, or an interactive REPL without it."""
        settings = self._settings.merged(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        model = name or settings.model
        history: list[dict[str, str]] = []
        if system:
            history.append({"role": "system", "content": system})

        def turn(text: str) -> None:
            history.append({"role": "user", "content": text})
            reply: list[str] = []
            for delta in self._engine.stream(model, messages=history, **settings.sampling()):
                reply.append(delta)
                sys.stdout.write(delta)
                sys.stdout.flush()
            sys.stdout.write("\n")
            history.append({"role": "assistant", "content": "".join(reply)})

        if prompt is not None:
            turn(prompt)
            return
        print(f"{model} — empty line or Ctrl-D to quit")
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not text:
                return
            turn(text)

    def logits(self, text: str, name: str | None = None, top: int = 5) -> list[dict[str, Any]]:
        """Top-`top` next-token candidates for `text`."""
        loaded = self._engine.load(name or self._settings.model)
        scores = loaded.model.logits(text=text)
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)[:top]
        return [
            {
                "token_id": token_id,
                "token": loaded.model.detokenize([token_id], False),
                "logit": round(float(score), 4),
            }
            for token_id, score in ranked
        ]

    def bench(self, name: str | None = None, max_new_tokens: int = 64) -> dict[str, Any]:
        """Time prefill and decode on a fixed prompt."""
        settings = self._settings.merged(max_new_tokens=max_new_tokens, temperature=0.0)
        generation = self._engine.generate(
            name or settings.model,
            "The capital of France is",
            **settings.sampling(),
        )
        return {
            "prompt_tokens": generation.prompt_tokens,
            "new_tokens": len(generation.tokens),
            "prefill_seconds": round(generation.prefill_seconds, 4),
            "decode_seconds": round(generation.decode_seconds, 4),
            "tokens_per_second": round(generation.tokens_per_second, 2),
        }

    # ---- quantization --------------------------------------------------

    def quantize(
        self,
        out: str,
        name: str | None = None,
        method: str = "mixed",
        group_size: int = 64,
        kl_budget: float = 0.02,
        embedding: bool = True,
        measure: bool = True,
    ) -> dict[str, Any]:
        """Writes a small copy of a checkpoint into the directory `out`.

        `method` is `mixed`, `int8` or `ternary`:

        * `mixed` gives ternary weights to the matrices that a measurement shows
          can carry them, and 8-bit weights to the rest. This is the default,
          and it is the only method that keeps the quality of a small model.
        * `int8` gives 8-bit weights to every matrix. It loses almost nothing.
        * `ternary` gives ternary weights to every matrix. Use it only to see
          what the model does then; the answers of a small model become useless.

        `kl_budget` controls `mixed`: it is the largest change of the answers
        that the result may show on the measurement text.
        """
        from . import plan as planner
        from .calibration import CALIBRATION_TEXTS
        from .evaluate import compare as compare_checkpoints
        from .quantize import quantize_checkpoint
        from .reference import Reference, collect_inputs

        source = Path(resolve(name or self._settings.model, download=not self._offline).path)
        destination = Path(out).expanduser()
        model = self._engine.load(str(source)).model
        bos = int(model.config["bos_token_id"])
        texts = [[bos] + list(model.tokenize(t)) for t in CALIBRATION_TEXTS]

        print(f"source: {source}")
        print(f"calibration: {len(texts)} texts, {sum(len(t) for t in texts)} tokens")
        calibration = collect_inputs(source, texts, max_rows=2048)
        names = Reference.load(source).matrix_names()

        if method == "int8":
            selection = {n: "int8" for n in names}
        elif method == "ternary":
            selection = {n: "ternary" for n in names}
        elif method == "mixed":
            tokens = planner.tokenize_evaluation(model, limit=96)
            selection, report = planner.choose(
                source,
                kl_budget=kl_budget,
                group_size=group_size,
                calibration=calibration,
                tokens=tokens,
            )
            print(
                f"plan: {report['ternary']} ternary and {report['int8']} 8-bit matrices, "
                f"divergence {report['kl_result']}"
            )
        else:
            raise ValueError("method must be mixed, int8 or ternary")

        if embedding:
            # The embedding is a large part of a small checkpoint, and 8 bits
            # cost it almost nothing.
            selection["transformer.wte.weight"] = "int8"
            if "lm_head.weight" in names:
                selection["lm_head.weight"] = "int8"

        result = quantize_checkpoint(
            source,
            destination,
            group_size=group_size,
            calibration=calibration,
            plan=selection,
        )
        summary = result.summary()
        summary["method"] = method
        print(json.dumps(summary, indent=1))

        if measure:
            quality = compare_checkpoints(source, destination, max_tokens=96).as_dict()
            (destination / "quality_report.json").write_text(
                json.dumps(quality, indent=1) + "\n", encoding="utf-8"
            )
            print(json.dumps(quality, indent=1))
            summary["quality"] = quality

        # Record the method in the report as well, for the model card.
        report_path = destination / "quantization_report.json"
        report = json.loads(report_path.read_text("utf-8"))
        report["summary"] = summary
        report_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        return summary

    def compare(
        self,
        candidate: str,
        name: str | None = None,
        max_tokens: int = 96,
    ) -> dict[str, Any]:
        """Measures a checkpoint against the float checkpoint."""
        from .evaluate import compare as compare_checkpoints

        reference = resolve(name or self._settings.model, download=not self._offline).path
        return compare_checkpoints(reference, candidate, max_tokens=max_tokens).as_dict()

    def push(
        self,
        directory: str,
        repo_id: str,
        base_model: str | None = None,
        private: bool = False,
        card_only: bool = False,
        research_only: bool = False,
    ) -> str:
        """Writes a model card and sends a quantized checkpoint to Hugging Face.

        The card holds the measured numbers from `quantize`. Set `card_only` to
        write the card and send nothing.
        """
        from . import __version__
        from .models import repo_id_for
        from .publish import build_card, read_reports, upload, write_card

        path = Path(directory).expanduser()
        quantization, quality = read_reports(path)
        base = base_model or repo_id_for(self._settings.model)
        card = build_card(
            repo_id=repo_id,
            base_model=base,
            quantization=quantization,
            quality=quality,
            producer_version=__version__,
            research_only=research_only,
        )
        write_card(path, card)
        if card_only:
            return str(path / "README.md")
        return upload(path, repo_id, private=private)

    # ---- server --------------------------------------------------------

    def serve(
        self,
        name: str | None = None,
        host: str | None = None,
        port: int | None = None,
        preload: bool = True,
    ) -> None:
        """Run the persistent OpenAI-compatible server."""
        _serve(model=name or self._settings.model, host=host, port=port, preload=preload)


def main() -> None:
    """Entry point: `bananamendy <command>`."""
    import fire

    try:
        fire.Fire(Cli, name="bananamendy")
    except ModelError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
