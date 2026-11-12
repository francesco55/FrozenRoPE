"""Short-context regression: HellaSwag, ARC-Easy, PIQA via lm-evaluation-harness.

Guards the "lose at most one point" success criterion for RQ1.

    uv sync --extra eval
    uv run python -m posext.eval.shortctx --run outputs/rq1_learned_freq_head
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from posext.export import export_full_model
from posext.models.build import DEFAULT_MODEL

logger = logging.getLogger("posext.eval.shortctx")

TASKS = ("hellaswag", "arc_easy", "piqa")


def evaluate_shortctx(
    model_path: str | Path,
    *,
    tasks: tuple[str, ...] = TASKS,
    batch_size: int = 8,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run lm-eval on an exported (or HF hub) model path with trust_remote_code."""
    try:
        from lm_eval import evaluator
    except ImportError as exc:
        raise ImportError(
            "lm-eval is required for short-context eval. Install with: uv sync --extra eval"
        ) from exc

    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},trust_remote_code=True,dtype=bfloat16",
        tasks=list(tasks),
        batch_size=batch_size,
        limit=limit,
    )
    # Keep the compact score table.
    scores = {}
    for task, payload in results.get("results", {}).items():
        # Prefer acc_norm when present (hellaswag), else acc.
        acc = payload.get("acc_norm,none", payload.get("acc,none", payload.get("acc")))
        scores[task] = acc
    return {"tasks": scores, "raw_keys": list(results.get("results", {}).keys())}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=None, help="Trained positional run to export then eval")
    p.add_argument("--model", type=str, default=None, help="HF path or hub id (baselines)")
    p.add_argument("--export-dir", type=Path, default=None, help="Where to write the full model export")
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="lm-eval limit (smoke)")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    if args.run:
        export_dir = args.export_dir or (args.run / "full_model")
        model_path = export_full_model(args.run, export_dir)
    else:
        model_path = Path(args.model or DEFAULT_MODEL)
    tasks = tuple(t for t in args.tasks.split(",") if t)
    results = evaluate_shortctx(model_path, tasks=tasks, batch_size=args.batch_size, limit=args.limit)
    results["model"] = str(model_path)
    text = json.dumps(results, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
