"""Grid search over YaRN settings on PG-19 validation + validation passkey.

Score = z-scored combination of (-val NLL @ 16k) and passkey accuracy. The winning
settings are written to `outputs/yarn_grid/best.json` and can be copied into
`configs/rq1/yarn_tuned.yaml`.

    uv run python -m posext.baselines.yarn_grid --device cuda
    # smoke (tiny grid):
    uv run python -m posext.baselines.yarn_grid --smoke --max-sequences 2 --n-per-cell 1
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from posext.eval.passkey import evaluate_passkey
from posext.eval.ppl import evaluate_ppl
from posext.models import load_model
from posext.models.build import DEFAULT_MODEL

logger = logging.getLogger("posext.baselines.yarn_grid")

FACTORS = (4, 6, 8)
BETA_FAST = (8, 16, 32, 64)
BETA_SLOW = (0.5, 1.0, 2.0)
ATTENTION_FACTORS = (1.0, 1.1, 1.2, 1.3)


def yarn_pos(factor: float, beta_fast: float, beta_slow: float, attention_factor: float) -> dict[str, Any]:
    """pos dict for one YaRN grid point."""
    return {
        "mode": "yarn",
        "yarn": {
            "factor": float(factor),
            "beta_fast": float(beta_fast),
            "beta_slow": float(beta_slow),
            "attention_factor": float(attention_factor),
        },
    }


def combined_score(nll: float, passkey_acc: float, nll_mean: float, nll_std: float, acc_mean: float, acc_std: float) -> float:
    """Higher is better: low NLL, high passkey. Uses population z-scores over the grid."""
    z_nll = 0.0 if nll_std < 1e-8 else (nll_mean - nll) / nll_std  # lower nll -> higher score
    z_acc = 0.0 if acc_std < 1e-8 else (passkey_acc - acc_mean) / acc_std
    return 0.5 * z_nll + 0.5 * z_acc


def run_grid(
    *,
    factors: tuple[float, ...] = FACTORS,
    beta_fast: tuple[float, ...] = BETA_FAST,
    beta_slow: tuple[float, ...] = BETA_SLOW,
    attention_factors: tuple[float, ...] = ATTENTION_FACTORS,
    model_name: str = DEFAULT_MODEL,
    device: torch.device,
    max_sequences: int = 20,
    n_per_cell: int = 2,
    passkey_lengths: tuple[int, ...] = (4096, 16384),
    ppl_lengths: tuple[int, ...] = (16384,),
    seed: int = 0,
    attn: str = "sdpa",
) -> dict[str, Any]:
    """Evaluate every YaRN setting; return ranked results + best."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    rows: list[dict[str, Any]] = []
    grid = list(itertools.product(factors, beta_fast, beta_slow, attention_factors))
    logger.info("YaRN grid: %d settings", len(grid))

    for i, (factor, bf, bs, af) in enumerate(grid, 1):
        pos = yarn_pos(factor, bf, bs, af)
        logger.info("[%d/%d] %s", i, len(grid), pos["yarn"])
        model = load_model(model_name, pos=pos, device=device, attn_implementation=attn, dtype=torch.bfloat16)
        model.eval()
        ppl = evaluate_ppl(
            model,
            split="validation",
            lengths=ppl_lengths,
            max_sequences=max_sequences,
            seed=seed,
        )
        nll = ppl["lengths"][str(ppl_lengths[-1])]["nll"]
        pk = evaluate_passkey(
            model,
            tokenizer,
            lengths=passkey_lengths,
            depths=(0.0, 0.5, 1.0),
            n_per_cell=n_per_cell,
            seed=seed,  # validation seed
        )
        rows.append(
            {
                "yarn": pos["yarn"],
                "nll": nll,
                "ppl": ppl["lengths"][str(ppl_lengths[-1])]["ppl"],
                "passkey_accuracy": pk["accuracy"],
                "passkey_by_length": pk["by_length"],
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    nlls = [r["nll"] for r in rows]
    accs = [r["passkey_accuracy"] for r in rows]
    nll_mean = sum(nlls) / len(nlls)
    acc_mean = sum(accs) / len(accs)
    nll_std = (sum((x - nll_mean) ** 2 for x in nlls) / len(nlls)) ** 0.5
    acc_std = (sum((x - acc_mean) ** 2 for x in accs) / len(accs)) ** 0.5
    for row in rows:
        row["score"] = combined_score(row["nll"], row["passkey_accuracy"], nll_mean, nll_std, acc_mean, acc_std)
    rows.sort(key=lambda r: r["score"], reverse=True)
    best = rows[0]
    return {
        "best": best,
        "n_settings": len(rows),
        "results": rows,
        "zscore_stats": {"nll_mean": nll_mean, "nll_std": nll_std, "acc_mean": acc_mean, "acc_std": acc_std},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default=None)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-sequences", type=int, default=20)
    p.add_argument("--n-per-cell", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/yarn_grid"))
    p.add_argument("--smoke", action="store_true", help="1x1x1x1 grid for plumbing checks")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    kwargs: dict[str, Any] = {}
    if args.smoke:
        kwargs = {
            "factors": (4,),
            "beta_fast": (32,),
            "beta_slow": (1.0,),
            "attention_factors": (1.0,),
        }
    results = run_grid(
        model_name=args.model,
        device=device,
        max_sequences=args.max_sequences,
        n_per_cell=args.n_per_cell,
        seed=args.seed,
        attn=args.attn,
        **kwargs,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "grid_results.json").write_text(json.dumps(results, indent=2))
    (args.out_dir / "best.json").write_text(json.dumps(results["best"], indent=2))
    logger.info("Best YaRN: %s  score=%.4f", results["best"]["yarn"], results["best"]["score"])
    print(json.dumps(results["best"], indent=2))


if __name__ == "__main__":
    main()
