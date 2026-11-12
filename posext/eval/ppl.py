"""PG-19 perplexity at 4k / 8k / 16k.

    uv run python -m posext.eval.ppl --run outputs/rq1_learned_freq_head --lengths 4096,8192,16384
    uv run python -m posext.eval.ppl --pos-mode yarn --lengths 4096,8192,16384  # training-free
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from posext.data import Collator, load_pg19_chunks
from posext.loss import chunked_lm_cross_entropy
from posext.models import load_model, load_trained
from posext.models.build import DEFAULT_MODEL
from posext.models.positional import PosConfig

logger = logging.getLogger("posext.eval.ppl")

DEFAULT_LENGTHS = (4096, 8192, 16384)


@torch.no_grad()
def perplexity_on_chunks(
    model,
    chunks: torch.Tensor,
    *,
    batch_size: int = 1,
    chunk_size: int = 2048,
    target_len: int | None = None,
) -> dict[str, float]:
    """Mean NLL / PPL over a (N, S) chunk tensor. Positions are contiguous 0..S-1."""
    model.eval()
    seq_len = chunks.shape[1]
    collator = Collator("long", target_len or seq_len)
    loader = DataLoader(
        TensorDataset(chunks),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda rows: collator(torch.stack([r[0] for r in rows])),
    )
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            use_cache=False,
        )
        # Recompute token count for averaging (ignore none; all labels valid).
        n = batch["labels"][:, 1:].numel()
        loss = chunked_lm_cross_entropy(
            model.lm_head, outputs.last_hidden_state, batch["labels"], chunk_size=chunk_size
        )
        total_loss += float(loss.item()) * n
        total_tokens += n
    nll = total_loss / max(total_tokens, 1)
    return {"nll": nll, "ppl": math.exp(min(nll, 20.0)), "tokens": float(total_tokens)}


def evaluate_ppl(
    model,
    *,
    split: str = "test",
    lengths: tuple[int, ...] = DEFAULT_LENGTHS,
    max_sequences: int | None = 50,
    seed: int = 0,
    batch_size: int = 1,
    logit_chunk_size: int = 2048,
) -> dict[str, Any]:
    """Perplexity at each length on PG-19 `split`."""
    results: dict[str, Any] = {"split": split, "seed": seed, "lengths": {}}
    for length in lengths:
        chunks = load_pg19_chunks(split, length, max_sequences=max_sequences, seed=seed)
        stats = perplexity_on_chunks(
            model, chunks, batch_size=batch_size, chunk_size=logit_chunk_size, target_len=length
        )
        results["lengths"][str(length)] = stats
        logger.info("PPL@%d (%s): %.3f  (nll=%.4f, %d tokens)", length, split, stats["ppl"], stats["nll"], int(stats["tokens"]))
    return results


def _load_eval_model(args: argparse.Namespace):
    """Load from a trained run dir, or a training-free pos mode."""
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.run:
        model = load_trained(args.run, device=device, attn_implementation=args.attn)
    else:
        pos = {"mode": args.pos_mode}
        if args.pos_mode == "pi":
            pos["pi_factor"] = args.factor
        if args.pos_mode == "yarn":
            pos["yarn"] = {
                "factor": args.factor,
                "beta_fast": args.beta_fast,
                "beta_slow": args.beta_slow,
                "attention_factor": args.attention_factor,
            }
        PosConfig.from_dict(pos)
        model = load_model(
            args.model, pos=pos, device=device, attn_implementation=args.attn, dtype=torch.bfloat16
        )
    model.eval()
    return model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=None, help="Trained run dir (positional.safetensors)")
    p.add_argument("--pos-mode", default="none", choices=["none", "pi", "yarn"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--lengths", default="4096,8192,16384")
    p.add_argument("--max-sequences", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--factor", type=float, default=4.0)
    p.add_argument("--beta-fast", type=float, default=32.0)
    p.add_argument("--beta-slow", type=float, default=1.0)
    p.add_argument("--attention-factor", type=float, default=None)
    p.add_argument("--out", type=Path, default=None, help="Write JSON results here")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    lengths = tuple(int(x) for x in args.lengths.split(",") if x)
    model = _load_eval_model(args)
    results = evaluate_ppl(
        model,
        split=args.split,
        lengths=lengths,
        max_sequences=args.max_sequences,
        seed=args.seed,
    )
    results["source"] = str(args.run) if args.run else {"pos_mode": args.pos_mode}
    text = json.dumps(results, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
