"""Synthetic passkey retrieval (Landmark Attention / Mohtashami & Jaggi style).

    uv run python -m posext.eval.passkey --run outputs/rq1_learned_freq_head
    uv run python -m posext.eval.passkey --pos-mode yarn --split validation
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from posext.eval.common import greedy_generate
from posext.models import load_model, load_trained
from posext.models.build import DEFAULT_MODEL
from posext.models.positional import PosConfig

logger = logging.getLogger("posext.eval.passkey")

FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. Soft and "
    "warm like summer days. "
)
KEY_TEMPLATE = "The pass key is {key}. Remember it. {key} is the pass key. "
QUERY = "What is the pass key? The pass key is"


@dataclass(frozen=True)
class PasskeyPrompt:
    """One retrieval prompt at a controlled length and depth."""

    input_ids: list[int]
    key: str
    length: int
    depth: float  # 0 = beginning, 1 = just before the query


def _repeat_to_token_budget(tokenizer, text: str, n_tokens: int) -> list[int]:
    """Repeat `text` until we have at least n_tokens, then truncate."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("filler text tokenized to empty")
    out: list[int] = []
    while len(out) < n_tokens:
        out.extend(ids)
    return out[:n_tokens]


def generate_passkey_prompts(
    tokenizer,
    *,
    lengths: tuple[int, ...] = (4096, 8192, 16384),
    depths: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    n_per_cell: int = 5,
    seed: int = 0,
) -> list[PasskeyPrompt]:
    """Build prompts: filler + key sentence at `depth` + filler + query.

    Validation and test must use different seeds (CLAUDE.md).
    """
    rng = torch.Generator().manual_seed(seed)
    query_ids = tokenizer(QUERY, add_special_tokens=False)["input_ids"]
    prompts: list[PasskeyPrompt] = []
    for length in lengths:
        for depth in depths:
            for _ in range(n_per_cell):
                key = f"{int(torch.randint(0, 100_000, (1,), generator=rng).item()):05d}"
                key_ids = tokenizer(KEY_TEMPLATE.format(key=key), add_special_tokens=False)["input_ids"]
                # Leave room for key + query inside `length`.
                budget = length - len(key_ids) - len(query_ids)
                if budget < 2:
                    raise ValueError(f"length {length} too short for key+query")
                prefix_len = int(budget * depth)
                suffix_len = budget - prefix_len
                prefix = _repeat_to_token_budget(tokenizer, FILLER, prefix_len)
                suffix = _repeat_to_token_budget(tokenizer, FILLER, suffix_len)
                input_ids = prefix + key_ids + suffix + query_ids
                assert len(input_ids) == length, (len(input_ids), length)
                prompts.append(PasskeyPrompt(input_ids, key, length, depth))
    return prompts


@torch.no_grad()
def answer_passkey(model, tokenizer, prompt: PasskeyPrompt, *, max_new_tokens: int = 16) -> str:
    """Greedy-generate a short continuation and return the decoded text."""
    new_tokens = greedy_generate(
        model, prompt.input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def key_in_text(key: str, text: str) -> bool:
    """Whether the 5-digit key appears as a whole number in the generation."""
    return re.search(rf"\b{re.escape(key)}\b", text) is not None


def evaluate_passkey(
    model,
    tokenizer,
    prompts: list[PasskeyPrompt] | None = None,
    *,
    lengths: tuple[int, ...] = (4096, 8192, 16384),
    depths: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    n_per_cell: int = 5,
    seed: int = 0,
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    """Accuracy overall and broken down by length and depth."""
    if prompts is None:
        prompts = generate_passkey_prompts(
            tokenizer, lengths=lengths, depths=depths, n_per_cell=n_per_cell, seed=seed
        )
    model.eval()
    rows = []
    correct = 0
    for i, prompt in enumerate(prompts):
        text = answer_passkey(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        ok = key_in_text(prompt.key, text)
        correct += int(ok)
        rows.append(
            {
                **{k: v for k, v in asdict(prompt).items() if k != "input_ids"},
                "ok": ok,
                "generation": text[:200],
            }
        )
        if (i + 1) % 10 == 0:
            logger.info("passkey %d/%d  running acc=%.2f", i + 1, len(prompts), correct / (i + 1))

    by_length: dict[str, list[bool]] = {}
    by_depth: dict[str, list[bool]] = {}
    for row in rows:
        by_length.setdefault(str(row["length"]), []).append(row["ok"])
        by_depth.setdefault(str(row["depth"]), []).append(row["ok"])

    def _acc(vals: list[bool]) -> float:
        return sum(vals) / max(len(vals), 1)

    return {
        "seed": seed,
        "n": len(rows),
        "accuracy": _acc([r["ok"] for r in rows]),
        "by_length": {k: _acc(v) for k, v in by_length.items()},
        "by_depth": {k: _acc(v) for k, v in by_depth.items()},
        "details": rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=None)
    p.add_argument("--pos-mode", default="none", choices=["none", "pi", "yarn"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--lengths", default="4096,8192,16384")
    p.add_argument("--depths", default="0,0.25,0.5,0.75,1")
    p.add_argument("--n-per-cell", type=int, default=5)
    p.add_argument("--seed", type=int, default=0, help="Use a different seed for validation vs test")
    p.add_argument("--device", default=None)
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--factor", type=float, default=4.0)
    p.add_argument("--beta-fast", type=float, default=32.0)
    p.add_argument("--beta-slow", type=float, default=1.0)
    p.add_argument("--attention-factor", type=float, default=None)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.run:
        model = load_trained(args.run, device=device, attn_implementation=args.attn)
    else:
        pos: dict[str, Any] = {"mode": args.pos_mode}
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
    lengths = tuple(int(x) for x in args.lengths.split(",") if x)
    depths = tuple(float(x) for x in args.depths.split(",") if x)
    results = evaluate_passkey(
        model,
        tokenizer,
        lengths=lengths,
        depths=depths,
        n_per_cell=args.n_per_cell,
        seed=args.seed,
    )
    results["source"] = str(args.run) if args.run else {"pos_mode": args.pos_mode}
    # Details are large; keep them only in the file.
    summary = {k: v for k, v in results.items() if k != "details"}
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
