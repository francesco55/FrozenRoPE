"""Shared evaluation helpers: model loading from CLI flags and greedy generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from posext.models import load_model, load_trained
from posext.models.build import DEFAULT_MODEL
from posext.models.positional import PosConfig


def add_model_args(p: argparse.ArgumentParser) -> None:
    """Flags selecting a trained run (`--run`) or a training-free positional mode."""
    p.add_argument("--run", type=Path, default=None, help="Trained run dir (positional.safetensors)")
    p.add_argument("--pos-mode", default="none", choices=["none", "pi", "yarn"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default=None)
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--factor", type=float, default=4.0)
    p.add_argument("--beta-fast", type=float, default=32.0)
    p.add_argument("--beta-slow", type=float, default=1.0)
    p.add_argument("--attention-factor", type=float, default=None)


def pos_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """The `pos` dict for a training-free mode (validated)."""
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
    return pos


def model_source(args: argparse.Namespace) -> str | dict[str, Any]:
    """What was evaluated, for result JSONs."""
    return str(args.run) if args.run else pos_from_args(args)


def load_eval_model(args: argparse.Namespace):
    """Load from a trained run dir, or the base model in a training-free pos mode (bf16)."""
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.run:
        model = load_trained(args.run, device=device, attn_implementation=args.attn)
    else:
        model = load_model(
            args.model,
            pos=pos_from_args(args),
            device=device,
            attn_implementation=args.attn,
            dtype=torch.bfloat16,
        )
    model.eval()
    return model


@torch.no_grad()
def greedy_generate(
    model, input_ids: list[int], *, max_new_tokens: int, pad_token_id: int
) -> list[int]:
    """Greedy KV-cached continuation of one prompt (batch size 1). Returns the new token ids.

    No explicit position_ids: generate() never extends a passed tensor, so every new token
    would reuse the last prompt position. Without it, positions come from the attention mask
    and keep counting (0..n-1 for the prompt, then n, n+1, ...). Batch size 1 also keeps
    AdaScale's `tokens` length source exact (no left padding).
    """
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], device=device)
    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=pad_token_id,
    )
    return out[0, ids.shape[1] :].tolist()
