"""Sanity-check the pinned environment against the real OLMo-2 checkpoint.

Checks that transformers loads allenai/OLMo-2-0425-1B, that the config matches the
model facts in CLAUDE.md, that a short forward/generate gives sensible output, and
whether stock transformers YaRN `rope_scaling` works for OLMo-2.

Usage: uv run python scripts/check_env.py [--model allenai/OLMo-2-0425-1B]
Needs ~4 GB RAM (bf16) and downloads ~6 GB on first run.
"""

from __future__ import annotations

import argparse
import math

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

EXPECTED = {
    "num_hidden_layers": 16,
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "rope_theta": 500_000,
    "max_position_embeddings": 4096,
}
PROMPT = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"


def check_config(name: str) -> None:
    cfg = AutoConfig.from_pretrained(name)
    for key, want in EXPECTED.items():
        got = getattr(cfg, key)
        assert got == want, f"{key}: expected {want}, got {got}"
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    assert head_dim == 128, f"head_dim: expected 128, got {head_dim}"
    print(f"[ok] config matches CLAUDE.md model facts (head_dim={head_dim})")


@torch.no_grad()
def check_forward(name: str) -> torch.nn.Module:
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    ids = tok(PROMPT, return_tensors="pt").input_ids
    loss = model(ids, labels=ids).loss.item()
    gen = model.generate(ids, max_new_tokens=3, do_sample=False)
    text = tok.decode(gen[0, ids.shape[1]:])
    print(f"[ok] loaded {type(model).__name__}: {n / 1e9:.3f}B params")
    print(f"[ok] prompt loss {loss:.3f} (ppl {math.exp(loss):.1f}); greedy continuation: {text!r}")
    return model


@torch.no_grad()
def check_stock_yarn(name: str, base_model: torch.nn.Module) -> None:
    cfg = AutoConfig.from_pretrained(name)
    cfg.rope_scaling = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096}
    try:
        model = AutoModelForCausalLM.from_pretrained(name, config=cfg, dtype=torch.bfloat16)
    except Exception as e:  # noqa: BLE001 - report any failure
        print(f"[FAIL] stock YaRN rope_scaling: {type(e).__name__}: {e}")
        return
    rot = model.model.rotary_emb
    changed = not torch.allclose(rot.inv_freq, base_model.model.rotary_emb.inv_freq)
    print(
        f"[ok] stock YaRN: rope_type={rot.rope_type}, attention_scaling={rot.attention_scaling:.4f}, "
        f"inv_freq changed={changed}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    args = parser.parse_args()
    print(f"transformers {transformers.__version__}, torch {torch.__version__}")
    check_config(args.model)
    model = check_forward(args.model)
    check_stock_yarn(args.model, model)


if __name__ == "__main__":
    main()
