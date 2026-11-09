"""Optional: identity against the real allenai/OLMo-2-0425-1B checkpoint.

Skipped unless POSEXT_REAL_CKPT=1. Uses CUDA if available, otherwise CPU (needs ~12 GB RAM in
float32; the weights download once, ~6 GB). On SCC: `POSEXT_REAL_CKPT=1 uv run pytest -m real`.
"""

from __future__ import annotations

import gc
import os

import pytest
import torch

pytestmark = [
    pytest.mark.real,
    pytest.mark.skipif(os.environ.get("POSEXT_REAL_CKPT") != "1", reason="set POSEXT_REAL_CKPT=1"),
]

NAME = "allenai/OLMo-2-0425-1B"
# DaRoPE's 0.5 offset gives ~2e-4 on logits of magnitude ~25 in float32; other modes are exact.
ATOL = {"learned_freq_temp": 1e-5, "darope": 1e-3}


@pytest.fixture(scope="module")
def reference() -> dict:
    """Stock logits for contiguous and gapped positions, computed once (model freed after)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = AutoConfig.from_pretrained(NAME)
    # Model facts from CLAUDE.md.
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads) == (16, 2048, 16, 16)
    assert (cfg.rope_theta, cfg.max_position_embeddings) == (500_000, 4096)
    torch.manual_seed(0)
    length = 1024 if device == "cuda" else 512
    ids = torch.randint(0, cfg.vocab_size, (1, length), device=device)
    gapped = torch.cat([torch.zeros(1), torch.sort(torch.randperm(4095)[: length - 1] + 1)[0]]).long()[None].to(device)
    stock = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").to(device).eval()
    with torch.no_grad():
        out = {"ids": ids, "gapped": gapped, "device": device, "config": cfg.to_dict(),
               "contig_logits": stock(ids).logits, "gapped_logits": stock(ids, position_ids=gapped).logits}
    del stock
    gc.collect()
    return out


@pytest.mark.parametrize(
    "posext",
    [
        {"mode": "learned_freq_temp", "granularity": "head"},
        {"mode": "darope"},
    ],
    ids=["learned_freq_temp-head", "darope"],
)
@torch.no_grad()
def test_real_checkpoint_identity(reference: dict, posext: dict) -> None:
    """Patched model at init matches stock OLMo-2-1B on contiguous and gapped positions < 4096."""
    from posext.models import Olmo2PosExtConfig, Olmo2PosExtForCausalLM

    cfg = Olmo2PosExtConfig(**reference["config"], posext=posext)
    model = Olmo2PosExtForCausalLM.from_pretrained(
        NAME, config=cfg, dtype=torch.float32, attn_implementation="sdpa"
    ).to(reference["device"]).eval()
    assert model.posext.slow_mask.nonzero().flatten().tolist() == list(range(32, 64))
    ids, gapped = reference["ids"], reference["gapped"]
    atol = ATOL[posext["mode"]]  # see top of file
    assert (model(ids).logits - reference["contig_logits"]).abs().max().item() < atol
    assert (model(ids, position_ids=gapped).logits - reference["gapped_logits"]).abs().max().item() < atol
    del model
    gc.collect()


@pytest.mark.parametrize(
    "posext,expected",
    [
        ({"mode": "learned_freq", "granularity": "shared"}, 64),
        ({"mode": "learned_freq_temp", "granularity": "head"}, 16 * 16 * 64 + 2 * 16 * 16),
        ({"mode": "darope"}, 16 * 16 * (2048 + 1)),
    ],
    ids=["shared", "learned_freq_temp-head", "darope"],
)
def test_real_checkpoint_build_and_freeze(posext: dict, expected: int) -> None:
    """load_model checks the model facts; after freezing, trainable counts match CLAUDE.md."""
    from posext.models.build import count_parameters, freeze_backbone, load_model

    model = load_model(NAME, posext)  # bf16 backbone, fp32 positional params
    freeze_backbone(model)
    counts = count_parameters(model)
    assert counts["trainable"] == expected
    assert 1.4e9 < counts["total"] < 1.6e9
    del model
    gc.collect()
