"""Shared fixtures: a tiny randomly initialized OLMo-2 and helpers to build posext twins."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from transformers import Olmo2Config, Olmo2ForCausalLM

from posext.models import Olmo2PosExtConfig, Olmo2PosExtForCausalLM

NATIVE_LEN = 64  # tiny model's native context L

# Small but structurally faithful: full multi-head attention (no GQA), QK-norm, reordered norm.
TINY = dict(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=4,
    max_position_embeddings=NATIVE_LEN,
    rope_theta=10000.0,
)

# Every identity-at-init configuration (mode name -> posext dict). DaRoPE's eps must be < 0.5/L.
IDENTITY_CONFIGS: dict[str, dict[str, Any]] = {
    "none": {},
    "learned_freq-shared": {"mode": "learned_freq", "granularity": "shared"},
    "learned_freq-layer": {"mode": "learned_freq", "granularity": "layer"},
    "learned_freq-head": {"mode": "learned_freq", "granularity": "head"},
    "temp-constant": {"mode": "learned_freq_temp", "temp": {"form": "constant"}},
    "temp-adascale-tokens": {"mode": "learned_freq_temp", "temp": {"form": "adascale"}},
    "temp-adascale-positions": {
        "mode": "learned_freq_temp",
        "temp": {"form": "adascale", "length_source": "positions", "ref_len": 8},
    },
    "darope": {"mode": "darope", "darope": {"eps": 1e-4}},
    "darope-no-rmsnorm": {"mode": "darope", "darope": {"eps": 1e-4, "rms_norm_input": False}},
}


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Make every test deterministic."""
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def stock() -> Olmo2ForCausalLM:
    """Stock Hugging Face OLMo-2 with random weights (the reference), float32, eager attention."""
    torch.manual_seed(1234)
    return Olmo2ForCausalLM(Olmo2Config(**TINY, attn_implementation="eager")).eval()


def make_stock(rope_scaling: dict[str, Any] | None = None, **overrides: Any) -> Olmo2ForCausalLM:
    """Stock model with optional rope_scaling; weights must be loaded by the caller.

    `rope_scaling` is set after construction because Olmo2Config's validator only accepts
    the keys `type`/`factor`.
    """
    cfg = Olmo2Config(**{**TINY, **overrides}, attn_implementation="eager")
    cfg.rope_scaling = rope_scaling
    return Olmo2ForCausalLM(cfg).eval()


def make_posext(
    posext: dict[str, Any], state_dict: dict[str, torch.Tensor] | None = None, **overrides: Any
) -> Olmo2PosExtForCausalLM:
    """posext model with the given positional config, optionally loading backbone weights."""
    overrides.setdefault("attn_implementation", "eager")
    model = Olmo2PosExtForCausalLM(Olmo2PosExtConfig(**{**TINY, **overrides}, posext=posext)).eval()
    if state_dict is not None:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        assert not unexpected, unexpected
        assert all(k.startswith("model.posext.") for k in missing), missing
    return model


def random_ids(batch: int = 2, length: int = 48) -> torch.Tensor:
    """Random token ids."""
    return torch.randint(0, TINY["vocab_size"], (batch, length))


def gapped_positions(batch: int, length: int, high: int, start_at_zero: bool = True) -> torch.Tensor:
    """Strictly increasing position ids with random gaps, all < `high` (PoSE-like)."""
    rows = []
    for _ in range(batch):
        if start_at_zero:  # 0 followed by length-1 distinct values from 1 .. high-1
            rest = torch.sort(torch.randperm(high - 1)[: length - 1] + 1)[0]
            rows.append(torch.cat([torch.zeros(1, dtype=rest.dtype), rest]))
        else:  # length distinct values from 1 .. high-1
            rows.append(torch.sort(torch.randperm(high - 1)[:length] + 1)[0])
    return torch.stack(rows)
