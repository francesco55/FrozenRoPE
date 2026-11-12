"""At initialization, every learned positional mode must reproduce stock OLMo-2 exactly.

This is the gate for the project (CLAUDE.md, "Testing"): if it fails, a learned method's
gains could come from a changed starting point rather than from learning.
Runs on CPU with a tiny randomly initialized OLMo-2 in float32.
"""

from __future__ import annotations

import tempfile

import pytest
import torch
from transformers import Olmo2Config, Olmo2ForCausalLM

from posext.models import Olmo2PosExtConfig, Olmo2PosExtForCausalLM

from .conftest import (
    IDENTITY_CONFIGS,
    NATIVE_LEN,
    TINY,
    gapped_positions,
    make_posext,
    random_ids,
)

ATOL = 1e-5  # float32; DaRoPE's 0.5 position offset gives ~3e-7, everything else is exact


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest absolute elementwise difference."""
    return (a - b).abs().max().item()


@pytest.mark.parametrize("name", list(IDENTITY_CONFIGS))
@pytest.mark.parametrize("positions", ["default", "contiguous", "gapped", "gapped-nonzero-start"])
@torch.no_grad()
def test_identity_logits(stock: Olmo2ForCausalLM, name: str, positions: str) -> None:
    """Logits equal stock OLMo-2 for default, explicit, and non-contiguous position ids < L."""
    model = make_posext(IDENTITY_CONFIGS[name], stock.state_dict())
    batch, length = 2, 48
    ids = random_ids(batch, length)
    pos = {
        "default": None,
        "contiguous": torch.arange(length).expand(batch, -1),
        "gapped": gapped_positions(batch, length, NATIVE_LEN),
        "gapped-nonzero-start": gapped_positions(batch, length, NATIVE_LEN, start_at_zero=False),
    }[positions]
    got = model(ids, position_ids=pos).logits
    want = stock(ids, position_ids=pos).logits
    assert max_diff(got, want) < ATOL


@pytest.mark.parametrize("name", ["temp-adascale-tokens", "darope"])
@pytest.mark.parametrize("length", [1, 7, NATIVE_LEN])
@torch.no_grad()
def test_identity_lengths(stock: Olmo2ForCausalLM, name: str, length: int) -> None:
    """Identity holds from a single token up to the full native length L.

    For DaRoPE, position L-1 is where a too-large eps would clip (m + 0.5)/L.
    """
    model = make_posext(IDENTITY_CONFIGS[name], stock.state_dict())
    ids = random_ids(1, length)
    assert max_diff(model(ids).logits, stock(ids).logits) < ATOL


@pytest.mark.parametrize(
    "name", [n for n in IDENTITY_CONFIGS if not n.startswith("darope")]
)
@torch.no_grad()
def test_identity_beyond_native_length(stock: Olmo2ForCausalLM, name: str) -> None:
    """Non-DaRoPE modes are the identity at any position (a = 0, lambda = 1), even > L.

    DaRoPE is excluded on purpose: positions >= L saturate by design (see test_positional.py).
    """
    model = make_posext(IDENTITY_CONFIGS[name], stock.state_dict())
    ids = random_ids(2, 32)
    pos = gapped_positions(2, 32, 4 * NATIVE_LEN)
    assert max_diff(model(ids, position_ids=pos).logits, stock(ids, position_ids=pos).logits) < ATOL


@pytest.mark.parametrize("name", ["learned_freq-head", "temp-adascale-tokens", "darope"])
@torch.no_grad()
def test_identity_sdpa_backend(stock: Olmo2ForCausalLM, name: str) -> None:
    """The SDPA kernel after our rotation gives the same logits as stock eager attention."""
    model = make_posext(IDENTITY_CONFIGS[name], stock.state_dict(), attn_implementation="sdpa")
    ids = random_ids()
    assert max_diff(model(ids).logits, stock(ids).logits) < ATOL


@pytest.mark.parametrize("name", ["none", "temp-adascale-tokens", "darope"])
@torch.no_grad()
def test_identity_kv_cache_generation(stock: Olmo2ForCausalLM, name: str) -> None:
    """Greedy generation with the KV cache produces the same tokens as stock."""
    model = make_posext(IDENTITY_CONFIGS[name], stock.state_dict())
    prompt = random_ids(1, 10)
    kwargs = dict(max_new_tokens=12, do_sample=False, pad_token_id=0)
    assert torch.equal(model.generate(prompt, **kwargs), stock.generate(prompt, **kwargs))


@pytest.mark.parametrize("name", ["learned_freq-layer", "temp-adascale-tokens", "darope"])
@torch.no_grad()
def test_identity_via_from_pretrained(stock: Olmo2ForCausalLM, name: str) -> None:
    """Loading a stock checkpoint into the posext class initializes positional params to identity."""
    ids = random_ids()
    with tempfile.TemporaryDirectory() as tmp:
        stock.save_pretrained(tmp)
        cfg = Olmo2PosExtConfig(
            **Olmo2Config.from_pretrained(tmp).to_dict(), posext=IDENTITY_CONFIGS[name]
        )
        model = Olmo2PosExtForCausalLM.from_pretrained(tmp, config=cfg).eval()
    assert max_diff(model(ids).logits, stock(ids).logits) < ATOL
    assert model.config.posext["native_len"] == TINY["max_position_embeddings"]
