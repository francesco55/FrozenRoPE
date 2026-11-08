"""Behaviour with non-identity (trained-like) parameters, and the PoSE masking regression.

test_identity.py proves we start from stock OLMo-2. These tests check that the positional
parameters do the right thing once they move away from initialization.
"""

from __future__ import annotations

import math

import pytest
import torch
from transformers import Olmo2ForCausalLM
from transformers.cache_utils import DynamicCache

from posext.models import PosConfig
from posext.models.positional import PositionalEncoding

from .conftest import NATIVE_LEN, make_posext, make_stock, random_ids


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest absolute elementwise difference."""
    return (a - b).abs().max().item()


def freeze_backbone(model: torch.nn.Module) -> None:
    """Only positional parameters train (stand-in until models/build.py exists)."""
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("model.posext."))


# ---- PoSE masking regression ---------------------------------------------------------
# transformers treats jumps in position_ids as packed-sequence boundaries when there is no
# attention mask and no cache; posext defaults to an all-ones mask to prevent that.

POSE_POSITIONS = torch.cat([torch.arange(8), torch.arange(192, 200)])[None]  # 2 chunks, skip of 184


def chunk2_sees_chunk1(model: torch.nn.Module, **kwargs) -> bool:
    """Whether changing chunk-1 tokens changes chunk-2 logits (i.e. chunk 2 attends to chunk 1)."""
    ids = random_ids(1, 16)
    other = ids.clone()
    other[0, :8] = (ids[0, :8] + 1) % 256
    out_a = model(ids, position_ids=POSE_POSITIONS, **kwargs).logits[0, 8:]
    out_b = model(other, position_ids=POSE_POSITIONS, **kwargs).logits[0, 8:]
    return max_diff(out_a, out_b) > 1e-6


@pytest.mark.parametrize("impl", ["eager", "sdpa"])
@pytest.mark.parametrize("use_cache", [True, False])
@torch.no_grad()
def test_pose_chunks_attend_across_skip(stock: Olmo2ForCausalLM, impl: str, use_cache: bool) -> None:
    """With and without the KV cache, PoSE chunk 2 attends to chunk 1 (no attention_mask passed)."""
    model = make_posext({"mode": "learned_freq"}, stock.state_dict(), attn_implementation=impl)
    assert chunk2_sees_chunk1(model, use_cache=use_cache)


def test_pose_chunks_attend_across_skip_in_training(stock: Olmo2ForCausalLM) -> None:
    """Training path (gradient checkpointing forces use_cache=False) keeps chunks connected."""
    model = make_posext({"mode": "learned_freq"}, stock.state_dict()).train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    with torch.no_grad():
        assert chunk2_sees_chunk1(model)


@torch.no_grad()
def test_identity_without_cache(stock: Olmo2ForCausalLM) -> None:
    """Cache off, PoSE positions: equal to stock given an explicit all-ones mask."""
    model = make_posext({"mode": "learned_freq_temp"}, stock.state_dict())
    ids = random_ids(1, 16)
    got = model(ids, position_ids=POSE_POSITIONS, use_cache=False).logits
    want = stock(ids, position_ids=POSE_POSITIONS, use_cache=False, attention_mask=torch.ones_like(ids)).logits
    assert max_diff(got, want) < 1e-5


# ---- KV-cache generation with trained-like parameters --------------------------------


def perturbed(posext: dict, stock: Olmo2ForCausalLM) -> torch.nn.Module:
    """posext model with its positional parameters moved away from identity."""
    model = make_posext(posext, stock.state_dict())
    with torch.no_grad():
        pe = model.posext
        if hasattr(pe, "a"):
            pe.a.normal_(0, 0.1)
        if hasattr(pe, "gamma"):
            pe.gamma.fill_(1.0)
            pe.tau.uniform_(0.8, 1.2)
        if hasattr(pe, "w"):
            pe.w.normal_(0, 0.1)
            pe.alpha.uniform_(0.8, 1.2)
    return model


CACHE_CONFIGS = {
    "learned_freq-head": {"mode": "learned_freq", "granularity": "head"},
    "adascale-tokens": {"mode": "learned_freq_temp", "temp": {"length_source": "tokens"}},
    "adascale-positions": {"mode": "learned_freq_temp", "temp": {"length_source": "positions"}},
    "darope": {"mode": "darope", "darope": {"eps": 1e-4}},
}


@pytest.mark.parametrize("name", list(CACHE_CONFIGS))
@torch.no_grad()
def test_cached_decoding_matches_full_forward(stock: Olmo2ForCausalLM, name: str) -> None:
    """Token-by-token decoding with the KV cache equals one full forward, for 96 > L tokens.

    Exercises the per-query AdaScale length (a whole-sequence length would break this) and
    DaRoPE's content coordinate with the cache.
    """
    model = perturbed(CACHE_CONFIGS[name], stock)
    ids = random_ids(1, 96)
    full = model(ids, use_cache=False).logits[0]
    cache = DynamicCache(config=model.config)
    steps = [model(ids[:, t : t + 1], past_key_values=cache, use_cache=True).logits[0, 0] for t in range(96)]
    assert max_diff(torch.stack(steps), full) < 1e-4


# ---- non-zero parameters give the right output ---------------------------------------


@torch.no_grad()
def test_shared_scale_minus_ln4_equals_pi(stock: Olmo2ForCausalLM) -> None:
    """Shared a = -ln 4 on every band slows all frequencies by 4: exactly PI with factor 4."""
    model = make_posext({"mode": "learned_freq", "granularity": "shared"}, stock.state_dict())
    model.posext.a.fill_(-math.log(4.0))
    pi = make_posext({"mode": "pi", "pi_factor": 4.0}, stock.state_dict())
    ids = random_ids()
    assert max_diff(model(ids).logits, pi(ids).logits) < 1e-5


@torch.no_grad()
def test_layer_granularity_only_affects_its_layer(stock: Olmo2ForCausalLM) -> None:
    """Changing a[2] leaves the hidden states entering layer 2 bit-identical."""
    model = make_posext({"mode": "learned_freq", "granularity": "layer"}, stock.state_dict())
    ids = random_ids()
    before = model(ids, output_hidden_states=True)
    model.posext.a[2].normal_(0, 0.5)
    after = model(ids, output_hidden_states=True)
    for i in range(3):  # embeddings, after layer 0, after layer 1
        assert torch.equal(before.hidden_states[i], after.hidden_states[i]), i
    assert max_diff(before.logits, after.logits) > 1e-3


@torch.no_grad()
def test_head_granularity_only_affects_its_head(stock: Olmo2ForCausalLM) -> None:
    """Changing a[l, h] changes only head h's attention output (before o_proj) in layer l."""
    layer, head = 1, 2
    model = make_posext({"mode": "learned_freq", "granularity": "head"}, stock.state_dict())
    captured: list[torch.Tensor] = []
    hook = model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(
        lambda _m, args: captured.append(args[0].clone())
    )
    ids = random_ids()
    model(ids)
    model.posext.a[layer, head].normal_(0, 0.5)
    model(ids)
    hook.remove()
    n_heads = model.config.num_attention_heads
    before, after = (t.view(*t.shape[:2], n_heads, -1) for t in captured)
    changed = [(before[:, :, h] - after[:, :, h]).abs().max().item() > 1e-6 for h in range(n_heads)]
    assert changed == [h == head for h in range(n_heads)]


@torch.no_grad()
def test_constant_temperature_equals_scaled_attention(stock: Olmo2ForCausalLM) -> None:
    """log_tau = ln 2 on every head == stock with every attention scaling doubled."""
    model = make_posext({"mode": "learned_freq_temp", "temp": {"form": "constant"}}, stock.state_dict())
    model.posext.log_tau.fill_(math.log(2.0))
    ref = make_stock()
    ref.load_state_dict(stock.state_dict())
    for layer in ref.model.layers:
        layer.self_attn.scaling *= 2.0
    ids = random_ids()
    assert max_diff(model(ids).logits, ref(ids).logits) < 1e-5


@pytest.mark.parametrize("source", ["tokens", "positions"])
@torch.no_grad()
def test_adascale_exact_value(source: str) -> None:
    """lambda_h(L_i) = ln(1 + max(L_i, L_ref)/L_ref)^gamma_h / tau_h, with L_i from the chosen source."""
    pos = PosConfig.from_dict({"mode": "learned_freq_temp", "temp": {"length_source": source}})
    module = PositionalEncoding(pos, n_layers=1, n_heads=3, n_kv_heads=3, head_dim=8,
                                hidden_size=8, rope_base=10_000.0, native_len=NATIVE_LEN)
    module.tau.copy_(torch.tensor([[0.5, 1.0, 2.0]]))
    module.gamma.copy_(torch.tensor([[0.0, 1.0, 2.5]]))
    position_ids = torch.tensor([[0, 5, 100, 300]])  # PoSE-like: positions != token indices
    cache_position = torch.arange(4) + 70  # token indices 70..73 (as if after 70 cached tokens)
    lam = module(0, position_ids, cache_position=cache_position).q_scale  # (B or 1, H, T, 1)
    index = position_ids[0].double() if source == "positions" else cache_position.double()
    length = index + 1
    growth = torch.log1p(length.clamp(min=NATIVE_LEN) / NATIVE_LEN)
    expected = growth[None, :] ** module.gamma[0].double()[:, None] / module.tau[0].double()[:, None]
    assert torch.allclose(lam[0, :, :, 0].double(), expected, rtol=1e-5)


# ---- training signal -----------------------------------------------------------------


def test_gradient_checkpointing_non_reentrant_gives_gradients(stock: Olmo2ForCausalLM) -> None:
    """With a frozen backbone, non-reentrant checkpointing still delivers posext gradients."""
    model = make_posext({"mode": "learned_freq", "granularity": "layer"}, stock.state_dict()).train()
    freeze_backbone(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ids = random_ids()
    model(ids, labels=ids).loss.backward()
    assert model.posext.a.grad is not None and (model.posext.a.grad.abs().sum(-1) > 0).all()


@pytest.mark.parametrize(
    "posext,param",
    [
        ({"mode": "learned_freq", "granularity": "layer"}, "a"),
        ({"mode": "learned_freq", "granularity": "head"}, "a"),
        ({"mode": "learned_freq_temp", "temp": {"form": "constant"}}, "log_tau"),
        ({"mode": "learned_freq_temp"}, "tau"),
        ({"mode": "darope", "darope": {"eps": 1e-4}}, "w"),
        ({"mode": "darope", "darope": {"eps": 1e-4}}, "alpha"),
    ],
)
def test_every_layer_and_head_gets_gradient(stock: Olmo2ForCausalLM, posext: dict, param: str) -> None:
    """Each layer row (and each head, where per-head) receives a non-zero gradient."""
    model = make_posext(posext, stock.state_dict()).train()
    freeze_backbone(model)
    ids = random_ids()
    model(ids, labels=ids).loss.backward()
    grad = getattr(model.posext, param).grad
    per_slot = grad.abs().reshape(grad.shape[0], grad.shape[1], -1).sum(-1)  # (layers, heads or 1)
    assert (per_slot > 0).all(), per_slot


# ---- float32 precision where it matters ----------------------------------------------


@torch.no_grad()
def test_rotation_precise_at_long_positions_in_bf16_model(stock: Olmo2ForCausalLM) -> None:
    """In a bf16 model, cos/sin at positions up to 16k stay accurate (fp32 frequencies).

    A logits comparison on the tiny random model is too insensitive, so check cos/sin directly
    against a float64 reference, and confirm bf16 frequencies would fail the same check.
    """
    model = make_posext({"mode": "learned_freq", "granularity": "head"}, stock.state_dict()).to(torch.bfloat16)
    position_ids = torch.tensor([[0, 1000, 4095, 8191, 16383]])
    cos = model.posext(0, position_ids).cos[0, 0]  # (T, D), head 0
    inv64 = 1.0 / (model.posext.rope_base ** (torch.arange(0, 16, 2, dtype=torch.float64) / 16))
    ref = torch.cos(position_ids[0, :, None].double() * torch.cat([inv64, inv64]))
    assert max_diff(cos.double(), ref) < 1e-2
    bf16_inv = inv64.to(torch.bfloat16).double()
    bad = torch.cos(position_ids[0, :, None].double() * torch.cat([bf16_inv, bf16_inv]))
    assert max_diff(bad, ref) > 0.1  # the check is sensitive to the failure it guards against
