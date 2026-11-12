"""Unit tests for posext/models/positional.py and the model's positional plumbing (CPU, tiny model)."""

from __future__ import annotations

import math
import tempfile

import pytest
import torch
from transformers import Olmo2Config, Olmo2ForCausalLM

from posext.models import Olmo2PosExtConfig, Olmo2PosExtForCausalLM, PosConfig
from posext.models.positional import PositionalEncoding, rope_inv_freq, slow_band_mask

from .conftest import IDENTITY_CONFIGS, NATIVE_LEN, TINY, make_posext, make_stock, random_ids

ALL_CONFIGS = IDENTITY_CONFIGS | {"pi": {"mode": "pi"}, "yarn": {"mode": "yarn"}}

# OLMo-2-0425-1B dimensions (CLAUDE.md, "Model facts").
OLMO = dict(n_layers=16, n_heads=16, n_kv_heads=16, head_dim=128, hidden_size=2048,
            rope_base=500_000.0, native_len=4096)


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest absolute elementwise difference."""
    return (a - b).abs().max().item()


# ---- slow-band mask -------------------------------------------------------------------


@pytest.mark.parametrize("head_dim,base,native_len", [(128, 500_000.0, 4096), (16, 10_000.0, 64), (64, 10_000.0, 2048)])
def test_slow_band_mask_matches_formula(head_dim: int, base: float, native_len: int) -> None:
    """Band j is slow iff its wavelength 2*pi*base^(2j/d) exceeds L (computed independently)."""
    mask = slow_band_mask(rope_inv_freq(head_dim, base), native_len)
    expected = [2 * math.pi * base ** (2 * j / head_dim) > native_len for j in range(head_dim // 2)]
    assert mask.tolist() == expected


def test_slow_bands_for_olmo2() -> None:
    """For OLMo-2-1B (d=128, base=500k, L=4096) the slow bands are exactly 32..63."""
    mask = slow_band_mask(rope_inv_freq(128, 500_000.0), 4096)
    assert mask.nonzero().flatten().tolist() == list(range(32, 64))


# ---- baselines match stock transformers ----------------------------------------------


YARN_SETTINGS = [
    {"factor": 4.0, "beta_fast": 32.0, "beta_slow": 1.0},
    {"factor": 6.0, "beta_fast": 16.0, "beta_slow": 2.0, "attention_factor": 1.2},
    {"factor": 8.0, "beta_fast": 64.0, "beta_slow": 0.5, "attention_factor": 1.0},
]


@pytest.mark.parametrize("yarn", YARN_SETTINGS)
@torch.no_grad()
def test_yarn_matches_stock(stock: Olmo2ForCausalLM, yarn: dict) -> None:
    """Our `yarn` mode equals transformers' rope_scaling yarn (frequencies and attention factor)."""
    ref = make_stock({"rope_type": "yarn", **yarn})
    ref.load_state_dict(stock.state_dict())
    model = make_posext({"mode": "yarn", "yarn": yarn}, stock.state_dict())
    ids = random_ids()
    assert max_diff(model(ids).logits, ref(ids).logits) < 1e-5


@pytest.mark.parametrize("factor", [2.0, 4.0])
@torch.no_grad()
def test_pi_matches_stock_linear(stock: Olmo2ForCausalLM, factor: float) -> None:
    """Our `pi` mode equals transformers' linear rope scaling."""
    ref = make_stock({"rope_type": "linear", "factor": factor})
    ref.load_state_dict(stock.state_dict())
    model = make_posext({"mode": "pi", "pi_factor": factor}, stock.state_dict())
    ids = random_ids()
    assert max_diff(model(ids).logits, ref(ids).logits) < 1e-5


@torch.no_grad()
def test_init_from_yarn_equals_yarn_frequencies(stock: Olmo2ForCausalLM) -> None:
    """learned_freq (no temperature) + init_from_yarn: YaRN's frequencies, no attention factor."""
    yarn = YARN_SETTINGS[0]
    ref = make_stock({"rope_type": "yarn", **yarn, "attention_factor": 1.0})
    ref.load_state_dict(stock.state_dict())
    model = make_posext({"mode": "learned_freq", "init_from_yarn": True, "yarn": yarn}, stock.state_dict())
    ids = random_ids()
    assert max_diff(model(ids).logits, ref(ids).logits) < 1e-5


@pytest.mark.parametrize("granularity", ["shared", "layer", "head"])
@pytest.mark.parametrize("temp", [{"form": "constant"}, {"form": "adascale"},
                                  {"form": "adascale", "length_source": "positions"}])
@pytest.mark.parametrize("yarn", YARN_SETTINGS[:2])
@torch.no_grad()
def test_init_from_yarn_with_temperature_equals_yarn(
    stock: Olmo2ForCausalLM, granularity: str, temp: dict, yarn: dict
) -> None:
    """learned_freq_temp + init_from_yarn equals `yarn` mode exactly (frequencies AND attention
    factor), also beyond the native length where AdaScale would grow if gamma were not 0."""
    ref = make_posext({"mode": "yarn", "yarn": yarn}, stock.state_dict())
    model = make_posext({"mode": "learned_freq_temp", "granularity": granularity, "init_from_yarn": True,
                         "temp": temp, "yarn": yarn}, stock.state_dict())
    ids = random_ids(batch=1, length=3 * NATIVE_LEN)
    assert max_diff(model(ids).logits, ref(ids).logits) < 1e-5


def test_init_from_yarn_temperature_values() -> None:
    """The stored temperature is YaRN's logit scale f^2 (f = 0.1 ln 4 + 1 by default)."""
    f2 = (0.1 * math.log(4.0) + 1.0) ** 2
    model = make_posext({"mode": "learned_freq_temp", "init_from_yarn": True, "temp": {"form": "constant"}})
    assert torch.allclose(model.model.posext.log_tau.exp(), torch.full_like(model.model.posext.log_tau, f2))
    model = make_posext({"mode": "learned_freq_temp", "init_from_yarn": True, "temp": {"form": "adascale"}})
    pe = model.model.posext
    assert torch.allclose(pe.tau, torch.full_like(pe.tau, 1 / f2)) and (pe.gamma == 0).all()


# ---- configuration -------------------------------------------------------------------


@pytest.mark.parametrize("name", list(ALL_CONFIGS))
def test_every_mode_constructs(name: str) -> None:
    """Every configuration builds (regression: modes without parameters once failed post_init)."""
    make_posext(ALL_CONFIGS[name])


@pytest.mark.parametrize(
    "bad",
    [
        {"mode": "nope"},
        {"granularity": "per_token"},
        {"temp": {"form": "softmax"}},
        {"temp": {"length_source": "span"}},
        {"typo_key": 1},
        {"yarn": {"facter": 4.0}},
    ],
)
def test_config_rejects_bad_values(bad: dict) -> None:
    """Unknown modes, values and keys fail loudly instead of being ignored."""
    with pytest.raises(ValueError):
        PosConfig.from_dict(bad)


def test_config_round_trip() -> None:
    """PosConfig -> dict -> PosConfig is lossless (the dict is what HF configs store)."""
    cfg = PosConfig.from_dict({"mode": "learned_freq_temp", "granularity": "layer",
                               "temp": {"form": "constant"}, "yarn": {"factor": 6.0}})
    assert PosConfig.from_dict(cfg.to_dict()) == cfg


def test_darope_eps_guard() -> None:
    """darope.eps >= 0.5/L would clip positions inside L, so it is rejected."""
    with pytest.raises(ValueError, match="eps"):
        make_posext({"mode": "darope", "darope": {"eps": 0.5 / NATIVE_LEN}})


@pytest.mark.parametrize("pos", [{"mode": "darope", "darope": {"eps": 1e-4}},
                                 {"mode": "learned_freq", "granularity": "head"}])
def test_per_head_modes_reject_gqa(pos: dict) -> None:
    """Per-head modes assume no grouped-query attention."""
    with pytest.raises(NotImplementedError):
        make_posext(pos, num_key_value_heads=2)


def test_rope_scaling_must_be_none() -> None:
    """Scaling must go through posext.mode, not config.rope_scaling (it would be ignored)."""
    cfg = Olmo2PosExtConfig(**TINY, posext={})
    cfg.rope_scaling = {"rope_type": "linear", "factor": 2.0}
    with pytest.raises(ValueError, match="rope_scaling"):
        Olmo2PosExtForCausalLM(cfg)


# ---- parameters: shapes, counts, registration ----------------------------------------


@pytest.mark.parametrize(
    "pos,expected",
    [
        ({"mode": "none"}, {}),
        ({"mode": "yarn"}, {}),
        ({"mode": "learned_freq", "granularity": "shared"}, {"a": (1, 1, 64)}),
        ({"mode": "learned_freq", "granularity": "layer"}, {"a": (16, 1, 64)}),
        ({"mode": "learned_freq", "granularity": "head"}, {"a": (16, 16, 64)}),
        ({"mode": "learned_freq_temp", "temp": {"form": "constant"}},
         {"a": (16, 16, 64), "log_tau": (16, 16)}),
        ({"mode": "learned_freq_temp"}, {"a": (16, 16, 64), "tau": (16, 16), "gamma": (16, 16)}),
        ({"mode": "darope"}, {"w": (16, 16, 2048), "alpha": (16, 16)}),
    ],
)
def test_parameter_shapes_at_olmo2_size(pos: dict, expected: dict) -> None:
    """Shapes match CLAUDE.md for OLMo-2-1B; DaRoPE totals 16*16*(2048+1) = 524,544."""
    module = PositionalEncoding(PosConfig.from_dict(pos), **OLMO)
    shapes = {n: tuple(p.shape) for n, p in module.named_parameters()}
    assert shapes == expected
    if pos["mode"] == "darope":
        assert sum(p.numel() for p in module.parameters()) == 524_544


def test_positional_params_registered_once() -> None:
    """Layers reach posext through a plain attribute, so its parameters appear exactly once."""
    model = make_posext({"mode": "darope", "darope": {"eps": 1e-4}})
    names = [n for n, _ in model.named_parameters() if "posext" in n]
    assert names == ["model.posext.w", "model.posext.alpha"]
    assert all(layer.self_attn._posext is model.posext for layer in model.model.layers)


# ---- float32 handling ----------------------------------------------------------------


@pytest.mark.parametrize("name", list(ALL_CONFIGS))
def test_positional_tensors_stay_fp32_after_to_bf16(name: str) -> None:
    """model.to(bfloat16) casts the backbone but leaves positional tensors float32, values exact.

    Checking values matters: a cast to bf16 and back reports float32 but has lost precision.
    """
    model = make_posext(ALL_CONFIGS[name])
    with torch.no_grad():
        for p in model.posext.parameters():
            p.add_(torch.randn_like(p) * 0.1234567)  # non-representable in bf16
    before = {n: t.detach().clone() for n, t in model.posext.state_dict(keep_vars=True).items()}
    before |= {n: b.clone() for n, b in model.posext.named_buffers()}
    model = model.to(torch.bfloat16)
    after = dict(model.posext.named_parameters()) | dict(model.posext.named_buffers())
    for n, t in after.items():
        if t.is_floating_point():
            assert t.dtype == torch.float32, n
        assert torch.equal(t, before[n]), n
    assert model.model.norm.weight.dtype == torch.bfloat16


@pytest.mark.parametrize("name", ["none", "yarn", "learned_freq-head", "temp-adascale-tokens", "darope"])
def test_positional_tensors_stay_fp32_on_bf16_load(stock: Olmo2ForCausalLM, name: str) -> None:
    """from_pretrained(dtype=bfloat16) keeps positional params float32."""
    with tempfile.TemporaryDirectory() as tmp:
        stock.save_pretrained(tmp)
        cfg = Olmo2PosExtConfig(**Olmo2Config.from_pretrained(tmp).to_dict(), posext=ALL_CONFIGS[name])
        model = Olmo2PosExtForCausalLM.from_pretrained(tmp, config=cfg, dtype=torch.bfloat16)
    tensors = list(model.posext.parameters()) + list(model.posext.buffers())
    assert all(t.dtype == torch.float32 for t in tensors if t.is_floating_point())
    assert torch.equal(model.posext.inv_freq, rope_inv_freq(TINY["hidden_size"] // TINY["num_attention_heads"], TINY["rope_theta"]))
    assert model.model.norm.weight.dtype == torch.bfloat16


@pytest.mark.parametrize("pos", [{"mode": "learned_freq_temp"},
                                 {"mode": "learned_freq_temp", "temp": {"form": "constant"}},
                                 {"mode": "darope", "darope": {"eps": 1e-4}}])
def test_trained_params_survive_save_and_bf16_reload(stock: Olmo2ForCausalLM, pos: dict) -> None:
    """Every non-initial positional value round-trips exactly through save_pretrained + bf16 load."""
    model = make_posext(pos, stock.state_dict())
    with torch.no_grad():
        for p in model.posext.parameters():
            p.add_(torch.randn_like(p) * 0.1234567)
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        loaded = Olmo2PosExtForCausalLM.from_pretrained(tmp, dtype=torch.bfloat16)
    for n, p in model.posext.named_parameters():
        assert torch.equal(getattr(loaded.posext, n), p), n
    assert loaded.config.posext == model.config.posext


# ---- learning signal -----------------------------------------------------------------


@pytest.mark.parametrize("name", [n for n in IDENTITY_CONFIGS if n != "none"])
def test_gradients_reach_positional_params(stock: Olmo2ForCausalLM, name: str) -> None:
    """With the backbone frozen, the LM loss gives every positional parameter a gradient."""
    model = make_posext(IDENTITY_CONFIGS[name], stock.state_dict()).train()
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("model.posext."))
    ids = random_ids()
    model(ids, labels=ids).loss.backward()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, n
        else:
            assert p.grad is not None and p.grad.abs().sum() > 0, n


@pytest.mark.parametrize("source", ["tokens", "positions"])
def test_adascale_gamma_identifiability_depends_on_length_source(
    stock: Olmo2ForCausalLM, source: str
) -> None:
    """PoSE-like input: 16 real tokens labelled with positions up to 4L.

    With `tokens`, every query sees <= L_ref keys, so the growth term is the constant ln 2 and
    lambda = (ln 2)^gamma / tau. gamma still gets a gradient, but it is an exact multiple of
    tau's (grad_gamma = -grad_tau * tau * ln(ln 2)): gamma is only a second constant scale and
    learns nothing about length (the RQ2 caveat in CLAUDE.md). With `positions`, lengths exceed
    L_ref, lambda varies with position, and the two gradients are no longer proportional.
    """
    pos_cfg = {"mode": "learned_freq_temp", "temp": {"form": "adascale", "length_source": source}}
    model = make_posext(pos_cfg, stock.state_dict()).train()
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("model.posext."))
    ids = random_ids(1, 16)
    positions = torch.cat([torch.arange(8), torch.arange(8) + 3 * NATIVE_LEN])[None]
    model(ids, position_ids=positions, labels=ids).loss.backward()
    g_gamma, g_tau, tau = model.posext.gamma.grad, model.posext.tau.grad, model.posext.tau.detach()
    redundant = torch.allclose(g_gamma, -g_tau * tau * math.log(math.log(2.0)), rtol=1e-4, atol=1e-9)
    assert g_gamma.abs().sum() > 0
    assert redundant == (source == "tokens")


@pytest.mark.parametrize("param", ["a", "tau", "w"])
@torch.no_grad()
def test_changing_params_changes_output(stock: Olmo2ForCausalLM, param: str) -> None:
    """Sanity: the learned parameters actually affect the model."""
    pos = {"a": {"mode": "learned_freq", "granularity": "head"},
           "tau": {"mode": "learned_freq_temp"},
           "w": {"mode": "darope", "darope": {"eps": 1e-4}}}[param]
    model = make_posext(pos, stock.state_dict())
    ids = random_ids()
    before = model(ids).logits
    getattr(model.posext, param).add_(0.5)
    assert max_diff(model(ids).logits, before) > 1e-3


# ---- DaRoPE and AdaScale behaviour ---------------------------------------------------


@torch.no_grad()
def test_darope_coordinate_saturates_beyond_native_length() -> None:
    """At init, slow-band angles follow m + 0.5 inside L and stop growing at L for m >= L."""
    pos = PosConfig.from_dict({"mode": "darope", "darope": {"eps": 1e-4}})
    module = PositionalEncoding(pos, n_layers=1, n_heads=2, n_kv_heads=2, head_dim=16,
                                hidden_size=8, rope_base=10_000.0, native_len=NATIVE_LEN)
    position_ids = torch.tensor([[0, 10, NATIVE_LEN - 1, NATIVE_LEN, 4 * NATIVE_LEN]])
    out = module(0, position_ids, hidden_states=torch.randn(1, 5, 8))
    slow = module.slow_mask.nonzero().flatten()[0].item()
    theta = module.inv_freq[slow].item()
    coord = torch.atan2(out.sin[0, 0, :, slow], out.cos[0, 0, :, slow]) / theta  # angle / theta
    assert torch.allclose(coord[:3], position_ids[0, :3].float() + 0.5, atol=1e-2)
    # Saturated at L * (1 - eps). (The saturated angle is < pi, so atan2 does not wrap here; an
    # unsaturated m = 4L would wrap and read back as ~58, which is why a bound check is not enough.)
    assert torch.allclose(coord[3:], torch.full((2,), NATIVE_LEN * (1 - 1e-4)), atol=1e-3)
    # Fast bands keep ordinary positions.
    assert torch.allclose(out.cos[0, 0, :, 0], torch.cos(position_ids[0].float() * module.inv_freq[0]), atol=1e-5)


@torch.no_grad()
def test_adascale_flat_up_to_ref_len_then_grows() -> None:
    """lambda is constant for lengths <= L_ref and increases beyond it when gamma > 0."""
    pos = PosConfig.from_dict({"mode": "learned_freq_temp", "temp": {"length_source": "positions"}})
    module = PositionalEncoding(pos, n_layers=1, n_heads=1, n_kv_heads=1, head_dim=8,
                                hidden_size=8, rope_base=10_000.0, native_len=NATIVE_LEN)
    module.gamma.fill_(1.0)
    position_ids = torch.tensor([[0, NATIVE_LEN // 2, NATIVE_LEN - 1, 2 * NATIVE_LEN, 4 * NATIVE_LEN]])
    lam = module(0, position_ids).q_scale.flatten()
    assert torch.allclose(lam[:3], torch.full((3,), math.log(2.0)))
    assert lam[2] < lam[3] < lam[4]
