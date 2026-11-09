"""Tests for posext/models/build.py: loading, freezing, counting, saving/loading positional params."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import Olmo2Config, Olmo2ForCausalLM

from posext.models.build import (
    CONFIG_FILE,
    MODEL_FACTS,
    assert_model_facts,
    count_parameters,
    enable_gradient_checkpointing,
    freeze_backbone,
    load_model,
    load_positional,
    load_trained,
    save_positional,
)

from .conftest import TINY, random_ids

L, H, F, D = TINY["num_hidden_layers"], TINY["num_attention_heads"], 8, TINY["hidden_size"]  # F = head_dim/2

# Expected trainable counts at the tiny size, from the shapes in CLAUDE.md.
EXPECTED_TRAINABLE = {
    "none": ({}, 0),
    "pi": ({"mode": "pi"}, 0),
    "yarn": ({"mode": "yarn"}, 0),
    "learned_freq-shared": ({"mode": "learned_freq", "granularity": "shared"}, F),
    "learned_freq-layer": ({"mode": "learned_freq", "granularity": "layer"}, L * F),
    "learned_freq-head": ({"mode": "learned_freq", "granularity": "head"}, L * H * F),
    "temp-constant": ({"mode": "learned_freq_temp", "temp": {"form": "constant"}}, L * H * F + L * H),
    "temp-adascale": ({"mode": "learned_freq_temp"}, L * H * F + 2 * L * H),
    "darope": ({"mode": "darope", "darope": {"eps": 1e-4}}, L * H * (D + 1)),
}


@pytest.fixture(scope="session")
def checkpoint(stock: Olmo2ForCausalLM, tmp_path_factory: pytest.TempPathFactory) -> str:
    """The tiny stock model saved as a checkpoint directory (what load_model reads)."""
    path = tmp_path_factory.mktemp("stock")
    stock.save_pretrained(path)
    return str(path)


def load(checkpoint: str, pos: dict) -> torch.nn.Module:
    """load_model on the tiny checkpoint, float32 for exact comparisons."""
    return load_model(checkpoint, pos, dtype=torch.float32, attn_implementation="eager")


# ---- loading -------------------------------------------------------------------------


@torch.no_grad()
def test_load_model_is_identity(stock: Olmo2ForCausalLM, checkpoint: str) -> None:
    """A freshly loaded model equals stock OLMo-2 (positional params at identity)."""
    model = load(checkpoint, {"mode": "learned_freq_temp"})
    ids = random_ids()
    assert (model(ids).logits - stock(ids).logits).abs().max() < 1e-5


def test_load_model_keeps_positional_fp32_in_bf16(checkpoint: str) -> None:
    """Default bf16 backbone, float32 positional parameters."""
    model = load_model(checkpoint, {"mode": "darope", "darope": {"eps": 1e-4}})
    assert model.model.norm.weight.dtype == torch.bfloat16
    assert all(p.dtype == torch.float32 for p in model.posext.parameters())


def olmo_config(**overrides) -> Olmo2Config:
    """An OLMo-2-1B-shaped config (no weights)."""
    facts = {k: v for k, v in MODEL_FACTS.items() if k != "head_dim"}
    return Olmo2Config(**{**facts, **overrides})


def test_model_facts_pass_for_olmo2_config() -> None:
    """The real OLMo-2-1B dimensions satisfy the CLAUDE.md facts."""
    assert_model_facts(olmo_config())


@pytest.mark.parametrize("override", [{"num_hidden_layers": 12}, {"num_key_value_heads": 8},
                                      {"rope_theta": 10_000}, {"max_position_embeddings": 8192}])
def test_model_facts_fail_for_other_configs(override: dict) -> None:
    """Any deviation from the facts is reported."""
    with pytest.raises(ValueError, match=next(iter(override))):
        assert_model_facts(olmo_config(**override))


def test_load_model_checks_facts_on_request(checkpoint: str) -> None:
    """check_facts=True on the tiny checkpoint fails (it is not OLMo-2-1B)."""
    with pytest.raises(ValueError, match="model facts"):
        load_model(checkpoint, {}, check_facts=True)


# ---- freezing and counting -----------------------------------------------------------


@pytest.mark.parametrize("name", list(EXPECTED_TRAINABLE))
def test_freeze_backbone_leaves_only_positional_params(checkpoint: str, name: str) -> None:
    """After freezing, exactly the positional parameters require grad, with the expected count."""
    pos, expected = EXPECTED_TRAINABLE[name]
    model = load(checkpoint, pos)
    names = freeze_backbone(model)
    for n, p in model.named_parameters():
        assert p.requires_grad == n.startswith("model.posext."), n
    assert names == [n for n, p in model.named_parameters() if p.requires_grad]
    counts = count_parameters(model)
    assert counts["trainable"] == expected
    assert counts["total"] == sum(p.numel() for p in model.parameters())


def test_frozen_model_only_updates_positional_params(checkpoint: str) -> None:
    """An optimizer step after freezing changes positional params and nothing else."""
    model = load(checkpoint, {"mode": "learned_freq_temp"}).train()
    freeze_backbone(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.0)
    ids = random_ids()
    model(ids, labels=ids).loss.backward()
    optimizer.step()
    for n, p in model.named_parameters():
        assert torch.equal(p, before[n]) != n.startswith("model.posext."), n


def test_enable_gradient_checkpointing_gives_positional_gradients(checkpoint: str) -> None:
    """The helper uses non-reentrant checkpointing, so frozen-backbone training still has gradients."""
    model = load(checkpoint, {"mode": "learned_freq", "granularity": "layer"}).train()
    freeze_backbone(model)
    enable_gradient_checkpointing(model)
    assert model.model.gradient_checkpointing
    ids = random_ids()
    model(ids, labels=ids).loss.backward()
    assert (model.posext.a.grad.abs().sum(-1) > 0).all()


# ---- saving and loading positional parameters ----------------------------------------


def trained(checkpoint: str, pos: dict) -> torch.nn.Module:
    """A loaded model whose positional parameters were moved away from identity."""
    model = load(checkpoint, pos)
    with torch.no_grad():
        for p in model.posext.parameters():
            p.add_(torch.randn_like(p) * 0.1234567)
    return model


@pytest.mark.parametrize("name", ["none", "learned_freq-head", "temp-adascale", "darope"])
@torch.no_grad()
def test_save_and_load_trained_round_trip(checkpoint: str, tmp_path: Path, name: str) -> None:
    """save_positional + load_trained reproduces the trained model exactly (params and logits)."""
    model = trained(checkpoint, EXPECTED_TRAINABLE[name][0])
    save_positional(model, tmp_path, extra={"seed": 0})
    loaded = load_trained(tmp_path, dtype=torch.float32, attn_implementation="eager")
    for n, p in model.posext.named_parameters():
        assert torch.equal(getattr(loaded.posext, n), p), n
    ids = random_ids()
    assert torch.equal(loaded(ids).logits, model(ids).logits)
    assert json.loads((tmp_path / CONFIG_FILE).read_text())["seed"] == 0


def test_saved_run_is_small(checkpoint: str, tmp_path: Path) -> None:
    """Only positional parameters are saved, not the backbone."""
    model = trained(checkpoint, {"mode": "learned_freq", "granularity": "head"})
    save_positional(model, tmp_path)
    size = sum(f.stat().st_size for f in tmp_path.iterdir())
    assert size < 4 * L * H * F + 4096  # float32 values + small header and JSON


def test_load_rejects_mismatched_config(checkpoint: str, tmp_path: Path) -> None:
    """Loading a run into a model with a different positional config raises."""
    save_positional(trained(checkpoint, {"mode": "learned_freq", "granularity": "head"}), tmp_path)
    other = load(checkpoint, {"mode": "learned_freq", "granularity": "layer"})
    with pytest.raises(ValueError, match="granularity"):
        load_positional(other, tmp_path)


def test_load_rejects_missing_weights(checkpoint: str, tmp_path: Path) -> None:
    """A run whose weights file lacks a parameter cannot load silently as the identity model."""
    from safetensors.torch import load_file, save_file

    save_positional(trained(checkpoint, {"mode": "learned_freq_temp"}), tmp_path)
    weights = load_file(tmp_path / "positional.safetensors")
    del weights["gamma"]
    save_file(weights, tmp_path / "positional.safetensors")
    with pytest.raises(ValueError, match="missing=\\['gamma'\\]"):
        load_trained(tmp_path, dtype=torch.float32)


def test_load_rejects_wrong_shape(checkpoint: str, tmp_path: Path) -> None:
    """A weights file with the right names but wrong shapes raises."""
    from safetensors.torch import load_file, save_file

    save_positional(trained(checkpoint, {"mode": "learned_freq", "granularity": "head"}), tmp_path)
    weights = load_file(tmp_path / "positional.safetensors")
    weights["a"] = weights["a"][:, :1].contiguous()
    save_file(weights, tmp_path / "positional.safetensors")
    with pytest.raises(ValueError, match="Shape mismatch"):
        load_trained(tmp_path, dtype=torch.float32)
