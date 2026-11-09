"""Build a ready-to-train posext model: load weights, set the positional mode, freeze the backbone.

Typical use:

    model = load_model("allenai/OLMo-2-0425-1B", pos={"mode": "learned_freq", "granularity": "head"})
    freeze_backbone(model)
    enable_gradient_checkpointing(model)
    log_parameter_counts(model)
    ...train...
    save_positional(model, "outputs/run1")            # kilobytes, not a full model copy

    model = load_trained("outputs/run1")               # base weights + trained positional params
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

from .olmo2_posext import Olmo2PosExtConfig, Olmo2PosExtForCausalLM
from .positional import PosConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "allenai/OLMo-2-0425-1B"

# CLAUDE.md, "Model facts". Checked at load time for the default checkpoint.
MODEL_FACTS: dict[str, Any] = {
    "num_hidden_layers": 16,
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "head_dim": 128,
    "rope_theta": 500_000,
    "max_position_embeddings": 4096,
}

POSITIONAL_PREFIX = "model.posext."
WEIGHTS_FILE = "positional.safetensors"
CONFIG_FILE = "positional_config.json"
FORMAT_VERSION = 1

# Keys of the stock config dict that must not be forwarded to Olmo2PosExtConfig.
_DROP_CONFIG_KEYS = ("architectures", "auto_map", "model_type", "transformers_version", "_name_or_path")


# ---- loading -------------------------------------------------------------------------


def assert_model_facts(config: Any, expected: dict[str, Any] = MODEL_FACTS) -> None:
    """Raise if the loaded config differs from the model facts in CLAUDE.md."""
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    actual = {k: head_dim if k == "head_dim" else getattr(config, k) for k in expected}
    wrong = {k: (actual[k], v) for k, v in expected.items() if actual[k] != v}
    if wrong:
        details = ", ".join(f"{k}: got {got}, expected {want}" for k, (got, want) in wrong.items())
        raise ValueError(f"Checkpoint does not match CLAUDE.md model facts: {details}")


def load_model(
    name_or_path: str = DEFAULT_MODEL,
    pos: dict[str, Any] | PosConfig | None = None,
    *,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
    device: str | torch.device | None = None,
    check_facts: bool | None = None,
) -> Olmo2PosExtForCausalLM:
    """Load a stock OLMo-2 checkpoint into the posext model with the given positional config.

    Positional parameters start at identity (the model equals stock OLMo-2) and stay float32
    whatever `dtype` the backbone uses. `check_facts=None` checks the CLAUDE.md model facts
    only for the default checkpoint.
    """
    pos_dict = pos.to_dict() if isinstance(pos, PosConfig) else dict(pos or {})
    stock = AutoConfig.from_pretrained(name_or_path)
    if check_facts or (check_facts is None and name_or_path == DEFAULT_MODEL):
        assert_model_facts(stock)
    base = {k: v for k, v in stock.to_dict().items() if k not in _DROP_CONFIG_KEYS}
    config = Olmo2PosExtConfig(**base, posext=pos_dict)
    model = Olmo2PosExtForCausalLM.from_pretrained(
        name_or_path, config=config, dtype=dtype, attn_implementation=attn_implementation
    )
    if device is not None:
        model = model.to(device)
    logger.info("Loaded %s with posext mode=%s", name_or_path, config.posext["mode"])
    return model


# ---- freezing and counting -----------------------------------------------------------


def is_positional(name: str) -> bool:
    """Whether a parameter name (from model.named_parameters()) is a positional parameter."""
    return name.startswith(POSITIONAL_PREFIX)


def freeze_backbone(model: torch.nn.Module) -> list[str]:
    """Only positional parameters train; returns their names."""
    trainable = []
    for name, param in model.named_parameters():
        param.requires_grad_(is_positional(name))
        if is_positional(name):
            trainable.append(name)
    return trainable


def count_parameters(model: torch.nn.Module) -> dict[str, Any]:
    """Trainable and total parameter counts, plus per-tensor trainable counts (store with results)."""
    params = list(model.named_parameters())
    trainable = {n: p.numel() for n, p in params if p.requires_grad}
    return {
        "trainable": sum(trainable.values()),
        "total": sum(p.numel() for _, p in params),
        "trainable_tensors": trainable,
    }


def log_parameter_counts(model: torch.nn.Module) -> dict[str, Any]:
    """Log the trainable-parameter count at startup (CLAUDE.md, "Freezing") and return it."""
    counts = count_parameters(model)
    logger.info(
        "Trainable parameters: %s of %s (%.6f%%) in %s",
        f"{counts['trainable']:,}",
        f"{counts['total']:,}",
        100 * counts["trainable"] / max(counts["total"], 1),
        list(counts["trainable_tensors"]) or "none",
    )
    return counts


def enable_gradient_checkpointing(model: Olmo2PosExtForCausalLM) -> None:
    """Checkpoint activations with use_reentrant=False.

    With a frozen backbone the layer inputs don't require grad, so the reentrant variant would
    silently give the positional parameters no gradient.
    """
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False


# ---- saving and loading trained positional parameters --------------------------------


def save_positional(model: Olmo2PosExtForCausalLM, out_dir: str | Path, extra: dict[str, Any] | None = None) -> Path:
    """Save only the positional parameters plus the config needed to rebuild the model."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensors = {n: p.detach().to("cpu", torch.float32).contiguous() for n, p in model.posext.named_parameters()}
    save_file(tensors, out / WEIGHTS_FILE)
    meta = {
        "format_version": FORMAT_VERSION,
        "base_model": model.config._name_or_path,
        "posext": model.config.posext,
        **(extra or {}),
    }
    (out / CONFIG_FILE).write_text(json.dumps(meta, indent=2))
    return out


def read_positional_config(run_dir: str | Path) -> dict[str, Any]:
    """Read the metadata written by save_positional."""
    meta = json.loads((Path(run_dir) / CONFIG_FILE).read_text())
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported positional format version: {meta.get('format_version')}")
    return meta


@torch.no_grad()
def load_positional(model: Olmo2PosExtForCausalLM, run_dir: str | Path) -> None:
    """Load trained positional parameters into `model`, strictly.

    Raises if the saved positional config differs from the model's, or if any parameter is
    missing, unexpected or has the wrong shape, so a broken run can never load silently as the
    untrained (identity) model.
    """
    meta = read_positional_config(run_dir)
    if meta["posext"] != model.config.posext:
        diff = {
            k: (meta["posext"].get(k), model.config.posext.get(k))
            for k in set(meta["posext"]) | set(model.config.posext)
            if meta["posext"].get(k) != model.config.posext.get(k)
        }
        raise ValueError(f"Saved posext config differs from the model's (saved, model): {diff}")
    saved = load_file(Path(run_dir) / WEIGHTS_FILE)
    params = dict(model.posext.named_parameters())
    missing, unexpected = sorted(set(params) - set(saved)), sorted(set(saved) - set(params))
    if missing or unexpected:
        raise ValueError(f"Positional weights mismatch: missing={missing}, unexpected={unexpected}")
    for name, param in params.items():
        if saved[name].shape != param.shape:
            raise ValueError(f"Shape mismatch for {name}: saved {tuple(saved[name].shape)}, model {tuple(param.shape)}")
        param.copy_(saved[name])  # in place: keeps float32 and the Parameter object


def load_trained(
    run_dir: str | Path,
    *,
    base_model: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
    device: str | torch.device | None = None,
) -> Olmo2PosExtForCausalLM:
    """Rebuild a trained model: base checkpoint + saved positional config and parameters."""
    meta = read_positional_config(run_dir)
    model = load_model(
        base_model or meta["base_model"],
        meta["posext"],
        dtype=dtype,
        attn_implementation=attn_implementation,
        device=device,
    )
    load_positional(model, run_dir)
    return model
