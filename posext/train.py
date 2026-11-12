"""Train only positional parameters on a frozen OLMo-2 (RQ1/RQ2 entry point).

    uv run python -m posext.train --config configs/rq1/learned_freq_head.yaml
    uv run python -m posext.train --config configs/rq1/learned_freq_head.yaml --no-wandb
    uv run python -m posext.train --config configs/smoke.yaml --device cpu  # local smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from posext.config import ExperimentConfig, load_config
from posext.data import Collator, load_pg19_chunks
from posext.loss import chunked_lm_cross_entropy
from posext.models import (
    enable_gradient_checkpointing,
    freeze_backbone,
    load_model,
    log_parameter_counts,
    save_positional,
)

logger = logging.getLogger("posext.train")

CHECKPOINT_NAME = "train_state.pt"
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def set_seed(seed: int) -> None:
    """Seed Python, numpy (if present), and torch."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str | None) -> torch.device:
    """Resolve --device; default CUDA > MPS > CPU."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loader(cfg: ExperimentConfig) -> DataLoader:
    """PG-19 chunks + collator for the configured regime."""
    data = cfg.data
    chunks = load_pg19_chunks(
        data.split,
        data.seq_len,
        max_sequences=data.max_sequences,
        max_chunks_per_book=data.max_chunks_per_book,
        seed=cfg.train.seed,
    )
    if chunks.numel() == 0:
        raise RuntimeError(f"No PG-19 chunks for {data.split} seq_len={data.seq_len}")
    collator = Collator(data.regime, data.target_len, n_chunks=data.n_chunks, seed=cfg.train.seed)
    return DataLoader(
        TensorDataset(chunks),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=data.num_workers,
        collate_fn=lambda rows: collator(torch.stack([r[0] for r in rows])),
        drop_last=False,
    )


def _param_groups(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split trainable params into frequency scales vs temperature (for per-group clipping)."""
    freq, temp = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Temperature tensors in positional.py: log_tau (constant), tau + gamma (adascale).
        leaf = name.rsplit(".", 1)[-1]
        if leaf in ("log_tau", "tau", "gamma"):
            temp.append(param)
        else:
            freq.append(param)
    return freq, temp


def build_optimizer(model: torch.nn.Module, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    """AdamW on positional params only, no weight decay by default."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(
            f"Mode {cfg.pos.get('mode')!r} has no trainable positional parameters; "
            "baselines (none/pi/yarn) should be evaluated, not trained."
        )
    return torch.optim.AdamW(
        params,
        lr=cfg.train.lr,
        betas=cfg.train.betas,
        weight_decay=cfg.train.weight_decay,
    )


def clip_gradients(model: torch.nn.Module, cfg: ExperimentConfig) -> float:
    """Clip grads; optional separate norms for frequency vs temperature groups."""
    freq, temp = _param_groups(model)
    total_norm = 0.0
    if cfg.train.clip_freq is not None and freq:
        total_norm = max(total_norm, float(torch.nn.utils.clip_grad_norm_(freq, cfg.train.clip_freq)))
    if cfg.train.clip_temp is not None and temp:
        total_norm = max(total_norm, float(torch.nn.utils.clip_grad_norm_(temp, cfg.train.clip_temp)))
    if cfg.train.clip_freq is None and cfg.train.clip_temp is None:
        params = [p for p in model.parameters() if p.requires_grad]
        total_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip))
    return total_norm


def forward_loss(model, batch: dict[str, torch.Tensor], chunk_size: int) -> torch.Tensor:
    """Backbone forward + chunked CE (avoids a full (B,S,V) logit tensor)."""
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model.model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        use_cache=False,
    )
    return chunked_lm_cross_entropy(
        model.lm_head,
        outputs.last_hidden_state,
        batch["labels"],
        chunk_size=chunk_size,
    )


def save_train_checkpoint(
    out_dir: Path,
    model,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: ExperimentConfig,
    counts: dict[str, Any],
) -> None:
    """Save positional weights + optimizer/step for resume."""
    save_positional(
        model,
        out_dir,
        extra={"experiment": cfg.to_dict(), "parameter_counts": counts, "step": step},
    )
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict(), "seed": cfg.train.seed},
        out_dir / CHECKPOINT_NAME,
    )


def maybe_init_wandb(cfg: ExperimentConfig, disabled: bool) -> Any:
    """Start a W&B run unless disabled via flag or config.

    Never blocks the job on a missing API key (common on SCC compute nodes).
    """
    if disabled or not cfg.wandb.enabled:
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; continuing without it")
        return None
    try:
        return wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name or cfg.name,
            config=cfg.to_dict(),
        )
    except Exception as exc:  # noqa: BLE001 — login/network failures must not kill training
        logger.warning("wandb.init failed (%s); continuing without wandb", exc)
        return None


def train(cfg: ExperimentConfig, *, device: torch.device, no_wandb: bool = False, resume: bool = False) -> Path:
    """Run positional-only training; returns the output directory."""
    set_seed(cfg.train.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(
        # Keep the resolved config next to the run for reproducibility.
        __import__("yaml").safe_dump(cfg.to_dict(), sort_keys=False)
    )

    dtype = DTYPES[cfg.dtype]
    model = load_model(
        cfg.model,
        pos=cfg.pos,
        dtype=dtype,
        attn_implementation=cfg.attn_implementation,
        device=device,
    )
    freeze_backbone(model)
    enable_gradient_checkpointing(model)
    model.train()
    counts = log_parameter_counts(model)
    optimizer = build_optimizer(model, cfg)

    start_step = 0
    if resume and (out_dir / CHECKPOINT_NAME).exists():
        state = torch.load(out_dir / CHECKPOINT_NAME, map_location="cpu")
        from posext.models import load_positional

        load_positional(model, out_dir)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        logger.info("Resumed from step %d", start_step)

    loader = build_loader(cfg)
    data_iter = iter(loader)
    wb = maybe_init_wandb(cfg, no_wandb)

    logger.info(
        "Training %s for %d steps on %s (regime=%s, seq_len=%d, target_len=%d)",
        cfg.name,
        cfg.train.steps,
        device,
        cfg.data.regime,
        cfg.data.seq_len,
        cfg.data.target_len,
    )

    t0 = time.time()
    running = 0.0
    for step in range(start_step + 1, cfg.train.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        optimizer.zero_grad(set_to_none=True)
        loss = forward_loss(model, batch, cfg.train.logit_chunk_size)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        grad_norm = clip_gradients(model, cfg)
        optimizer.step()

        running += float(loss.item())
        if step % cfg.train.log_every == 0 or step == 1:
            avg = running / min(step - start_step, cfg.train.log_every)
            running = 0.0
            ppl = math.exp(min(avg, 20.0))
            logger.info(
                "step %d/%d  loss=%.4f  ppl?%.2f  grad_norm=%.3f  %.1fs",
                step,
                cfg.train.steps,
                avg,
                ppl,
                grad_norm,
                time.time() - t0,
            )
            if wb is not None:
                wb.log({"train/loss": avg, "train/ppl": ppl, "train/grad_norm": grad_norm}, step=step)

        if step % cfg.train.save_every == 0 or step == cfg.train.steps:
            save_train_checkpoint(out_dir, model, optimizer, step, cfg, counts)
            logger.info("Saved checkpoint at step %d -> %s", step, out_dir)

    results = {
        "name": cfg.name,
        "seed": cfg.train.seed,
        "steps": cfg.train.steps,
        "parameter_counts": counts,
        "output_dir": str(out_dir),
        "seconds": time.time() - t0,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(results, indent=2))
    if wb is not None:
        wb.summary.update(results)
        wb.finish()
    return out_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI for `python -m posext.train`."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="Experiment YAML")
    p.add_argument("--device", type=str, default=None, help="cuda | mps | cpu (default: auto)")
    p.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases")
    p.add_argument("--resume", action="store_true", help="Resume from output_dir/train_state.pt")
    p.add_argument("--output-dir", type=Path, default=None, help="Override config output_dir")
    p.add_argument("--steps", type=int, default=None, help="Override train.steps (smoke tests)")
    p.add_argument("--max-sequences", type=int, default=None, help="Override data.max_sequences")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load config, apply CLI overrides, train."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.output_dir is not None:
        cfg.output_dir = str(args.output_dir)
    if args.steps is not None:
        cfg.train.steps = args.steps
    if args.max_sequences is not None:
        cfg.data.max_sequences = args.max_sequences
    device = pick_device(args.device)
    train(cfg, device=device, no_wandb=args.no_wandb, resume=args.resume)


if __name__ == "__main__":
    main()
