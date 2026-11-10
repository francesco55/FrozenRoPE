"""Batches for the two training regimes (CLAUDE.md, "Training").

- `long`: chunks of the target length, position ids 0..L-1.
- `pose`: short chunks (e.g. 4096) with PoSE position ids up to the target length.

Both regimes pass `labels` (the model shifts them) and an all-ones `attention_mask`. The mask
matters: without it and without a KV cache, transformers treats jumps in position ids as
boundaries between packed sequences and masks PoSE chunks off from each other. Our model
defaults to an all-ones mask too; passing it here also protects code outside our model class.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .pose import pose_batch_position_ids

REGIMES = ("long", "pose")


class Collator:
    """DataLoader collate_fn producing model-ready batches.

    PoSE positions are resampled for every example from this collator's own seeded generator,
    so a run is reproducible from its seed. Use num_workers=0 (or one Collator per worker with
    distinct seeds), otherwise workers would share the generator state and repeat samples.
    """

    def __init__(self, regime: str, target_len: int, n_chunks: int = 2, seed: int = 0) -> None:
        """Configure the regime; `target_len` is L' (e.g. 16384)."""
        if regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
        self.regime, self.target_len, self.n_chunks = regime, target_len, n_chunks
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, rows: Sequence[torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor]:
        """Stack token rows (each (seq_len,)) into a batch dict."""
        input_ids = torch.stack(list(rows)).long() if not isinstance(rows, torch.Tensor) else rows.long()
        batch_size, seq_len = input_ids.shape
        if self.regime == "long":
            if seq_len > self.target_len:
                raise ValueError(f"long regime: seq_len {seq_len} exceeds target_len {self.target_len}")
            position_ids = torch.arange(seq_len).expand(batch_size, -1).clone()
        else:
            position_ids = pose_batch_position_ids(batch_size, seq_len, self.target_len, self.n_chunks, self.generator)
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
            "position_ids": position_ids,
        }
