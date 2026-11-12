# PoSE position-ID skipping.
#
# Attribution: the two-chunk sampling scheme follows `train_preprocess_function_pose` in PoSE
# (github.com/dwzhu-pku/PoSE, src/train_pose.py), MIT License, Copyright (c) Dawei Zhu et al.
# Differences: (1) only position ids are produced; the text stays one contiguous chunk, as
# specified in CLAUDE.md (PoSE also takes the second chunk's text from later in the document);
# (2) a generalization to more than two chunks.
"""PoSE position ids: train on short sequences while exposing long relative distances.

A sequence of `seq_len` real tokens is split into `n_chunks` contiguous pieces. The first piece
keeps positions 0, 1, 2, ...; each later piece is shifted by a random skip, so positions keep
increasing and the last one stays below `target_len`. Distances up to `target_len - 1` then
appear between tokens of different pieces, although only `seq_len` tokens are present.
"""

from __future__ import annotations

import torch


def _randint(low: int, high: int, generator: torch.Generator | None) -> int:
    """Uniform integer in [low, high] (inclusive), like random.randint."""
    return int(torch.randint(low, high + 1, (1,), generator=generator).item())


def pose_position_ids(
    seq_len: int,
    target_len: int,
    n_chunks: int = 2,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample PoSE position ids for one sequence: shape (seq_len,), int64.

    Guarantees: first id is 0, ids strictly increase, last id < target_len, each piece is a
    contiguous run of ids.

    n_chunks = 2 reproduces PoSE: the first piece has a uniform length in [1, ceil(seq_len/2)],
    and the second piece is shifted by a uniform skip in [0, target_len - seq_len].
    For n_chunks > 2, the cut points are uniform distinct positions and the cumulative skips
    are sorted uniform draws from [0, target_len - seq_len].
    """
    if not 1 <= n_chunks <= seq_len:
        raise ValueError(f"n_chunks must be in [1, seq_len], got {n_chunks}")
    if target_len < seq_len:
        raise ValueError(f"target_len ({target_len}) must be >= seq_len ({seq_len})")
    budget = target_len - seq_len  # total skip available
    positions = torch.arange(seq_len, dtype=torch.long)
    if n_chunks == 1 or budget == 0:
        return positions

    if n_chunks == 2:
        cuts = [_randint(1, (seq_len + 1) // 2, generator)]
        offsets = [_randint(0, budget, generator)]
    else:
        cuts = (torch.randperm(seq_len - 1, generator=generator)[: n_chunks - 1] + 1).sort().values.tolist()
        offsets = torch.randint(0, budget + 1, (n_chunks - 1,), generator=generator).sort().values.tolist()

    # Piece i (i >= 1) starts at token index cuts[i-1] and is shifted by offsets[i-1].
    for cut, offset in zip(cuts, offsets, strict=True):
        positions[cut:] = torch.arange(cut, seq_len) + offset
    return positions


def pose_batch_position_ids(
    batch_size: int,
    seq_len: int,
    target_len: int,
    n_chunks: int = 2,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Independent PoSE position ids for every row: shape (batch_size, seq_len)."""
    return torch.stack(
        [pose_position_ids(seq_len, target_len, n_chunks, generator) for _ in range(batch_size)]
    )
