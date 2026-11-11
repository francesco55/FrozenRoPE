"""Memory-efficient next-token cross-entropy for long sequences.

A full (B, S, V) logit tensor at S=16384 and V~100k is several GB. We run the backbone once,
then apply `lm_head` in token chunks so peak memory stays manageable under gradient
checkpointing on a single SCC GPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def chunked_lm_cross_entropy(
    lm_head: nn.Module,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = 2048,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Mean CE over non-ignored next-token positions, computing logits in chunks of `chunk_size`.

    Args:
        lm_head: language-model head mapping hidden -> vocab logits.
        hidden_states: (B, S, H) last-layer hidden states (same length as `labels`).
        labels: (B, S) token ids; typically the input ids (CausalLM shifts internally; we shift).
        chunk_size: number of token positions per lm_head call.
        ignore_index: label value to skip (padding).
    """
    # Standard CausalLM shift: predict token t+1 from hidden at t.
    shift_hidden = hidden_states[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    batch, seq, _ = shift_hidden.shape
    flat_h = shift_hidden.reshape(batch * seq, -1)
    flat_y = shift_labels.reshape(batch * seq)

    total_loss = flat_h.new_zeros(())
    total_n = 0
    for start in range(0, flat_h.shape[0], chunk_size):
        end = min(start + chunk_size, flat_h.shape[0])
        logits = lm_head(flat_h[start:end])  # (chunk, V)
        chunk_y = flat_y[start:end]
        # sum reduction so we can average over the whole sequence, not per chunk
        loss = F.cross_entropy(logits.float(), chunk_y, ignore_index=ignore_index, reduction="sum")
        n = int((chunk_y != ignore_index).sum().item())
        total_loss = total_loss + loss
        total_n += n

    if total_n == 0:
        return total_loss  # zero; avoids /0
    return total_loss / total_n
