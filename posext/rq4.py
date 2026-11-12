"""RQ4 stretch: unfreeze weights in stages to locate the remaining gap.

TODO (not part of the RQ1 deliverable):
1. LoRA adapters on W_Q / W_K via peft (`uv sync --extra lora`).
2. Full fine-tuning of the backbone.
3. For each stage, report the fraction of the gap between tuned YaRN and full FT
   closed on PG-19 PPL, passkey, and RULER.

This module is a stub so imports and docs can point at a stable path.
"""

from __future__ import annotations


def run_rq4(*_args, **_kwargs):  # noqa: ANN001, ANN002
    """Placeholder -- implement after RQ1-RQ3."""
    raise NotImplementedError(
        "RQ4 is a stretch goal. See CLAUDE.md / the proposal: LoRA on W_Q/W_K, then full FT."
    )
