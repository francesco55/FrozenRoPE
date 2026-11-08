# The positional parameters: a guide

A short orientation to what `posext` adds to OLMo-2, why, and where it lives in the code.
For the research questions, see the proposal; for engineering rules, see `CLAUDE.md`.

## Summary

We keep every original OLMo-2 weight frozen and learn only a few small sets of numbers that
control **how positions are encoded in attention**, so the model can read 16k tokens instead
of 4k.

## RoPE 

- Each attention head splits its 128-dim query and key vectors into **64 bands** (pairs of
  numbers). Band *j* of the token at position *m* is rotated by the angle *m · θ_j*.
- θ_j = 500000^(−2j/128): band 0 rotates fast (once every ~6 tokens), band 63 extremely slowly.
- Attention compares a query and a key with a dot product. Only the **difference** of their
  angles survives, so attention sees *relative* distance.
- **Fast bands** (0–31) complete many full turns within 4096 tokens: the model has seen every
  angle they can produce. **Slow bands** (32–63) never complete a turn within 4096. Beyond
  4k they produce angles the model has never seen, and that is what breaks long context.

Every method in this project changes either **how fast the bands rotate**, **what position
the slow bands use**, or **how sharp attention is**.

## What we changed in the model

What happens inside each attention layer, step by step:

| Step | Stock OLMo-2 | Ours (posext) |
|---|---|---|
| 1. Project and normalize | q, k, v from the hidden state; QK-norm on q and k | **Same** |
| 2. Rotation angles | Computed **once** for the whole model from `position_ids` and fixed θ, shared by all 16 layers | Computed **in each layer** by `model.model.posext`, from that layer's own parameters, `position_ids` and (DaRoPE only) the hidden state |
| 3. Rotate q and k | RoPE rotation | **Same** rotation, same place, with this layer's angles |
| 4. Temperature | None | `learned_freq_temp` only: multiply each head's queries by its temperature |
| 5. Attention | SDPA / FlashAttention kernel | **Same** kernel |
| Learned positional numbers | None (θ is a constant) | Depends on the mode: see the table below |

- All positional parameters live in **one module, `model.model.posext`** (`PositionalEncoding`).
  Each layer asks it for its own rotation, passing its layer index.
- Angles always come from the `position_ids` passed to the model, never from counting tokens
  (token index 0, 1, 2, …). Normally the two are identical. They differ under PoSE, which
  labels 4096 real tokens with gapped positions (e.g. tokens 2048–4095 get positions
  10,000–12,047) so the model practices long distances on short inputs.
- Everything else (weights, norms, MLPs, attention kernel) is the stock OLMo-2 code, copied
  from transformers 4.57.1.

## The modes (`pos.mode` in the config)

| mode | what it does | learned numbers on OLMo-2-1B |
|---|---|---|
| `none` | stock RoPE | 0 |
| `pi` | divide every position by a factor (e.g. 4) | 0 |
| `yarn` | slow bands slowed down, fast bands kept, plus a fixed sharpening | 0 |
| `learned_freq` | learn a speed for every band | 64 (shared) · 1,024 (per layer) · 16,384 (per head) |
| `learned_freq_temp` | as above, plus a learned per-head temperature | + 256 (constant) or 512 (AdaScale) |
| `darope` | slow bands use a learned, content-dependent position | 524,544 |

The first three are training-free baselines; the last three are what we train. For scale,
OLMo-2-1B has 1.485 billion frozen parameters.

## The learned parameters, one by one

**Frequency scales `a`** (`learned_freq`, `learned_freq_temp`). Band *j* rotates at
exp(*a_j*) · θ_j instead of θ_j. *a* = 0 means "unchanged"; *a* < 0 slows a band down (which
is what YaRN does by hand to slow bands). **Granularity** sets how many independent copies
exist: one set for the whole model (`shared`), one per layer, or one per head per layer.
RQ1 asks which granularity is needed.

**Temperature** (`learned_freq_temp`). A per-head number that multiplies the **query**,
making that head's attention sharper (>1) or flatter (<1). With 16k keys, softmax spreads
attention more thinly than with 4k ("dilution"), and a temperature can counter that.
- `constant`: one learned number per head.
- `adascale` (default, from AdaRoPE): the sharpening grows with how many keys the query can
  see, L_i = i + 1, and is flat up to 4096. Learned per head: a base temperature τ and a
  growth rate γ.
- `temp.length_source` decides whether L_i counts real tokens or position ids. This only
  matters for PoSE training (RQ2): with `tokens`, every training length is ≤ 4096, so γ
  can't learn how λ should grow with length (it only duplicates τ as a constant scale).
  Run RQ2 both ways. `CLAUDE.md` has the full explanation.

**DaRoPE coordinates** (`darope`, RQ3). On slow bands only, the position *m* is replaced
by a learned coordinate *c* = 4096 · sigmoid(**w** · x + α · logit((*m* + 0.5)/4096)),
where *x* is the token's hidden state. **w** (2048 numbers per head) lets the coordinate
depend on content, so related tokens can be placed "close" however far apart they are in
the text. α scales the ordinary position term. Positions ≥ 4096 saturate instead of
producing unseen angles. Fast bands keep normal RoPE.

## Why everything starts at "identity"

Every learned parameter is initialized so the model is **exactly** stock OLMo-2:
*a* = 0, τ = 1 and γ = 0 (or log τ = 0), **w** = 0 and α = 1. So any improvement after training
was learned rather than built in, which is the point of comparing against YaRN.

This is checked by comparing logits with the stock model (also with gapped position ids).
All modes match exactly, except DaRoPE, which differs by ~3e-7 on a tiny model and ~2e-4 on
the real one (logits ~25) from float rounding. `tests/test_identity.py` (Step 3) will make
this a permanent test.

## Practical rules

- **Positional parameters stay float32**, even when the model runs in bf16. Angles grow with
  position, so bf16's ~0.4% rounding becomes ~64 radians of error at position 16k, and small
  training updates would round away. This is automatic; don't fight it.
- **Training with gradient checkpointing needs `use_reentrant=False`.** Otherwise, with a
  frozen backbone, the positional parameters silently get no gradient.
- **Pass `position_ids` explicitly** when using PoSE, and keep the first position at 0.
  Without an attention mask and KV cache (i.e. in training), transformers would treat PoSE's
  position jumps as separate sequences that can't see each other; the model now passes an
  all-ones mask by default to prevent this.
- **Per-head modes assume no grouped-query attention.** True for OLMo-2-1B.

## Using it

Until `models/build.py` (Step 4) wraps this, loading looks like:

```python
import torch
from transformers import AutoConfig
from posext.models import Olmo2PosExtConfig, Olmo2PosExtForCausalLM

name = "allenai/OLMo-2-0425-1B"
cfg = Olmo2PosExtConfig(
    **AutoConfig.from_pretrained(name).to_dict(),
    posext={"mode": "learned_freq_temp", "granularity": "head", "temp": {"form": "adascale"}},
)
model = Olmo2PosExtForCausalLM.from_pretrained(name, config=cfg, dtype=torch.bfloat16)

# Freeze the backbone: only positional parameters train.
for n, p in model.named_parameters():
    p.requires_grad_(n.startswith("model.posext."))
print(model.posext)  # mode, granularity, slow bands
```

## Where to look in the code

| What | Where |
|---|---|
| Config options and defaults | `posext/models/positional.py`: `PosConfig`, `TempConfig`, `YarnConfig`, `DaRoPEConfig` |
| Base frequencies, slow-band mask, YaRN | `positional.py`: `rope_inv_freq`, `slow_band_mask`, `yarn_inv_freq` |
| All parameters and their initialization | `positional.py`: `PositionalEncoding.__init__`, `reset_parameters` |
| Per-layer rotation | `positional.py`: `PositionalEncoding.forward`, `layer_inv_freq` |
| DaRoPE coordinate | `positional.py`: `_darope_angles` |
| Temperature | `positional.py`: `_query_scale` |
| Where the rotation is applied | `posext/models/olmo2_posext.py`: `Olmo2PosExtAttention.forward` (search `posext:`) |
| Real-checkpoint sanity check | `scripts/check_env.py` |
