# Project status and team guide

*As of 2026-11-09, branch `chore/skeleton-setup`. `CLAUDE.md` holds the live status checklist.*

## Completed

| Step | What |
|---|---|
| 0–1 | uv environment (Python 3.11, `transformers==4.57.1`, `torch==2.9.1`), pin verified on the real checkpoint |
| 2 | Patched OLMo-2 with all 6 positional modes |
| 3 | Test suite: identity, module and behaviour tests (204 CPU tests, ~1.5 s, plus opt-in real-checkpoint and real-data tests) |
| 4 | `build.py`: load, freeze, count, strict save/load of positional parameters |
| 5 | `posext/data/`: PG-19 chunks, PoSE position ids, batch builder |

Bugs found and fixed along the way:
- PoSE chunks masked off from each other in training
- bf16 casts silently losing positional precision
- `none`/`pi`/`yarn` crashing at construction
- the PG-19 loader downloading all 18 GB

## Remaining before RQ1 can run

RQ1 compares `learned_freq` (shared, layer, head) with and without temperature against `none`,
`pi`, default YaRN and tuned YaRN.

| Needed | Step |
|---|---|
| `train.py` + YAML configs (memory-efficient loss for 16k, per-group clipping, resume) | 6 |
| Full-model export for RULER and `lm-eval` | 6 |
| Perplexity at 4k/8k/16k and passkey retrieval | 7 |
| RULER wrapper (16k RULER is part of the success criterion) | 7 |
| HellaSwag/ARC-Easy/PIQA via `lm-eval` (the "lose at most 1 point" criterion) | 7 |
| YaRN grid search on validation (the tuned-YaRN baseline) | 8 |
| SCC job templates, uv setup on SCC, FlashAttention-2 + PoSE GPU check | 9 |
| Then the actual runs: tune learning rate and steps on validation, train 5 learned configs, evaluate all 9 configs | — |

## Remaining repo setup

- Push the branch and open a PR into `main`
- README: setup, running tests, training, evaluation
- SCC: qsub templates, `HF_HOME` and `POSEXT_DATA_CACHE` in `/projectnb`, W&B team project, offline-mode fallback
- RQ4 stub (LoRA and full fine-tuning) with a TODO
- Not yet verified against BU's documentation: SCC flags, and whether compute nodes have internet access

## How to work with the repo

**Setup:**
```bash
uv sync --all-extras          # installs Python 3.11 + pinned dependencies
uv run pytest                 # 204 tests, ~1.5 s, CPU
```

**Where things live:**

| What | Where |
|---|---|
| Engineering rules, status, gotchas | `CLAUDE.md` |
| Plain-language guide to the positional parameters | `Documentation/positional_parameters.md` |
| The model (stock OLMo-2 copy; search `posext:` for our changes) | `posext/models/olmo2_posext.py` |
| All positional logic and parameters | `posext/models/positional.py` → `model.model.posext` |
| Load, freeze, save, reload | `posext/models/build.py` |
| PG-19, PoSE, batches | `posext/data/` |

**OLMo-2 weights:** downloaded once (~6 GB) to the Hugging Face cache on first use
(`~/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B`) and reused by every run. Only the
positional parameters are trained and saved; a trained run is kilobytes. On SCC, set
`HF_HOME=/projectnb/<project>/...` (home quota is ~10 GB).

**Choosing what to train** is one `pos` dict:

| `pos` | Trains | Count (OLMo-2-1B) |
|---|---|---|
| `{"mode": "learned_freq", "granularity": "shared" / "layer" / "head"}` | frequency scales `a` | 64 / 1,024 / 16,384 |
| `{"mode": "learned_freq_temp", "granularity": ..., "temp": {"form": "adascale" / "constant"}}` | `a` + temperature | +512 / +256 |
| `{"mode": "darope"}` | DaRoPE `w`, α | 524,544 |
| `{"mode": "none" / "pi" / "yarn"}` | nothing (baselines) | 0 |

```python
from posext.models import load_model, freeze_backbone, enable_gradient_checkpointing, save_positional, load_trained

model = load_model(pos={"mode": "learned_freq", "granularity": "head"})  # equals stock OLMo-2 at start
freeze_backbone(model)
enable_gradient_checkpointing(model)
# ... train (train.py, Step 6) ...
save_positional(model, "outputs/run1")  # kilobytes; load_trained("outputs/run1") restores it
```

**Adding a new positional mode:**
- **`positional.py`:** add the mode name, its parameters in `__init__`, identity values in
  `reset_parameters`, and its math in `layer_inv_freq` / `forward`.
- **Tests:** add it to `IDENTITY_CONFIGS` in `tests/conftest.py`, so the identity tests cover it
  automatically.

**Rules that bite if ignored** (details in `CLAUDE.md`):
- Positional parameters stay float32; this is automatic.
- Use `enable_gradient_checkpointing`; reentrant checkpointing silently gives no gradients.
- With PoSE, pass `position_ids` explicitly and keep the first position at 0.
- PoSE + `temp.length_source: tokens` can't learn length dependence for γ; run RQ2 both ways.
