# CLAUDE.md — Learned RoPE parameters for context extension (CS599 B1 project)

Read this file and `Documentation/Learning RoPE Parameters for Context Extension.pdf` (the proposal)
before writing code. Reference papers are in `Documentation/papers/`.
The proposal states the research questions; this file states the engineering decisions.

## Project in one paragraph

We extend a **frozen** pretrained LM (OLMo-2-0425-1B, native context L = 4096) to
L' = 16384 by training **only positional parameters** inside RoPE, and compare against
training-free baselines (Position Interpolation, YaRN default and grid-tuned).
RQ1: learned frequency scales (+ per-head temperature) at three granularities.
RQ2: same parameters learned from 4k sequences only, using PoSE position skipping.
RQ3: DaRoPE-style learned slow-band coordinates retrofitted onto the frozen model.
RQ4 (stretch): add LoRA on W_Q/W_K, then full fine-tuning, to locate the remaining gap.

Team of 3. Compute: Boston University SCC (SGE scheduler, A100/L40S-class GPUs).

## Current task: build the skeleton

Build a working, tested skeleton. Do **not** run large experiments yet. In order:

1. Package layout and `requirements.txt` with **pinned** versions (especially `transformers`;
   see "Proposed choices" for the suggested version).
2. Patched OLMo-2 model file supporting every positional mode below.
3. `tests/test_identity.py` — must pass before anything else is merged (see Testing).
4. Utilities to freeze the backbone and count trainable parameters.
5. PG-19 data pipeline (long chunks) and PoSE position-ID generator.
6. Training script for positional-only training, driven by YAML configs.
7. Evaluation: perplexity at 4k/8k/16k and synthetic passkey retrieval. Stubs/wrappers for RULER and `lm-evaluation-harness`.
8. YaRN grid-search script (evaluation only, no training).
9. SCC job-script templates and a README with setup and run instructions.

Leave RQ4 (LoRA, full fine-tuning) as a stub with a TODO.

## Status and implementation notes

Branch `chore/skeleton-setup`. Update this list as steps land.

- [x] 1. uv env (Python 3.11): `pyproject.toml`, `uv.lock`, `requirements.txt` exported via
  `uv export`. Pin verified on the real checkpoint: `uv run python scripts/check_env.py`.
- [x] 2. `posext/models/olmo2_posext.py` + `positional.py`, all six modes.
- [x] 3. Tests: `uv run pytest` (204 tests incl. steps 4–5, CPU, ~1.5 s).
  - `tests/test_identity.py`: identity for every mode (default/contiguous/gapped positions,
    lengths 1..L, beyond L, SDPA, KV-cache generation, from_pretrained).
  - `tests/test_positional.py`: slow-band mask, yarn/pi vs stock, config guards, shapes at
    OLMo-2 size, fp32 values, save/reload, gradients, DaRoPE saturation, AdaScale.
  - `tests/test_behavior.py`: trained-like (non-identity) parameters: cached decoding = full
    forward beyond L, exact equivalences (shared a = -ln 4 == PI, constant temperature ==
    doubled scaling, exact AdaScale values), per-layer/per-head isolation and gradients,
    non-reentrant checkpointing, bf16 rotation precision, and the PoSE masking regression.
  - Real checkpoint, opt-in: `POSEXT_REAL_CKPT=1 uv run pytest -m real` (~1 min CPU).
  - Still to add: FA2 + PoSE test on an SCC GPU.
  - Review of the suite and its follow-ups: `Documentation/test_review.md`.
- [x] 4. `posext/models/build.py`: `load_model` (checks model facts for OLMo-2-1B),
  `freeze_backbone`, `count_parameters` / `log_parameter_counts`,
  `enable_gradient_checkpointing` (non-reentrant), `save_positional` / `load_positional` /
  `load_trained`. A run is saved as `positional.safetensors` + `positional_config.json`
  (base model, posext config, extras), kilobytes not GBs; loading is strict (config, keys
  and shapes must match). Tests: `tests/test_build.py`; real-checkpoint counts 64 / 16,896 /
  524,544 in `test_real_checkpoint.py`.
- [x] 5. `posext/data/`: `pg19.py` (`load_pg19_chunks(split, seq_len, max_sequences=...,
  max_chunks_per_book=..., seed=...)`: shuffled books, non-overlapping chunks within one book,
  tail dropped, cached), `pose.py` (`pose_position_ids`), `collate.py` (`Collator` for the
  `long` / `pose` regimes; always passes labels and an all-ones attention mask). Tests:
  `tests/test_data.py` (offline fakes) + opt-in real data: `POSEXT_NETWORK=1 uv run pytest -m network`.
- [x] 6. `posext/train.py` + `posext/config.py` + `posext/loss.py` (chunked CE) +
  `posext/export.py` + `configs/rq1/*.yaml` + `configs/smoke.yaml`. Resume via
  `train_state.pt`; per-group grad clip for freq vs temperature.
- [x] 7. `posext/eval/ppl.py`, `passkey.py`, `shortctx.py` (lm-eval); `ruler.py` is a
  stub that exports the model and documents the NVIDIA RULER checkout.
- [x] 8. `posext/baselines/yarn_grid.py` (val PPL @16k + val passkey → combined z-score).
- [x] 9. `scripts/scc/{train,eval,yarn_grid}.qsub` (placeholders) + README run instructions.
  RQ4 stub: `posext/rq4.py`.

Notes for the remaining steps:
- **All positional parameters live in one module, `model.model.posext`** (`PositionalEncoding`).
  Freeze everything else; save/load only `posext.*`. Attention layers reach it through a
  plain attribute (`_posext`), not a submodule, so it is registered once.
- **Positional params stay float32, values exact,** even when the model is bf16:
  `PositionalEncoding._apply` keeps the original fp32 tensors and restores them after `.to()`
  (casting to bf16 and back would report float32 but lose precision; tests check values),
  and `post_init` sets `_keep_in_fp32_modules_strict = ["posext"]` for
  `from_pretrained(dtype=bf16)` (the non-strict list only covers fp16). It is set per
  instance and only when posext has parameters, because HF rejects listed modules that own
  none (modes `none`/`pi`/`yarn`).
- **Gradient checkpointing must use `use_reentrant=False`.** With a frozen backbone the layer
  inputs don't require grad, so reentrant checkpointing gives `posext` no gradients, silently.
- **Identity tolerances (fp32):** all modes exact except DaRoPE, ~3e-7 on the tiny model and
  ~2e-4 on the real model (logits ~25), from the 0.5 position offset. `darope.eps` must be
  < 0.5/L (enforced in code).
- **Stock YaRN cross-check:** `Olmo2Config` rejects `rope_scaling` with keys beyond
  `type`/`factor`; set `config.rope_scaling = {...}` after construction. Our `yarn` and `pi`
  modes match stock YaRN / linear scaling exactly.
- **Per-head modes assume no GQA** (true for OLMo-2-1B; code raises otherwise).
- **Gapped `position_ids` and packed-sequence detection (fixed in the model).** In
  transformers 4.57.1, when there is no `attention_mask` and no KV cache (e.g. training, where
  gradient checkpointing turns the cache off), mask creation treats every jump in
  `position_ids` as the start of a new packed sequence, for every attention backend. PoSE
  chunks would then be masked off from each other. `Olmo2PosExtModel.forward` therefore
  defaults to an all-ones `attention_mask` in that case (we never pack sequences). Regression
  tests in `tests/test_behavior.py` fail without this. Stock OLMo-2 has the same behaviour, so
  compare against stock with an explicit all-ones mask when the cache is off.
- **FlashAttention-2** has a separate packed-sequence path keyed on `position_ids` (new
  sequence at every position 0, batch size 1). PoSE ids start at 0 exactly once, so it should
  be fine, and with the all-ones mask FA2 may take its masked path instead. **Unverified:**
  needs a GPU test on SCC with PoSE positions.
- **PG-19 source: the `emozilla/pg19` Parquet mirror** (`deepmind/pg19` needs a loading
  script that `datasets` >= 4 refuses to run). Validation (50) and test (100) are complete and
  checked on load; train has 13,684 of 28,602 books (plenty for positional-only training).
  `load_pg19_texts` passes `data_files` for the one split it needs: a plain `load_dataset`
  downloads and converts every split (~18 GB on disk even for the 20 MB test split).
- **Data cache:** chunks are cached as `.pt` files under `$POSEXT_DATA_CACHE` (default
  `./data_cache`, gitignored), keyed by split, length, counts, seed, tokenizer and dataset.
  On SCC, set `POSEXT_DATA_CACHE` and `HF_HOME` to `/projectnb/<project>/...` (home quota).
- **PoSE vs the original:** with 2 chunks we match PoSE's sampling (first piece length uniform
  in [1, ceil(n/2)], skip uniform in [0, L' - n]). PoSE also takes the second piece's *text*
  from later in the document; per this file we keep the text contiguous and only skip
  positions. Revisit if RQ2 results look off.
- **Collator randomness:** PoSE positions come from the collator's own seeded generator; use
  `num_workers=0` (or one collator per worker with distinct seeds).
- The copied model file keeps upstream style (ruff per-file ignores in `pyproject.toml`) so it
  diffs cleanly against transformers 4.57.1's `modeling_olmo2.py`.

## Repository layout

```
posext/
  models/olmo2_posext.py      # copy of transformers' modeling_olmo2.py + our changes
  models/positional.py        # learned-frequency, temperature, DaRoPE modules
  models/build.py             # load pretrained weights, set mode, freeze backbone
  data/pg19.py                # tokenize + chunk PG-19 into fixed-length sequences
  data/pose.py                # PoSE position-ID skipping
  train.py                    # trains only positional params; config-driven
  eval/ppl.py  eval/passkey.py  eval/ruler.py  eval/shortctx.py
  baselines/yarn_grid.py      # grid search over YaRN settings on a validation split
configs/                      # one YAML per experiment
scripts/scc/                  # qsub templates
tests/
Documentation/                # proposal PDF; papers/ holds reference papers
```

## Model facts (verify against the loaded config at runtime; assert them in code)

- `allenai/OLMo-2-0425-1B`: 16 layers, hidden size 2048, 16 query heads and 16 KV heads
  (full multi-head attention, no GQA), head dim 128 (64 rotary bands), RoPE base 500,000,
  max position 4096.
- Band j has θ_j = base^(-2j/head_dim). **Slow bands** are those with wavelength
  2π/θ_j > L. With these settings that is bands 32–63 (half). Compute the mask in code;
  don't hard-code it.
- OLMo-2 applies **QK-norm** (RMSNorm on query and key projections) **before** RoPE.
  Keep that order: our rotation goes after the norm, exactly where stock RoPE is applied.
- OLMo-2 uses **reordered norm** (normalization on the outputs of attention/MLP), so the
  hidden state entering attention is the un-normalized residual stream. Relevant to DaRoPE.

## Key architectural change

In stock Hugging Face OLMo-2, one rotary module computes `cos`/`sin` once per forward
and every layer reuses them. Per-layer and per-head frequencies break this, so each
attention layer must compute its own rotation from **its own parameters and the
`position_ids`**. Implement this inside the attention forward of the copied model file.
Keep using the stock attention backend (SDPA / FlashAttention) after rotation.

Always derive angles from `position_ids` passed into the model (needed for PoSE), never
from `arange(seq_len)`.

## Positional modes (one config field: `pos.mode`)

| mode | what changes | trainable params |
|---|---|---|
| `none` | stock RoPE | 0 |
| `pi` | positions divided by factor s | 0 |
| `yarn` | YaRN frequency ramp + attention factor (settings from config) | 0 |
| `learned_freq` | θ̂ = exp(a)·θ, a init 0; `granularity` ∈ {shared, layer, head} | a |
| `learned_freq_temp` | as above + per-head temperature (`temp.form`: `constant` or `adascale`) | a + temperature params |
| `darope` | slow bands use a learned coordinate (below); fast bands stock | w_h, α_h |

Parameter shapes: shared `a ∈ R^{64}`; layer `R^{16×64}`; head `R^{16×16×64}`.
Temperature: applied by scaling each head's **queries only** before attention (keys,
values and the attention kernel unchanged). Two forms, selected by `temp.form`:
- `constant`: one `log_tau ∈ R^{16}` per layer, init 0; scale = exp(log_tau_h).
- `adascale` (proposed default, see "Proposed choices"): AdaRoPE's length-dependent form,
  scale λ_h(L) = (1/T_h) · [ln(1 + max(L, L_ref)/L_ref)]^{γ_h}, with learned T_h and γ_h per
  head and `L_ref` a config hyperparameter (`temp.ref_len`, default L = 4096).
  **L is per query, not the whole sequence length** (as in AdaRoPE): the query at index i
  attends causally to i + 1 keys, so L_i = i + 1. A whole-sequence L would make a token's
  output depend on text after it (non-causal) and break KV-cache generation.
  Because of the `max(L, L_ref)`, λ is constant up to L_ref and only varies beyond it.
  Initialize T_h = 1 and γ_h = 0, which gives λ_h = 1 at every length, so the identity
  test still holds. This matches AdaRoPE's initialization.

  **`temp.length_source`: `tokens` | `positions`** (default `tokens`) chooses what i is:
  the token's index in the input (= the number of keys it really attends to) or its
  position id. The two are identical everywhere except PoSE training, so RQ1 training
  and all evaluation are unaffected. Under PoSE they diverge: with 4096 tokens split as
  indices 0–2047 → positions 0–2047 and indices 2048–4095 → positions 10000–12047, the token
  at index 3000 has L = 3001 under `tokens` (below L_ref, λ flat) but L ≈ 10951 under
  `positions` (λ grows as if at ~11k, although only 3001 keys are present).
  Consequence for RQ2: with `tokens`, L never exceeds 4096 during PoSE training, so the
  growth term is the constant ln 2 and λ_h = (ln 2)^{γ_h} / T_h. γ_h still gets a gradient,
  but only as a **second constant scale, redundant with T_h** (its gradient is an exact
  multiple of T_h's; tested in `test_positional.py`). It learns nothing about length, yet
  whatever value it drifts to sets how λ grows at real 16k lengths during evaluation. The
  drift has a fixed direction: because ln(ln 2) < 0, wanting sharper attention in training
  pushes γ_h negative, which then makes attention at 16k *flatter* than at 4k
  (λ(16k)/λ(4k) = (ln 5 / ln 2)^{γ_h} < 1), the opposite of what dilution needs. That makes "temperatures cannot
  be learned from short sequences" largely true by construction, not an empirical finding.
  With `positions`, γ_h does get a length-dependent gradient, but from a mismatched
  signal (16k-like positions, 4k-key softmax); whether it transfers to real 16k inputs is
  the actual test. **Run RQ2 with both settings and report the difference**: `tokens` as
  the by-construction baseline, `positions` as the real test of the hypothesis.
  (Minor: in batched generation with left padding, the token index counts pad tokens;
  `positions` is correct there. Evaluation should use batch size 1 or `positions`.)

Optional field `pos.init_from_yarn: bool` (default false) initializes `a` from YaRN's
frequencies instead of 0, to compare against AdaRoPE's initialization.

### DaRoPE coordinate

For each layer and head, on slow bands only, replace position m with
`c_h(x_m) = L · sigmoid(w_hᵀ x̃_m + α_h · β_m)`, `β_m = logit((m + 0.5) / L)`,
where `(m + 0.5) / L` is clipped to `[ε, 1 − ε]` so positions ≥ L saturate.
Angle on slow band j becomes `c_h(x_m) · θ_j`; queries and keys use the coordinate of
their own token. Init `w_h = 0`, `α_h = 1` so that inside L the coordinate equals m + 0.5
(the constant offset cancels in relative angles). `x̃_m` is the attention input hidden
state; because OLMo-2's residual stream is un-normalized, apply a parameter-free RMS
normalization to x before the projection (make this a config flag, default on).
Parameters per head: 2048 + 1, ≈ 525k total.

## Freezing

`requires_grad=False` on every original parameter; only positional parameters train.
Gradients still flow through all layers, so activation memory ≈ full fine-tuning:
use bf16, gradient checkpointing, and FlashAttention/SDPA. Log the trainable-parameter
count at startup and store it with results.

## Data

- **PG-19** (Hugging Face `pg19` or equivalent mirror): train split for training,
  validation split for YaRN grid search and hyperparameter choices, test split only for
  final numbers. Tokenize with the OLMo-2 tokenizer; chunk books into non-overlapping
  sequences of length 4096 / 8192 / 16384.
- **PoSE** (`data/pose.py`): given a 4096-token chunk and target length L', split into
  `n_chunks` (default 2) contiguous pieces with random lengths; first piece starts at
  position 0; each later piece gets a random skip so positions are increasing and the
  last position < L'. Resample per example. Return `position_ids` only — text unchanged.

## Training (`train.py`)

- Objective: next-token cross-entropy on long text (or PoSE-positioned short text).
  Retrieval tasks are **never** used for training.
- Config fields: model, `pos.*`, data regime (`long` | `pose`), target length,
  number of sequences, optimizer (AdamW on positional params only, no weight decay,
  separate gradient clipping), lr, steps, seed, output dir.
- Positional parameters can be finicky: keep them in float32 even if the model runs in bf16.
- Save only the positional parameters plus the config, not the full model.
- Also support exporting a full model directory with custom code (`auto_map` +
  `trust_remote_code`) so RULER and `lm-evaluation-harness` can load it by path.

## Evaluation

- `eval/ppl.py`: PG-19 test perplexity at 4k, 8k, 16k.
- `eval/passkey.py`: synthetic passkey retrieval (repeated filler text with a random
  5-digit key at controlled depths), accuracy over lengths 4k–16k and depths 0–100%.
  Generate validation and test prompts with different seeds.
- `eval/ruler.py`: thin wrapper around NVIDIA's RULER repo for a subset of tasks
  (needle-in-a-haystack variants, variable tracking) at 4k/8k/16k.
- `eval/shortctx.py`: HellaSwag, ARC-Easy, PIQA via `lm-evaluation-harness`.
- RQ3 extra probe: key–value recall with many competing pairs (stub for now).
- All results are written as JSON keyed by config name and seed.

## Baselines

- Untouched model (`none`), `pi`, `yarn` default, `yarn` grid-tuned.
- `baselines/yarn_grid.py`: grid over `factor` {4, 6, 8}, `beta_fast` {8, 16, 32, 64},
  `beta_slow` {0.5, 1, 2}, `attention_factor` {1.0, 1.1, 1.2, 1.3}; score on the PG-19
  **validation** split + validation passkey prompts with one fixed combined score.
- DaRoPE at initialization (no training) is itself a baseline for RQ3.

## Testing (required)

- `tests/test_identity.py`: for modes `learned_freq` (all granularities, a = 0),
  `learned_freq_temp` (log τ = 0) and `darope` (w = 0, α = 1), logits must match the stock
  Hugging Face OLMo-2 on random inputs of length ≤ 4096 (tolerance appropriate to dtype;
  run in float32 for the test). Also test with non-contiguous `position_ids`.
- Test that only positional parameters have `requires_grad=True`.
- Test the PoSE generator: increasing positions, first = 0, max < L', correct length.
- Test the slow-band mask against the formula.
- Tests must run on CPU with a tiny randomly initialized OLMo-2 config, plus one
  optional GPU test with the real checkpoint.

## SCC

Provide `scripts/scc/train.qsub` and `eval.qsub` templates (SGE: `qsub`, project flag,
GPU count and type requests, wall time, module loads, conda env activation). Leave
project name and resource values as clearly marked placeholders; check flags against
BU SCC documentation rather than guessing.

## Conventions

- Python 3.10+, type hints, small functions, no notebooks in the main package.
- Every experiment is reproducible from its YAML config + seed.
- Log to Weights & Biases (team project) with a `--no-wandb` flag.
- Attribute any code copied from other repos (transformers, PoSE, YaRN) in file
  headers and check their licenses.

## Proposed choices (follow by default, but evaluate as we go)

These are the team's current suggestions, not settled decisions. Implement them as the
default, keep the alternative available behind a config flag or a clearly isolated
change, and flag in the README or PR description if something argues against them.

- **Pin `transformers==4.57.1`.** This is the version AdaRoPE's code requires, so copying
  `modeling_olmo2.py` from it makes porting AdaRoPE's modules straightforward. Before
  committing, confirm that this version's OLMo-2 implementation loads
  `allenai/OLMo-2-0425-1B` correctly and works with our evaluation tools; if it doesn't,
  propose a different pin and explain why.
- **Use AdaRoPE's length-dependent temperature (AdaScale) as the default `temp.form`.**
  It makes the RQ1 comparison with AdaRoPE faithful. Keep `constant` as a simpler
  alternative; we will compare the two empirically and may switch.

## Things to verify (don't assume)

- That the proposed `transformers` pin (above) works for OLMo-2, our tests and evaluation tools.
- Whether stock `transformers` YaRN `rope_scaling` works for OLMo-2 (useful as a
  cross-check for our own `yarn` mode).
- Released code (as of early Nov 2026):
  - **AdaRoPE:** `github.com/Outsider565/AdaRoPE` (Apache 2.0). `src/adarope/modules.py` holds
    AdaFreq and AdaScale; `modeling_adarope.py` is a Llama implementation only. Port the
    modules into our OLMo-2 attention rather than using their model class; attribute in
    file headers. The repo is early (single commit) and has no context-extension training
    code, so its training setup is not available to copy.
  - **LeRoPE, DaRoPE:** no public code found; implement from the papers. Re-check before
    the RQ3 runs.
- Reference repos worth reading: `dwzhu-pku/PoSE` (position skipping, training loop),
  `jquesnelle/yarn` (PI/YaRN, passkey and perplexity scripts), NVIDIA `RULER`.

## Out of scope for the skeleton

Running full experiment sweeps, RQ4 implementation, head-level interpretability
analysis, plotting beyond a simple results table.
