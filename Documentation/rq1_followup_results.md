# RQ1 follow-up report: learning rate, initialization and granularity

**Branch:** `feat/rq1-wo-temp-ruler`; RULER and 50-per-cell passkey on `feat/rq1-wo-temp-ruler`
**Cluster:** BU SCC (training and PPL/passkey on A100, `scc-212`; RULER on mixed ≥40 GB GPUs)
**Dates:** 2026-11-11 (training, validation and test evaluation); 2026-11-12 (RULER, passkey re-test)
**Model:** `allenai/OLMo-2-0425-1B`, native context L = 4096, target L' = 16384
**Follows up on:** `Documentation/rq1_experiment.md`

## Summary

- **The first RQ1 sweep under-trained the learned frequencies.** With AdamW each parameter moves
  at most about `lr` per step, so 500 steps at lr 1e-3 let each scale `a` move at most about 0.5.
  YaRN's slow bands correspond to `a = -ln 4 = -1.39`. At lr 3e-2 the learned runs train properly.
- **Perplexity: learned frequencies from `a = 0` beat grid-tuned YaRN at 16k.** The best setting
  is **per layer** (1,024 parameters): PG-19 **test** PPL@16k **19.48** vs tuned YaRN **21.38**
  (validation: 17.40 vs 18.92). Per head is close behind, and shared doesn't beat tuned YaRN.
- **Passkey: every learned run gets 0% at 16k**, while tuned YaRN gets 48% (test) and 60%
  (validation). A re-test with 50 prompts per cell (750 per model) confirms it: 0–2% at 16k for
  all three granularities. At 16k the learned models write fluent filler instead of the key. Per
  layer is perfect up to 8k (100%), so retrieval breaks somewhere between 8k and 16k.
- **The 16k failure is not about distance.** At 16k the learned models get 0% at *every* depth,
  including depth 1.0, where the key sits right before the question (at 8k depth 1.0 they get
  100%). So at 16k they stop answering the question at all; it isn't that the key is too far back.
- **RULER (8 needle-in-a-haystack tasks, test) confirms the collapse:** per layer scores 39.3 /
  21.9 / **1.3** at 4k / 8k / 16k, against tuned YaRN's 45.7 / 35.4 / **24.0**. Shared (0.2) and
  per head (0.0) are no better at 16k.
- **Temperature is what this method is missing for RULER.** A teammate's `learned_freq_temp` per
  layer run, same lr and start (`a = 0`), scores **33.3** on 16k RULER vs our **1.3** (shared:
  36.8 vs 0.2). All temperature runs except per head from `a = 0` (5.9) reach 33–53 at 16k, above
  tuned YaRN. Passkey at 16k is different: most
  temperature runs still get 0–18% there, so passkey has a second problem (see Interpretation).
- **So RQ1's success criterion is not met by frequencies alone.** Lower perplexity didn't turn into
  16k retrieval without a learned temperature.
- **Per head damages the model even at 4k:** RULER 5.8 at 4k (stock 72.8) and passkey 59% at 4k
  (stock 100%). Its 16,384 frequency parameters hurt retrieval inside the original context.
- **The short-context cost is real, and the same for tuned YaRN.** On test, PPL@4k rises from
  16.40 (stock) to 19.99 (per layer) and 19.95 (tuned YaRN). The test 4k score comes from a
  single book, though (see Caveats).
- **Our setup reproduces the first sweep exactly:** stock and default YaRN test perplexities
  match `rq1_experiment.md` to two decimals.
- **Grid tuning matters a lot for YaRN itself:** default YaRN gets 35.76 at 16k on validation,
  tuned YaRN 18.92. Tuned YaRN is a much harder bar than default YaRN.
- **Little sign of memorizing the training books:** validation tracks training loss closely
  (e.g. per head lr 3e-2: train PPL ≈ 16.6, validation PPL@16k 17.86).
- **The first sweep's passkey inconsistency is resolved:** default YaRN really fails at 16k (0%),
  and tuned YaRN really works there, but at 48–60% rather than the grid's 83% (small sample).

## Verdict: best learned run vs tuned YaRN

Per layer, lr 3e-2, from `a = 0` (ours) against grid-tuned YaRN (best baseline), PG-19 / passkey
**test** set.

| Measure | Ours | Tuned YaRN | Winner |
|---|---:|---:|---|
| PPL@16k (lower is better) | **19.48** | 21.38 | ours (about 9% lower) |
| PPL@8k | **21.33** | 22.88 | ours |
| PPL@4k | 19.99 | 19.95 | tie (both worse than stock's 16.40) |
| Passkey @4k | **100%** | 94% | ours, slightly |
| Passkey @8k | **100%** | 66% | ours |
| Passkey @16k | 0% | **48%** | tuned YaRN |
| RULER @4k (avg of 8 NIAH tasks) | 39.3 | **45.7** | tuned YaRN |
| RULER @8k | 21.9 | **35.4** | tuned YaRN |
| RULER @16k | 1.3 | **24.0** | tuned YaRN |
| Short-context benchmarks | — | — | not run |

Passkey: 10 prompts per cell for both columns. Ours with 50 per cell: 100% / 100% / 0%.

**Better:** more fluent at 8k and 16k (validation and test agree), and better passkey retrieval
up to 8k, with parameters learned from the original model rather than set by hand.
**Not better:** RULER at every length, and any retrieval at 16k, which is what the proposal's
success criterion uses. Short-context perplexity is no better than tuned YaRN's.

**Confidence:** the perplexity win holds on both validation and test, and for both per layer and
per head (fairly confident; one seed, few books). The 16k retrieval failure is 0 of 250 passkey
prompts and 1.3% on 4,000 RULER prompts for per layer (very confident).

**One-line takeaway:** learned frequencies alone beat tuned YaRN on long-text perplexity and on
passkey up to 8k, but they fail at 16k retrieval (passkey and RULER). Adding a learned temperature
(teammate's runs) brings 16k RULER above tuned YaRN, so frequencies-only is best read as the
ablation showing that the temperature is necessary.

## What changed from the first sweep

| | First sweep (`rq1_experiment.md`) | This follow-up |
|---|---|---|
| Learning rate | 1e-3 | 1e-2, **3e-2**, 1e-1 (per head), then 3e-2 for all granularities |
| Initialization | `a = 0` | `a = 0`, plus one `init_from_yarn` run |
| YaRN baseline on test | default only | default **and grid-tuned** |
| Passkey | had a generation bug | fixed (see below), 10 prompts per cell instead of 5 |
| Selection | test | **validation** for every choice; test reported once |

### Passkey fix

`answer_passkey` passed explicit `position_ids` to `model.generate()`. Transformers never extends
that tensor, so every generated token after the first got the last prompt position. Now
`generate()` builds positions from the attention mask. Regression test:
`tests/test_rq1_pipeline.py::test_passkey_generation_matches_full_forward` (fails on the old code).
All passkey numbers below use the fixed code.

## Experimental design

### Training (unchanged except where stated)

- 100 PG-19 **train** chunks of 16,384 tokens (one per book), seed 0, regime `long`.
- 500 steps, batch size 1, AdamW, no weight decay, grad clip 1.0, constant lr, bf16 backbone,
  fp32 positional parameters, gradient checkpointing (non-reentrant), one A100, ~13 min per run.
- Mode `learned_freq` (no temperature) in every run.

| Config (`configs/rq1/followup/`) | Granularity | Trainable | lr | Start |
|---|---|---:|---|---|
| `learned_freq_head_lr1e-2.yaml` | head | 16,384 | 1e-2 | `a = 0` |
| `learned_freq_head_lr3e-2.yaml` | head | 16,384 | 3e-2 | `a = 0` |
| `learned_freq_head_lr1e-1.yaml` | head | 16,384 | 1e-1 | `a = 0` |
| `learned_freq_head_yarninit.yaml` | head | 16,384 | 1e-3 | YaRN default |
| `learned_freq_layer_lr3e-2.yaml` | layer | 1,024 | 3e-2 | `a = 0` |
| `learned_freq_shared_lr3e-2.yaml` | shared | 64 | 3e-2 | `a = 0` |

### Baselines (no training)

| Name | Settings |
|---|---|
| stock | `pos.mode: none` |
| YaRN default | factor 4, beta_fast 32, beta_slow 1, attention factor 0.1 ln 4 + 1 ≈ 1.14 |
| YaRN tuned | factor 4, beta_fast 64, beta_slow 2, attention factor 1.2 (grid winner, `rq1_experiment.md`) |

### Evaluation

- **Perplexity:** PG-19 validation or test split, 50 chunks per length (4k / 8k / 16k), seed 0.
- **Passkey:** 3 lengths × 5 depths × 10 prompts = 150 prompts. Validation prompts seed 0, test
  prompts seed 1. Greedy, 16 new tokens. Correct if the 5-digit key appears in the answer.
  **Re-test:** the same test setup (seed 1) with 50 prompts per cell (750 per model), for the
  three lr 3e-2 runs.
- **RULER** (`posext/eval/ruler.py`, from the RULER branch): the 8 needle-in-a-haystack tasks at
  4k / 8k / 16k, RULER's own data and scoring, our inference. Test split: seed 42, 500 samples per
  task per length (12,000 per model). Greedy, batch size 1. Score in percent, averaged over tasks.
  Baselines and temperature runs were scored by teammates on the same test data and settings.
- Job scripts: `configs/rq1/followup/val_ppl.qsub` (`EVAL=ppl|passkey`, `SPLIT=validation|test`);
  RULER `scripts/scc/ruler_prepare.qsub` + `scripts/scc/ruler.qsub`.

## Results

### Training loss

Mean cross-entropy over the previous 10 steps, on the training chunks (seen ~5 times each, not
held out).

| Run | Step 1 | Step 100 | Step 300 | Step 500 | Notes |
|---|---:|---:|---:|---:|---|
| first sweep, lr 1e-3 | ~6.7 | | | 5.2–5.6 | from `rq1_experiment.md` |
| head, lr 1e-2 | 6.70 | 4.97 | 4.12 | 2.96 | still falling at step 500 |
| head, lr 3e-2 | 6.70 | 4.50 | 3.10 | 2.81 | |
| head, lr 1e-1 | 6.70 | 3.46 | 2.99 | 2.91 | grad norm ~65,000 from step 400 (unstable) |
| head, YaRN init, lr 1e-3 | 4.19 | 3.14 | 2.78 | **2.67** | steady |
| shared, lr 3e-2 | | | | 2.88 | |
| layer, lr 3e-2 | | | | 2.73 | |

Differences of about 0.1 are within step-to-step noise.

### Validation perplexity (used for all choices)

Lower is better. Same 50 chunks per length for every row.

| Method | Start | lr | PPL@4k | PPL@8k | PPL@16k | Δ@16k vs tuned YaRN |
|---|---|---|---:|---:|---:|---:|
| YaRN default | | | 25.55 | 31.90 | 35.76 | +16.84 |
| **YaRN tuned** | | | 18.45 | 18.03 | 18.92 | — |
| head | 0 | 1e-2 | 18.38 | 17.53 | 20.64 | +1.72 |
| head | 0 | 3e-2 | 19.50 | 17.57 | 17.86 | −1.06 |
| head | 0 | 1e-1 | 22.93 | 20.30 | 20.13 | +1.21 |
| shared | 0 | 3e-2 | 20.65 | 19.21 | 20.04 | +1.12 |
| **layer** | 0 | 3e-2 | 18.88 | **17.20** | **17.40** | **−1.52** |
| head, YaRN init | YaRN | 1e-3 | **18.13** | **16.49** | **16.49** | **−2.43** |

The stock model has no validation perplexity row (not run).

- **The learning rate matters more than anything else in the from-zero runs.** 1e-2 is still
  under-trained (its loss was still falling at step 500), 1e-1 is unstable and costs the most at
  4k, and 3e-2 is best.
- **Only the YaRN-init run beats tuned YaRN at 4k.** Per layer is the closest from-zero run
  (+0.43).
- **Best from zero vs best overall:** per layer (17.40) is about 0.9 behind the YaRN-init run
  (16.49) at 16k, and about 0.75 behind it at 4k (18.88 vs 18.13).
- PPL@8k is lower than PPL@4k for every method except default YaRN, as expected: more context
  helps prediction once position encoding works.
- The grid search's own report of **20.1** for tuned YaRN at 16k (`rq1_experiment.md`) used 20
  chunks. The 18.92 above uses the same 50 chunks as every other row.

### Test perplexity (reported once; not used for choices)

| Method | PPL@4k | PPL@8k | PPL@16k | Δ@16k vs tuned YaRN |
|---|---:|---:|---:|---:|
| stock | **16.40** | 124.64 | 430.30 | +408.92 |
| YaRN default | 27.11 | 40.14 | 41.67 | +20.29 |
| **YaRN tuned** | 19.95 | 22.88 | 21.38 | — |
| shared, lr 3e-2 | 21.82 | 23.87 | 22.52 | +1.14 |
| **layer, lr 3e-2** | 19.99 | **21.33** | **19.48** | **−1.90** |
| head, lr 3e-2 | 20.74 | 21.77 | 19.73 | −1.65 |

Stock and YaRN default match `rq1_experiment.md` exactly (16.40 / 124.64 / 430.30 and
27.11 / 40.14 / 41.67). The YaRN-init run was not evaluated on test.

**Against the first sweep (same test chunks):** the first sweep's learned runs at lr 1e-3 scored
178.49 (per head), 240.18 (per layer) and 257.90 (shared) at 16k, with 145.73 the best
(`learned_freq_temp` per head). At lr 3e-2 the same granularities score 19.73, 19.48 and 22.52.

Unlike validation, the learned runs and tuned YaRN score **worse at 8k than at 16k** on test. More
context shouldn't hurt, so this is most likely the few-books problem (test 8k chunks come from 3
books, 16k from 6; see Caveats), not a real effect.

### Passkey accuracy

| Method | Val overall | Val 4k | Val 8k | Val 16k | Test overall | Test 4k | Test 8k | Test 16k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stock | 0.33 | 1.00 | 0.00 | 0.00 | 0.33 | 1.00 | 0.00 | 0.00 |
| YaRN default | 0.36 | 0.68 | 0.40 | 0.00 | 0.36 | 0.68 | 0.40 | 0.00 |
| **YaRN tuned** | **0.73** | 1.00 | 0.58 | **0.60** | **0.69** | 0.94 | 0.66 | **0.48** |
| shared, lr 3e-2 | 0.59 | 0.80 | 0.96 | 0.00 | 0.63 | 0.90 | 0.98 | 0.00 |
| **layer, lr 3e-2** | 0.66 | 0.98 | **1.00** | 0.00 | 0.67 | **1.00** | **1.00** | 0.00 |
| head, lr 3e-2 | 0.30 | 0.66 | 0.24 | 0.00 | 0.27 | 0.54 | 0.28 | 0.00 |

50 prompts per length cell. Per-depth results are in the JSON files.

What the models write at 16k (validation, from `"details"` in the result files):

| Method | Typical answer to "...The pass key is" |
|---|---|
| layer | ` yellow. Here we go. The sun is yellow. Here we go.` (all 50 prompts) |
| head | ` blue. The sun is yellow. Here we go.` |
| stock | ` The green The green The green ...` (broken text) |
| YaRN tuned | ` 76505. Here we go. Soft and warm like summer days.` (correct key) |

**The first sweep's passkey inconsistency, resolved.** `rq1_experiment.md` reported tuned YaRN at
83% on 16k validation passkey (grid search) but default YaRN at 0% on 16k test passkey. With the
fixed code and 50 prompts per length:
- Default YaRN really does fail at 16k (0% on both validation and test). It was not an evaluation bug.
- Tuned YaRN really does work at 16k, but at 60% (validation) and 48% (test). The 83% came from
  the grid search's small sample (2 prompts per cell, 10 at 16k).
- So the mismatch was mostly default vs tuned YaRN plus small samples. The generation bug was
  real but small: default YaRN's test passkey changed from 0.64 / 0.44 / 0.00 to
  0.68 / 0.40 / 0.00 (4k / 8k / 16k), with a different number of prompts per cell.

### Passkey re-test: 50 prompts per cell (test, seed 1)

750 prompts per model. Accuracy by depth (0 = key at the start, 1 = key just before the question).

| Run | Length | d 0 | d 0.25 | d 0.5 | d 0.75 | d 1.0 | All depths |
|---|---|---:|---:|---:|---:|---:|---:|
| shared | 4k | 0.94 | 0.78 | 0.52 | 0.98 | 0.98 | 0.84 |
| | 8k | 1.00 | 1.00 | 0.86 | 0.86 | 1.00 | 0.94 |
| | 16k | 0.00 | 0.00 | 0.00 | 0.12 | 0.00 | **0.02** |
| **layer** | 4k | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | **1.00** |
| | 8k | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| | 16k | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** |
| head | 4k | 0.22 | 0.46 | 0.50 | 0.78 | 0.98 | 0.59 |
| | 8k | 0.00 | 0.00 | 0.00 | 0.36 | 0.82 | 0.24 |
| | 16k | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** |

Overall: shared 0.60, layer 0.67, head 0.28 (10-per-cell run: 0.63, 0.67, 0.27). The 10-per-cell
results hold up. For comparison, tuned YaRN's 10-per-cell test passkey at 16k by depth is
1.0 / 0.3 / 0.2 / 0.6 / 0.3: it succeeds at every depth some of the time, ours at none.

Per head at 4k fails mostly when the key is far back (22% at depth 0, 98% at depth 1.0): it has
lost some long-range lookup *inside* the original 4k context.

### RULER (test, 8 NIAH tasks, 500 samples each)

Average over the 8 tasks, in percent. Rows other than ours are from teammates
(`results/sweeps/*/test_ruler_*.json`), same data, seed and scoring.

| Method | 4k | 8k | 16k | Avg |
|---|---:|---:|---:|---:|
| stock | **72.8** | 0.0 | 0.0 | 24.3 |
| PI (factor 4) | 0.0 | 0.0 | 0.0 | 0.0 |
| YaRN default | 8.3 | 6.6 | 1.2 | 5.4 |
| **YaRN tuned** | 45.7 | 35.4 | **24.0** | 35.0 |
| ours: shared, lr 3e-2 | 12.2 | 4.7 | 0.2 | 5.7 |
| **ours: layer, lr 3e-2** | 39.3 | 21.9 | 1.3 | 20.8 |
| ours: head, lr 3e-2 | 5.8 | 0.8 | 0.0 | 2.2 |
| + temp: shared, `a = 0`, lr 3e-2 | 61.0 | 49.1 | 36.8 | 49.0 |
| + temp: layer, `a = 0`, lr 3e-2 | 59.9 | 43.7 | 33.3 | 45.7 |
| + temp: head, `a = 0`, lr 3e-2 | 24.5 | 21.3 | 5.9 | 17.3 |
| + temp: head, YaRN init, lr 1e-2 | 69.8 | 62.9 | **53.1** | **62.0** |

Per task, per layer (ours) vs tuned YaRN at 16k: single_1 0.0 vs 50.4, single_2 2.2 vs 58.6,
multikey_1 6.2 vs 29.4, multiquery 1.2 vs 21.9; both near 0 on multikey_2/3. Per-task scores for
every length are in `results/ruler_rq1f_learned_freq_*_lr3e-2_test.json`.

- **Ours vs tuned YaRN:** per layer is below tuned YaRN at every length, and collapses at 16k.
- **The frequencies-only vs temperature pair** (same granularity, lr, start and training data):
  per layer 1.3 → 33.3 at 16k, shared 0.2 → 36.8. Adding the temperature is the only change.
- **Same granularity ranking as perplexity and passkey:** layer > shared > head.
- **4k cost:** every method that works at 16k loses retrieval at 4k compared with stock (72.8);
  ours (39.3) loses more than the temperature runs (60–70).

## Interpretation

1. **The method can learn once the learning rate is right.** From the original model (`a = 0`),
   per-layer learned frequencies beat tuned YaRN on perplexity at 8k and 16k, on both
   validation and test. The first sweep's failure was the optimizer budget, not the method.
2. **Per layer is the best granularity so far.** It beats per head (16× more parameters) on
   every perplexity column and on passkey up to 8k. Shared is not enough. This disagrees with
   AdaRoPE's claim that per-head is needed. One seed, so it needs repeating.
3. **Perplexity and retrieval disagree at 16k.** The learned models are fluent at 16k (low
   perplexity, grammatical filler) but continue the filler ("...is yellow. Here we go.") instead
   of giving the key. Stock, by contrast, produces broken text. *An earlier version of this report
   said they had "lost long-range lookup"; the per-depth results rule that out.* At 16k they get
   0% even at depth 1.0, where the key is a sentence before the question, and at 8k they get 100%
   at that depth. So at 16k they stop treating the question as a question at all, whatever the
   distance. PG-19 perplexity mostly rewards local prediction, so it can improve while this breaks.
4. **Why: mostly the missing temperature.** These runs use `learned_freq` with no temperature. At
   16k each query spreads its attention over 4× more keys than at 4k ("attention dilution"), so the
   few relevant tokens get too little weight. Tuned YaRN counters this by multiplying attention
   logits by 1.2² ≈ 1.44, and `learned_freq_temp` learns such a scale. The RULER pairs above
   support this: with everything else equal, adding the temperature takes 16k RULER from 1.3 to
   33.3 (per layer) and from 0.2 to 36.8 (shared).
5. **Passkey at 16k has a second problem.** With temperature, most runs still get 0–18% on 16k
   passkey (layer `a = 0`: 0%; head: 0–18%), even where they reach 33–53 on 16k RULER. Only shared
   + temperature works (96%). Passkey's haystack is one sentence repeated thousands of times; RULER's
   is real essays. Two untested explanations: very repetitive filler is a special case where
   dilution is extreme, or the sped-up slow bands (`a` up to +6.8 in per layer) alias, making
   distant identical filler look nearby. Shared has a single `a` per band for all layers, which
   may be why it avoids this. Teammates' passkey numbers use 10 prompts per cell.
6. **Starting from YaRN's values gives the best validation perplexity** (16.49 at 16k), but that is
   AdaRoPE's initialization, so it's an ablation for us. It hasn't been evaluated on passkey or RULER.
7. **Short context:** on test, both tuned YaRN and per layer lose about 3.6 PPL at 4k relative to
   stock (16.40 → ~20), and per layer loses about half of stock's 4k RULER score (72.8 → 39.3). The
   success criterion is measured on HellaSwag / ARC-Easy / PIQA, which haven't been run.

### Success criteria (from the proposal)

Beat tuned YaRN on 16k RULER and passkey while losing at most 1 point of short-context accuracy:
**not met** by frequencies alone. 16k passkey is 0% vs tuned YaRN's 48%, and 16k RULER is 1.3 vs
24.0. Short-context benchmarks were not run. (With temperature, teammates' runs beat tuned YaRN on
16k RULER but not on 16k passkey, except shared + temperature.)

## Learned frequency scales

`a` from `outputs/<run>/positional.safetensors`, slow bands 32–63, over all layers (and heads).
Default YaRN uses `a = -1.39` on every slow band.

| Run | Mean | Min | Max | Std |
|---|---:|---:|---:|---:|
| head, lr 1e-2 | -0.06 | -3.12 | +4.74 | 1.05 |
| head, lr 3e-2 | -0.09 | -4.53 | +8.36 | 1.50 |
| head, lr 1e-1 | -0.17 | -6.34 | +12.78 | 2.11 |
| head, YaRN init | -1.37 | -1.91 | -0.82 | 0.17 |
| layer, lr 3e-2 | -0.62 | -5.22 | +6.77 | — |

A slow-band `a` of +8 means about e⁸ ≈ 3,000× faster rotation, so the from-zero runs turn some
slow bands into fast ones, while others are slowed more than YaRN does.

Per layer, by layer (slow-band mean): layers 4, 5, 8, 9, 12, 13 slow down most (−1.0 to −1.5,
close to YaRN), while layers 0 and 10 speed up about as many slow bands as they slow down. Fast
bands stay near 0 in every run (per layer: mean −0.08).

## Caveats

- **Few books per evaluation.** `posext/eval/ppl.py` takes 50 chunks from shuffled books with no
  per-book cap. Validation uses 3 books at 4k, 6 at 8k and 9 at 16k. **Test uses 1 book at 4k,
  3 at 8k and 6 at 16k.** Rankings are fair (identical chunks for every method), but absolute
  values are noisy, and **test PPL@4k reflects a single book**. This probably explains why
  test PPL@8k is higher than PPL@16k for the learned runs. Fix (pending): expose
  `max_chunks_per_book` in `ppl.py` and rerun.
- **One seed per run.** Seed-to-seed variation is unknown. With fixed `a = 0` initialization, the
  training seed only changes which 100 books are used and their order.
- **Selection on validation.** The learning rate and granularity were picked on validation, so
  validation numbers for the winners are slightly optimistic. Test numbers are reported once.
- **Passkey is synthetic and uses one filler text.** 10 prompts per cell (50 per length), and a
  re-test of the three lr 3e-2 runs with 50 per cell (250 per length).
- **RULER GPUs varied** (L40S, A100 and others, one job per model and length). Decoding is greedy,
  so this should only cause tiny bf16 differences.
- **Cross-team comparisons:** the temperature runs come from a different training sweep (the
  pipeline on the RULER branch); the RULER pairs match lr, start and granularity, but not every
  detail was checked to be identical.
- Not run: stock validation perplexity, YaRN-init passkey / test / RULER, short-context benchmarks.

## Next steps

1. **Find what breaks passkey at 16k** (frequencies alone and most temperature runs). Cheap
   tests: passkey at 10k / 12k / 14k (find the cliff); passkey with PG-19 text as the haystack (is it
   the repetitive filler?); per layer `a` plus a fixed 1.44× sharpening (constant temperature,
   `log_tau = ln 1.44 ≈ 0.365`, no training).
2. **Fix the few-books evaluation** (`max_chunks_per_book`) and rerun validation and test perplexity.
3. **Seeds 1 and 2** for per layer and per head at lr 3e-2.
4. Stock validation perplexity, and passkey / RULER for the YaRN-init run.
5. Short-context benchmarks (`posext/eval/shortctx.py`).
6. Training-recipe options: warmup and decay, more training books, mixing in 4k chunks.

## How to reproduce

From the repo root on SCC:

```bash
# training (one per config)
qsub -v POSEXT_ROOT=$PWD,POSEXT_STUDENT_ROOT=/projectnb/cs599b1/students/$USER,CONFIG=configs/rq1/followup/learned_freq_layer_lr3e-2.yaml scripts/scc/train.qsub

# evaluation: pick a model (RUN=<outputs dir name>, YARN=default|tuned, STOCK=1),
# EVAL=ppl|passkey (default ppl), SPLIT=validation|test (default validation)
qsub -v RUN=rq1f_learned_freq_layer_lr3e-2 configs/rq1/followup/val_ppl.qsub
qsub -v RUN=rq1f_learned_freq_layer_lr3e-2,EVAL=passkey,SPLIT=test configs/rq1/followup/val_ppl.qsub
qsub -v YARN=tuned,SPLIT=test configs/rq1/followup/val_ppl.qsub

# RULER (once: bash scripts/scc/ruler_setup.sh on a login node), then one GPU job per length
MYENV=POSEXT_ROOT=$PWD,POSEXT_STUDENT_ROOT=/projectnb/cs599b1/students/$USER
qsub -v $MYENV,SPLIT=test scripts/scc/ruler_prepare.qsub
qsub -hold_jid <prepare job> -v $MYENV,RUN_DIR=outputs/rq1f_learned_freq_layer_lr3e-2,SPLIT=test,LENGTHS=16384 scripts/scc/ruler.qsub
# combine the lengths into one file (each job only writes its own length)
uv run python -m posext.eval.ruler score --pred-dir outputs/ruler/rq1f_learned_freq_layer_lr3e-2/test \
  --out results/ruler_rq1f_learned_freq_layer_lr3e-2_test.json

# passkey re-test, 50 per cell
uv run python -m posext.eval.passkey --run outputs/rq1f_learned_freq_layer_lr3e-2 --seed 1 \
  --n-per-cell 50 --device cuda --out results/passkey_n50/test_passkey_n50_nandini_layer_lr3e-2.json
```

Outputs: `outputs/rq1f_*/`, RULER predictions `outputs/ruler/<run>/test/`. Logs: `logs/`.
Results: `results/<val|test>_<ppl|passkey>_<name>.json`, `results/ruler_<run>_test.json`,
`results/passkey_n50/`.
