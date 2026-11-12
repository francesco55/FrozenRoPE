# RQ1 follow-up: learning rate and initialization

**Status:** 4 training runs and 6 validation evaluations done (2026-11-11). Perplexity only,
one seed. Follows up on `Documentation/rq1_experiment.md`.

## Summary

- The first RQ1 sweep (lr 1e-3) **under-trained the learned frequencies**. With AdamW each
  parameter moves at most about `lr` per step, so 500 steps at 1e-3 let each `a` move at most about 0.5,
  while YaRN's slow bands correspond to `a = -ln 4 = -1.39`.
- Raising the learning rate fixes this. Final training loss drops from **5.2–5.6** to **2.8–3.0**.
- **On PG-19 validation PPL@16k, two learned runs beat grid-tuned YaRN (18.92):**
  - learned from `a = 0` at lr 3e-2: **17.86** (−1.06, about 6% lower),
  - learned from YaRN's values (`init_from_yarn`): **16.49** (−2.43, about 13% lower), and also
    better than tuned YaRN at 4k and 8k.
- The from-zero run pays at 4k: 19.50 vs tuned YaRN's 18.45. The stock model's validation PPL@4k
  hasn't been measured yet, so the size of the short-context cost relative to the original model
  is still unknown.
- Grid tuning matters a lot for YaRN itself: default YaRN gets 35.76 at 16k, tuned YaRN 18.92.
- Validation tracks training loss closely (e.g. lr 3e-2: train PPL ≈ 16.6, val PPL@16k 17.86), so
  there's little sign of memorizing the 100 training books.
- The from-zero runs do **not** rediscover YaRN. They spread the slow-band scales widely
  (some bands slowed, some sped up a lot) instead of slowing every slow band evenly. Validation
  suggests this solution generalizes.

## Setup

Identical to `configs/rq1/learned_freq_head.yaml` (per-head `learned_freq`, 16,384 trainable
parameters, 100 PG-19 train chunks of 16,384 tokens, 500 steps, batch size 1, AdamW, no weight
decay, grad clip 1.0, bf16 backbone, one A100, about 13 minutes per run) except:

| Config (`configs/rq1/followup/`) | lr | Start |
|---|---|---|
| `learned_freq_head_lr1e-2.yaml` | 1e-2 | `a = 0` |
| `learned_freq_head_lr3e-2.yaml` | 3e-2 | `a = 0` |
| `learned_freq_head_lr1e-1.yaml` | 1e-1 | `a = 0` |
| `learned_freq_head_yarninit.yaml` | 1e-3 | YaRN default (`init_from_yarn: true`) |

Seed 0, one seed per run.

## Training loss

Mean cross-entropy over the previous 10 steps, on the 100 training chunks. These books are seen
about 5 times each, so these numbers are **not** held-out.

| Run | Step 1 | Step 100 | Step 300 | Step 500 | Notes |
|---|---:|---:|---:|---:|---|
| original lr 1e-3 (first sweep) | ~6.7 | | | 5.2–5.6 | from `rq1_experiment.md` |
| lr 1e-2 | 6.70 | 4.97 | 4.12 | 2.96 | still falling at step 500 |
| lr 3e-2 | 6.70 | 4.50 | 3.10 | 2.81 | |
| lr 1e-1 | 6.70 | 3.46 | 2.99 | 2.91 | grad norm about 65,000 from step 400 (unstable) |
| YaRN init, lr 1e-3 | 4.19 | 3.14 | 2.78 | **2.67** | steady |

Differences of about 0.1 are within step-to-step noise.

## Learned frequency scales (slow bands 32–63)

`a` from `outputs/<run>/positional.safetensors`, over all 16 layers × 16 heads. For reference,
default YaRN uses `a = -1.39` on every slow band and `0` on the fastest bands.

| Run | Mean | Min | Max | Std |
|---|---:|---:|---:|---:|
| lr 1e-2 | -0.06 | -3.12 | +4.74 | 1.05 |
| lr 3e-2 | -0.09 | -4.53 | +8.36 | 1.50 |
| lr 1e-1 | -0.17 | -6.34 | +12.78 | 2.11 |
| YaRN init | -1.37 | -1.91 | -0.82 | 0.17 |

Fast bands (0–13) stay near 0 in every run. A slow-band `a` of +8 means about 3,000× faster
rotation, so the from-zero runs turn some slow bands into fast ones. Whether this generalizes is
what validation tests.

## Validation perplexity (PG-19 validation, 50 chunks per length)

Lower is better. All rows use the same 50 chunks per length (seed 0).

| Method | Trained? | PPL@4k | PPL@8k | PPL@16k | Δ@16k vs tuned YaRN |
|---|---|---:|---:|---:|---:|
| YaRN default | no | 25.55 | 31.90 | 35.76 | +16.84 |
| YaRN tuned (grid winner) | no | 18.45 | 18.03 | 18.92 | — |
| learned, `a = 0`, lr 1e-2 | yes | 18.38 | 17.53 | 20.64 | +1.72 |
| learned, `a = 0`, lr 3e-2 | yes | 19.50 | 17.57 | **17.86** | **−1.06** |
| learned, `a = 0`, lr 1e-1 | yes | 22.93 | 20.30 | 20.13 | +1.21 |
| learned, YaRN init, lr 1e-3 | yes | **18.13** | **16.49** | **16.49** | **−2.43** |

Tuned YaRN = factor 4, beta_fast 64, beta_slow 2, attention_factor 1.2 (from
`outputs/yarn_grid/best.json`).

### Granularity at lr 3e-2 (from `a = 0`)

Configs `learned_freq_{shared,layer}_lr3e-2.yaml`, otherwise identical to the per-head lr 3e-2 run.

| Granularity | Trainable | Final train loss | PPL@4k | PPL@8k | PPL@16k | Δ@16k vs tuned YaRN |
|---|---:|---:|---:|---:|---:|---:|
| shared | 64 | 2.88 | 20.65 | 19.21 | 20.04 | +1.12 |
| per layer | 1,024 | 2.73 | 18.88 | **17.20** | **17.40** | **−1.52** |
| per head | 16,384 | 2.81 | 19.50 | 17.57 | 17.86 | −1.06 |

- **Per layer is the best from-zero setting** at every length, and the closest to tuned YaRN at 4k
  (+0.43). Per head, with 16× more parameters, is not better.
- Shared (one set of 64 scales for the whole model) doesn't beat tuned YaRN.
- One seed each, so the per-layer vs per-head gap (0.46 at 16k) needs repeating before claiming
  an order.

Observations:
- **lr matters more than anything else in the from-zero runs.** 1e-2 is still under-trained (its
  loss was still falling at step 500), 1e-1 is unstable and costs the most at 4k, and 3e-2 is best.
- **The YaRN start wins everywhere** and is the only learned run better than tuned YaRN at 4k.
  Its slow-band scales stay close to YaRN's (mean −1.37), so the gain comes from small per-head
  adjustments. This is AdaRoPE's initialization, so it's an ablation for us, not the main method.
- **From zero, lr 3e-2 beats tuned YaRN at 8k and 16k** but is about 1 PPL worse at 4k. That
  trade-off is the main thing to fix.
- PPL@8k ≤ PPL@4k for most methods, as expected: more context helps prediction once position
  encoding works.

Caveats:
- **One seed per run, and differences are about 1–2 PPL.** Seed-to-seed variation is unknown, so
  the from-zero vs tuned-YaRN gap (−1.06) is suggestive, not established.
- **Picked on validation.** The best learning rate was chosen on these same validation numbers
  (as was tuned YaRN), so final claims need the test split.
- The grid search's own report of **20.1** for tuned YaRN used 20 chunks. The 18.92 above uses the
  same 50 chunks as every other row.
- **No stock-model (`none`) validation row yet.** It's needed to judge the short-context cost: on
  test the stock model is 16.40 at 4k.
- **Perplexity only.** The success criterion also needs 16k passkey and RULER, plus short-context
  benchmarks (HellaSwag, ARC-Easy, PIQA). None have been run on these models.
- Test-set numbers are deliberately not reported here. Test stays for final results.

## Open questions

1. Is the from-zero gain over tuned YaRN robust across seeds?
2. How much does PPL@4k rise relative to the stock model (the short-context criterion)?
3. Does the lower perplexity translate into retrieval (passkey, RULER)?
4. Passkey: the first sweep's numbers looked inconsistent (tuned YaRN 83% at 16k on validation,
   default YaRN 0% at 16k on test). Check this before running passkey on these runs.

## Possible next runs

- Stock model (`none`) on validation, to get the 4k reference.
- lr 3e-2 from `a = 0` with 2 more seeds.
- lr 3e-2 with warmup and decay (AdaRoPE uses warmup-stable-decay), to keep the speed without the
  lr 1e-1 instability.
- Mix some 4k chunks into training, to reduce the 4k cost of the from-zero run.
- More training sequences (300–500 instead of 100) to reduce repetition.
- 2–3 seeds for the best setting.
- `learned_freq_temp` with separate learning rates for frequency and temperature (AdaRoPE: 3e-2
  and 3e-3). This needs a small change to `posext/train.py`, which uses one lr for all parameters.

## Reproduce

From the repo root on SCC:

```bash
# training (one per config)
qsub -v POSEXT_ROOT=$PWD,POSEXT_STUDENT_ROOT=/projectnb/cs599b1/students/$USER,CONFIG=configs/rq1/followup/learned_freq_head_lr1e-2.yaml scripts/scc/train.qsub

# validation perplexity
qsub -v RUN=lr1e-2 configs/rq1/followup/val_ppl.qsub
qsub -v YARN=tuned configs/rq1/followup/val_ppl.qsub
```

Outputs: `outputs/rq1f_learned_freq_head_<run>/`. Logs: `logs/`.
Results: `results/val_ppl_<name>.json`.
