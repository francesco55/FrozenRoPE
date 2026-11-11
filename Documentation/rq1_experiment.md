# RQ1 Experiment Report

**Branch:** `feat/rq1-learned-rope`  
**Cluster:** BU SCC (`scc-212`, A100)  
**Project path:** `/projectnb/cs599b1/students/adithyad/FrozenRoPE`  
**Dates:** 2026-11-10 (train/eval) – 2026-11-11 (results collected)  
**Model:** `allenai/OLMo-2-0425-1B` (native context L = 4096 ? target L? = 16384)

## Research question

**RQ1:** Starting from a pretrained LLM, what does learning per-head RoPE parameters on a **frozen** backbone add beyond YaRN?

We learn frequency scales \(a\) (and optionally AdaScale temperature) at three granularities (shared / per-layer / per-head), initialize at identity (\(a = 0\)), and compare to PI, default YaRN, and grid-tuned YaRN.

## Code and environment setup

### Repo layout (RQ1-relevant)

| Path | Role |
|---|---|
| `posext/models/` | Patched OLMo-2 + positional modules (from skeleton) |
| `posext/data/` | PG-19 chunks + PoSE collator (from skeleton) |
| `posext/train.py` | Positional-only training |
| `posext/loss.py` | Chunked CE (fits 16k vocab logits in memory) |
| `posext/eval/{ppl,passkey,shortctx,ruler}.py` | Evaluation |
| `posext/baselines/yarn_grid.py` | YaRN validation grid |
| `configs/rq1/*.yaml` | One config per RQ1 cell |
| `scripts/scc/` | `env.sh`, `setup.sh`, `train.qsub`, `eval.qsub`, `yarn_grid.qsub` |

### SCC storage (avoid `$HOME` quota)

All heavy artifacts under `/projectnb/cs599b1/students/adithyad/`:

| Directory | Contents |
|---|---|
| `FrozenRoPE/` | code + `.venv` |
| `uv_cache/` | uv download cache |
| `hf_home/` | OLMo-2 weights |
| `posext_data/` | PG-19 chunk caches |
| `wandb/`, `tmp/`, … | other caches |

Always:

```bash
cd /projectnb/cs599b1/students/adithyad/FrozenRoPE
source scripts/scc/env.sh
bash scripts/scc/setup.sh   # first time
```

### Software

- Python 3.11 via `uv`
- `transformers==4.57.1`, `torch==2.9.1`, bf16, SDPA
- W&B disabled on SCC (`--no-wandb`; no API key on compute nodes)

### GPU request

```
#$ -P cs599b1
#$ -l gpus=1
#$ -l gpu_type=A100
#$ -l gpu_memory=40G
#$ -pe omp 8
```

Jobs ran on `a100@scc-212.scc.bu.edu`.

## Experimental design

### Data

- **Corpus:** PG-19 via Hugging Face mirror `emozilla/pg19`
- **Train:** 100 non-overlapping **16384**-token chunks from the train split (`max_chunks_per_book=1`)
- **YaRN grid:** validation split (PPL @16k + val passkey)
- **Final metrics:** **test** split only

### Training (6 learned runs)

| Config | Mode | Granularity | Trainable params (approx.) |
|---|---|---|---|
| `learned_freq_shared` | `learned_freq` | shared | 64 |
| `learned_freq_layer` | `learned_freq` | layer | 1,024 |
| `learned_freq_head` | `learned_freq` | head | 16,384 |
| `learned_freq_temp_shared` | `learned_freq_temp` + AdaScale | shared | 64 + temp |
| `learned_freq_temp_layer` | `learned_freq_temp` + AdaScale | layer | … |
| `learned_freq_temp_head` | `learned_freq_temp` + AdaScale | head | … |

Shared hyperparameters (YAML):

- regime: `long` (not PoSE — PoSE is RQ2)
- steps: **500**, batch size **1**, lr **1e-3**, AdamW, no weight decay
- grad clip 1.0 (per-group freq/temp when both present)
- backbone **frozen**; only `model.posext.*` trains
- gradient checkpointing (`use_reentrant=False`), bf16 backbone, fp32 positional params
- init: identity (`a = 0`, AdaScale \(T=1,\gamma=0\)) — **not** YaRN init

Wall time: ~**12–13 minutes per run** on one A100 (~1.5 s/step). Six runs serialized ? **1.5 hours**.

### Baselines (no training)

| Name | Settings |
|---|---|
| `none` | stock RoPE |
| `pi` | Position Interpolation, factor 4 |
| `yarn` (default) | factor 4, beta_fast 32, beta_slow 1, default attention_factor |
| `yarn_tuned` | from validation grid (below) — **test eval of tuned settings not completed in this sweep**; default YaRN was evaluated on test |

### YaRN validation grid

- **144** settings: factor ? {4,6,8} × beta_fast ? {8,16,32,64} × beta_slow ? {0.5,1,2} × attention_factor ? {1.0,1.1,1.2,1.3}
- Score: z-scored mix of (?val NLL @16k) and val passkey
- Runtime: ~**50 minutes** on A100

**Winner (`outputs/yarn_grid/best.json`):**

```json
{
  "yarn": {
    "factor": 4.0,
    "beta_fast": 64.0,
    "beta_slow": 2.0,
    "attention_factor": 1.2
  },
  "nll": 3.001,
  "ppl": 20.108,
  "passkey_accuracy": 0.917,
  "passkey_by_length": {"4096": 1.0, "16384": 0.833}
}
```

### Evaluation protocol

For each method:

1. **PG-19 test perplexity** at 4096 / 8192 / 16384 (50 chunks each)
2. **Passkey retrieval** — 75 prompts (3 lengths × 5 depths × 5 seeds), seed=1 (test)

Short-context lm-eval and RULER were **not** run in this sweep (`ruler.py` remains a stub).

## Results

### Test perplexity (lower is better)

| Method | PPL@4k | PPL@8k | PPL@16k |
|---|---:|---:|---:|
| none | 16.40 | 124.64 | 430.30 |
| PI | 109.09 | 118.56 | 110.89 |
| **YaRN (default)** | 27.11 | **40.14** | **41.67** |
| learned_freq shared | 16.58 | 52.79 | 257.90 |
| learned_freq layer | 16.47 | 70.76 | 240.18 |
| learned_freq head | 16.68 | 64.26 | 178.49 |
| learned_freq_temp shared | 19.23 | 52.17 | 191.23 |
| learned_freq_temp layer | 18.92 | 61.43 | 171.15 |
| learned_freq_temp head | 17.79 | 57.89 | **145.73** |

### Test passkey accuracy

| Method | overall | @4k | @8k | @16k |
|---|---:|---:|---:|---:|
| none | 0.333 | 1.00 | 0.00 | 0.00 |
| PI | 0.000 | 0.00 | 0.00 | 0.00 |
| YaRN (default) | 0.360 | 0.64 | **0.44** | 0.00 |
| learned_freq shared | 0.333 | 1.00 | 0.00 | 0.00 |
| learned_freq layer | 0.333 | 1.00 | 0.00 | 0.00 |
| learned_freq head | 0.333 | 1.00 | 0.00 | 0.00 |
| learned_freq_temp shared | 0.333 | 1.00 | 0.00 | 0.00 |
| learned_freq_temp layer | 0.280 | 0.84 | 0.00 | 0.00 |
| learned_freq_temp head | 0.333 | 1.00 | 0.00 | 0.00 |

Raw JSON on SCC: `/projectnb/cs599b1/students/adithyad/FrozenRoPE/results/`.

### Training loss notes (not final metrics)

Final train CE after 500 steps (16k chunks) was roughly 5.2–5.6 depending on run. Temp variants reached lower train loss than freq-only. These numbers are **not** comparable to eval PPL.

## Interpretation (RQ1)

1. **Learned RoPE did not beat YaRN** on this setup.
2. Best learned 16k PPL ? **146** (temp + head) vs YaRN default ? **42**.
3. Trends within learned methods: more granularity helps somewhat (shared ? head); adding AdaScale temperature helps further; still far from YaRN.
4. YaRN trades some 4k PPL (27 vs ~16) for stable 8k/16k.
5. PI keeps long PPL flat-ish (~110) but destroys short context (4k ? 109).
6. Untouched model fails beyond 4k as expected.
7. Passkey beyond 4k is near-zero for learned methods; YaRN alone gets non-zero @8k. **Caveat:** generation logged warnings that the model’s predefined max length is still **4096**, so long passkey numbers may be partly an eval bug. Re-run passkey after raising `max_position_embeddings` before strong claims on retrieval.

### Success criteria (from proposal)

Beat tuned YaRN on 16k RULER + passkey while losing ?1 short-context point — **not met** in this sweep (and RULER/short-context harness not run yet).

## Issues encountered

| Issue | Fix |
|---|---|
| `$HOME` uv cache (~7 GB) after first `uv sync` | `scripts/scc/env.sh` + wipe `~/.cache/uv` |
| W&B login crash on compute nodes | `--no-wandb` in `train.qsub`; harden `maybe_init_wandb` |
| Typo `yarntuned.yaml` | renamed to `yarn_tuned.yaml` |
| Duplicate default-YaRN eval job | optional `qdel`; tuned YaRN test eval still outstanding |
| `generate()` max-length 4096 warning | open follow-up: set `max_position_embeddings=16384` on load/eval |

## How to reproduce

```bash
# on SCC
cd /projectnb/cs599b1/students/adithyad/FrozenRoPE
source scripts/scc/env.sh

qsub scripts/scc/yarn_grid.qsub
# copy outputs/yarn_grid/best.json -> configs/rq1/yarn_tuned.yaml

for c in shared layer head; do
  qsub -v CONFIG=configs/rq1/learned_freq_${c}.yaml scripts/scc/train.qsub
  qsub -v CONFIG=configs/rq1/learned_freq_temp_${c}.yaml scripts/scc/train.qsub
done

for d in outputs/rq1_learned_freq_{shared,layer,head} outputs/rq1_learned_freq_temp_{shared,layer,head}; do
  qsub -v RUN_DIR=$d scripts/scc/eval.qsub
done
qsub -v POS_MODE=none scripts/scc/eval.qsub
qsub -v POS_MODE=pi,FACTOR=4 scripts/scc/eval.qsub
qsub -v POS_MODE=yarn,FACTOR=4 scripts/scc/eval.qsub
# tuned YaRN test eval still to be wired with beta_fast=64 beta_slow=2 attention_factor=1.2
```

## Next steps

1. Fix long-context `generate` (`max_position_embeddings`) and re-eval passkey.
2. Run **tuned** YaRN on the test set (grid winner settings).
3. Short-context regression (`posext.eval.shortctx`) and RULER.
4. Decide RQ2 (PoSE / short data) vs RQ3 (DaRoPE) given RQ1 outcome: learned ? or < tuned YaRN on PPL so far ? parameterization / training recipe may need to change before claiming gains over YaRN.
5. Optional: longer training, LR sweep on validation, `init_from_yarn: true` ablation (AdaRoPE-style).

## Artifact checklist (SCC)

- [x] YaRN grid `outputs/yarn_grid/best.json`
- [x] Six trained runs under `outputs/rq1_learned_freq*` (`positional.safetensors` + configs)
- [x] Test PPL + passkey JSON under `results/`
- [ ] Tuned-YaRN test JSON
- [ ] Short-context / RULER JSON
- [ ] This report committed on `feat/rq1-learned-rope`
