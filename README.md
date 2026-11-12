# FrozenRoPE

How Much of Context Extension Is Positional? Learning RoPE Parameters on a Frozen Language Model.

Extend **OLMo-2-0425-1B** from 4k to 16k by training only RoPE parameters on a frozen backbone, and compare against PI / YaRN.

Engineering rules and status: [`CLAUDE.md`](CLAUDE.md). Team guide: [`Documentation/project_status.md`](Documentation/project_status.md).

## Setup

```bash
uv sync --all-extras          # Python 3.11 + pinned deps
uv run pytest                 # CPU suite (~seconds)
uv run python scripts/check_env.py   # optional: real checkpoint pin check
```

On SCC, **never install into `$HOME`** (≈10 GB quota). Source the env file so venv,
uv cache, HF weights, and PG-19 all live under projectnb. Each teammate works in their own
folder, `/projectnb/cs599b1/students/$USER`; `env.sh` and the `.qsub` scripts default to it
(override with `POSEXT_STUDENT_ROOT` / `POSEXT_ROOT`, passed to `qsub` with `-v`).

```bash
cd /projectnb/cs599b1/students/$USER/FrozenRoPE
source scripts/scc/env.sh          # redirects every cache off $HOME
bash scripts/scc/setup.sh          # uv sync into projectnb .venv
```

## RQ1 at a glance

| Config | Train? | What |
|---|---|---|
| `configs/rq1/none.yaml` | no | stock RoPE |
| `configs/rq1/pi.yaml` | no | Position Interpolation |
| `configs/rq1/yarn_default.yaml` | no | YaRN paper defaults |
| `configs/rq1/yarn_tuned.yaml` | no | best from YaRN grid |
| `configs/rq1/learned_freq_{shared,layer,head}.yaml` | yes | learned `a` |
| `configs/rq1/learned_freq_temp_*.yaml` | yes | learned `a` + AdaScale temp |

### Local vs SCC

- **Local (e.g. Mac M4 16 GB):** use `configs/smoke.yaml` and short eval lengths only. Full 16k training/eval will OOM or be impractically slow without a datacenter GPU.
- **SCC (A100 / L40S):** all real RQ1 runs -- YaRN grid, 16k training, PPL / passkey / RULER / short-context.

### Smoke (local)

```bash
uv run python -m posext.train --config configs/smoke.yaml --device cpu --no-wandb
```

### SCC training

```bash
mkdir -p logs
qsub -v CONFIG=configs/rq1/learned_freq_head.yaml scripts/scc/train.qsub
```

Edit placeholders in `scripts/scc/*.qsub` (`<PROJECT>`, GPU type, modules) against current BU SCC docs first.

### YaRN grid (validation) -> fill `yarn_tuned.yaml`

```bash
qsub scripts/scc/yarn_grid.qsub
# then copy outputs/yarn_grid/best.json yarn.* fields into configs/rq1/yarn_tuned.yaml
```

### Evaluation

```bash
# Trained run
uv run python -m posext.eval.ppl --run outputs/rq1_learned_freq_head --split test --out results/ppl.json
uv run python -m posext.eval.passkey --run outputs/rq1_learned_freq_head --seed 1 --out results/passkey.json

# Training-free baseline
uv run python -m posext.eval.ppl --pos-mode yarn --factor 4 --split test
uv run python -m posext.eval.passkey --pos-mode yarn --factor 4 --seed 1

# Short-context regression (needs --extra eval)
uv run python -m posext.eval.shortctx --run outputs/rq1_learned_freq_head
```

### RULER (8 needle-in-a-haystack tasks at 4k / 8k / 16k)

RULER's own scripts generate the data and its metric scores the answers; inference runs
through our loaders, so trained runs and training-free modes work the same way. Validation
uses 100 samples per task per length (seed 1234), test uses 500 (seed 42).

```bash
# Local
uv run python -m posext.eval.ruler setup                        # internet: RULER + essays + nltk
uv run python -m posext.eval.ruler prepare --split validation   # CPU, ~1 min per length
uv run python -m posext.eval.ruler run --run outputs/rq1_learned_freq_head --split validation
uv run python -m posext.eval.ruler run --pos-mode yarn --factor 4 --split validation
uv run python -m posext.eval.ruler score --pred-dir outputs/ruler/<tag>/validation

# SCC: setup on a login node (needs internet), generation and inference as batch jobs
bash scripts/scc/ruler_setup.sh
qsub -v SPLIT=validation scripts/scc/ruler_prepare.qsub
qsub -v RUN_DIR=outputs/rq1_learned_freq_head,SPLIT=validation scripts/scc/ruler.qsub
qsub -v POS_MODE=none,SPLIT=validation scripts/scc/ruler.qsub
```

Results: `results/ruler_<tag>_<split>.json` (score per task and length, averages over tasks,
`complete` flag); per-sample predictions in `outputs/ruler/<tag>/<split>/`. `run` resumes
where it stopped, so a job that hits its wall time can be resubmitted unchanged. Use
`--max-samples N` for a quick check and `--tasks vt,...` for other RULER tasks.

## Layout

```
posext/           # model, data, train, eval, baselines
configs/rq1/      # one YAML per RQ1 cell
scripts/scc/      # qsub templates
tests/            # identity + plumbing tests (CPU)
Documentation/    # proposal + papers
```
