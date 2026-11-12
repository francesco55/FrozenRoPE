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
uv cache, HF weights, and PG-19 all live under projectnb:

```bash
cd /projectnb/cs599b1/students/adithyad/FrozenRoPE
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

# RULER wrapper is still a stub; export then use NVIDIA RULER manually if needed
uv run python -m posext.eval.ruler --run outputs/rq1_learned_freq_head
```

## Layout

```
posext/           # model, data, train, eval, baselines
configs/rq1/      # one YAML per RQ1 cell
scripts/scc/      # qsub templates
tests/            # identity + plumbing tests (CPU)
Documentation/    # proposal + papers
```
