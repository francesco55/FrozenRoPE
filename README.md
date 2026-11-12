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
TMPDIR=/tmp uv run pytest          # tests on SCC: env.sh puts TMPDIR on projectnb (NFS), where
                                   # 6 save/reload tests fail to delete their temp dirs
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

### Overnight pipeline: sweep -> select on validation -> test (one call)

Run on an SCC **login node**. It queues every job at once; each waits for the ones it needs
(`-hold_jid`), so you can submit and walk away.

```bash
bash scripts/scc/pipeline.sh configs/rq1/learned_freq_head.yaml --sweep train.lr=1e-3,1e-2,3e-2
# grid: repeat --sweep (e.g. --sweep pos.temp.form=adascale,constant); --dry-run prints the plan only
bash scripts/scc/pipeline.sh configs/rq1/yarn_tuned.yaml          # baselines (none/pi/yarn): same call, no training
uv run python -m posext.pipeline status outputs/sweeps/<name>     # check progress / results
```

| # | Job | Runs | Waits for |
|---|---|---|---|
| 1 | `train` (GPU array, one task per setting) | train (skipped for baselines), then PG-19 **validation** PPL and passkey (seed 0) | — |
| 2 | `rulerprep` (CPU) | generate RULER **test** data; only submitted if missing | — |
| 3 | `select` (CPU) | rank settings on validation, write `selection.json` | 1 |
| 4 | `test` (GPU) | PG-19 **test** PPL + passkey (seed 1), winner only | 3 |
| 5 | `ruler` (GPU array, one task per length) | RULER **test** (8 NIAH tasks) at 4k / 8k / 16k in parallel, winner only | 3 (+ 2) |
| 6 | `rulermerge` (CPU) | combine the lengths into one RULER result | 5 |

- **Selection:** the YaRN grid's combined score (val NLL at 16k + val passkey) by default;
  `--metric ppl` or `--metric passkey` to change. Test data is used only by the winner.
  Sweep hyperparameters, not seeds.
- **Where things go:** `outputs/sweeps/<name>/` (plan, generated configs, `runs/<setting>/`,
  `selection.json`, `logs/`); test results in `results/sweeps/<name>/test_{ppl,passkey,ruler}_<setting>.json`.
- **Safe to re-run:** the same command with the same `--name` resubmits and skips finished
  work (training resumes, RULER resumes). A different sweep under an existing name is refused.
- **Failures:** a setting that crashes is listed as missing and selection uses the others. If
  nothing finishes validation, `select` fails and so do 4 and 5.
- **Baselines** (`none`, `pi`, `yarn` configs) go through the same jobs with the same eval
  settings, loaded with `--pos-mode` flags instead of a trained run; a setting those flags
  can't express is refused at submit time.
- **Typical times** (one model): train + validate ~20 min; test PPL + passkey ~5 min; RULER
  test ~3 h at 16k (2.3–2.9 s/sample), much less at 4k / 8k (answers stop early); RULER test
  data generation ~20 min per length, once.
- **More test prompts later:** `bash scripts/scc/pipeline.sh retest outputs/sweeps/* --passkey-n-per-cell 50`
  re-runs test passkey (seed 1) on each sweep's selected model and writes
  `test_passkey_n50_<setting>.json` next to the original; sweeps still running wait for their
  selection. `status` shows the extra result.
- **Options:** `--name`, `--gpu-type A100` (pin one GPU type for final numbers; SCC has few
  A100s, so expect longer waits), `--max-concurrent N`, `--no-ruler`, `--train-hours` (6, or 1
  for baselines), `--test-hours 4`, `--ruler-hours 12` (per length).
  All: `uv run python -m posext.pipeline submit --help`.

**Training curves.** Loss and gradient norm from the training logs (W&B is off in SCC jobs);
matplotlib comes from `uv run --with`, so it is not a project dependency:

```bash
uv run --with matplotlib python -m posext.plot_training outputs/sweeps/temp_*            # panel per sweep
uv run --with matplotlib python -m posext.plot_training logs/train_8576388.log ...       # train.qsub logs
```

Writes `outputs/plots/<name>/`: `loss.png`, `grad_norm.png`, `compare.png` (each sweep's selected
setting), `curves.csv` and `summary.md` (final train loss, val PPL / passkey, selection, GPU, minutes).
Run it on SCC, or copy the sweeps' `logs/` and JSON files locally first (no checkpoints needed).

**Testing the pipeline before a real overnight run.** Each tier catches what the previous one can't:

1. **Unit tests (laptop, seconds):** `uv run pytest tests/test_pipeline.py`. Covers the plan,
   selection, job chaining and skip logic, with qsub and the GPU stages faked.
2. **Dry run (SCC login node, seconds):** add `--dry-run`. Checks the config and sweep keys,
   SCC paths and RULER data status, and prints the exact qsub commands. Submits nothing.
3. **Smoke run (SCC, a few GPU-minutes plus queue time):** every real job and dependency on a tiny budget:

   ```bash
   bash scripts/scc/pipeline.sh configs/rq1/learned_freq_shared.yaml --name smoke_pipeline \
     --sweep train.lr=1e-2,3e-2 --sweep train.steps=20 \
     --ppl-max-sequences 2 --passkey-n-per-cell 1 \
     --ruler-tasks niah_single_1 --ruler-lengths 4096 --ruler-max-samples 20 \
     --train-hours 1 --test-hours 1 --ruler-hours 1
   ```

   It works if, once `qstat -u $USER` is empty:
   - `status outputs/sweeps/smoke_pipeline` shows both settings trained with val numbers, one
     marked selected, and test PPL / passkey / RULER filled in (`complete=True`).
   - For a baseline, run the same check with `configs/rq1/pi.yaml --name smoke_baseline` and the
     same eval flags (no `--sweep`). Use `--ruler-lengths 4096,8192` so the length array has more than one task.
   - `grep -l Traceback outputs/sweeps/smoke_pipeline/logs/*` prints nothing.
   - Re-running the identical command finishes in minutes, with logs saying "Already trained"
     and no new results (proves resubmission is safe).
   - Optional failure check: `qdel <train job id>.1` (task 1 only) while it runs; `select` should list
     that setting as missing and still pick the other.

   Smoke numbers are meaningless (20 steps, tiny evals); delete `outputs/sweeps/smoke_*`
   and `results/sweeps/smoke_*` afterwards. `--ruler-max-samples` is for smoke runs only.

## Layout

```
posext/           # model, data, train, eval, baselines
configs/rq1/      # one YAML per RQ1 cell
scripts/scc/      # qsub templates
tests/            # identity + plumbing tests (CPU)
Documentation/    # proposal + papers
```
