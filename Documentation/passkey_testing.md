# Passkey testing with the same data setup

How a teammate can test specific models on exactly the same passkey prompts as our results.
No new code is needed; it's the existing `posext.eval.passkey` command.

**The same data setup means:**
- **Test seed 1**
- **Defaults kept:** lengths 4k / 8k / 16k, depths 0 / 25 / 50 / 75 / 100%, greedy 16 tokens, bf16. Don't pass `--lengths`, `--depths` or `--max-new-tokens`.
- **Prompt count:** `--n-per-cell 10` to compare with the current table (150 prompts), or `--n-per-cell 50` to match the queued re-test (750 prompts). Results are only comparable at the same count.

**What they need:** the one-time setup (clone, `setup.sh`, `source scripts/scc/env.sh`, weights download). They also need the models, either as trained run folders or as baseline settings:
- **A trained run folder** holds `positional.safetensors` and `positional_config.json`, a few KB. If the models are ours, copy just those two files from my folder, e.g. `outputs/sweeps/temp_head_inityarn/runs/rq1_learned_freq_temp_head__lr1e-2/`, since teammates probably can't read my project folder directly.
- **Baselines** are given as flags. Tuned YaRN is `--pos-mode yarn --factor 4 --beta-fast 64 --beta-slow 2 --attention-factor 1.2`; PI is `--pos-mode pi --factor 4`; the stock model is `--pos-mode none`.

**Option 1: a batch job for several models.** Submit from the repo root on a login node. qsub reads the job script from standard input, so no file is needed:

```bash
cd /projectnb/cs599b1/students/$USER/FrozenRoPE && mkdir -p logs
qsub -P cs599b1 -N passkey_n50 -S /bin/bash -cwd -j y -o logs/ \
  -l h_rt=4:00:00 -l gpus=1 -l gpu_c=8.0 -l gpu_memory=40G -pe omp 8 <<'EOF'
source scripts/scc/env.sh
N=50                                      # 10 = current table, 50 = re-test
OUT=results/passkey_n$N && mkdir -p $OUT
# trained models: "<unique name> <run dir>"
while read NAME RUN; do
  uv run python -m posext.eval.passkey --run "$RUN" --seed 1 --n-per-cell $N --device cuda \
    --out $OUT/test_passkey_n${N}_${NAME}.json
done <<'RUNS'
head_inityarn   outputs/models/head_inityarn
shared_init0    outputs/models/shared_init0
RUNS
# a baseline, e.g. tuned YaRN
uv run python -m posext.eval.passkey --pos-mode yarn --factor 4 --beta-fast 64 --beta-slow 2 \
  --attention-factor 1.2 --seed 1 --n-per-cell $N --device cuda --out $OUT/test_passkey_n${N}_yarn_tuned.json
EOF
```

At 50 per cell, each model takes about 15 min of GPU time, so set `h_rt` to roughly 20 min × the number of models.

**Option 2: interactively**, for one or two models:

```bash
qrsh -P cs599b1 -l gpus=1 -l gpu_c=8.0 -l gpu_memory=40G -l h_rt=2:00:00 -pe omp 4
cd /projectnb/cs599b1/students/$USER/FrozenRoPE && source scripts/scc/env.sh
uv run python -m posext.eval.passkey --run outputs/models/head_inityarn --seed 1 --n-per-cell 50 \
  --device cuda --out results/passkey_n50/test_passkey_n50_head_inityarn.json
```

**Reading the results:** each JSON has `seed`, `n`, overall `accuracy`, `by_length`, `by_depth`, and every prompt's key and generation under `details`. The command also prints a summary.

Three cautions:
- **Use a unique name per model in `--out`.** Run folder names repeat across sweeps (e.g. `rq1_learned_freq_temp_head__lr1e-2` exists in both the from-0 and from-YaRN sweeps), and the second result would overwrite the first.
- **Avoid `eval.qsub` for this.** It runs passkey at the default 5 per cell (not 10 or 50) and reruns test PPL.
- **`configs/rq1/followup/val_ppl.qsub` with `EVAL=passkey SPLIT=test`** uses the right settings at 10 per cell, but it's pinned to A100s (slow to start) and expects models under `outputs/`.
