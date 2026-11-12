#!/bin/bash -l
# One call for an overnight run: train every sweep variant, validate, pick the best on validation,
# then evaluate the winner on the test sets (PG-19 PPL, passkey, RULER). All stages are queued
# at once with -hold_jid dependencies. Run on an SCC LOGIN node (needs qsub):
#
#   bash scripts/scc/pipeline.sh configs/rq1/learned_freq_head.yaml --sweep train.lr=1e-3,1e-2,3e-2
#   bash scripts/scc/pipeline.sh configs/rq1/learned_freq_temp_head.yaml \
#       --sweep train.lr=1e-2,3e-2 --sweep pos.temp.form=adascale,constant --name temp_head_v1
#   bash scripts/scc/pipeline.sh <config> --sweep ... --dry-run     # show plan + qsub commands
#
# Next morning:
#   uv run python -m posext.pipeline status outputs/sweeps/<name>
#
# Options (all optional): --name, --metric combined|ppl|passkey, --gpu-type A100, --no-ruler,
# --max-concurrent N, --train-hours 6, --test-hours 4, --ruler-hours 24, --passkey-n-per-cell 10.
# Full help: uv run python -m posext.pipeline submit --help
# Prerequisites (once): scripts/scc/setup.sh; for RULER also scripts/scc/ruler_setup.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This checkout, not whatever POSEXT_ROOT env.sh would default to.
export POSEXT_ROOT="${POSEXT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh" >/dev/null
cd "$POSEXT_ROOT"

if [[ " $* " != *" --dry-run "* ]] && ! command -v qsub >/dev/null 2>&1; then
  echo "qsub not found: run this on an SCC login node (or add --dry-run)" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  PY=(uv run python)
elif [[ -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
  PY=("${UV_PROJECT_ENVIRONMENT}/bin/python")
else
  echo "No uv and no venv at $UV_PROJECT_ENVIRONMENT — run scripts/scc/setup.sh first" >&2
  exit 1
fi

"${PY[@]}" -m posext.pipeline submit "$@"
