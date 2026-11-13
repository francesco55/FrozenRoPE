#!/bin/bash -l
# One-time RULER setup. Run on a LOGIN node (needs internet; light CPU only):
#   bash scripts/scc/ruler_setup.sh
#
# Clones NVIDIA RULER at the pinned commit into $POSEXT_RULER_ROOT, downloads the Paul Graham
# essays (NIAH haystack) and nltk punkt data into $NLTK_DATA, and caches the OLMo-2 tokenizer.
# Data generation itself is CPU-heavy, so it runs as a batch job (login-node processes are
# killed after ~15 CPU-minutes): qsub scripts/scc/ruler_prepare.qsub

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
cd "$POSEXT_ROOT"

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --extra ruler python)
elif [[ -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
  PY=("${UV_PROJECT_ENVIRONMENT}/bin/python")
else
  echo "No uv and no venv at $UV_PROJECT_ENVIRONMENT — run scripts/scc/setup.sh first" >&2
  exit 1
fi

"${PY[@]}" -m posext.eval.ruler setup

echo
echo "Next: qsub -v SPLIT=validation scripts/scc/ruler_prepare.qsub"
echo "      qsub -v SPLIT=test       scripts/scc/ruler_prepare.qsub"
