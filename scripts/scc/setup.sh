#!/bin/bash -l
# One-time (or after pulling) SCC setup. Run on a login node:
#   bash scripts/scc/setup.sh
#
# Installs the venv + deps under projectnb. Does NOT use $HOME for caches.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

cd "$POSEXT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv into $UV_TOOL_BIN_DIR (not \$HOME)..."
  curl -LsSf https://astral.sh/uv/install.sh | \
    UV_INSTALL_DIR="$UV_TOOL_BIN_DIR" sh
  export PATH="${UV_TOOL_BIN_DIR}:$PATH"
fi

echo "Installing project deps into $UV_PROJECT_ENVIRONMENT ..."
uv sync --all-extras

echo
echo "Done. Quick checks:"
du -sh "$UV_PROJECT_ENVIRONMENT" "$UV_CACHE_DIR" "$HF_HOME" 2>/dev/null || true
echo
echo "Home usage (should stay small):"
du -sh "$HOME" 2>/dev/null || true
quota -s 2>/dev/null || true
echo
echo "Next: qsub scripts/scc/yarn_grid.qsub"
echo "Always: source $SCRIPT_DIR/env.sh   before interactive uv/python work."
