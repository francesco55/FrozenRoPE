# SCC paths: keep ALL large artifacts off $HOME (~10 GB quota).
# Source this before `uv sync`, training, or eval:
#   source /projectnb/cs599b1/students/adithyad/FrozenRoPE/scripts/scc/env.sh
#
# Layout under projectnb:
#   FrozenRoPE/          code + .venv
#   hf_home/             OLMo-2 weights (~6 GB) + HF caches
#   posext_data/         PG-19 chunk cache
#   uv_cache/            uv package download cache (torch wheels are big)
#   pip_cache/           pip fallback cache
#   torch_home/          torch hub
#   wandb/               W&B runs
#   tmp/                 scratch for temp files

# Do not `set -u` here: this file is meant to be sourced from interactive shells too.
POSEXT_STUDENT_ROOT="${POSEXT_STUDENT_ROOT:-/projectnb/cs599b1/students/adithyad}"
export POSEXT_ROOT="${POSEXT_ROOT:-${POSEXT_STUDENT_ROOT}/FrozenRoPE}"

export HF_HOME="${HF_HOME:-${POSEXT_STUDENT_ROOT}/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export POSEXT_DATA_CACHE="${POSEXT_DATA_CACHE:-${POSEXT_STUDENT_ROOT}/posext_data}"

# uv: venv next to the repo (on projectnb), caches also on projectnb — never ~/.cache/uv
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${POSEXT_ROOT}/.venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${POSEXT_STUDENT_ROOT}/uv_cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${POSEXT_STUDENT_ROOT}/uv_python}"
export UV_TOOL_DIR="${UV_TOOL_DIR:-${POSEXT_STUDENT_ROOT}/uv_tools}"
export UV_TOOL_BIN_DIR="${UV_TOOL_BIN_DIR:-${POSEXT_STUDENT_ROOT}/uv_tools/bin}"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${POSEXT_STUDENT_ROOT}/pip_cache}"
export TORCH_HOME="${TORCH_HOME:-${POSEXT_STUDENT_ROOT}/torch_home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${POSEXT_STUDENT_ROOT}/xdg_cache}"
export WANDB_DIR="${WANDB_DIR:-${POSEXT_STUDENT_ROOT}/wandb}"
export TMPDIR="${TMPDIR:-${POSEXT_STUDENT_ROOT}/tmp}"

mkdir -p \
  "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" \
  "$POSEXT_DATA_CACHE" \
  "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_TOOL_DIR" "$UV_TOOL_BIN_DIR" \
  "$PIP_CACHE_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME" "$WANDB_DIR" "$TMPDIR" \
  "${POSEXT_ROOT}/logs" "${POSEXT_ROOT}/results" "${POSEXT_ROOT}/outputs"

# Guardrail: warn if anything is still resolving under $HOME/cache-like paths.
_home="${HOME:-}"
if [[ -n "$_home" ]]; then
  for _var in UV_CACHE_DIR HF_HOME POSEXT_DATA_CACHE PIP_CACHE_DIR TORCH_HOME XDG_CACHE_HOME WANDB_DIR UV_PROJECT_ENVIRONMENT; do
    _val="${!_var}"
    case "$_val" in
      "$_home"|"$_home"/*)
        echo "WARNING: $_var=$_val is under \$HOME — risk of quota. Fix scripts/scc/env.sh." >&2
        ;;
    esac
  done
fi

echo "POSEXT_ROOT=$POSEXT_ROOT"
echo "UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT"
echo "UV_CACHE_DIR=$UV_CACHE_DIR"
echo "HF_HOME=$HF_HOME"
echo "POSEXT_DATA_CACHE=$POSEXT_DATA_CACHE"
