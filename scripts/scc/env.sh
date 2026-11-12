# SCC paths: keep ALL large artifacts off $HOME (10 GB quota, 11 GB hard limit, shared with
# everything else you do on SCC). Source this before `uv sync`, training, or eval:
#   source /projectnb/cs599b1/students/$USER/FrozenRoPE/scripts/scc/env.sh
#
# Defaults to the current user's student folder, so the same scripts work for every teammate.
# Override with POSEXT_STUDENT_ROOT / POSEXT_ROOT if your layout differs (pass them to qsub
# with -v: SGE does not forward your shell environment).
#
# Layout under projectnb (POSEXT_STUDENT_ROOT):
#   FrozenRoPE/          code + .venv
#   hf_home/             OLMo-2 weights (~6 GB) + HF caches
#   posext_data/         PG-19 chunk cache (+ ruler/ data)
#   RULER/  nltk_data/   NVIDIA RULER checkout + its nltk data
#   uv_cache/            uv package download cache (torch wheels are big)
#   uv_python/ uv_tools/ uv-managed Python interpreters and tools
#   pip_cache/           pip fallback cache
#   torch_home/          torch hub
#   xdg_cache/           anything else that honours XDG_CACHE_HOME (~/.cache otherwise)
#   triton_cache/        Triton kernel cache (~/.triton otherwise)
#   cuda_cache/          CUDA JIT cache (~/.nv/ComputeCache otherwise)
#   wandb/               W&B runs, cache and artifacts
#   tmp/                 scratch for temp files (batch jobs keep SGE's node-local $TMPDIR)

# Do not `set -u` here: this file is meant to be sourced from interactive shells too.

# Cache locations are always set from POSEXT_STUDENT_ROOT, even if your login profile already
# exports them (HW2-style ~/.bashrc lines like `export HF_HOME=...` are common). That keeps
# every teammate's layout identical; a note is printed when an inherited value is replaced.
# Only the POSEXT_* variables (and SGE's TMPDIR) can be overridden from outside.
_posext_set() {
  local _name="$1" _value="$2"
  if [[ -n "${!_name:-}" && "${!_name}" != "$_value" ]]; then
    echo "env.sh: $_name was ${!_name}; using $_value" >&2
  fi
  export "$_name=$_value"
}
POSEXT_STUDENT_ROOT="${POSEXT_STUDENT_ROOT:-/projectnb/cs599b1/students/${USER:-$(id -un)}}"
export POSEXT_ROOT="${POSEXT_ROOT:-${POSEXT_STUDENT_ROOT}/FrozenRoPE}"

_posext_set HF_HOME "${POSEXT_STUDENT_ROOT}/hf_home"
_posext_set HUGGINGFACE_HUB_CACHE "${HF_HOME}/hub"
_posext_set HF_HUB_CACHE "${HF_HOME}/hub"
# TRANSFORMERS_CACHE is deprecated, and transformers 4.57 still uses it as the download dir:
# pointing it anywhere but the hub cache splits the cache and breaks HF_HUB_OFFLINE=1 jobs.
if [[ -n "${TRANSFORMERS_CACHE:-}" ]]; then
  echo "env.sh: unsetting TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} (deprecated; HF_HOME is used)" >&2
  unset TRANSFORMERS_CACHE
fi
export POSEXT_DATA_CACHE="${POSEXT_DATA_CACHE:-${POSEXT_STUDENT_ROOT}/posext_data}"
# NVIDIA RULER checkout + its nltk data (set up once on a login node: scripts/scc/ruler_setup.sh)
export POSEXT_RULER_ROOT="${POSEXT_RULER_ROOT:-${POSEXT_STUDENT_ROOT}/RULER}"
_posext_set NLTK_DATA "${POSEXT_STUDENT_ROOT}/nltk_data"

# uv: venv next to the repo (on projectnb), caches also on projectnb - never ~/.cache/uv
_posext_set UV_PROJECT_ENVIRONMENT "${POSEXT_ROOT}/.venv"
_posext_set UV_CACHE_DIR "${POSEXT_STUDENT_ROOT}/uv_cache"
_posext_set UV_PYTHON_INSTALL_DIR "${POSEXT_STUDENT_ROOT}/uv_python"
_posext_set UV_TOOL_DIR "${POSEXT_STUDENT_ROOT}/uv_tools"
_posext_set UV_TOOL_BIN_DIR "${POSEXT_STUDENT_ROOT}/uv_tools/bin"

_posext_set PIP_CACHE_DIR "${POSEXT_STUDENT_ROOT}/pip_cache"
_posext_set TORCH_HOME "${POSEXT_STUDENT_ROOT}/torch_home"
_posext_set XDG_CACHE_HOME "${POSEXT_STUDENT_ROOT}/xdg_cache"
_posext_set TRITON_CACHE_DIR "${POSEXT_STUDENT_ROOT}/triton_cache"
_posext_set CUDA_CACHE_PATH "${POSEXT_STUDENT_ROOT}/cuda_cache"
_posext_set WANDB_DIR "${POSEXT_STUDENT_ROOT}/wandb"
_posext_set WANDB_CACHE_DIR "${WANDB_DIR}/cache"
_posext_set WANDB_DATA_DIR "${WANDB_DIR}/data"
export TMPDIR="${TMPDIR:-${POSEXT_STUDENT_ROOT}/tmp}"

mkdir -p \
  "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" \
  "$POSEXT_DATA_CACHE" \
  "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_TOOL_DIR" "$UV_TOOL_BIN_DIR" \
  "$PIP_CACHE_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" \
  "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$TMPDIR" "$NLTK_DATA" \
  "${POSEXT_ROOT}/logs" "${POSEXT_ROOT}/results" "${POSEXT_ROOT}/outputs"

# Guardrail: warn if anything is still resolving under $HOME.
_home="${HOME:-}"
if [[ -n "$_home" ]]; then
  for _var in UV_CACHE_DIR UV_PYTHON_INSTALL_DIR HF_HOME POSEXT_DATA_CACHE PIP_CACHE_DIR TORCH_HOME \
      XDG_CACHE_HOME TRITON_CACHE_DIR CUDA_CACHE_PATH WANDB_DIR WANDB_CACHE_DIR WANDB_DATA_DIR \
      UV_PROJECT_ENVIRONMENT POSEXT_RULER_ROOT NLTK_DATA; do
    _val="${!_var}"
    case "$_val" in
      "$_home"|"$_home"/*)
        echo "WARNING: $_var=$_val is under \$HOME - risk of quota. Fix scripts/scc/env.sh." >&2
        ;;
    esac
  done
fi

echo "POSEXT_ROOT=$POSEXT_ROOT"
echo "UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT"
echo "UV_CACHE_DIR=$UV_CACHE_DIR"
echo "HF_HOME=$HF_HOME"
echo "POSEXT_DATA_CACHE=$POSEXT_DATA_CACHE"
