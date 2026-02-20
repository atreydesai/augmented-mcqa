#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Submit a single local-model smoke test via srun (1 GPU).

Usage:
  jobs/srun_local_model_smoke.sh --model <model_name> [options]

Required:
  --model <name>                Model alias/id (e.g. Nanbeige/Nanbeige4.1-3B)

Options:
  --run-tag <tag>               Log tag (default: local_model_smoke_<utc>)
  --repo-root <path>            Repo path (default: /fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa)
  --partition <name>            SLURM partition (default: clip)
  --account <name>              SLURM account (default: clip)
  --qos <name>                  SLURM qos (default: high)
  --gpu <gres>                  GPU request (default: gpu:rtxa6000:1)
  --time <hh:mm:ss>             Time limit (default: 00:30:00)
  --mem <value>                 Memory (default: 32G)
  --cpus <int>                  CPUs per task (default: 4)
  --max-tokens <int>            Generation max tokens (default: 64)
  --temperature <float>         Generation temperature (default: 0.0)
  --gpu-memory-util <float>     vLLM gpu memory utilization (default: 0.9)
  --tp-size <int>               Tensor parallel size (default: 1)
  --dtype <name>                Optional dtype override
  --max-model-len <int>         Optional max model len override
  --tokenizer-mode <mode>       Optional tokenizer mode override (auto|slow)
  --stop-token-id <int>         Optional stop token id (repeatable)
  --no-local-snapshot           Disable local snapshot resolution
  --no-sync                     Skip uv sync before run
  --help                        Show this help

Examples:
  jobs/srun_local_model_smoke.sh --model Nanbeige/Nanbeige4.1-3B --tokenizer-mode auto
  jobs/srun_local_model_smoke.sh --model Qwen/Qwen3-4B-Instruct-2507 --dtype bfloat16 --max-model-len 32768
USAGE
}

MODEL=""
RUN_TAG="local_model_smoke_$(date -u +%Y%m%d_%H%M%S)"
REPO_ROOT="/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa"
PARTITION="clip"
ACCOUNT="clip"
QOS="high"
GPU_GRES="gpu:rtxa6000:1"
TIME_LIMIT="00:30:00"
MEMORY="32G"
CPUS=4
MAX_TOKENS=64
TEMPERATURE=0.0
GPU_MEMORY_UTIL=0.9
TP_SIZE=1
DTYPE=""
MAX_MODEL_LEN=""
TOKENIZER_MODE=""
USE_LOCAL_SNAPSHOT=1
DO_SYNC=1
VLLM_INSTALL_SPEC="${VLLM_INSTALL_SPEC:-vllm==0.11.2}"
VLLM_TRANSFORMERS_SPEC="${VLLM_TRANSFORMERS_SPEC:-transformers<5}"
VLLM_NUMPY_SPEC="${VLLM_NUMPY_SPEC:-numpy<2.3}"
declare -a STOP_TOKEN_IDS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:-}"; shift 2 ;;
    --run-tag) RUN_TAG="${2:-}"; shift 2 ;;
    --repo-root) REPO_ROOT="${2:-}"; shift 2 ;;
    --partition) PARTITION="${2:-}"; shift 2 ;;
    --account) ACCOUNT="${2:-}"; shift 2 ;;
    --qos) QOS="${2:-}"; shift 2 ;;
    --gpu) GPU_GRES="${2:-}"; shift 2 ;;
    --time) TIME_LIMIT="${2:-}"; shift 2 ;;
    --mem) MEMORY="${2:-}"; shift 2 ;;
    --cpus) CPUS="${2:-}"; shift 2 ;;
    --max-tokens) MAX_TOKENS="${2:-}"; shift 2 ;;
    --temperature) TEMPERATURE="${2:-}"; shift 2 ;;
    --gpu-memory-util) GPU_MEMORY_UTIL="${2:-}"; shift 2 ;;
    --tp-size) TP_SIZE="${2:-}"; shift 2 ;;
    --dtype) DTYPE="${2:-}"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="${2:-}"; shift 2 ;;
    --tokenizer-mode) TOKENIZER_MODE="${2:-}"; shift 2 ;;
    --stop-token-id) STOP_TOKEN_IDS+=("${2:-}"); shift 2 ;;
    --no-local-snapshot) USE_LOCAL_SNAPSHOT=0; shift ;;
    --no-sync) DO_SYNC=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "--model is required" >&2
  usage
  exit 1
fi

safe_name() {
  printf '%s' "$1" | tr '/ .:' '_' | tr -cd '[:alnum:]_-'
}

MODEL_SAFE="$(safe_name "$MODEL")"
LOG_DIR="$REPO_ROOT/logs/slurm/local_model_smoke/$RUN_TAG"
TMP_DIR="$REPO_ROOT/results/local_eval/$RUN_TAG/tmp_smoke_jobs"
mkdir -p "$LOG_DIR" "$TMP_DIR"

JOB_SCRIPT="$TMP_DIR/${MODEL_SAFE}.sh"

cat > "$JOB_SCRIPT" <<EOS
#!/usr/bin/env bash
set -euo pipefail

if [[ -f "\$HOME/.bashrc" ]]; then
  # shellcheck disable=SC1090
  source "\$HOME/.bashrc" || true
fi

cd "$REPO_ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export UV_LINK_MODE="\${UV_LINK_MODE:-copy}"
export MODEL_CACHE_DIR="/fs/nexus-scratch/adesai10/hub"
export HF_HOME="/fs/nexus-scratch/adesai10/hub"
export HF_DATASETS_CACHE="/fs/nexus-scratch/adesai10/hub/datasets"
export TRANSFORMERS_CACHE="/fs/nexus-scratch/adesai10/hub/transformers"
export PYTHONUNBUFFERED=1
export UV_NO_SYNC=1

if [[ "$DO_SYNC" == "1" ]]; then
  uv sync --inexact
fi

if ! uv run --no-sync python - <<'PY'
import importlib.util
import sys
if importlib.util.find_spec("vllm") is None:
    sys.exit(1)
try:
    import transformers  # type: ignore
    major = int(str(transformers.__version__).split(".")[0])
except Exception:
    sys.exit(2)
if major >= 5:
    sys.exit(3)
try:
    import numpy as np  # type: ignore
    parts = str(np.__version__).split(".")
    n_major = int(parts[0])
    n_minor = int(parts[1]) if len(parts) > 1 else 0
except Exception:
    sys.exit(4)
sys.exit(0 if (n_major < 2 or (n_major == 2 and n_minor < 3)) else 5)
PY
then
  uv pip install --only-binary=:all: "$VLLM_INSTALL_SPEC" "$VLLM_TRANSFORMERS_SPEC" "$VLLM_NUMPY_SPEC"
fi

cmd=(
  uv run --no-sync python scripts/smoke_local_model.py
  --model "$MODEL"
  --max-tokens "$MAX_TOKENS"
  --temperature "$TEMPERATURE"
  --gpu-memory-utilization "$GPU_MEMORY_UTIL"
  --tensor-parallel-size "$TP_SIZE"
)

if [[ -n "$DTYPE" ]]; then
  cmd+=(--dtype "$DTYPE")
fi
if [[ -n "$MAX_MODEL_LEN" ]]; then
  cmd+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [[ -n "$TOKENIZER_MODE" ]]; then
  cmd+=(--tokenizer-mode "$TOKENIZER_MODE")
fi
if [[ "$USE_LOCAL_SNAPSHOT" == "0" ]]; then
  cmd+=(--no-use-local-snapshot)
fi
EOS

for token_id in "${STOP_TOKEN_IDS[@]:-}"; do
  if [[ -n "$token_id" ]]; then
    printf 'cmd+=(--stop-token-id "%s")\n' "$token_id" >> "$JOB_SCRIPT"
  fi
done

cat >> "$JOB_SCRIPT" <<'EOS'
"${cmd[@]}"
EOS

chmod +x "$JOB_SCRIPT"

OUT_LOG="$LOG_DIR/${MODEL_SAFE}_%j.out"
ERR_LOG="$LOG_DIR/${MODEL_SAFE}_%j.err"

echo "Submitting smoke test via srun"
echo "  model: $MODEL"
echo "  logs:  $LOG_DIR"

action_cmd=(
  srun
  --partition="$PARTITION"
  --account="$ACCOUNT"
  --qos="$QOS"
  --gres="$GPU_GRES"
  --time="$TIME_LIMIT"
  --mem="$MEMORY"
  --cpus-per-task="$CPUS"
  --job-name="qgqa_smoke_${MODEL_SAFE}"
  --output "$OUT_LOG"
  --error "$ERR_LOG"
  "$JOB_SCRIPT"
)

"${action_cmd[@]}"

echo "Smoke run complete for $MODEL"
