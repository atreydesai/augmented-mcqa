#!/usr/bin/env bash
#SBATCH --job-name=vllm-a6000
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/vllm/%x_%j.log

set -euo pipefail

# vLLM server for Inspect AI.
# Runs vLLM inside Apptainer on an RTX A6000 node and binds the shared
# Hugging Face cache at /fs/clip-scratch/adesai10.
# This script does not activate or modify the project .venv.

PROJECT_DIR="/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa"
HF_ROOT="${HF_ROOT:-/fs/clip-scratch/adesai10}"
SIF="${VLLM_SIF:-/fs/clip-scratch/adesai10/containers/vllm-gemma4.sandbox}"
IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:gemma4}"
MODEL="${VLLM_MODEL:-google/gemma-4-E4B-it}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
API_KEY="${VLLM_API_KEY:-local-dev-key}"
TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
DTYPE="${VLLM_DTYPE:-auto}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-$MODEL}"
EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

mkdir -p "$PROJECT_DIR/logs/slurm/vllm" "$HF_ROOT/hub" "$HF_ROOT/vllm"
cd "$PROJECT_DIR"

if [[ ! -e "$SIF" ]]; then
  echo "Missing Apptainer image/sandbox: $SIF"
  echo "Building local sandbox from: $IMAGE"
  mkdir -p "$(dirname "$SIF")" /fs/clip-scratch/adesai10/apptainer-cache /tmp/adesai10-apptainer-tmp
  APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/fs/clip-scratch/adesai10/apptainer-cache}" \
  APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp/adesai10-apptainer-tmp}" \
    apptainer build --sandbox "$SIF" "$IMAGE"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Warning: HF_TOKEN is not set. Gated Hugging Face models may fail." >&2
fi

echo "vLLM node: $(hostname)"
echo "vLLM URL: http://$(hostname):${PORT}/v1"
echo "Inspect setup:"
echo "  export VLLM_BASE_URL=http://$(hostname):${PORT}/v1"
echo "  export VLLM_API_KEY=${API_KEY}"
echo "  inspect eval <task.py> --model vllm/${SERVED_MODEL_NAME}"
echo
echo "SIF: $SIF"
echo "Model: $MODEL"
echo "HF cache root: $HF_ROOT"
echo "GPU allocation:"
nvidia-smi || true

args=(
  --model "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --api-key "$API_KEY"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --dtype "$DTYPE"
)

if [[ -n "$MAX_MODEL_LEN" ]]; then
  args+=(--max-model-len "$MAX_MODEL_LEN")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args_array=($EXTRA_ARGS)
  args+=("${extra_args_array[@]}")
fi

exec apptainer run --nv \
  --bind "${HF_ROOT}:/root/.cache/huggingface" \
  --bind "${HF_ROOT}/vllm:/root/.cache/vllm" \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  --env "HF_HOME=/root/.cache/huggingface" \
  --env "HF_HUB_CACHE=/root/.cache/huggingface/hub" \
  --env "VLLM_USAGE_STATS_SERVER=" \
  "$SIF" \
  "${args[@]}"
