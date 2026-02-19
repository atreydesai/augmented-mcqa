#!/bin/bash
#SBATCH --job-name=eval_matrix
#SBATCH --output=logs/eval_matrix/slurm_%A_%a.out
#SBATCH --error=logs/eval_matrix/slurm_%A_%a.err
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=medium
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail

# Required exports from submit script:
# MODEL, DATASET_PATH, NUM_SHARDS

MODEL="${MODEL:?MODEL is required}"
DATASET_PATH="${DATASET_PATH:?DATASET_PATH is required}"
NUM_SHARDS="${NUM_SHARDS:?NUM_SHARDS is required}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID}"

PRESET="${PRESET:-core16}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
EVAL_MODE="${EVAL_MODE:-behavioral}"
LIMIT="${LIMIT:-}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
THINKING_LEVEL="${THINKING_LEVEL:-}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-100}"
DATASET_TYPES="${DATASET_TYPES:-}"
DISTRACTOR_SOURCES="${DISTRACTOR_SOURCES:-}"

conda deactivate 2>/dev/null || true
cd /fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa
source .venv/bin/activate
mkdir -p logs/eval_matrix

CMD=(
  uv run python scripts/eval_matrix.py run
  --preset "$PRESET"
  --model "$MODEL"
  --dataset-path "$DATASET_PATH"
  --eval-mode "$EVAL_MODE"
  --temperature "$TEMPERATURE"
  --max-tokens "$MAX_TOKENS"
  --output-dir "$OUTPUT_DIR"
  --num-shards "$NUM_SHARDS"
  --shard-index "$SHARD_INDEX"
  --skip-existing
)

if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi
if [[ -n "$REASONING_EFFORT" ]]; then
  CMD+=(--reasoning-effort "$REASONING_EFFORT")
fi
if [[ -n "$THINKING_LEVEL" ]]; then
  CMD+=(--thinking-level "$THINKING_LEVEL")
fi
if [[ -n "$DATASET_TYPES" ]]; then
  IFS=',' read -r -a DATASET_TYPE_ARR <<< "$DATASET_TYPES"
  CMD+=(--dataset-types "${DATASET_TYPE_ARR[@]}")
fi
if [[ -n "$DISTRACTOR_SOURCES" ]]; then
  IFS=',' read -r -a DISTRACTOR_SOURCE_ARR <<< "$DISTRACTOR_SOURCES"
  CMD+=(--distractor-source "${DISTRACTOR_SOURCE_ARR[@]}")
fi

printf 'Running shard %s/%s with command: %q ' "$SHARD_INDEX" "$((NUM_SHARDS - 1))" "${CMD[@]}"
printf '\n'

"${CMD[@]}"
