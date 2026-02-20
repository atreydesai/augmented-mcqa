#!/bin/bash
#SBATCH --job-name=local_eval
#SBATCH --output=logs/local_eval/slurm_%A_%a.out
#SBATCH --error=logs/local_eval/slurm_%A_%a.err
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtxa6000:1

set -euo pipefail

# Required exports from submit script:
# MODEL, DATASET_PATH, GENERATOR_DATASET_LABEL, NUM_SHARDS

MODEL="${MODEL:?MODEL is required}"
DATASET_PATH="${DATASET_PATH:?DATASET_PATH is required}"
GENERATOR_DATASET_LABEL="${GENERATOR_DATASET_LABEL:?GENERATOR_DATASET_LABEL is required}"
NUM_SHARDS="${NUM_SHARDS:?NUM_SHARDS is required}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID}"

PRESET="${PRESET:-core16}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
EVAL_MODE="${EVAL_MODE:-}"
LIMIT="${LIMIT:-}"
MAX_TOKENS="${MAX_TOKENS:-150}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-2}"
DATASET_TYPES="${DATASET_TYPES:-}"
DISTRACTOR_SOURCES="${DISTRACTOR_SOURCES:-}"

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
mkdir -p logs/local_eval

CMD=(
  uv run python scripts/eval_matrix.py run
  --model "$MODEL"
  --dataset-path "$DATASET_PATH"
  --generator-dataset-label "$GENERATOR_DATASET_LABEL"
  --output-dir "$OUTPUT_DIR"
  --num-shards "$NUM_SHARDS"
  --shard-index "$SHARD_INDEX"
  --skip-existing
)

[[ -n "$PRESET" ]]           && CMD+=(--preset "$PRESET")
[[ -n "$EVAL_MODE" ]]        && CMD+=(--eval-mode "$EVAL_MODE")
[[ -n "$LIMIT" ]]            && CMD+=(--limit "$LIMIT")
[[ -n "$MAX_TOKENS" ]]       && CMD+=(--max-tokens "$MAX_TOKENS")
[[ -n "$SAVE_INTERVAL" ]]    && CMD+=(--save-interval "$SAVE_INTERVAL")
[[ -n "$KEEP_CHECKPOINTS" ]] && CMD+=(--keep-checkpoints "$KEEP_CHECKPOINTS")

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
