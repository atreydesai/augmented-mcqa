#!/bin/bash
set -euo pipefail

# Sequential evaluation orchestration.
# For sharded SLURM runs, prefer jobs/eval_matrix_array.sbatch.

MODEL="${MODEL:-gpt-4.1}"
DATASET_PATH="${DATASET_PATH:-datasets/augmented/unified_processed}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
LIMIT="${LIMIT:-}"
PRESET="${PRESET:-}"
EVAL_MODE="${EVAL_MODE:-}"
GENERATOR_DATASET_LABEL="${GENERATOR_DATASET_LABEL:-default}"
TEMPERATURE="${TEMPERATURE:-}"
MAX_TOKENS="${MAX_TOKENS:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-}"

DATASET_TYPES=(mmlu_pro gpqa arc_easy arc_challenge)
DISTRACTOR_SOURCES=(scratch dhuman dmodel)

echo "Running sequential eval matrix"
echo "  Model: $MODEL"
echo "  Dataset path: $DATASET_PATH"
echo "  Output dir: $OUTPUT_DIR"
echo "  Preset: ${PRESET:-<eval_matrix default>}"
echo "  Eval mode: ${EVAL_MODE:-<eval_matrix default>}"
echo "  Generator dataset label: $GENERATOR_DATASET_LABEL"
echo "  Temperature: ${TEMPERATURE:-<eval_matrix default>}"
echo "  Max tokens: ${MAX_TOKENS:-<eval_matrix default>}"
echo "  Save interval: ${SAVE_INTERVAL:-<eval_matrix default>}"
echo "  Keep checkpoints: ${KEEP_CHECKPOINTS:-<eval_matrix default>}"

CMD=(
  uv run python scripts/eval_matrix.py run
  --model "$MODEL"
  --dataset-path "$DATASET_PATH"
  --generator-dataset-label "$GENERATOR_DATASET_LABEL"
  --dataset-types "${DATASET_TYPES[@]}"
  --distractor-source "${DISTRACTOR_SOURCES[@]}"
  --output-dir "$OUTPUT_DIR"
  --skip-existing
)

if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi
if [[ -n "$PRESET" ]]; then
  CMD+=(--preset "$PRESET")
fi
if [[ -n "$EVAL_MODE" ]]; then
  CMD+=(--eval-mode "$EVAL_MODE")
fi
if [[ -n "$TEMPERATURE" ]]; then
  CMD+=(--temperature "$TEMPERATURE")
fi
if [[ -n "$MAX_TOKENS" ]]; then
  CMD+=(--max-tokens "$MAX_TOKENS")
fi
if [[ -n "$SAVE_INTERVAL" ]]; then
  CMD+=(--save-interval "$SAVE_INTERVAL")
fi
if [[ -n "$KEEP_CHECKPOINTS" ]]; then
  CMD+=(--keep-checkpoints "$KEEP_CHECKPOINTS")
fi

printf 'Command: %q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "Done."
echo "For SLURM arrays, use jobs/eval_matrix_array.sbatch + jobs/submit_eval_array.sh"
