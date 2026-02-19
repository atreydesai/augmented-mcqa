#!/bin/bash
set -euo pipefail

# Sequential evaluation orchestration.
# For sharded SLURM runs, prefer jobs/eval_matrix_array.sbatch.

MODEL="${MODEL:-gpt-4.1}"
DATASET_PATH="${DATASET_PATH:-datasets/augmented/unified_processed}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
LIMIT="${LIMIT:-}"
EVAL_MODE="${EVAL_MODE:-behavioral}"

DATASET_TYPES=(mmlu_pro gpqa arc_easy arc_challenge)
DISTRACTOR_SOURCES=(scratch dhuman dmodel)

echo "Running sequential eval matrix"
echo "  Model: $MODEL"
echo "  Dataset path: $DATASET_PATH"
echo "  Output dir: $OUTPUT_DIR"
echo "  Eval mode: $EVAL_MODE"

CMD=(
  uv run python scripts/eval_matrix.py run
  --preset core16
  --model "$MODEL"
  --dataset-path "$DATASET_PATH"
  --dataset-types "${DATASET_TYPES[@]}"
  --distractor-source "${DISTRACTOR_SOURCES[@]}"
  --eval-mode "$EVAL_MODE"
  --output-dir "$OUTPUT_DIR"
  --skip-existing
)

if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi

printf 'Command: %q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "Done."
echo "For SLURM arrays, use jobs/eval_matrix_array.sbatch + jobs/submit_eval_array.sh"
