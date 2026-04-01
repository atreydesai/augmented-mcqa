#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage: ./scripts/api-smoke.sh [RUN_NAME] [GENERATOR_MODEL] [EVALUATION_MODEL] [PROCESSED_DATASET]
USAGE
  exit 0
fi

if (( $# > 4 )); then
  echo "Unexpected extra arguments. Use scripts/local-generate.sh or scripts/local-evaluate.sh for command-specific flags." >&2
  exit 2
fi

run_name="${1:-${RUN_NAME:-api-smoke}}"
generator_model="${2:-${GENERATOR_MODEL:-gpt-5.2-2025-12-11}}"
evaluation_model="${3:-${EVALUATION_MODEL:-Qwen/Qwen3-4B-Instruct-2507}}"
processed_dataset="${4:-${PROCESSED_DATASET:-datasets/processed/unified_processed_v3}}"

dataset_types="${DATASET_TYPES:-arc_challenge}"
limit="${LIMIT:-1}"
max_tokens="${MAX_TOKENS:-256}"

run_main \
  generate \
  --run-name "$run_name" \
  --model "$generator_model" \
  --processed-dataset "$processed_dataset" \
  --dataset-types "$dataset_types" \
  --generation-strategies "${GENERATION_STRATEGIES:-model_from_scratch}" \
  --limit "$limit" \
  --max-tokens "$max_tokens" \
  --materialize-cache

run_main \
  evaluate \
  --run-name "${run_name}-eval" \
  --model "$evaluation_model" \
  --generator-run-name "$run_name" \
  --generator-model "$generator_model" \
  --processed-dataset "$processed_dataset" \
  --dataset-types "$dataset_types" \
  --settings "${SETTINGS:-model_from_scratch}" \
  --modes "${MODES:-choices_only}" \
  --limit "$limit" \
  --max-tokens "$max_tokens"
