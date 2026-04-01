#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  run_main generate --help
  exit 0
fi

run_name="${1:-${RUN_NAME:-gen_gpt52_v2}}"
model="${2:-${MODEL:-gpt-5.2-2025-12-11}}"
processed_dataset="${3:-${PROCESSED_DATASET:-datasets/processed/unified_processed_v3}}"
extra_args=("${@:4}")

args=(
  generate
  --run-name "$run_name"
  --model "$model"
  --processed-dataset "$processed_dataset"
  --materialize-cache
)
append_if_set args --dataset-types "${DATASET_TYPES:-}"
append_if_set args --generation-strategies "${GENERATION_STRATEGIES:-model_from_scratch,augment_human,augment_model,augment_ablation}"
append_if_set args --limit "${LIMIT:-}"
append_if_set args --question-start "${QUESTION_START:-}"

run_main "${args[@]}" "${extra_args[@]}"
