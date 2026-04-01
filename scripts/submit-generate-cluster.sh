#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  run_main submit-generate-cluster --help
  exit 0
fi

run_name="${1:-${RUN_NAME:-gen_gpt52_v2}}"
models="${2:-${MODELS:-gpt-5.2-2025-12-11}}"
processed_dataset="${3:-${PROCESSED_DATASET:-datasets/processed/unified_processed_v3}}"
extra_args=("${@:4}")

args=(
  submit-generate-cluster
  --run-name "$run_name"
  --models "$models"
  --processed-dataset "$processed_dataset"
  --generation-strategies "${GENERATION_STRATEGIES:-model_from_scratch,augment_human,augment_model,augment_ablation}"
  --questions-per-job "${QUESTIONS_PER_JOB:-200}"
)
append_if_set args --dataset-types "${DATASET_TYPES:-}"
append_if_set args --gpu-count "${GPU_COUNT:-}"
append_if_set args --output-dir "${OUTPUT_DIR:-}"
if [[ "${SUBMIT:-0}" != "1" ]]; then
  args+=(--write-only)
fi

run_main "${args[@]}" "${extra_args[@]}"
