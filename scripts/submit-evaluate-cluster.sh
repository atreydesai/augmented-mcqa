#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  run_main submit-evaluate-cluster --help
  exit 0
fi

run_name="${1:-${RUN_NAME:-eval_qwen4b_on_gen_gpt52_v2}}"
generator_run_name="${2:-${GENERATOR_RUN_NAME:-gen_gpt52_v2}}"
generator_model="${3:-${GENERATOR_MODEL:-gpt-5.2-2025-12-11}}"
models="${4:-${MODELS:-Qwen/Qwen3-4B-Instruct-2507,allenai/Olmo-3-7B-Instruct}}"
processed_dataset="${5:-${PROCESSED_DATASET:-datasets/processed/unified_processed_v3}}"
extra_args=("${@:6}")

args=(
  submit-evaluate-cluster
  --run-name "$run_name"
  --generator-run-name "$generator_run_name"
  --generator-model "$generator_model"
  --models "$models"
  --processed-dataset "$processed_dataset"
  --settings "${SETTINGS:-human_from_scratch,model_from_scratch,augment_human,augment_model,augment_ablation}"
  --modes "${MODES:-full_question,choices_only}"
  --questions-per-job "${QUESTIONS_PER_JOB:-200}"
)
append_if_set args --dataset-types "${DATASET_TYPES:-}"
append_if_set args --gpu-count "${GPU_COUNT:-}"
append_if_set args --output-dir "${OUTPUT_DIR:-}"
if [[ "${SUBMIT:-0}" != "1" ]]; then
  args+=(--write-only)
fi

run_main "${args[@]}" "${extra_args[@]}"
