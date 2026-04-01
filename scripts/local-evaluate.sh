#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  run_main evaluate --help
  exit 0
fi

run_name="${1:-${RUN_NAME:-eval_qwen4b_on_gen_gpt52_v2}}"
model="${2:-${MODEL:-Qwen/Qwen3-4B-Instruct-2507}}"
generator_run_name="${3:-${GENERATOR_RUN_NAME:-gen_gpt52_v2}}"
generator_model="${4:-${GENERATOR_MODEL:-gpt-5.2-2025-12-11}}"
processed_dataset="${5:-${PROCESSED_DATASET:-datasets/processed/unified_processed_v3}}"
extra_args=("${@:6}")

args=(
  evaluate
  --run-name "$run_name"
  --model "$model"
  --generator-run-name "$generator_run_name"
  --generator-model "$generator_model"
  --processed-dataset "$processed_dataset"
)
append_if_set args --dataset-types "${DATASET_TYPES:-}"
append_if_set args --settings "${SETTINGS:-human_from_scratch,model_from_scratch,augment_human,augment_model,augment_ablation}"
append_if_set args --modes "${MODES:-full_question,choices_only}"
append_if_set args --limit "${LIMIT:-}"
append_if_set args --question-start "${QUESTION_START:-}"

run_main "${args[@]}" "${extra_args[@]}"
