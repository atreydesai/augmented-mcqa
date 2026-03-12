#!/bin/bash
set -euo pipefail

# Edit these variables, then run:
#   bash jobs/submit_evaluate_scheduler.sh

RUN_NAME="eval_scheduler_run"
GENERATOR_RUN_NAME="gen_scheduler_run"
GENERATOR_MODEL="gpt-5.2-2025-12-11"
GENERATOR_BACKEND=""
PROCESSED_DATASET="datasets/processed/unified_processed_v3"
OUTPUT_DIR="jobs/generated/evaluate/${RUN_NAME}"

MODELS="Qwen/Qwen3-4B-Instruct-2507,allenai/Olmo-3-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct"
BACKEND=""
DATASET_TYPES="arc_challenge,mmlu_pro,gpqa"
SETTINGS="human_from_scratch,model_from_scratch,augment_human,augment_model,augment_ablation"
MODES="full_question,choices_only"
QUESTIONS_PER_JOB=""
GPU_COUNT=""

MAX_CONNECTIONS=""
MAX_TOKENS=""
TEMPERATURE=""
REASONING_EFFORT=""
RETRY_ON_ERROR=""
MODEL_BASE_URL=""
STOP_SEQS=""

FORCE=0
RENDER_STATUS=1
WRITE_ONLY=0

PARTITION="clip"
ACCOUNT="clip"
QOS="high"
TIME_LIMIT="12:00:00"
MEMORY="32G"
CPUS_PER_TASK="4"
GPU_TYPE="rtxa6000"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

cmd=(
  uv run python main.py submit-evaluate-cluster
  --run-name "$RUN_NAME"
  --generator-run-name "$GENERATOR_RUN_NAME"
  --generator-model "$GENERATOR_MODEL"
  --processed-dataset "$PROCESSED_DATASET"
  --output-dir "$OUTPUT_DIR"
  --write-only
  --partition "$PARTITION"
  --account "$ACCOUNT"
  --qos "$QOS"
  --time-limit "$TIME_LIMIT"
  --mem "$MEMORY"
  --cpus-per-task "$CPUS_PER_TASK"
  --gpu-type "$GPU_TYPE"
)

[[ -n "$MODELS" ]] && cmd+=(--models "$MODELS")
[[ -n "$BACKEND" ]] && cmd+=(--backend "$BACKEND")
[[ -n "$GENERATOR_BACKEND" ]] && cmd+=(--generator-backend "$GENERATOR_BACKEND")
[[ -n "$DATASET_TYPES" ]] && cmd+=(--dataset-types "$DATASET_TYPES")
[[ -n "$SETTINGS" ]] && cmd+=(--settings "$SETTINGS")
[[ -n "$MODES" ]] && cmd+=(--modes "$MODES")
[[ -n "$QUESTIONS_PER_JOB" ]] && cmd+=(--questions-per-job "$QUESTIONS_PER_JOB")
[[ -n "$GPU_COUNT" ]] && cmd+=(--gpu-count "$GPU_COUNT")
[[ -n "$MAX_CONNECTIONS" ]] && cmd+=(--max-connections "$MAX_CONNECTIONS")
[[ -n "$MAX_TOKENS" ]] && cmd+=(--max-tokens "$MAX_TOKENS")
[[ -n "$TEMPERATURE" ]] && cmd+=(--temperature "$TEMPERATURE")
[[ -n "$REASONING_EFFORT" ]] && cmd+=(--reasoning-effort "$REASONING_EFFORT")
[[ -n "$RETRY_ON_ERROR" ]] && cmd+=(--retry-on-error "$RETRY_ON_ERROR")
[[ -n "$MODEL_BASE_URL" ]] && cmd+=(--model-base-url "$MODEL_BASE_URL")
[[ -n "$STOP_SEQS" ]] && cmd+=(--stop-seqs "$STOP_SEQS")
[[ "$FORCE" == "1" ]] && cmd+=(--force)
[[ "$RENDER_STATUS" == "1" ]] && cmd+=(--render-status)

printf 'Running:'
for token in "${cmd[@]}"; do
  printf ' %q' "$token"
done
printf '\n'

"${cmd[@]}"

submit_script="$(ls -td "$OUTPUT_DIR"/submissions/*/submit_all.sh | head -n1)"
echo "Bundle written to: $submit_script"

if [[ "$WRITE_ONLY" == "1" ]]; then
  exit 0
fi

echo "Submitting: $submit_script"
bash "$submit_script"
