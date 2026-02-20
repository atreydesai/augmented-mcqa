#!/bin/bash
set -euo pipefail

# Orchestrator for local vLLM model evaluation via SLURM array jobs.
#
# Prerequisites (one-time setup, not enforced here):
#   uv sync
#   uv pip install --no-build-isolation 'vllm==0.11.2' 'transformers<5' 'numpy<2.3'
#   huggingface-cli download <model_id> --local-dir /path/to/cache
#
# Submit from repo root. Uses jobs/local_model_eval.sbatch.

usage() {
  cat <<'USAGE'
Usage:
  jobs/run_local_eval.sh --model <model> --generator-dataset-label <label> \
    --dataset-path <path> [options]

Required:
  --model <model>                    Model alias (must match config/model_aliases.toml)
  --generator-dataset-label <label>  Generator label (e.g. gpt-4.1, opus)
  --dataset-path <path>              Path to augmented dataset directory

Options:
  --num-shards <int>         Number of SLURM array shards (default: 8)
  --phase <smoke|main|both>  Which phase(s) to run (default: both)
  --smoke-limit <int>        Entry limit per config in smoke phase (default: 2)
  --preset <name>            Matrix preset: core16 or branching21 (default: core16)
  --dataset-types <csv>      Comma-separated dataset types (default: all)
  --distractor-source <csv>  Comma-separated distractor sources (default: all)
  --output-dir <path>        Results output directory (default: results)
  --save-interval <int>      Checkpoint save interval (default: 200)
  --keep-checkpoints <int>   Number of checkpoints to keep (default: 2)
  --max-tokens <int>         Max tokens for generation (default: 150)
  --help                     Show this help

Examples:
  # Smoke + main run (default):
  jobs/run_local_eval.sh \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --generator-dataset-label gpt-4.1 \
    --dataset-path datasets/augmented/unified_processed_gpt-4.1_20260213_033916

  # Smoke only, 3 shards:
  jobs/run_local_eval.sh \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --generator-dataset-label gpt-4.1 \
    --dataset-path datasets/augmented/unified_processed_gpt-4.1_20260213_033916 \
    --phase smoke --num-shards 3

  # Main only, specific dataset types:
  jobs/run_local_eval.sh \
    --model allenai/Olmo-3-7B-Instruct \
    --generator-dataset-label opus \
    --dataset-path datasets/augmented/unified_processed_opus_20260213_041708 \
    --phase main --num-shards 8 \
    --dataset-types mmlu_pro,gpqa
USAGE
}

# Defaults
MODEL=""
GENERATOR_DATASET_LABEL=""
DATASET_PATH=""
NUM_SHARDS=8
PHASE="both"
SMOKE_LIMIT=2
PRESET="core16"
DATASET_TYPES=""
DISTRACTOR_SOURCES=""
OUTPUT_DIR="results"
SAVE_INTERVAL=200
KEEP_CHECKPOINTS=2
MAX_TOKENS=150

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)                    MODEL="$2";                    shift 2 ;;
    --generator-dataset-label)  GENERATOR_DATASET_LABEL="$2"; shift 2 ;;
    --dataset-path)             DATASET_PATH="$2";             shift 2 ;;
    --num-shards)               NUM_SHARDS="$2";               shift 2 ;;
    --phase)                    PHASE="$2";                    shift 2 ;;
    --smoke-limit)              SMOKE_LIMIT="$2";              shift 2 ;;
    --preset)                   PRESET="$2";                   shift 2 ;;
    --dataset-types)            DATASET_TYPES="$2";            shift 2 ;;
    --distractor-source)        DISTRACTOR_SOURCES="$2";       shift 2 ;;
    --output-dir)               OUTPUT_DIR="$2";               shift 2 ;;
    --save-interval)            SAVE_INTERVAL="$2";            shift 2 ;;
    --keep-checkpoints)         KEEP_CHECKPOINTS="$2";         shift 2 ;;
    --max-tokens)               MAX_TOKENS="$2";               shift 2 ;;
    --help|-h)                  usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

[[ -n "$MODEL" ]]                   || { echo "Error: --model is required"; usage; exit 1; }
[[ -n "$GENERATOR_DATASET_LABEL" ]] || { echo "Error: --generator-dataset-label is required"; usage; exit 1; }
[[ -n "$DATASET_PATH" ]]            || { echo "Error: --dataset-path is required"; usage; exit 1; }

if [[ "$PHASE" != "smoke" && "$PHASE" != "main" && "$PHASE" != "both" ]]; then
  echo "Error: --phase must be smoke, main, or both"
  exit 1
fi

ARRAY_RANGE="0-$((NUM_SHARDS - 1))"

COMMON_EXPORTS=(
  MODEL="$MODEL"
  DATASET_PATH="$DATASET_PATH"
  GENERATOR_DATASET_LABEL="$GENERATOR_DATASET_LABEL"
  NUM_SHARDS="$NUM_SHARDS"
  PRESET="$PRESET"
  OUTPUT_DIR="$OUTPUT_DIR"
  SAVE_INTERVAL="$SAVE_INTERVAL"
  KEEP_CHECKPOINTS="$KEEP_CHECKPOINTS"
  MAX_TOKENS="$MAX_TOKENS"
  DATASET_TYPES="$DATASET_TYPES"
  DISTRACTOR_SOURCES="$DISTRACTOR_SOURCES"
)

submit_phase() {
  local phase_name="$1"
  local limit="$2"

  local export_str
  export_str=$(IFS=','; echo "ALL,${COMMON_EXPORTS[*]},LIMIT=$limit")

  echo ""
  echo "Submitting $phase_name phase"
  echo "  model=$MODEL"
  echo "  generator_dataset_label=$GENERATOR_DATASET_LABEL"
  echo "  dataset_path=$DATASET_PATH"
  echo "  num_shards=$NUM_SHARDS"
  echo "  array=$ARRAY_RANGE"
  [[ -n "$limit" ]] && echo "  limit=$limit (smoke)"

  local job_id
  job_id=$(sbatch \
    --array="$ARRAY_RANGE" \
    --export="$export_str" \
    --parsable \
    jobs/local_model_eval.sbatch)

  echo "Submitted $phase_name job: $job_id"
  echo "  Logs: logs/local_eval/slurm_${job_id}_*.{out,err}"
}

if [[ "$PHASE" == "smoke" || "$PHASE" == "both" ]]; then
  submit_phase "smoke" "$SMOKE_LIMIT"
fi

if [[ "$PHASE" == "main" || "$PHASE" == "both" ]]; then
  submit_phase "main" ""
fi
