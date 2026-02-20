#!/bin/bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  cat <<'USAGE'
Usage:
  jobs/submit_eval_array.sh <model> <dataset_path> <generator_dataset_label> <num_shards> [options]

Options:
  --preset <name>                Matrix preset (default: eval_matrix.py default)
  --output-dir <path>            Output base directory (default: results)
  --dataset-types <csv>          Comma-separated dataset types
  --distractor-source <csv>      Comma-separated distractor sources
  --limit <int>                  Entry limit per config
  --eval-mode <mode>             accuracy|behavioral (default: eval_matrix.py default)
  --reasoning-effort <value>     OpenAI reasoning effort
  --thinking-level <value>       Anthropic/Gemini thinking level
  --temperature <float>          Generation temperature (default: eval_matrix.py default)
  --max-tokens <int>             Max tokens (default: eval_matrix.py default)
  --save-interval <int>          Checkpoint save interval (default: eval_matrix.py default)
  --keep-checkpoints <int>       Keep newest checkpoints per root (default: eval_matrix.py default)

Example:
  jobs/submit_eval_array.sh gpt-4.1 datasets/augmented/unified_processed my-gen-label 8 \
    --dataset-types mmlu_pro,gpqa --distractor-source scratch,dhuman --limit 200
USAGE
  exit 1
fi

MODEL="$1"
DATASET_PATH="$2"
GENERATOR_DATASET_LABEL="$3"
NUM_SHARDS="$4"
shift 4

PRESET=""
OUTPUT_DIR="results"
DATASET_TYPES=""
DISTRACTOR_SOURCES=""
LIMIT=""
EVAL_MODE=""
REASONING_EFFORT=""
THINKING_LEVEL=""
TEMPERATURE=""
MAX_TOKENS=""
SAVE_INTERVAL=""
KEEP_CHECKPOINTS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset)
      PRESET="$2"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --dataset-types)
      DATASET_TYPES="$2"; shift 2 ;;
    --distractor-source|--distractor-sources)
      DISTRACTOR_SOURCES="$2"; shift 2 ;;
    --limit)
      LIMIT="$2"; shift 2 ;;
    --eval-mode)
      EVAL_MODE="$2"; shift 2 ;;
    --reasoning-effort)
      REASONING_EFFORT="$2"; shift 2 ;;
    --thinking-level)
      THINKING_LEVEL="$2"; shift 2 ;;
    --temperature)
      TEMPERATURE="$2"; shift 2 ;;
    --max-tokens)
      MAX_TOKENS="$2"; shift 2 ;;
    --save-interval)
      SAVE_INTERVAL="$2"; shift 2 ;;
    --keep-checkpoints)
      KEEP_CHECKPOINTS="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1 ;;
  esac
done

if ! [[ "$NUM_SHARDS" =~ ^[0-9]+$ ]] || [[ "$NUM_SHARDS" -le 0 ]]; then
  echo "num_shards must be a positive integer"
  exit 1
fi

ARRAY_RANGE="0-$((NUM_SHARDS - 1))"

echo "Submitting eval matrix array"
echo "  model=$MODEL"
echo "  dataset_path=$DATASET_PATH"
echo "  generator_dataset_label=$GENERATOR_DATASET_LABEL"
echo "  num_shards=$NUM_SHARDS"
echo "  array=$ARRAY_RANGE"

sbatch \
  --array="$ARRAY_RANGE" \
  --export=ALL,MODEL="$MODEL",DATASET_PATH="$DATASET_PATH",GENERATOR_DATASET_LABEL="$GENERATOR_DATASET_LABEL",NUM_SHARDS="$NUM_SHARDS",PRESET="$PRESET",OUTPUT_DIR="$OUTPUT_DIR",DATASET_TYPES="$DATASET_TYPES",DISTRACTOR_SOURCES="$DISTRACTOR_SOURCES",LIMIT="$LIMIT",EVAL_MODE="$EVAL_MODE",REASONING_EFFORT="$REASONING_EFFORT",THINKING_LEVEL="$THINKING_LEVEL",TEMPERATURE="$TEMPERATURE",MAX_TOKENS="$MAX_TOKENS",SAVE_INTERVAL="$SAVE_INTERVAL",KEEP_CHECKPOINTS="$KEEP_CHECKPOINTS" \
  jobs/eval_matrix_array.sbatch
