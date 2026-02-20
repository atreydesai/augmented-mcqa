#!/bin/bash
#SBATCH --job-name=aug
#SBATCH --output=logs/augmentation/slurm_%j.out
#SBATCH --error=logs/augmentation/slurm_%j.err
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Usage:
#   sbatch jobs/aug.sh --model gpt-4.1 --input datasets/processed/unified_processed \
#     --output datasets/finished_sets/gpt-4.1 [--limit N] [--save-interval N]
#
# Submit from repo root. All paths are relative to repo root.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
mkdir -p logs/augmentation

MODEL=""
INPUT=""
OUTPUT=""
LIMIT=""
SAVE_INTERVAL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)        MODEL="$2";         shift 2 ;;
    --input)        INPUT="$2";         shift 2 ;;
    --output)       OUTPUT="$2";        shift 2 ;;
    --limit)        LIMIT="$2";         shift 2 ;;
    --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

[[ -n "$MODEL" ]]  || { echo "Error: --model is required"; exit 1; }
[[ -n "$INPUT" ]]  || { echo "Error: --input is required"; exit 1; }
[[ -n "$OUTPUT" ]] || { echo "Error: --output is required"; exit 1; }

CMD=(
  uv run python scripts/generate_distractors.py
  --model "$MODEL"
  --input "$INPUT"
  --output "$OUTPUT"
  --parallel
)

[[ -n "$LIMIT" ]]         && CMD+=(--limit "$LIMIT")
[[ -n "$SAVE_INTERVAL" ]] && CMD+=(--save-interval "$SAVE_INTERVAL")

echo "Running: ${CMD[*]}"
"${CMD[@]}"
