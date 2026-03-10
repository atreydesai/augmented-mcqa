#!/bin/bash
#SBATCH --job-name=final5_evaluate
#SBATCH --output=logs/final5_evaluate_%A_%a.out
#SBATCH --error=logs/final5_evaluate_%A_%a.err
#SBATCH --array=0-0

set -euo pipefail

IFS=',' read -r -a MODELS <<< "${MODEL_LIST_CSV:?set MODEL_LIST_CSV}"
SHARD_COUNT="${SHARD_COUNT:?set SHARD_COUNT}"
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / SHARD_COUNT))
SHARD_INDEX=$((SLURM_ARRAY_TASK_ID % SHARD_COUNT))

MODEL="${MODELS[$MODEL_INDEX]}"

bash jobs/run_evaluate_shard.sh \
  --model "$MODEL" \
  --run-name "${RUN_NAME:?set RUN_NAME}" \
  --generator-run-name "${GENERATOR_RUN_NAME:?set GENERATOR_RUN_NAME}" \
  --generator-model "${GENERATOR_MODEL:?set GENERATOR_MODEL}" \
  --processed-dataset "${PROCESSED_DATASET:-datasets/processed/unified_processed_v2}" \
  --shard-count "$SHARD_COUNT" \
  --shard-index "$SHARD_INDEX"
