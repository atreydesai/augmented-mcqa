#!/bin/bash
#SBATCH --job-name=final5_pair__gpt-5.2-2025-12-11__Qwen_Qwen3-4B-Instruct-2507
#SBATCH --output=${LOG_DIR:-logs/final5_eval}/final5_pair__gpt-5.2-2025-12-11__Qwen_Qwen3-4B-Instruct-2507_%A_%a.out
#SBATCH --error=${LOG_DIR:-logs/final5_eval}/final5_pair__gpt-5.2-2025-12-11__Qwen_Qwen3-4B-Instruct-2507_%A_%a.err
#SBATCH --array=0-9
#SBATCH --partition=${SLURM_PARTITION_OVERRIDE:-clip}
#SBATCH --account=${SLURM_ACCOUNT_OVERRIDE:-clip}
#SBATCH --qos=${SLURM_QOS_OVERRIDE:-high}
#SBATCH --time=${SLURM_TIME_OVERRIDE:-12:00:00}
#SBATCH --mem=${SLURM_MEM_OVERRIDE:-32G}
#SBATCH --cpus-per-task=${SLURM_CPUS_OVERRIDE:-4}
#SBATCH --gres=${SLURM_GRES_OVERRIDE:-gpu:rtxa6000:1}

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"
DEFAULT_VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
if [[ ! -f "$DEFAULT_VENV_ACTIVATE" ]]; then
  DEFAULT_VENV_ACTIVATE="/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/.venv/bin/activate"
fi
VENV_ACTIVATE="${VENV_ACTIVATE:-$DEFAULT_VENV_ACTIVATE}"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Error: venv activate script not found: $VENV_ACTIVATE"
  exit 1
fi
source "$VENV_ACTIVATE"
PYTHON_BIN="${PYTHON_BIN:-$(dirname "$VENV_ACTIVATE")/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Error: python executable not found after activating venv."
  exit 1
fi
mkdir -p "${LOG_DIR:-logs/final5_eval}"

GENERATOR_LABEL="gpt-5.2-2025-12-11"
GENERATOR_DATASET_PATH="/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/datasets/augmented/final5_smoke_20260225_181039"
EVAL_MODEL="Qwen/Qwen3-4B-Instruct-2507"
OUTPUT_BASE="${OUTPUT_BASE:-/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/results/final5_smoke_20260225_181039}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2}"
MAX_TOKENS="${MAX_TOKENS:-32}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_UNITS_FILE="${WORK_UNITS_FILE:-$SCRIPT_DIR/final5_pair__gpt-5.2-2025-12-11__Qwen_Qwen3-4B-Instruct-2507.work_units.json}"

readarray -t UNIT_EXPORTS < <("$PYTHON_BIN" - "$WORK_UNITS_FILE" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json
import shlex
import sys

units = json.loads(open(sys.argv[1], encoding='utf-8').read())
idx = int(sys.argv[2])
if idx < 0 or idx >= len(units):
    raise SystemExit(f"Invalid SLURM_ARRAY_TASK_ID={idx} for work_units size={len(units)}")
unit = units[idx]
for key in ("mode", "dataset", "entry_shards", "entry_shard_index", "entry_shard_strategy", "choices_only", "expected_rows"):
    val = unit[key]
    print(f"{key.upper()}={shlex.quote(str(val))}")
PY
)

for kv in "${UNIT_EXPORTS[@]}"; do
  eval "$kv"
done

CMD=(
  "$PYTHON_BIN" scripts/eval_matrix.py run
  --preset final5
  --model "$EVAL_MODEL"
  --dataset-path "$GENERATOR_DATASET_PATH"
  --generator-dataset-label "$GENERATOR_LABEL"
  --dataset-types "$DATASET"
  --output-dir "$OUTPUT_BASE"
  --entry-shards "$ENTRY_SHARDS"
  --entry-shard-index "$ENTRY_SHARD_INDEX"
  --entry-shard-strategy "$ENTRY_SHARD_STRATEGY"
  --save-interval "$SAVE_INTERVAL"
  --max-tokens "$MAX_TOKENS"
  --skip-existing
)

if [[ "$CHOICES_ONLY" == "True" || "$CHOICES_ONLY" == "true" || "$CHOICES_ONLY" == "1" ]]; then
  CMD+=(--choices-only)
fi

printf 'Running task %s | mode=%s dataset=%s part=%s/%s expected_rows=%s\n'   "$SLURM_ARRAY_TASK_ID" "$MODE" "$DATASET" "$ENTRY_SHARD_INDEX" "$ENTRY_SHARDS" "$EXPECTED_ROWS"
printf 'Running: %q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
