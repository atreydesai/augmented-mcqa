#!/bin/bash
#SBATCH --job-name=final5_pair__gpt-5.2-2025-12-11__Qwen_Qwen3-4B-Instruct-2507
#SBATCH --output=logs/final5_eval/final5_pair__gpt-5.2-2025-12-11__Qwen_Qwen3-4B-Instruct-2507_%j.out
#SBATCH --error=logs/final5_eval/final5_pair__gpt-5.2-2025-12-11__Qwen_Qwen3-4B-Instruct-2507_%j.err
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtxa6000:1

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
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_MANIFEST_FILE="${RUN_MANIFEST_FILE:-jobs/generated/final5_full_gpt-5.2-2025-12-11_20260225_210132/final5_pair__gpt-5.2-2025-12-11__Qwen_Qwen3-4B-Instruct-2507.run_manifest.json}"

CMD=(
  "$PYTHON_BIN" scripts/eval_matrix.py run
  --manifest "$RUN_MANIFEST_FILE"
  --generator-dataset-label "$GENERATOR_LABEL"
  --save-interval "$SAVE_INTERVAL"
  --skip-existing
)

printf 'Running pair job with %s work-units\n' "10"
printf 'Run manifest: %s\n' "$RUN_MANIFEST_FILE"
printf 'Running: %q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
