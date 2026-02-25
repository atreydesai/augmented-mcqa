#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

echo "Submitting final5_pair__gpt-5.2-2025-12-11__Qwen_Qwen3-4B-Instruct-2507.sbatch"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "  sbatch $SCRIPT_DIR/final5_pair__gpt-5.2-2025-12-11__Qwen_Qwen3-4B-Instruct-2507.sbatch"
else
  sbatch "$SCRIPT_DIR/final5_pair__gpt-5.2-2025-12-11__Qwen_Qwen3-4B-Instruct-2507.sbatch"
fi

echo "Submitting final5_pair__gpt-5.2-2025-12-11__allenai_Olmo-3-7B-Instruct.sbatch"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "  sbatch $SCRIPT_DIR/final5_pair__gpt-5.2-2025-12-11__allenai_Olmo-3-7B-Instruct.sbatch"
else
  sbatch "$SCRIPT_DIR/final5_pair__gpt-5.2-2025-12-11__allenai_Olmo-3-7B-Instruct.sbatch"
fi

echo "Submitting final5_pair__gpt-5.2-2025-12-11__meta-llama_Llama-3.1-8B-Instruct.sbatch"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "  sbatch $SCRIPT_DIR/final5_pair__gpt-5.2-2025-12-11__meta-llama_Llama-3.1-8B-Instruct.sbatch"
else
  sbatch "$SCRIPT_DIR/final5_pair__gpt-5.2-2025-12-11__meta-llama_Llama-3.1-8B-Instruct.sbatch"
fi

