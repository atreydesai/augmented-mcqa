#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ARGS=()
[[ -n "${SLURM_PARTITION_OVERRIDE:-}" ]] && SBATCH_ARGS+=(--partition "$SLURM_PARTITION_OVERRIDE")
[[ -n "${SLURM_ACCOUNT_OVERRIDE:-}" ]] && SBATCH_ARGS+=(--account "$SLURM_ACCOUNT_OVERRIDE")
[[ -n "${SLURM_QOS_OVERRIDE:-}" ]] && SBATCH_ARGS+=(--qos "$SLURM_QOS_OVERRIDE")
[[ -n "${SLURM_TIME_OVERRIDE:-}" ]] && SBATCH_ARGS+=(--time "$SLURM_TIME_OVERRIDE")
[[ -n "${SLURM_MEM_OVERRIDE:-}" ]] && SBATCH_ARGS+=(--mem "$SLURM_MEM_OVERRIDE")
[[ -n "${SLURM_CPUS_OVERRIDE:-}" ]] && SBATCH_ARGS+=(--cpus-per-task "$SLURM_CPUS_OVERRIDE")
[[ -n "${SLURM_GRES_OVERRIDE:-}" ]] && SBATCH_ARGS+=(--gres "$SLURM_GRES_OVERRIDE")
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

echo "Submitting final5_pair__gpt-5.2-2025-12-11__Qwen_Qwen3-4B-Instruct-2507.sbatch"
if [[ "$DRY_RUN" == "1" ]]; then
  printf "  sbatch "
  printf "%q " "${SBATCH_ARGS[@]}"
  printf "%q\n" "$SCRIPT_DIR/final5_pair__gpt-5.2-2025-12-11__Qwen_Qwen3-4B-Instruct-2507.sbatch"
else
  sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/final5_pair__gpt-5.2-2025-12-11__Qwen_Qwen3-4B-Instruct-2507.sbatch"
fi

echo "Submitting final5_pair__gpt-5.2-2025-12-11__allenai_Olmo-3-7B-Instruct.sbatch"
if [[ "$DRY_RUN" == "1" ]]; then
  printf "  sbatch "
  printf "%q " "${SBATCH_ARGS[@]}"
  printf "%q\n" "$SCRIPT_DIR/final5_pair__gpt-5.2-2025-12-11__allenai_Olmo-3-7B-Instruct.sbatch"
else
  sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/final5_pair__gpt-5.2-2025-12-11__allenai_Olmo-3-7B-Instruct.sbatch"
fi

echo "Submitting final5_pair__gpt-5.2-2025-12-11__meta-llama_Llama-3.1-8B-Instruct.sbatch"
if [[ "$DRY_RUN" == "1" ]]; then
  printf "  sbatch "
  printf "%q " "${SBATCH_ARGS[@]}"
  printf "%q\n" "$SCRIPT_DIR/final5_pair__gpt-5.2-2025-12-11__meta-llama_Llama-3.1-8B-Instruct.sbatch"
else
  sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/final5_pair__gpt-5.2-2025-12-11__meta-llama_Llama-3.1-8B-Instruct.sbatch"
fi

