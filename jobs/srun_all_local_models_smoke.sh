#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa"
RUN_TAG="local_model_smoke_all_$(date -u +%Y%m%d_%H%M%S)"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      REPO_ROOT="${2:-}"; shift 2 ;;
    --run-tag)
      RUN_TAG="${2:-}"; shift 2 ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

models=(
  "Nanbeige/Nanbeige4.1-3B"
  "Qwen/Qwen3-4B-Instruct-2507"
  "allenai/Olmo-3-7B-Instruct"
)

for model in "${models[@]}"; do
  echo "=== Smoke test: $model ==="
  cmd=(
    "$REPO_ROOT/jobs/srun_local_model_smoke.sh"
    --repo-root "$REPO_ROOT"
    --run-tag "$RUN_TAG"
    --model "$model"
  )

  # Nanbeige needs tokenizer_mode=auto in current vLLM runtime.
  if [[ "$model" == "Nanbeige/Nanbeige4.1-3B" ]]; then
    cmd+=(--tokenizer-mode auto)
  fi

  if (( ${#EXTRA_ARGS[@]} > 0 )); then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  "${cmd[@]}"
done

printf '\nAll smoke tests finished. run_tag=%s\n' "$RUN_TAG"
