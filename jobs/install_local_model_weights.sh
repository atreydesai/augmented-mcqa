#!/usr/bin/env bash
set -euo pipefail

# Stage local evaluation model weights into scratch/HF cache.
# Default behavior executes downloads. Use --dry-run to print commands only.

ENV_FILE=".env"
DRY_RUN=0
TARGET_DIR=""

MODELS=(
  "Qwen/Qwen3-4B-Instruct-2507"
  "allenai/Olmo-3-7B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
)

usage() {
  cat <<'USAGE'
Usage:
  jobs/install_local_model_weights.sh [options]

Options:
  --env-file <path>     Path to env file (default: .env)
  --target-dir <path>   Override target cache root
  --dry-run             Print download commands without executing
  --help                Show this help

Notes:
  - Reads MODEL_CACHE_DIR and HF_HOME from .env.
  - If MODEL_CACHE_DIR does not end with /hub, /hub is appended.
  - Downloads are staged into <target>/<model_id_with_slashes_replaced>.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --target-dir)
      TARGET_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

if [[ -z "$TARGET_DIR" ]]; then
  if [[ -n "${MODEL_CACHE_DIR:-}" ]]; then
    TARGET_DIR="$MODEL_CACHE_DIR"
  elif [[ -n "${HF_HOME:-}" ]]; then
    TARGET_DIR="$HF_HOME"
  else
    echo "Error: MODEL_CACHE_DIR/HF_HOME not found and --target-dir not provided"
    exit 1
  fi
fi

if [[ "$(basename "$TARGET_DIR")" != "hub" ]]; then
  TARGET_DIR="${TARGET_DIR%/}/hub"
fi

export HF_HOME="$TARGET_DIR"
mkdir -p "$TARGET_DIR"

echo "Using target cache: $TARGET_DIR"
echo "Dry run: $DRY_RUN"

for model in "${MODELS[@]}"; do
  safe_name="${model//\//__}"
  local_dir="$TARGET_DIR/$safe_name"
  cmd=(
    huggingface-cli
    download
    "$model"
    --local-dir
    "$local_dir"
    --local-dir-use-symlinks
    False
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY RUN: %q ' "${cmd[@]}"
    printf '\n'
  else
    echo "Staging weights for $model"
    "${cmd[@]}"
  fi
done

echo "Done."
