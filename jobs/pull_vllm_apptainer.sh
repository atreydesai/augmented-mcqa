#!/usr/bin/env bash
set -euo pipefail

# Pull a vLLM OpenAI Docker image into an Apptainer SIF.
# This does not touch the project .venv.

IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:gemma4}"
OUT="${VLLM_SIF:-/fs/clip-scratch/adesai10/containers/vllm-gemma4.sif}"
FORCE=0

usage() {
  cat <<'USAGE'
Usage:
  jobs/pull_vllm_apptainer.sh [options]

Options:
  --image <docker-uri>  Docker image to pull
                       default: docker://vllm/vllm-openai:gemma4
  --out <path>          Output .sif path
                       default: /fs/clip-scratch/adesai10/containers/vllm-gemma4.sif
  --force              Replace an existing .sif
  --help               Show this help

Environment overrides:
  VLLM_IMAGE, VLLM_SIF
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --out)
      OUT="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! command -v apptainer >/dev/null 2>&1; then
  echo "Error: apptainer is not on PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")" \
  /fs/clip-scratch/adesai10/apptainer-cache \
  /fs/clip-scratch/adesai10/apptainer-tmp
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/fs/clip-scratch/adesai10/apptainer-cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/fs/clip-scratch/adesai10/apptainer-tmp}"

if [[ -e "$OUT" && "$FORCE" != "1" ]]; then
  echo "SIF already exists: $OUT"
  echo "Use --force to replace it."
  exit 0
fi

echo "Pulling $IMAGE"
echo "Output: $OUT"
echo "Apptainer cache: $APPTAINER_CACHEDIR"
echo "Apptainer tmp: $APPTAINER_TMPDIR"

if [[ "$FORCE" == "1" ]]; then
  apptainer pull --force "$OUT" "$IMAGE"
else
  apptainer pull "$OUT" "$IMAGE"
fi

echo "Done: $OUT"
