#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  run_main prepare-data --help
  exit 0
fi

output_path="${1:-${PROCESSED_DATASET:-datasets/processed/unified_processed_v3}}"
extra_args=("${@:2}")

args=(
  prepare-data
  --step "${STEP:-all}"
  --output-path "$output_path"
)
append_if_set args --dataset "${DATASET:-}"
append_if_set args --limit "${LIMIT:-}"

run_main "${args[@]}" "${extra_args[@]}"
