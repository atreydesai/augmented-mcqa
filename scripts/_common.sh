#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

run_main() {
  uv run python main.py "$@"
}

append_if_set() {
  local -n target=$1
  local flag=$2
  local value=${3:-}
  if [[ -n "$value" ]]; then
    target+=("$flag" "$value")
  fi
}
