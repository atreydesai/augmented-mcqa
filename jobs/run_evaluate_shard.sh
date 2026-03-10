#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" main.py evaluate "$@"
