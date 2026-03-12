#!/bin/bash
set -euo pipefail

MANIFEST_PATH="$1"
TASK_INDEX="$2"
PROJECT_ROOT="${3:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${4:-python}"

cd "$PROJECT_ROOT"

"$PYTHON_BIN" - <<'PY' "$MANIFEST_PATH" "$TASK_INDEX" "$PROJECT_ROOT" "$PYTHON_BIN"
import json
import subprocess
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
task_index = int(sys.argv[2])
project_root = Path(sys.argv[3])
python_bin = sys.argv[4]

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
task = manifest["tasks"][task_index]
cmd = [python_bin, "main.py", *task["argv"]]
print(f"Running: {' '.join(cmd)}")
sys.exit(subprocess.run(cmd, cwd=str(project_root), check=False).returncode)
PY
