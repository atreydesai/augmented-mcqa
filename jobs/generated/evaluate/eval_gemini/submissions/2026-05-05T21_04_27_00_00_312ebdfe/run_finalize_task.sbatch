#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMPDIR_ROOT="${SLURM_TMPDIR:-/tmp}"
export TMPDIR="$(mktemp -d "$TMPDIR_ROOT/finalizer.XXXXXX")"
trap 'rm -rf "$TMPDIR"' EXIT
export MODEL_HUB_ROOT="${MODEL_HUB_ROOT:-/fs/nexus-scratch/adesai10/hub}"
export RUNTIME_CACHE_ROOT="$MODEL_HUB_ROOT/runtime-cache"
export XDG_CACHE_HOME="$RUNTIME_CACHE_ROOT/xdg-cache"
export XDG_DATA_HOME="$RUNTIME_CACHE_ROOT/xdg-data"
export XDG_STATE_HOME="$RUNTIME_CACHE_ROOT/xdg-state"
export HF_HOME="$MODEL_HUB_ROOT"
export HUGGINGFACE_HUB_CACHE="$MODEL_HUB_ROOT"
export TRANSFORMERS_CACHE="$MODEL_HUB_ROOT/transformers"
export TORCH_HOME="$MODEL_HUB_ROOT/torch"
export TRITON_CACHE_DIR="$RUNTIME_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUNTIME_CACHE_ROOT/torchinductor"
mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

MANIFEST_PATH="$1"
FINALIZER_INDEX="$2"
PROJECT_ROOT="${3:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${4:-python}"
PYTHON_DIR="$(dirname "$PYTHON_BIN")"

cd "$PROJECT_ROOT"
export PATH="$PYTHON_DIR:$PATH"

"$PYTHON_BIN" - <<'PY' "$MANIFEST_PATH" "$FINALIZER_INDEX" "$PROJECT_ROOT" "$PYTHON_BIN"
import json
import subprocess
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
finalizer_index = int(sys.argv[2])
project_root = Path(sys.argv[3])
python_bin = sys.argv[4]

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
task = manifest["finalizers"][finalizer_index]
cmd = [python_bin, "main.py", *task["argv"]]
print(f"Running: {' '.join(cmd)}")
sys.exit(subprocess.run(cmd, cwd=str(project_root), check=False).returncode)
PY
