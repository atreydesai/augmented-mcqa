#!/usr/bin/env bash
set -euo pipefail
source "$HOME/.bashrc" || true
cd /fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa
export MODEL_CACHE_DIR=/fs/nexus-scratch/adesai10/hub
export HF_HOME=/fs/nexus-scratch/adesai10/hub
export PYTHONUNBUFFERED=1
export UV_NO_SYNC=1
uv run --no-sync python scripts/test_olmo_inference.py
