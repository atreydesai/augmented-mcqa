#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run a minimal end-to-end Final5 eval sharding smoke test on a single GPU host.

Usage:
  scripts/run_final5_remote_smoke.sh [options]

Options:
  --gen-full-ds <path>        Source generated dataset (default: newest datasets/augmented/final5_full_*)
  --generator-label <label>   Generator label in results path (default: inferred from source dataset dir)
  --hf-home <path>            HF cache root (default: /fs/nexus-scratch/adesai10/hub)
  --arc-limit <int>           Smoke rows for arc_challenge (default: 5)
  --mmlu-limit <int>          Smoke rows for mmlu_pro (default: 5)
  --gpqa-limit <int>          Smoke rows for gpqa (default: 2)
  --target-rows <int>         Target rows per sub-split for bundle builder (default: 3)
  --save-interval <int>       save_interval passed to eval runs (default: 2)
  --max-tokens <int>          max_tokens passed to eval runs (default: 32)
  --timestamp <value>         Optional fixed timestamp suffix for outputs
  --skip-run                  Build bundle only; do not execute arrays/merge/plot
  --skip-plot                 Skip plot generation
  --help                      Show this help

Notes:
  - Runs all eval models from scripts/build_eval_slurm_bundle.py.
  - Each task runs preset=final5, so all 5 settings are exercised.
  - Eval currently does not emit periodic temp JSON checkpoints; this script still checks for new temp_final5 files.
USAGE
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
GEN_FULL_DS=""
GENERATOR_LABEL=""
HF_HOME_OVERRIDE="/fs/nexus-scratch/adesai10/hub"
ARC_LIMIT=5
MMLU_LIMIT=5
GPQA_LIMIT=2
TARGET_ROWS=3
SAVE_INTERVAL=2
MAX_TOKENS=32
TIMESTAMP_OVERRIDE=""
SKIP_RUN=0
SKIP_PLOT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gen-full-ds)
      GEN_FULL_DS="$2"
      shift 2
      ;;
    --generator-label)
      GENERATOR_LABEL="$2"
      shift 2
      ;;
    --hf-home)
      HF_HOME_OVERRIDE="$2"
      shift 2
      ;;
    --arc-limit)
      ARC_LIMIT="$2"
      shift 2
      ;;
    --mmlu-limit)
      MMLU_LIMIT="$2"
      shift 2
      ;;
    --gpqa-limit)
      GPQA_LIMIT="$2"
      shift 2
      ;;
    --target-rows)
      TARGET_ROWS="$2"
      shift 2
      ;;
    --save-interval)
      SAVE_INTERVAL="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --timestamp)
      TIMESTAMP_OVERRIDE="$2"
      shift 2
      ;;
    --skip-run)
      SKIP_RUN=1
      shift
      ;;
    --skip-plot)
      SKIP_PLOT=1
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

cd "$PROJECT_ROOT"
PROJECT_ROOT="$(pwd -P)"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if command -v uv >/dev/null 2>&1; then
  PY_RUNNER=(uv run python)
else
  if ! command -v python >/dev/null 2>&1; then
    echo "Error: neither 'uv' nor 'python' is available on PATH."
    exit 1
  fi
  PY_RUNNER=(python)
  echo "Warning: 'uv' not found; using 'python' from current environment."
fi

export DATASETS_DIR="$PROJECT_ROOT/datasets"
export RESULTS_DIR="$PROJECT_ROOT/results"
mkdir -p "$DATASETS_DIR" "$RESULTS_DIR"

export HF_HOME="$HF_HOME_OVERRIDE"
export MODEL_CACHE_DIR="$HF_HOME_OVERRIDE"
if [[ ! -d "$HF_HOME" ]]; then
  mkdir -p "$HF_HOME" 2>/dev/null || true
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "Error: jq is required for this smoke script."
  exit 1
fi

if [[ -z "$GEN_FULL_DS" ]]; then
  GEN_FULL_DS="$(ls -dt "$DATASETS_DIR"/augmented/final5_full_* 2>/dev/null | head -n1 || true)"
fi

if [[ -n "$GEN_FULL_DS" && "$GEN_FULL_DS" != /* ]]; then
  GEN_FULL_DS="$PROJECT_ROOT/$GEN_FULL_DS"
fi

if [[ -z "$GEN_FULL_DS" ]]; then
  echo "Error: no source dataset found. Pass --gen-full-ds."
  exit 1
fi
if [[ ! -d "$GEN_FULL_DS" ]]; then
  echo "Error: source dataset path is not a directory: $GEN_FULL_DS"
  exit 1
fi

if [[ -z "$GENERATOR_LABEL" ]]; then
  GENERATOR_LABEL="$(basename "$GEN_FULL_DS" | sed -E 's/^final5_full_[0-9]{8}_[0-9]{6}_//')"
fi

TS="${TIMESTAMP_OVERRIDE:-$(date +%Y%m%d_%H%M%S)}"
SMOKE_DS="$DATASETS_DIR/augmented/final5_smoke_${TS}"
SMOKE_MANIFEST="$DATASETS_DIR/augmented/final5_smoke_manifest_${TS}.json"
BUNDLE_DIR="$PROJECT_ROOT/jobs/generated/final5_smoke_${TS}"
SMOKE_OUT="$RESULTS_DIR/final5_smoke_${TS}"
MARKER="/tmp/final5_eval_smoke_${TS}.marker"
touch "$MARKER"

echo "Project root: $PROJECT_ROOT"
echo "Source dataset: $GEN_FULL_DS"
echo "Generator label: $GENERATOR_LABEL"
echo "Smoke dataset: $SMOKE_DS"
echo "Bundle dir: $BUNDLE_DIR"
echo "Smoke output: $SMOKE_OUT"

echo
echo "[1/9] Building tiny smoke dataset (arc=${ARC_LIMIT}, mmlu_pro=${MMLU_LIMIT}, gpqa=${GPQA_LIMIT})"
export GEN_FULL_DS SMOKE_DS ARC_LIMIT MMLU_LIMIT GPQA_LIMIT
"${PY_RUNNER[@]}" - <<'PY'
from datasets import DatasetDict, load_from_disk
import os

src = os.environ["GEN_FULL_DS"]
dst = os.environ["SMOKE_DS"]
arc_limit = int(os.environ["ARC_LIMIT"])
mmlu_limit = int(os.environ["MMLU_LIMIT"])
gpqa_limit = int(os.environ["GPQA_LIMIT"])

if arc_limit <= 0 or mmlu_limit <= 0 or gpqa_limit <= 0:
    raise ValueError("All split limits must be > 0")

ds = load_from_disk(src)
mini = DatasetDict({
    "arc_challenge": ds["arc_challenge"].select(range(min(arc_limit, len(ds["arc_challenge"])))),
    "mmlu_pro": ds["mmlu_pro"].select(range(min(mmlu_limit, len(ds["mmlu_pro"])))),
    "gpqa": ds["gpqa"].select(range(min(gpqa_limit, len(ds["gpqa"])))),
})
mini.save_to_disk(dst)
print("smoke dataset:", dst)
print({k: len(mini[k]) for k in mini.keys()})
PY

echo
echo "[2/9] Writing minimal regen manifest"
cat > "$SMOKE_MANIFEST" <<JSON
{
  "manifest_version": 1,
  "schema_version": "final5_v1",
  "generators": [
    {"model": "${GENERATOR_LABEL}", "dataset_path": "${SMOKE_DS}", "returncode": 0}
  ]
}
JSON

echo
echo "[3/9] Building per-pair bundle"
"${PY_RUNNER[@]}" scripts/build_eval_slurm_bundle.py \
  --manifest "$SMOKE_MANIFEST" \
  --output-dir "$BUNDLE_DIR" \
  --output-base "$SMOKE_OUT" \
  --target-rows-per-subsplit "$TARGET_ROWS" \
  --save-interval "$SAVE_INTERVAL" \
  --max-tokens "$MAX_TOKENS"

echo "Bundle quick check:"
jq '.total_sbatch_files, .total_array_tasks, .dataset_part_counts' "$BUNDLE_DIR/bundle_manifest.json"
for wu in "$BUNDLE_DIR"/*.work_units.json; do
  echo "$(basename "$wu") units=$(jq 'length' "$wu")"
done

if [[ "$SKIP_RUN" == "0" ]]; then
  echo
  echo "[4/9] Running all array tasks directly (no sbatch)"
  mapfile -t sbatch_files < <(find "$BUNDLE_DIR" -maxdepth 1 -type f -name 'final5_pair__*.sbatch' | sort)
  if [[ "${#sbatch_files[@]}" -eq 0 ]]; then
    echo "Error: no sbatch files found in $BUNDLE_DIR"
    exit 1
  fi

  for sb in "${sbatch_files[@]}"; do
    echo "Running $(basename "$sb")"
    wu="${sb%.sbatch}.work_units.json"
    if [[ ! -f "$wu" ]]; then
      echo "Error: missing work units file for $sb"
      exit 1
    fi
    num_units="$(jq 'length' "$wu")"
    if [[ "$num_units" -le 0 ]]; then
      echo "Error: $wu has no work units"
      exit 1
    fi
    for idx in $(seq 0 $((num_units - 1))); do
      SLURM_ARRAY_TASK_ID="$idx" \
      SLURM_SUBMIT_DIR="$PWD" \
      PROJECT_ROOT="$PWD" \
      OUTPUT_BASE="$SMOKE_OUT" \
      SAVE_INTERVAL="$SAVE_INTERVAL" \
      MAX_TOKENS="$MAX_TOKENS" \
      LOG_DIR="$SMOKE_OUT/logs" \
      bash "$sb"
    done
  done
else
  echo
  echo "[4/9] Skipping run step (--skip-run)"
  echo "Bundle ready at: $BUNDLE_DIR"
  exit 0
fi

echo
echo "[5/9] Merging partial entry shards"
"${PY_RUNNER[@]}" scripts/merge_eval_subshards.py \
  --bundle-manifest "$BUNDLE_DIR/bundle_manifest.json" \
  --strict

echo
echo "[6/9] Validating counts"
expected_canonical="$(jq '.config_roots | length' "$BUNDLE_DIR/bundle_manifest.json")"
expected_partial="$(jq '[.expected_entry_shards_by_config_root[]] | add' "$BUNDLE_DIR/bundle_manifest.json")"

if command -v rg >/dev/null 2>&1; then
  actual_canonical="$(find "$SMOKE_OUT" -path '*/summary.json' | rg '/(human_from_scratch|model_from_scratch|augment_human|augment_model|augment_ablation)/summary.json$' | wc -l | tr -d ' ')"
else
  actual_canonical="$(find "$SMOKE_OUT" -path '*/summary.json' | grep -E '/(human_from_scratch|model_from_scratch|augment_human|augment_model|augment_ablation)/summary.json$' | wc -l | tr -d ' ')"
fi
actual_partial="$(find "$SMOKE_OUT" -path '*/_partials/entry_shard_*_of_*/summary.json' | wc -l | tr -d ' ')"

echo "Canonical summaries: ${actual_canonical} (expected ${expected_canonical})"
echo "Partial summaries: ${actual_partial} (expected ${expected_partial})"

if [[ "$actual_canonical" != "$expected_canonical" ]]; then
  echo "Error: canonical summary count mismatch"
  exit 1
fi
if [[ "$actual_partial" != "$expected_partial" ]]; then
  echo "Error: partial summary count mismatch"
  exit 1
fi

echo
echo "[7/9] Checking question_idx uniqueness in merged Arrow rows"
export SMOKE_OUT
"${PY_RUNNER[@]}" - <<'PY'
from collections import Counter
from pathlib import Path
import os
import sys

from datasets import load_from_disk

root = Path(os.environ["SMOKE_OUT"])
bad = 0
for rows_dir in sorted(root.glob("*/*/*/*/*/rows")):
    idxs = [int(r["question_idx"]) for r in load_from_disk(str(rows_dir))]
    if any(v > 1 for v in Counter(idxs).values()):
        bad += 1
print("configs_with_duplicate_question_idx:", bad)
sys.exit(1 if bad else 0)
PY

if [[ "$SKIP_PLOT" == "0" ]]; then
  echo
  echo "[8/9] Plotting Final5 visuals"
  "${PY_RUNNER[@]}" scripts/plot_final5.py \
    --results-root "$SMOKE_OUT" \
    --output-dir "$SMOKE_OUT/final5_plots"
  find "$SMOKE_OUT/final5_plots" -maxdepth 1 -type f | sort
else
  echo
  echo "[8/9] Skipping plots (--skip-plot)"
fi

echo
echo "[9/9] Checking for new temp_final5 checkpoint files (eval-side currently expected to be none)"
find "$RESULTS_DIR" -maxdepth 1 -name 'temp_final5_*.json' -newer "$MARKER" -print || true

echo
echo "Smoke run complete."
echo "Bundle manifest: $BUNDLE_DIR/bundle_manifest.json"
echo "Merged summary: $BUNDLE_DIR/merged_summary.json"
echo "Results root: $SMOKE_OUT"
