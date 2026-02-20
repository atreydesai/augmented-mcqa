#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Remote SLURM local-model eval orchestration (smoke + main, sharded, non-branching).

Usage:
  jobs/clip_local_eval_master.sh [options]

Options:
  --phase <smoke|main|both>     Which phase(s) to run (default: both)
  --run-tag <tag>               Run tag (default: UTC timestamp)
  --max-concurrent-jobs <int>   Max active shard jobs across all combos (default: 12)
  --num-shards-smoke <int>      Shards per combo for smoke phase (default: 3)
  --num-shards-main <int>       Shards per combo for main phase (default: 12)
  --save-interval <int>         Mid-eval checkpoint interval (default: 50)
  --keep-checkpoints <int>      Keep newest checkpoint files per root (default: 2)
  --max-tokens <int>            Eval max_tokens (default: 2048)
  --force-refresh               Redownload HF generator datasets to remote disk
  --skip-push                   Do not push final artifacts to HuggingFace Hub
  --repo-root <path>            Remote repo path (default: /fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa)
  --help                        Show this help text

Examples:
  jobs/clip_local_eval_master.sh --phase smoke
  jobs/clip_local_eval_master.sh --phase both --max-concurrent-jobs 12 --num-shards-main 12
USAGE
}

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

PHASE="both"
RUN_TAG="local_eval_$(date -u +%Y%m%d_%H%M%S)"
MAX_CONCURRENT_JOBS=12
NUM_SHARDS_SMOKE=3
NUM_SHARDS_MAIN=12
SAVE_INTERVAL=50
KEEP_CHECKPOINTS=2
MAX_TOKENS=2048
FORCE_REFRESH=0
SKIP_PUSH=0
REPO_ROOT="/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa"

CACHE_ROOT="/fs/nexus-scratch/adesai10/hub"
DATASETS_REMOTE_ROOT_REL="datasets/finished_sets_remote"
RESULTS_ROOT_REL="results/local_eval"
LOG_ROOT_REL="logs/slurm/local_eval"

EVAL_MODELS=(
  "Nanbeige/Nanbeige4.1-3B"
  "Qwen/Qwen3-4B-Instruct-2507"
  "allenai/Olmo-3-7B-Instruct"
)

GENERATOR_LABELS=(
  "opus"
  "gpt-4.1"
  "gpt-5.2"
)

PROMPT_MODES=(
  "full"
  "choices_only"
)

DATASET_TYPES=(
  "mmlu_pro"
  "gpqa"
  "arc_easy"
  "arc_challenge"
)

DISTRACTOR_SOURCES=(
  "scratch"
  "dhuman"
  "dmodel"
)

# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase)
      PHASE="${2:-}"; shift 2 ;;
    --run-tag)
      RUN_TAG="${2:-}"; shift 2 ;;
    --max-concurrent-jobs)
      MAX_CONCURRENT_JOBS="${2:-}"; shift 2 ;;
    --num-shards-smoke)
      NUM_SHARDS_SMOKE="${2:-}"; shift 2 ;;
    --num-shards-main)
      NUM_SHARDS_MAIN="${2:-}"; shift 2 ;;
    --save-interval)
      SAVE_INTERVAL="${2:-}"; shift 2 ;;
    --keep-checkpoints)
      KEEP_CHECKPOINTS="${2:-}"; shift 2 ;;
    --max-tokens)
      MAX_TOKENS="${2:-}"; shift 2 ;;
    --force-refresh)
      FORCE_REFRESH=1; shift ;;
    --skip-push)
      SKIP_PUSH=1; shift ;;
    --repo-root)
      REPO_ROOT="${2:-}"; shift 2 ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$PHASE" != "smoke" && "$PHASE" != "main" && "$PHASE" != "both" ]]; then
  echo "Invalid --phase: $PHASE (expected smoke|main|both)" >&2
  exit 1
fi

for n in "$MAX_CONCURRENT_JOBS" "$NUM_SHARDS_SMOKE" "$NUM_SHARDS_MAIN" "$SAVE_INTERVAL" "$KEEP_CHECKPOINTS" "$MAX_TOKENS"; do
  if ! [[ "$n" =~ ^[0-9]+$ ]]; then
    echo "Numeric option expected integer but got: $n" >&2
    exit 1
  fi
done

if [[ "$MAX_CONCURRENT_JOBS" -le 0 || "$NUM_SHARDS_SMOKE" -le 0 || "$NUM_SHARDS_MAIN" -le 0 || "$SAVE_INTERVAL" -le 0 || "$MAX_TOKENS" -le 0 ]]; then
  echo "Expected positive integers for jobs/shards/save-interval/max-tokens" >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------

RUN_ROOT="$REPO_ROOT/$RESULTS_ROOT_REL/$RUN_TAG"
LOG_ROOT="$REPO_ROOT/$LOG_ROOT_REL/$RUN_TAG"
STATUS_ROOT="$RUN_ROOT/status"
MANIFEST_ROOT="$RUN_ROOT/manifests"
TMP_JOB_ROOT="$RUN_ROOT/tmp_jobs"
DATASETS_REMOTE_ROOT="$REPO_ROOT/$DATASETS_REMOTE_ROOT_REL"

HEARTBEAT_LOG="$STATUS_ROOT/heartbeat.log"
DASHBOARD_JSON="$STATUS_ROOT/dashboard.json"
DASHBOARD_TSV="$STATUS_ROOT/dashboard.tsv"
FINAL_MANIFEST_JSON="$STATUS_ROOT/final_manifest.json"

mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$STATUS_ROOT" "$MANIFEST_ROOT" "$TMP_JOB_ROOT" "$DATASETS_REMOTE_ROOT"
touch "$HEARTBEAT_LOG"

LAST_HEARTBEAT_EPOCH=0
declare -a ACTIVE_JOB_IDS=()
POLL_CAPACITY_SECONDS=20
POLL_WAIT_SECONDS=30
VLLM_INSTALL_SPEC="${VLLM_INSTALL_SPEC:-vllm==0.11.2}"
UV_SYNC_INEXACT="${UV_SYNC_INEXACT:-1}"

safe_name() {
  printf '%s' "$1" | tr '/ .:' '_' | tr -cd '[:alnum:]_-'
}

generator_repo_for_label() {
  local label="$1"
  case "$label" in
    opus) printf '%s' "atreydesai/qgqa-gpqa-migrate-20260219-141149-claude-opus-4-6" ;;
    gpt-4.1) printf '%s' "atreydesai/qgqa-gpqa-migrate-20260219-141149-gpt-4-1" ;;
    gpt-5.2) printf '%s' "atreydesai/qgqa-gpqa-migrate-20260219-141149-gpt-5-2" ;;
    *)
      echo "Unknown generator label for repo mapping: $label" >&2
      return 1
      ;;
  esac
}

generator_display_for_label() {
  local label="$1"
  case "$label" in
    opus) printf '%s' "claude-opus-4-6" ;;
    gpt-4.1) printf '%s' "gpt-4.1" ;;
    gpt-5.2) printf '%s' "gpt-5.2" ;;
    *)
      echo "Unknown generator label for display mapping: $label" >&2
      return 1
      ;;
  esac
}

now_utc() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

log_line() {
  local msg="$1"
  printf '[%s] %s\n' "$(now_utc)" "$msg"
}

append_heartbeat() {
  local msg="$1"
  # Log to file + stderr so command substitutions receive clean stdout.
  log_line "$msg" | tee -a "$HEARTBEAT_LOG" >&2
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Required command not found: $cmd" >&2
    exit 1
  fi
}

update_dashboard() {
  uv run python - "$STATUS_ROOT" "$DASHBOARD_JSON" "$DASHBOARD_TSV" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

status_root = Path(sys.argv[1])
dashboard_json = Path(sys.argv[2])
dashboard_tsv = Path(sys.argv[3])
shards_root = status_root / "shards"

def load_status(path: Path) -> str:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return str(data.get("state", "unknown"))
    except Exception:
        return "unknown"

payload = {
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "phases": {},
    "totals": {
        "submitted": 0,
        "running": 0,
        "success": 0,
        "failed": 0,
        "unknown": 0,
        "completed": 0,
        "total": 0,
    },
}

if shards_root.exists():
    for phase_dir in sorted([p for p in shards_root.iterdir() if p.is_dir()]):
        phase_name = phase_dir.name
        phase_payload = {
            "combos": {},
            "totals": {
                "submitted": 0,
                "running": 0,
                "success": 0,
                "failed": 0,
                "unknown": 0,
                "completed": 0,
                "total": 0,
            },
        }
        for combo_dir in sorted([p for p in phase_dir.iterdir() if p.is_dir()]):
            counts = {
                "submitted": 0,
                "running": 0,
                "success": 0,
                "failed": 0,
                "unknown": 0,
                "completed": 0,
                "total": 0,
            }
            for sf in sorted(combo_dir.glob("shard_*.json")):
                state = load_status(sf)
                counts["total"] += 1
                if state in counts:
                    counts[state] += 1
                else:
                    counts["unknown"] += 1
            counts["completed"] = counts["success"] + counts["failed"]
            phase_payload["combos"][combo_dir.name] = counts
            for k in counts:
                phase_payload["totals"][k] += counts[k]
        payload["phases"][phase_name] = phase_payload
        for k in payload["totals"]:
            payload["totals"][k] += phase_payload["totals"][k]

dashboard_json.parent.mkdir(parents=True, exist_ok=True)
with open(dashboard_json, "w") as f:
    json.dump(payload, f, indent=2)

with open(dashboard_tsv, "w") as f:
    f.write("phase\tcombo\tsubmitted\trunning\tsuccess\tfailed\tunknown\tcompleted\ttotal\n")
    for phase_name, phase_data in sorted(payload["phases"].items()):
        for combo_name, counts in sorted(phase_data["combos"].items()):
            f.write(
                f"{phase_name}\t{combo_name}\t{counts['submitted']}\t{counts['running']}\t"
                f"{counts['success']}\t{counts['failed']}\t{counts['unknown']}\t"
                f"{counts['completed']}\t{counts['total']}\n"
            )

totals = payload["totals"]
print(
    f"submitted={totals['submitted']} running={totals['running']} "
    f"success={totals['success']} failed={totals['failed']} "
    f"unknown={totals['unknown']} completed={totals['completed']}"
)
PY
}

maybe_heartbeat() {
  local now_epoch
  now_epoch="$(date +%s)"
  if (( now_epoch - LAST_HEARTBEAT_EPOCH >= 60 )); then
    local summary
    summary="$(update_dashboard)"
    append_heartbeat "heartbeat $summary"
    LAST_HEARTBEAT_EPOCH="$now_epoch"
  fi
}

refresh_active_jobs() {
  if (( ${#ACTIVE_JOB_IDS[@]} == 0 )); then
    ACTIVE_JOB_IDS=()
    return 0
  fi

  local joined
  joined="$(IFS=,; printf '%s' "${ACTIVE_JOB_IDS[*]}")"
  local active_ids_raw
  active_ids_raw="$(squeue -h -j "$joined" -o '%A' 2>/dev/null || true)"
  local active_ids=" $(echo "$active_ids_raw" | tr '\n' ' ') "

  local new_active=()
  local jid
  for jid in "${ACTIVE_JOB_IDS[@]:-}"; do
    if [[ "$active_ids" == *" $jid "* ]]; then
      new_active+=("$jid")
    fi
  done
  ACTIVE_JOB_IDS=("${new_active[@]}")
}

wait_for_capacity() {
  while true; do
    refresh_active_jobs
    if (( ${#ACTIVE_JOB_IDS[@]} < MAX_CONCURRENT_JOBS )); then
      break
    fi
    maybe_heartbeat
    sleep "$POLL_CAPACITY_SECONDS"
  done
}

wait_for_jobs() {
  local job_ids=("$@")
  if (( ${#job_ids[@]} == 0 )); then
    return 0
  fi

  local pending=("${job_ids[@]}")
  while (( ${#pending[@]} > 0 )); do
    local joined
    joined="$(IFS=,; printf '%s' "${pending[*]}")"
    local active_ids_raw
    active_ids_raw="$(squeue -h -j "$joined" -o '%A' 2>/dev/null || true)"
    local active_ids=" $(echo "$active_ids_raw" | tr '\n' ' ') "

    local next_pending=()
    local jid
    for jid in "${pending[@]}"; do
      if [[ "$active_ids" == *" $jid "* ]]; then
        next_pending+=("$jid")
      fi
    done
    pending=("${next_pending[@]:-}")
    maybe_heartbeat
    if (( ${#pending[@]} > 0 )); then
      sleep "$POLL_WAIT_SECONDS"
    fi
  done
  refresh_active_jobs
  update_dashboard >/dev/null
}

write_shard_status() {
  local status_file="$1"
  local phase="$2"
  local combo="$3"
  local shard_index="$4"
  local state="$5"
  local job_id="$6"
  local attempt="$7"
  local exit_code="$8"
  local message="$9"

  mkdir -p "$(dirname "$status_file")"
  uv run python - "$status_file" "$phase" "$combo" "$shard_index" "$state" "$job_id" "$attempt" "$exit_code" "$message" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "phase": sys.argv[2],
    "combo_id": sys.argv[3],
    "shard_index": int(sys.argv[4]),
    "state": sys.argv[5],
    "job_id": sys.argv[6],
    "attempt": int(sys.argv[7]),
    "exit_code": int(sys.argv[8]),
    "message": sys.argv[9],
}
with open(path, "w") as f:
    json.dump(payload, f, indent=2)
PY
}

check_prereqs_and_env() {
  require_cmd uv
  require_cmd sbatch
  require_cmd squeue

  if [[ -f "$HOME/.bashrc" ]]; then
    # shellcheck disable=SC1090
    source "$HOME/.bashrc" || true
  fi

  cd "$REPO_ROOT"

  export MODEL_CACHE_DIR="$CACHE_ROOT"
  export HF_HOME="$CACHE_ROOT"
  export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"
  export TRANSFORMERS_CACHE="$CACHE_ROOT/transformers"
  export PYTHONUNBUFFERED=1
  if [[ "$SKIP_PUSH" -eq 1 ]]; then
    export HF_SKIP_PUSH=1
  else
    export HF_SKIP_PUSH=0
  fi

  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is not set. Export HF_TOKEN before running this script." >&2
    exit 1
  fi

  if [[ "$UV_SYNC_INEXACT" == "1" ]]; then
    append_heartbeat "syncing uv environment mode=inexact"
    uv sync --inexact
  else
    append_heartbeat "syncing uv environment mode=exact"
    uv sync
  fi

  # Local model eval requires vLLM; install a wheel-only build if missing.
  if ! uv run python - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("vllm") is not None else 1)
PY
  then
    append_heartbeat "vllm_missing installing_with_uv_pip spec=$VLLM_INSTALL_SPEC"
    if ! uv pip install --only-binary=:all: "$VLLM_INSTALL_SPEC"; then
      echo "Failed to install $VLLM_INSTALL_SPEC as a prebuilt wheel." >&2
      echo "Set VLLM_INSTALL_SPEC to another wheel-backed version or install CUDA toolkit+nvcc for source builds." >&2
      echo "Example retry: VLLM_INSTALL_SPEC='vllm==0.10.2' jobs/clip_local_eval_master.sh --phase smoke ..." >&2
      exit 1
    fi
    if ! uv run python - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("vllm") is not None else 1)
PY
    then
      echo "Failed to install/import vllm in uv environment." >&2
      exit 1
    fi
    append_heartbeat "vllm_install_complete"
  fi
}

materialize_single_dataset() {
  local label="$1"
  local repo_id="$2"
  local target_dir="$DATASETS_REMOTE_ROOT/$label/combined"

  append_heartbeat "dataset_sync label=$label repo=$repo_id target=$target_dir force_refresh=$FORCE_REFRESH"

  uv run python - "$repo_id" "$target_dir" "$FORCE_REFRESH" <<'PY'
import json
import shutil
import sys
from pathlib import Path

from datasets import load_dataset, load_from_disk

repo_id = sys.argv[1]
target = Path(sys.argv[2])
force_refresh = bool(int(sys.argv[3]))
required_splits = ["arc_easy", "arc_challenge", "mmlu_pro", "gpqa"]

if force_refresh and target.exists():
    shutil.rmtree(target)

if not target.exists():
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f"{target.name}_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    ds = load_dataset(repo_id)
    if not hasattr(ds, "keys"):
        raise SystemExit(f"{repo_id} did not load as DatasetDict")
    ds.save_to_disk(str(tmp))
    if target.exists():
        shutil.rmtree(target)
    tmp.rename(target)

loaded = load_from_disk(str(target))
if not hasattr(loaded, "keys"):
    raise SystemExit(f"Saved dataset at {target} is not a DatasetDict")
missing = [s for s in required_splits if s not in loaded]
if missing:
    raise SystemExit(
        f"{repo_id} at {target} missing required splits: {missing}. "
        f"present={sorted(list(loaded.keys()))}"
    )

sizes = {k: len(loaded[k]) for k in required_splits}
print(json.dumps({"repo_id": repo_id, "target": str(target), "sizes": sizes}, indent=2))
PY
}

materialize_datasets() {
  local label
  for label in "${GENERATOR_LABELS[@]}"; do
    materialize_single_dataset "$label" "$(generator_repo_for_label "$label")"
  done
}

build_combo_id() {
  local phase="$1"
  local eval_model="$2"
  local gen_label="$3"
  local mode="$4"
  printf '%s__%s__%s__%s' "$phase" "$(safe_name "$eval_model")" "$(safe_name "$gen_label")" "$(safe_name "$mode")"
}

build_repo_name() {
  local phase="$1"
  local eval_model="$2"
  local gen_label="$3"
  local mode="$4"
  local base="qgqa-local-eval-${RUN_TAG}-${phase}-$(safe_name "$eval_model")-$(safe_name "$gen_label")-$(safe_name "$mode")"
  if ((${#base} > 95)); then
    base="${base:0:95}"
  fi
  printf '%s' "$base"
}

plan_combo_manifest() {
  local phase="$1"
  local eval_model="$2"
  local gen_label="$3"
  local mode="$4"
  local num_shards="$5"
  local limit_value="$6"  # empty => full

  local combo_id
  combo_id="$(build_combo_id "$phase" "$eval_model" "$gen_label" "$mode")"
  local combo_root="$RUN_ROOT/$phase/$combo_id"
  local results_base="$combo_root/results/$gen_label"
  local summary_dir="$combo_root/summaries"
  local manifest_path="$MANIFEST_ROOT/$phase/$combo_id.json"
  local dataset_path="$DATASETS_REMOTE_ROOT/$gen_label/combined"

  mkdir -p "$combo_root" "$results_base" "$summary_dir" "$(dirname "$manifest_path")"

  local cmd=(
    uv run python scripts/eval_matrix.py plan
    --preset core16
    --model "$eval_model"
    --dataset-path "$dataset_path"
    --generator-dataset-label "$gen_label"
    --dataset-types "${DATASET_TYPES[@]}"
    --distractor-sources "${DISTRACTOR_SOURCES[@]}"
    --eval-mode behavioral
    --max-tokens "$MAX_TOKENS"
    --save-interval "$SAVE_INTERVAL"
    --output-dir "$results_base"
    --manifest-out "$manifest_path"
  )
  if [[ "$mode" == "choices_only" ]]; then
    cmd+=(--choices-only)
  fi
  if [[ -n "$limit_value" ]]; then
    cmd+=(--limit "$limit_value")
  fi

  log_line "planning combo=$combo_id phase=$phase model=$eval_model gen_label=$gen_label mode=$mode" >&2
  "${cmd[@]}" >/dev/null
  cp "$manifest_path" "$combo_root/manifest.json"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$combo_id" "$eval_model" "$gen_label" "$mode" "$manifest_path" "$combo_root" "$results_base" "$summary_dir" "$num_shards"
}

submit_shard_job() {
  local phase="$1"
  local combo_id="$2"
  local eval_model="$3"
  local gen_label="$4"
  local mode="$5"
  local manifest_path="$6"
  local combo_root="$7"
  local summary_dir="$8"
  local num_shards="$9"
  local shard_index="${10}"
  local attempt="${11}"

  local shard_status_file="$STATUS_ROOT/shards/$phase/$combo_id/shard_${shard_index}.json"
  local phase_log_dir="$LOG_ROOT/$phase"
  local job_script="$TMP_JOB_ROOT/${combo_id}_shard${shard_index}_attempt${attempt}.sh"
  local job_name="qgqa_$(safe_name "$phase")_$(safe_name "$combo_id")_s${shard_index}_a${attempt}"
  local out_log="$phase_log_dir/${combo_id}_shard${shard_index}_attempt${attempt}_%j.out"
  local err_log="$phase_log_dir/${combo_id}_shard${shard_index}_attempt${attempt}_%j.err"

  mkdir -p "$(dirname "$shard_status_file")" "$phase_log_dir" "$(dirname "$job_script")" "$summary_dir"

  write_shard_status "$shard_status_file" "$phase" "$combo_id" "$shard_index" "submitted" "pending" "$attempt" -1 "submitted_to_scheduler"

  cat >"$job_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

STATUS_FILE="$shard_status_file"
PHASE="$phase"
COMBO_ID="$combo_id"
SHARD_INDEX="$shard_index"
ATTEMPT="$attempt"
MANIFEST_PATH="$manifest_path"
GEN_LABEL="$gen_label"
SUMMARY_DIR="$summary_dir"
REPO_ROOT="$REPO_ROOT"
SAVE_INTERVAL="$SAVE_INTERVAL"
KEEP_CHECKPOINTS="$KEEP_CHECKPOINTS"
NUM_SHARDS="$num_shards"
CACHE_ROOT="$CACHE_ROOT"

write_status() {
  local state="\$1"
  local exit_code="\$2"
  local message="\$3"
  python - "\$STATUS_FILE" "\$PHASE" "\$COMBO_ID" "\$SHARD_INDEX" "\$state" "\${SLURM_JOB_ID:-unknown}" "\$ATTEMPT" "\$exit_code" "\$message" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "phase": sys.argv[2],
    "combo_id": sys.argv[3],
    "shard_index": int(sys.argv[4]),
    "state": sys.argv[5],
    "job_id": sys.argv[6],
    "attempt": int(sys.argv[7]),
    "exit_code": int(sys.argv[8]),
    "message": sys.argv[9],
}
with open(path, "w") as f:
    json.dump(payload, f, indent=2)
PY
}

if [[ -f "\$HOME/.bashrc" ]]; then
  # shellcheck disable=SC1090
  source "\$HOME/.bashrc" || true
fi

cd "\$REPO_ROOT"

export MODEL_CACHE_DIR="\$CACHE_ROOT"
export HF_HOME="\$CACHE_ROOT"
export HF_DATASETS_CACHE="\$CACHE_ROOT/datasets"
export TRANSFORMERS_CACHE="\$CACHE_ROOT/transformers"
export PYTHONUNBUFFERED=1

write_status "running" -1 "started"

cmd=(
  uv run python scripts/eval_matrix.py run
  --manifest "\$MANIFEST_PATH"
  --generator-dataset-label "\$GEN_LABEL"
  --output-dir "\$SUMMARY_DIR"
  --num-shards "\$NUM_SHARDS"
  --shard-index "\$SHARD_INDEX"
  --save-interval "\$SAVE_INTERVAL"
  --keep-checkpoints "\$KEEP_CHECKPOINTS"
  --skip-existing
)

set +e
"\${cmd[@]}"
rc=\$?
set -e

if [[ \$rc -eq 0 ]]; then
  write_status "success" "\$rc" "completed"
else
  write_status "failed" "\$rc" "eval_matrix_run_failed"
fi
exit \$rc
EOF
  chmod +x "$job_script"

  wait_for_capacity
  local jid
  jid="$(
    sbatch \
      --parsable \
      --job-name "$job_name" \
      --output "$out_log" \
      --error "$err_log" \
      --partition=clip \
      --account=clip \
      --qos=high \
      --gres=gpu:rtxa6000:1 \
      --time=12:00:00 \
      --mem=32G \
      --cpus-per-task=4 \
      "$job_script"
  )"

  write_shard_status "$shard_status_file" "$phase" "$combo_id" "$shard_index" "submitted" "$jid" "$attempt" -1 "submitted_to_scheduler"
  ACTIVE_JOB_IDS+=("$jid")
  append_heartbeat "submitted phase=$phase combo=$combo_id shard=$shard_index attempt=$attempt job_id=$jid"
  update_dashboard >/dev/null
  printf '%s' "$jid"
}

compute_combo_incomplete_shards() {
  local manifest_path="$1"
  local num_shards="$2"
  local shard_status_dir="$3"
  local check_out="$4"

  uv run python - "$manifest_path" "$num_shards" "$shard_status_dir" "$check_out" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
num_shards = int(sys.argv[2])
status_dir = Path(sys.argv[3])
check_out = Path(sys.argv[4])

with open(manifest_path, "r") as f:
    manifest = json.load(f)

configs = sorted(manifest.get("configs", []), key=lambda x: x.get("config_id", ""))
missing_indices = []
invalid_indices = []
for idx, cfg in enumerate(configs):
    output_dir = Path(cfg["output_dir"])
    results_path = output_dir / "results.json"
    if not results_path.exists():
        missing_indices.append(idx)
        continue
    try:
        with open(results_path, "r") as f:
            payload = json.load(f)
        summary = payload.get("summary", {})
        successful = int(summary.get("successful_entries", 0))
        attempted = int(summary.get("attempted_entries", 0))
    except Exception:
        invalid_indices.append(idx)
        continue
    # Structural validity: at least one successful eval entry should exist.
    if attempted > 0 and successful <= 0:
        invalid_indices.append(idx)

missing_result_shards = sorted(set(i % num_shards for i in missing_indices))
invalid_result_shards = sorted(set(i % num_shards for i in invalid_indices))
non_success_shards = []
missing_status_shards = []

for shard in range(num_shards):
    sf = status_dir / f"shard_{shard}.json"
    if not sf.exists():
        missing_status_shards.append(shard)
        continue
    try:
        with open(sf, "r") as f:
            payload = json.load(f)
        state = str(payload.get("state", "unknown"))
    except Exception:
        state = "unknown"
    if state != "success":
        non_success_shards.append(shard)

incomplete = sorted(
    set(
        missing_result_shards
        + invalid_result_shards
        + non_success_shards
        + missing_status_shards
    )
)
result = {
    "manifest_path": str(manifest_path),
    "num_shards": num_shards,
    "expected_configs": len(configs),
    "missing_result_indices": missing_indices,
    "missing_result_count": len(missing_indices),
    "missing_result_shards": missing_result_shards,
    "invalid_result_indices": invalid_indices,
    "invalid_result_count": len(invalid_indices),
    "invalid_result_shards": invalid_result_shards,
    "non_success_shards": sorted(set(non_success_shards)),
    "missing_status_shards": missing_status_shards,
    "incomplete_shards": incomplete,
    "complete": len(incomplete) == 0,
}
check_out.parent.mkdir(parents=True, exist_ok=True)
with open(check_out, "w") as f:
    json.dump(result, f, indent=2)

print(" ".join(str(x) for x in incomplete))
PY
}

generate_combo_plots() {
  local combo_root="$1"
  local results_base="$2"
  local eval_model="$3"
  local generator_label="$4"
  local generator_display="$5"

  append_heartbeat "plotting combo_root=$combo_root eval_model=$eval_model generator=$generator_label"

  local dt
  for dt in "${DATASET_TYPES[@]}"; do
    local plot_dir="$combo_root/plots/$dt"
    mkdir -p "$plot_dir"
    uv run python - "$results_base" "$eval_model" "$plot_dir" "$dt" "$generator_display" <<'PY'
import sys
from pathlib import Path

from analysis.visualize import (
    plot_rq1_combined,
    plot_rq2_human_distractors,
    plot_rq3_model_distractors,
)

base_dir = Path(sys.argv[1])
eval_model = sys.argv[2]
plot_dir = Path(sys.argv[3])
dataset_type = sys.argv[4]
generator_display = sys.argv[5]

plot_rq1_combined(
    base_dir,
    eval_model,
    output_dir=plot_dir,
    show=False,
    dataset_type=dataset_type,
    generator_model=generator_display,
    evaluation_model=eval_model,
)
plot_rq2_human_distractors(
    base_dir,
    eval_model,
    output_dir=plot_dir,
    show=False,
    dataset_type=dataset_type,
    generator_model=generator_display,
    evaluation_model=eval_model,
)
plot_rq3_model_distractors(
    base_dir,
    eval_model,
    output_dir=plot_dir,
    show=False,
    dataset_type=dataset_type,
    generator_model=generator_display,
    evaluation_model=eval_model,
)
PY
  done
}

push_combo_to_hub() {
  local combo_root="$1"
  local repo_name="$2"
  uv run python - "$combo_root" "$repo_name" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

folder = Path(sys.argv[1])
repo_name = sys.argv[2]

token = os.getenv("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN is required for push")

api = HfApi(token=token)
owner = api.whoami().get("name")
if not owner:
    raise SystemExit("Could not resolve HF owner via whoami()")
repo_id = f"{owner}/{repo_name}"
api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path=str(folder),
    path_in_repo="",
)
print(f"https://huggingface.co/datasets/{repo_id}")
PY
}

append_final_manifest_entry() {
  local phase="$1"
  local combo_id="$2"
  local eval_model="$3"
  local gen_label="$4"
  local mode="$5"
  local combo_root="$6"
  local results_base="$7"
  local manifest_path="$8"
  local complete="$9"
  local hf_url="${10}"
  local notes="${11}"

  uv run python - "$FINAL_MANIFEST_JSON" "$phase" "$combo_id" "$eval_model" "$gen_label" "$mode" "$combo_root" "$results_base" "$manifest_path" "$complete" "$hf_url" "$notes" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
entry = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "phase": sys.argv[2],
    "combo_id": sys.argv[3],
    "eval_model": sys.argv[4],
    "generator_label": sys.argv[5],
    "prompt_mode": sys.argv[6],
    "combo_root": sys.argv[7],
    "results_base": sys.argv[8],
    "manifest_path": sys.argv[9],
    "complete": bool(int(sys.argv[10])),
    "hf_url": sys.argv[11] if sys.argv[11] else None,
    "notes": sys.argv[12] if sys.argv[12] else None,
}
payload = {"created_at": datetime.now(timezone.utc).isoformat(), "entries": []}
if path.exists():
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except Exception:
        payload = {"created_at": datetime.now(timezone.utc).isoformat(), "entries": []}
payload.setdefault("entries", [])
payload["entries"].append(entry)
with open(path, "w") as f:
    json.dump(payload, f, indent=2)
PY
}

run_phase() {
  local phase="$1"
  local num_shards="$2"
  local limit_value="$3"  # empty => full

  append_heartbeat "phase_start phase=$phase num_shards=$num_shards limit=${limit_value:-all}"

  local phase_manifest_dir="$MANIFEST_ROOT/$phase"
  local phase_status_dir="$STATUS_ROOT/$phase"
  local phase_checks_dir="$STATUS_ROOT/checks/$phase"
  local combos_tsv="$phase_status_dir/combos.tsv"
  local phase_summary_json="$phase_status_dir/summary.json"
  mkdir -p "$phase_manifest_dir" "$phase_status_dir" "$phase_checks_dir"
  : >"$combos_tsv"

  # 1) Plan all combos
  local eval_model
  local gen_label
  local mode
  for eval_model in "${EVAL_MODELS[@]}"; do
    for gen_label in "${GENERATOR_LABELS[@]}"; do
      for mode in "${PROMPT_MODES[@]}"; do
        plan_combo_manifest "$phase" "$eval_model" "$gen_label" "$mode" "$num_shards" "$limit_value" >>"$combos_tsv"
      done
    done
  done
  append_heartbeat "phase_plan_complete phase=$phase combos=$(wc -l <"$combos_tsv" | tr -d ' ')"

  # 2) Submit initial shard jobs for all combos
  local phase_job_ids=()
  while IFS=$'\t' read -r combo_id combo_eval combo_gen combo_mode manifest_path combo_root results_base summary_dir combo_shards; do
    local shard
    for ((shard = 0; shard < combo_shards; shard++)); do
      local jid
      jid="$(
        submit_shard_job \
          "$phase" "$combo_id" "$combo_eval" "$combo_gen" "$combo_mode" \
          "$manifest_path" "$combo_root" "$summary_dir" "$combo_shards" "$shard" 0
      )"
      phase_job_ids+=("$jid")
    done
  done <"$combos_tsv"

  append_heartbeat "phase_initial_submissions_done phase=$phase submitted_jobs=${#phase_job_ids[@]}"
  wait_for_jobs "${phase_job_ids[@]}"
  append_heartbeat "phase_initial_jobs_complete phase=$phase"

  # 3) Completeness check + one retry round per combo
  local complete_count=0
  local failed_count=0

  while IFS=$'\t' read -r combo_id combo_eval combo_gen combo_mode manifest_path combo_root results_base summary_dir combo_shards; do
    local shard_status_dir="$STATUS_ROOT/shards/$phase/$combo_id"
    local check_file="$phase_checks_dir/${combo_id}.json"
    local incomplete_shards
    incomplete_shards="$(compute_combo_incomplete_shards "$manifest_path" "$combo_shards" "$shard_status_dir" "$check_file")"

    if [[ -n "$incomplete_shards" ]]; then
      append_heartbeat "combo_retry phase=$phase combo=$combo_id shards=$incomplete_shards"
      local retry_job_ids=()
      local retry_shard
      for retry_shard in $incomplete_shards; do
        local retry_jid
        retry_jid="$(
          submit_shard_job \
            "$phase" "$combo_id" "$combo_eval" "$combo_gen" "$combo_mode" \
            "$manifest_path" "$combo_root" "$summary_dir" "$combo_shards" "$retry_shard" 1
        )"
        retry_job_ids+=("$retry_jid")
      done
      wait_for_jobs "${retry_job_ids[@]}"
      incomplete_shards="$(compute_combo_incomplete_shards "$manifest_path" "$combo_shards" "$shard_status_dir" "$check_file")"
    fi

    local combo_complete=1
    local notes=""
    if [[ -n "$incomplete_shards" ]]; then
      combo_complete=0
      notes="incomplete_shards_after_retry=$incomplete_shards"
      ((failed_count += 1))
      append_heartbeat "combo_incomplete phase=$phase combo=$combo_id shards=$incomplete_shards"
    else
      ((complete_count += 1))
      append_heartbeat "combo_complete phase=$phase combo=$combo_id"
      generate_combo_plots "$combo_root" "$results_base" "$combo_eval" "$combo_gen" "$(generator_display_for_label "$combo_gen")"
      notes="complete"
    fi

    local hf_url=""
    if [[ "$combo_complete" -eq 1 && "$SKIP_PUSH" -eq 0 ]]; then
      local repo_name
      repo_name="$(build_repo_name "$phase" "$combo_eval" "$combo_gen" "$combo_mode")"
      append_heartbeat "combo_push_start phase=$phase combo=$combo_id repo=$repo_name"
      hf_url="$(push_combo_to_hub "$combo_root" "$repo_name")"
      append_heartbeat "combo_push_done phase=$phase combo=$combo_id url=$hf_url"
    fi

    append_final_manifest_entry \
      "$phase" "$combo_id" "$combo_eval" "$combo_gen" "$combo_mode" \
      "$combo_root" "$results_base" "$manifest_path" "$combo_complete" "$hf_url" "$notes"
  done <"$combos_tsv"

  uv run python - "$phase_summary_json" "$phase" "$complete_count" "$failed_count" "$combos_tsv" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
phase = sys.argv[2]
complete = int(sys.argv[3])
failed = int(sys.argv[4])
combos_tsv = Path(sys.argv[5])
total = sum(1 for _ in combos_tsv.open("r")) if combos_tsv.exists() else 0
payload = {
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "phase": phase,
    "total_combos": total,
    "complete_combos": complete,
    "failed_combos": failed,
}
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(payload, f, indent=2)
PY

  append_heartbeat "phase_done phase=$phase complete_combos=$complete_count failed_combos=$failed_count"
}

main() {
  append_heartbeat "run_start run_tag=$RUN_TAG phase=$PHASE"
  append_heartbeat "run_config repo_root=$REPO_ROOT max_concurrent_jobs=$MAX_CONCURRENT_JOBS num_shards_smoke=$NUM_SHARDS_SMOKE num_shards_main=$NUM_SHARDS_MAIN save_interval=$SAVE_INTERVAL keep_checkpoints=$KEEP_CHECKPOINTS max_tokens=$MAX_TOKENS force_refresh=$FORCE_REFRESH skip_push=$SKIP_PUSH"

  check_prereqs_and_env
  materialize_datasets

  case "$PHASE" in
    smoke)
      run_phase "smoke" "$NUM_SHARDS_SMOKE" "5"
      ;;
    main)
      run_phase "main" "$NUM_SHARDS_MAIN" ""
      ;;
    both)
      run_phase "smoke" "$NUM_SHARDS_SMOKE" "5"
      run_phase "main" "$NUM_SHARDS_MAIN" ""
      ;;
  esac

  update_dashboard >/dev/null
  append_heartbeat "run_complete run_tag=$RUN_TAG final_manifest=$FINAL_MANIFEST_JSON dashboard_json=$DASHBOARD_JSON dashboard_tsv=$DASHBOARD_TSV"
  log_line "Done. Run root: $RUN_ROOT"
}

main "$@"
