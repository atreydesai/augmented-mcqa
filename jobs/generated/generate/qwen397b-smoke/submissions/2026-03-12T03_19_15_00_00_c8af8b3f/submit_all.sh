#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PROJECT_ROOT="${PROJECT_ROOT:-/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" - <<'PY' "/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/jobs/generated/generate/qwen397b-smoke/submissions/2026-03-12T03_19_15_00_00_c8af8b3f/manifest.json" "/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/jobs/generated/generate/qwen397b-smoke/submissions/2026-03-12T03_19_15_00_00_c8af8b3f/run_local_task.sbatch" "/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/jobs/generated/generate/qwen397b-smoke/submissions/2026-03-12T03_19_15_00_00_c8af8b3f/run_api_task.sbatch" "$PROJECT_ROOT" "$PYTHON_BIN"
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path(sys.argv[1])
local_wrapper = Path(sys.argv[2])
api_wrapper = Path(sys.argv[3])
project_root = sys.argv[4]
python_bin = sys.argv[5]

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
job_ids = {}
slot_previous = {}
slot_counters = {}
concurrency_caps = dict(manifest.get("concurrency_caps", {}))

def record_submission(task_index, job_id):
    task_record = manifest["tasks"][task_index]
    task_record["submitted_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    task_record["submitted_job_id"] = job_id
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

for task in manifest["tasks"]:
    resource_class = task["resource_class"]
    wrapper = local_wrapper if resource_class == "local" else api_wrapper
    logical_deps = []
    throttle_deps = []
    for dependency_ref in task.get("submit_dependency_refs", []):
        dependency_job_id = job_ids.get(dependency_ref)
        if dependency_job_id:
            logical_deps.append(dependency_job_id)

    cap = concurrency_caps.get(resource_class)
    if cap:
        slot_index = slot_counters.get(resource_class, 0) % int(cap)
        slot_key = (resource_class, slot_index)
        previous_job_id = slot_previous.get(slot_key)
        if previous_job_id:
            throttle_deps.append(previous_job_id)
        slot_counters[resource_class] = slot_counters.get(resource_class, 0) + 1

    cmd = [
        "sbatch",
        "--parsable",
        "--job-name", task["job_name"],
        "--output", task["task_stdout"],
        "--error", task["task_stderr"],
        "--partition", task["resources"]["partition"],
        "--account", task["resources"]["account"],
        "--qos", task["resources"]["qos"],
        "--time", task["resources"]["time_limit"],
        "--mem", task["resources"]["memory"],
        "--cpus-per-task", str(task["resources"]["cpus_per_task"]),
    ]
    if resource_class == "local":
        cmd.extend(["--gres", f"gpu:{task['resources']['gpu_type']}:1"])
    dependency_parts = []
    if logical_deps:
        ordered = []
        seen = set()
        for dependency_job_id in logical_deps:
            if dependency_job_id not in seen:
                seen.add(dependency_job_id)
                ordered.append(dependency_job_id)
        dependency_parts.append("afterok:" + ":".join(ordered))
    if throttle_deps:
        ordered = []
        seen = set()
        for dependency_job_id in throttle_deps:
            if dependency_job_id not in seen:
                seen.add(dependency_job_id)
                ordered.append(dependency_job_id)
        dependency_parts.append("afterany:" + ":".join(ordered))
    if dependency_parts:
        cmd.extend(["--dependency", ",".join(dependency_parts)])
    cmd.extend(
        [
            str(wrapper),
            str(manifest_path),
            str(task["task_index"]),
            project_root,
            python_bin,
        ]
    )

    print("Submitting:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
        sys.exit(result.returncode)

    job_id = (result.stdout.strip().split(";", 1)[0] if result.stdout else "").strip()
    if not job_id:
        print("Failed to parse sbatch job id", file=sys.stderr)
        sys.exit(1)

    job_ids[task["slice_ref"]] = job_id
    record_submission(task["task_index"], job_id)
    if cap:
        slot_previous[(resource_class, slot_index)] = job_id
PY
