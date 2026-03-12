from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from utils.modeling import safe_name
from utils.scheduler_state import iso_now


WRAPPER_TEMPLATE = """#!/bin/bash
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
"""


FINALIZER_WRAPPER_TEMPLATE = """#!/bin/bash
set -euo pipefail

MANIFEST_PATH="$1"
FINALIZER_INDEX="$2"
PROJECT_ROOT="${3:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${4:-python}"

cd "$PROJECT_ROOT"

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
"""


MASTER_SUBMIT_TEMPLATE = """#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PROJECT_ROOT="${PROJECT_ROOT:-__PROJECT_ROOT__}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" - <<'PY' "__MANIFEST_PATH__" "__LOCAL_WRAPPER__" "__API_WRAPPER__" "__FINALIZER_WRAPPER__" "$PROJECT_ROOT" "$PYTHON_BIN"
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path(sys.argv[1])
local_wrapper = Path(sys.argv[2])
api_wrapper = Path(sys.argv[3])
finalizer_wrapper = Path(sys.argv[4])
project_root = sys.argv[5]
python_bin = sys.argv[6]

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
job_ids = {}
slot_previous = {}
slot_counters = {}
concurrency_caps = dict(manifest.get("concurrency_caps", {}))

def record_submission(task_index, job_id):
    task_record = manifest["tasks"][task_index]
    task_record["submitted_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    task_record["submitted_job_id"] = job_id
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\\n", encoding="utf-8")

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

for finalizer in manifest.get("finalizers", []):
    dependency_job_ids = []
    for dependency_ref in finalizer.get("dependency_refs", []):
        dependency_job_id = job_ids.get(dependency_ref)
        if dependency_job_id:
            dependency_job_ids.append(dependency_job_id)
    cmd = [
        "sbatch",
        "--parsable",
        "--job-name", finalizer["job_name"],
        "--output", finalizer["task_stdout"],
        "--error", finalizer["task_stderr"],
        "--partition", finalizer["resources"]["partition"],
        "--account", finalizer["resources"]["account"],
        "--qos", finalizer["resources"]["qos"],
        "--time", finalizer["resources"]["time_limit"],
        "--mem", finalizer["resources"]["memory"],
        "--cpus-per-task", str(finalizer["resources"]["cpus_per_task"]),
    ]
    if dependency_job_ids:
        ordered = []
        seen = set()
        for dependency_job_id in dependency_job_ids:
            if dependency_job_id not in seen:
                seen.add(dependency_job_id)
                ordered.append(dependency_job_id)
        cmd.extend(["--dependency", "afterany:" + ":".join(ordered)])
    cmd.extend(
        [
            str(finalizer_wrapper),
            str(manifest_path),
            str(finalizer["finalizer_index"]),
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
PY
"""


@dataclass
class ClusterTask:
    stage: str
    run_name: str
    model: str
    model_slug: str
    dataset_type: str
    dataset_slug: str
    resource_class: str
    slice_ref: str
    task_slug: str
    question_start: int
    question_end: int
    chunk_index: int
    argv: list[str]
    resources: dict[str, Any]
    strategy: str | None = None
    setting: str | None = None
    mode: str | None = None
    state_dependency_refs: list[str] | None = None
    submit_dependency_refs: list[str] | None = None
    force: bool = False
    generation_run_name: str | None = None
    generation_model: str | None = None

    def as_dict(self, *, task_index: int, submission_created_at: str, submission_id: str, task_log_dir: Path) -> dict[str, Any]:
        task_stdout = task_log_dir / f"{self.task_slug}__%j.out"
        task_stderr = task_log_dir / f"{self.task_slug}__%j.err"
        return {
            "task_index": task_index,
            "stage": self.stage,
            "run_name": self.run_name,
            "submission_id": submission_id,
            "submission_created_at": submission_created_at,
            "model": self.model,
            "model_slug": self.model_slug,
            "dataset_type": self.dataset_type,
            "dataset_slug": self.dataset_slug,
            "resource_class": self.resource_class,
            "slice_ref": self.slice_ref,
            "task_slug": self.task_slug,
            "question_start": self.question_start,
            "question_end": self.question_end,
            "chunk_index": self.chunk_index,
            "strategy": self.strategy,
            "setting": self.setting,
            "mode": self.mode,
            "force": self.force,
            "generation_run_name": self.generation_run_name,
            "generation_model": self.generation_model,
            "state_dependency_refs": list(self.state_dependency_refs or []),
            "submit_dependency_refs": list(self.submit_dependency_refs or []),
            "argv": list(self.argv),
            "resources": dict(self.resources),
            "task_log_dir": str(task_log_dir),
            "task_stdout": str(task_stdout),
            "task_stderr": str(task_stderr),
            "job_name": f"final5-{self.stage}-{self.task_slug}",
            "submitted_at": "",
            "submitted_job_id": "",
        }


@dataclass(frozen=True)
class ClusterBundlePaths:
    run_dir: Path
    submission_dir: Path
    manifest_path: Path
    submit_path: Path
    local_wrapper_path: Path
    api_wrapper_path: Path
    finalizer_wrapper_path: Path
    log_dir: Path
    state_path: Path
    dashboard_path: Path
    submission_id: str
    submission_created_at: str


def build_bundle_paths(*, stage: str, run_name: str, output_dir: str | Path | None) -> ClusterBundlePaths:
    run_slug = safe_name(run_name)
    run_dir = (Path(output_dir) if output_dir else Path("jobs/generated") / stage / run_slug).resolve()
    submission_created_at = iso_now()
    submission_id = safe_name(f"{submission_created_at}_{uuid4().hex[:8]}")
    submission_dir = run_dir / "submissions" / submission_id
    log_dir = (Path("logs/slurm") / stage / run_slug).resolve()
    return ClusterBundlePaths(
        run_dir=run_dir,
        submission_dir=submission_dir,
        manifest_path=submission_dir / "manifest.json",
        submit_path=submission_dir / "submit_all.sh",
        local_wrapper_path=submission_dir / "run_local_task.sbatch",
        api_wrapper_path=submission_dir / "run_api_task.sbatch",
        finalizer_wrapper_path=submission_dir / "run_finalize_task.sbatch",
        log_dir=log_dir,
        state_path=run_dir / "scheduler_state.json",
        dashboard_path=run_dir / "scheduler_status.html",
        submission_id=submission_id,
        submission_created_at=submission_created_at,
    )


def _topological_order(tasks: list[ClusterTask]) -> list[ClusterTask]:
    tasks_by_ref = {task.slice_ref: task for task in tasks}
    incoming: dict[str, set[str]] = {}
    for task in tasks:
        incoming[task.slice_ref] = {ref for ref in task.submit_dependency_refs or [] if ref in tasks_by_ref}

    ordered: list[ClusterTask] = []
    ready = sorted([task.slice_ref for task in tasks if not incoming[task.slice_ref]])
    while ready:
        slice_ref = ready.pop(0)
        ordered.append(tasks_by_ref[slice_ref])
        for candidate in tasks:
            if slice_ref in incoming[candidate.slice_ref]:
                incoming[candidate.slice_ref].remove(slice_ref)
                if not incoming[candidate.slice_ref]:
                    ready.append(candidate.slice_ref)
                    ready.sort()
    if len(ordered) != len(tasks):
        raise ValueError("Task dependencies contain a cycle.")
    return ordered


def render_manifest(
    *,
    stage: str,
    run_name: str,
    resources: dict[str, Any],
    tasks: list[ClusterTask],
    paths: ClusterBundlePaths,
    concurrency_caps: dict[str, int | None],
) -> str:
    ordered_tasks = _topological_order(tasks)
    finalizers: list[dict[str, Any]] = []
    if stage == "generate":
        models = sorted({task.model for task in ordered_tasks})
        for finalizer_index, model in enumerate(models):
            representative_task = next(task for task in ordered_tasks if task.model == model)
            model_slug = safe_name(model)
            dependency_refs = [task.slice_ref for task in ordered_tasks if task.model == model]
            if not dependency_refs:
                continue
            argv = [
                "materialize-generation-cache",
                "--run-name",
                run_name,
                "--model",
                model,
                "--processed-dataset",
                str(representative_task.argv[representative_task.argv.index("--processed-dataset") + 1]),
            ]
            finalizers.append(
                {
                    "finalizer_index": finalizer_index,
                    "kind": "materialize_generation_cache",
                    "model": model,
                    "dependency_refs": dependency_refs,
                    "argv": argv,
                    "resources": {
                        "partition": resources["partition"],
                        "account": resources["account"],
                        "qos": resources["qos"],
                        "time_limit": "00:30:00",
                        "memory": "8G",
                        "cpus_per_task": 2,
                    },
                    "task_stdout": str(paths.log_dir / f"materialize__{model_slug}__%j.out"),
                    "task_stderr": str(paths.log_dir / f"materialize__{model_slug}__%j.err"),
                    "job_name": f"final5-generate-materialize__{model_slug}",
                }
            )
    payload = {
        "stage": stage,
        "run_name": run_name,
        "submission_id": paths.submission_id,
        "submission_created_at": paths.submission_created_at,
        "task_count": len(ordered_tasks),
        "resources": resources,
        "concurrency_caps": concurrency_caps,
        "finalizers": finalizers,
        "tasks": [
            task.as_dict(
                task_index=index,
                submission_created_at=paths.submission_created_at,
                submission_id=paths.submission_id,
                task_log_dir=paths.log_dir,
            )
            for index, task in enumerate(ordered_tasks)
        ],
    }
    return json.dumps(payload, indent=2) + "\n"


def render_wrapper_script() -> str:
    return WRAPPER_TEMPLATE


def render_finalizer_wrapper_script() -> str:
    return FINALIZER_WRAPPER_TEMPLATE


def render_submit_script(paths: ClusterBundlePaths) -> str:
    return (
        MASTER_SUBMIT_TEMPLATE.replace("__MANIFEST_PATH__", str(paths.manifest_path))
        .replace("__LOCAL_WRAPPER__", str(paths.local_wrapper_path))
        .replace("__API_WRAPPER__", str(paths.api_wrapper_path))
        .replace("__FINALIZER_WRAPPER__", str(paths.finalizer_wrapper_path))
        .replace("__PROJECT_ROOT__", str(Path(__file__).resolve().parent.parent))
    )


def write_bundle(
    *,
    paths: ClusterBundlePaths,
    manifest_text: str,
    submit_text: str,
    local_wrapper_text: str,
    api_wrapper_text: str,
    finalizer_wrapper_text: str,
) -> None:
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.submission_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_path.write_text(manifest_text, encoding="utf-8")
    paths.submit_path.write_text(submit_text, encoding="utf-8")
    paths.local_wrapper_path.write_text(local_wrapper_text, encoding="utf-8")
    paths.api_wrapper_path.write_text(api_wrapper_text, encoding="utf-8")
    paths.finalizer_wrapper_path.write_text(finalizer_wrapper_text, encoding="utf-8")
    paths.submit_path.chmod(0o755)
    paths.local_wrapper_path.chmod(0o755)
    paths.api_wrapper_path.chmod(0o755)
    paths.finalizer_wrapper_path.chmod(0o755)


def submit_bundle(paths: ClusterBundlePaths) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", paths.submit_path.name],
        cwd=str(paths.submission_dir),
        capture_output=True,
        text=True,
        check=False,
    )

__all__ = [
    "ClusterBundlePaths",
    "ClusterTask",
    "build_bundle_paths",
    "render_finalizer_wrapper_script",
    "render_manifest",
    "render_submit_script",
    "render_wrapper_script",
    "submit_bundle",
    "write_bundle",
]
