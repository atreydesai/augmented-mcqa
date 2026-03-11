from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.modeling import safe_name


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={bootstrap_log_dir}/{job_name}__%A_%a.bootstrap.out
#SBATCH --error={bootstrap_log_dir}/{job_name}__%A_%a.bootstrap.err
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --qos={qos}
#SBATCH --time={time_limit}
#SBATCH --mem={memory}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gres=gpu:{gpu_type}:1
#SBATCH --array={array_spec}

set -euo pipefail

PROJECT_ROOT="${{PROJECT_ROOT:-$SLURM_SUBMIT_DIR}}"
cd "$PROJECT_ROOT"
PYTHON_BIN="${{PYTHON_BIN:-python}}"

"$PYTHON_BIN" - <<'PY' "{manifest_path}" "$SLURM_ARRAY_TASK_ID" "$PROJECT_ROOT" "$PYTHON_BIN" "{task_log_dir}"
import json
import os
import subprocess
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
task_index = int(sys.argv[2])
project_root = Path(sys.argv[3])
python_bin = sys.argv[4]
task_log_dir = Path(sys.argv[5])

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
task = manifest["tasks"][task_index]
job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID", "job")
array_id = os.environ.get("SLURM_ARRAY_TASK_ID", str(task_index))
stdout_path = task_log_dir / f"{{task['model_slug']}}__{{task['dataset_slug']}}__{{job_id}}_{{array_id}}.out"
stderr_path = task_log_dir / f"{{task['model_slug']}}__{{task['dataset_slug']}}__{{job_id}}_{{array_id}}.err"
stdout_path.parent.mkdir(parents=True, exist_ok=True)

cmd = [python_bin, "main.py", *task["argv"]]
print(f"Running: {{' '.join(cmd)}}")
print(f"Stdout: {{stdout_path}}")
print(f"Stderr: {{stderr_path}}")

with stdout_path.open("a", encoding="utf-8") as out, stderr_path.open("a", encoding="utf-8") as err:
    rc = subprocess.run(cmd, cwd=str(project_root), stdout=out, stderr=err, check=False).returncode
sys.exit(rc)
PY
"""


SUBMIT_TEMPLATE = """#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR"
sbatch "{sbatch_name}"
"""


@dataclass(frozen=True)
class ClusterTask:
    stage: str
    model: str
    model_slug: str
    dataset_type: str
    dataset_slug: str
    argv: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "model": self.model,
            "model_slug": self.model_slug,
            "dataset_type": self.dataset_type,
            "dataset_slug": self.dataset_slug,
            "argv": list(self.argv),
        }


@dataclass(frozen=True)
class ClusterBundlePaths:
    bundle_dir: Path
    manifest_path: Path
    sbatch_path: Path
    submit_path: Path
    log_dir: Path
    bootstrap_log_dir: Path


def build_bundle_paths(*, stage: str, run_name: str, output_dir: str | Path | None) -> ClusterBundlePaths:
    run_slug = safe_name(run_name)
    if output_dir:
        bundle_dir = Path(output_dir)
    else:
        bundle_dir = Path("jobs/generated") / stage / run_slug
    log_dir = Path("logs/slurm") / stage / run_slug
    bootstrap_log_dir = log_dir / "_bootstrap"
    return ClusterBundlePaths(
        bundle_dir=bundle_dir,
        manifest_path=bundle_dir / "manifest.json",
        sbatch_path=bundle_dir / f"final5-{stage}-{run_slug}.sbatch",
        submit_path=bundle_dir / "submit_all.sh",
        log_dir=log_dir,
        bootstrap_log_dir=bootstrap_log_dir,
    )


def render_array_spec(task_count: int, gpu_count: int | None) -> str:
    if task_count <= 0:
        raise ValueError("task_count must be > 0")
    spec = f"0-{task_count - 1}"
    if gpu_count is not None:
        if gpu_count <= 0:
            raise ValueError("gpu_count must be > 0 when provided")
        spec = f"{spec}%{gpu_count}"
    return spec


def render_manifest(
    *,
    stage: str,
    run_name: str,
    resources: dict[str, Any],
    tasks: list[ClusterTask],
) -> str:
    payload = {
        "stage": stage,
        "run_name": run_name,
        "task_count": len(tasks),
        "resources": resources,
        "tasks": [task.as_dict() for task in tasks],
    }
    return json.dumps(payload, indent=2) + "\n"


def render_sbatch(
    *,
    paths: ClusterBundlePaths,
    stage: str,
    run_name: str,
    resources: dict[str, Any],
    task_count: int,
    gpu_count: int | None,
) -> str:
    return SBATCH_TEMPLATE.format(
        job_name=f"final5-{stage}-{safe_name(run_name)}",
        bootstrap_log_dir=str(paths.bootstrap_log_dir),
        partition=resources["partition"],
        account=resources["account"],
        qos=resources["qos"],
        time_limit=resources["time_limit"],
        memory=resources["memory"],
        cpus_per_task=resources["cpus_per_task"],
        gpu_type=resources["gpu_type"],
        array_spec=render_array_spec(task_count, gpu_count),
        manifest_path=str(paths.manifest_path),
        task_log_dir=str(paths.log_dir),
    )


def render_submit_script(paths: ClusterBundlePaths) -> str:
    return SUBMIT_TEMPLATE.format(sbatch_name=paths.sbatch_path.name)


def write_bundle(
    *,
    paths: ClusterBundlePaths,
    manifest_text: str,
    sbatch_text: str,
    submit_text: str,
) -> None:
    paths.bundle_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    paths.bootstrap_log_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_path.write_text(manifest_text, encoding="utf-8")
    paths.sbatch_path.write_text(sbatch_text, encoding="utf-8")
    paths.submit_path.write_text(submit_text, encoding="utf-8")
    paths.submit_path.chmod(0o755)


def submit_bundle(paths: ClusterBundlePaths) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sbatch", paths.sbatch_path.name],
        cwd=str(paths.bundle_dir),
        capture_output=True,
        text=True,
        check=False,
    )


__all__ = [
    "ClusterBundlePaths",
    "ClusterTask",
    "build_bundle_paths",
    "render_array_spec",
    "render_manifest",
    "render_sbatch",
    "render_submit_script",
    "submit_bundle",
    "write_bundle",
]
