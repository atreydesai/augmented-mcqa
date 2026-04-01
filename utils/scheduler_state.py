from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.logs import iter_log_summaries
from utils.modeling import safe_name
from utils.recipes import get_setting_recipe, schedulable_generation_strategies

SCHEDULABLE_GENERATION_STRATEGIES = schedulable_generation_strategies()

GENERATION_STRATEGY_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    strategy: (
        (recipe.prerequisite_setting,) if recipe.prerequisite_setting else ()
    )
    for strategy in SCHEDULABLE_GENERATION_STRATEGIES
    for recipe in [get_setting_recipe(strategy)]
}

EVALUATION_SETTING_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "human_from_scratch": (),
    "model_from_scratch": ("model_from_scratch",),
    "augment_human": ("augment_human",),
    "augment_model": ("augment_model",),
    "augment_ablation": ("augment_ablation",),
}

STATUS_PENDING = "pending"
STATUS_PLANNED = "planned"
STATUS_CURRENT = "current"
STATUS_STALE = "stale"
STATUS_FAILED = "failed"


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def resource_class_for_model(model: str) -> str:
    return "local" if str(model).startswith(("vllm/", "hf/")) else "api"


def _joined_safe_parts(separator: str, *parts: object) -> str:
    return separator.join(str(part) if isinstance(part, int) else safe_name(str(part)) for part in parts)


def chunk_ranges(total_questions: int, questions_per_job: int | None) -> list[tuple[int, int, int]]:
    if total_questions <= 0:
        return []
    if questions_per_job is None or questions_per_job <= 0 or questions_per_job >= total_questions:
        return [(0, 0, total_questions)]
    return [
        (chunk_index, start, min(total_questions, start + questions_per_job))
        for chunk_index, start in enumerate(range(0, total_questions, questions_per_job))
    ]


def generation_slice_ref(
    *,
    run_name: str,
    model: str,
    dataset_type: str,
    strategy: str,
    question_start: int,
    question_end: int,
) -> str:
    return _joined_safe_parts("|", "generation", run_name, model, dataset_type, strategy, question_start, question_end)


def evaluation_slice_ref(
    *,
    run_name: str,
    model: str,
    dataset_type: str,
    setting: str,
    mode: str,
    question_start: int,
    question_end: int,
) -> str:
    return _joined_safe_parts("|", "evaluation", run_name, model, dataset_type, setting, mode, question_start, question_end)


def task_slug(
    *,
    stage: str,
    model: str,
    dataset_type: str,
    question_start: int,
    question_end: int,
    strategy: str | None = None,
    setting: str | None = None,
    mode: str | None = None,
) -> str:
    parts = [stage, model, dataset_type]
    if strategy:
        parts.append(strategy)
    if setting:
        parts.append(setting)
    if mode:
        parts.append(mode)
    return _joined_safe_parts(
        "__",
        *parts,
        f"{question_start}-{max(question_start, question_end - 1)}",
    )


def _attempt_status_from_summary(log_summary: dict[str, Any], *, kind: str) -> tuple[str, int]:
    log_status = str(log_summary.get("status", "") or "")
    scores = [float(value) for value in list(log_summary.get("score_values", []) or [])]
    sample_statuses = [str(value or "") for value in list(log_summary.get("sample_statuses", []) or [])]
    has_sample_errors = any(status and status != "success" for status in sample_statuses)
    if log_status and log_status != "success":
        return "failed", len(scores)
    if kind == "generation":
        status = "success" if scores and all(value >= 1.0 for value in scores) and not has_sample_errors else "failed"
        return status, len(scores)
    total_samples = int(log_summary.get("summary_count", 0) or 0)
    status = "success" if total_samples > 0 and len(scores) == total_samples else "failed"
    return status, len(scores)


def _attempt_record(log_path: Path | str, log_summary: dict[str, Any], *, kind: str) -> dict[str, Any] | None:
    eval_metadata = dict(log_summary.get("metadata", {}) or {})
    slice_ref = str(eval_metadata.get("slice_ref", "") or "")
    if not slice_ref:
        return None
    status, sample_count = _attempt_status_from_summary(log_summary, kind=kind)
    return {
        "slice_ref": slice_ref,
        "status": status,
        "completed_at": str(log_summary.get("completed_at", "") or ""),
        "log_path": str(log_path),
        "sample_count": sample_count,
    }


def load_scheduler_manifests(run_dir: Path | str) -> list[dict[str, Any]]:
    root = Path(run_dir)
    manifests: list[dict[str, Any]] = []
    if not root.exists():
        return manifests
    for path in sorted(root.glob("submissions/*/manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = str(path)
        manifests.append(payload)
    return manifests


def collect_slice_attempts(log_dir: Path | str, *, kind: str) -> dict[str, list[dict[str, Any]]]:
    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for log_path, log_summary in iter_log_summaries(log_dir, kind=kind):
        record = _attempt_record(log_path, log_summary, kind=kind)
        if record is None:
            continue
        attempts[str(record["slice_ref"])].append(record)
    for records in attempts.values():
        records.sort(key=lambda record: record.get("completed_at") or "")
    return dict(attempts)


def _latest_record(records: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    latest_ts: datetime | None = None
    for record in records:
        record_ts = parse_iso(str(record.get(key, "") or ""))
        if record_ts is None:
            continue
        if latest is None or latest_ts is None or record_ts >= latest_ts:
            latest = record
            latest_ts = record_ts
    return latest


def _live_slurm_job_ids(job_ids: list[str]) -> set[str]:
    normalized = [job_id.strip() for job_id in job_ids if str(job_id or "").strip()]
    if not normalized:
        return set()
    try:
        result = subprocess.run(
            ["squeue", "-h", "-o", "%i", "-j", ",".join(normalized)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return set()
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _scheduler_state_from_planned(
    *,
    stage: str,
    run_name: str,
    submission_count: int,
    planned: dict[str, list[dict[str, Any]]],
    attempts_by_slice: dict[str, list[dict[str, Any]]],
    live_job_ids: set[str],
) -> dict[str, Any]:
    slices: list[dict[str, Any]] = []
    all_slice_refs = sorted(set(planned.keys()) | set(attempts_by_slice.keys()))
    for slice_ref in all_slice_refs:
        plans = sorted(planned.get(slice_ref, []), key=lambda plan: str(plan.get("submission_created_at", "")))
        latest_plan = dict(plans[-1]) if plans else {"slice_ref": slice_ref}
        attempts = list(attempts_by_slice.get(slice_ref, []))
        latest_attempt = _latest_record(attempts, "completed_at")
        latest_success = _latest_record([attempt for attempt in attempts if attempt.get("status") == "success"], "completed_at")

        latest_submission_at = parse_iso(str(latest_plan.get("submission_created_at", "") or ""))
        latest_submitted_at = parse_iso(str(latest_plan.get("submitted_at", "") or ""))
        latest_attempt_at = parse_iso(str((latest_attempt or {}).get("completed_at", "") or ""))
        latest_success_at = parse_iso(str((latest_success or {}).get("completed_at", "") or ""))
        latest_submitted_job_id = str(latest_plan.get("submitted_job_id", "") or "")
        submitted_job_live = not latest_submitted_job_id or latest_submitted_job_id in live_job_ids
        pending = (
            latest_submitted_at is not None
            and (latest_attempt_at is None or latest_submitted_at > latest_attempt_at)
            and submitted_job_live
        )

        if pending:
            status = STATUS_PENDING
        elif latest_attempt and latest_attempt.get("status") == "failed":
            status = STATUS_FAILED
        elif latest_success is not None:
            status = STATUS_CURRENT
        else:
            status = STATUS_PLANNED

        latest_force_request = _latest_record(
            [plan for plan in plans if plan.get("force") and plan.get("submitted_at")],
            "submitted_at",
        )
        dependency_change_candidates: list[datetime] = []
        if latest_attempt_at is not None:
            dependency_change_candidates.append(latest_attempt_at)
        if latest_force_request is not None:
            forced_at = parse_iso(str(latest_force_request.get("submitted_at", "") or ""))
            if forced_at is not None:
                dependency_change_candidates.append(forced_at)
        dependency_change_at = max(dependency_change_candidates) if dependency_change_candidates else None

        slices.append(
            {
                **latest_plan,
                "slice_ref": slice_ref,
                "submission_count": len(plans),
                "latest_attempt": latest_attempt,
                "latest_success": latest_success,
                "latest_submitted_at": latest_submitted_at.isoformat() if latest_submitted_at else "",
                "latest_attempt_at": latest_attempt_at.isoformat() if latest_attempt_at else "",
                "latest_success_at": latest_success_at.isoformat() if latest_success_at else "",
                "dependency_change_at": dependency_change_at.isoformat() if dependency_change_at else "",
                "status": status,
            }
        )

    indexed = {entry["slice_ref"]: entry for entry in slices}
    for entry in slices:
        if entry["status"] != STATUS_CURRENT:
            continue
        latest_success_at = parse_iso(str(entry.get("latest_success_at", "") or ""))
        if latest_success_at is None:
            continue
        for dependency_ref in entry.get("state_dependency_refs", []):
            dependency = indexed.get(str(dependency_ref))
            if dependency is None:
                continue
            dependency_change_at = parse_iso(str(dependency.get("dependency_change_at", "") or ""))
            if dependency_change_at is not None and dependency_change_at > latest_success_at:
                entry["status"] = STATUS_STALE
                break

    return {
        "stage": stage,
        "run_name": run_name,
        "generated_at": iso_now(),
        "submission_count": submission_count,
        "slice_count": len(slices),
        "slices": slices,
    }


def build_scheduler_state(
    *,
    manifests: list[dict[str, Any]],
    attempts_by_slice: dict[str, list[dict[str, Any]]],
    live_job_ids: set[str] | None = None,
) -> dict[str, Any]:
    planned: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for manifest in manifests:
        for task in manifest.get("tasks", []):
            enriched = dict(task)
            enriched["submission_id"] = manifest.get("submission_id")
            enriched["submission_created_at"] = manifest.get("submission_created_at")
            enriched["submission_path"] = manifest.get("_path")
            planned[str(task["slice_ref"])].append(enriched)

    if live_job_ids is None:
        submitted_job_ids = [
            str(task.get("submitted_job_id", "") or "")
            for tasks in planned.values()
            for task in tasks
        ]
        live_job_ids = _live_slurm_job_ids(submitted_job_ids)

    return _scheduler_state_from_planned(
        stage=str(manifests[-1].get("stage") or "") if manifests else "",
        run_name=str(manifests[-1].get("run_name") or "") if manifests else "",
        submission_count=len(manifests),
        planned=planned,
        attempts_by_slice=attempts_by_slice,
        live_job_ids=live_job_ids,
    )


def build_scheduler_state_from_tasks(
    *,
    stage: str,
    run_name: str,
    planned_tasks: list[dict[str, Any]],
    attempts_by_slice: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    planned: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in planned_tasks:
        slice_ref = str(task.get("slice_ref", "") or "")
        if not slice_ref:
            continue
        planned[slice_ref].append(dict(task))
    return _scheduler_state_from_planned(
        stage=stage,
        run_name=run_name,
        submission_count=0,
        planned=planned,
        attempts_by_slice=attempts_by_slice,
        live_job_ids=set(),
    )


__all__ = [
    "EVALUATION_SETTING_DEPENDENCIES",
    "GENERATION_STRATEGY_DEPENDENCIES",
    "SCHEDULABLE_GENERATION_STRATEGIES",
    "STATUS_CURRENT",
    "STATUS_FAILED",
    "STATUS_PENDING",
    "STATUS_PLANNED",
    "STATUS_STALE",
    "build_scheduler_state",
    "build_scheduler_state_from_tasks",
    "chunk_ranges",
    "collect_slice_attempts",
    "evaluation_slice_ref",
    "generation_slice_ref",
    "iso_now",
    "load_scheduler_manifests",
    "resource_class_for_model",
    "task_slug",
]
