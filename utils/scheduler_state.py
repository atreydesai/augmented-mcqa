from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.constants import FINAL5_SETTINGS, MODE_CHOICES
from utils.logs import iter_eval_logs
from utils.modeling import safe_name

SCHEDULABLE_GENERATION_STRATEGIES = (
    "model_from_scratch",
    "augment_human",
    "augment_model",
    "augment_ablation",
)

GENERATION_STRATEGY_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "model_from_scratch": (),
    "augment_human": (),
    "augment_model": ("model_from_scratch",),
    "augment_ablation": (),
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
    return "local" if str(model).startswith("vllm/") else "api"


def chunk_ranges(total_questions: int, questions_per_job: int | None) -> list[tuple[int, int, int]]:
    if total_questions <= 0:
        return []
    if questions_per_job is None or questions_per_job <= 0 or questions_per_job >= total_questions:
        return [(0, 0, total_questions)]
    ranges: list[tuple[int, int, int]] = []
    start = 0
    chunk_index = 0
    while start < total_questions:
        end = min(total_questions, start + questions_per_job)
        ranges.append((chunk_index, start, end))
        start = end
        chunk_index += 1
    return ranges


def generation_slice_ref(
    *,
    run_name: str,
    model: str,
    dataset_type: str,
    strategy: str,
    question_start: int,
    question_end: int,
) -> str:
    return "|".join(
        [
            "generation",
            safe_name(run_name),
            safe_name(model),
            safe_name(dataset_type),
            safe_name(strategy),
            str(question_start),
            str(question_end),
        ]
    )


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
    return "|".join(
        [
            "evaluation",
            safe_name(run_name),
            safe_name(model),
            safe_name(dataset_type),
            safe_name(setting),
            safe_name(mode),
            str(question_start),
            str(question_end),
        ]
    )


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
    parts = [stage, safe_name(model), safe_name(dataset_type)]
    if strategy:
        parts.append(safe_name(strategy))
    if setting:
        parts.append(safe_name(setting))
    if mode:
        parts.append(safe_name(mode))
    parts.append(f"{question_start}-{max(question_start, question_end - 1)}")
    return "__".join(parts)


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
    for log_path, log in iter_eval_logs(log_dir, kind=kind):
        eval_metadata = dict(getattr(log.eval, "metadata", {}) or {})
        slice_ref = str(eval_metadata.get("slice_ref", "") or "")
        if not slice_ref:
            continue
        log_status = str(getattr(log, "status", "") or "")
        scores = []
        for sample in getattr(log, "samples", []):
            if not getattr(sample, "scores", None):
                continue
            score = next(iter(sample.scores.values()))
            value = getattr(score, "value", None)
            if value is not None:
                scores.append(float(value))
        if log_status and log_status != "success":
            status = "failed"
        elif kind == "generation":
            status = "success" if scores and all(value >= 1.0 for value in scores) else "failed"
        else:
            total_samples = len(getattr(log, "samples", []) or [])
            status = "success" if total_samples > 0 and len(scores) == total_samples else "failed"
        completed_at = str(getattr(log, "completed_at", "") or "")
        attempts[slice_ref].append(
            {
                "slice_ref": slice_ref,
                "status": status,
                "completed_at": completed_at,
                "log_path": str(log_path),
                "sample_count": len(scores),
            }
        )
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


def build_scheduler_state(
    *,
    manifests: list[dict[str, Any]],
    attempts_by_slice: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    planned: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for manifest in manifests:
        for task in manifest.get("tasks", []):
            enriched = dict(task)
            enriched["submission_id"] = manifest.get("submission_id")
            enriched["submission_created_at"] = manifest.get("submission_created_at")
            enriched["submission_path"] = manifest.get("_path")
            planned[str(task["slice_ref"])].append(enriched)

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
        pending = latest_submitted_at is not None and (latest_attempt_at is None or latest_submitted_at > latest_attempt_at)

        if pending:
            status = STATUS_PENDING
        elif latest_attempt and latest_attempt.get("status") == "failed":
            status = STATUS_FAILED
        elif latest_success is not None:
            status = STATUS_CURRENT
        elif plans:
            status = STATUS_PLANNED
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
        "stage": manifests[-1].get("stage") if manifests else "",
        "run_name": manifests[-1].get("run_name") if manifests else "",
        "generated_at": iso_now(),
        "submission_count": len(manifests),
        "slice_count": len(slices),
        "slices": slices,
    }


def render_scheduler_dashboard(state: dict[str, Any]) -> str:
    slices = list(state.get("slices", []))
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"columns": set(), "rows": defaultdict(dict)})
    for entry in slices:
        group_key = (str(entry.get("model", "")), str(entry.get("dataset_type", "")))
        if entry.get("stage") == "generation":
            column = str(entry.get("strategy", ""))
        else:
            column = f"{entry.get('setting', '')}:{entry.get('mode', '')}"
        chunk_end = int(entry.get("question_end", 0)) - 1
        row_label = f"{int(entry.get('question_start', 0))}-{max(int(entry.get('question_start', 0)), chunk_end)}"
        groups[group_key]["columns"].add(column)
        groups[group_key]["rows"][row_label][column] = entry

    legend = {
        STATUS_PLANNED: "#6e7781",
        STATUS_PENDING: "#d73a49",
        STATUS_CURRENT: "#2ea043",
        STATUS_STALE: "#8250df",
        STATUS_FAILED: "#fb8500",
    }
    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>Scheduler Status: {state.get('stage')} {state.get('run_name')}</title>",
        "<style>",
        "body{font-family:Menlo,Monaco,monospace;margin:24px;background:#f7f7f8;color:#111;}",
        "h1,h2{margin:0 0 12px 0;}",
        ".meta{margin-bottom:20px;color:#555;}",
        ".legend{display:flex;gap:12px;margin:12px 0 24px 0;flex-wrap:wrap;}",
        ".legend span{display:inline-flex;align-items:center;gap:8px;}",
        ".swatch{width:14px;height:14px;border-radius:3px;display:inline-block;}",
        ".group{margin:0 0 28px 0;padding:16px;border:1px solid #ddd;border-radius:10px;background:#fff;}",
        "table{border-collapse:collapse;width:100%;}",
        "th,td{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top;}",
        "th{background:#fafafa;}",
        ".cell{color:#fff;font-weight:700;border-radius:6px;padding:6px 8px;display:block;}",
        ".small{font-size:12px;font-weight:400;opacity:.9;}",
        "</style></head><body>",
        f"<h1>{state.get('stage')} scheduler status</h1>",
        f"<div class='meta'>run={state.get('run_name')} | submissions={state.get('submission_count')} | slices={state.get('slice_count')} | generated_at={state.get('generated_at')}</div>",
        "<div class='legend'>",
    ]
    for status, color in legend.items():
        html.append(f"<span><i class='swatch' style='background:{color}'></i>{status}</span>")
    html.append("</div>")

    for (model, dataset_type), payload in sorted(groups.items()):
        columns = sorted(payload["columns"])
        rows = payload["rows"]
        html.append(f"<div class='group'><h2>{model} | {dataset_type}</h2><table>")
        html.append("<tr><th>Questions</th>" + "".join(f"<th>{column}</th>" for column in columns) + "</tr>")
        for row_label in sorted(rows.keys(), key=lambda value: int(value.split("-", 1)[0])):
            html.append(f"<tr><th>{row_label}</th>")
            row = rows[row_label]
            for column in columns:
                entry = row.get(column)
                if entry is None:
                    html.append("<td></td>")
                    continue
                color = legend[str(entry.get("status"))]
                latest_attempt = entry.get("latest_attempt") or {}
                html.append(
                    "<td>"
                    f"<span class='cell' style='background:{color}'>{entry.get('status')}"
                    f"<span class='small'><br>{entry.get('task_slug')}</span>"
                    f"<span class='small'><br>latest={latest_attempt.get('status', 'none')}</span>"
                    "</span></td>"
                )
            html.append("</tr>")
        html.append("</table></div>")

    html.append("</body></html>")
    return "\n".join(html)


@dataclass(frozen=True)
class RenderedStatePaths:
    state_path: Path
    dashboard_path: Path


__all__ = [
    "EVALUATION_SETTING_DEPENDENCIES",
    "GENERATION_STRATEGY_DEPENDENCIES",
    "RenderedStatePaths",
    "SCHEDULABLE_GENERATION_STRATEGIES",
    "STATUS_CURRENT",
    "STATUS_FAILED",
    "STATUS_PENDING",
    "STATUS_PLANNED",
    "STATUS_STALE",
    "build_scheduler_state",
    "chunk_ranges",
    "collect_slice_attempts",
    "evaluation_slice_ref",
    "generation_slice_ref",
    "iso_now",
    "load_scheduler_manifests",
    "render_scheduler_dashboard",
    "resource_class_for_model",
    "task_slug",
]
