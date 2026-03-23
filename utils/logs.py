from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Iterator


def find_eval_logs(path: Path | str) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root] if root.suffix == ".eval" else []
    return sorted(root.glob("**/*.eval"))


def read_log(path: Path | str):
    from inspect_ai.log import read_eval_log

    return read_eval_log(str(path))


def iter_eval_logs(path: Path | str, *, kind: str | None = None) -> Iterator[tuple[Path, object]]:
    for log_path in find_eval_logs(path):
        log = read_log(log_path)
        metadata = getattr(log.eval, "metadata", {}) or {}
        if kind is not None and metadata.get("kind") != kind:
            continue
        yield log_path, log


def _coerce_score_value(score: Any) -> float | None:
    value: Any
    if isinstance(score, dict):
        value = score.get("value")
    else:
        value = getattr(score, "value", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_log_summary_from_archive(path: Path) -> dict[str, Any] | None:
    try:
        with zipfile.ZipFile(path) as archive:
            header = json.loads(archive.read("header.json"))
            summaries = json.loads(archive.read("summaries.json"))
    except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError):
        return None

    eval_payload = header.get("eval", {}) if isinstance(header, dict) else {}
    metadata = dict(eval_payload.get("metadata", {}) or {})
    stats = header.get("stats", {}) if isinstance(header, dict) else {}
    score_values: list[float] = []
    sample_statuses: list[str] = []
    summary_count = 0
    if isinstance(summaries, list):
        summary_count = len(summaries)
        for summary in summaries:
            if not isinstance(summary, dict):
                continue
            scores = summary.get("scores", {})
            if not isinstance(scores, dict) or not scores:
                continue
            score = next(iter(scores.values()))
            score_metadata = score.get("metadata", {}) if isinstance(score, dict) else {}
            sample_statuses.append(str(score_metadata.get("status", "") or ""))
            value = _coerce_score_value(score)
            if value is not None:
                score_values.append(value)

    return {
        "status": str(header.get("status", "") or ""),
        "metadata": metadata,
        "completed_at": str(stats.get("completed_at", "") or ""),
        "score_values": score_values,
        "sample_statuses": sample_statuses,
        "summary_count": summary_count,
    }


def _read_log_payload_from_archive(path: Path) -> dict[str, Any] | None:
    try:
        with zipfile.ZipFile(path) as archive:
            header = json.loads(archive.read("header.json"))
            summaries = json.loads(archive.read("summaries.json"))
    except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError):
        return None

    eval_payload = header.get("eval", {}) if isinstance(header, dict) else {}
    metadata = dict(eval_payload.get("metadata", {}) or {})
    stats = header.get("stats", {}) if isinstance(header, dict) else {}
    return {
        "status": str(header.get("status", "") or ""),
        "metadata": metadata,
        "completed_at": str(stats.get("completed_at", "") or ""),
        "summaries": summaries if isinstance(summaries, list) else [],
    }


def _normalize_log_summary_from_object(log: Any) -> dict[str, Any]:
    metadata = dict(getattr(log.eval, "metadata", {}) or {})
    score_values: list[float] = []
    sample_statuses: list[str] = []
    samples = list(getattr(log, "samples", []) or [])
    for sample in samples:
        scores = getattr(sample, "scores", None)
        if not scores:
            continue
        score = next(iter(scores.values()))
        score_metadata = dict(getattr(score, "metadata", {}) or {})
        sample_statuses.append(str(score_metadata.get("status", "") or ""))
        value = _coerce_score_value(score)
        if value is not None:
            score_values.append(value)
    stats = getattr(log, "stats", None)
    return {
        "status": str(getattr(log, "status", "") or ""),
        "metadata": metadata,
        "completed_at": str(getattr(stats, "completed_at", "") or ""),
        "score_values": score_values,
        "sample_statuses": sample_statuses,
        "summary_count": len(samples),
    }


def read_log_summary(path: Path | str) -> dict[str, Any] | None:
    log_path = Path(path)
    summary = _normalize_log_summary_from_archive(log_path)
    if summary is not None:
        return summary
    try:
        return _normalize_log_summary_from_object(read_log(log_path))
    except Exception:
        return None


def _read_log_payload_from_object(log: Any) -> dict[str, Any]:
    metadata = dict(getattr(log.eval, "metadata", {}) or {})
    stats = getattr(log, "stats", None)
    samples: list[dict[str, Any]] = []
    for sample in list(getattr(log, "samples", []) or []):
        sample_payload: dict[str, Any] = {
            "id": getattr(sample, "id", ""),
            "input": getattr(sample, "input", ""),
            "choices": list(getattr(sample, "choices", []) or []),
            "target": getattr(sample, "target", ""),
            "metadata": dict(getattr(sample, "metadata", {}) or {}),
            "scores": {},
        }
        scores = getattr(sample, "scores", None) or {}
        for name, score in dict(scores).items():
            sample_payload["scores"][str(name)] = {
                "value": getattr(score, "value", None),
                "answer": getattr(score, "answer", None),
                "explanation": getattr(score, "explanation", None),
                "metadata": dict(getattr(score, "metadata", {}) or {}),
            }
        samples.append(sample_payload)
    return {
        "status": str(getattr(log, "status", "") or ""),
        "metadata": metadata,
        "completed_at": str(getattr(stats, "completed_at", "") or ""),
        "summaries": samples,
    }


def read_log_payload(path: Path | str) -> dict[str, Any] | None:
    log_path = Path(path)
    payload = _read_log_payload_from_archive(log_path)
    if payload is not None:
        return payload
    try:
        return _read_log_payload_from_object(read_log(log_path))
    except Exception:
        return None


def iter_log_payloads(path: Path | str, *, kind: str | None = None) -> Iterator[tuple[Path, dict[str, Any]]]:
    for log_path in find_eval_logs(path):
        payload = read_log_payload(log_path)
        if payload is None:
            continue
        metadata = dict(payload.get("metadata", {}) or {})
        if kind is not None and metadata.get("kind") != kind:
            continue
        yield log_path, payload


def iter_log_summaries(path: Path | str, *, kind: str | None = None) -> Iterator[tuple[Path, dict[str, Any]]]:
    for log_path in find_eval_logs(path):
        summary = read_log_summary(log_path)
        if summary is None:
            continue
        metadata = dict(summary.get("metadata", {}) or {})
        if kind is not None and metadata.get("kind") != kind:
            continue
        yield log_path, summary
