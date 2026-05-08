from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from utils.modeling import extra_body_for_model, reasoning_effort_for_model, vllm_server_args


@contextmanager
def temporary_env(updates: dict[str, str]):
    if not updates:
        yield
        return
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def inspect_eval(tasks, *, model: str, log_dir: Path, args):
    from inspect_ai import eval as run_eval

    log_dir.mkdir(parents=True, exist_ok=True)
    env_updates: dict[str, str] = {}
    current_bin = str(Path(sys.executable).parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
    if current_bin and current_bin not in path_entries:
        env_updates["PATH"] = current_bin if not path_entries else f"{current_bin}{os.pathsep}{os.environ['PATH']}"
    if not getattr(args, "model_base_url", None) and "VLLM_DEFAULT_SERVER_ARGS" not in os.environ:
        server_args = vllm_server_args(model)
        if server_args:
            env_updates["VLLM_DEFAULT_SERVER_ARGS"] = json.dumps(server_args)
    extra_body = extra_body_for_model(model)
    with temporary_env(env_updates):
        return run_eval(
            tasks,
            model=model,
            model_base_url=args.model_base_url,
            log_dir=str(log_dir),
            display="plain",
            fail_on_error=False,
            retry_on_error=args.retry_on_error,
            max_connections=args.max_connections,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            reasoning_effort=reasoning_effort_for_model(model, args.reasoning_effort),
            extra_body=extra_body,
            stop_seqs=args.stop_seqs,
        )


def _score_value(score) -> float | None:
    value = getattr(score, "value", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _status_is_success(status) -> bool:
    normalized = str(status or "")
    return not normalized or normalized == "success"


def _sample_scores(sample) -> list[object]:
    scores = getattr(sample, "scores", None) or {}
    return list(dict(scores).values())


def _validated_samples(logs):
    if not logs:
        return None
    samples = []
    for log in logs:
        if not _status_is_success(getattr(log, "status", "")):
            return None
        log_samples = list(getattr(log, "samples", []) or [])
        if not log_samples:
            return None
        samples.extend(log_samples)
    return samples


def generation_logs_succeeded(logs) -> bool:
    samples = _validated_samples(logs)
    if samples is None:
        return False
    for sample in samples:
        scores = _sample_scores(sample)
        if not scores:
            return False
        for score in scores:
            metadata = dict(getattr(score, "metadata", {}) or {})
            if metadata.get("status") != "success":
                return False
            value = _score_value(score)
            if value is None or value < 1.0:
                return False
    return True


def evaluation_logs_completed(logs) -> bool:
    samples = _validated_samples(logs)
    if samples is None:
        return False
    for sample in samples:
        if not _sample_scores(sample):
            return False
    return True
