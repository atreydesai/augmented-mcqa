from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.parsing import extract_answer_letter_from_json


SCORE_NAME = "augmented_mcqa_eval"
CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _prediction_type(
    prediction: str,
    gold_index: int,
    human_indices: list[int],
    model_indices: list[int],
) -> str:
    if not prediction or len(prediction) != 1:
        return "?"
    predicted_index = ord(prediction) - ord("A")
    if predicted_index == gold_index:
        return "G"
    if predicted_index in human_indices:
        return "H"
    if predicted_index in model_indices:
        return "M"
    return "?"


def _as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def _score_payload(sample: dict[str, Any]) -> dict[str, Any] | None:
    scores = sample.get("scores")
    if not isinstance(scores, dict):
        return None
    score = scores.get(SCORE_NAME)
    return score if isinstance(score, dict) else None


def _sample_key(sample: dict[str, Any], member_name: str) -> str:
    sample_id = sample.get("id")
    if isinstance(sample_id, str) and sample_id:
        return sample_id
    metadata = sample.get("metadata")
    if isinstance(metadata, dict):
        metadata_id = metadata.get("sample_id")
        if isinstance(metadata_id, str) and metadata_id:
            return metadata_id
    return member_name


def _valid_letters(sample: dict[str, Any]) -> str:
    choices = sample.get("choices")
    if isinstance(choices, list):
        return CHOICE_LABELS[: len(choices)]
    return CHOICE_LABELS


def _target_letter(sample: dict[str, Any], metadata: dict[str, Any]) -> str:
    target = str(sample.get("target") or metadata.get("gold_answer_letter") or "")
    return target.strip().upper()


def _update_score(sample: dict[str, Any], prediction: str) -> bool:
    score = _score_payload(sample)
    if score is None:
        return False
    metadata = score.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        score["metadata"] = metadata

    target = _target_letter(sample, metadata)
    gold_index = int(metadata.get("gold_index", -1))
    human_indices = _as_int_list(metadata.get("human_option_indices"))
    model_indices = _as_int_list(metadata.get("model_option_indices"))
    prediction_type = _prediction_type(prediction, gold_index, human_indices, model_indices)

    metadata["prediction"] = prediction
    metadata["prediction_type"] = prediction_type
    score["value"] = 1.0 if prediction and prediction == target else 0.0
    score["answer"] = prediction

    sample_metadata = sample.get("metadata")
    if isinstance(sample_metadata, dict):
        evaluation = sample_metadata.get("evaluation")
        if isinstance(evaluation, dict):
            evaluation["prediction"] = prediction
    return True


def _score_state(sample: dict[str, Any], member_name: str) -> dict[str, Any] | None:
    score = _score_payload(sample)
    if score is None:
        return None
    metadata = score.get("metadata")
    if not isinstance(metadata, dict):
        return None

    prediction = str(metadata.get("prediction", "") or "").strip().upper()
    recovered = False
    if not prediction:
        raw_output = str(metadata.get("raw_output", "") or "")
        prediction = extract_answer_letter_from_json(raw_output, _valid_letters(sample))
        if not prediction:
            return {
                "key": _sample_key(sample, member_name),
                "prediction": "",
                "prediction_type": str(metadata.get("prediction_type", "") or "?"),
                "value": score.get("value"),
                "answer": score.get("answer"),
                "recovered": False,
                "correct": False,
                "syncable": False,
            }
        recovered = True
        _update_score(sample, prediction)
        score = _score_payload(sample)
        if score is None:
            return None
        metadata = score.get("metadata")
        if not isinstance(metadata, dict):
            return None

    return {
        "key": _sample_key(sample, member_name),
        "prediction": prediction,
        "prediction_type": str(metadata.get("prediction_type", "") or "?"),
        "value": score.get("value"),
        "answer": score.get("answer", prediction),
        "recovered": recovered,
        "correct": prediction == _target_letter(sample, metadata),
        "syncable": True,
    }


def _states_for_archive(path: Path) -> tuple[dict[str, dict[str, Any]], Counter]:
    states: dict[str, dict[str, Any]] = {}
    counts: Counter = Counter()
    with zipfile.ZipFile(path) as archive:
        for member_name in archive.namelist():
            if not member_name.startswith("samples/") or not member_name.endswith(".json"):
                continue
            counts["samples"] += 1
            sample = json.loads(archive.read(member_name))
            score = _score_payload(sample)
            if score is None:
                continue
            metadata = score.get("metadata")
            if not isinstance(metadata, dict):
                continue
            prediction = str(metadata.get("prediction", "") or "").strip().upper()
            if prediction:
                counts["already_predicted"] += 1
            else:
                counts["blank_prediction"] += 1
            state = _score_state(sample, member_name)
            if state is None:
                continue
            states[state["key"]] = state
            if state["recovered"]:
                counts["recoverable"] += 1
            if state["recovered"] and state["correct"]:
                counts["recoverable_correct"] += 1
    return states, counts


def _set_if_different(container: dict[str, Any], key: str, value: Any) -> bool:
    if container.get(key) == value:
        return False
    container[key] = value
    return True


def _sync_score_dict(score: dict[str, Any], state: dict[str, Any]) -> bool:
    if not state.get("syncable", True):
        return False
    changed = False
    metadata = score.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        score["metadata"] = metadata
        changed = True

    changed = _set_if_different(metadata, "prediction", state["prediction"]) or changed
    changed = _set_if_different(metadata, "prediction_type", state["prediction_type"]) or changed
    changed = _set_if_different(score, "value", state["value"]) or changed
    if "answer" in score or state.get("answer") is not None:
        changed = _set_if_different(score, "answer", state.get("answer")) or changed
    return changed


def _sync_sample_score(sample: dict[str, Any], state: dict[str, Any]) -> bool:
    score = _score_payload(sample)
    if score is None:
        return False
    changed = _sync_score_dict(score, state)
    sample_metadata = sample.get("metadata")
    if isinstance(sample_metadata, dict):
        evaluation = sample_metadata.get("evaluation")
        if isinstance(evaluation, dict):
            changed = _set_if_different(evaluation, "prediction", state["prediction"]) or changed
    return changed


def _sync_summary_tree(node: Any, states: dict[str, dict[str, Any]], source_name: str) -> int:
    changed = 0
    if isinstance(node, list):
        for item in node:
            changed += _sync_summary_tree(item, states, source_name)
    elif isinstance(node, dict):
        key = _sample_key(node, source_name)
        state = states.get(key)
        if state and _sync_sample_score(node, state):
            changed += 1
    return changed


def _sync_reductions(node: Any, states: dict[str, dict[str, Any]]) -> int:
    changed = 0
    if isinstance(node, list):
        for item in node:
            changed += _sync_reductions(item, states)
    elif isinstance(node, dict):
        samples = node.get("samples")
        if isinstance(samples, list):
            for score in samples:
                if not isinstance(score, dict):
                    continue
                sample_id = score.get("sample_id")
                state = states.get(sample_id) if isinstance(sample_id, str) else None
                if state and _sync_score_dict(score, state):
                    changed += 1
        else:
            for value in node.values():
                changed += _sync_reductions(value, states)
    return changed


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stderr(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return math.sqrt(variance) / math.sqrt(n)


def _sync_header(header: dict[str, Any], states: dict[str, dict[str, Any]]) -> int:
    results = header.get("results")
    if not isinstance(results, dict):
        return 0
    scores = results.get("scores")
    if not isinstance(scores, list):
        return 0

    values: list[float] = []
    for state in states.values():
        try:
            values.append(float(state["value"]))
        except (TypeError, ValueError):
            continue

    changed = 0
    for score in scores:
        if not isinstance(score, dict):
            continue
        if score.get("name") != SCORE_NAME and score.get("scorer") != SCORE_NAME:
            continue
        metrics = score.get("metrics")
        if not isinstance(metrics, dict):
            continue
        metric_values = {"mean": _mean(values), "stderr": _stderr(values)}
        for name, value in metric_values.items():
            metric = metrics.get(name)
            if isinstance(metric, dict):
                if metric.get("value") != value:
                    metric["value"] = value
                    changed += 1
    return 1 if changed else 0


def _rewrite_archive(path: Path, states: dict[str, dict[str, Any]], backup_suffix: str | None) -> Counter:
    counts: Counter = Counter()

    with zipfile.ZipFile(path) as source:
        infos = source.infolist()
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            with zipfile.ZipFile(temp_path, "w") as dest:
                for info in infos:
                    data = source.read(info.filename)
                    if info.filename.startswith("samples/") and info.filename.endswith(".json"):
                        sample = json.loads(data)
                        key = _sample_key(sample, info.filename)
                        state = states.get(key)
                        if state and _sync_sample_score(sample, state):
                            data = json.dumps(sample, ensure_ascii=False).encode("utf-8")
                            counts["sample_records_updated"] += 1
                    elif info.filename.startswith("_journal/summaries/") and info.filename.endswith(".json"):
                        summaries = json.loads(data)
                        changed = _sync_summary_tree(summaries, states, info.filename)
                        if changed:
                            data = json.dumps(summaries, ensure_ascii=False).encode("utf-8")
                            counts["summary_records_updated"] += changed
                    elif info.filename == "summaries.json":
                        summaries = json.loads(data)
                        changed = _sync_summary_tree(summaries, states, info.filename)
                        if changed:
                            data = json.dumps(summaries, ensure_ascii=False).encode("utf-8")
                            counts["root_summary_records_updated"] += changed
                    elif info.filename == "reductions.json":
                        reductions = json.loads(data)
                        changed = _sync_reductions(reductions, states)
                        if changed:
                            data = json.dumps(reductions, ensure_ascii=False).encode("utf-8")
                            counts["reduction_records_updated"] += changed
                    elif info.filename == "header.json":
                        header = json.loads(data)
                        changed = _sync_header(header, states) if isinstance(header, dict) else 0
                        if changed:
                            data = json.dumps(header, ensure_ascii=False).encode("utf-8")
                            counts["headers_updated"] += changed
                    dest.writestr(info, data)
            if sum(counts.values()):
                if backup_suffix:
                    backup_path = path.with_name(path.name + backup_suffix)
                    shutil.copy2(path, backup_path)
                    counts["backups_created"] += 1
                shutil.move(temp_path, path)
            else:
                temp_path.unlink(missing_ok=True)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair blank predictions caused by trailing-JSON parse bug in Inspect .eval archives.")
    parser.add_argument("root", type=Path, help="Directory containing .eval archives.")
    parser.add_argument("--apply", action="store_true", help="Rewrite affected archives. Without this flag, only report.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak backups when applying.")
    args = parser.parse_args()

    files = sorted(args.root.rglob("*.eval"))
    total: Counter = Counter(files=len(files))
    affected: list[tuple[Path, Counter]] = []

    for path in files:
        states, counts = _states_for_archive(path)
        total.update(counts)
        if states:
            rewrite_counts = Counter()
            if args.apply:
                suffix = None if args.no_backup else f".parser-repair-{datetime.now().strftime('%Y%m%dT%H%M%S')}.bak"
                rewrite_counts = _rewrite_archive(path, states, suffix)
                total.update(rewrite_counts)
            if counts.get("recoverable") or rewrite_counts:
                affected.append((path, counts + rewrite_counts))

    print(json.dumps({"root": str(args.root), **dict(total), "affected_files": len(affected)}, indent=2, sort_keys=True))
    for path, counts in affected:
        print(json.dumps({"path": str(path), **dict(counts)}, sort_keys=True))


if __name__ == "__main__":
    main()
