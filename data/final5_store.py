from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from inspect_ai.dataset import MemoryDataset, Sample

from config import ACTIVE_DATASET_TYPES
from utils.constants import CHOICE_LABELS, FINAL5_SETTINGS, MODE_CHOICES
from utils.logs import iter_eval_logs
from utils.scheduler_state import SCHEDULABLE_GENERATION_STRATEGIES
from utils.sharding import sample_id_for_row, select_shard


def _latest_mtime(path: Path, *, suffix: str | None = None) -> float | None:
    if not path.exists():
        return None
    if path.is_file():
        if suffix is not None and path.suffix != suffix:
            return None
        return path.stat().st_mtime

    latest: float | None = None
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        if suffix is not None and candidate.suffix != suffix:
            continue
        candidate_mtime = candidate.stat().st_mtime
        if latest is None or candidate_mtime > latest:
            latest = candidate_mtime
    return latest


def _load_dataset_dict(path: Path | str):
    dataset_path = Path(path)
    dataset_dict_file = dataset_path / "dataset_dict.json"
    if dataset_dict_file.exists():
        payload = json.loads(dataset_dict_file.read_text(encoding="utf-8"))
        rebuilt: dict[str, Dataset] = {}
        for split_name in payload.get("splits", []):
            split_path = dataset_path / split_name
            state_path = split_path / "state.json"
            if not state_path.exists():
                continue
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("_data_files"):
                rebuilt[split_name] = Dataset.load_from_disk(str(split_path))
            else:
                rebuilt[split_name] = Dataset.from_list([])
        return DatasetDict(rebuilt)

    dataset = load_from_disk(str(dataset_path))
    if isinstance(dataset, DatasetDict):
        return dataset
    if hasattr(dataset, "keys"):
        return dataset
    raise TypeError(f"Expected DatasetDict at {path}")


def _answer_text(row: dict[str, Any]) -> str:
    answer = str(row.get("answer", "") or "").strip()
    if answer:
        return answer
    choices_answer = row.get("choices_answer") or []
    if isinstance(choices_answer, list) and choices_answer:
        return str(choices_answer[0]).strip()
    return ""


def iter_processed_rows(
    processed_dataset_path: Path | str,
    dataset_types: list[str] | None = None,
    *,
    question_start: int = 0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    dataset_dict = _load_dataset_dict(processed_dataset_path)
    wanted = dataset_types or list(ACTIVE_DATASET_TYPES)
    rows: list[dict[str, Any]] = []
    for dataset_type in wanted:
        if dataset_type not in dataset_dict:
            continue
        split = dataset_dict[dataset_type]
        selected_for_dataset = 0
        for row_index, row in enumerate(split):
            if row_index < question_start:
                continue
            if limit is not None and selected_for_dataset >= limit:
                break
            payload = dict(row)
            payload["dataset_type"] = dataset_type
            payload["row_index"] = row_index
            payload["sample_id"] = sample_id_for_row(dataset_type, payload, row_index)
            payload["answer"] = _answer_text(payload)
            rows.append(payload)
            selected_for_dataset += 1
    return rows


def build_generation_dataset(
    processed_dataset_path: Path | str,
    *,
    strategy: str = "model_from_scratch",
    dataset_types: list[str] | None = None,
    question_start: int = 0,
    limit: int | None = None,
    generation_log_dir: Path | str | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
    shard_strategy: str = "contiguous",
) -> MemoryDataset:
    if strategy not in SCHEDULABLE_GENERATION_STRATEGIES:
        raise ValueError(f"Unknown schedulable generation strategy: {strategy}")

    rows = iter_processed_rows(
        processed_dataset_path,
        dataset_types=dataset_types,
        question_start=question_start,
        limit=limit,
    )
    rows = select_shard(rows, shard_count=shard_count, shard_index=shard_index, strategy=shard_strategy)
    prior_payloads = _generation_payloads(generation_log_dir) if generation_log_dir else {}

    samples: list[Sample] = []
    for row in rows:
        metadata = {
            "sample_id": row["sample_id"],
            "dataset_type": row["dataset_type"],
            "row_index": int(row["row_index"]),
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", "")),
            "choices_human": list(row.get("choices_human") or []),
            "category": str(row.get("category", "")),
            "question_id": row.get("question_id"),
            "generation_strategy": strategy,
        }
        if strategy == "augment_model":
            prior = prior_payloads.get(row["sample_id"], {})
            existing_model = list(prior.get("model_from_scratch") or [])
            if len(existing_model) < 3:
                continue
            metadata["existing_model_from_scratch"] = existing_model[:3]
        samples.append(
            Sample(
                input=str(row.get("question", "")),
                target="",
                id=row["sample_id"],
                metadata=metadata,
            )
        )
    return MemoryDataset(samples)


def _merge_generation_payload(target: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(target)
    if not merged:
        merged.update(
            {
                "sample_id": payload.get("sample_id"),
                "dataset_type": payload.get("dataset_type"),
                "row_index": payload.get("row_index"),
                "question": payload.get("question"),
                "answer": payload.get("answer"),
                "category": payload.get("category", ""),
            }
        )
    merged["status"] = "success"

    if list(payload.get("human_from_scratch") or []):
        merged["human_from_scratch"] = list(payload.get("human_from_scratch") or [])
        merged["human_from_scratch_options_randomized"] = list(payload.get("human_from_scratch_options_randomized") or [])
        merged["human_from_scratch_correct_answer_letter"] = str(payload.get("human_from_scratch_correct_answer_letter", "") or "")

    for key in FINAL5_SETTINGS:
        generated_values = list(payload.get(key) or [])
        randomized_values = list(payload.get(f"{key}_options_randomized") or [])
        correct_letter = str(payload.get(f"{key}_correct_answer_letter", "") or "")
        if generated_values and randomized_values and correct_letter:
            merged[key] = generated_values
            merged[f"{key}_options_randomized"] = randomized_values
            merged[f"{key}_correct_answer_letter"] = correct_letter
    return merged


def _generation_payloads(log_dir: Path | str) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for _path, log in iter_eval_logs(log_dir, kind="generation"):
        for sample in log.samples:
            if not sample.scores:
                continue
            score = next(iter(sample.scores.values()))
            metadata = dict(getattr(score, "metadata", {}) or {})
            if not metadata or metadata.get("status") != "success":
                continue
            sample_id = str(sample.id)
            payloads[sample_id] = _merge_generation_payload(payloads.get(sample_id, {}), metadata)
    return payloads


def _empty_generated_row() -> dict[str, Any]:
    payload: dict[str, Any] = {"schema_version": "final5_inspect_v1"}
    for setting in FINAL5_SETTINGS:
        payload[setting] = []
        payload[f"{setting}_options_randomized"] = []
        payload[f"{setting}_correct_answer_letter"] = ""
    return payload


def _empty_augmented_split(source_split: Dataset, *, dataset_type: str) -> Dataset:
    columns: dict[str, list[Any]] = {name: [] for name in source_split.column_names}
    columns.setdefault("dataset_type", [])
    columns.setdefault("row_index", [])
    columns.setdefault("sample_id", [])
    columns.setdefault("schema_version", [])
    columns.setdefault("generation_status", [])
    for key in _empty_generated_row().keys():
        columns.setdefault(key, [])
    return Dataset.from_dict(columns)


def materialize_augmented_dataset(
    processed_dataset_path: Path | str,
    generation_log_dir: Path | str,
    output_path: Path | str,
    *,
    dataset_types: list[str] | None = None,
) -> Path:
    dataset_dict = _load_dataset_dict(processed_dataset_path)
    generated = _generation_payloads(generation_log_dir)
    wanted = dataset_types or list(ACTIVE_DATASET_TYPES)

    rebuilt: dict[str, Dataset] = {}
    for dataset_type in wanted:
        if dataset_type not in dataset_dict:
            continue
        rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(dataset_dict[dataset_type]):
            payload = dict(row)
            sample_id = sample_id_for_row(dataset_type, payload, row_index)
            generated_row = generated.get(sample_id)
            if generated_row is None:
                continue
            payload["dataset_type"] = dataset_type
            payload["row_index"] = row_index
            payload["sample_id"] = sample_id
            payload["schema_version"] = "final5_inspect_v1"
            payload.update(_empty_generated_row())
            payload["generation_status"] = str(generated_row.get("status", "") or "")
            for key in FINAL5_SETTINGS:
                payload[key] = list(generated_row.get(key) or [])
                payload[f"{key}_options_randomized"] = list(generated_row.get(f"{key}_options_randomized") or [])
                payload[f"{key}_correct_answer_letter"] = str(generated_row.get(f"{key}_correct_answer_letter", "") or "")
            rows.append(payload)
        if rows:
            rebuilt[dataset_type] = Dataset.from_list(rows)
        else:
            rebuilt[dataset_type] = _empty_augmented_split(dataset_dict[dataset_type], dataset_type=dataset_type)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict(rebuilt).save_to_disk(str(out))
    return out


def ensure_augmented_dataset(
    processed_dataset_path: Path | str,
    generation_log_dir: Path | str,
    output_path: Path | str,
    *,
    dataset_types: list[str] | None = None,
    rebuild: bool = False,
) -> Path:
    out = Path(output_path)
    if out.exists() and not rebuild:
        cache_mtime = _latest_mtime(out)
        log_mtime = _latest_mtime(Path(generation_log_dir), suffix=".eval")
        if cache_mtime is not None and (log_mtime is None or log_mtime <= cache_mtime):
            return out
    return materialize_augmented_dataset(
        processed_dataset_path=processed_dataset_path,
        generation_log_dir=generation_log_dir,
        output_path=out,
        dataset_types=dataset_types,
    )


def build_evaluation_dataset(
    augmented_dataset_path: Path | str,
    *,
    setting: str,
    mode: str,
    dataset_types: list[str] | None = None,
    question_start: int = 0,
    limit: int | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
    shard_strategy: str = "contiguous",
) -> MemoryDataset:
    if setting not in FINAL5_SETTINGS:
        raise ValueError(f"Unknown setting: {setting}")
    if mode not in MODE_CHOICES:
        raise ValueError(f"Unknown mode: {mode}")

    dataset_dict = _load_dataset_dict(augmented_dataset_path)
    wanted = dataset_types or list(ACTIVE_DATASET_TYPES)
    entries: list[Sample] = []

    for dataset_type in wanted:
        if dataset_type not in dataset_dict:
            continue
        split = dataset_dict[dataset_type]
        question_end = question_start + limit if limit is not None else None
        for row_index, row in enumerate(split):
            payload = dict(row)
            original_row_index = int(payload.get("row_index", row_index))
            if original_row_index < question_start:
                continue
            if question_end is not None and original_row_index >= question_end:
                break
            sample_id = str(payload.get("sample_id") or sample_id_for_row(dataset_type, payload, original_row_index))
            options = list(payload.get(f"{setting}_options_randomized") or [])
            correct_letter = str(payload.get(f"{setting}_correct_answer_letter", "") or "")
            if not options or correct_letter not in CHOICE_LABELS[: len(options)]:
                continue

            if setting == "human_from_scratch":
                selected_human = list(payload.get("human_from_scratch") or [])[:3]
                selected_model = []
            elif setting == "model_from_scratch":
                selected_human = []
                selected_model = list(payload.get("model_from_scratch") or [])[:3]
            elif setting == "augment_human":
                selected_human = list(payload.get("human_from_scratch") or [])[:3]
                selected_model = list(payload.get("augment_human") or [])[:6]
            elif setting == "augment_model":
                selected_human = []
                selected_model = list(payload.get("augment_model") or [])[:9]
            else:
                selected_human = []
                selected_model = list(payload.get("augment_ablation") or [])[:9]

            try:
                gold_index = CHOICE_LABELS.index(correct_letter)
                human_indices = [options.index(text) for text in selected_human]
                model_indices = [options.index(text) for text in selected_model]
            except ValueError:
                continue

            entries.append(
                Sample(
                    input=str(payload.get("question", "")),
                    choices=options,
                    target=correct_letter,
                    id=sample_id,
                    metadata={
                        "sample_id": sample_id,
                        "dataset_type": dataset_type,
                        "row_index": original_row_index,
                        "question": str(payload.get("question", "")),
                        "category": str(payload.get("category", "")),
                        "setting": setting,
                        "mode": mode,
                        "gold_answer": str(payload.get("answer", "") or ""),
                        "gold_index": gold_index,
                        "selected_human_distractors": selected_human,
                        "selected_model_distractors": selected_model,
                        "human_option_indices": human_indices,
                        "model_option_indices": model_indices,
                    },
                )
            )
    entries = select_shard(entries, shard_count=shard_count, shard_index=shard_index, strategy=shard_strategy)
    return MemoryDataset(entries)


def export_generation_summary(log_dir: Path | str, output_path: Path | str) -> Path:
    payloads = _generation_payloads(log_dir)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"count": len(payloads), "sample_ids": sorted(payloads.keys())}, indent=2), encoding="utf-8")
    return out
