from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from inspect_ai.dataset import MemoryDataset, Sample

from config import ACTIVE_DATASET_TYPES
from utils.constants import (
    AUGMENTED_STORE_MANIFEST,
    AUGMENTED_STORE_SCHEMA_VERSION,
    CHOICE_LABELS,
    FINAL5_SETTINGS,
    MODE_CHOICES,
)
from utils.logs import iter_eval_logs
from utils.recipes import get_setting_recipe
from utils.scheduler_state import SCHEDULABLE_GENERATION_STRATEGIES
from utils.sharding import sample_id_for_row, select_shard

AUGMENTED_RECORD_COLUMNS = (
    "id",
    "question_id",
    "dataset_type",
    "row_index",
    "sample_id",
    "question",
    "answer",
    "category",
    "options",
    "answer_index",
    "choices_human",
    "setting",
    "generation_strategy",
    "status",
    "num_human",
    "num_model",
    "num_choices",
    "human_distractors",
    "model_distractors",
    "distractors",
    "options_randomized",
    "correct_answer_letter",
    "traces",
)

DATASET_MANIFEST_SCHEMA_VERSION = "augmented_mcqa_dataset_manifest_v1"


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


def _materialized_cache_mtime(path: Path) -> float | None:
    if not path.exists():
        return None

    manifest_path = path / AUGMENTED_STORE_MANIFEST
    dataset_dict_path = path / "dataset_dict.json"
    if not manifest_path.exists() and not dataset_dict_path.exists():
        return None

    seed_path = manifest_path if manifest_path.exists() else dataset_dict_path
    latest = seed_path.stat().st_mtime
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        if "_cluster_slices" in candidate.parts:
            continue
        candidate_mtime = candidate.stat().st_mtime
        if candidate_mtime > latest:
            latest = candidate_mtime
    return latest


def _validate_augmented_output_path(
    processed_dataset_path: Path | str,
    output_path: Path | str,
) -> None:
    processed_root = Path(processed_dataset_path).resolve(strict=False)
    output_root = Path(output_path).resolve(strict=False)
    if (
        output_root == processed_root
        or processed_root in output_root.parents
        or output_root in processed_root.parents
    ):
        raise ValueError(
            f"Augmented output path must not overlap processed dataset path: {output_root}"
        )


def _load_dataset_dict(path: Path | str):
    dataset_path = Path(path)
    if dataset_path.is_file() and dataset_path.suffix == ".json":
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") == DATASET_MANIFEST_SCHEMA_VERSION:
            return _load_manifest_dataset_dict(dataset_path, payload)
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
                rebuilt[split_name] = load_from_disk(str(split_path))
            else:
                rebuilt[split_name] = Dataset.from_list([])
        return DatasetDict(rebuilt)

    dataset = load_from_disk(str(dataset_path))
    if isinstance(dataset, DatasetDict):
        return dataset
    if hasattr(dataset, "keys"):
        return dataset
    raise TypeError(f"Expected DatasetDict at {path}")


def _load_manifest_rows(spec: dict[str, Any], *, base_dir: Path) -> list[dict[str, Any]]:
    source_path = Path(str(spec.get("path", "") or ""))
    if not source_path.is_absolute():
        source_path = (base_dir / source_path).resolve(strict=False)
    source_format = str(spec.get("format", "jsonl") or "jsonl").lower()

    if source_format == "jsonl":
        rows: list[dict[str, Any]] = []
        for line in source_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(dict(json.loads(line)))
        return rows

    if source_format in {"dataset", "hf_dataset"}:
        loaded = load_from_disk(str(source_path))
        if isinstance(loaded, Dataset):
            return [dict(row) for row in loaded]
        if isinstance(loaded, DatasetDict):
            split_name = str(spec.get("split", "") or "")
            if split_name:
                return [dict(row) for row in loaded[split_name]]
            first_split = next(iter(loaded.keys()), None)
            if first_split is None:
                return []
            return [dict(row) for row in loaded[first_split]]
    raise ValueError(f"Unsupported dataset manifest format: {source_format}")


def _map_manifest_row(raw_row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    def get_value(key_name: str) -> Any:
        source_key = str(spec.get(key_name, "") or "")
        return raw_row.get(source_key) if source_key else None

    question = get_value("question_key")
    answer = get_value("answer_key")
    choices_human = get_value("choices_human_key")
    options = get_value("options_key")
    answer_index = get_value("answer_index_key")
    category = get_value("category_key")
    question_id = get_value("question_id_key")
    identifier = get_value("id_key")

    return {
        "id": identifier,
        "question_id": question_id,
        "question": "" if question is None else str(question),
        "answer": "" if answer is None else str(answer),
        "choices_human": list(choices_human or []),
        "options": list(options or []),
        "answer_index": answer_index,
        "category": "" if category is None else str(category),
    }


def _load_manifest_dataset_dict(path: Path, payload: dict[str, Any]) -> DatasetDict:
    rebuilt: dict[str, Dataset] = {}
    for dataset_type, spec in dict(payload.get("datasets", {}) or {}).items():
        rows = [_map_manifest_row(raw_row, dict(spec or {})) for raw_row in _load_manifest_rows(dict(spec or {}), base_dir=path.parent)]
        rebuilt[str(dataset_type)] = Dataset.from_list(rows)
    return DatasetDict(rebuilt)


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


def _augmented_manifest_path(root: Path) -> Path:
    return root / AUGMENTED_STORE_MANIFEST


def _normalize_augmented_root(root: Path | str) -> Path:
    root_path = Path(root)
    if root_path.is_file() and root_path.name in {AUGMENTED_STORE_MANIFEST, "dataset_dict.json"}:
        return root_path.parent
    return root_path


def _is_augmented_record_store(root: Path) -> bool:
    return _augmented_manifest_path(root).exists()


def _is_legacy_augmented_store(root: Path) -> bool:
    return (root / "dataset_dict.json").exists() and not _is_augmented_record_store(root)


def _setting_store_path(root: Path, dataset_type: str, setting: str) -> Path:
    return root / dataset_type / setting


def _load_augmented_manifest(root: Path | str) -> dict[str, Any]:
    root_path = _normalize_augmented_root(root)
    if _is_legacy_augmented_store(root_path):
        migrate_augmented_dataset_in_place(root_path)
    manifest_path = _augmented_manifest_path(root_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing augmented manifest at {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def iter_augmented_rows(
    root: Path | str,
    *,
    dataset_types: list[str] | None = None,
    settings: list[str] | None = None,
):
    root_path = _normalize_augmented_root(root)
    manifest = _load_augmented_manifest(root_path)
    wanted_datasets = dataset_types or list(manifest.get("dataset_types", []) or [])
    wanted_settings = settings or list(manifest.get("settings", []) or [])
    for dataset_type in wanted_datasets:
        for setting in wanted_settings:
            for row in _load_setting_dataset(root_path, dataset_type, setting):
                yield dict(row)


def _empty_augmented_dataset() -> Dataset:
    return Dataset.from_dict({column: [] for column in AUGMENTED_RECORD_COLUMNS})


def _write_augmented_manifest(root: Path, *, dataset_types: list[str]) -> None:
    payload = {
        "schema_version": AUGMENTED_STORE_SCHEMA_VERSION,
        "storage_kind": "setting_records",
        "dataset_types": dataset_types,
        "settings": list(FINAL5_SETTINGS),
    }
    _augmented_manifest_path(root).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_setting_dataset(root: Path | str, dataset_type: str, setting: str) -> Dataset:
    root_path = _normalize_augmented_root(root)
    if _is_legacy_augmented_store(root_path):
        migrate_augmented_dataset_in_place(root_path)
    path = _setting_store_path(root_path, dataset_type, setting)
    if not path.exists():
        return _empty_augmented_dataset()
    state_path = path / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not state.get("_data_files"):
            return _empty_augmented_dataset()
    dataset = load_from_disk(str(path))
    if isinstance(dataset, Dataset):
        return dataset
    raise TypeError(f"Expected Dataset at {path}")


def _base_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "question_id": row.get("question_id"),
        "dataset_type": str(row.get("dataset_type", "") or ""),
        "row_index": int(row.get("row_index", 0) or 0),
        "sample_id": str(row.get("sample_id", "") or ""),
        "question": str(row.get("question", "") or ""),
        "answer": str(row.get("answer", "") or ""),
        "category": str(row.get("category", "") or ""),
        "options": list(row.get("options") or []),
        "answer_index": row.get("answer_index"),
        "choices_human": list(row.get("choices_human") or []),
    }


def _record_from_setting_values(
    base: dict[str, Any],
    *,
    setting: str,
    status: str,
    human_distractors: list[str],
    model_distractors: list[str],
    options_randomized: list[str],
    correct_answer_letter: str,
    traces: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not options_randomized or correct_answer_letter not in CHOICE_LABELS[: len(options_randomized)]:
        return None
    recipe = get_setting_recipe(setting)
    distractors = [*human_distractors, *model_distractors]
    return {
        **base,
        "setting": setting,
        "generation_strategy": recipe.generation_strategy,
        "status": status,
        "num_human": len(human_distractors),
        "num_model": len(model_distractors),
        "num_choices": len(options_randomized),
        "human_distractors": list(human_distractors),
        "model_distractors": list(model_distractors),
        "distractors": distractors,
        "options_randomized": list(options_randomized),
        "correct_answer_letter": str(correct_answer_letter or ""),
        "traces": dict(traces or {}),
    }


def _empty_setting_record(
    base: dict[str, Any],
    *,
    setting: str,
    status: str,
    human_distractors: list[str] | None = None,
    model_distractors: list[str] | None = None,
    traces: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recipe = get_setting_recipe(setting)
    selected_human = list(human_distractors or [])
    selected_model = list(model_distractors or [])
    return {
        **base,
        "setting": setting,
        "generation_strategy": recipe.generation_strategy,
        "status": status,
        "num_human": len(selected_human),
        "num_model": len(selected_model),
        "num_choices": 0,
        "human_distractors": selected_human,
        "model_distractors": selected_model,
        "distractors": [*selected_human, *selected_model],
        "options_randomized": [],
        "correct_answer_letter": "",
        "traces": dict(traces or {}),
    }


def _record_from_generation_payload(
    base: dict[str, Any],
    payload: dict[str, Any],
    *,
    setting: str,
) -> dict[str, Any] | None:
    human = [str(item).strip() for item in list(payload.get("human_from_scratch") or []) if str(item).strip()]
    if setting == "human_from_scratch":
        human_distractors = human
        model_distractors = []
    elif setting == "model_from_scratch":
        human_distractors = []
        model_distractors = [str(item).strip() for item in list(payload.get("model_from_scratch") or []) if str(item).strip()]
    elif setting == "augment_human":
        human_distractors = human
        model_distractors = [str(item).strip() for item in list(payload.get("augment_human") or []) if str(item).strip()]
    elif setting == "augment_model":
        human_distractors = []
        model_distractors = [str(item).strip() for item in list(payload.get("augment_model") or []) if str(item).strip()]
    else:
        human_distractors = []
        model_distractors = [str(item).strip() for item in list(payload.get("augment_ablation") or []) if str(item).strip()]
    record = _record_from_setting_values(
        base,
        setting=setting,
        status=str(payload.get("status", "") or ""),
        human_distractors=human_distractors,
        model_distractors=model_distractors,
        options_randomized=list(payload.get(f"{setting}_options_randomized") or []),
        correct_answer_letter=str(payload.get(f"{setting}_correct_answer_letter", "") or ""),
        traces=dict((payload.get("traces") or {}).get(setting, {}) or {}),
    )
    if record is not None:
        return record
    return _empty_setting_record(
        base,
        setting=setting,
        status=str(payload.get("status", "") or ""),
        human_distractors=human_distractors,
        model_distractors=model_distractors,
        traces=dict((payload.get("traces") or {}).get(setting, {}) or {}),
    )


def _record_from_legacy_row(row: dict[str, Any], *, setting: str) -> dict[str, Any] | None:
    base = _base_record(
        {
            **row,
            "dataset_type": row.get("dataset_type"),
            "row_index": row.get("row_index", 0),
            "sample_id": row.get("sample_id", ""),
            "answer": _answer_text(row),
        }
    )
    human = [str(item).strip() for item in list(row.get("human_from_scratch") or row.get("choices_human") or []) if str(item).strip()]
    if setting == "human_from_scratch":
        human_distractors = human
        model_distractors = []
    elif setting == "model_from_scratch":
        human_distractors = []
        model_distractors = [str(item).strip() for item in list(row.get("model_from_scratch") or []) if str(item).strip()]
    elif setting == "augment_human":
        human_distractors = human
        model_distractors = [str(item).strip() for item in list(row.get("augment_human") or []) if str(item).strip()]
    elif setting == "augment_model":
        human_distractors = []
        model_distractors = [str(item).strip() for item in list(row.get("augment_model") or []) if str(item).strip()]
    else:
        human_distractors = []
        model_distractors = [str(item).strip() for item in list(row.get("augment_ablation") or []) if str(item).strip()]
    return _record_from_setting_values(
        base,
        setting=setting,
        status=str(row.get("generation_status", "success") or "success"),
        human_distractors=human_distractors,
        model_distractors=model_distractors,
        options_randomized=list(row.get(f"{setting}_options_randomized") or []),
        correct_answer_letter=str(row.get(f"{setting}_correct_answer_letter", "") or ""),
        traces={},
    )


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
    recipe = get_setting_recipe(strategy)

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
            "recipe_name": recipe.name,
            "generated_count": recipe.generated_count,
        }
        if recipe.prerequisite_setting:
            prior = prior_payloads.get(row["sample_id"], {})
            existing = list(prior.get(recipe.prerequisite_setting) or [])
            if len(existing) < get_setting_recipe(recipe.prerequisite_setting).num_model:
                continue
            metadata["existing_prerequisite_distractors"] = existing
            metadata["existing_prerequisite_setting"] = recipe.prerequisite_setting
            if recipe.prerequisite_setting == "model_from_scratch":
                metadata["existing_model_from_scratch"] = existing
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
    merged.setdefault("traces", {})
    for setting in FINAL5_SETTINGS:
        generated_values = list(payload.get(setting) or [])
        randomized_values = list(payload.get(f"{setting}_options_randomized") or [])
        correct_letter = str(payload.get(f"{setting}_correct_answer_letter", "") or "")
        if generated_values or setting == "human_from_scratch":
            merged[setting] = generated_values
        if randomized_values and correct_letter:
            merged[f"{setting}_options_randomized"] = randomized_values
            merged[f"{setting}_correct_answer_letter"] = correct_letter
        trace = dict((payload.get("traces") or {}).get(setting, {}) or {})
        if trace:
            merged["traces"][setting] = trace
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


def _write_augmented_store(
    output_path: Path,
    *,
    dataset_types: list[str],
    records: dict[str, dict[str, list[dict[str, Any]]]],
) -> Path:
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    backup_path = output_path.with_name(f".{output_path.name}.bak")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    if backup_path.exists():
        shutil.rmtree(backup_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    _write_augmented_manifest(tmp_path, dataset_types=dataset_types)
    for dataset_type in dataset_types:
        for setting in FINAL5_SETTINGS:
            setting_rows = records.get(dataset_type, {}).get(setting, [])
            dataset = Dataset.from_list(setting_rows) if setting_rows else _empty_augmented_dataset()
            path = _setting_store_path(tmp_path, dataset_type, setting)
            path.parent.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(path))
    if output_path.exists():
        output_path.rename(backup_path)
    tmp_path.rename(output_path)
    if backup_path.exists():
        shutil.rmtree(backup_path, ignore_errors=True)
    return output_path


def materialize_augmented_dataset(
    processed_dataset_path: Path | str,
    generation_log_dir: Path | str,
    output_path: Path | str,
    *,
    dataset_types: list[str] | None = None,
) -> Path:
    _validate_augmented_output_path(processed_dataset_path, output_path)
    generated = _generation_payloads(generation_log_dir)
    wanted = dataset_types or list(ACTIVE_DATASET_TYPES)
    rows = iter_processed_rows(processed_dataset_path, dataset_types=wanted)

    records: dict[str, dict[str, list[dict[str, Any]]]] = {
        dataset_type: {setting: [] for setting in FINAL5_SETTINGS} for dataset_type in wanted
    }
    for row in rows:
        sample_id = row["sample_id"]
        generated_row = generated.get(sample_id)
        base = _base_record(row)
        dataset_type = row["dataset_type"]
        for setting in FINAL5_SETTINGS:
            if generated_row is None:
                record = _empty_setting_record(base, setting=setting, status="missing")
            else:
                record = _record_from_generation_payload(base, generated_row, setting=setting)
            records[dataset_type][setting].append(record)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return _write_augmented_store(out, dataset_types=wanted, records=records)


def migrate_augmented_dataset_in_place(path: Path | str) -> Path:
    root = Path(path)
    if _is_augmented_record_store(root):
        return root
    if not _is_legacy_augmented_store(root):
        raise ValueError(f"No legacy augmented dataset found at {root}")

    dataset_dict = _load_dataset_dict(root)
    dataset_types = [dataset_type for dataset_type in ACTIVE_DATASET_TYPES if dataset_type in dataset_dict]
    records: dict[str, dict[str, list[dict[str, Any]]]] = {
        dataset_type: {setting: [] for setting in FINAL5_SETTINGS} for dataset_type in dataset_types
    }
    for dataset_type in dataset_types:
        split = dataset_dict[dataset_type]
        for row_index, row in enumerate(split):
            payload = dict(row)
            payload.setdefault("dataset_type", dataset_type)
            payload.setdefault("row_index", row_index)
            payload.setdefault("sample_id", sample_id_for_row(dataset_type, payload, row_index))
            for setting in FINAL5_SETTINGS:
                record = _record_from_legacy_row(payload, setting=setting)
                if record is not None:
                    records[dataset_type][setting].append(record)
    return _write_augmented_store(root, dataset_types=dataset_types, records=records)


def ensure_augmented_dataset(
    processed_dataset_path: Path | str,
    generation_log_dir: Path | str,
    output_path: Path | str,
    *,
    dataset_types: list[str] | None = None,
    rebuild: bool = False,
) -> Path:
    _validate_augmented_output_path(processed_dataset_path, output_path)
    out = Path(output_path)
    if out.exists() and _is_legacy_augmented_store(out):
        migrate_augmented_dataset_in_place(out)
    if out.exists() and not rebuild:
        cache_mtime = _materialized_cache_mtime(out)
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

    root = Path(augmented_dataset_path)
    root = _normalize_augmented_root(root)
    if _is_legacy_augmented_store(root):
        migrate_augmented_dataset_in_place(root)

    wanted = dataset_types or list(ACTIVE_DATASET_TYPES)
    entries: list[Sample] = []
    question_end = question_start + limit if limit is not None else None

    for dataset_type in wanted:
        split = _load_setting_dataset(root, dataset_type, setting)
        for row in split:
            payload = dict(row)
            original_row_index = int(payload.get("row_index", -1))
            if original_row_index < question_start:
                continue
            if question_end is not None and original_row_index >= question_end:
                continue
            sample_id = str(payload.get("sample_id") or "")
            options = list(payload.get("options_randomized") or [])
            correct_letter = str(payload.get("correct_answer_letter", "") or "")
            if not options or correct_letter not in CHOICE_LABELS[: len(options)]:
                continue

            selected_human = [str(item) for item in list(payload.get("human_distractors") or [])]
            selected_model = [str(item) for item in list(payload.get("model_distractors") or [])]
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
