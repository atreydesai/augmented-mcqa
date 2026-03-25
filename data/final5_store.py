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
    DEFAULT_AUGMENTED_CACHE_ROOT,
    EVALUATED_STORE_MANIFEST,
    EVALUATED_STORE_SCHEMA_VERSION,
    FINAL5_SETTINGS,
    MODE_CHOICES,
)
from utils.logs import iter_eval_logs, iter_log_payloads
from utils.modeling import safe_name
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
EVALUATED_RECORD_COLUMNS = AUGMENTED_RECORD_COLUMNS + (
    "evaluation_status",
    "evaluation_is_correct",
    "evaluation_score",
    "evaluation_prediction",
    "evaluation_prediction_type",
    "evaluation_raw_output",
    "evaluation_prompt",
    "evaluation_question_idx",
    "evaluation_log_path",
)


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
    status = str(payload.get("status", "") or "")
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
    if status and status != "success":
        return _empty_setting_record(
            base,
            setting=setting,
            status=status,
            human_distractors=human_distractors,
            model_distractors=model_distractors,
            traces=dict((payload.get("traces") or {}).get(setting, {}) or {}),
        )
    record = _record_from_setting_values(
        base,
        setting=setting,
        status=status,
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
    augmented_dataset_path: Path | str | None = None,
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
    cached_prerequisites: dict[str, list[str]] = {}
    if recipe.prerequisite_setting and augmented_dataset_path:
        try:
            wanted_dataset_types = list(dataset_types or [])
            if not wanted_dataset_types:
                manifest = _load_augmented_manifest(augmented_dataset_path)
                wanted_dataset_types = list(manifest.get("dataset_types") or [])
            for dataset_type in wanted_dataset_types:
                for row in _load_setting_dataset(augmented_dataset_path, dataset_type, recipe.prerequisite_setting):
                    payload = dict(row)
                    sample_id = str(payload.get("sample_id", "") or "")
                    if not sample_id:
                        continue
                    distractors = [
                        str(item).strip()
                        for item in list(payload.get("model_distractors") or [])
                        if str(item).strip()
                    ]
                    if distractors:
                        cached_prerequisites[sample_id] = distractors
        except FileNotFoundError:
            cached_prerequisites = {}

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
            if not existing:
                existing = list(cached_prerequisites.get(row["sample_id"], []) or [])
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
    payload_status = str(payload.get("status", "") or "")
    if payload_status:
        merged["status"] = payload_status
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
            if not metadata:
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
    include_missing_rows: bool = True,
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
        if generated_row is None and not include_missing_rows:
            continue
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
        include_missing_rows=False,
    )


def _evaluated_manifest_path(root: Path) -> Path:
    return root / EVALUATED_STORE_MANIFEST


def _evaluated_store_path(root: Path, dataset_type: str, setting: str, mode: str) -> Path:
    return root / dataset_type / setting / mode


def _empty_evaluated_dataset() -> Dataset:
    return Dataset.from_dict({column: [] for column in EVALUATED_RECORD_COLUMNS})


def _write_evaluated_manifest(
    root: Path,
    *,
    dataset_types: list[str],
    settings: list[str],
    modes: list[str],
    generation_run_name: str,
    generation_model: str,
    evaluation_model: str,
    source_results_root: str,
) -> None:
    payload = {
        "schema_version": EVALUATED_STORE_SCHEMA_VERSION,
        "storage_kind": "evaluated_setting_mode_records",
        "dataset_types": dataset_types,
        "settings": settings,
        "modes": modes,
        "generation_run_name": generation_run_name,
        "generation_model": generation_model,
        "evaluation_model": evaluation_model,
        "source_results_root": source_results_root,
    }
    _evaluated_manifest_path(root).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_evaluated_store(
    output_path: Path,
    *,
    dataset_types: list[str],
    settings: list[str],
    modes: list[str],
    records: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
    generation_run_name: str,
    generation_model: str,
    evaluation_model: str,
    source_results_root: str,
) -> Path:
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    backup_path = output_path.with_name(f".{output_path.name}.bak")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    if backup_path.exists():
        shutil.rmtree(backup_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    _write_evaluated_manifest(
        tmp_path,
        dataset_types=dataset_types,
        settings=settings,
        modes=modes,
        generation_run_name=generation_run_name,
        generation_model=generation_model,
        evaluation_model=evaluation_model,
        source_results_root=source_results_root,
    )
    for dataset_type in dataset_types:
        for setting in settings:
            for mode in modes:
                mode_rows = records.get(dataset_type, {}).get(setting, {}).get(mode, [])
                dataset = Dataset.from_list(mode_rows) if mode_rows else _empty_evaluated_dataset()
                path = _evaluated_store_path(tmp_path, dataset_type, setting, mode)
                path.parent.mkdir(parents=True, exist_ok=True)
                dataset.save_to_disk(str(path))
    if output_path.exists():
        output_path.rename(backup_path)
    tmp_path.rename(output_path)
    if backup_path.exists():
        shutil.rmtree(backup_path, ignore_errors=True)
    return output_path


def _evaluation_score_value(score: Any) -> float | None:
    value = score.get("value") if isinstance(score, dict) else getattr(score, "value", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _evaluated_root_mtime(path: Path) -> float | None:
    if not path.exists():
        return None
    latest = _latest_mtime(path)
    if latest is None:
        return None
    return latest


def _existing_evaluated_outputs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob(EVALUATED_STORE_MANIFEST))


def _augmented_record_lookup(
    root: Path,
    dataset_type: str,
    setting: str,
    cache: dict[tuple[str, str, str], dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    key = (str(root), str(dataset_type), str(setting))
    if key not in cache:
        rows: dict[str, dict[str, Any]] = {}
        for row in _load_setting_dataset(root, dataset_type, setting):
            payload = dict(row)
            sample_id = str(payload.get("sample_id", "") or "")
            if sample_id:
                rows[sample_id] = payload
        cache[key] = rows
    return cache[key]


def _fallback_augmented_record(
    *,
    sample: Any,
    sample_meta: dict[str, Any],
    score_meta: dict[str, Any],
    setting: str,
) -> dict[str, Any]:
    sample_id = str(sample_meta.get("sample_id") or getattr(sample, "id", "") or "")
    raw_question_idx = score_meta.get("question_idx", sample_meta.get("row_index", -1))
    question_idx = -1 if raw_question_idx is None else int(raw_question_idx)
    answer_index = score_meta.get("gold_index", sample_meta.get("gold_index"))
    choices = list(getattr(sample, "choices", []) or [])
    if answer_index is not None:
        try:
            answer_index_int = int(answer_index)
        except (TypeError, ValueError):
            answer_index_int = None
    else:
        answer_index_int = None
    answer_text = str(sample_meta.get("gold_answer", "") or "")
    if not answer_text and answer_index_int is not None and 0 <= answer_index_int < len(choices):
        answer_text = str(choices[answer_index_int])
    human = [
        str(item)
        for item in list(
            score_meta.get("selected_human_distractors")
            or sample_meta.get("selected_human_distractors")
            or []
        )
    ]
    model = [
        str(item)
        for item in list(
            score_meta.get("selected_model_distractors")
            or sample_meta.get("selected_model_distractors")
            or []
        )
    ]
    base = _base_record(
        {
            "id": sample_id,
            "question_id": sample_id,
            "dataset_type": str(score_meta.get("dataset_type") or sample_meta.get("dataset_type") or ""),
            "row_index": question_idx,
            "sample_id": sample_id,
            "question": str(sample_meta.get("question") or getattr(sample, "input", "") or ""),
            "answer": answer_text,
            "category": str(score_meta.get("category") or sample_meta.get("category") or ""),
            "options": choices,
            "answer_index": answer_index_int,
            "choices_human": list(sample_meta.get("choices_human") or human),
        }
    )
    record = _record_from_setting_values(
        base,
        setting=setting,
        status="evaluated",
        human_distractors=human,
        model_distractors=model,
        options_randomized=choices,
        correct_answer_letter=str(score_meta.get("gold_answer_letter") or getattr(sample, "target", "") or ""),
        traces={},
    )
    if record is not None:
        return record
    return _empty_setting_record(base, setting=setting, status="evaluated", human_distractors=human, model_distractors=model)


def _empty_evaluation_payload(*, row_index: int) -> dict[str, Any]:
    return {
        "evaluation_status": "missing",
        "evaluation_is_correct": None,
        "evaluation_score": None,
        "evaluation_prediction": "",
        "evaluation_prediction_type": "",
        "evaluation_raw_output": "",
        "evaluation_prompt": "",
        "evaluation_question_idx": int(row_index),
        "evaluation_log_path": "",
    }


def materialize_evaluated_datasets(
    evaluation_log_root: Path | str,
    output_root: Path | str = DEFAULT_AUGMENTED_CACHE_ROOT.parent / "evaluated",
    *,
    augmented_root: Path | str = DEFAULT_AUGMENTED_CACHE_ROOT,
) -> list[Path]:
    output_root_path = Path(output_root)
    eval_root = Path(evaluation_log_root)
    augmented_root_path = Path(augmented_root)

    cache_mtime = _evaluated_root_mtime(output_root_path)
    eval_mtime = _latest_mtime(eval_root, suffix=".eval")
    augmented_mtime = _latest_mtime(augmented_root_path)
    if (
        cache_mtime is not None
        and (eval_mtime is None or eval_mtime <= cache_mtime)
        and (augmented_mtime is None or augmented_mtime <= cache_mtime)
    ):
        return _existing_evaluated_outputs(output_root_path)

    observed_by_group: dict[tuple[str, str, str], dict[tuple[str, str, str, str], dict[str, Any]]] = {}
    group_meta: dict[tuple[str, str, str], dict[str, Any]] = {}
    augmented_cache: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    source_root = str(eval_root)

    for log_path, log in iter_log_payloads(eval_root, kind="evaluation"):
        log_meta = dict(log.get("metadata", {}) or {})
        generation_run_name = str(log_meta.get("generation_run_name", "") or "")
        generation_model = str(log_meta.get("generation_model", "") or "")
        evaluation_model = str(log_meta.get("evaluation_model", "") or "")
        setting = str(log_meta.get("setting", "") or "")
        mode = str(log_meta.get("mode", "") or "")
        group_key = (generation_run_name, generation_model, evaluation_model)
        augmented_path = (
            augmented_root_path
            / safe_name(generation_run_name or "unknown_generation_run")
            / safe_name(generation_model or "unknown_generation_model")
        )
        group_meta.setdefault(
            group_key,
            {
                "generation_run_name": generation_run_name,
                "generation_model": generation_model,
                "evaluation_model": evaluation_model,
                "dataset_types": set(),
                "settings": set(),
                "modes": set(),
                "augmented_path": augmented_path,
            },
        )
        group_meta[group_key]["settings"].add(setting)
        group_meta[group_key]["modes"].add(mode)
        group_observed = observed_by_group.setdefault(group_key, {})

        for sample in list(log.get("summaries", []) or []):
            scores = dict(sample.get("scores", {}) or {})
            if not scores:
                continue
            score = next(iter(scores.values()))
            score_meta = dict(score.get("metadata", {}) or {})
            sample_meta = dict(sample.get("metadata", {}) or {})
            dataset_type = str(score_meta.get("dataset_type") or sample_meta.get("dataset_type") or "")
            if not dataset_type:
                continue
            sample_id = str(score_meta.get("sample_id") or sample_meta.get("sample_id") or sample.get("id", "") or "")
            group_meta[group_key]["dataset_types"].add(dataset_type)
            score_value = _evaluation_score_value(score)
            evaluation_meta = sample_meta.get("evaluation", {}) or {}
            if not isinstance(evaluation_meta, dict):
                evaluation_meta = {}
            group_observed[(dataset_type, setting, mode, sample_id)] = {
                "evaluation_is_correct": bool(score_value) if score_value is not None else False,
                "evaluation_score": score_value,
                "evaluation_prediction": str(score_meta.get("prediction") or evaluation_meta.get("prediction") or ""),
                "evaluation_prediction_type": str(score_meta.get("prediction_type", "") or ""),
                "evaluation_raw_output": str(score_meta.get("raw_output") or evaluation_meta.get("raw_output") or ""),
                "evaluation_prompt": str(score_meta.get("prompt") or evaluation_meta.get("prompt") or ""),
                "evaluation_status": str(score_meta.get("status") or log.get("status", "") or "success"),
                "evaluation_question_idx": (
                    -1
                    if score_meta.get("question_idx", sample_meta.get("row_index", -1)) is None
                    else int(score_meta.get("question_idx", sample_meta.get("row_index", -1)))
                ),
                "evaluation_log_path": str(log_path),
            }

    outputs: list[Path] = []
    for group_key, meta in group_meta.items():
        generation_run_name, generation_model, evaluation_model = group_key
        augmented_path = Path(meta["augmented_path"])
        dataset_types = sorted(meta["dataset_types"])
        if augmented_path.exists():
            manifest = _load_augmented_manifest(augmented_path)
            manifest_datasets = list(manifest.get("dataset_types") or [])
            if manifest_datasets:
                dataset_types = [dataset for dataset in manifest_datasets if dataset in set(dataset_types)]
        settings = sorted(meta["settings"], key=lambda value: list(FINAL5_SETTINGS).index(value) if value in FINAL5_SETTINGS else len(FINAL5_SETTINGS))
        modes = sorted(meta["modes"], key=lambda value: list(MODE_CHOICES).index(value) if value in MODE_CHOICES else len(MODE_CHOICES))
        group_records: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {
            dataset_type: {setting: {mode: [] for mode in modes} for setting in settings}
            for dataset_type in dataset_types
        }
        group_observed = observed_by_group.get(group_key, {})

        for dataset_type in dataset_types:
            for setting in settings:
                augmented_rows = _augmented_record_lookup(augmented_path, dataset_type, setting, augmented_cache)
                seen_sample_ids = set()
                for sample_id, augmented_row in augmented_rows.items():
                    seen_sample_ids.add(sample_id)
                    base_record = {column: augmented_row.get(column) for column in AUGMENTED_RECORD_COLUMNS}
                    raw_row_index = augmented_row.get("row_index", -1)
                    row_index = -1 if raw_row_index is None else int(raw_row_index)
                    for mode in modes:
                        evaluation_payload = group_observed.get((dataset_type, setting, mode, sample_id))
                        group_records[dataset_type][setting][mode].append(
                            {
                                **base_record,
                                **(
                                    dict(evaluation_payload)
                                    if evaluation_payload is not None
                                    else _empty_evaluation_payload(row_index=row_index)
                                ),
                            }
                        )

                for mode in modes:
                    observed_only = [
                        (sample_id, payload)
                        for (payload_dataset, payload_setting, payload_mode, sample_id), payload in group_observed.items()
                        if payload_dataset == dataset_type
                        and payload_setting == setting
                        and payload_mode == mode
                        and sample_id not in seen_sample_ids
                    ]
                    for sample_id, payload in observed_only:
                        fallback = _fallback_augmented_record(
                            sample=type("SampleProxy", (), {"id": sample_id, "input": "", "choices": []})(),
                            sample_meta={"sample_id": sample_id, "dataset_type": dataset_type},
                            score_meta={"dataset_type": dataset_type, "question_idx": payload["evaluation_question_idx"]},
                            setting=setting,
                        )
                        group_records[dataset_type][setting][mode].append({**fallback, **payload})

        output_path = (
            output_root_path
            / safe_name(generation_run_name or "unknown_generation_run")
            / safe_name(generation_model or "unknown_generation_model")
            / safe_name(evaluation_model or "unknown_evaluation_model")
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        outputs.append(
            _write_evaluated_store(
                output_path,
                dataset_types=dataset_types,
                settings=settings,
                modes=modes,
                records=group_records,
                generation_run_name=generation_run_name,
                generation_model=generation_model,
                evaluation_model=evaluation_model,
                source_results_root=source_root,
            )
        )

    return outputs


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
