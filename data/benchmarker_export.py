"""Export setting-scoped augmented datasets into benchmarker JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.store import _load_setting_dataset, _normalize_augmented_root
from datasets import Dataset, load_from_disk
from utils.constants import AUGMENTED_STORE_MANIFEST, CHOICE_LABELS, EVALUATED_STORE_MANIFEST, MODE_CHOICES, SETTING_NAMES


VARIANT_NAMES = ("original", *SETTING_NAMES)


class ExportValidationError(ValueError):
    """Raised when a required row cannot be exported."""


def _question_text(row: dict[str, Any]) -> str:
    question = row.get("question")
    if not isinstance(question, str) or not question:
        raise ExportValidationError("missing question")
    return question


def _coerce_choices(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not value:
        return None

    choices: list[str] = []
    for item in value:
        if item is None:
            return None
        choices.append(item if isinstance(item, str) else str(item))
    return choices


def _answer_letter_from_index(index: Any, choice_count: int) -> str:
    if index is None:
        raise ExportValidationError("missing answer_index")

    try:
        idx = int(index)
    except (TypeError, ValueError) as exc:
        raise ExportValidationError(f"invalid answer_index: {index!r}") from exc

    if idx < 0 or idx >= choice_count or idx >= len(CHOICE_LABELS):
        raise ExportValidationError(f"answer_index out of range for {choice_count} choices: {idx}")
    return CHOICE_LABELS[idx]


def _normalize_answer_letter(letter: Any, choice_count: int) -> str | None:
    if not isinstance(letter, str):
        return None

    normalized = letter.strip().upper()
    if len(normalized) != 1:
        return None
    if normalized not in CHOICE_LABELS[:choice_count]:
        return None
    return normalized


def _row_identifier(row: dict[str, Any], row_index: int) -> str:
    for key in ("sample_id", "id", "question_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return f"row_index:{row_index}"


def _skip_metadata(row: dict[str, Any], row_index: int, reason: str) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "identifier": _row_identifier(row, row_index),
        "reason": reason,
    }


def _build_original_row(split_name: str, row: dict[str, Any]) -> dict[str, Any]:
    question = _question_text(row)

    if split_name in {"arc_challenge", "mmlu_pro"}:
        choices = _coerce_choices(row.get("options"))
        if choices is None:
            raise ExportValidationError("missing original options")
        answer = _answer_letter_from_index(row.get("answer_index"), len(choices))
        return {"question": question, "choices": choices, "answer": answer}

    if split_name == "gpqa":
        answer_text = row.get("answer")
        if not isinstance(answer_text, str) or not answer_text:
            raise ExportValidationError("missing gpqa answer text")
        human_choices = _coerce_choices(row.get("choices_human"))
        if human_choices is None:
            raise ExportValidationError("missing gpqa human distractors")
        choices = [answer_text, *human_choices]
        return {"question": question, "choices": choices, "answer": "A"}

    raise ExportValidationError(f"unsupported split: {split_name}")


def _build_generated_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    question = _question_text(row)
    choices = _coerce_choices(row.get("options_randomized"))
    if choices is None:
        return None, "missing choices in options_randomized"

    answer_letter = _normalize_answer_letter(row.get("correct_answer_letter"), len(choices))
    if answer_letter is None:
        return None, "invalid answer letter in correct_answer_letter"

    return {"question": question, "choices": choices, "answer": answer_letter}, None


def _evaluated_manifest_path(root: Path) -> Path:
    return root / EVALUATED_STORE_MANIFEST


def _load_evaluated_manifest(path: Path) -> dict[str, Any]:
    manifest_path = _evaluated_manifest_path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing evaluated manifest at {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _source_kind(path: Path) -> str:
    if path.name in {AUGMENTED_STORE_MANIFEST, EVALUATED_STORE_MANIFEST}:
        raise ValueError(f"Input path must be a root directory, not a manifest file: {path}")
    if (path / AUGMENTED_STORE_MANIFEST).exists():
        return "augmented"
    if (path / EVALUATED_STORE_MANIFEST).exists():
        return "evaluated"
    try:
        _normalize_augmented_root(path)
    except ValueError as exc:
        if "unsupported augmented cache layout" in str(exc):
            raise
    raise FileNotFoundError(
        f"Input path must contain either {AUGMENTED_STORE_MANIFEST} or {EVALUATED_STORE_MANIFEST}: {path}"
    )


def _load_evaluated_setting_dataset(root: Path, split_name: str, setting: str):
    for mode in MODE_CHOICES:
        path = root / split_name / setting / mode
        if path.exists():
            dataset = load_from_disk(str(path))
            if isinstance(dataset, Dataset):
                return dataset
    return Dataset.from_list([])


def _first_available_evaluated_rows(input_path: Path, split_name: str):
    manifest = _load_evaluated_manifest(input_path)
    settings = list(manifest.get("settings") or [])
    for setting in settings:
        rows = _load_evaluated_setting_dataset(input_path, split_name, setting)
        if len(rows) > 0:
            return rows
    return _load_evaluated_setting_dataset(input_path, split_name, "human_from_scratch")


def _generator_identity(manifest: dict[str, Any]) -> tuple[str, str]:
    return (
        str(manifest.get("generation_run_name", "") or ""),
        str(manifest.get("generation_model", "") or ""),
    )


def _evaluated_group_roots(collected_root: Path) -> list[Path]:
    return sorted(path.parent for path in collected_root.rglob(EVALUATED_STORE_MANIFEST))


def _sample_id(row: dict[str, Any], row_index: int) -> str:
    value = row.get("sample_id")
    if value not in (None, ""):
        return str(value)
    return _row_identifier(row, row_index)


def _valid_generated_row(row: dict[str, Any]) -> bool:
    exported, _reason = _build_generated_row(row)
    return exported is not None


def _collected_shared_support(input_path: Path) -> dict[str, set[str]]:
    collected_root = input_path.parent.parent.parent
    group_roots = _evaluated_group_roots(collected_root)
    source_manifest = _load_evaluated_manifest(input_path)
    required_settings = list(source_manifest.get("settings") or [])
    generator_keys: set[tuple[str, str]] = set()
    observed: dict[tuple[str, str, str], set[tuple[str, str]]] = {}

    for group_root in group_roots:
        manifest = _load_evaluated_manifest(group_root)
        generator_key = _generator_identity(manifest)
        generator_keys.add(generator_key)
        dataset_types = list(manifest.get("dataset_types") or [])
        settings = list(manifest.get("settings") or [])
        for split_name in dataset_types:
            for setting in settings:
                rows = _load_evaluated_setting_dataset(group_root, split_name, setting)
                for row_index, raw_row in enumerate(rows):
                    row = dict(raw_row)
                    if not _valid_generated_row(row):
                        continue
                    key = (split_name, setting, _sample_id(row, row_index))
                    observed.setdefault(key, set()).add(generator_key)

    eligible_by_setting: dict[tuple[str, str], set[str]] = {}
    required_generators = set(generator_keys)
    for split_name, setting, sample_id in observed:
        present = observed[(split_name, setting, sample_id)]
        if present == required_generators:
            eligible_by_setting.setdefault((split_name, setting), set()).add(sample_id)

    eligible_by_split: dict[str, set[str]] = {}
    for split_name in list(source_manifest.get("dataset_types") or []):
        setting_sets = [set(eligible_by_setting.get((split_name, setting), set())) for setting in required_settings]
        if not setting_sets:
            eligible_by_split[split_name] = set()
            continue
        shared = set(setting_sets[0])
        for sample_ids in setting_sets[1:]:
            shared.intersection_update(sample_ids)
        eligible_by_split[split_name] = shared
    return eligible_by_split


def _source_output_dir(input_path: Path, *, source_kind: str) -> Path:
    if source_kind == "augmented":
        return Path(input_path.name)
    manifest = _load_evaluated_manifest(input_path)
    run_name = str(manifest.get("generation_run_name", "") or "")
    generator = str(manifest.get("generation_model", "") or "")
    safe_generator = generator.replace("/", "_").replace("\\", "_").replace(" ", "_")
    safe_run = run_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return Path(f"{safe_run}__{safe_generator}")


def _source_split_rows(input_path: Path, split_name: str, variant_name: str, *, source_kind: str):
    if source_kind == "augmented":
        if variant_name == "original":
            return _first_available_split_rows(input_path, split_name)
        return _load_setting_dataset(input_path, split_name, variant_name)
    if variant_name == "original":
        return _first_available_evaluated_rows(input_path, split_name)
    return _load_evaluated_setting_dataset(input_path, split_name, variant_name)
 
    raise ValueError(f"Unsupported export source kind: {source_kind}")


def _export_variant(
    split_name: str,
    split_rows: Any,
    variant_name: str,
    output_path: Path,
    *,
    eligible_sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    written = 0
    skipped: list[dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as handle:
        for row_index, raw_row in enumerate(split_rows):
            row = dict(raw_row)
            identifier = _sample_id(row, row_index)
            if eligible_sample_ids is not None and identifier not in eligible_sample_ids:
                skipped.append(_skip_metadata(row, row_index, "not in shared support"))
                continue
            if variant_name == "original":
                try:
                    exported = _build_original_row(split_name, row)
                except ExportValidationError as exc:
                    raise ExportValidationError(
                        f"{split_name}/{variant_name} row {row_index} "
                        f"({_row_identifier(row, row_index)}): {exc}"
                    ) from exc
            else:
                exported, reason = _build_generated_row(row)
                if exported is None:
                    skipped.append(_skip_metadata(row, row_index, reason or "invalid row"))
                    continue

            handle.write(json.dumps(exported, ensure_ascii=False))
            handle.write("\n")
            written += 1

    return {
        "output_path": str(output_path.resolve()),
        "rows_written": written,
        "skipped_row_count": len(skipped),
        "skipped_rows": skipped,
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    path = _normalize_augmented_root(path)
    manifest_path = path / AUGMENTED_STORE_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing augmented manifest at {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _first_available_split_rows(input_path: Path, split_name: str):
    for setting in SETTING_NAMES:
        rows = _load_setting_dataset(input_path, split_name, setting)
        if len(rows) > 0:
            return rows
    return _load_setting_dataset(input_path, split_name, "human_from_scratch")


def export_benchmarker_items(input_path: Path | str, output_root: Path | str) -> Path:
    """Export a setting-scoped augmented dataset into benchmarker JSONL files."""
    input_path = Path(input_path)
    output_root = Path(output_root)
    source_kind = _source_kind(input_path)
    manifest = _load_manifest(input_path) if source_kind == "augmented" else _load_evaluated_manifest(input_path)
    output_dir = output_root / _source_output_dir(input_path, source_kind=source_kind)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "source_dataset_path": str(input_path.resolve()),
        "source_kind": source_kind,
        "output_directory": str(output_dir.resolve()),
        "files": {},
    }

    for split_name in manifest.get("dataset_types", []):
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        summary["files"][split_name] = {}
        original_rows = _source_split_rows(input_path, split_name, "original", source_kind=source_kind)
        for variant_name in VARIANT_NAMES:
            rows = (
                original_rows
                if variant_name == "original"
                else _source_split_rows(input_path, split_name, variant_name, source_kind=source_kind)
            )
            summary["files"][split_name][variant_name] = _export_variant(
                split_name,
                rows,
                variant_name,
                split_dir / f"{variant_name}.jsonl",
            )

    summary_path = output_dir / "export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path
