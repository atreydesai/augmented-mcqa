"""Export setting-scoped augmented datasets into benchmarker JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.store import _load_setting_dataset, _normalize_augmented_root
from utils.constants import AUGMENTED_STORE_MANIFEST, CHOICE_LABELS, SETTING_NAMES


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


def _export_variant(
    split_name: str,
    split_rows: Any,
    variant_name: str,
    output_path: Path,
) -> dict[str, Any]:
    written = 0
    skipped: list[dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as handle:
        for row_index, raw_row in enumerate(split_rows):
            row = dict(raw_row)
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
    input_path = _normalize_augmented_root(Path(input_path))
    output_root = Path(output_root)

    manifest = _load_manifest(input_path)
    output_dir = output_root / input_path.name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "source_dataset_path": str(input_path.resolve()),
        "output_directory": str(output_dir.resolve()),
        "files": {},
    }

    for split_name in manifest.get("dataset_types", []):
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        summary["files"][split_name] = {}
        original_rows = _first_available_split_rows(input_path, split_name)
        for variant_name in VARIANT_NAMES:
            rows = original_rows if variant_name == "original" else _load_setting_dataset(input_path, split_name, variant_name)
            summary["files"][split_name][variant_name] = _export_variant(
                split_name,
                rows,
                variant_name,
                split_dir / f"{variant_name}.jsonl",
            )

    summary_path = output_dir / "export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path
