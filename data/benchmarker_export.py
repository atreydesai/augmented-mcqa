"""Export Final5 augmented datasets into benchmarker JSONL files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk

from config import CHOICE_LABELS


@dataclass(frozen=True)
class VariantSpec:
    name: str
    options_key: str | None = None
    answer_key: str | None = None


VARIANT_SPECS = (
    VariantSpec(name="original"),
    VariantSpec(
        name="human_from_scratch",
        options_key="human_from_scratch_options_randomized",
        answer_key="human_from_scratch_correct_answer_letter",
    ),
    VariantSpec(
        name="model_from_scratch",
        options_key="model_from_scratch_options_randomized",
        answer_key="model_from_scratch_correct_answer_letter",
    ),
    VariantSpec(
        name="augment_human",
        options_key="augment_human_options_randomized",
        answer_key="augment_human_correct_answer_letter",
    ),
    VariantSpec(
        name="augment_model",
        options_key="augment_model_options_randomized",
        answer_key="augment_model_correct_answer_letter",
    ),
    VariantSpec(
        name="augment_ablation",
        options_key="augment_ablation_options_randomized",
        answer_key="augment_ablation_correct_answer_letter",
    ),
)


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
        raise ExportValidationError(
            f"answer_index out of range for {choice_count} choices: {idx}"
        )
    return CHOICE_LABELS[idx]


def _answer_index_from_letter(letter: Any, choice_count: int) -> int | None:
    if not isinstance(letter, str) or not letter:
        return None

    idx = CHOICE_LABELS.find(letter.upper())
    if idx < 0 or idx >= choice_count:
        return None
    return idx


def _row_identifier(row: dict[str, Any], row_index: int) -> str:
    for key in ("id", "question_id"):
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


def _build_generated_row(
    row: dict[str, Any],
    *,
    options_key: str,
    answer_key: str,
) -> tuple[dict[str, Any] | None, str | None]:
    question = _question_text(row)
    choices = _coerce_choices(row.get(options_key))
    if choices is None:
        return None, f"missing choices in {options_key}"

    answer_letter = row.get(answer_key)
    answer_index = _answer_index_from_letter(answer_letter, len(choices))
    if answer_index is None:
        return None, f"invalid answer letter in {answer_key}"

    return {
        "question": question,
        "choices": choices,
        "answer": CHOICE_LABELS[answer_index],
    }, None


def _export_variant(
    split_name: str,
    split_rows: Any,
    spec: VariantSpec,
    output_path: Path,
) -> dict[str, Any]:
    written = 0
    skipped: list[dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(split_rows):
            if spec.name == "original":
                try:
                    exported = _build_original_row(split_name, row)
                except ExportValidationError as exc:
                    raise ExportValidationError(
                        f"{split_name}/{spec.name} row {row_index} "
                        f"({_row_identifier(row, row_index)}): {exc}"
                    ) from exc
            else:
                assert spec.options_key is not None
                assert spec.answer_key is not None
                exported, reason = _build_generated_row(
                    row,
                    options_key=spec.options_key,
                    answer_key=spec.answer_key,
                )
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


def export_benchmarker_items(input_path: Path | str, output_root: Path | str) -> Path:
    """Export a Final5 DatasetDict into benchmarker JSONL files."""
    input_path = Path(input_path)
    output_root = Path(output_root)

    dataset = load_from_disk(str(input_path))
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict at {input_path}")

    output_dir = output_root / input_path.name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "source_dataset_path": str(input_path.resolve()),
        "output_directory": str(output_dir.resolve()),
        "splits": list(dataset.keys()),
        "files": {},
    }

    for split_name, split_rows in dataset.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        split_summary: dict[str, Any] = {}

        for spec in VARIANT_SPECS:
            file_path = split_dir / f"{spec.name}.jsonl"
            split_summary[spec.name] = _export_variant(split_name, split_rows, spec, file_path)

        summary["files"][split_name] = split_summary

    summary_path = output_dir / "export_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    return summary_path


__all__ = ["export_benchmarker_items", "ExportValidationError", "VARIANT_SPECS"]
