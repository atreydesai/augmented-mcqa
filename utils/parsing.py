from __future__ import annotations

import re
from typing import Iterable


class LabeledParseError(ValueError):
    pass


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def format_choice_lines(options: Iterable[str]) -> str:
    lines = []
    for idx, option in enumerate(options):
        lines.append(f"{chr(ord('A') + idx)}. {str(option).strip()}")
    return "\n".join(lines)


def parse_labeled_distractors(
    text: str,
    labels: list[str],
    *,
    forbidden: Iterable[str] | None = None,
) -> list[str]:
    payload = strip_code_fences(text)
    lines = [line.strip() for line in payload.splitlines() if line.strip()]
    if len(lines) != len(labels):
        raise LabeledParseError(f"Expected {len(labels)} labeled lines, got {len(lines)}")

    seen_normalized: set[str] = set()
    forbidden_normalized = {normalize_text(item).casefold() for item in forbidden or [] if normalize_text(item)}
    distractors: list[str] = []

    for expected_label, line in zip(labels, lines):
        match = re.fullmatch(r"([A-J])[\.\):]\s*(.+)", line)
        if not match:
            raise LabeledParseError(f"Line does not match expected labeled format: {line!r}")
        label = match.group(1)
        text_value = normalize_text(match.group(2))
        if label != expected_label:
            raise LabeledParseError(f"Expected label {expected_label}, got {label}")
        if not text_value:
            raise LabeledParseError(f"Empty distractor for label {label}")
        normalized = text_value.casefold()
        if normalized in seen_normalized:
            raise LabeledParseError(f"Duplicate distractor for label {label}: {text_value!r}")
        if normalized in forbidden_normalized:
            raise LabeledParseError(f"Forbidden distractor generated for label {label}: {text_value!r}")
        seen_normalized.add(normalized)
        distractors.append(text_value)

    return distractors


def extract_answer_letter(text: str, valid_letters: str) -> str:
    payload = strip_code_fences(text)
    compact = payload.strip().upper()
    if not compact:
        return ""
    valid = set(valid_letters)
    if compact in valid:
        return compact

    patterns = (
        r"ANSWER\s*[:\-]?\s*([A-J])\b",
        r"THE ANSWER IS\s*\(?([A-J])\)?",
        r"^([A-J])[\.\):]?\s*$",
        r"\b([A-J])\b",
    )
    for pattern in patterns:
        matches = re.findall(pattern, compact)
        if matches:
            for letter in reversed(matches):
                if letter in valid:
                    return letter
    return ""

