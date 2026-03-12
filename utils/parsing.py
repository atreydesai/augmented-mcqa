from __future__ import annotations

import json
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


def _extract_last_json_object(blob: str) -> dict[str, object] | None:
    decoder = json.JSONDecoder()
    for start in reversed([match.start() for match in re.finditer(r"\{", blob)]):
        candidate = blob[start:].lstrip()
        if not candidate.startswith("{"):
            continue
        try:
            parsed, _end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_json_object(text: str) -> dict[str, object]:
    payload = strip_code_fences(text)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        candidate_blobs = [payload, *[block.strip() for block in fenced_blocks if block.strip()]]
        for blob in candidate_blobs:
            parsed = _extract_last_json_object(blob)
            if parsed is not None:
                break
        else:
            raise LabeledParseError("Response does not contain a valid JSON object") from None
    if not isinstance(parsed, dict):
        raise LabeledParseError("Expected a JSON object")
    return parsed


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def format_choice_lines(options: Iterable[str]) -> str:
    lines = []
    for idx, option in enumerate(options):
        lines.append(f"{chr(ord('A') + idx)}. {str(option).strip()}")
    return "\n".join(lines)


def parse_distractors(
    text: str,
    count: int,
    *,
    forbidden: Iterable[str] | None = None,
) -> list[str]:
    forbidden_normalized = {normalize_text(item).casefold() for item in forbidden or [] if normalize_text(item)}
    payload = _extract_json_object(text)

    unexpected = sorted(set(payload.keys()) - {"distractors"})
    if "distractors" not in payload:
        raise LabeledParseError('Missing required key: "distractors"')
    if unexpected:
        raise LabeledParseError(f"Unexpected distractor keys: {', '.join(unexpected)}")

    raw_distractors = payload.get("distractors")
    if not isinstance(raw_distractors, list):
        raise LabeledParseError('Expected "distractors" to be a list')
    if len(raw_distractors) != count:
        raise LabeledParseError(f'Expected {count} distractors, got {len(raw_distractors)}')

    seen_normalized: set[str] = set()
    distractors: list[str] = []
    for idx, raw_value in enumerate(raw_distractors, start=1):
        if not isinstance(raw_value, str):
            raise LabeledParseError(f"Distractor {idx} must be a string")
        text_value = normalize_text(raw_value)
        if not text_value:
            raise LabeledParseError(f"Distractor {idx} is empty")
        normalized = text_value.casefold()
        if normalized in seen_normalized:
            raise LabeledParseError(f"Duplicate distractor at position {idx}: {text_value!r}")
        if normalized in forbidden_normalized:
            raise LabeledParseError(f"Forbidden distractor at position {idx}: {text_value!r}")
        seen_normalized.add(normalized)
        distractors.append(text_value)
    return distractors


def parse_labeled_distractors(
    text: str,
    labels: list[str],
    *,
    forbidden: Iterable[str] | None = None,
) -> list[str]:
    return parse_distractors(text, len(labels), forbidden=forbidden)


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


def extract_answer_letter_from_json(text: str, valid_letters: str) -> str:
    try:
        payload = _extract_json_object(text)
    except LabeledParseError:
        return ""

    answer = normalize_text(payload.get("answer", ""))
    if not answer:
        return ""
    candidate = answer.upper()
    if len(candidate) == 1 and candidate in set(valid_letters):
        return candidate
    return ""
