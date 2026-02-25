"""Final5 distractor generation utilities.

Generation steps per question:
A) human_from_scratch: Q+A -> 3H (passthrough from choices_human)
B) model_from_scratch: Q+A -> 3M
C) augment_human: (Q+A+3H) -> 6M
D) augment_model: (Q+A+3M from B) -> 6M delta, then combine to 9M
E) augment_ablation: Q+A -> 9M
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm

from config import (
    RESULTS_DIR,
    DISTRACTOR_GENERATION_PROMPT_QA_TEMPLATE,
    DISTRACTOR_GENERATION_PROMPT_CONDITIONED_TEMPLATE,
)
from data.hub_utils import push_dataset_to_hub
from models import get_client


CHOICE_LABELS = "ABCDEFGHIJ"
FINAL5_SCHEMA_VERSION = "final5_v1"


class AugmentorMode(Enum):
    FINAL5 = "final5"
    # Legacy compatibility aliases.
    FROM_SCRATCH = "from_scratch"
    CONDITIONED_HUMAN = "conditioned_human"
    CONDITIONED_SYNTHETIC = "conditioned_synthetic"


class NonRetryableAugmentationError(RuntimeError):
    """Raised for deterministic schema/data issues."""


class StructuredParseError(ValueError):
    """Raised when structured distractor parsing fails."""


@dataclass
class GenerationConfig:
    mode: AugmentorMode = AugmentorMode.FINAL5
    model_provider: str = "openai"
    model_name: str = "gpt-5.2-2025-12-11"
    max_tokens: int = 512
    max_retries: int = 3
    retry_delay: float = 1.0
    save_interval: int = 25
    skip_failed_entries: bool = False
    force_overwrite: bool = False

    # Explicit reasoning control policy.
    reasoning_effort: Optional[str] = None
    anthropic_thinking: Optional[Dict[str, Any]] = None

    # Additional provider kwargs applied to every generation call.
    generate_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Optional request-level visibility controls.
    request_log_path: Optional[Path] = None
    slow_call_seconds: float = 45.0


STRATEGY_IDS = [
    "human_from_scratch",
    "model_from_scratch",
    "augment_human",
    "augment_model",
    "augment_ablation",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_request_log(path: Optional[Path], payload: Dict[str, Any]) -> None:
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _log_event(config: GenerationConfig, event: str, **fields: Any) -> None:
    payload = {
        "ts_utc": _utc_now_iso(),
        "event": event,
        "model": config.model_name,
        "provider": config.model_provider,
        **fields,
    }
    _append_request_log(config.request_log_path, payload)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _require_column_list(entry: Dict[str, Any], column_name: str) -> List[str]:
    values = entry.get(column_name)
    if values is None:
        raise NonRetryableAugmentationError(
            f"Missing required column '{column_name}' in dataset entry"
        )
    if not isinstance(values, list):
        raise NonRetryableAugmentationError(
            f"Column '{column_name}' must be a list, got {type(values).__name__}"
        )
    return [str(v).strip() for v in values if str(v).strip()]


def _expected_letters(start_letter: str, expected_count: int) -> List[str]:
    start = start_letter.upper()
    return [chr(ord(start) + i) for i in range(expected_count)]


def parse_generated_distractors(
    response: str,
    expected_count: int,
    start_letter: str,
) -> List[str]:
    if expected_count <= 0:
        raise ValueError(f"expected_count must be > 0, got {expected_count}")

    del start_letter

    response = response.strip()
    if not response:
        raise ValueError("Model output is empty")

    json_values = _extract_json_distractors(response, expected_count=expected_count)
    if json_values is not None:
        return json_values

    raise ValueError(
        f"Could not decode structured distractors JSON with {expected_count} entries. "
        f"Raw preview: '{_preview_text(response)}'"
    )


def _json_candidates_from_text(text: str) -> List[str]:
    candidates = [text.strip()]
    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())
    return candidates


def _normalize_distractor_text(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    text = re.sub(r"^\s*[A-Z]\s*(?:[:\.\)\-])\s*", "", text)
    return text.strip()


def _extract_json_distractors(text: str, expected_count: int) -> Optional[List[str]]:
    for candidate in _json_candidates_from_text(text):
        try:
            parsed = json.loads(candidate)
        except Exception:  # noqa: BLE001
            recovered = _extract_truncated_json_distractors(
                candidate,
                expected_count=expected_count,
            )
            if recovered is not None:
                return recovered
            continue

        distractors: Any = parsed.get("distractors") if isinstance(parsed, dict) else None

        if not isinstance(distractors, list):
            continue

        values = [_normalize_distractor_text(x) for x in distractors]
        values = [x for x in values if x]
        if len(values) >= expected_count:
            return values[:expected_count]
    return None


def _extract_truncated_json_distractors(
    candidate: str,
    expected_count: int,
) -> Optional[List[str]]:
    # Structured-output specific fallback: provider sometimes truncates tail
    # after producing valid quoted distractor strings.
    m = re.search(r'"distractors"\s*:\s*\[(.*)', candidate, flags=re.DOTALL)
    if not m:
        return None

    body = m.group(1)
    quoted = re.findall(r'"((?:\\.|[^"\\])*)"', body)
    if not quoted:
        return None

    values: List[str] = []
    for item in quoted:
        try:
            decoded = json.loads(f'"{item}"')
        except Exception:  # noqa: BLE001
            decoded = item
        normalized = _normalize_distractor_text(decoded)
        if normalized:
            values.append(normalized)

    if len(values) >= expected_count:
        return values[:expected_count]
    return None


def _build_q_a_prompt(question: str, answer: str, count: int, start_letter: str) -> str:
    letters = ", ".join(_expected_letters(start_letter, count))
    return DISTRACTOR_GENERATION_PROMPT_QA_TEMPLATE.format(
        question=question,
        gold_answer=answer,
        count=count,
        target_letters=letters,
    )


def _build_conditioned_prompt(
    question: str,
    answer: str,
    context_distractors: List[str],
    count: int,
) -> str:
    lines = [f"A: {answer}"]
    for i, d in enumerate(context_distractors, start=1):
        lines.append(f"{chr(ord('A') + i)}: {d}")

    start_letter = chr(ord("A") + len(context_distractors) + 1)
    num_existing_options = len(lines)
    total_options = num_existing_options + count
    total_distractors = total_options - 1
    letters = ", ".join(_expected_letters(start_letter, count))
    existing_options_block = "\n".join(lines)

    return DISTRACTOR_GENERATION_PROMPT_CONDITIONED_TEMPLATE.format(
        question=question,
        gold_answer=answer,
        existing_options_block=existing_options_block,
        num_existing_options=num_existing_options,
        total_options=total_options,
        total_distractors=total_distractors,
        count=count,
        target_letters=letters,
    )


def _build_structured_schema(expected_count: int, provider: str) -> Dict[str, Any]:
    provider_key = provider.lower().strip()
    array_schema: Dict[str, Any] = {
        "type": "array",
        "items": {"type": "string"},
    }
    if provider_key == "anthropic":
        # Anthropic JSON schema currently rejects minItems > 1 for arrays.
        # Keep schema permissive and enforce exact count in parser.
        array_schema["minItems"] = 1
    else:
        array_schema["minItems"] = expected_count
        array_schema["maxItems"] = expected_count

    return {
        "type": "object",
        "properties": {
            "distractors": array_schema
        },
        "required": ["distractors"],
        "additionalProperties": False,
    }


def _preview_text(value: str, max_chars: int = 300) -> str:
    compact = re.sub(r"\s+", " ", value or "").strip()
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3]}..."


def _structured_request_kwargs(provider: str, expected_count: int) -> Dict[str, Any]:
    provider_key = provider.lower().strip()
    schema = _build_structured_schema(expected_count, provider=provider_key)
    schema_name = f"distractors_{expected_count}"

    if provider_key in {"openai", "gemini"}:
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            }
        }
    if provider_key == "anthropic":
        return {
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": schema,
                }
            }
        }
    return {}


def _shuffle_options(answer: str, distractors: List[str], seed: int) -> tuple[List[str], str]:
    options = [answer] + distractors
    idx = list(range(len(options)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    shuffled = [options[i] for i in idx]
    gold_idx = idx.index(0)
    return shuffled, CHOICE_LABELS[gold_idx]


def _ensure_trace_columns(entry: Dict[str, Any]) -> None:
    for strategy in STRATEGY_IDS:
        entry.setdefault(strategy, [])
        entry.setdefault(f"{strategy}_full_question", "")
        entry.setdefault(f"{strategy}_model_input", "")
        entry.setdefault(f"{strategy}_model_output", "")
        entry.setdefault(f"{strategy}_options_randomized", [])
        entry.setdefault(f"{strategy}_correct_answer_letter", "")

    entry.setdefault("augment_model_delta_6m", [])
    entry.setdefault("schema_version", FINAL5_SCHEMA_VERSION)


def _client_kwargs_for_config(config: GenerationConfig) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    provider = config.model_provider.lower().strip()

    if provider in {"openai", "gemini"} and config.reasoning_effort:
        kwargs["reasoning_effort"] = config.reasoning_effort
    return kwargs


def _call_generate_with_observability(
    *,
    client,
    prompt: str,
    max_tokens: int,
    generate_kwargs: Dict[str, Any],
    config: GenerationConfig,
    context: str,
    entry_index: int,
    attempt: int,
    expected_count: Optional[int] = None,
    structured_output_type: Optional[str] = None,
):
    started = time.time()
    _log_event(
        config,
        "request_start",
        context=context,
        entry_index=entry_index,
        attempt=attempt,
        prompt_chars=len(prompt),
        max_tokens=max_tokens,
        expected_count=expected_count,
        structured_output_type=structured_output_type,
    )

    try:
        response = client.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            **generate_kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = round(time.time() - started, 3)
        _log_event(
            config,
            "request_error",
            context=context,
            entry_index=entry_index,
            attempt=attempt,
            elapsed_s=elapsed,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise

    elapsed = round(time.time() - started, 3)
    output_chars = len(_safe_text(getattr(response, "text", "")))
    if elapsed >= config.slow_call_seconds:
        msg = (
            f"[slow-call] model={config.model_name} context={context} "
            f"entry={entry_index} attempt={attempt} elapsed={elapsed:.1f}s"
        )
        print(msg)
        _log_event(
            config,
            "slow_call",
            context=context,
            entry_index=entry_index,
            attempt=attempt,
            elapsed_s=elapsed,
        )

    _log_event(
        config,
        "request_success",
        context=context,
        entry_index=entry_index,
        attempt=attempt,
        elapsed_s=elapsed,
        output_chars=output_chars,
    )
    return response


def _generate_with_retries(
    *,
    client,
    prompt: str,
    expected_count: int,
    start_letter: str,
    config: GenerationConfig,
    generate_kwargs: Dict[str, Any],
    context: str,
    entry_index: int,
) -> tuple[List[str], str]:
    last_error: Optional[Exception] = None
    parse_retry_used = False
    merged_kwargs = dict(generate_kwargs)
    merged_kwargs.update(_structured_request_kwargs(config.model_provider, expected_count))
    structured_output_type = (
        "response_format"
        if "response_format" in merged_kwargs
        else "output_config"
        if "output_config" in merged_kwargs
        else None
    )

    for attempt in range(config.max_retries):
        attempt_number = attempt + 1
        try:
            response = _call_generate_with_observability(
                client=client,
                prompt=prompt,
                max_tokens=config.max_tokens,
                generate_kwargs=merged_kwargs,
                config=config,
                context=context,
                entry_index=entry_index,
                attempt=attempt_number,
                expected_count=expected_count,
                structured_output_type=structured_output_type,
            )
            text = _safe_text(getattr(response, "text", ""))
            if not text:
                raw = getattr(response, "raw_response", None)
                stop_reason = getattr(raw, "stop_reason", None) if raw is not None else None
                model = getattr(raw, "model", None) if raw is not None else None
                content_types: List[str] = []
                if raw is not None:
                    content = getattr(raw, "content", None)
                    if content:
                        for block in content:
                            block_type = getattr(block, "type", None)
                            if block_type is None and isinstance(block, dict):
                                block_type = block.get("type")
                            content_types.append(str(block_type))
                raise ValueError(
                    "Model returned empty output "
                    f"(model={model}, stop_reason={stop_reason}, content_types={content_types})"
                )
            try:
                parsed = parse_generated_distractors(text, expected_count, start_letter)
            except ValueError as parse_exc:
                if not parse_retry_used and attempt < config.max_retries - 1:
                    parse_retry_used = True
                    _log_event(
                        config,
                        "parse_retry",
                        context=context,
                        entry_index=entry_index,
                        attempt=attempt_number,
                        expected_count=expected_count,
                        error_type=type(parse_exc).__name__,
                        error=str(parse_exc),
                    )
                    time.sleep(config.retry_delay * (2**attempt))
                    continue
                raise StructuredParseError(
                    f"{parse_exc} Raw preview: '{_preview_text(text)}'"
                ) from parse_exc
            return parsed, text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            _log_event(
                config,
                "request_retry",
                context=context,
                entry_index=entry_index,
                attempt=attempt_number,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            if isinstance(exc, StructuredParseError):
                raise RuntimeError(
                    f"Generation failed after structured parse retry for {context} "
                    f"(entry_index={entry_index}): {exc}"
                ) from exc
            if attempt < config.max_retries - 1:
                time.sleep(config.retry_delay * (2**attempt))
    raise RuntimeError(
        f"Generation failed after {config.max_retries} attempts for {context} "
        f"(entry_index={entry_index}): {last_error}"
    )


def _is_missing(values: Any) -> bool:
    return not values


def _write_trace(
    entry: Dict[str, Any],
    strategy: str,
    *,
    prompt: str,
    output: str,
    answer: str,
    distractors_for_eval: List[str],
    seed: int,
) -> None:
    shuffled, correct_letter = _shuffle_options(answer, distractors_for_eval, seed)
    entry[f"{strategy}_full_question"] = f"Question: {entry.get('question', '')}\nAnswer: {answer}"
    entry[f"{strategy}_model_input"] = prompt
    entry[f"{strategy}_model_output"] = output
    entry[f"{strategy}_options_randomized"] = shuffled
    entry[f"{strategy}_correct_answer_letter"] = correct_letter


def _generate_conditioned_options(
    *,
    client,
    question: str,
    answer: str,
    base_context_distractors: List[str],
    total_count: int,
    config: GenerationConfig,
    generate_kwargs: Dict[str, Any],
    context_label: str,
    entry_index: int,
) -> tuple[List[str], str]:
    prompt = _build_conditioned_prompt(question, answer, base_context_distractors, count=total_count)
    start_letter = chr(ord("A") + len(base_context_distractors) + 1)
    return _generate_with_retries(
        client=client,
        prompt=prompt,
        expected_count=total_count,
        start_letter=start_letter,
        config=config,
        generate_kwargs=generate_kwargs,
        context=context_label,
        entry_index=entry_index,
    )


def _generate_qa_options(
    *,
    client,
    question: str,
    answer: str,
    total_count: int,
    start_letter: str,
    config: GenerationConfig,
    generate_kwargs: Dict[str, Any],
    context_label: str,
    entry_index: int,
) -> tuple[List[str], str]:
    prompt = _build_q_a_prompt(question, answer, count=total_count, start_letter=start_letter)
    return _generate_with_retries(
        client=client,
        prompt=prompt,
        expected_count=total_count,
        start_letter=start_letter,
        config=config,
        generate_kwargs=generate_kwargs,
        context=context_label,
        entry_index=entry_index,
    )


def _apply_generation_for_entry(
    *,
    entry: Dict[str, Any],
    idx: int,
    client,
    config: GenerationConfig,
    generate_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    row = dict(entry)
    _ensure_trace_columns(row)
    row["schema_version"] = FINAL5_SCHEMA_VERSION

    question = _safe_text(row.get("question"))
    answer = _safe_text(row.get("answer"))
    if not answer:
        choices_answer = _require_column_list(row, "choices_answer")
        answer = _safe_text(choices_answer[0]) if choices_answer else ""

    if not question or not answer:
        raise NonRetryableAugmentationError("Missing required question/answer")

    human = _require_column_list(row, "choices_human")
    if len(human) < 3:
        raise NonRetryableAugmentationError("Need at least 3 human distractors in choices_human")
    human3 = human[:3]

    # A: human_from_scratch (passthrough)
    row["human_from_scratch"] = human3
    _write_trace(
        row,
        "human_from_scratch",
        prompt="passthrough: choices_human[:3]",
        output="",
        answer=answer,
        distractors_for_eval=human3,
        seed=idx * 10_000 + 1,
    )

    # B: model_from_scratch (3M)
    need_b = config.force_overwrite or _is_missing(row.get("model_from_scratch"))
    if need_b:
        model3, raw_b = _generate_qa_options(
            client=client,
            question=question,
            answer=answer,
            total_count=3,
            start_letter="B",
            config=config,
            generate_kwargs=generate_kwargs,
            context_label="model_from_scratch",
            entry_index=idx,
        )
        prompt_b = _build_q_a_prompt(question, answer, count=3, start_letter="B")
        row["model_from_scratch"] = model3
        _write_trace(
            row,
            "model_from_scratch",
            prompt=prompt_b,
            output=raw_b,
            answer=answer,
            distractors_for_eval=model3,
            seed=idx * 10_000 + 2,
        )

    model3 = _require_column_list(row, "model_from_scratch")
    if len(model3) < 3:
        raise NonRetryableAugmentationError("model_from_scratch must contain 3 distractors")
    model3 = model3[:3]
    row["model_from_scratch"] = model3

    # C: augment_human (delta 6M conditioned on 3H)
    need_c = config.force_overwrite or _is_missing(row.get("augment_human"))
    if need_c:
        c_delta6, raw_c = _generate_conditioned_options(
            client=client,
            question=question,
            answer=answer,
            base_context_distractors=human3,
            total_count=6,
            config=config,
            generate_kwargs=generate_kwargs,
            context_label="augment_human",
            entry_index=idx,
        )
        prompt_c = _build_conditioned_prompt(question, answer, human3, count=6)
        row["augment_human"] = c_delta6
        # Eval setting uses A + C.
        _write_trace(
            row,
            "augment_human",
            prompt=prompt_c,
            output=raw_c,
            answer=answer,
            distractors_for_eval=human3 + c_delta6,
            seed=idx * 10_000 + 3,
        )

    c_delta6 = _require_column_list(row, "augment_human")
    if len(c_delta6) < 6:
        raise NonRetryableAugmentationError("augment_human must contain 6 distractors")
    c_delta6 = c_delta6[:6]
    row["augment_human"] = c_delta6

    # D: augment_model_delta_6m + augment_model combined 9M
    need_d = config.force_overwrite or _is_missing(row.get("augment_model"))
    if need_d:
        d_delta6, raw_d = _generate_conditioned_options(
            client=client,
            question=question,
            answer=answer,
            base_context_distractors=model3,
            total_count=6,
            config=config,
            generate_kwargs=generate_kwargs,
            context_label="augment_model_delta_6m",
            entry_index=idx,
        )
        prompt_d = _build_conditioned_prompt(question, answer, model3, count=6)
        combined9 = model3 + d_delta6
        row["augment_model_delta_6m"] = d_delta6
        row["augment_model"] = combined9
        _write_trace(
            row,
            "augment_model",
            prompt=prompt_d,
            output=raw_d,
            answer=answer,
            distractors_for_eval=combined9,
            seed=idx * 10_000 + 4,
        )

    d_delta6 = _require_column_list(row, "augment_model_delta_6m")
    combined9 = _require_column_list(row, "augment_model")
    if len(d_delta6) < 6 or len(combined9) < 9:
        raise NonRetryableAugmentationError("augment_model requires 6M delta and 9M combined")
    row["augment_model_delta_6m"] = d_delta6[:6]
    row["augment_model"] = combined9[:9]

    # E: augment_ablation (9M direct)
    need_e = config.force_overwrite or _is_missing(row.get("augment_ablation"))
    if need_e:
        e_9m, raw_e = _generate_qa_options(
            client=client,
            question=question,
            answer=answer,
            total_count=9,
            start_letter="B",
            config=config,
            generate_kwargs=generate_kwargs,
            context_label="augment_ablation",
            entry_index=idx,
        )
        prompt_e = _build_q_a_prompt(question, answer, count=9, start_letter="B")
        row["augment_ablation"] = e_9m
        _write_trace(
            row,
            "augment_ablation",
            prompt=prompt_e,
            output=raw_e,
            answer=answer,
            distractors_for_eval=e_9m,
            seed=idx * 10_000 + 5,
        )

    e_9m = _require_column_list(row, "augment_ablation")
    if len(e_9m) < 9:
        raise NonRetryableAugmentationError("augment_ablation must contain 9 distractors")
    row["augment_ablation"] = e_9m[:9]

    return row


def augment_single_dataset(
    dataset: Dataset | Iterable[Dict[str, Any]],
    config: GenerationConfig,
    limit: Optional[int] = None,
    resume_from: Optional[Path] = None,
) -> Dataset:
    entries = list(dataset)
    if limit is not None:
        entries = entries[:limit]

    client = get_client(config.model_name, **_client_kwargs_for_config(config))

    generate_kwargs = dict(config.generate_kwargs)
    provider = config.model_provider.lower().strip()
    if provider == "anthropic" and config.anthropic_thinking:
        generate_kwargs.setdefault("thinking", dict(config.anthropic_thinking))

    processed_rows: List[Dict[str, Any]] = []
    start_idx = 0

    if resume_from is not None:
        p = Path(resume_from)
        if not p.exists():
            raise FileNotFoundError(f"resume_from checkpoint not found: {p}")
        loaded = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError("resume checkpoint must be a JSON list")
        processed_rows = loaded
        start_idx = len(processed_rows)

    temp_path = RESULTS_DIR / f"temp_final5_{config.model_name.replace('/', '_')}_{int(time.time())}.json"
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    _log_event(
        config,
        "dataset_start",
        rows=len(entries),
        limit=limit,
        resume_from=str(resume_from) if resume_from else None,
        request_log_path=str(config.request_log_path) if config.request_log_path else None,
    )

    failures = 0
    for idx in tqdm(range(start_idx, len(entries)), desc="Generating Final5"):
        row = dict(entries[idx])
        try:
            out = _apply_generation_for_entry(
                entry=row,
                idx=idx,
                client=client,
                config=config,
                generate_kwargs=generate_kwargs,
            )
            processed_rows.append(out)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            if config.skip_failed_entries:
                row.setdefault("generation_failures", [])
                row["generation_failures"].append(
                    {
                        "index": idx,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                processed_rows.append(row)
                _log_event(
                    config,
                    "row_failed_skipped",
                    entry_index=idx,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            else:
                raise

        if (idx + 1) % config.save_interval == 0:
            temp_path.write_text(json.dumps(processed_rows, indent=2), encoding="utf-8")

    temp_path.write_text(json.dumps(processed_rows, indent=2), encoding="utf-8")
    if failures:
        print(f"Generation completed with {failures} failed rows")
    _log_event(config, "dataset_end", rows_processed=len(processed_rows), failures=failures)

    return Dataset.from_list(processed_rows)


def augment_dataset(
    dataset_path: Path,
    config: GenerationConfig,
    output_path: Optional[Path] = None,
    limit: Optional[int] = None,
    resume_from: Optional[Path] = None,
    push_to_hub: bool = True,
    splits: Optional[List[str]] = None,
) -> Dataset | DatasetDict:
    ds = load_from_disk(str(dataset_path))

    if isinstance(ds, DatasetDict):
        selected = list(ds.keys())
        if splits:
            missing = sorted(set(splits) - set(selected))
            if missing:
                raise ValueError(f"Requested splits not found: {missing}")
            selected = [x for x in selected if x in splits]

        out: Dict[str, Dataset] = {}
        for split in selected:
            out[split] = augment_single_dataset(ds[split], config, limit=limit, resume_from=resume_from)
        result: Dataset | DatasetDict = DatasetDict(out)
    else:
        result = augment_single_dataset(ds, config, limit=limit, resume_from=resume_from)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save_to_disk(str(output_path))
        print(f"Saved Final5 dataset to {output_path}")
        if push_to_hub:
            push_dataset_to_hub(result, dataset_name=output_path.name)

    return result


def get_output_column(mode: AugmentorMode) -> str:
    """Legacy compatibility helper."""
    if mode in {AugmentorMode.FROM_SCRATCH, AugmentorMode.FINAL5}:
        return "model_from_scratch"
    if mode == AugmentorMode.CONDITIONED_HUMAN:
        return "augment_human"
    if mode == AugmentorMode.CONDITIONED_SYNTHETIC:
        return "augment_model"
    raise ValueError(f"Unsupported mode: {mode}")
