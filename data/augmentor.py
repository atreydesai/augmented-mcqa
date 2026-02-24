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
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm

from config import RESULTS_DIR
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


@dataclass
class GenerationConfig:
    mode: AugmentorMode = AugmentorMode.FINAL5
    model_provider: str = "openai"
    model_name: str = "gpt-5.2-2025-12-11"
    max_tokens: int = 2048
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


STRATEGY_IDS = [
    "human_from_scratch",
    "model_from_scratch",
    "augment_human",
    "augment_model",
    "augment_ablation",
]


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

    expected_letters = _expected_letters(start_letter, expected_count)
    expected_set = set(expected_letters)
    by_letter: Dict[str, str] = {}
    pattern = r"^\s*([A-Z])\s*(?:[:\.\)\-])\s*(.+?)\s*$"

    for line in response.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(pattern, line, re.IGNORECASE)
        if not m:
            continue
        letter = m.group(1).upper()
        if letter not in expected_set:
            continue
        text = m.group(2).strip().rstrip(".")
        if text and letter not in by_letter:
            by_letter[letter] = text

    missing = [x for x in expected_letters if x not in by_letter]
    if missing:
        raise ValueError(
            f"Expected letters {expected_letters}, missing {missing}. Output invalid."
        )

    return [by_letter[x] for x in expected_letters]


def _build_q_a_prompt(question: str, answer: str, count: int, start_letter: str) -> str:
    letters = ", ".join(_expected_letters(start_letter, count))
    return (
        "I have a multiple-choice question with one known correct answer.\n"
        "Generate plausible but incorrect distractor options.\n\n"
        f"Question: {question}\n"
        f"Correct Answer: A: {answer}\n\n"
        f"Generate exactly {count} incorrect options using labels: {letters}.\n"
        "Output one option per line with format '<LETTER>: <option>'."
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
    letters = ", ".join(_expected_letters(start_letter, count))

    return (
        "I have a multiple-choice question with existing options where A is correct.\n"
        "Generate additional plausible but incorrect distractors.\n\n"
        f"Question: {question}\n"
        "Existing Options:\n"
        + "\n".join(lines)
        + "\n\n"
        + f"Generate exactly {count} new incorrect options: {letters}.\n"
        + "Output one option per line with format '<LETTER>: <option>'."
    )


def _build_repair_prompt(raw_output: str, start_letter: str, expected_count: int) -> str:
    letters = ", ".join(_expected_letters(start_letter, expected_count))
    return (
        "Reformat the following text into strict distractor lines.\n"
        f"Output exactly {expected_count} lines using labels: {letters}.\n"
        "Line format must be '<LETTER>: <option>' and nothing else.\n\n"
        "Text:\n"
        f"{raw_output}"
    )


def _parse_or_repair(
    *,
    client,
    text: str,
    expected_count: int,
    start_letter: str,
    max_tokens: int,
    generate_kwargs: Dict[str, Any],
) -> tuple[List[str], str]:
    try:
        return parse_generated_distractors(text, expected_count, start_letter), text
    except ValueError:
        repair_prompt = _build_repair_prompt(text, start_letter, expected_count)
        repaired = client.generate(
            prompt=repair_prompt,
            max_tokens=max_tokens,
            **generate_kwargs,
        )
        parsed = parse_generated_distractors(repaired.text, expected_count, start_letter)
        return parsed, repaired.text


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

    if provider == "openai":
        if config.reasoning_effort:
            kwargs["reasoning_effort"] = config.reasoning_effort
    return kwargs


def _generate_with_retries(
    *,
    client,
    prompt: str,
    expected_count: int,
    start_letter: str,
    config: GenerationConfig,
    generate_kwargs: Dict[str, Any],
) -> tuple[List[str], str]:
    last_error: Optional[Exception] = None
    for attempt in range(config.max_retries):
        try:
            response = client.generate(prompt=prompt, max_tokens=config.max_tokens, **generate_kwargs)
            return _parse_or_repair(
                client=client,
                text=response.text,
                expected_count=expected_count,
                start_letter=start_letter,
                max_tokens=config.max_tokens,
                generate_kwargs=generate_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < config.max_retries - 1:
                time.sleep(config.retry_delay * (2**attempt))
    raise RuntimeError(f"Generation failed after {config.max_retries} attempts: {last_error}")


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
        prompt_b = _build_q_a_prompt(question, answer, count=3, start_letter="B")
        model3, raw_b = _generate_with_retries(
            client=client,
            prompt=prompt_b,
            expected_count=3,
            start_letter="B",
            config=config,
            generate_kwargs=generate_kwargs,
        )
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
        prompt_c = _build_conditioned_prompt(question, answer, human3, count=6)
        c_delta6, raw_c = _generate_with_retries(
            client=client,
            prompt=prompt_c,
            expected_count=6,
            start_letter="E",
            config=config,
            generate_kwargs=generate_kwargs,
        )
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
        prompt_d = _build_conditioned_prompt(question, answer, model3, count=6)
        d_delta6, raw_d = _generate_with_retries(
            client=client,
            prompt=prompt_d,
            expected_count=6,
            start_letter="E",
            config=config,
            generate_kwargs=generate_kwargs,
        )
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
        prompt_e = _build_q_a_prompt(question, answer, count=9, start_letter="B")
        e_9m, raw_e = _generate_with_retries(
            client=client,
            prompt=prompt_e,
            expected_count=9,
            start_letter="B",
            config=config,
            generate_kwargs=generate_kwargs,
        )
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
            else:
                raise

        if (idx + 1) % config.save_interval == 0:
            temp_path.write_text(json.dumps(processed_rows, indent=2), encoding="utf-8")

    temp_path.write_text(json.dumps(processed_rows, indent=2), encoding="utf-8")
    if failures:
        print(f"Generation completed with {failures} failed rows")

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
