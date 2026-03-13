from __future__ import annotations

import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import HF_TOKEN


DATASET_PATH = "/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/datasets/augmented/gen_gpt52_v2/openai_gpt-5.2-2025-12-11"
REPO_ID = "atreydesai/gen_gpt52_v2"

CANONICAL_FEATURES = Features(
    {
        "id": Value("string"),
        "question_id": Value("int64"),
        "question": Value("string"),
        "options": Sequence(Value("string")),
        "labels": Sequence(Value("string")),
        "answer": Value("string"),
        "choices_answer": Sequence(Value("string")),
        "answer_index": Value("int64"),
        "answer_letter": Value("string"),
        "category": Value("string"),
        "choices_human": Sequence(Value("string")),
        "src": Value("string"),
        "subfield": Value("string"),
        "difficulty": Value("string"),
        "discipline": Value("string"),
        "dataset_type": Value("string"),
        "schema_version": Value("string"),
        "row_index": Value("int64"),
        "sample_id": Value("string"),
        "human_from_scratch": Sequence(Value("string")),
        "human_from_scratch_options_randomized": Sequence(Value("string")),
        "human_from_scratch_correct_answer_letter": Value("string"),
        "model_from_scratch": Sequence(Value("string")),
        "model_from_scratch_options_randomized": Sequence(Value("string")),
        "model_from_scratch_correct_answer_letter": Value("string"),
        "augment_human": Sequence(Value("string")),
        "augment_human_options_randomized": Sequence(Value("string")),
        "augment_human_correct_answer_letter": Value("string"),
        "augment_model": Sequence(Value("string")),
        "augment_model_options_randomized": Sequence(Value("string")),
        "augment_model_correct_answer_letter": Value("string"),
        "augment_ablation": Sequence(Value("string")),
        "augment_ablation_options_randomized": Sequence(Value("string")),
        "augment_ablation_correct_answer_letter": Value("string"),
        "generation_status": Value("string"),
        "cot_content": Value("string"),
        "whitespace_bug_fixed": Value("bool"),
    }
)

LIST_COLUMNS = {
    "options",
    "labels",
    "choices_answer",
    "choices_human",
    "human_from_scratch",
    "human_from_scratch_options_randomized",
    "model_from_scratch",
    "model_from_scratch_options_randomized",
    "augment_human",
    "augment_human_options_randomized",
    "augment_model",
    "augment_model_options_randomized",
    "augment_ablation",
    "augment_ablation_options_randomized",
}

INT_COLUMNS = {"question_id", "answer_index", "row_index"}
BOOL_COLUMNS = {"whitespace_bug_fixed"}


def _normalize_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _normalize_int(value: object, *, default: int = -1) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_bool(value: object) -> bool:
    return bool(value) if value is not None else False


def _normalize_string(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_row(row: dict, split_name: str) -> dict:
    normalized: dict[str, object] = {}
    for column in CANONICAL_FEATURES:
        value = row.get(column)
        if column in LIST_COLUMNS:
            normalized[column] = _normalize_list(value)
        elif column in INT_COLUMNS:
            normalized[column] = _normalize_int(value)
        elif column in BOOL_COLUMNS:
            normalized[column] = _normalize_bool(value)
        else:
            normalized[column] = _normalize_string(value)

    if not normalized["id"]:
        fallback_id = row.get("sample_id")
        if fallback_id in (None, ""):
            fallback_id = row.get("question_id")
        normalized["id"] = _normalize_string(fallback_id)

    if normalized["question_id"] == -1:
        normalized["question_id"] = _normalize_int(row.get("question_id"))

    if not normalized["dataset_type"]:
        normalized["dataset_type"] = split_name

    return normalized


def normalize_dataset_dict(dataset: DatasetDict) -> DatasetDict:
    normalized = {}
    for split_name, split in dataset.items():
        rows = [_normalize_row(dict(row), split_name) for row in split]
        normalized[split_name] = Dataset.from_list(rows, features=CANONICAL_FEATURES)
    return DatasetDict(normalized)


def main() -> None:
    print("Loading dataset from local disk...")
    dataset = load_from_disk(DATASET_PATH)

    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict at {DATASET_PATH}")

    print("Normalizing split schemas...")
    dataset = normalize_dataset_dict(dataset)

    print("Pushing to hub")
    push_kwargs = {}
    if HF_TOKEN:
        push_kwargs["token"] = HF_TOKEN

    try:
        dataset.push_to_hub(REPO_ID, **push_kwargs)
    except (HfHubHTTPError, RepositoryNotFoundError) as exc:
        message = str(exc)
        if "401" in message or "Invalid username or password" in message:
            raise RuntimeError(
                "Hugging Face authentication failed. Set HF_TOKEN to a valid write-scoped token "
                "for the account or org that owns the target dataset repo, then rerun "
                "`python data/pushtohub.py`."
            ) from exc
        raise


if __name__ == "__main__":
    main()
