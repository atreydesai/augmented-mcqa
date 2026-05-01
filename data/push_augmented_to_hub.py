"""Push augmented datasets to the Hugging Face Hub.

Directory layout under datasets/augmented/:
    <run_name>/
        <model_name>/
            <benchmark>/          (arc_challenge | gpqa | mmlu_pro)
                <setting>/        (human_from_scratch | model_from_scratch | ...)
                    <arrow files>

Each leaf dataset has per-row columns:
  distractors, options_randomized, correct_answer_letter
  (plus shared metadata: id, sample_id, question, answer, options, ...)

This script produces a **wide** dataset where all settings for a question are
merged into a single row using the CANONICAL_FEATURES schema from pushtohub.py:
  human_from_scratch, human_from_scratch_options_randomized,
  human_from_scratch_correct_answer_letter, model_from_scratch, ...

Three repos are pushed:
  atreydesai/augmented-mcqa-gemini-augmented
  atreydesai/augmented-mcqa-gpt-augmented
  atreydesai/augmented-mcqa-together-augmented
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Features, Sequence, Value, concatenate_datasets, load_from_disk
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import AUGMENTED_DATASETS_DIR, HF_TOKEN
from utils.constants import SETTING_NAMES

# ---------------------------------------------------------------------------
# Repository map: local run folder -> HF repo id
# ---------------------------------------------------------------------------
RUNS: dict[str, str] = {
    "gemini_from_scratch_testing": "atreydesai/augmented-mcqa-gemini-augmented",
    "gen_gpt52_v2": "atreydesai/augmented-mcqa-gpt-augmented",
    "together_from_scratch_testing": "atreydesai/augmented-mcqa-together-augmented",
}

# ---------------------------------------------------------------------------
# Wide-format canonical features (mirrors pushtohub.py CANONICAL_FEATURES)
# ---------------------------------------------------------------------------
WIDE_FEATURES = Features(
    {
        "id": Value("string"),
        "sample_id": Value("string"),
        "question_id": Value("int64"),
        "row_index": Value("int64"),
        "dataset_type": Value("string"),
        "question": Value("string"),
        "answer": Value("string"),
        "answer_index": Value("int64"),
        "category": Value("string"),
        "options": Sequence(Value("string")),
        "choices_human": Sequence(Value("string")),
        # per-setting columns
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
    }
)

# Shared metadata columns that come from the first available setting
SHARED_COLS = [
    "id",
    "sample_id",
    "question_id",
    "row_index",
    "dataset_type",
    "question",
    "answer",
    "answer_index",
    "category",
    "options",
    "choices_human",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_str(v: object) -> str:
    return "" if v is None else str(v)


def _to_int(v: object, default: int = -1) -> int:
    if v in (None, ""):
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _to_list(v: object) -> list[str]:
    if not isinstance(v, list):
        return []
    return [str(x) for x in v if x is not None]


def load_setting(run_root: Path, setting: str) -> Dataset | None:
    """Load and concatenate all benchmarks for a given setting under run_root."""
    parts: list[Dataset] = []
    for benchmark_dir in sorted(run_root.iterdir()):
        if not benchmark_dir.is_dir():
            continue
        setting_dir = benchmark_dir / setting
        if not setting_dir.is_dir():
            continue
        try:
            ds = load_from_disk(str(setting_dir))
            parts.append(ds)
        except Exception as exc:  # noqa: BLE001
            print(f"    WARNING: could not load {setting_dir}: {exc}")
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def build_wide_dataset(run_root: Path) -> Dataset:
    """Merge all settings by sample_id into a single wide dataset."""
    # wide_rows: sample_id -> dict accumulating columns
    wide_rows: dict[str, dict[str, Any]] = {}

    for setting in SETTING_NAMES:
        print(f"    Loading setting: {setting} ...", end=" ", flush=True)
        ds = load_setting(run_root, setting)
        if ds is None:
            print("SKIPPED (no data)")
            continue
        print(f"{len(ds)} rows")

        for row in ds:
            sid = _to_str(row.get("sample_id") or row.get("id") or row.get("question_id"))
            if not sid:
                continue

            if sid not in wide_rows:
                base: dict[str, Any] = {col: None for col in WIDE_FEATURES}
                # populate shared columns from this first row
                base["id"] = _to_str(row.get("id") or row.get("sample_id"))
                base["sample_id"] = sid
                base["question_id"] = _to_int(row.get("question_id"))
                base["row_index"] = _to_int(row.get("row_index"))
                base["dataset_type"] = _to_str(row.get("dataset_type"))
                base["question"] = _to_str(row.get("question"))
                base["answer"] = _to_str(row.get("answer"))
                base["answer_index"] = _to_int(row.get("answer_index"))
                base["category"] = _to_str(row.get("category"))
                base["options"] = _to_list(row.get("options"))
                base["choices_human"] = _to_list(row.get("choices_human"))
                wide_rows[sid] = base

            entry = wide_rows[sid]
            entry[setting] = _to_list(row.get("distractors"))
            entry[f"{setting}_options_randomized"] = _to_list(row.get("options_randomized"))
            entry[f"{setting}_correct_answer_letter"] = _to_str(row.get("correct_answer_letter"))

    if not wide_rows:
        raise ValueError(f"No data found under {run_root}")

    # Fill any None list fields with [] and None str fields with ""
    rows_list: list[dict[str, Any]] = []
    for entry in wide_rows.values():
        clean: dict[str, Any] = {}
        for feat_name, feat_type in WIDE_FEATURES.items():
            val = entry.get(feat_name)
            if isinstance(feat_type, Sequence):
                clean[feat_name] = val if isinstance(val, list) else []
            elif isinstance(feat_type, Value) and feat_type.dtype == "int64":
                clean[feat_name] = val if isinstance(val, int) else -1
            else:
                clean[feat_name] = val if isinstance(val, str) else ""
        rows_list.append(clean)

    return Dataset.from_list(rows_list, features=WIDE_FEATURES)


def push_run(run_name: str, repo_id: str) -> None:
    run_base = AUGMENTED_DATASETS_DIR / run_name
    print(f"\n[{run_name}] -> {repo_id}")

    # Each sub-directory of the run is a model folder
    model_dirs = sorted(p for p in run_base.iterdir() if p.is_dir())
    if not model_dirs:
        print(f"  No model directories found under {run_base}, skipping.")
        return

    # Build a DatasetDict with one config (model name) per model directory.
    # For simplicity we push one config named after the model, with a "train" split.
    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"  Model: {model_name}")
        ds = build_wide_dataset(model_dir)
        print(f"  Wide dataset: {len(ds)} rows, {len(ds.column_names)} columns")

        dataset_dict = DatasetDict({"train": ds})
        push_kwargs: dict[str, Any] = {"config_name": model_name}
        if HF_TOKEN:
            push_kwargs["token"] = HF_TOKEN

        print(f"  Pushing to {repo_id} (config={model_name}) ...")
        dataset_dict.push_to_hub(repo_id, **push_kwargs)
        print(f"  Done.")


def main() -> None:
    try:
        for run_name, repo_id in RUNS.items():
            push_run(run_name, repo_id)
    except (HfHubHTTPError, RepositoryNotFoundError) as exc:
        message = str(exc)
        if "401" in message or "Invalid username or password" in message:
            raise RuntimeError(
                "Hugging Face authentication failed. Set HF_TOKEN to a valid write-scoped token and rerun."
            ) from exc
        raise

    print("\nAll done!")


if __name__ == "__main__":
    main()
