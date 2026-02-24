#!/usr/bin/env python3
"""Process raw datasets into the Final5 unified schema.

Active datasets:
- arc_challenge
- mmlu_pro
- gpqa

MMLU-Pro filtering-vs-MMLU remains delegated to data.mmlu_pro_processor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, concatenate_datasets

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PROCESSED_DATASETS_DIR
from data import (
    process_arc_for_experiments,
    process_gpqa_for_experiments,
    process_mmlu_pro as process_mmlu_pro_func,
)


SCHEMA_VERSION = "final5_v1"
DEFAULT_PER_DATASET_LIMIT = 1000


STRATEGY_IDS = [
    "human_from_scratch",
    "model_from_scratch",
    "augment_human",
    "augment_model",
    "augment_ablation",
]


def _limit_dataset(ds: Dataset, limit: Optional[int]) -> Dataset:
    if limit is None:
        return ds
    return ds.select(range(min(limit, len(ds))))


def _add_or_replace_column(ds: Dataset, name: str, values: list) -> Dataset:
    if name in ds.column_names:
        ds = ds.remove_columns([name])
    return ds.add_column(name, values)


def _init_strategy_columns(ds: Dataset) -> Dataset:
    n = len(ds)

    list_columns = [
        "human_from_scratch",
        "model_from_scratch",
        "augment_human",
        "augment_model_delta_6m",
        "augment_model",
        "augment_ablation",
    ]

    for col in list_columns:
        ds = _add_or_replace_column(ds, col, [[] for _ in range(n)])

    trace_cols = []
    for strategy in STRATEGY_IDS:
        trace_cols.extend(
            [
                f"{strategy}_full_question",
                f"{strategy}_model_input",
                f"{strategy}_model_output",
                f"{strategy}_options_randomized",
                f"{strategy}_correct_answer_letter",
            ]
        )

    for col in trace_cols:
        if col.endswith("_options_randomized"):
            ds = _add_or_replace_column(ds, col, [[] for _ in range(n)])
        else:
            ds = _add_or_replace_column(ds, col, ["" for _ in range(n)])

    ds = _add_or_replace_column(ds, "schema_version", [SCHEMA_VERSION for _ in range(n)])
    return ds


def _ensure_common_columns(ds: Dataset, dataset_type: str) -> Dataset:
    n = len(ds)
    defaults = {
        "question": "",
        "options": [],
        "answer": "",
        "answer_index": -1,
        "category": "",
        "src": "",
        "subfield": "",
        "difficulty": "",
        "discipline": "",
        "labels": [],
        "choices_answer": [],
        "choices_human": [],
    }

    for col, default in defaults.items():
        if col not in ds.column_names:
            fill = [default for _ in range(n)]
            ds = ds.add_column(col, fill)

    ds = _add_or_replace_column(ds, "dataset_type", [dataset_type for _ in range(n)])
    ds = _init_strategy_columns(ds)
    return ds


def _process_mmlu_pro(limit: Optional[int]) -> Dataset:
    output_path = PROCESSED_DATASETS_DIR / "mmlu_pro_processed"
    processed = process_mmlu_pro_func(output_path=output_path, limit=None)

    if "train" in processed and "test" in processed:
        merged = concatenate_datasets([processed["train"], processed["test"]])
    elif "train" in processed:
        merged = processed["train"]
    elif "test" in processed:
        merged = processed["test"]
    else:
        raise ValueError("MMLU-Pro processor returned no train/test split")

    merged = _limit_dataset(merged, limit)
    return _ensure_common_columns(merged, "mmlu_pro")


def _process_arc_challenge(limit: Optional[int]) -> Dataset:
    ds = process_arc_for_experiments(
        difficulty="challenge",
        output_path=PROCESSED_DATASETS_DIR / "arc_processed" / "arc_challenge",
        limit=None,
    )
    ds = _limit_dataset(ds, limit)
    return _ensure_common_columns(ds, "arc_challenge")


def _process_gpqa(limit: Optional[int]) -> Dataset:
    ds = process_gpqa_for_experiments(
        output_path=PROCESSED_DATASETS_DIR / "gpqa_processed",
        limit=None,
    )
    ds = _limit_dataset(ds, limit)
    return _ensure_common_columns(ds, "gpqa")


def run_all(limit: Optional[int] = None, output_path: Optional[Path] = None) -> DatasetDict:
    effective_limit = DEFAULT_PER_DATASET_LIMIT if limit is None else limit

    print("Processing MMLU-Pro with preserved MMLU filtering...")
    mmlu_pro = _process_mmlu_pro(effective_limit)

    print("Processing ARC-Challenge...")
    arc_challenge = _process_arc_challenge(effective_limit)

    print("Processing GPQA...")
    gpqa = _process_gpqa(effective_limit)

    unified = DatasetDict(
        {
            "arc_challenge": arc_challenge,
            "mmlu_pro": mmlu_pro,
            "gpqa": gpqa,
        }
    )

    if output_path is None:
        output_path = PROCESSED_DATASETS_DIR / "unified_processed_v2"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unified.save_to_disk(str(output_path))

    print(f"Saved unified dataset to {output_path}")
    for split in unified.keys():
        print(f"  - {split}: {len(unified[split])} rows")

    return unified


def main() -> int:
    parser = argparse.ArgumentParser(description="Process all active datasets for Final5")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Per-dataset row cap. If omitted, defaults to deterministic first 1000 rows per dataset"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(PROCESSED_DATASETS_DIR / "unified_processed_v2"),
        help="Output path for unified processed dataset",
    )
    args = parser.parse_args()

    run_all(limit=args.limit, output_path=Path(args.output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
