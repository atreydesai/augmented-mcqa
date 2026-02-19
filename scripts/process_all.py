#!/usr/bin/env python
"""
Process downloaded datasets into a unified experiment-ready dataset.

Usage:
    python scripts/process_all.py
    python scripts/process_all.py --dataset mmlu_pro
    python scripts/process_all.py --dataset arc
    python scripts/process_all.py --dataset gpqa
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PROCESSED_DATASETS_DIR
from data import (
    process_arc_for_experiments,
    process_gpqa_for_experiments,
    process_mmlu_pro as process_mmlu_pro_func,
)


def run_mmlu_pro(limit: Optional[int] = None):
    """Sort MMLU-Pro into human vs synthetic distractors."""
    print("\n" + "=" * 60)
    print("Processing MMLU-Pro (sorting human vs synthetic distractors)")
    print("=" * 60)

    output_path = PROCESSED_DATASETS_DIR / "mmlu_pro_processed"
    result = process_mmlu_pro_func(output_path=output_path, limit=limit)

    print(f"✅ MMLU-Pro processed -> {output_path}")
    return result


def run_arc(limit: Optional[int] = None):
    """Process ARC-Easy and ARC-Challenge into unified format."""
    print("\n" + "=" * 60)
    print("Processing ARC (Easy + Challenge)")
    print("=" * 60)

    arc_easy = process_arc_for_experiments(
        difficulty="easy",
        output_path=PROCESSED_DATASETS_DIR / "arc_processed" / "arc_easy",
        limit=limit,
    )
    print(f"  ARC-Easy: {len(arc_easy)} entries")

    arc_challenge = process_arc_for_experiments(
        difficulty="challenge",
        output_path=PROCESSED_DATASETS_DIR / "arc_processed" / "arc_challenge",
        limit=limit,
    )
    print(f"  ARC-Challenge: {len(arc_challenge)} entries")

    print(f"✅ ARC processed -> {PROCESSED_DATASETS_DIR / 'arc_processed'}")
    return {"easy": arc_easy, "challenge": arc_challenge}


def run_gpqa(limit: Optional[int] = None):
    """Process GPQA into unified format."""
    print("\n" + "=" * 60)
    print("Processing GPQA (gpqa_main/train)")
    print("=" * 60)

    result = process_gpqa_for_experiments(
        output_path=PROCESSED_DATASETS_DIR / "gpqa_processed",
        limit=limit,
    )

    print(f"✅ GPQA processed -> {PROCESSED_DATASETS_DIR / 'gpqa_processed'}")
    return result


def add_empty_columns(dataset):
    """Initialize empty generation placeholder columns."""
    new_columns = {
        # Scratch path
        "cond_model_q_a_scratch": None,
        "qa_options_randomized": None,
        "qa_correct_answer_letter": None,
        "qa_full_question": None,
        "qa_model_input": None,
        "qa_model_output": None,
        # Distractor-Human path
        "cond_model_q_a_dhuman": None,
        "qadh_options_randomized": None,
        "qadh_correct_answer_letter": None,
        "qadh_full_question": None,
        "qadh_model_input": None,
        "qadh_model_output": None,
        # Distractor-Model path
        "cond_model_q_a_dmodel": None,
        "qadm_options_randomized": None,
        "qadm_correct_answer_letter": None,
        "qadm_full_question": None,
        "qadm_model_input": None,
        "qadm_model_output": None,
    }

    for col_name, default_val in new_columns.items():
        if col_name not in dataset.features:
            dataset = dataset.add_column(col_name, [default_val] * len(dataset))
    return dataset


def harmonize_features(dataset_dict):
    """Ensure all splits have the exact same column schema."""
    from datasets import Features, Sequence, Value

    all_columns = set()
    for ds in dataset_dict.values():
        all_columns.update(ds.column_names)

    print(f"\nHarmonizing schemas. Union of columns: {sorted(list(all_columns))}")

    datasets_with_cols = {}
    for name, ds in dataset_dict.items():
        missing_cols = all_columns - set(ds.column_names)
        if missing_cols:
            print(f"  {name}: Adding missing columns {list(missing_cols)}")
            for col in missing_cols:
                ds = ds.add_column(col, [None] * len(ds))
        datasets_with_cols[name] = ds

    print("\nPhase 2: Determining unified feature types...")
    unified_features = {}
    for col in list(all_columns):
        found_feat = None
        for ds in datasets_with_cols.values():
            feat = ds.features.get(col)
            if feat and hasattr(feat, "dtype") and feat.dtype != "null":
                found_feat = feat
                break
            if isinstance(feat, Sequence):
                found_feat = feat
                break

        if found_feat is None:
            found_feat = Value("string")

        unified_features[col] = found_feat

    target_features = Features(unified_features)

    final_datasets = {}
    print("Phase 3: Casting datasets to unified schema...")
    for name, ds in datasets_with_cols.items():
        final_datasets[name] = ds.cast(target_features)

    return final_datasets


def run_all(limit: Optional[int] = None):
    """Process all datasets and combine into a unified processed dataset."""
    from datasets import DatasetDict, concatenate_datasets

    print("Step 1: Processing individual datasets...")
    mmlu_pro_dict = run_mmlu_pro(limit=limit)
    arc_results = run_arc(limit=limit)
    gpqa_ds = run_gpqa(limit=limit)

    print("\nStep 2: Aggregating into unified dataset...")
    unified_splits = {
        "arc_easy": arc_results["easy"],
        "arc_challenge": arc_results["challenge"],
        "gpqa": gpqa_ds,
    }

    if "train" in mmlu_pro_dict:
        ds = mmlu_pro_dict["train"]
        if "test" in mmlu_pro_dict:
            print("  Combining MMLU-Pro train + test for unified 'mmlu_pro' split...")
            ds = concatenate_datasets([ds, mmlu_pro_dict["test"]])
        unified_splits["mmlu_pro"] = ds
    elif "test" in mmlu_pro_dict:
        unified_splits["mmlu_pro"] = mmlu_pro_dict["test"]

    print("\nStep 3: Initializing empty columns for generation placeholders...")
    datasets_with_empty_cols = {}
    for split_name, dataset in unified_splits.items():
        print(f"  Initializing standard columns for {split_name}...")
        datasets_with_empty_cols[split_name] = add_empty_columns(dataset)

    print("\nStep 4: Harmonizing schemas (union of all columns)...")
    final_splits = harmonize_features(datasets_with_empty_cols)

    ordered_columns = [
        "question",
        "options",
        "answer",
        "answer_index",
        "category",
        "src",
        "subfield",
        "difficulty",
        "choices_answer",
        "choices_human",
        "cond_model_q_a_scratch",
        "qa_options_randomized",
        "qa_correct_answer_letter",
        "qa_full_question",
        "qa_model_input",
        "qa_model_output",
        "cond_model_q_a_dhuman",
        "qadh_options_randomized",
        "qadh_correct_answer_letter",
        "qadh_full_question",
        "qadh_model_input",
        "qadh_model_output",
        "cond_model_q_a_dmodel",
        "qadm_options_randomized",
        "qadm_correct_answer_letter",
        "qadm_full_question",
        "qadm_model_input",
        "qadm_model_output",
    ]

    print("\nStep 4.5: Reordering columns for better visibility...")
    reordered_splits = {}
    for name, ds in final_splits.items():
        remaining_cols = [c for c in ds.column_names if c not in ordered_columns]
        final_column_order = [c for c in ordered_columns if c in ds.column_names] + remaining_cols
        reordered_splits[name] = ds.select_columns(final_column_order)

    final_dataset = DatasetDict(reordered_splits)
    output_path = PROCESSED_DATASETS_DIR / "unified_processed"

    print(f"\nStep 5: Saving unified dataset to {output_path}...")
    final_dataset.save_to_disk(str(output_path))

    from data.hub_utils import push_dataset_to_hub

    push_dataset_to_hub(final_dataset, repo_id="atreydesai/qgqa-unified-processed")

    print("\n✅ Verification:")
    print(f"  Splits: {list(final_dataset.keys())}")
    for split in final_dataset.keys():
        print(f"  - {split}: {len(final_dataset[split])} rows")
        print(f"    First 5 columns: {final_dataset[split].column_names[:5]}")
        print(
            "    ✓ 'qa_model_input' column initialized"
            if "qa_model_input" in final_dataset[split].features
            else "    ❌ 'qa_model_input' MISSING"
        )

    return final_dataset


def main():
    parser = argparse.ArgumentParser(description="Process all datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mmlu_pro", "arc", "gpqa", "all"],
        default="all",
        help="Which dataset to process (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit entries per dataset (for testing)",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        run_all(limit=args.limit)
    elif args.dataset == "mmlu_pro":
        run_mmlu_pro(limit=args.limit)
    elif args.dataset == "arc":
        run_arc(limit=args.limit)
    elif args.dataset == "gpqa":
        run_gpqa(limit=args.limit)


if __name__ == "__main__":
    main()
