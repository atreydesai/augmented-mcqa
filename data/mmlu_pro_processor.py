#!/usr/bin/env python3
"""
MMLU-Pro Dataset Processor.

Processes MMLU-Pro to:
1. Identify human vs synthetic distractors by matching against original MMLU.
2. Fix leading whitespace issues in STEM categories (chemistry, physics, math).
3. Filter to 10-option questions for consistent distractor sets.
4. Push processed datasets to Hugging Face Hub.
"""

import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict

from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATASETS_DIR,
    RAW_DATASETS_DIR,
    PROCESSED_DATASETS_DIR,
    DistractorType,
    RANDOM_SEED,
)
from data.hub_utils import push_dataset_to_hub


# Categories with known leading whitespace issues in MMLU-Pro
WHITESPACE_BUG_CATEGORIES = {"chemistry", "physics", "math"}


def clean_whitespace(text: str) -> str:
    """Strip leading/trailing whitespace from text."""
    if isinstance(text, str):
        return text.strip()
    return text


def clean_options(options: List[str]) -> List[str]:
    """Clean whitespace from all options in a list."""
    return [clean_whitespace(opt) for opt in options]


def build_mmlu_lookup(mmlu_path: Path) -> Dict[str, Set[str]]:
    """
    Build a lookup table of all MMLU options indexed by question.
    Uses raw strings (NO CLEANING/LOWERING) for exact matching.
    
    Returns:
        Dict mapping raw_question -> set of raw_options
    """
    lookup = defaultdict(set)
    
    print("Building MMLU option lookup table...")
    
    # Use the 'all' configuration which contains all subjects
    all_path = mmlu_path / "all"
    if not all_path.exists():
        print(f"  Warning: {all_path} not found, falling back to all subdirectories")
        dirs_to_scan = [d for d in mmlu_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    else:
        dirs_to_scan = [all_path]
    
    for subset_dir in dirs_to_scan:
        if subset_dir.name == "auxiliary_train":
            continue
            
        try:
            subset_data = load_from_disk(str(subset_dir))
        except Exception as e:
            print(f"  Warning: Failed to load {subset_dir.name}: {e}")
            continue
            
        for split_name in subset_data.keys():
            split_data = subset_data[split_name]
            for entry in split_data:
                if "question" not in entry:
                    continue
                
                # Use RAW strings
                question = entry["question"]
                choices = entry.get("choices", [])
                for choice in choices:
                    lookup[question].add(str(choice))
    
    print(f"  Built lookup with {len(lookup)} questions")
    return lookup


def sort_distractors(
    entry: Dict,
    mmlu_lookup: Dict[str, Set[str]],
) -> Optional[Tuple[List[str], List[str], str]]:
    """
    Sort distractors into human (from MMLU) vs synthetic (added by MMLU-Pro).
    ONLY returns values if the question matches MMLU exactly.
    
    Returns:
        Tuple of (human_distractors, synthetic_distractors, gold_answer) or None if no match
    """
    raw_question = entry["question"]
    if raw_question not in mmlu_lookup:
        return None
        
    mmlu_options = mmlu_lookup[raw_question]
    
    answer_idx = entry.get("answer_index", 0)
    options_raw = entry.get("options", [])
    
    if not options_raw:
        return None
    
    human_distractors_raw = []
    synthetic_distractors_raw = []
    
    for i, opt in enumerate(options_raw):
        if i == answer_idx:
            continue
            
        # Match using RAW strings
        if str(opt) in mmlu_options:
            human_distractors_raw.append(opt)
        else:
            synthetic_distractors_raw.append(opt)
    
    # POST-PROCESSING: Clean whitespaces and build final gold answer
    gold_answer = clean_whitespace(options_raw[answer_idx]) if answer_idx < len(options_raw) else ""
    human_distractors = [clean_whitespace(opt) for opt in human_distractors_raw]
    synthetic_distractors = [clean_whitespace(opt) for opt in synthetic_distractors_raw]
    
    return human_distractors, synthetic_distractors, gold_answer


def process_mmlu_pro(
    mmlu_pro_path: Optional[Path] = None,
    mmlu_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    report_whitespace_bugs: bool = True,
    filter_10_options: bool = True,
    fix_whitespace: bool = True,
    limit: Optional[int] = None,
) -> DatasetDict:
    """
    Process MMLU-Pro to separate human and synthetic distractors.
    """
    # Set default paths
    if mmlu_pro_path is None:
        mmlu_pro_path = RAW_DATASETS_DIR / "mmlu_pro"
    if mmlu_path is None:
        mmlu_path = RAW_DATASETS_DIR / "mmlu_all"
    if output_path is None:
        output_path = PROCESSED_DATASETS_DIR / "mmlu_pro_processed"
    
    mmlu_pro_path = Path(mmlu_pro_path)
    mmlu_path = Path(mmlu_path)
    output_path = Path(output_path)
    
    # Load datasets
    print(f"Loading MMLU-Pro from {mmlu_pro_path}...")
    mmlu_pro = load_from_disk(str(mmlu_pro_path))
    
    # Build lookup
    mmlu_lookup = build_mmlu_lookup(mmlu_path)
    
    # Track whitespace bugs
    whitespace_bugs = []
    
    def process_entry(entry):
        # Sort distractors (filters by exact match)
        sort_res = sort_distractors(entry, mmlu_lookup)
        if sort_res is None:
            return None
            
        human_distractors, synthetic_distractors, gold_answer = sort_res
        
        # Now apply cleaning and fixes to the entry
        category = entry.get("category", "").lower()
        question = clean_whitespace(entry["question"])
        
        had_whitespace_bug = False
        if fix_whitespace and category in WHITESPACE_BUG_CATEGORIES:
            # Check for leading whitespace in the original gold answer
            raw_options = entry.get("options", [])
            answer_idx = entry.get("answer_index", 0)
            if answer_idx < len(raw_options):
                orig = raw_options[answer_idx]
                if orig != orig.strip():
                    had_whitespace_bug = True
                    if report_whitespace_bugs:
                        whitespace_bugs.append({
                            "category": category,
                            "question": question[:100],
                            "original_answer": repr(orig),
                        })

        return {
            **entry,
            "question": question,
            # Schema Unification
            "choices_human": human_distractors,
            "legacy_choices_synthetic": synthetic_distractors,
            "answer": gold_answer,
            "choices_answer": [gold_answer] if gold_answer else [],
            "whitespace_bug_fixed": had_whitespace_bug,
        }
    
    # Process all splits
    processed = {}
    
    for split_name, split_data in mmlu_pro.items():
        if split_name == "validation":
            print(f"Skipping validation split")
            continue
            
        target_name = "train" if split_name == "test" else split_name
        print(f"Processing {split_name} split -> {target_name} ({len(split_data)} entries)...")
        
        processed_entries = []
        skipped_not_matched = 0
        skipped_options = 0
        
        for entry in tqdm(split_data, desc=f"  {split_name}"):
            options = entry.get("options", [])
            
            if filter_10_options and len(options) != 10:
                skipped_options += 1
                continue
            
            res = process_entry(dict(entry))
            if res is None:
                skipped_not_matched += 1
                continue
                
            processed_entries.append(res)
            
            if limit and len(processed_entries) >= limit:
                break
        
        processed[target_name] = Dataset.from_list(processed_entries)
        print(f"    Kept {len(processed_entries)} entries in '{target_name}'")
        if skipped_not_matched > 0:
            print(f"    Skipped (no MMLU match): {skipped_not_matched}")
        if skipped_options > 0:
            print(f"    Skipped (not 10 options): {skipped_options}")
    
    if not processed:
        print("Warning: No splits were processed!")
        return DatasetDict()
        
    result = DatasetDict(processed)
    
    # Report whitespace bugs
    if report_whitespace_bugs and whitespace_bugs:
        print(f"\n⚠️ Found {len(whitespace_bugs)} entries with whitespace bugs")
    
    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    result.save_to_disk(str(output_path))
    print(f"\n✓ Saved processed dataset to {output_path}")
    
    # Push to Hugging Face
    push_dataset_to_hub(result, repo_id="atreydesai/qgqa-mmlu-pro-processed")
    
    return result


def verify_sorting(output_path: Path, num_samples: int = 5) -> None:
    """Verify sorting by printing sample entries."""
    print("\n=== Verification Samples ===")
    dataset = load_from_disk(str(output_path))
    
    # Handle DatasetDict
    if isinstance(dataset, DatasetDict):
        splits = dataset.keys()
    else:
        splits = ["main"]
        dataset = {"main": dataset}
        
    for split_name in splits:
        split_data = dataset[split_name]
        print(f"\n--- {split_name} ---")
        for i in range(min(num_samples, len(split_data))):
            entry = split_data[i]
            print(f"\nEntry {i}:")
            print(f"  Question: {entry['question'][:80]}...")
            print(f"  Gold: {entry.get('choices_answer', [])}")
            print(f"  Human distractors: {len(entry.get(DistractorType.COND_HUMAN_Q_A.value, []))}")
            print(f"  Model distractors: {len(entry.get(DistractorType.COND_MODEL_Q_A.value, []))}")


def main():
    parser = argparse.ArgumentParser(description="Process MMLU-Pro dataset")
    parser.add_argument("--input", type=str, default=str(RAW_DATASETS_DIR / "mmlu_pro"))
    parser.add_argument("--mmlu", type=str, default=str(RAW_DATASETS_DIR / "mmlu_all"))
    parser.add_argument("--output", type=str, default=str(PROCESSED_DATASETS_DIR / "mmlu_pro_processed"))
    parser.add_argument("--fix-whitespace", action="store_true", default=True)
    parser.add_argument("--report-whitespace", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--limit", type=int)
    
    args = parser.parse_args()
    
    result = process_mmlu_pro(
        mmlu_pro_path=Path(args.input),
        mmlu_path=Path(args.mmlu),
        output_path=Path(args.output),
        fix_whitespace=args.fix_whitespace,
        report_whitespace_bugs=args.report_whitespace,
        limit=args.limit,
    )
    
    if args.verify:
        verify_sorting(Path(args.output))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
