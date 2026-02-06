"""
Dataset sorter for separating human vs synthetic distractors.

This module processes MMLU-Pro dataset entries to:
1. Identify which options came from original MMLU (human distractors)
2. Identify which options were added by MMLU-Pro (synthetic distractors)
3. Track the original MMLU subset each question came from
4. Fix leading whitespace issues in certain STEM categories
"""

import re
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict

from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
from tqdm import tqdm

from config import (
    DATASETS_DIR,
    DistractorType,
    RANDOM_SEED,
)


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


def find_mmlu_subset(
    question: str,
    mmlu_datasets: Dict[str, Dataset],
) -> Optional[str]:
    """
    Find which MMLU subset a question originated from.
    
    Args:
        question: The question text to search for
        mmlu_datasets: Dict mapping subset name to MMLU Dataset
        
    Returns:
        The subset name if found, None otherwise
    """
    question_clean = clean_whitespace(question.lower())
    
    for subset_name, subset_data in mmlu_datasets.items():
        for entry in subset_data:
            if clean_whitespace(entry["question"].lower()) == question_clean:
                return subset_name
    
    return None


def build_mmlu_lookup(mmlu_path: Path) -> Dict[str, Set[str]]:
    """
    Build a lookup table of all MMLU options indexed by question.
    
    Args:
        mmlu_path: Path to the downloaded MMLU dataset directory
        
    Returns:
        Dict mapping question -> set of options from MMLU
    """
    lookup = defaultdict(set)
    
    print("Building MMLU option lookup table...")
    
    # MMLU is organized by subject subdirectories
    if not mmlu_path.exists():
        raise FileNotFoundError(
            f"MMLU not found at {mmlu_path}. "
            "Run: python -m data.downloader --dataset mmlu"
        )
    
    # Handle both single dataset and multi-config structure
    for subset_dir in mmlu_path.iterdir():
        if not subset_dir.is_dir() or subset_dir.name.startswith("."):
            continue
            
        try:
            subset_data = load_from_disk(str(subset_dir))
        except Exception:
            continue
            
        # Process all splits
        for split_name, split_data in subset_data.items():
            for entry in split_data:
                question = clean_whitespace(entry["question"].lower())
                choices = entry.get("choices", [])
                for choice in choices:
                    lookup[question].add(clean_whitespace(str(choice).lower()))
    
    print(f"  Built lookup with {len(lookup)} questions")
    return lookup


def sort_distractors(
    entry: Dict,
    mmlu_lookup: Dict[str, Set[str]],
) -> Tuple[List[str], List[str], str]:
    """
    Sort distractors into human (from MMLU) vs synthetic (added by MMLU-Pro).
    
    Args:
        entry: A single MMLU-Pro dataset entry
        mmlu_lookup: Lookup table from build_mmlu_lookup()
        
    Returns:
        Tuple of (human_distractors, synthetic_distractors, gold_answer)
    """
    question = clean_whitespace(entry["question"].lower())
    mmlu_options = mmlu_lookup.get(question, set())
    
    # Get the correct answer
    answer_idx = entry.get("answer_index", 0)
    options = entry.get("options", [])
    
    if not options:
        return [], [], ""
    
    # Clean all options (fixes whitespace bug)
    options = clean_options(options)
    gold_answer = options[answer_idx] if answer_idx < len(options) else ""
    
    # Identify distractors (non-gold options)
    human_distractors = []
    synthetic_distractors = []
    
    for i, opt in enumerate(options):
        if i == answer_idx:
            continue  # Skip the gold answer
            
        opt_lower = opt.lower()
        if opt_lower in mmlu_options:
            human_distractors.append(opt)
        else:
            synthetic_distractors.append(opt)
    
    return human_distractors, synthetic_distractors, gold_answer


def process_mmlu_pro(
    mmlu_pro_path: Optional[Path] = None,
    mmlu_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    report_whitespace_bugs: bool = True,
    filter_10_options: bool = True,
) -> DatasetDict:
    """
    Process MMLU-Pro to separate human and synthetic distractors.
    
    IMPORTANT: By default, only 10-option questions are processed because:
    - 10 options = 1 gold + 3 human + 6 synthetic
    - Questions with fewer options have incomplete distractor sets
    
    Creates a new dataset with additional columns:
    - cond_human_q_a: List of human distractors (from MMLU, up to 3)
    - cond_model_q_a: List of synthetic distractors (added by MMLU-Pro, up to 6)
    - choices_answer: The gold answer (as a single-item list for consistency)
    - whitespace_bug_fixed: Whether whitespace was cleaned
    
    Args:
        mmlu_pro_path: Path to MMLU-Pro dataset
        mmlu_path: Path to MMLU dataset (for building lookup)
        output_path: Where to save the processed dataset
        report_whitespace_bugs: If True, report questions with whitespace issues
        filter_10_options: If True, only process questions with exactly 10 options
        
    Returns:
        Processed DatasetDict
    """
    # Set default paths
    if mmlu_pro_path is None:
        mmlu_pro_path = DATASETS_DIR / "mmlu_pro"
    if mmlu_path is None:
        mmlu_path = DATASETS_DIR / "mmlu_all"
    if output_path is None:
        output_path = DATASETS_DIR / "mmlu_pro_sorted"
    
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
        """Process a single entry."""
        # Check for whitespace issues
        had_whitespace_bug = False
        category = entry.get("category", "").lower()
        
        if category in WHITESPACE_BUG_CATEGORIES:
            options = entry.get("options", [])
            answer_idx = entry.get("answer_index", 0)
            if answer_idx < len(options):
                original_answer = options[answer_idx]
                if original_answer != original_answer.strip():
                    had_whitespace_bug = True
                    if report_whitespace_bugs:
                        whitespace_bugs.append({
                            "category": category,
                            "question": entry["question"][:100],
                            "original_answer": repr(original_answer),
                        })
        
        # Sort distractors
        human_distractors, synthetic_distractors, gold_answer = sort_distractors(
            entry, mmlu_lookup
        )
        
        # Use unified naming convention
        return {
            **entry,
            DistractorType.COND_HUMAN_Q_A.value: human_distractors,
            DistractorType.COND_MODEL_Q_A.value: synthetic_distractors,
            "choices_answer": [gold_answer] if gold_answer else [],
            "whitespace_bug_fixed": had_whitespace_bug,
        }
    
    # Process all splits
    processed = {}
    skipped_counts = {}
    
    for split_name, split_data in mmlu_pro.items():
        print(f"Processing {split_name} split ({len(split_data)} entries)...")
        processed_entries = []
        skipped = 0
        
        for entry in tqdm(split_data, desc=f"  {split_name}"):
            options = entry.get("options", [])
            
            # Filter to 10-option questions if requested
            if filter_10_options and len(options) != 10:
                skipped += 1
                continue
            
            processed_entries.append(process_entry(dict(entry)))
        
        processed[split_name] = Dataset.from_list(processed_entries)
        skipped_counts[split_name] = skipped
        
        if filter_10_options and skipped > 0:
            print(f"    Filtered: kept {len(processed_entries)}, skipped {skipped} (non-10-option)")
    
    result = DatasetDict(processed)
    
    # Report whitespace bugs
    if report_whitespace_bugs and whitespace_bugs:
        print(f"\n⚠️  Found {len(whitespace_bugs)} entries with whitespace bugs:")
        by_category = defaultdict(int)
        for bug in whitespace_bugs:
            by_category[bug["category"]] += 1
        for cat, count in sorted(by_category.items()):
            print(f"    {cat}: {count} entries")
    
    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    result.save_to_disk(str(output_path))
    print(f"\n✓ Saved processed dataset to {output_path}")
    
    # Print summary
    for split_name, split_data in result.items():
        sample = split_data[0]
        human_count = len(sample.get(DistractorType.COND_HUMAN_Q_A.value, []))
        model_count = len(sample.get(DistractorType.COND_MODEL_Q_A.value, []))
        print(f"\n  {split_name}: {len(split_data)} entries")
        print(f"    Sample human distractors: {human_count}")
        print(f"    Sample model distractors: {model_count}")
    
    return result


def verify_sorting(dataset: DatasetDict, num_samples: int = 5) -> None:
    """
    Verify the sorting by printing sample entries.
    """
    print("\n=== Verification Samples ===")
    
    for split_name, split_data in dataset.items():
        print(f"\n--- {split_name} ---")
        
        for i in range(min(num_samples, len(split_data))):
            entry = split_data[i]
            print(f"\nEntry {i}:")
            print(f"  Question: {entry['question'][:80]}...")
            print(f"  Gold: {entry.get('choices_answer', [])}")
            print(f"  Human distractors: {entry.get(DistractorType.COND_HUMAN_Q_A.value, [])}")
            print(f"  Model distractors: {entry.get(DistractorType.COND_MODEL_Q_A.value, [])}")
            print(f"  Whitespace fixed: {entry.get('whitespace_bug_fixed', False)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sort MMLU-Pro distractors")
    parser.add_argument(
        "--mmlu-pro-path",
        type=str,
        default=None,
        help="Path to MMLU-Pro dataset",
    )
    parser.add_argument(
        "--mmlu-path",
        type=str,
        default=None,
        help="Path to MMLU dataset",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Where to save processed dataset",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Print verification samples after processing",
    )
    args = parser.parse_args()
    
    result = process_mmlu_pro(
        mmlu_pro_path=Path(args.mmlu_pro_path) if args.mmlu_pro_path else None,
        mmlu_path=Path(args.mmlu_path) if args.mmlu_path else None,
        output_path=Path(args.output_path) if args.output_path else None,
    )
    
    if args.verify:
        verify_sorting(result)
