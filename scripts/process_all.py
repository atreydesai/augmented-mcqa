#!/usr/bin/env python
"""
Process all downloaded datasets into formats ready for experiments.

Usage:
    python scripts/process_all.py
    python scripts/process_all.py --dataset mmlu_pro
    python scripts/process_all.py --dataset arc
    python scripts/process_all.py --dataset supergpqa
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATASETS_DIR


def process_mmlu_pro():
    """Sort MMLU-Pro into human vs synthetic distractors."""
    print("\n" + "=" * 60)
    print("Processing MMLU-Pro (sorting human vs synthetic distractors)")
    print("=" * 60)
    
    from data.sorter import process_mmlu_pro as _process
    
    output_path = DATASETS_DIR / "mmlu_pro_sorted"
    result = _process(output_path=output_path)
    
    print(f"✅ MMLU-Pro processed -> {output_path}")
    return result


def process_arc():
    """Process ARC-Easy and ARC-Challenge into unified format."""
    print("\n" + "=" * 60)
    print("Processing ARC (Easy + Challenge)")
    print("=" * 60)
    
    from data.arc_processor import process_arc_for_experiments
    
    arc_easy = process_arc_for_experiments(
        difficulty="easy",
        output_path=DATASETS_DIR / "arc_processed" / "arc_easy.json"
    )
    print(f"  ARC-Easy: {len(arc_easy)} entries")
    
    arc_challenge = process_arc_for_experiments(
        difficulty="challenge",
        output_path=DATASETS_DIR / "arc_processed" / "arc_challenge.json"
    )
    print(f"  ARC-Challenge: {len(arc_challenge)} entries")
    
    print(f"✅ ARC processed -> {DATASETS_DIR / 'arc_processed'}")
    return {"easy": arc_easy, "challenge": arc_challenge}


def process_supergpqa():
    """Process SuperGPQA into unified format (10-option questions only)."""
    print("\n" + "=" * 60)
    print("Processing SuperGPQA (filtering to 10-option questions)")
    print("=" * 60)
    
    from data.supergpqa_processor import process_supergpqa_for_experiments
    
    result = process_supergpqa_for_experiments(
        output_path=DATASETS_DIR / "supergpqa_processed" / "supergpqa.json"
    )
    
    print(f"  SuperGPQA: {len(result)} entries (10-option only)")
    print(f"✅ SuperGPQA processed -> {DATASETS_DIR / 'supergpqa_processed'}")
    return result


def process_all():
    """Process all datasets."""
    results = {}
    
    results["mmlu_pro"] = process_mmlu_pro()
    results["arc"] = process_arc()
    results["supergpqa"] = process_supergpqa()
    
    print("\n" + "=" * 60)
    print("All datasets processed!")
    print("=" * 60)
    print(f"  datasets/mmlu_pro_sorted/")
    print(f"  datasets/arc_processed/")
    print(f"  datasets/supergpqa_processed/")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Process all datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mmlu_pro", "arc", "supergpqa", "all"],
        default="all",
        help="Which dataset to process (default: all)"
    )
    args = parser.parse_args()
    
    if args.dataset == "all":
        process_all()
    elif args.dataset == "mmlu_pro":
        process_mmlu_pro()
    elif args.dataset == "arc":
        process_arc()
    elif args.dataset == "supergpqa":
        process_supergpqa()


if __name__ == "__main__":
    main()
