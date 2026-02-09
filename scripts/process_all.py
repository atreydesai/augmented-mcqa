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
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATASETS_DIR, PROCESSED_DATASETS_DIR
from data import (
    process_mmlu_pro as process_mmlu_pro_func,
    process_arc_for_experiments,
    process_supergpqa_for_experiments,
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
        output_path=PROCESSED_DATASETS_DIR / "arc_processed" / "arc_easy.json",
        limit=limit
    )
    print(f"  ARC-Easy: {len(arc_easy)} entries")
    
    arc_challenge = process_arc_for_experiments(
        difficulty="challenge",
        output_path=PROCESSED_DATASETS_DIR / "arc_processed" / "arc_challenge.json",
        limit=limit
    )
    print(f"  ARC-Challenge: {len(arc_challenge)} entries")
    
    print(f"✅ ARC processed -> {PROCESSED_DATASETS_DIR / 'arc_processed'}")
    return {"easy": arc_easy, "challenge": arc_challenge}


def run_supergpqa(limit: Optional[int] = None):
    """Process SuperGPQA into unified format (10-option questions only)."""
    print("\n" + "=" * 60)
    print("Processing SuperGPQA (filtering to 10-option questions)")
    print("=" * 60)
    
    result = process_supergpqa_for_experiments(
        output_path=PROCESSED_DATASETS_DIR / "supergpqa_processed" / "supergpqa.json",
        limit=limit
    )
    
    print(f"  SuperGPQA: {len(result)} entries (10-option only)")
    print(f"✅ SuperGPQA processed -> {PROCESSED_DATASETS_DIR / 'supergpqa_processed'}")
    return result


def run_all(limit: Optional[int] = None):
    """Process all datasets."""
    results = {}
    
    results["mmlu_pro"] = run_mmlu_pro(limit=limit)
    results["arc"] = run_arc(limit=limit)
    results["supergpqa"] = run_supergpqa(limit=limit)
    
    print("\n" + "=" * 60)
    print("All datasets processed!")
    print("=" * 60)
    print(f"  datasets/processed/mmlu_pro_processed/")
    print(f"  datasets/processed/arc_processed/")
    print(f"  datasets/processed/supergpqa_processed/")
    
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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit entries per dataset (for testing)"
    )
    args = parser.parse_args()
    
    if args.dataset == "all":
        run_all(limit=args.limit)
    elif args.dataset == "mmlu_pro":
        run_mmlu_pro(limit=args.limit)
    elif args.dataset == "arc":
        run_arc(limit=args.limit)
    elif args.dataset == "supergpqa":
        run_supergpqa(limit=args.limit)


if __name__ == "__main__":
    main()
