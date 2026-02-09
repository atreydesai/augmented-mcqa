#!/usr/bin/env python3
"""
Process MMLU-Pro dataset: sort distractors and fix whitespace.

Usage:
    python data/mmlu_pro_processor.py --input datasets/mmlu_pro \
        --mmlu datasets/mmlu --output datasets/mmlu_pro_processed
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import process_mmlu_pro, verify_sorting
from config import DATASETS_DIR


def main():
    parser = argparse.ArgumentParser(description="Process MMLU-Pro dataset")
    
    parser.add_argument(
        "--input",
        type=str,
        default=str(DATASETS_DIR / "mmlu_pro"),
        help="Input MMLU-Pro dataset path",
    )
    parser.add_argument(
        "--mmlu",
        type=str,
        default=str(DATASETS_DIR / "mmlu"),
        help="MMLU dataset path for comparison",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATASETS_DIR / "mmlu_pro_processed"),
        help="Output path for processed dataset",
    )
    parser.add_argument(
        "--fix-whitespace",
        action="store_true",
        default=True,
        help="Fix leading whitespace in STEM categories",
    )
    parser.add_argument(
        "--report-whitespace",
        action="store_true",
        help="Report whitespace issues",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify sorting after processing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit entries (for testing)",
    )
    
    args = parser.parse_args()
    
    print("Processing MMLU-Pro dataset...")
    print(f"  Input: {args.input}")
    print(f"  MMLU: {args.mmlu}")
    print(f"  Output: {args.output}")
    
    result = process_mmlu_pro(
        mmlu_pro_path=Path(args.input),
        mmlu_path=Path(args.mmlu),
        output_path=Path(args.output),
        fix_whitespace=args.fix_whitespace,
        report_whitespace_bugs=args.report_whitespace,
        limit=args.limit,
    )
    
    print(f"\n✓ Processed {len(result)} entries")
    
    if args.verify:
        print("\nVerifying sorting...")
        verify_sorting(Path(args.output))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
