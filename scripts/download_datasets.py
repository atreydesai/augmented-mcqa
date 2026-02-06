#!/usr/bin/env python3
"""
Download and process datasets.

Usage:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --dataset mmlu_pro
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import (
    download_mmlu_pro,
    download_mmlu_all_configs,
    download_arc,
    download_supergpqa,
)
from config import DATASETS_DIR


def main():
    parser = argparse.ArgumentParser(description="Download datasets")
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mmlu_pro", "mmlu", "arc", "supergpqa"],
        help="Download a specific dataset",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATASETS_DIR),
        help="Output directory for datasets",
    )
    
    args = parser.parse_args()
    
    if not args.all and not args.dataset:
        parser.error("Must specify --dataset or --all")
    
    output_dir = Path(args.output_dir)
    
    datasets_to_download = []
    if args.all:
        datasets_to_download = ["mmlu_pro", "mmlu", "arc", "supergpqa"]
    else:
        datasets_to_download = [args.dataset]
    
    for ds in datasets_to_download:
        print(f"\n{'='*50}")
        print(f"Downloading: {ds}")
        print('='*50)
        
        try:
            if ds == "mmlu_pro":
                download_mmlu_pro(output_dir / "mmlu_pro")
            elif ds == "mmlu":
                download_mmlu_all_configs(output_dir / "mmlu")
            elif ds == "arc":
                download_arc(output_dir / "arc")
            elif ds == "supergpqa":
                download_supergpqa(output_dir / "supergpqa")
            
            print(f"✓ Downloaded: {ds}")
            
        except Exception as e:
            print(f"✗ Failed to download {ds}: {e}")
    
    print(f"\nDatasets saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
