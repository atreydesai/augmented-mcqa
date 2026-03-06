#!/usr/bin/env python3
"""Export Final5 augmented datasets into benchmarker JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import export_benchmarker_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export benchmarker JSONL files from a Final5 dataset")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the augmented Final5 DatasetDict on disk",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="datasets/benchmarker_items",
        help="Root directory that will contain the exported dataset folder",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = export_benchmarker_items(args.input, args.output_root)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    print(f"Exported benchmarker items to {summary['output_directory']}")
    for split_name in summary["splits"]:
        print(f"{split_name}:")
        for variant_name, meta in summary["files"][split_name].items():
            print(
                f"  {variant_name}: rows_written={meta['rows_written']} "
                f"skipped={meta['skipped_row_count']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
