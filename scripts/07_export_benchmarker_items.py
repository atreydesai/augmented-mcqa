#!/usr/bin/env python3
"""Export Final5 augmented datasets into benchmarker JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import main as app_main


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
    return app_main.main(["export", "--input", args.input, "--output-root", args.output_root])


if __name__ == "__main__":
    raise SystemExit(main())
