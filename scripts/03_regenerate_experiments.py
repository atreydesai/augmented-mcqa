#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import main as app_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for Inspect-first generate-all")
    parser.add_argument("--processed-dataset", default="datasets/processed/unified_processed_v2")
    parser.add_argument("--output-root", default="datasets/augmented")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--models", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    argv = [
        "generate-all",
        "--run-name",
        args.run_name,
        "--processed-dataset",
        args.processed_dataset,
        "--cache-root",
        args.output_root,
        "--materialize-cache",
    ]
    if args.models:
        argv.extend(["--models", args.models])
    if args.limit is not None:
        argv.extend(["--limit", str(args.limit)])
    return app_main.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
