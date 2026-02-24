#!/usr/bin/env python3
"""Run a tiny live-API generation smoke test for Final5."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_GENERATORS = [
    "gpt-5.2-2025-12-11",
    "claude-opus-4-6",
    "gemini-3.1-pro-preview",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live API smoke test for Final5 generation")
    parser.add_argument(
        "--processed-dataset",
        type=str,
        default="datasets/processed/unified_processed_v2",
        help="Input processed dataset containing arc_challenge/mmlu_pro/gpqa splits",
    )
    parser.add_argument("--output-root", type=str, default="datasets/augmented/smoke")
    parser.add_argument("--limit", type=int, default=2, help="Rows per split to generate")
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_GENERATORS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset_path = Path(args.processed_dataset)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}")

    for model in args.models:
        model_safe = model.replace("/", "_")
        out_path = output_root / f"{model_safe}_smoke"

        cmd = [
            "uv",
            "run",
            "python",
            "scripts/generate_distractors.py",
            "--input",
            str(dataset_path),
            "--output",
            str(out_path),
            "--model",
            model,
            "--limit",
            str(args.limit),
            "--skip-push",
        ]

        print(" ".join(cmd))
        if args.dry_run:
            continue

        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            return proc.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
