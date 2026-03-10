#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import main as app_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for Inspect-first evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--preset", default="final5")
    plan.add_argument("--model", required=False)
    plan.add_argument("--dataset-path", required=False)
    plan.add_argument("--generator-dataset-label", required=False)

    run = sub.add_parser("run")
    run.add_argument("--model", default=None)
    run.add_argument("--generator-run-name", default=None)
    run.add_argument("--generator-model", default=None)
    run.add_argument("--generation-log-dir", default=None)
    run.add_argument("--processed-dataset", default="datasets/processed/unified_processed_v2")
    run.add_argument("--augmented-dataset", default=None)
    run.add_argument("--settings", default=None)
    run.add_argument("--modes", default=None)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--run-name", default="compat-eval")
    run.add_argument("--shard-count", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-strategy", choices=["contiguous", "modulo"], default="contiguous")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "plan":
        print("Planning manifests is no longer required. Use scripts/05_build_eval_slurm_bundle.py or main.py evaluate/evaluate-all.")
        return 0

    if not args.model or not args.generator_run_name or not args.generator_model:
        raise ValueError("--model, --generator-run-name, and --generator-model are required for run")
    forwarded = [
        "evaluate",
        "--model",
        args.model,
        "--run-name",
        args.run_name,
        "--generator-run-name",
        args.generator_run_name,
        "--generator-model",
        args.generator_model,
        "--processed-dataset",
        args.processed_dataset,
        "--shard-count",
        str(args.shard_count),
        "--shard-index",
        str(args.shard_index),
        "--shard-strategy",
        args.shard_strategy,
    ]
    if args.generation_log_dir:
        forwarded.extend(["--generation-log-dir", args.generation_log_dir])
    if args.augmented_dataset:
        forwarded.extend(["--augmented-dataset", args.augmented_dataset])
    if args.settings:
        forwarded.extend(["--settings", args.settings])
    if args.modes:
        forwarded.extend(["--modes", args.modes])
    if args.limit is not None:
        forwarded.extend(["--limit", str(args.limit)])
    return app_main.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
