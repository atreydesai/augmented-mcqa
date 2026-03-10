#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import main as app_main
from utils.constants import DEFAULT_GENERATION_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for Inspect-first generation")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--input", "-i", type=str, help="Processed dataset path")
    parser.add_argument("--output", "-o", type=str, help="Optional augmented dataset cache output")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_GENERATION_MODELS[0])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="compat-generate")
    parser.add_argument("--reasoning-effort", type=str, default=None)
    parser.add_argument("--model-base-url", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_models:
        for model in DEFAULT_GENERATION_MODELS:
            print(model)
        return 0
    if not args.input:
        print("--input is required")
        return 1

    argv = [
        "generate",
        "--model",
        args.model,
        "--run-name",
        args.run_name,
        "--processed-dataset",
        args.input,
        "--materialize-cache",
    ]
    if args.output:
        argv.extend(["--augmented-dataset", args.output])
    if args.limit is not None:
        argv.extend(["--limit", str(args.limit)])
    if args.reasoning_effort:
        argv.extend(["--reasoning-effort", args.reasoning_effort])
    if args.model_base_url:
        argv.extend(["--model-base-url", args.model_base_url])
    if args.backend:
        argv.extend(["--backend", args.backend])
    return app_main.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
