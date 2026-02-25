#!/usr/bin/env python3
"""Run a tiny live-API generation smoke test for Final5."""

from __future__ import annotations

import argparse
import subprocess
import sys
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
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--skip-failed-entries", action="store_true")
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_GENERATORS)
    parser.add_argument(
        "--request-log-dir",
        type=str,
        default="results/live_smoke_logs",
        help="Directory for per-model JSONL request logs",
    )
    parser.add_argument(
        "--slow-call-seconds",
        type=float,
        default=45.0,
        help="Slow-call threshold passed to generator",
    )
    parser.add_argument(
        "--concurrent-models",
        action="store_true",
        help="Launch one subprocess per model concurrently",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset_path = Path(args.processed_dataset)
    output_root = Path(args.output_root)
    request_log_dir = Path(args.request_log_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    request_log_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}")

    commands: list[tuple[str, list[str]]] = []
    for model in args.models:
        model_safe = model.replace("/", "_")
        out_path = output_root / f"{model_safe}_smoke"
        request_log = request_log_dir / f"{model_safe}.jsonl"

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
            "--save-interval",
            str(args.save_interval),
            "--max-retries",
            str(args.max_retries),
            "--retry-delay",
            str(args.retry_delay),
            "--skip-push",
            "--request-log",
            str(request_log),
            "--slow-call-seconds",
            str(args.slow_call_seconds),
        ]
        if args.skip_failed_entries:
            cmd.append("--skip-failed-entries")

        commands.append((model, cmd))

    for _, cmd in commands:
        print(" ".join(cmd))
    if args.dry_run:
        return 0

    if not args.concurrent_models:
        for model, cmd in commands:
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                print(f"[FAIL] {model} exited with code {proc.returncode}")
                return proc.returncode
        return 0

    running: list[tuple[str, subprocess.Popen[str]]] = []
    for model, cmd in commands:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        running.append((model, proc))

    failures: list[tuple[str, int]] = []
    for model, proc in running:
        assert proc.stdout is not None
        output = proc.stdout.read()
        rc = proc.wait()
        print(f"\n=== {model} (rc={rc}) ===")
        if output:
            sys.stdout.write(output)
        if rc != 0:
            failures.append((model, rc))

    if failures:
        print("\nFailures:")
        for model, rc in failures:
            print(f"  - {model}: rc={rc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
