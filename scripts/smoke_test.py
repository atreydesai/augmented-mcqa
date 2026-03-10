#!/usr/bin/env python3
"""Inspect-first smoke tests for generation and evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.constants import (  # noqa: E402
    DEFAULT_AUGMENTED_CACHE_ROOT,
    DEFAULT_EVALUATION_LOG_ROOT,
    DEFAULT_EVALUATION_MODELS,
    DEFAULT_GENERATION_LOG_ROOT,
    DEFAULT_GENERATION_MODELS,
    DEFAULT_PROCESSED_DATASET,
)


@dataclass
class RunResult:
    name: str
    command: list[str]
    returncode: int
    output: str


def _run_commands(
    commands: list[tuple[str, list[str]]],
    *,
    dry_run: bool,
    concurrent: bool,
) -> int:
    for _, command in commands:
        print(" ".join(command))
    if dry_run:
        return 0

    if not concurrent:
        for name, command in commands:
            proc = subprocess.run(command, capture_output=True, text=True)
            if proc.stdout:
                sys.stdout.write(proc.stdout)
            if proc.stderr:
                sys.stderr.write(proc.stderr)
            if proc.returncode != 0:
                print(f"[FAIL] {name}: rc={proc.returncode}")
                return proc.returncode
        return 0

    running: list[tuple[str, subprocess.Popen[str]]] = []
    for name, command in commands:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        running.append((name, proc))

    failures: list[RunResult] = []
    for name, proc in running:
        assert proc.stdout is not None
        output = proc.stdout.read()
        rc = proc.wait()
        print(f"\n=== {name} (rc={rc}) ===")
        if output:
            sys.stdout.write(output)
        if rc != 0:
            failures.append(RunResult(name=name, command=[], returncode=rc, output=output))

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure.name}: rc={failure.returncode}")
        return 1
    return 0


def _append_runtime_flags(command: list[str], args: argparse.Namespace) -> None:
    if args.backend:
        command.extend(["--backend", args.backend])
    if args.model_base_url:
        command.extend(["--model-base-url", args.model_base_url])
    if args.reasoning_effort:
        command.extend(["--reasoning-effort", args.reasoning_effort])
    if args.max_tokens is not None:
        command.extend(["--max-tokens", str(args.max_tokens)])


def cmd_generate(args: argparse.Namespace) -> int:
    dataset_path = Path(args.processed_dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}")

    commands: list[tuple[str, list[str]]] = []
    for model in args.models:
        command = [
            "uv",
            "run",
            "python",
            "main.py",
            "generate",
            "--model",
            model,
            "--run-name",
            args.run_name,
            "--processed-dataset",
            str(dataset_path),
            "--limit",
            str(args.limit),
            "--log-root",
            args.log_root,
            "--cache-root",
            args.cache_root,
            "--materialize-cache",
        ]
        if args.dataset_types:
            command.extend(["--dataset-types", args.dataset_types])
        _append_runtime_flags(command, args)
        commands.append((model, command))
    return _run_commands(commands, dry_run=args.dry_run, concurrent=args.concurrent_models)


def cmd_evaluate(args: argparse.Namespace) -> int:
    commands: list[tuple[str, list[str]]] = []
    for model in args.models:
        command = [
            "uv",
            "run",
            "python",
            "main.py",
            "evaluate",
            "--model",
            model,
            "--run-name",
            args.run_name,
            "--generator-run-name",
            args.generator_run_name,
            "--generator-model",
            args.generator_model,
            "--processed-dataset",
            args.processed_dataset,
            "--generation-log-root",
            args.generation_log_root,
            "--cache-root",
            args.cache_root,
            "--log-root",
            args.log_root,
            "--limit",
            str(args.limit),
        ]
        if args.settings:
            command.extend(["--settings", args.settings])
        if args.modes:
            command.extend(["--modes", args.modes])
        if args.dataset_types:
            command.extend(["--dataset-types", args.dataset_types])
        if args.generator_backend:
            command.extend(["--generator-backend", args.generator_backend])
        _append_runtime_flags(command, args)
        commands.append((model, command))
    return _run_commands(commands, dry_run=args.dry_run, concurrent=args.concurrent_models)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect-first smoke tests")
    subparsers = parser.add_subparsers(dest="command")

    generate = subparsers.add_parser("generate", help="Run small Inspect generation jobs")
    generate.add_argument("--models", nargs="+", default=list(DEFAULT_GENERATION_MODELS))
    generate.add_argument("--run-name", default="smoke-generate")
    generate.add_argument("--processed-dataset", default=str(DEFAULT_PROCESSED_DATASET))
    generate.add_argument("--dataset-types", default=None)
    generate.add_argument("--limit", type=int, default=2)
    generate.add_argument("--log-root", default=str(DEFAULT_GENERATION_LOG_ROOT))
    generate.add_argument("--cache-root", default=str(DEFAULT_AUGMENTED_CACHE_ROOT))
    generate.add_argument("--backend", default=None)
    generate.add_argument("--model-base-url", default=None)
    generate.add_argument("--reasoning-effort", default=None)
    generate.add_argument("--max-tokens", type=int, default=256)
    generate.add_argument("--dry-run", action="store_true")
    generate.add_argument("--concurrent-models", action="store_true")
    generate.set_defaults(handler=cmd_generate)

    evaluate = subparsers.add_parser("evaluate", help="Run small Inspect evaluation jobs")
    evaluate.add_argument("--models", nargs="+", default=list(DEFAULT_EVALUATION_MODELS))
    evaluate.add_argument("--run-name", default="smoke-evaluate")
    evaluate.add_argument("--generator-run-name", required=True)
    evaluate.add_argument("--generator-model", required=True)
    evaluate.add_argument("--generator-backend", default=None)
    evaluate.add_argument("--processed-dataset", default=str(DEFAULT_PROCESSED_DATASET))
    evaluate.add_argument("--dataset-types", default=None)
    evaluate.add_argument("--settings", default=None)
    evaluate.add_argument("--modes", default=None)
    evaluate.add_argument("--limit", type=int, default=2)
    evaluate.add_argument("--generation-log-root", default=str(DEFAULT_GENERATION_LOG_ROOT))
    evaluate.add_argument("--cache-root", default=str(DEFAULT_AUGMENTED_CACHE_ROOT))
    evaluate.add_argument("--log-root", default=str(DEFAULT_EVALUATION_LOG_ROOT))
    evaluate.add_argument("--backend", default=None)
    evaluate.add_argument("--model-base-url", default=None)
    evaluate.add_argument("--reasoning-effort", default=None)
    evaluate.add_argument("--max-tokens", type=int, default=128)
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.add_argument("--concurrent-models", action="store_true")
    evaluate.set_defaults(handler=cmd_evaluate)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
