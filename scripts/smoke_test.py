#!/usr/bin/env python3
"""Live API smoke tests for structured outputs and generation.

Subcommands:
    clients   - Quick structured-output test across all providers
    generate  - End-to-end generation smoke with real datasets

Usage:
    python scripts/smoke_test.py clients --models gpt-5.2-2025-12-11 claude-opus-4-6
    python scripts/smoke_test.py generate --limit 2 --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# clients subcommand (from former test_structured_outputs_clients.py)
# ---------------------------------------------------------------------------

DEFAULT_MODELS = [
    "gpt-5.2-2025-12-11",
    "claude-opus-4-6",
    "gemini-3.1-pro-preview",
]


@dataclass
class RunResult:
    model: str
    provider: str
    ok: bool
    elapsed_s: float
    error: str | None = None
    raw_output: str = ""


def _schema(count: int, provider: str) -> dict[str, Any]:
    provider_key = provider.lower().strip()
    array_schema: dict[str, Any] = {
        "type": "array",
        "items": {"type": "string"},
    }
    if provider_key == "anthropic":
        array_schema["minItems"] = 1
    else:
        array_schema["minItems"] = count
        array_schema["maxItems"] = count

    return {
        "type": "object",
        "properties": {
            "distractors": array_schema
        },
        "required": ["distractors"],
        "additionalProperties": False,
    }


def _prompt(count: int) -> str:
    return (
        "I have a multiple-choice question with the single correct answer, and I need to "
        "expand it with additional multiple-choice options. Please generate "
        f"{count} additional plausible but incorrect options (B, C, D) to accompany the correct answer choice.\n\n"
        "Input:\n\n"
        "Question: What is the capital of France?\n\n"
        "Answer: A: Paris\n\n"
        "Please generate only the new incorrect options. Output each option on a separate line "
        'in the format "<LETTER>: <option>".'
    )


def _extract_distractors(raw_text: str, expected_count: int) -> list[str]:
    from data.augmentor import parse_generated_distractors

    candidates = [raw_text.strip()]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", raw_text, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(block.strip() for block in fenced if block.strip())
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("distractors"), list):
            values = [str(x).strip() for x in parsed["distractors"] if str(x).strip()]
            if len(values) == expected_count:
                return values
        if isinstance(parsed, list):
            values = [str(x).strip() for x in parsed if str(x).strip()]
            if len(values) == expected_count:
                return values
    return parse_generated_distractors(raw_text, expected_count=expected_count, start_letter="B")


def _run_single(model: str, count: int, max_tokens: int) -> RunResult:
    from models import get_client, resolve_model

    provider, _, _ = resolve_model(model)
    provider = provider.lower().strip()

    client_kwargs: dict[str, Any] = {}
    request_kwargs: dict[str, Any] = {}

    if model == "gpt-5.2-2025-12-11":
        client_kwargs["reasoning_effort"] = "medium"
    if provider == "anthropic":
        request_kwargs["thinking"] = {"type": "adaptive"}
        request_kwargs["output_config"] = {
            "format": {
                "type": "json_schema",
                "schema": _schema(count, provider=provider),
            }
        }
    else:
        request_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": f"quick_distractors_{count}",
                "strict": True,
                "schema": _schema(count, provider=provider),
            },
        }

    started = time.time()
    try:
        client = get_client(model, **client_kwargs)
        out = client.generate(
            prompt=_prompt(count),
            max_tokens=max_tokens,
            **request_kwargs,
        )
        text = (out.text or "").strip()
        _extract_distractors(text, expected_count=count)
        return RunResult(
            model=model,
            provider=provider,
            ok=True,
            elapsed_s=round(time.time() - started, 3),
            raw_output=text,
        )
    except Exception as exc:  # noqa: BLE001
        return RunResult(
            model=model,
            provider=provider,
            ok=False,
            elapsed_s=round(time.time() - started, 3),
            error=f"{type(exc).__name__}: {exc}",
        )


def cmd_clients(args: argparse.Namespace) -> int:
    results: list[RunResult] = []
    for model in args.models:
        results.append(_run_single(model=model, count=args.count, max_tokens=args.max_tokens))

    failures = [x for x in results if not x.ok]
    for row in results:
        status = "OK" if row.ok else "FAIL"
        print(f"[{status}] model={row.model} provider={row.provider} elapsed_s={row.elapsed_s}")
        if row.ok:
            preview = row.raw_output.replace("\n", " ")[:220]
            print(f"  output={preview}")
        else:
            preview = (row.raw_output or "").replace("\n", " ")[:220]
            print(f"  error={row.error}")
            print(f"  output={preview}")

    return 1 if failures else 0


# ---------------------------------------------------------------------------
# generate subcommand (from former live_api_smoke.py)
# ---------------------------------------------------------------------------

DEFAULT_GENERATORS = [
    "gpt-5.2-2025-12-11",
    "claude-opus-4-6",
    "gemini-3.1-pro-preview",
]


def cmd_generate(args: argparse.Namespace) -> int:
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
            "scripts/02_generate_distractors.py",
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live API smoke tests for structured outputs and generation"
    )
    subparsers = parser.add_subparsers(dest="command")

    # clients subcommand
    cl = subparsers.add_parser("clients", help="Quick structured-output test across all providers")
    cl.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    cl.add_argument("--count", type=int, default=3)
    cl.add_argument("--max-tokens", type=int, default=2048)
    cl.set_defaults(handler=cmd_clients)

    # generate subcommand
    gen = subparsers.add_parser("generate", help="End-to-end generation smoke with real datasets")
    gen.add_argument(
        "--processed-dataset",
        type=str,
        default="datasets/processed/unified_processed_v2",
        help="Input processed dataset containing arc_challenge/mmlu_pro/gpqa splits",
    )
    gen.add_argument("--output-root", type=str, default="datasets/augmented/smoke")
    gen.add_argument("--limit", type=int, default=2, help="Rows per split to generate")
    gen.add_argument("--save-interval", type=int, default=25)
    gen.add_argument("--max-retries", type=int, default=3)
    gen.add_argument("--retry-delay", type=float, default=1.0)
    gen.add_argument("--skip-failed-entries", action="store_true")
    gen.add_argument("--models", type=str, nargs="+", default=DEFAULT_GENERATORS)
    gen.add_argument(
        "--request-log-dir",
        type=str,
        default="results/live_smoke_logs",
    )
    gen.add_argument("--slow-call-seconds", type=float, default=45.0)
    gen.add_argument("--concurrent-models", action="store_true")
    gen.add_argument("--dry-run", action="store_true")
    gen.set_defaults(handler=cmd_generate)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
