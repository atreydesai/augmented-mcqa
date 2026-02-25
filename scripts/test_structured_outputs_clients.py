#!/usr/bin/env python3
"""Quick live test for structured outputs across all generation providers."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.augmentor import parse_generated_distractors
from models import get_client, resolve_model


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
            raw_output=(locals().get("out").text if "out" in locals() and getattr(out, "text", None) else ""),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structured-output live client smoke")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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


if __name__ == "__main__":
    raise SystemExit(main())
