#!/usr/bin/env python3
"""Diagnostic tools for debugging generation issues.

Subcommands:
    failures  - Report rows with missing required distractor columns
    trace     - Run structured-generation trace and save per-step JSON

Usage:
    python scripts/diagnose.py failures --dataset-path datasets/augmented/...
    python scripts/diagnose.py trace --model gemini-3.1-pro-preview --dataset-path datasets/processed/unified_processed_v2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# failures subcommand (from former diagnose_failures.py)
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    ("model_from_scratch", 3),
    ("augment_human", 6),
    ("augment_model", 9),
    ("augment_ablation", 9),
]


def cmd_failures(args: argparse.Namespace) -> int:
    from datasets import load_from_disk

    ds = load_from_disk(str(Path(args.dataset_path)))

    total_failed = 0
    for split in ds.keys():
        failed = []
        for i, row in enumerate(ds[split]):
            missing = [k for k, n in REQUIRED_COLUMNS if len(row.get(k) or []) < n]
            if missing:
                failed.append({
                    "idx": i,
                    "id": row.get("id"),
                    "question_id": row.get("question_id"),
                    "missing": missing,
                    "question": row.get("question", "")[:140],
                })

        print(f"\n{split}: failed_rows={len(failed)}")
        for r in failed:
            print(r)
        total_failed += len(failed)

    return 1 if total_failed > 0 else 0


# ---------------------------------------------------------------------------
# trace subcommand (from former diagnose_structured_generation_trace.py)
# ---------------------------------------------------------------------------

DEFAULT_SPLITS = ["arc_challenge", "mmlu_pro", "gpqa"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _default_output_path(model: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_safe = model.replace("/", "_")
    out_dir = Path("results/live_smoke_logs") / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{model_safe}_step_trace.json"


def _pick_rows(ds, splits: list[str], per_split: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for split in splits:
        if split not in ds:
            continue
        split_ds = ds[split]
        take = min(per_split, len(split_ds))
        for idx in range(take):
            row = split_ds[idx]
            answer = _safe_text(row.get("answer"))
            if not answer:
                choices_answer = row.get("choices_answer") or []
                answer = _safe_text(choices_answer[0]) if choices_answer else ""

            human = [_safe_text(x) for x in (row.get("choices_human") or []) if _safe_text(x)]
            selected.append(
                {
                    "split": split,
                    "entry_index": idx,
                    "question": _safe_text(row.get("question")),
                    "answer": answer,
                    "human3": human[:3],
                }
            )
    return selected


def _client_policy_kwargs(model: str, provider: str) -> tuple[dict[str, Any], dict[str, Any]]:
    client_kwargs: dict[str, Any] = {}
    request_common_kwargs: dict[str, Any] = {}

    if model == "gpt-5.2-2025-12-11":
        client_kwargs["reasoning_effort"] = "medium"
    if model == "claude-opus-4-6" or provider == "anthropic":
        request_common_kwargs["thinking"] = {"type": "adaptive"}

    return client_kwargs, request_common_kwargs


def _run_step(
    *,
    client,
    provider: str,
    prompt: str,
    context: str,
    expected_count: int,
    start_letter: str,
    max_tokens: int,
    max_attempts: int,
    request_common_kwargs: dict[str, Any],
) -> dict[str, Any]:
    from data.augmentor import _structured_request_kwargs, parse_generated_distractors

    step: dict[str, Any] = {
        "context": context,
        "expected_count": expected_count,
        "start_letter": start_letter,
        "success": False,
        "parsed": None,
        "attempts": [],
    }

    structured_kwargs = _structured_request_kwargs(provider, expected_count)

    for attempt in range(1, max_attempts + 1):
        req_kwargs = dict(request_common_kwargs)
        req_kwargs.update(structured_kwargs)

        attempt_rec: dict[str, Any] = {
            "attempt": attempt,
            "started_utc": _now_iso(),
            "prompt": prompt,
            "request_kwargs": req_kwargs,
        }

        started = time.time()
        try:
            out = client.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                **req_kwargs,
            )
            raw_output = _safe_text(out.text)
            attempt_rec["elapsed_s"] = round(time.time() - started, 3)
            attempt_rec["finish_reason"] = out.finish_reason
            attempt_rec["usage"] = out.usage
            attempt_rec["raw_output"] = raw_output

            try:
                parsed = parse_generated_distractors(
                    raw_output,
                    expected_count=expected_count,
                    start_letter=start_letter,
                )
                attempt_rec["parse_ok"] = True
                attempt_rec["parse_error"] = None
                attempt_rec["parsed"] = parsed
                step["success"] = True
                step["parsed"] = parsed
                step["attempts"].append(attempt_rec)
                break
            except Exception as parse_exc:  # noqa: BLE001
                attempt_rec["parse_ok"] = False
                attempt_rec["parse_error"] = f"{type(parse_exc).__name__}: {parse_exc}"
                attempt_rec["parsed"] = None
                step["attempts"].append(attempt_rec)
        except Exception as req_exc:  # noqa: BLE001
            attempt_rec["elapsed_s"] = round(time.time() - started, 3)
            attempt_rec["request_error"] = f"{type(req_exc).__name__}: {req_exc}"
            attempt_rec["raw_output"] = ""
            step["attempts"].append(attempt_rec)

    return step


def cmd_trace(args: argparse.Namespace) -> int:
    from datasets import DatasetDict, load_from_disk

    from data.augmentor import _build_conditioned_prompt, _build_q_a_prompt
    from models import get_client, resolve_model

    ds = load_from_disk(str(Path(args.dataset_path)))
    if not isinstance(ds, DatasetDict):
        raise TypeError(f"Expected DatasetDict at {args.dataset_path}")

    rows = _pick_rows(ds, splits=args.splits, per_split=args.per_split)
    if not rows:
        raise RuntimeError("No rows selected for diagnostics")

    provider, _, _ = resolve_model(args.model)
    provider = provider.lower().strip()
    client_kwargs, request_common_kwargs = _client_policy_kwargs(args.model, provider)
    client = get_client(args.model, **client_kwargs)

    trace: dict[str, Any] = {
        "ts_utc": _now_iso(),
        "model": args.model,
        "provider": provider,
        "dataset_path": args.dataset_path,
        "splits": args.splits,
        "per_split": args.per_split,
        "max_tokens": args.max_tokens,
        "max_attempts": args.max_attempts,
        "question_count": len(rows),
        "questions": [],
    }

    for row in rows:
        question = row["question"]
        answer = row["answer"]
        human3 = row["human3"]

        item: dict[str, Any] = {
            "split": row["split"],
            "entry_index": row["entry_index"],
            "question": question,
            "answer": answer,
            "human3": human3,
            "steps": [],
        }

        prompt_b = _build_q_a_prompt(question, answer, count=3, start_letter="B")
        step_b = _run_step(
            client=client,
            provider=provider,
            prompt=prompt_b,
            context="model_from_scratch",
            expected_count=3,
            start_letter="B",
            max_tokens=args.max_tokens,
            max_attempts=args.max_attempts,
            request_common_kwargs=request_common_kwargs,
        )
        item["steps"].append(step_b)

        prompt_c = _build_conditioned_prompt(question, answer, human3, count=6)
        step_c = _run_step(
            client=client,
            provider=provider,
            prompt=prompt_c,
            context="augment_human",
            expected_count=6,
            start_letter="E",
            max_tokens=args.max_tokens,
            max_attempts=args.max_attempts,
            request_common_kwargs=request_common_kwargs,
        )
        item["steps"].append(step_c)

        model3 = step_b.get("parsed") if isinstance(step_b.get("parsed"), list) else []
        if len(model3) == 3:
            prompt_d = _build_conditioned_prompt(question, answer, model3, count=6)
            step_d = _run_step(
                client=client,
                provider=provider,
                prompt=prompt_d,
                context="augment_model_delta_6m",
                expected_count=6,
                start_letter="E",
                max_tokens=args.max_tokens,
                max_attempts=args.max_attempts,
                request_common_kwargs=request_common_kwargs,
            )
        else:
            step_d = {
                "context": "augment_model_delta_6m",
                "expected_count": 6,
                "start_letter": "E",
                "success": False,
                "parsed": None,
                "attempts": [],
                "skipped": True,
                "skip_reason": "model_from_scratch did not produce exactly 3 parsed distractors",
            }
        item["steps"].append(step_d)

        prompt_e = _build_q_a_prompt(question, answer, count=9, start_letter="B")
        step_e = _run_step(
            client=client,
            provider=provider,
            prompt=prompt_e,
            context="augment_ablation",
            expected_count=9,
            start_letter="B",
            max_tokens=args.max_tokens,
            max_attempts=args.max_attempts,
            request_common_kwargs=request_common_kwargs,
        )
        item["steps"].append(step_e)

        trace["questions"].append(item)

    out_path = Path(args.output) if args.output else _default_output_path(args.model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")

    print(str(out_path))
    for q in trace["questions"]:
        statuses = ", ".join(
            [f"{s['context']}={'ok' if s.get('success') else 'fail'}" for s in q["steps"]]
        )
        print(f"{q['split']} {statuses}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnostic tools for debugging generation issues"
    )
    subparsers = parser.add_subparsers(dest="command")

    # failures subcommand
    fail = subparsers.add_parser("failures", help="Report rows with missing distractor columns")
    fail.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to augmented dataset on disk",
    )
    fail.set_defaults(handler=cmd_failures)

    # trace subcommand
    tr = subparsers.add_parser("trace", help="Run structured-generation trace and save per-step JSON")
    tr.add_argument("--model", type=str, default="gemini-3.1-pro-preview")
    tr.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/processed/unified_processed_v2",
    )
    tr.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    tr.add_argument("--per-split", type=int, default=1)
    tr.add_argument("--max-tokens", type=int, default=2048)
    tr.add_argument("--max-attempts", type=int, default=2)
    tr.add_argument("--output", type=str, default=None)
    tr.set_defaults(handler=cmd_trace)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
