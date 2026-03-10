#!/usr/bin/env python3
"""Diagnostic tools for Inspect-first generation runs.

Subcommands:
    failures  - Report rows in an augmented dataset cache with missing Final5 columns
    trace     - Dump generation traces from Inspect `.eval` logs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))


REQUIRED_COLUMNS = [
    ("model_from_scratch", 3),
    ("augment_human", 6),
    ("augment_model", 9),
    ("augment_ablation", 9),
]


def _load_dataset_dict(path: str):
    from data.final5_store import _load_dataset_dict

    return _load_dataset_dict(Path(path))


def cmd_failures(args: argparse.Namespace) -> int:
    ds = _load_dataset_dict(args.dataset_path)

    total_failed = 0
    for split in ds.keys():
        failed = []
        for i, row in enumerate(ds[split]):
            missing = [key for key, count in REQUIRED_COLUMNS if len(row.get(key) or []) < count]
            if missing:
                failed.append(
                    {
                        "idx": i,
                        "id": row.get("id"),
                        "question_id": row.get("question_id"),
                        "missing": missing,
                        "question": str(row.get("question", ""))[:140],
                    }
                )

        print(f"\n{split}: failed_rows={len(failed)}")
        for row in failed:
            print(json.dumps(row, ensure_ascii=True))
        total_failed += len(failed)

    return 1 if total_failed > 0 else 0


def _generation_trace_records(
    *,
    log_dir: str,
    sample_id: str | None,
    only_errors: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    from utils.logs import iter_eval_logs

    records: list[dict[str, Any]] = []
    for log_path, log in iter_eval_logs(log_dir, kind="generation"):
        eval_metadata = dict(getattr(log.eval, "metadata", {}) or {})
        for sample in getattr(log, "samples", []):
            if not sample.scores:
                continue
            score = next(iter(sample.scores.values()))
            metadata = dict(getattr(score, "metadata", {}) or {})
            if not metadata:
                continue
            current_sample_id = str(getattr(sample, "id", ""))
            status = str(metadata.get("status", ""))
            if sample_id and current_sample_id != sample_id:
                continue
            if only_errors and status != "error":
                continue

            records.append(
                {
                    "log_path": str(log_path),
                    "run_name": eval_metadata.get("run_name"),
                    "generation_model": eval_metadata.get("generation_model"),
                    "sample_id": current_sample_id,
                    "status": status,
                    "error": metadata.get("error"),
                    "dataset_type": metadata.get("dataset_type"),
                    "row_index": metadata.get("row_index"),
                    "question": metadata.get("question"),
                    "answer": metadata.get("answer"),
                    "category": metadata.get("category"),
                    "traces": metadata.get("traces", {}),
                    "human_from_scratch": metadata.get("human_from_scratch", []),
                    "model_from_scratch": metadata.get("model_from_scratch", []),
                    "augment_human": metadata.get("augment_human", []),
                    "augment_model": metadata.get("augment_model", []),
                    "augment_ablation": metadata.get("augment_ablation", []),
                }
            )
            if limit is not None and len(records) >= limit:
                return records
    return records


def cmd_trace(args: argparse.Namespace) -> int:
    records = _generation_trace_records(
        log_dir=args.log_dir,
        sample_id=args.sample_id,
        only_errors=args.only_errors,
        limit=args.limit,
    )
    if not records:
        print("No generation traces matched the requested filters.")
        return 1

    if args.summary:
        for record in records:
            print(
                json.dumps(
                    {
                        "sample_id": record["sample_id"],
                        "status": record["status"],
                        "dataset_type": record["dataset_type"],
                        "error": record["error"],
                        "log_path": record["log_path"],
                    },
                    ensure_ascii=True,
                )
            )
        return 0

    payload = json.dumps(records, indent=2, ensure_ascii=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
        print(output_path)
    else:
        print(payload)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect-first Final5 diagnostics")
    subparsers = parser.add_subparsers(dest="command")

    failures = subparsers.add_parser("failures", help="Report rows missing Final5 distractor columns")
    failures.add_argument("--dataset-path", required=True, help="Augmented dataset cache path")
    failures.set_defaults(handler=cmd_failures)

    trace = subparsers.add_parser("trace", help="Dump generation traces from Inspect `.eval` logs")
    trace.add_argument("--log-dir", required=True, help="Generation log directory or `.eval` file")
    trace.add_argument("--sample-id", default=None, help="Optional sample id filter")
    trace.add_argument("--only-errors", action="store_true", help="Only include failed generation samples")
    trace.add_argument("--limit", type=int, default=None, help="Maximum number of records to emit")
    trace.add_argument("--summary", action="store_true", help="Print compact JSON lines instead of full traces")
    trace.add_argument("--output", default=None, help="Optional path for the emitted JSON payload")
    trace.set_defaults(handler=cmd_trace)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
