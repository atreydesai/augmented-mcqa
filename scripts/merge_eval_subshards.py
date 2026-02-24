#!/usr/bin/env python3
"""Merge Final5 eval entry sub-shards into canonical results.json files."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ENTRY_SHARD_RE = re.compile(r"entry_shard_(\d+)_of_(\d+)")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_config_roots(results_root: Path) -> list[Path]:
    roots = []
    for partial_root in sorted(results_root.glob("*/*/*/*/*/_partials")):
        roots.append(partial_root.parent)
    return roots


def _extract_shard_info(path: Path) -> tuple[int, int]:
    match = ENTRY_SHARD_RE.search(str(path.parent))
    if not match:
        raise ValueError(f"Could not parse entry shard from path: {path}")
    return int(match.group(1)), int(match.group(2))


def _compute_accuracy_by_category(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("category", ""))].append(bool(row.get("is_correct", False)))
    out = {}
    for category, values in grouped.items():
        if not values:
            out[category] = 0.0
            continue
        out[category] = sum(1 for v in values if v) / len(values)
    return out


def _compute_behavioral_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter({"G": 0, "H": 0, "M": 0, "?": 0})
    for row in rows:
        label = str(row.get("prediction_type") or "?")
        if label not in {"G", "H", "M", "?"}:
            label = "?"
        counts[label] += 1
    return dict(counts)


def _build_summary(
    rows: list[dict[str, Any]],
    attempted_entries: int,
    successful_entries: int,
    failed_entries: int,
    entry_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if bool(row.get("is_correct", False)))
    accuracy = (correct / total) if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "attempted_entries": attempted_entries,
        "successful_entries": successful_entries,
        "failed_entries": failed_entries,
        "entry_failure_count": len(entry_failures),
        "behavioral_counts": _compute_behavioral_counts(rows),
        "accuracy_by_category": _compute_accuracy_by_category(rows),
    }


def merge_config_root(
    config_root: Path,
    *,
    expected_entry_shards: int | None,
    strict: bool,
) -> dict[str, Any]:
    partial_files = sorted(config_root.glob("_partials/entry_shard_*_of_*/results.json"))
    canonical_path = config_root / "results.json"

    if not partial_files:
        status = "already_merged" if canonical_path.exists() else "missing_partials"
        return {
            "config_root": str(config_root),
            "status": status,
            "partial_count": 0,
            "merged_rows": 0,
        }

    shard_payloads: list[tuple[int, int, dict[str, Any], Path]] = []
    shard_totals = set()
    for path in partial_files:
        shard_idx, shard_total = _extract_shard_info(path)
        shard_totals.add(shard_total)
        payload = _load_json(path)
        shard_payloads.append((shard_idx, shard_total, payload, path))

    if len(shard_totals) != 1:
        raise ValueError(
            f"{config_root}: inconsistent shard totals in partials: {sorted(shard_totals)}"
        )

    shard_total = next(iter(shard_totals))
    if expected_entry_shards is not None and shard_total != expected_entry_shards:
        msg = (
            f"{config_root}: expected entry_shards={expected_entry_shards} but partials are _of_{shard_total}"
        )
        if strict:
            raise ValueError(msg)
        print(f"WARNING: {msg}")

    seen_indices = {idx for idx, _total, _payload, _path in shard_payloads}
    expected_indices = set(range(shard_total))
    missing_indices = sorted(expected_indices - seen_indices)
    if missing_indices and strict:
        raise ValueError(f"{config_root}: missing entry shards: {missing_indices}")

    shard_payloads.sort(key=lambda item: item[0])

    merged_by_idx: dict[int, dict[str, Any]] = {}
    duplicates = 0
    conflicting = 0
    entry_failures: list[dict[str, Any]] = []
    attempted_entries = 0
    successful_entries = 0
    failed_entries = 0

    first_payload = shard_payloads[0][2]
    start_times = []
    end_times = []

    for shard_idx, _total, payload, path in shard_payloads:
        summary = payload.get("summary", {})
        attempted_entries += int(summary.get("attempted_entries", 0) or 0)
        successful_entries += int(summary.get("successful_entries", 0) or 0)
        failed_entries += int(summary.get("failed_entries", 0) or 0)

        timing = payload.get("timing", {})
        if timing.get("start"):
            start_times.append(str(timing["start"]))
        if timing.get("end"):
            end_times.append(str(timing["end"]))

        for failure in payload.get("entry_failures", []):
            enriched = dict(failure)
            enriched.setdefault("entry_shard_index", shard_idx)
            enriched.setdefault("partial_path", str(path))
            entry_failures.append(enriched)

        for row in payload.get("results", []):
            idx = int(row.get("question_idx", -1))
            if idx < 0:
                continue
            if idx in merged_by_idx:
                duplicates += 1
                if merged_by_idx[idx] != row:
                    conflicting += 1
                continue
            merged_by_idx[idx] = row

    if conflicting and strict:
        raise ValueError(f"{config_root}: found {conflicting} conflicting duplicate question_idx rows")

    merged_rows = [merged_by_idx[i] for i in sorted(merged_by_idx.keys())]

    if attempted_entries == 0:
        attempted_entries = len(merged_rows)
    if successful_entries == 0:
        successful_entries = len(merged_rows)

    merged_payload = {
        "config": first_payload.get("config", {}),
        "summary": _build_summary(
            rows=merged_rows,
            attempted_entries=attempted_entries,
            successful_entries=successful_entries,
            failed_entries=failed_entries,
            entry_failures=entry_failures,
        ),
        "timing": {
            "start": min(start_times) if start_times else None,
            "end": max(end_times) if end_times else None,
            "merged_at": datetime.now(timezone.utc).isoformat(),
        },
        "entry_failures": entry_failures,
        "results": merged_rows,
        "merge_metadata": {
            "source_partial_files": [str(p) for _idx, _total, _payload, p in shard_payloads],
            "entry_shards_total": shard_total,
            "missing_entry_shards": missing_indices,
            "duplicate_question_idx_rows": duplicates,
            "conflicting_duplicate_rows": conflicting,
        },
    }

    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.write_text(json.dumps(merged_payload, indent=2), encoding="utf-8")

    status = "merged_incomplete" if missing_indices else "merged"
    return {
        "config_root": str(config_root),
        "status": status,
        "partial_count": len(partial_files),
        "merged_rows": len(merged_rows),
        "missing_entry_shards": missing_indices,
        "duplicate_rows": duplicates,
        "conflicting_rows": conflicting,
        "canonical_path": str(canonical_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Final5 eval entry sub-shards")
    parser.add_argument("--bundle-manifest", type=str, default="")
    parser.add_argument("--results-root", type=str, default="")
    parser.add_argument("--entry-shards", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--summary-out", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_roots: list[Path]
    expected_entry_shards: int | None = args.entry_shards if args.entry_shards > 0 else None
    summary_base_dir: Path

    if args.bundle_manifest:
        manifest_path = Path(args.bundle_manifest)
        manifest = _load_json(manifest_path)
        summary_base_dir = manifest_path.parent

        if expected_entry_shards is None:
            expected_entry_shards = int(manifest.get("entry_shards", 1))

        config_root_values = manifest.get("config_roots", [])
        if config_root_values:
            config_roots = [Path(p) for p in config_root_values]
        else:
            output_base = Path(manifest.get("output_base", "results"))
            config_roots = _discover_config_roots(output_base)
    else:
        if not args.results_root:
            raise ValueError("Provide either --bundle-manifest or --results-root")
        summary_base_dir = Path(args.results_root)
        config_roots = _discover_config_roots(Path(args.results_root))

    if not config_roots:
        print("No config roots found for merging.")
        return 1

    merged = 0
    missing = 0
    errors = 0
    results: list[dict[str, Any]] = []

    for root in sorted(config_roots):
        try:
            item = merge_config_root(
                root,
                expected_entry_shards=expected_entry_shards,
                strict=args.strict,
            )
            results.append(item)
            if item["status"].startswith("merged"):
                merged += 1
            elif item["status"] == "missing_partials":
                missing += 1
        except Exception as exc:  # noqa: BLE001
            errors += 1
            results.append(
                {
                    "config_root": str(root),
                    "status": "error",
                    "error": str(exc),
                }
            )
            if args.strict:
                break

    summary = {
        "merged_at": datetime.now(timezone.utc).isoformat(),
        "bundle_manifest": args.bundle_manifest or None,
        "results_root": args.results_root or None,
        "entry_shards": expected_entry_shards,
        "strict": bool(args.strict),
        "total_config_roots": len(config_roots),
        "merged": merged,
        "missing_partials": missing,
        "errors": errors,
        "configs": results,
    }

    summary_out = Path(args.summary_out) if args.summary_out else summary_base_dir / "merged_summary.json"
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Merged configs: {merged}/{len(config_roots)}")
    print(f"Missing partials: {missing}")
    print(f"Errors: {errors}")
    print(f"Summary: {summary_out}")

    if errors > 0:
        return 1
    if args.strict and missing > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
