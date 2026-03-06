#!/usr/bin/env python3
"""Merge Final5 eval entry sub-shards into canonical Arrow + summary outputs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, load_from_disk


ENTRY_SHARD_RE = re.compile(r"entry_shard_(\d+)_of_(\d+)")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_config_roots(results_root: Path) -> list[Path]:
    roots = []
    for partial_root in sorted(results_root.glob("*/*/*/*/*/_partials")):
        if any(partial_root.glob("entry_shard_*_of_*/rows")):
            roots.append(partial_root.parent)
    return roots


def _extract_shard_info(rows_path: Path) -> tuple[int, int]:
    match = ENTRY_SHARD_RE.search(str(rows_path.parent))
    if not match:
        raise ValueError(f"Could not parse entry shard from path: {rows_path}")
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


def _read_partial_summary(path: Path) -> dict[str, Any]:
    summary_path = path.parent / "summary.json"
    if not summary_path.exists():
        return {}
    return _load_json(summary_path)


def _resolve_expected_for_root(
    config_root: Path,
    expected_by_config_root: dict[str, int],
    default_expected: int | None,
) -> int | None:
    if expected_by_config_root:
        candidates = {str(config_root), str(config_root.resolve())}
        try:
            candidates.add(str(config_root.relative_to(Path.cwd())))
        except ValueError:
            pass
        for key in candidates:
            if key in expected_by_config_root:
                return int(expected_by_config_root[key])
    return default_expected


def merge_config_root(
    config_root: Path,
    *,
    expected_entry_shards: int | None,
    strict: bool,
) -> dict[str, Any]:
    partial_row_dirs = sorted(config_root.glob("_partials/entry_shard_*_of_*/rows"))
    canonical_summary_path = config_root / "summary.json"
    canonical_rows_path = config_root / "rows"
    merge_metadata_path = config_root / "merge_metadata.json"

    if not partial_row_dirs:
        status = "already_merged" if canonical_summary_path.exists() and canonical_rows_path.exists() else "missing_partials"
        return {
            "config_root": str(config_root),
            "status": status,
            "partial_count": 0,
            "merged_rows": 0,
        }

    shard_payloads: list[tuple[int, int, Any, dict[str, Any], Path]] = []
    shard_totals = set()
    for rows_dir in partial_row_dirs:
        shard_idx, shard_total = _extract_shard_info(rows_dir)
        shard_totals.add(shard_total)
        dataset = load_from_disk(str(rows_dir))
        summary_payload = _read_partial_summary(rows_dir)
        shard_payloads.append((shard_idx, shard_total, dataset, summary_payload, rows_dir))

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

    seen_indices = {idx for idx, _total, _ds, _summary, _rows_dir in shard_payloads}
    expected_indices = set(range(shard_total))
    missing_indices = sorted(expected_indices - seen_indices)
    if missing_indices and strict:
        raise ValueError(f"{config_root}: missing entry shards: {missing_indices}")

    shard_payloads.sort(key=lambda item: item[0])

    datasets_in_order = [ds for _idx, _total, ds, _summary, _rows_dir in shard_payloads]
    if len(datasets_in_order) == 1:
        combined = datasets_in_order[0]
    else:
        combined = concatenate_datasets(datasets_in_order)

    merged_by_idx: dict[int, dict[str, Any]] = {}
    duplicates = 0
    conflicting = 0
    for row in combined:
        idx = int(row.get("question_idx", -1))
        if idx < 0:
            continue
        row_obj = dict(row)
        if idx in merged_by_idx:
            duplicates += 1
            if merged_by_idx[idx] != row_obj:
                conflicting += 1
            continue
        merged_by_idx[idx] = row_obj

    if conflicting and strict:
        raise ValueError(f"{config_root}: found {conflicting} conflicting duplicate question_idx rows")

    merged_rows = [merged_by_idx[i] for i in sorted(merged_by_idx.keys())]

    entry_failures: list[dict[str, Any]] = []
    attempted_entries = 0
    successful_entries = 0
    failed_entries = 0
    start_times: list[str] = []
    end_times: list[str] = []
    first_summary = next((summary for _idx, _tot, _ds, summary, _rows in shard_payloads if summary), {})

    for shard_idx, _total, _ds, summary_payload, rows_dir in shard_payloads:
        summary = summary_payload.get("summary", {})
        attempted_entries += int(summary.get("attempted_entries", 0) or 0)
        successful_entries += int(summary.get("successful_entries", 0) or 0)
        failed_entries += int(summary.get("failed_entries", 0) or 0)

        timing = summary_payload.get("timing", {})
        if timing.get("start"):
            start_times.append(str(timing["start"]))
        if timing.get("end"):
            end_times.append(str(timing["end"]))

        for failure in summary_payload.get("entry_failures", []):
            enriched = dict(failure)
            enriched.setdefault("entry_shard_index", shard_idx)
            enriched.setdefault("partial_rows_path", str(rows_dir))
            entry_failures.append(enriched)

    if attempted_entries == 0:
        attempted_entries = len(merged_rows)
    if successful_entries == 0:
        successful_entries = len(merged_rows)

    if canonical_rows_path.exists():
        if canonical_rows_path.is_dir():
            shutil.rmtree(canonical_rows_path)
        else:
            canonical_rows_path.unlink()
    Dataset.from_list(merged_rows).save_to_disk(str(canonical_rows_path))

    summary_payload = {
        "config": first_summary.get("config", {}),
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
        "rows_path": str(canonical_rows_path),
    }
    canonical_summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    merge_metadata = {
        "source_partial_rows": [str(rows_dir) for _idx, _total, _ds, _summary, rows_dir in shard_payloads],
        "entry_shards_total": shard_total,
        "missing_entry_shards": missing_indices,
        "duplicate_question_idx_rows": duplicates,
        "conflicting_duplicate_rows": conflicting,
        "merged_rows_path": str(canonical_rows_path),
    }
    merge_metadata_path.write_text(json.dumps(merge_metadata, indent=2), encoding="utf-8")

    status = "merged_incomplete" if missing_indices else "merged"
    return {
        "config_root": str(config_root),
        "status": status,
        "partial_count": len(partial_row_dirs),
        "merged_rows": len(merged_rows),
        "missing_entry_shards": missing_indices,
        "duplicate_rows": duplicates,
        "conflicting_rows": conflicting,
        "canonical_summary_path": str(canonical_summary_path),
        "canonical_rows_path": str(canonical_rows_path),
        "merge_metadata_path": str(merge_metadata_path),
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
    default_expected_entry_shards: int | None = args.entry_shards if args.entry_shards > 0 else None
    expected_by_config_root: dict[str, int] = {}
    summary_base_dir: Path

    if args.bundle_manifest:
        manifest_path = Path(args.bundle_manifest)
        manifest = _load_json(manifest_path)
        summary_base_dir = manifest_path.parent

        if default_expected_entry_shards is None:
            if "entry_shards" in manifest:
                default_expected_entry_shards = int(manifest.get("entry_shards", 1))

        raw_expected_map = manifest.get("expected_entry_shards_by_config_root", {})
        if isinstance(raw_expected_map, dict):
            expected_by_config_root = {str(k): int(v) for k, v in raw_expected_map.items()}

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
        expected_for_root = _resolve_expected_for_root(
            root,
            expected_by_config_root=expected_by_config_root,
            default_expected=default_expected_entry_shards,
        )
        try:
            item = merge_config_root(
                root,
                expected_entry_shards=expected_for_root,
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
        "entry_shards": default_expected_entry_shards,
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
