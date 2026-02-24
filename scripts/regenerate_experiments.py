#!/usr/bin/env python3
"""Orchestrate Final5 dataset regeneration for all generator models."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATASETS_DIR, PROCESSED_DATASETS_DIR
from data.downloader import (
    download_arc,
    download_gpqa,
    download_mmlu_all_configs,
    download_mmlu_pro,
)
from scripts.process_all import run_all as process_all_run


GENERATOR_MODELS = [
    "gpt-5.2-2025-12-11",
    "claude-opus-4-6",
    "gemini-3.1-pro-preview",
]

ACTIVE_SPLITS = ["arc_challenge", "mmlu_pro", "gpqa"]
EXPECTED_PER_DATASET = 1000
EXPECTED_GENERATION_TASKS_PER_ROW = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate Final5 datasets for all generators")
    parser.add_argument(
        "--processed-dataset",
        type=str,
        default=str(PROCESSED_DATASETS_DIR / "unified_processed_v2"),
        help="Processed dataset path. If missing, preprocessing is run automatically.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DATASETS_DIR / "augmented"),
        help="Base output directory for generated datasets",
    )
    parser.add_argument("--limit", type=int, default=EXPECTED_PER_DATASET)
    parser.add_argument("--skip-push", action="store_true")
    parser.add_argument(
        "--manifest-out",
        type=str,
        default="",
        help="Optional explicit manifest output path",
    )
    return parser.parse_args()


def _ensure_raw_inputs() -> None:
    raw_dir = DATASETS_DIR / "raw"
    needed = {
        "mmlu_pro": raw_dir / "mmlu_pro",
        "mmlu_all": raw_dir / "mmlu_all",
        "arc": raw_dir / "arc",
        "gpqa": raw_dir / "gpqa",
    }

    if not needed["mmlu_pro"].exists():
        print("Raw mmlu_pro missing -> downloading")
        download_mmlu_pro(needed["mmlu_pro"])
    if not needed["mmlu_all"].exists():
        print("Raw mmlu_all missing -> downloading")
        download_mmlu_all_configs(needed["mmlu_all"])
    if not needed["arc"].exists():
        print("Raw arc missing -> downloading")
        download_arc(needed["arc"])
    if not needed["gpqa"].exists():
        print("Raw gpqa missing -> downloading")
        download_gpqa(needed["gpqa"])


def _ensure_processed_dataset(path: Path, limit: int) -> Path:
    if path.exists():
        return path

    _ensure_raw_inputs()
    process_all_run(limit=limit, output_path=path)
    return path


def _run_generator(
    *,
    model: str,
    processed_dataset_path: Path,
    output_root: Path,
    limit: int,
    push: bool,
) -> Dict[str, Any]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace("/", "_")
    output_path = output_root / f"{safe_model}_{ts}"

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/generate_distractors.py",
        "--input",
        str(processed_dataset_path),
        "--output",
        str(output_path),
        "--model",
        model,
        "--limit",
        str(limit),
    ]
    if not push:
        cmd.append("--skip-push")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "model": model,
        "output_path": str(output_path),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout.splitlines()[-20:],
        "stderr_tail": proc.stderr.splitlines()[-20:],
    }


def _summarize_generated(path: Path) -> Dict[str, Any]:
    ds = load_from_disk(str(path))
    if not hasattr(ds, "keys"):
        raise ValueError(f"Expected DatasetDict at {path}")

    split_counts = {split: len(ds[split]) for split in ds.keys()}

    # B/C/D/E generation tasks only.
    rows = sum(split_counts.get(split, 0) for split in ACTIVE_SPLITS)
    observed_generation_rows = rows * EXPECTED_GENERATION_TASKS_PER_ROW

    return {
        "split_counts": split_counts,
        "rows_active_splits": rows,
        "observed_generation_rows": observed_generation_rows,
    }


def _warn_generation_count(manifest: Dict[str, Any]) -> None:
    expected = int(manifest["totals"]["expected_generation_rows"])
    observed = manifest["totals"]["observed_generation_rows"]
    if observed != expected:
        print(
            "WARNING: generation count mismatch | "
            f"expected={expected} observed={observed}"
        )


def main() -> int:
    args = parse_args()

    processed_path = _ensure_processed_dataset(Path(args.processed_dataset), args.limit)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    push = not args.skip_push

    futures = []
    results: list[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(GENERATOR_MODELS)) as ex:
        for model in GENERATOR_MODELS:
            futures.append(
                ex.submit(
                    _run_generator,
                    model=model,
                    processed_dataset_path=processed_path,
                    output_root=output_root,
                    limit=args.limit,
                    push=push,
                )
            )

        for fut in as_completed(futures):
            results.append(fut.result())

    failures = [r for r in results if r["returncode"] != 0]
    for item in sorted(results, key=lambda x: x["model"]):
        print(f"[{item['model']}] rc={item['returncode']}")
        for line in item["stdout_tail"][-8:]:
            print(f"  {line}")
        for line in item["stderr_tail"][-4:]:
            print(f"  [stderr] {line}")

    if failures:
        print("One or more generator runs failed; manifest will include failure diagnostics.")

    manifest_generators = []
    observed_total = 0
    for item in sorted(results, key=lambda x: x["model"]):
        summary: Dict[str, Any] = {}
        out_path = Path(item["output_path"])
        if item["returncode"] == 0 and out_path.exists():
            summary = _summarize_generated(out_path)
            observed_total += summary["observed_generation_rows"]

        manifest_generators.append(
            {
                "model": item["model"],
                "dataset_path": item["output_path"],
                "returncode": item["returncode"],
                "stdout_tail": item["stdout_tail"],
                "stderr_tail": item["stderr_tail"],
                **summary,
            }
        )

    manifest = {
        "manifest_version": 1,
        "schema_version": "final5_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "processed_dataset_path": str(processed_path),
        "active_splits": list(ACTIVE_SPLITS),
        "generator_models": list(GENERATOR_MODELS),
        "limit_per_dataset": args.limit,
        "generators": manifest_generators,
        "totals": {
            "expected_generation_rows": len(GENERATOR_MODELS)
            * len(ACTIVE_SPLITS)
            * args.limit
            * EXPECTED_GENERATION_TASKS_PER_ROW,
            "observed_generation_rows": observed_total,
        },
    }

    if args.manifest_out:
        manifest_path = Path(args.manifest_out)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        manifest_path = output_root / f"final5_regeneration_manifest_{ts}.json"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written: {manifest_path}")

    _warn_generation_count(manifest)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
