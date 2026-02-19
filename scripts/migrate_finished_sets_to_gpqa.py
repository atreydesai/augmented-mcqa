#!/usr/bin/env python3
"""
Migrate finished-set datasets to GPQA usage.

Workflow per finished-set family:
1) Download source dataset repo from HF and overwrite local combined directory
2) Add mapped GPQA split (Idavidrein/gpqa, gpqa_main/train)
3) Generate GPQA distractor columns (scratch, dhuman, dmodel)
4) Merge generated GPQA split back into combined dataset
5) Push updated dataset to a new public HF dataset repo
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset, load_from_disk

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    AUGMENTED_DATASETS_DIR,
    HF_SKIP_PUSH,
    HF_TOKEN,
    PROJECT_ROOT,
    get_api_key,
)
from data.augmentor import AugmentorMode, GenerationConfig, augment_dataset
from data.gpqa_processor import load_gpqa_dataset
from data.hub_utils import push_dataset_to_hub
from models import resolve_model


@dataclass(frozen=True)
class TargetSpec:
    source_repo: str
    local_combined_dir: Path
    model_alias: str
    model_slug: str


TARGETS: List[TargetSpec] = [
    TargetSpec(
        source_repo="atreydesai/qgqa-gemini-3-flash-20260213-041708",
        local_combined_dir=PROJECT_ROOT / "datasets" / "finished_sets" / "gemini-3-flash_20260213_041708" / "combined",
        model_alias="gemini-3-flash-preview",
        model_slug="gemini-3-flash",
    ),
    TargetSpec(
        source_repo="atreydesai/qgqa-claude-opus-4-6-20260213-041708",
        local_combined_dir=PROJECT_ROOT / "datasets" / "finished_sets" / "claude-opus-4-6_20260213_041708" / "combined",
        model_alias="claude-opus-4-6",
        model_slug="claude-opus-4-6",
    ),
    TargetSpec(
        source_repo="atreydesai/qgqa-gpt-5.2-20260213-041705",
        local_combined_dir=PROJECT_ROOT / "datasets" / "finished_sets" / "gpt-5.2_20260213_041705" / "combined",
        model_alias="gpt-5.2",
        model_slug="gpt-5-2",
    ),
    TargetSpec(
        source_repo="atreydesai/qgqa-gpt-4.1-20260213-041705",
        local_combined_dir=PROJECT_ROOT / "datasets" / "finished_sets" / "gpt-4.1_20260213_041705" / "combined",
        model_alias="gpt-4.1",
        model_slug="gpt-4-1",
    ),
]


REQUIRED_GENERATED_COLUMNS = [
    "cond_model_q_a_scratch",
    "cond_model_q_a_dhuman",
    "cond_model_q_a_dmodel",
]


def _sanitize_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")


def _default_for_feature(feature: Any) -> Any:
    if isinstance(feature, Sequence):
        return []
    if isinstance(feature, Value):
        if feature.dtype == "string":
            return ""
        return None
    return None


def _aligned_gpqa_dataset(
    gpqa_rows: List[Dict[str, Any]],
    template_features: Features,
) -> Dataset:
    rows: List[Dict[str, Any]] = []
    for base in gpqa_rows:
        row: Dict[str, Any] = {}
        for col, feature in template_features.items():
            if col in base:
                row[col] = base[col]
            else:
                row[col] = _default_for_feature(feature)
        rows.append(row)

    return Dataset.from_list(rows, features=template_features)


def _load_hf_datasetdict(repo_id: str) -> DatasetDict:
    kwargs: Dict[str, Any] = {}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    ds = load_dataset(repo_id, **kwargs)
    if isinstance(ds, DatasetDict):
        return ds
    # Should not happen for these repos, but keep strict cast behavior explicit.
    return DatasetDict({"train": ds})


def _save_datasetdict(path: Path, dataset: DatasetDict, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists (use --overwrite)")
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path))


def _validate_gpqa_generated(dataset: DatasetDict) -> Dict[str, Any]:
    if "gpqa" not in dataset:
        raise ValueError("Missing gpqa split after generation")

    gpqa = dataset["gpqa"]
    if len(gpqa) == 0:
        raise ValueError("gpqa split is empty")

    for col in REQUIRED_GENERATED_COLUMNS:
        if col not in gpqa.column_names:
            raise ValueError(f"Missing required generated column: {col}")

    empty_counts: Dict[str, int] = {}
    for col in REQUIRED_GENERATED_COLUMNS:
        empty_counts[col] = sum(1 for x in gpqa[col] if not x)

    if any(v > 0 for v in empty_counts.values()):
        raise ValueError(f"Found empty generated cells in GPQA columns: {empty_counts}")

    bad_human_count = sum(1 for x in gpqa["choices_human"] if len(x) != 3)
    if bad_human_count:
        raise ValueError(f"GPQA rows with non-3 human distractors: {bad_human_count}")

    return {
        "gpqa_rows": len(gpqa),
        "empty_generated_cells": empty_counts,
        "bad_human_count": bad_human_count,
    }


def _worker_run(
    target: TargetSpec,
    gpqa_rows: List[Dict[str, Any]],
    *,
    overwrite: bool,
    push_public: bool,
    run_tag: str,
) -> Dict[str, Any]:
    print(f"\n=== [{target.model_alias}] start ===")

    # 1) Download source repo and overwrite local combined dir.
    source_ds = _load_hf_datasetdict(target.source_repo)
    _save_datasetdict(target.local_combined_dir, source_ds, overwrite=overwrite)
    print(f"[{target.model_alias}] downloaded -> {target.local_combined_dir}")

    # 2) Add GPQA split aligned to existing schema.
    combined = load_from_disk(str(target.local_combined_dir))
    if not isinstance(combined, DatasetDict):
        raise ValueError(f"Expected DatasetDict at {target.local_combined_dir}")

    template_split = "mmlu_pro" if "mmlu_pro" in combined else next(iter(combined.keys()))
    template_features = combined[template_split].features
    gpqa_ds = _aligned_gpqa_dataset(gpqa_rows, template_features)
    combined["gpqa"] = gpqa_ds
    _save_datasetdict(target.local_combined_dir, combined, overwrite=True)
    print(f"[{target.model_alias}] gpqa split added ({len(gpqa_ds)} rows)")

    # 3) Generate GPQA distractors only.
    provider, _, _ = resolve_model(target.model_alias)
    gen_cfg = GenerationConfig(
        mode=AugmentorMode.FROM_SCRATCH,
        model_provider=provider,
        model_name=target.model_alias,
        num_distractors=9,
        reasoning_effort=None,  # keep provider defaults unless user sets explicitly
        generate_branching_prefix_columns=False,
    )

    tmp_output = AUGMENTED_DATASETS_DIR / f"tmp_gpqa_migrate_{target.model_slug}_{run_tag}"
    if tmp_output.exists():
        shutil.rmtree(tmp_output)

    augment_dataset(
        dataset_path=target.local_combined_dir,
        config=gen_cfg,
        output_path=tmp_output,
        limit=None,
        push_to_hub=False,
        splits=["gpqa"],
    )
    generated = load_from_disk(str(tmp_output))
    if not isinstance(generated, DatasetDict) or "gpqa" not in generated:
        raise ValueError(f"Unexpected generated output at {tmp_output}")

    gpqa_aug = generated["gpqa"]
    combined = load_from_disk(str(target.local_combined_dir))
    combined["gpqa"] = gpqa_aug
    _save_datasetdict(target.local_combined_dir, combined, overwrite=True)
    print(f"[{target.model_alias}] gpqa generation merged")

    # 4) Validate.
    validation = _validate_gpqa_generated(combined)
    print(f"[{target.model_alias}] validation ok: {validation}")

    # 5) Push to new public repo.
    repo_id = f"atreydesai/qgqa-{run_tag}-{target.model_slug}"
    pushed_url = None
    if push_public:
        pushed_url = push_dataset_to_hub(
            combined,
            repo_id=repo_id,
            private=False,
        )
        if not pushed_url:
            raise RuntimeError(f"Push failed for {repo_id}")

    # Cleanup temporary generated split directory.
    if tmp_output.exists():
        shutil.rmtree(tmp_output)

    print(f"=== [{target.model_alias}] done ===")
    return {
        "source_repo": target.source_repo,
        "model_alias": target.model_alias,
        "local_combined_dir": str(target.local_combined_dir),
        "target_repo_id": repo_id,
        "pushed_url": pushed_url,
        **validation,
    }


def _ensure_env() -> None:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required")

    required_providers = sorted({resolve_model(t.model_alias)[0] for t in TARGETS})
    for provider in required_providers:
        if provider in {"openai", "anthropic", "gemini", "deepseek"}:
            # Raises if missing.
            get_api_key(provider)

    hf_home = os.getenv("HF_HOME", "")
    if (not hf_home) or hf_home.startswith("/fs"):
        local_cache = str(PROJECT_ROOT / ".hf_cache")
        os.environ["HF_HOME"] = local_cache
        os.environ["HF_DATASETS_CACHE"] = str(Path(local_cache) / "datasets")
        os.environ["TRANSFORMERS_CACHE"] = str(Path(local_cache) / "transformers")


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate finished-set datasets to GPQA split")
    parser.add_argument("--gpqa-subset", type=str, default="gpqa_main")
    parser.add_argument("--gpqa-split", type=str, default="train")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-public", action="store_true")
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--manifest-out", type=str, default=None)
    args = parser.parse_args()

    if args.push_public and HF_SKIP_PUSH:
        raise RuntimeError("HF_SKIP_PUSH is set; cannot push while --push-public is enabled")

    _ensure_env()

    run_tag = args.run_tag or datetime.now(timezone.utc).strftime("gpqa-migrate-%Y%m%d-%H%M%S")
    run_tag = _sanitize_slug(run_tag)
    print(f"Run tag: {run_tag}")

    gpqa_rows = load_gpqa_dataset(split=args.gpqa_split, subset=args.gpqa_subset, limit=None)
    if len(gpqa_rows) != 448:
        print(f"Warning: expected 448 GPQA rows, loaded {len(gpqa_rows)}")
    print(f"Loaded GPQA rows: {len(gpqa_rows)}")

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                _worker_run,
                target,
                gpqa_rows,
                overwrite=args.overwrite,
                push_public=args.push_public,
                run_tag=run_tag,
            )
            for target in TARGETS
        ]

        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:
                errors.append({"error": str(exc)})

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_tag": run_tag,
        "gpqa_subset": args.gpqa_subset,
        "gpqa_split": args.gpqa_split,
        "max_workers": args.max_workers,
        "results": sorted(results, key=lambda x: x["model_alias"]),
        "errors": errors,
    }

    if args.manifest_out:
        manifest_path = Path(args.manifest_out)
    else:
        manifest_path = PROJECT_ROOT / "results" / run_tag / "migration_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest written to: {manifest_path}")

    if errors:
        print(f"Completed with {len(errors)} error(s).")
        return 1

    print("Migration completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
