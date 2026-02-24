#!/usr/bin/env python3
"""Generate Final5 distractor columns for a processed dataset."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.augmentor import AugmentorMode, GenerationConfig, augment_dataset
from data.hub_utils import homogenize_features, push_dataset_to_hub
from models import list_available_models, resolve_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Final5 distractors")
    parser.add_argument("--list-models", action="store_true", help="List available model aliases")
    parser.add_argument("--input", "-i", type=str, help="Input processed dataset path")
    parser.add_argument("--output", "-o", type=str, help="Output dataset path")
    parser.add_argument("--model", "-m", type=str, default="gpt-5.2-2025-12-11")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--skip-push", action="store_true")
    parser.add_argument("--split", type=str, help="Run a single split")
    parser.add_argument("--parallel", action="store_true", help="Spawn one process per split")
    parser.add_argument("--combine", type=str, help="Combine split_* directories")
    parser.add_argument("--force-overwrite", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _model_policy(model_name: str) -> tuple[str, dict, dict]:
    provider, _, _ = resolve_model(model_name)
    provider = provider.lower().strip()

    client_kwargs: dict = {}
    generate_kwargs: dict = {}

    if model_name == "gpt-5.2-2025-12-11":
        client_kwargs["reasoning_effort"] = "medium"
    elif model_name == "claude-opus-4-6":
        generate_kwargs["thinking"] = {"type": "adaptive"}

    return provider, client_kwargs, generate_kwargs


def _build_config(model_name: str, save_interval: int, force_overwrite: bool) -> GenerationConfig:
    provider, client_kwargs, generate_kwargs = _model_policy(model_name)

    return GenerationConfig(
        mode=AugmentorMode.FINAL5,
        model_provider=provider,
        model_name=model_name,
        save_interval=save_interval,
        force_overwrite=force_overwrite,
        reasoning_effort=client_kwargs.get("reasoning_effort"),
        anthropic_thinking=generate_kwargs.get("thinking") if provider == "anthropic" else None,
        generate_kwargs={} if provider == "anthropic" else generate_kwargs,
    )


def combine_splits(base_dir: Path, push: bool = True) -> int:
    from datasets import DatasetDict, load_from_disk

    if not base_dir.exists():
        print(f"Missing combine directory: {base_dir}")
        return 1

    split_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("split_")])
    if not split_dirs:
        print(f"No split_* directories found in {base_dir}")
        return 1

    merged = {}
    for split_dir in split_dirs:
        split_name = split_dir.name.replace("split_", "")
        ds = load_from_disk(str(split_dir))
        if isinstance(ds, DatasetDict):
            for k, v in ds.items():
                merged[k] = v
        else:
            merged[split_name] = ds

    result = homogenize_features(DatasetDict(merged))
    out = base_dir / "combined"
    result.save_to_disk(str(out))
    print(f"Saved combined dataset to {out}")

    if push:
        push_dataset_to_hub(result, dataset_name=base_dir.name)

    return 0


def run_parallel(args: argparse.Namespace) -> int:
    from datasets import load_from_disk

    input_path = Path(args.input)
    ds = load_from_disk(str(input_path))
    if not hasattr(ds, "keys"):
        print("Parallel mode requires DatasetDict input")
        return 1

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_output = Path(args.output) if args.output else input_path.parent / f"{input_path.name}_{args.model}_{timestamp}"
    base_output.mkdir(parents=True, exist_ok=True)

    script = str(Path(__file__).resolve())
    base_cmd = [
        sys.executable,
        script,
        "--input",
        str(input_path),
        "--model",
        args.model,
        "--save-interval",
        str(args.save_interval),
        "--skip-push",
    ]
    if args.limit is not None:
        base_cmd.extend(["--limit", str(args.limit)])
    if args.force_overwrite:
        base_cmd.append("--force-overwrite")

    procs = []
    for split in ds.keys():
        out = base_output / f"split_{split}"
        cmd = base_cmd + ["--split", split, "--output", str(out)]
        procs.append((split, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))

    failed = []
    for split, proc in procs:
        stdout, _ = proc.communicate()
        if proc.returncode != 0:
            failed.append(split)
        print(f"[{split}] rc={proc.returncode}")
        if stdout:
            for line in stdout.strip().splitlines()[-8:]:
                print(f"  {line}")

    if failed:
        print(f"Failed splits: {failed}")
        return 1

    return combine_splits(base_output, push=not args.skip_push)


def main() -> int:
    args = parse_args()

    if args.list_models:
        for name in list_available_models(include_providers=False):
            print(name)
        return 0

    if args.combine:
        return combine_splits(Path(args.combine), push=not args.skip_push)

    if not args.input:
        print("--input is required")
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return 1

    if args.parallel:
        return run_parallel(args)

    cfg = _build_config(args.model, args.save_interval, args.force_overwrite)

    if args.dry_run:
        print(f"Model: {args.model}")
        print(f"Provider: {cfg.model_provider}")
        print(f"Reasoning effort: {cfg.reasoning_effort}")
        print(f"Anthropic thinking: {cfg.anthropic_thinking}")
        print(f"Generate kwargs: {cfg.generate_kwargs}")
        return 0

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else input_path.parent / f"{input_path.name}_{args.model}_{timestamp}"

    split_filter = [args.split] if args.split else None
    augment_dataset(
        dataset_path=input_path,
        config=cfg,
        output_path=output_path,
        limit=args.limit,
        resume_from=Path(args.resume_from) if args.resume_from else None,
        push_to_hub=not args.skip_push,
        splits=split_filter,
    )

    print(f"Done: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
