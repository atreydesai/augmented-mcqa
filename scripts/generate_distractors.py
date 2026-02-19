#!/usr/bin/env python
"""
Generate Synthetic Distractors.

Uses models from the models/ module directly.
Run with --list-models to see available options.

Usage:
    # List available models
    python scripts/generate_distractors.py --list-models

    # Generate with a specific model (sequential, all splits)
    python scripts/generate_distractors.py --input data.json --model gpt-4.1

    # Run a SINGLE split (for parallelization)
    python scripts/generate_distractors.py --input data.json --model gpt-4.1 --split arc_easy

    # Run ALL splits in parallel (spawns one process per split)
    python scripts/generate_distractors.py --input data.json --model gpt-4.1 --parallel

    # Combine split results into a single dataset
    python scripts/generate_distractors.py --combine datasets/augmented/run_20260213_040000
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_client, list_available_models, resolve_model
from data.augmentor import augment_dataset, AugmentorMode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic distractors using models/ module",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )

    parser.add_argument("--input", "-i", type=str, help="Input dataset JSON file")
    parser.add_argument("--output", "-o", type=str, help="Output file")
    parser.add_argument("--model", "-m", type=str, default="gpt-4.1", help="Model alias/name")

    parser.add_argument(
        "--mode",
        type=str,
        default="from_scratch",
        choices=["from_scratch", "conditioned_human", "conditioned_synthetic"],
    )
    parser.add_argument("--num-distractors", type=int, default=9)
    parser.add_argument("--limit", type=int, help="Limit entries per split")
    parser.add_argument("--save-interval", type=int, default=5, help="Save intermediate results every N entries")
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["minimal", "low", "medium", "high", "none"],
        default="minimal",
        help="Reasoning effort for OpenAI GPT-5 family models",
    )
    parser.add_argument(
        "--generate-branching-prefix-columns",
        action="store_true",
        help="Generate branching prefix columns (cond_model_q_a_dhuman_h1/h2/h3). Off by default.",
    )
    parser.add_argument("--skip-push", action="store_true", help="Skip pushing to HF Hub")
    parser.add_argument("--dry-run", action="store_true")

    # Parallelization
    parser.add_argument("--split", type=str, help="Process only this specific split (for parallelization)")
    parser.add_argument("--parallel", action="store_true", help="Run all splits in parallel (spawns subprocesses)")
    parser.add_argument("--combine", type=str, help="Combine per-split results from this directory into a unified dataset")

    return parser.parse_args()


def combine_splits(combine_dir: str, push_to_hub: bool = True):
    """Combine per-split result directories into a single DatasetDict."""
    from datasets import DatasetDict, load_from_disk
    from data.hub_utils import push_dataset_to_hub, homogenize_features

    combine_path = Path(combine_dir)
    if not combine_path.exists():
        print(f"Error: {combine_path} not found")
        return 1

    # Find all split_* subdirectories
    split_dirs = sorted([d for d in combine_path.iterdir() if d.is_dir() and d.name.startswith("split_")])

    if not split_dirs:
        print(f"No split_* directories found in {combine_path}")
        return 1

    print(f"\n📦 Combining {len(split_dirs)} split directories from {combine_path}")

    combined = {}
    for split_dir in split_dirs:
        split_name = split_dir.name.replace("split_", "")
        print(f"  Loading {split_name} from {split_dir}...")
        ds = load_from_disk(str(split_dir))
        # If it's a DatasetDict with a single split, unwrap it
        if isinstance(ds, DatasetDict):
            for k, v in ds.items():
                combined[k] = v
        else:
            combined[split_name] = ds

    result = DatasetDict(combined)

    # Homogenize features
    result = homogenize_features(result)

    # Save combined
    output_path = combine_path / "combined"
    result.save_to_disk(str(output_path))
    print(f"\n✅ Combined dataset saved to {output_path}")
    print(f"   Splits: {list(result.keys())}")
    for split in result:
        print(f"   - {split}: {len(result[split])} rows")

    if push_to_hub:
        dataset_name = combine_path.name
        print(f"\nPushing to Hub as {dataset_name}...")
        push_dataset_to_hub(result, dataset_name=dataset_name)

    return 0


def run_parallel(args):
    """Spawn one subprocess per split, all running concurrently."""
    from datasets import load_from_disk

    input_path = Path(args.input)
    dataset = load_from_disk(str(input_path))
    split_names = list(dataset.keys())

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output:
        base_output = Path(f"{args.output}_{timestamp}")
    else:
        base_output = input_path.parent / f"{input_path.name}_{args.model}_{timestamp}"

    base_output.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Parallel mode: spawning {len(split_names)} processes")
    print(f"   Model: {args.model}")
    print(f"   Splits: {split_names}")
    print(f"   Output: {base_output}")

    # Build base command
    script = str(Path(__file__).resolve())
    base_cmd = [
        sys.executable, script,
        "--input", str(input_path),
        "--model", args.model,
        "--mode", args.mode,
        "--num-distractors", str(args.num_distractors),
        "--save-interval", str(args.save_interval),
        "--reasoning-effort", args.reasoning_effort,
        "--skip-push",  # Don't push individual splits
    ]
    if args.generate_branching_prefix_columns:
        base_cmd.append("--generate-branching-prefix-columns")
    if args.limit:
        base_cmd += ["--limit", str(args.limit)]

    # Spawn one process per split
    processes = []
    for split_name in split_names:
        split_output = base_output / f"split_{split_name}"
        cmd = base_cmd + [
            "--split", split_name,
            "--output", str(split_output),
        ]
        print(f"   Starting {split_name}...")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        processes.append((split_name, proc))

    # Wait for all and stream output
    print(f"\n⏳ Waiting for {len(processes)} processes...\n")
    failed = []
    for split_name, proc in processes:
        stdout, _ = proc.communicate()
        status = "✅" if proc.returncode == 0 else "❌"
        print(f"{status} {split_name} (exit code {proc.returncode})")
        if stdout:
            # Print last few lines
            lines = stdout.strip().split("\n")
            for line in lines[-5:]:
                print(f"   {line}")
        if proc.returncode != 0:
            failed.append(split_name)
        print()

    if failed:
        print(f"\n⚠️ {len(failed)} splits failed: {failed}")
        return 1

    # Auto-combine
    print(f"\n📦 All splits done. Combining...")
    return combine_splits(str(base_output), push_to_hub=not args.skip_push)


def main():
    args = parse_args()

    if args.list_models:
        print("\n📋 Available Models (from config/model_aliases.toml + providers):")
        print("=" * 50)
        for model_name in list_available_models(include_providers=True):
            print(f"  {model_name}")
        return 0

    # Combine mode
    if args.combine:
        return combine_splits(args.combine, push_to_hub=not args.skip_push)

    if not args.input:
        print("Error: --input required")
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return 1

    # Parallel mode
    if args.parallel:
        return run_parallel(args)

    # Get client directly from models module
    provider_name, _, _ = resolve_model(args.model)
    client_kwargs = {}
    if provider_name == "openai":
        client_kwargs["reasoning_effort"] = args.reasoning_effort

    try:
        client = get_client(args.model, **client_kwargs)
    except ValueError as e:
        print(f"Error: {e}")
        print("Use --list-models to see available options")
        return 1

    mode_map = {
        "from_scratch": AugmentorMode.FROM_SCRATCH,
        "conditioned_human": AugmentorMode.CONDITIONED_HUMAN,
        "conditioned_synthetic": AugmentorMode.CONDITIONED_SYNTHETIC,
    }

    if args.dry_run:
        print(f"\n🔧 Model: {client.name}")
        print(f"   Mode: {args.mode}")
        print(f"   Input: {input_path}")
        print("\n🔍 Dry run - done")
        return 0

    print("\n🚀 Starting generation...")

    # Determine provider from client type
    provider = client.__class__.__name__.replace("Client", "").lower()

    from data.augmentor import GenerationConfig, augment_dataset

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_path = Path(f"{args.output}_{timestamp}")
    else:
        output_path = input_path.parent / f"{input_path.name}_{args.model}_{timestamp}"

    print(f"\n🔧 Model: {client.name}")
    print(f"   Mode: {args.mode}")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Reasoning effort: {args.reasoning_effort}")
    print(f"   Branching prefix cols: {args.generate_branching_prefix_columns}")
    if args.split:
        print(f"   Split: {args.split}")

    config = GenerationConfig(
        mode=mode_map[args.mode],
        model_provider=provider,
        model_name=args.model,
        num_distractors=args.num_distractors,
        save_interval=args.save_interval,
        reasoning_effort=args.reasoning_effort,
        generate_branching_prefix_columns=args.generate_branching_prefix_columns,
    )

    print(f"🚀 Starting generation for {input_path}...")

    # Pass split filter if specified
    split_filter = [args.split] if args.split else None

    augment_dataset(
        dataset_path=input_path,
        config=config,
        output_path=output_path,
        limit=args.limit,
        push_to_hub=not args.skip_push,
        splits=split_filter,
    )

    print(f"\n✅ Done. Augmented dataset saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
