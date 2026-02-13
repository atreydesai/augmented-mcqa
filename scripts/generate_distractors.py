#!/usr/bin/env python
"""
Generate Synthetic Distractors.

Uses models from the models/ module directly.
Run with --list-models to see available options.

Usage:
    # List available models
    python scripts/generate_distractors.py --list-models
    
    # Generate with a specific model
    python scripts/generate_distractors.py --input data.json --model gpt-4.1
    
    # Generate with conditioned mode
    python scripts/generate_distractors.py --input data.json --model claude-sonnet-4-5 \\
        --mode conditioned_human
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_client, _CLIENT_REGISTRY
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
    parser.add_argument("--model", "-m", type=str, default="gpt-4.1", help="Model name from registry")
    
    parser.add_argument(
        "--mode",
        type=str,
        default="from_scratch",
        choices=["from_scratch", "conditioned_human", "conditioned_synthetic"],
    )
    parser.add_argument("--num-distractors", type=int, default=9)
    parser.add_argument("--limit", type=int, help="Limit entries")
    parser.add_argument("--save-interval", type=int, default=5, help="Save intermediate results every N entries")
    parser.add_argument("--skip-push", action="store_true", help="Skip pushing to HF Hub")
    parser.add_argument("--dry-run", action="store_true")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.list_models:
        print("\n📋 Available Models (from models/_CLIENT_REGISTRY):")
        print("=" * 50)
        for model_name in sorted(_CLIENT_REGISTRY.keys()):
            print(f"  {model_name}")
        return 0
    
    if not args.input:
        print("Error: --input required")
        return 1
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return 1
    
    # Get client directly from models module
    try:
        client = get_client(args.model)
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
    output_path = Path(args.output) if args.output else input_path.parent / f"{input_path.name}_{args.model}_{timestamp}"
    
    print(f"\n🔧 Model: {client.name}")
    print(f"   Mode: {args.mode}")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    
    config = GenerationConfig(
        mode=mode_map[args.mode],
        model_provider=provider,
        model_name=args.model,
        num_distractors=args.num_distractors,
        save_interval=args.save_interval,
    )
    
    print(f"🚀 Starting generation for {input_path}...")
    
    augmented = augment_dataset(
        dataset_path=input_path,
        config=config,
        output_path=output_path,
        limit=args.limit,
        push_to_hub=not args.skip_push
    )
    
    print(f"\n✅ Done. Augmented dataset saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
