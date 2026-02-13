#!/usr/bin/env python3
"""
Branching experiment runner for Augmented MCQA.

Runs the branching analysis experiments: 1H, 2H, 3H baselines
with progressive model distractor additions (0M through 6M).

Experiment Matrix per (dataset_type, distractor_source):
  - 1H+0M through 1H+6M (7 configs)
  - 2H+0M through 2H+6M (7 configs)
  - 3H+0M through 3H+6M (7 configs) -- overlaps with main, skipped if exists
  Total: up to 21 configs per combination

After experiments complete, generates branching analysis plots.

Usage:
    python scripts/run_branching_experiments.py \
        --model gpt-4.1 \
        --dataset-path datasets/augmented/unified_processed_gemini-3-flash-preview \
        --limit 5 --dry-run

    python scripts/run_branching_experiments.py \
        --model gpt-4.1 \
        --dataset-path datasets/augmented/unified_processed_gemini-3-flash-preview \
        --distractor-source scratch \
        --dataset-types mmlu_pro \
        --plot-only
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments import ExperimentConfig, run_experiment
from config import RESULTS_DIR, DistractorType
from analysis.branching_analysis import (
    plot_human_distractor_branching,
    plot_human_benefit_comparison,
)


# All dataset types in the unified dataset
ALL_DATASET_TYPES = ["mmlu_pro", "supergpqa", "arc_easy", "arc_challenge"]

# Distractor source -> DistractorType mapping
DISTRACTOR_SOURCE_MAP = {
    "scratch": DistractorType.COND_MODEL_Q_A_SCRATCH,
    "dhuman": DistractorType.COND_MODEL_Q_A_DHUMAN,
    "dmodel": DistractorType.COND_MODEL_Q_A_DMODEL,
}


def build_branching_configs(
    model: str,
    dataset_path: Path,
    dataset_types: List[str],
    distractor_sources: List[str],
    output_base: Optional[Path] = None,
    limit: Optional[int] = None,
    eval_mode: str = "behavioral",
    seed: int = 42,
    reasoning_effort: Optional[str] = None,
    thinking_level: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 100,
) -> List[ExperimentConfig]:
    """Build branching experiment configs (1H/2H/3H x 0M-6M)."""
    if output_base is None:
        output_base = RESULTS_DIR

    configs = []

    for dataset_type in dataset_types:
        for source_name in distractor_sources:
            distractor_type = DISTRACTOR_SOURCE_MAP[source_name]

            for h in range(1, 4):     # 1H, 2H, 3H
                for m in range(0, 7):  # 0M through 6M
                    config_str = f"{h}H{m}M"
                    name = f"{model.replace('/', '_')}_{dataset_type}_{source_name}_{config_str}"
                    output_dir = (
                        output_base
                        / f"{model.replace('/', '_')}_{dataset_type}_{source_name}"
                        / config_str
                    )

                    config = ExperimentConfig(
                        name=name,
                        dataset_path=dataset_path,
                        model_name=model,
                        num_human=h,
                        num_model=m,
                        model_distractor_type=distractor_type,
                        eval_mode=eval_mode,
                        limit=limit,
                        seed=seed,
                        reasoning_effort=reasoning_effort,
                        thinking_level=thinking_level,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        output_dir=output_dir,
                        dataset_type_filter=dataset_type,
                        distractor_source=source_name,
                    )
                    configs.append(config)

    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Run branching analysis experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to unified dataset")

    # Filtering
    parser.add_argument(
        "--distractor-source",
        type=str,
        nargs="+",
        choices=list(DISTRACTOR_SOURCE_MAP.keys()),
        default=list(DISTRACTOR_SOURCE_MAP.keys()),
        help="Distractor sources to run",
    )
    parser.add_argument(
        "--dataset-types",
        type=str,
        nargs="+",
        choices=ALL_DATASET_TYPES,
        default=ALL_DATASET_TYPES,
        help="Dataset types to run",
    )

    # Evaluation settings
    parser.add_argument("--limit", type=int, help="Limit entries per experiment")
    parser.add_argument("--eval-mode", type=str, choices=["accuracy", "behavioral"], default="behavioral")
    parser.add_argument("--seed", type=int, default=42)

    # Model settings
    parser.add_argument("--reasoning-effort", type=str, help="OpenAI reasoning effort")
    parser.add_argument("--thinking-level", type=str, help="Anthropic/Gemini thinking level")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=100)

    # Output
    parser.add_argument("--output-dir", type=str, help="Base output directory")

    # Execution modes
    parser.add_argument("--dry-run", action="store_true", help="Print configs without running")
    parser.add_argument("--skip-existing", action="store_true", help="Skip experiments with existing results")
    parser.add_argument("--plot-only", action="store_true", help="Skip experiments, only generate plots")

    args = parser.parse_args()

    output_base = Path(args.output_dir) if args.output_dir else RESULTS_DIR

    # Build configs
    configs = build_branching_configs(
        model=args.model,
        dataset_path=Path(args.dataset_path),
        dataset_types=args.dataset_types,
        distractor_sources=args.distractor_source,
        output_base=output_base,
        limit=args.limit,
        eval_mode=args.eval_mode,
        seed=args.seed,
        reasoning_effort=args.reasoning_effort,
        thinking_level=args.thinking_level,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print(f"Branching experiment matrix: {len(configs)} configurations")
    print(f"  Model: {args.model}")
    print(f"  Dataset types: {args.dataset_types}")
    print(f"  Distractor sources: {args.distractor_source}")

    if args.dry_run:
        print("\n=== DRY RUN ===\n")
        for i, cfg in enumerate(configs):
            exists = (cfg.output_dir / "results.json").exists()
            marker = " [EXISTS]" if exists else ""
            print(
                f"  [{i+1:3d}] {cfg.distractor_config_str} | "
                f"type={cfg.dataset_type_filter} | "
                f"source={cfg.distractor_source}{marker}"
            )
        print(f"\nTotal: {len(configs)} configs")
        return 0

    if not args.plot_only:
        # Filter existing
        if args.skip_existing:
            original = len(configs)
            configs = [c for c in configs if not (c.output_dir / "results.json").exists()]
            print(f"Skipping {original - len(configs)} existing results")

        # Run experiments
        if configs:
            print(f"\nRunning {len(configs)} experiments...")
            for i, config in enumerate(configs):
                print(f"\n[{i+1}/{len(configs)}] {config.name}")
                try:
                    results = run_experiment(config)
                    print(f"  Accuracy: {results.accuracy:.2%}")
                except Exception as e:
                    print(f"  ERROR: {e}")
        else:
            print("No new experiments to run.")

    # Generate branching plots
    print("\n=== Generating Branching Analysis Plots ===")
    plot_dir = output_base / "plots"

    for dataset_type in args.dataset_types:
        for source in args.distractor_source:
            print(f"\nPlotting: {dataset_type} / {source}")

            plot_human_distractor_branching(
                output_base,
                model=args.model,
                dataset_type=dataset_type,
                distractor_source=source,
                output_dir=plot_dir,
            )

            plot_human_benefit_comparison(
                output_base,
                model=args.model,
                dataset_type=dataset_type,
                distractor_source=source,
                output_dir=plot_dir,
            )

    print(f"\nAll plots saved to: {plot_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
