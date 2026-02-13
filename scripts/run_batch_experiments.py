#!/usr/bin/env python3
"""
Batch experiment runner for Augmented MCQA.

Generates and runs the full experiment matrix across dataset_types and
distractor sources. Replaces the old api_mcqa_eval_essential.sh shell script.

Experiment Matrix per (dataset_type, distractor_source):
  - 3H+M Progressive: 3H+0M through 3H+6M (7 configs)
  - Human-Only: 1H+0M, 2H+0M, 3H+0M (3 configs)
  - Model-Only: 0H+1M through 0H+6M (6 configs)
  Total: 16 configs per combination
  Grand total: 16 * 4 dataset_types * 3 distractor_sources = 192 configs

Usage:
    python scripts/run_batch_experiments.py \
        --model gpt-4.1 \
        --dataset-path datasets/augmented/unified_processed_gemini-3-flash-preview \
        --limit 5 --dry-run

    python scripts/run_batch_experiments.py \
        --model gpt-4.1 \
        --dataset-path datasets/augmented/unified_processed_gemini-3-flash-preview \
        --distractor-source scratch \
        --dataset-types mmlu_pro supergpqa \
        --parallel 4
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments import ExperimentConfig, ExperimentRunner, run_experiment
from config import RESULTS_DIR, DistractorType


# All dataset types in the unified dataset
ALL_DATASET_TYPES = ["mmlu_pro", "supergpqa", "arc_easy", "arc_challenge"]

# Distractor source -> DistractorType mapping
DISTRACTOR_SOURCE_MAP = {
    "scratch": DistractorType.COND_MODEL_Q_A_SCRATCH,
    "dhuman": DistractorType.COND_MODEL_Q_A_DHUMAN,
    "dmodel": DistractorType.COND_MODEL_Q_A_DMODEL,
}

# Experiment configurations (num_human, num_model)
EXPERIMENT_3H_PROGRESSIVE = [(3, m) for m in range(0, 7)]    # 3H+0M..3H+6M
EXPERIMENT_HUMAN_ONLY = [(h, 0) for h in range(1, 4)]        # 1H+0M..3H+0M
EXPERIMENT_MODEL_ONLY = [(0, m) for m in range(1, 7)]        # 0H+1M..0H+6M


def build_experiment_configs(
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
    """
    Build the full experiment matrix.

    Returns:
        List of ExperimentConfig objects for all combinations.
    """
    if output_base is None:
        output_base = RESULTS_DIR

    configs = []

    for dataset_type in dataset_types:
        for source_name in distractor_sources:
            distractor_type = DISTRACTOR_SOURCE_MAP[source_name]

            # Combine all distractor configs, deduplicating (3H+0M appears in both progressive and human-only)
            seen = set()
            all_distractor_configs = []
            for dc_list in [EXPERIMENT_3H_PROGRESSIVE, EXPERIMENT_HUMAN_ONLY, EXPERIMENT_MODEL_ONLY]:
                for nh, nm in dc_list:
                    key = (nh, nm)
                    if key not in seen:
                        seen.add(key)
                        all_distractor_configs.append(key)

            for num_human, num_model in all_distractor_configs:
                config_str = f"{num_human}H{num_model}M"
                name = f"{model.replace('/', '_')}_{dataset_type}_{source_name}_{config_str}"
                output_dir = output_base / f"{model.replace('/', '_')}_{dataset_type}_{source_name}" / config_str

                config = ExperimentConfig(
                    name=name,
                    dataset_path=dataset_path,
                    model_name=model,
                    num_human=num_human,
                    num_model=num_model,
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


def run_single_config(config: ExperimentConfig) -> dict:
    """Run a single experiment config and return summary."""
    try:
        results = run_experiment(config)
        return {
            "name": config.name,
            "config": config.distractor_config_str,
            "accuracy": results.accuracy,
            "total": len(results.results),
            "status": "success",
        }
    except Exception as e:
        return {
            "name": config.name,
            "config": config.distractor_config_str,
            "accuracy": 0.0,
            "total": 0,
            "status": f"error: {e}",
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run batch MCQA evaluation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gpt-4.1)")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to unified dataset")

    # Filtering
    parser.add_argument(
        "--distractor-source",
        type=str,
        nargs="+",
        choices=list(DISTRACTOR_SOURCE_MAP.keys()),
        default=list(DISTRACTOR_SOURCE_MAP.keys()),
        help="Distractor sources to run (default: all)",
    )
    parser.add_argument(
        "--dataset-types",
        type=str,
        nargs="+",
        choices=ALL_DATASET_TYPES,
        default=ALL_DATASET_TYPES,
        help="Dataset types to run (default: all)",
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

    # Execution
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without running")
    parser.add_argument("--skip-existing", action="store_true", help="Skip experiments with existing results")

    args = parser.parse_args()

    output_base = Path(args.output_dir) if args.output_dir else RESULTS_DIR

    # Build all configs
    configs = build_experiment_configs(
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

    print(f"Generated {len(configs)} experiment configurations")
    print(f"  Model: {args.model}")
    print(f"  Dataset types: {args.dataset_types}")
    print(f"  Distractor sources: {args.distractor_source}")
    print(f"  Output base: {output_base}")

    if args.dry_run:
        print("\n=== DRY RUN - Configs to execute ===\n")
        for i, cfg in enumerate(configs):
            skip_marker = ""
            if args.skip_existing and (cfg.output_dir / "results.json").exists():
                skip_marker = " [SKIP - exists]"
            print(
                f"  [{i+1:3d}] {cfg.name} | "
                f"{cfg.distractor_config_str} | "
                f"type={cfg.dataset_type_filter} | "
                f"source={cfg.distractor_source}{skip_marker}"
            )
        print(f"\nTotal: {len(configs)} configs")
        return 0

    # Filter out existing results if requested
    if args.skip_existing:
        original_count = len(configs)
        configs = [c for c in configs if not (c.output_dir / "results.json").exists()]
        skipped = original_count - len(configs)
        if skipped > 0:
            print(f"Skipping {skipped} experiments with existing results")

    if not configs:
        print("No experiments to run.")
        return 0

    print(f"\nRunning {len(configs)} experiments...")

    # Run experiments
    summaries = []
    if args.parallel > 1:
        # Suppress tqdm progress bars in parallel mode to avoid thread-safety crashes
        for cfg in configs:
            cfg._quiet = True
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(run_single_config, cfg): cfg for cfg in configs}
            for future in as_completed(futures):
                summary = future.result()
                summaries.append(summary)
                status = summary["status"]
                if status == "success":
                    print(f"  Completed: {summary['name']} -> {summary['accuracy']:.2%}")
                else:
                    print(f"  FAILED: {summary['name']} -> {status}")
    else:
        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] {config.name}")
            summary = run_single_config(config)
            summaries.append(summary)
            status = summary["status"]
            if status == "success":
                print(f"  Accuracy: {summary['accuracy']:.2%}")
            else:
                print(f"  FAILED: {status}")

    # Save batch summary
    batch_summary_path = output_base / f"batch_summary_{args.model.replace('/', '_')}.json"
    batch_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(batch_summary_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "total_configs": len(summaries),
                "successful": sum(1 for s in summaries if s["status"] == "success"),
                "failed": sum(1 for s in summaries if s["status"] != "success"),
                "results": summaries,
            },
            f,
            indent=2,
        )
    print(f"\nBatch summary saved to: {batch_summary_path}")

    # Print final summary
    successful = [s for s in summaries if s["status"] == "success"]
    failed = [s for s in summaries if s["status"] != "success"]
    print(f"\n=== Batch Complete ===")
    print(f"  Successful: {len(successful)}/{len(summaries)}")
    if failed:
        print(f"  Failed: {len(failed)}")
        for f_summary in failed:
            print(f"    - {f_summary['name']}: {f_summary['status']}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
