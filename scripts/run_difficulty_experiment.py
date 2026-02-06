#!/usr/bin/env python3
"""
Run difficulty scaling experiments.

Usage:
    python scripts/run_difficulty_experiment.py --level easy --model gpt-5.2-2025-12-11 --limit 25
    python scripts/run_difficulty_experiment.py --all --model gpt-5.2-2025-12-11 --limit 25
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.difficulty import (
    DifficultyLevel,
    DIFFICULTY_DATASETS,
    prepare_difficulty_evaluation,
    compute_difficulty_comparison,
    save_difficulty_results,
)
from experiments import ExperimentConfig, run_experiment
from config import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description="Run difficulty scaling experiments")
    
    parser.add_argument(
        "--level",
        type=str,
        choices=["easy", "medium", "hard"],
        help="Difficulty level to evaluate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all difficulty levels",
    )
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--limit", type=int, default=None, help="Limit entries per dataset")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--name", type=str, default="difficulty_scaling", help="Experiment name")
    
    # Model settings
    parser.add_argument("--reasoning-effort", type=str, help="OpenAI reasoning effort")
    parser.add_argument("--thinking-level", type=str, help="Anthropic/Gemini thinking level")
    
    args = parser.parse_args()
    
    if not args.all and not args.level:
        parser.error("Must specify --level or --all")
    
    # Determine levels to run
    if args.all:
        levels = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
    else:
        levels = [DifficultyLevel(args.level)]
    
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / args.name
    
    all_results = {}
    
    for level in levels:
        print(f"\n{'='*60}")
        print(f"Difficulty Level: {level.value.upper()}")
        print('='*60)
        
        # Get datasets for this difficulty
        datasets = prepare_difficulty_evaluation(level, limit=args.limit)
        
        if not datasets:
            print(f"No datasets found for difficulty: {level.value}")
            continue
        
        level_results = {}
        
        for dataset_name, entries in datasets.items():
            print(f"\n--- Dataset: {dataset_name} ({len(entries)} entries) ---")
            
            config = ExperimentConfig(
                name=f"{args.name}_{level.value}_{dataset_name}",
                dataset_path=Path(DIFFICULTY_DATASETS[dataset_name].path),
                model_name=args.model,
                num_human=3,
                num_model=0,
                limit=args.limit,
                reasoning_effort=args.reasoning_effort,
                thinking_level=args.thinking_level,
                output_dir=output_dir / level.value / dataset_name,
            )
            
            try:
                result = run_experiment(config)
                level_results[dataset_name] = {
                    "accuracy": result.accuracy,
                    "total": len(result.results),
                    "behavioral_counts": result.behavioral_counts,
                }
            except Exception as e:
                print(f"Error running {dataset_name}: {e}")
                level_results[dataset_name] = {"error": str(e)}
        
        all_results[level.value] = level_results
    
    # Compute comparison across difficulty levels
    comparison = compute_difficulty_comparison({
        level: {
            "accuracy": sum(d.get("accuracy", 0) for d in results.values()) / max(len(results), 1),
            "total": sum(d.get("total", 0) for d in results.values()),
        }
        for level, results in all_results.items()
    })
    
    all_results["comparison"] = comparison
    
    # Save overall results
    save_difficulty_results(all_results, args.name, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("DIFFICULTY SCALING SUMMARY")
    print("="*60)
    
    for level, results in all_results.items():
        if level == "comparison":
            continue
        print(f"\n{level.upper()}:")
        for dataset, data in results.items():
            if "error" in data:
                print(f"  {dataset}: ERROR - {data['error']}")
            else:
                print(f"  {dataset}: {data['accuracy']:.2%} ({data['total']} entries)")
    
    if "comparison" in all_results and "trends" in all_results["comparison"]:
        trends = all_results["comparison"]["trends"]
        if "accuracy_drop" in trends:
            print(f"\nAccuracy drop (easy → hard): {trends['accuracy_drop']:.2%}")
            print(f"Relative drop: {trends['relative_drop']:.2%}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
