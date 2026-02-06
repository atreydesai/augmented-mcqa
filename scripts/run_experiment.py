#!/usr/bin/env python3
"""
Run MCQA evaluation experiment.

Usage:
    python scripts/run_experiment.py --dataset datasets/mmlu_pro_sorted \
        --model gpt-5.2-2025-12-11 --name test_run --limit 25
        
    python scripts/run_experiment.py --config experiments/my_config.json
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments import ExperimentConfig, run_experiment
from config import DATASETS_DIR, RESULTS_DIR, DistractorType


def main():
    parser = argparse.ArgumentParser(description="Run MCQA evaluation experiment")
    
    # Config file mode
    parser.add_argument(
        "--config",
        type=str,
        help="Path to experiment config JSON file",
    )
    
    # Direct configuration mode
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument("--model", type=str, help="Model name")
    
    # Distractor configuration
    parser.add_argument("--num-human", type=int, default=3, help="Number of human distractors")
    parser.add_argument("--num-model", type=int, default=0, help="Number of model distractors")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["cond_model_q_a", "cond_model_q_a_dhuman", "cond_model_q_a_dmodel"],
        default="cond_model_q_a",
        help="Type of model distractors",
    )
    
    # Evaluation settings
    parser.add_argument("--eval-mode", type=str, choices=["accuracy", "behavioral"], default="behavioral")
    parser.add_argument("--limit", type=int, help="Limit number of entries")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model settings
    parser.add_argument("--reasoning-effort", type=str, help="OpenAI reasoning effort")
    parser.add_argument("--thinking-level", type=str, help="Anthropic/Gemini thinking level")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens")
    
    # Output
    parser.add_argument("--output-dir", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = ExperimentConfig.load(Path(args.config))
        print(f"Loaded config from: {args.config}")
    else:
        if not all([args.name, args.dataset, args.model]):
            parser.error("Must provide --config OR (--name, --dataset, --model)")
        
        model_type = DistractorType(args.model_type)
        
        config = ExperimentConfig(
            name=args.name,
            dataset_path=Path(args.dataset),
            model_name=args.model,
            num_human=args.num_human,
            num_model=args.num_model,
            model_distractor_type=model_type,
            eval_mode=args.eval_mode,
            limit=args.limit,
            seed=args.seed,
            reasoning_effort=args.reasoning_effort,
            thinking_level=args.thinking_level,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )
    
    # Run experiment
    results = run_experiment(config)
    
    print(f"\nFinal Accuracy: {results.accuracy:.2%}")
    if results.behavioral_counts:
        print(f"Behavioral Counts: {results.behavioral_counts}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
