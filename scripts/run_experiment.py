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
from config import DistractorType
from experiments.defaults import (
    DEFAULT_EVAL_MAX_TOKENS,
    DEFAULT_EVAL_MODE,
    DEFAULT_EVAL_SEED,
    DEFAULT_EVAL_TEMPERATURE,
    DEFAULT_GENERATOR_DATASET_LABEL,
    DEFAULT_NUM_HUMAN_DISTRACTORS,
    DEFAULT_NUM_MODEL_DISTRACTORS,
)

MODEL_TYPE_CHOICES = [
    DistractorType.COND_MODEL_Q_A_SCRATCH.value,
    DistractorType.COND_MODEL_Q_A_DHUMAN.value,
    DistractorType.COND_MODEL_Q_A_DMODEL.value,
]


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
    parser.add_argument(
        "--generator-dataset-label",
        type=str,
        default=DEFAULT_GENERATOR_DATASET_LABEL,
        help="Generator dataset label used for output isolation",
    )
    
    # Distractor configuration
    parser.add_argument(
        "--num-human",
        type=int,
        default=DEFAULT_NUM_HUMAN_DISTRACTORS,
        help="Number of human distractors",
    )
    parser.add_argument(
        "--num-model",
        type=int,
        default=DEFAULT_NUM_MODEL_DISTRACTORS,
        help="Number of model distractors",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=MODEL_TYPE_CHOICES,
        default=DistractorType.COND_MODEL_Q_A_SCRATCH.value,
        help="Type of model distractors",
    )
    
    # Evaluation settings
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["accuracy", "behavioral"],
        default=DEFAULT_EVAL_MODE,
    )
    parser.add_argument("--choices-only", action="store_true", help="Use choices-only prompt")
    parser.add_argument("--limit", type=int, help="Limit number of entries")
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED, help="Random seed")
    
    # Model settings
    parser.add_argument("--reasoning-effort", type=str, help="OpenAI reasoning effort")
    parser.add_argument("--thinking-level", type=str, help="Anthropic/Gemini thinking level")
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_EVAL_TEMPERATURE,
        help="Sampling temperature (provider default if omitted)",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_EVAL_MAX_TOKENS, help="Max tokens")
    
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
            generator_dataset_label=args.generator_dataset_label,
            num_human=args.num_human,
            num_model=args.num_model,
            model_distractor_type=model_type,
            eval_mode=args.eval_mode,
            choices_only=args.choices_only,
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
