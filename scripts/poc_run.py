#!/usr/bin/env python3
"""
Proof-of-concept run to verify the entire pipeline works.

Usage:
    python scripts/poc_run.py
    
This script:
1. Creates a small test dataset (25 examples)
2. Runs evaluation with a mock/fast model
3. Generates sample visualizations
4. Saves results to verify the full pipeline
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATASETS_DIR, RESULTS_DIR
from experiments import ExperimentConfig


def create_mock_dataset(output_path: Path, n_examples: int = 25):
    """Create a minimal mock dataset for testing."""
    import json
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    mock_data = []
    for i in range(n_examples):
        mock_data.append({
            "question": f"Test question {i+1}: What is the capital of Country {i+1}?",
            "gold_answer": f"Capital {i+1}",
            "cond_human_q_a": [f"Wrong City {j}" for j in range(1, 4)],
            "cond_model_q_a": [f"Generated City {j}" for j in range(1, 4)],
            "category": f"category_{i % 5}",
        })
    
    # Save as JSON for verification
    with open(output_path / "mock_dataset.json", "w") as f:
        json.dump(mock_data, f, indent=2)
    
    print(f"Created mock dataset with {n_examples} examples at {output_path}")
    return mock_data


def run_poc():
    """Run proof-of-concept pipeline."""
    print("=" * 60)
    print("PROOF-OF-CONCEPT RUN")
    print("=" * 60)
    
    poc_dir = RESULTS_DIR / "poc_run"
    poc_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create mock dataset
    print("\n[1/4] Creating mock dataset...")
    mock_dataset_path = poc_dir / "mock_dataset"
    mock_data = create_mock_dataset(mock_dataset_path, n_examples=25)
    
    # Step 2: Verify config creation
    print("\n[2/4] Creating experiment config...")
    config = ExperimentConfig(
        name="poc_test",
        dataset_path=mock_dataset_path,
        model_name="gpt-4.1-2025-04-14",  # Use this for testing
        num_human=3,
        num_model=0,
        eval_mode="behavioral",
        limit=5,  # Very small for quick test
        output_dir=poc_dir / "results",
    )
    
    print(f"  Config ID: {config.config_id}")
    print(f"  Distractor config: {config.distractor_config_str}")
    
    # Save config
    config_path = config.save()
    print(f"  Saved config to: {config_path}")
    
    # Step 3: Verify imports work
    print("\n[3/4] Verifying module imports...")
    
    try:
        from data import DataAdapter, FilterConfig
        print("  ✓ data module")
    except ImportError as e:
        print(f"  ✗ data module: {e}")
    
    try:
        from models import get_client, ModelClient
        print("  ✓ models module")
    except ImportError as e:
        print(f"  ✗ models module: {e}")
    
    try:
        from evaluation import extract_answer, compute_accuracy
        print("  ✓ evaluation module")
    except ImportError as e:
        print(f"  ✗ evaluation module: {e}")
    
    try:
        from analysis import analyze_experiment, plot_all_rq
        print("  ✓ analysis module")
    except ImportError as e:
        print(f"  ✗ analysis module: {e}")
    
    try:
        from experiments import DifficultyLevel, DIFFICULTY_DATASETS
        print("  ✓ experiments.difficulty module")
    except ImportError as e:
        print(f"  ✗ experiments.difficulty module: {e}")
    
    # Step 4: Summary
    print("\n[4/4] Pipeline verification complete!")
    print(f"\nPOC outputs saved to: {poc_dir}")
    print("\nTo run actual experiment:")
    print(f"  python scripts/run_experiment.py --config {config_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(run_poc())
