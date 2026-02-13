#!/usr/bin/env python3
"""
Unified Orchestration Script for MCQA Experiments.

Runs the 3 core experiments:
1. RQ1: 3H + M variants (M=0 to 6)
2. RQ2: Human-only variants (H=1 to 3, M=0)
3. RQ3: Model-only variants (M=1 to 6, H=0)

Each experiment is run across:
- MMLU-Pro (Normal) and MMLU-Aug (Augmented) datasets
- Full Question and Choices-Only prompt templates
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATASETS_DIR, RESULTS_DIR


def run_command(cmd, name):
    """Run a shell command and print status."""
    print(f"Starting: {name}")
    try:
        subprocess.run(cmd, check=True, shell=True)
        print(f"✓ {name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {name} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Unified MCQA Experiment Runner")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gpt-4.1-2025-04-14)")
    parser.add_argument("--name", type=str, required=True, help="Parent experiment name (e.g., 2025-02-13)")
    parser.add_argument("--limit", type=int, help="Limit entries per experiment")
    parser.add_argument("--max-threads", type=int, default=10, help="Max parallel experiments")
    parser.add_argument("--dataset-normal", type=str, default="datasets/processed/mmlu_pro_processed", 
                        help="Path to normal MMLU-Pro dataset")
    parser.add_argument("--dataset-augmented", type=str, 
                        default="datasets/augmented/unified_processed_gpt-4.1_20260213_033916",
                        help="Path to augmented MMLU-Aug dataset")
    
    args = parser.parse_args()

    # Paths
    normal_ds = args.dataset_normal
    aug_ds = args.dataset_augmented
    parent_res_dir = f"{args.name}"

    commands = []

    # =========================================================================
    # Experiment 1: 3H + M Variants (RQ1)
    # =========================================================================
    for m in range(7):  # 0 to 6 model distractors
        for ds_type, ds_path in [("normal", normal_ds), ("augmented", aug_ds)]:
            for choices_only in [False, True]:
                suffix = "_choices_only" if choices_only else ""
                exp_name = f"3H{m}M_{ds_type}{suffix}"
                
                cmd = f"python scripts/run_experiment.py --name {parent_res_dir}/{exp_name} " \
                      f"--dataset {ds_path} --model {args.model} " \
                      f"--num-human 3 --num-model {m} --eval-mode behavioral "
                
                if choices_only:
                    cmd += "--choices-only "
                if args.limit:
                    cmd += f"--limit {args.limit} "
                
                commands.append((cmd, exp_name))

    # =========================================================================
    # Experiment 2: Human-only (RQ2)
    # =========================================================================
    for h in range(1, 4):  # 1 to 3 human distractors
        for ds_type, ds_path in [("normal", normal_ds), ("augmented", aug_ds)]:
            for choices_only in [False, True]:
                suffix = "_choices_only" if choices_only else ""
                exp_name = f"{h}H0M_{ds_type}{suffix}"
                
                cmd = f"python scripts/run_experiment.py --name {parent_res_dir}/{exp_name} " \
                      f"--dataset {ds_path} --model {args.model} " \
                      f"--num-human {h} --num-model 0 --eval-mode behavioral "
                
                if choices_only:
                    cmd += "--choices-only "
                if args.limit:
                    cmd += f"--limit {args.limit} "
                
                commands.append((cmd, exp_name))

    # =========================================================================
    # Experiment 3: Model-only (RQ3)
    # =========================================================================
    for m in range(1, 7):  # 1 to 6 model distractors
        for ds_type, ds_path in [("normal", normal_ds), ("augmented", aug_ds)]:
            for choices_only in [False, True]:
                suffix = "_choices_only" if choices_only else ""
                exp_name = f"0H{m}M_{ds_type}{suffix}"
                
                cmd = f"python scripts/run_experiment.py --name {parent_res_dir}/{exp_name} " \
                      f"--dataset {ds_path} --model {args.model} " \
                      f"--num-human 0 --num-model {m} --eval-mode behavioral "
                
                if choices_only:
                    cmd += "--choices-only "
                if args.limit:
                    cmd += f"--limit {args.limit} "
                
                commands.append((cmd, exp_name))

    print(f"Total experiments to run: {len(commands)}")
    print(f"Max parallel threads: {args.max_threads}")

    # Run experiments in parallel
    with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        futures = [executor.submit(run_command, cmd, name) for cmd, name in commands]
        results = [f.result() for f in futures]

    success_count = sum(1 for r in results if r)
    print(f"\nCompleted {success_count}/{len(commands)} experiments successfully.")
    
    if success_count == len(commands):
        print("\nAll experiments finished. You can now run visualization.")
        print(f"Command: python -c \"from analysis.visualize import plot_all_rq; from pathlib import Path; plot_all_rq(Path('results/{args.name}'))\"")
    else:
        print("\nSome experiments failed. Please check the logs.")


if __name__ == "__main__":
    main()
