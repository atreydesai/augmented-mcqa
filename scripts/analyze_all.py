#!/usr/bin/env python3
"""
Aggregate and Analyze MCQA Experiment Results.

Finds all experiment results in a directory and generates a consolidated 
behavioral signature table.
"""

import argparse
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.analyzer import analyze_experiment, format_signature_table


def main():
    parser = argparse.ArgumentParser(description="MCQA Results Aggregator")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing experiment results")
    parser.add_argument("--output", type=str, help="Output file for the table (optional)")
    
    args = parser.parse_args()
    base_dir = Path(args.dir)
    
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        sys.exit(1)
    
    # Find all experiment directories containing summary.json or results.json.
    exp_files = []
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        if (p / "summary.json").exists():
            exp_files.append(p / "summary.json")
        elif (p / "results.json").exists():
            exp_files.append(p / "results.json")

    if not exp_files:
        # Check nested subdirectories.
        summary_files = sorted(base_dir.glob("**/summary.json"))
        summary_parents = {p.parent for p in summary_files}
        result_files = [
            p for p in sorted(base_dir.glob("**/results.json"))
            if p.parent not in summary_parents
        ]
        exp_files.extend(summary_files)
        exp_files.extend(result_files)

    if not exp_files:
        print(f"No experiment results found in {base_dir}")
        sys.exit(0)

    print(f"Found {len(exp_files)} experiments. Analyzing...")
    
    results = {}
    for exp_file in sorted(exp_files):
        try:
            analysis = analyze_experiment(exp_file)
            name = exp_file.parent.name
            results[name] = {
                "signature": analysis["overall"]["signature"],
                "gold_rate": analysis["overall"]["gold_rate"]
            }
        except Exception as e:
            print(f"Warning: Failed to analyze {exp_file.parent.name}: {e}")
            
    # Format table
    table = format_signature_table(results, include_counts=True)
    
    print("\nConsolidated Behavioral Signatures:")
    print(table)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(table)
        print(f"\nTable saved to {args.output}")


if __name__ == "__main__":
    main()
