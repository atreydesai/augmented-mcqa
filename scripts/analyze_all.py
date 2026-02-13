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
    
    # Find all directories containing results.json
    exp_dirs = []
    for p in base_dir.iterdir():
        if p.is_dir() and (p / "results.json").exists():
            exp_dirs.append(p)
    
    if not exp_dirs:
        # Check subdirectories if they are nested (e.g., results/parent/exp)
        for p in base_dir.glob("**/results.json"):
            exp_dirs.append(p.parent)
            
    if not exp_dirs:
        print(f"No experiment results found in {base_dir}")
        sys.exit(0)
        
    print(f"Found {len(exp_dirs)} experiments. Analyzing...")
    
    results = {}
    for exp_dir in sorted(exp_dirs):
        try:
            analysis = analyze_experiment(exp_dir / "results.json")
            name = exp_dir.name
            results[name] = {
                "signature": analysis["overall"]["signature"],
                "gold_rate": analysis["overall"]["gold_rate"]
            }
        except Exception as e:
            print(f"Warning: Failed to analyze {exp_dir.name}: {e}")
            
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
