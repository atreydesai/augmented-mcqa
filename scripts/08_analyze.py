#!/usr/bin/env python3
"""Analyze experiment results and generate plots.

Subcommands:
    table  - Generate consolidated behavioral signature tables
    plot   - Generate Final5 pairwise plots and summary CSVs

Usage:
    python scripts/08_analyze.py table --dir results/
    python scripts/08_analyze.py plot --results-root results --output-dir results/final5_plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.analyzer import analyze_experiment, format_signature_table
from analysis.visualize import plot_final5_pairwise


def cmd_table(args: argparse.Namespace) -> int:
    base_dir = Path(args.dir)

    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        return 1

    exp_files = sorted(base_dir.glob("**/*.eval"))

    if not exp_files:
        print(f"No experiment results found in {base_dir}")
        return 0

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

    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    import main as app_main

    argv = ["analyze", "--results-root", args.results_root, "--output-dir", args.output_dir]
    if args.skip_tables:
        argv.append("--skip-tables")
    return app_main.main(argv)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze experiment results and generate plots"
    )
    subparsers = parser.add_subparsers(dest="command")

    # table subcommand
    tbl = subparsers.add_parser("table", help="Generate consolidated behavioral signature tables")
    tbl.add_argument("--dir", type=str, required=True, help="Directory containing experiment results")
    tbl.add_argument("--output", type=str, help="Output file for the table (optional)")
    tbl.set_defaults(handler=cmd_table)

    # plot subcommand
    plt_cmd = subparsers.add_parser("plot", help="Generate Final5 pairwise plots and summary CSVs")
    plt_cmd.add_argument("--results-root", type=str, default="results")
    plt_cmd.add_argument("--output-dir", type=str, default="results/final5_plots")
    plt_cmd.add_argument("--skip-tables", action="store_true")
    plt_cmd.set_defaults(handler=cmd_plot)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
