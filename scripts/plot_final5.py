#!/usr/bin/env python3
"""Generate Final5 pairwise plots and summary CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.visualize import plot_final5_pairwise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Final5 pairwise comparisons")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results/final5_plots")
    parser.add_argument("--skip-tables", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = plot_final5_pairwise(
        results_root=Path(args.results_root),
        output_dir=Path(args.output_dir),
        include_tables=not args.skip_tables,
    )

    if not outputs:
        print("No Final5 results found; nothing plotted.")
        return 1

    print("Generated outputs:")
    for out in outputs:
        print(f"  - {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
