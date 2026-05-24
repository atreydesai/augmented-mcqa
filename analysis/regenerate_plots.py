"""Regenerate Augmented MCQA IRT figures from cached tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.irt_figures import (
    plot_final_ablation_quality,
    plot_final_grouped_quality,
    plot_pareto_quality,
    plot_stacked_extended_summary,
    plot_stacked_quality_summary,
)
from analysis.irt_model import run_irt_analysis

DEFAULT_COLLECTED_ROOT = REPO_ROOT / "datasets" / "collected"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "augmented_mcqa_irt"


def csv_values(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return values or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate IRT plots from cached tables.")
    parser.add_argument("--collected-root", default=str(DEFAULT_COLLECTED_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--run-irt",
        action="store_true",
        help="Refit the IRT model and refresh cached tables before regenerating plots.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Deprecated: this script always regenerates plots from existing tables only.",
    )
    parser.add_argument("--generators", default=None, help="Optional comma-separated generator model filter.")
    parser.add_argument("--evaluators", default=None, help="Optional comma-separated evaluator model filter.")
    parser.add_argument("--datasets", default=None, help="Optional comma-separated dataset filter.")
    parser.add_argument("--settings", default=None, help="Optional comma-separated setting filter.")
    parser.add_argument("--modes", default="full_question", help="Comma-separated mode filter. Defaults to full_question.")
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--maxfun", type=int, default=50000)
    parser.add_argument("--gtol", type=float, default=1e-5)
    return parser


def regenerate_final_figures(output_dir: Path) -> list[Path]:
    tables = output_dir / "tables"
    figures = output_dir / "figures" / "final_figures"

    final_grouped = pd.read_csv(tables / "final_grouped_question_quality.csv")
    final_ablation = pd.read_csv(tables / "final_ablation_question_quality.csv")

    outputs = [
        # plot_final_grouped_quality(final_grouped, figures / "question_quality_grouped.png"),
        # plot_final_ablation_quality(final_ablation, figures / "ablation_quality.png"),
        # plot_stacked_quality_summary(
        #     final_grouped,
        #     final_ablation,
        #     figures / "quality_stacked_half.png",
        #     scale=0.5,
        # ),
        plot_stacked_extended_summary(
            final_grouped,
            final_ablation,
            figures / "quality_stacked_extended.png",
        ),
        plot_pareto_quality(
            final_grouped,
            final_ablation,
            figures / "quality_pareto_frontier.png",
        ),
        plot_pareto_quality(
            final_grouped,
            final_ablation,
            figures / "quality_pareto_frontier_with_ablation.png",
            include_ablation=True,
        ),
    ]
    return outputs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs: list[Path] = []
    if args.run_irt:
        outputs.extend(
            run_irt_analysis(
                collected_root=Path(args.collected_root),
                output_dir=Path(args.output_dir),
                generators=csv_values(args.generators),
                evaluators=csv_values(args.evaluators),
                datasets=csv_values(args.datasets),
                settings=csv_values(args.settings),
                modes=csv_values(args.modes) or ["full_question"],
                maxiter=int(args.maxiter),
                maxfun=int(args.maxfun),
                gtol=float(args.gtol),
            )
        )
    outputs.extend(regenerate_final_figures(Path(args.output_dir)))
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
