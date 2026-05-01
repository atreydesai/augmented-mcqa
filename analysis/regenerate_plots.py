"""Regenerate Augmented MCQA IRT tables and figures from collected datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.irt import run_irt_analysis


DEFAULT_COLLECTED_ROOT = REPO_ROOT / "datasets" / "collected"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "augmented_mcqa_irt"


def csv_values(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return values or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate IRT tables and plots from collected datasets.")
    parser.add_argument("--collected-root", default=str(DEFAULT_COLLECTED_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--generators", default=None, help="Optional comma-separated generator model filter.")
    parser.add_argument("--evaluators", default=None, help="Optional comma-separated evaluator model filter.")
    parser.add_argument("--datasets", default=None, help="Optional comma-separated dataset filter.")
    parser.add_argument("--settings", default=None, help="Optional comma-separated setting filter.")
    parser.add_argument("--modes", default="full_question", help="Comma-separated mode filter. Defaults to full_question.")
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--maxfun", type=int, default=50000)
    parser.add_argument("--gtol", type=float, default=1e-5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = run_irt_analysis(
        collected_root=Path(args.collected_root),
        output_dir=Path(args.output_dir),
        generators=csv_values(args.generators),
        evaluators=csv_values(args.evaluators),
        datasets=csv_values(args.datasets),
        settings=csv_values(args.settings),
        modes=csv_values(args.modes),
        maxiter=args.maxiter,
        maxfun=args.maxfun,
        gtol=args.gtol,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
