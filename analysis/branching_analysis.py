
#Branching Human Distractor Analysis.
#benefit of adding 1 vs 2 vs 3 human distractors
#with branching lines showing model distractor additions off each base.

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR


HUMAN_COLORS = {
    1: "#e74c3c",  # Red for 1H
    2: "#f39c12",  # Orange for 2H
    3: "#27ae60",  # Green for 3H
}


LINE_STYLES = {
    1: "-",
    2: "--",
    3: "-.",
}


def load_branching_results(
    base_dir: Path,
    model: Optional[str] = None,
    dataset_type: Optional[str] = None,
    distractor_source: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    #new directory structure:
    #  base_dir/{model}_{dataset_type}_{distractor_source}/{nH}{mM}/results.json
    results = {}

    for h in range(1, 4):  # 1H, 2H, 3H
        for m in range(0, 7):  # 0M to 6M
            config = f"{h}H{m}M"

            if model and dataset_type and distractor_source:
                dir_name = f"{model.replace('/', '_')}_{dataset_type}_{distractor_source}"
                results_path = base_dir / dir_name / config / "results.json"
            else:
                results_path = base_dir / config / "results.json"

            if not results_path.exists():
                continue

            with open(results_path) as f:
                data = json.load(f)

            summary = data.get("summary", {})
            results[config] = {
                "accuracy": summary.get("accuracy", 0),
                "correct": summary.get("correct", 0),
                "total": summary.get("total", 0),
            }

    return results


def plot_human_distractor_branching(
    base_dir: Path,
    model: Optional[str] = None,
    dataset_type: Optional[str] = None,
    distractor_source: Optional[str] = None,
    output_dir: Optional[Path] = None,
    show: bool = False,
) -> Dict[str, Path]:
    base_dir = Path(base_dir)
    if output_dir is None:
        output_dir = base_dir / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}

    results = load_branching_results(base_dir, model, dataset_type, distractor_source)

    if not results:
        print("No results found for branching analysis")
        return output_paths

    fig, ax = plt.subplots(figsize=(14, 9))

    # Plot branching lines for each human baseline
    for h in [1, 2, 3]:
        color = HUMAN_COLORS[h]
        style = LINE_STYLES[h]

        # Collect data points
        x_vals = []
        y_vals = []

        for m in range(0, 7):
            config = f"{h}H{m}M"
            if config in results:
                x_vals.append(h + m)  # Total distractors
                y_vals.append(results[config]["accuracy"])

        if x_vals:
            ax.plot(
                x_vals, y_vals,
                marker='o', linewidth=2.5, markersize=10,
                color=color, linestyle=style,
                label=f'{h}H + M (base: {h} human)',
                alpha=0.9,
            )

            # Add annotation at first point
            if y_vals:
                ax.annotate(
                    f'{h}H',
                    (x_vals[0], y_vals[0]),
                    textcoords="offset points",
                    xytext=(-15, 10),
                    fontsize=11,
                    fontweight='bold',
                    color=color,
                )

    # Styling
    title_parts = ["Benefit of Human Distractors with Model Additions"]
    if dataset_type:
        title_parts.append(f"Dataset: {dataset_type}")
    if distractor_source:
        title_parts.append(f"Source: {distractor_source}")
    if model:
        title_parts.append(f"Model: {model}")

    ax.set_xlabel('Total Number of Distractors', fontsize=16)
    ax.set_ylabel('Accuracy', fontsize=16)
    ax.set_title('\n'.join(title_parts), fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()

    # Save
    suffix_parts = [p for p in [dataset_type, distractor_source] if p]
    suffix = "_".join(suffix_parts) if suffix_parts else "all"
    output_path = output_dir / f"branching_{suffix}.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    output_paths["branching"] = output_path
    print(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()

    return output_paths


def plot_human_benefit_comparison(
    base_dir: Path,
    model: Optional[str] = None,
    dataset_type: Optional[str] = None,
    distractor_source: Optional[str] = None,
    output_dir: Optional[Path] = None,
    show: bool = False,
) -> Optional[Path]:
    ## degradation comparison showing how accuracy drops with model distractors
    base_dir = Path(base_dir)
    if output_dir is None:
        output_dir = base_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_branching_results(base_dir, model, dataset_type, distractor_source)
    if not results:
        print("No results found for benefit comparison")
        return None

    fig, ax = plt.subplots(figsize=(12, 7))

    # Calculate degradation rates
    for h in [1, 2, 3]:
        baseline = results.get(f"{h}H0M", {}).get("accuracy", 0)

        degradation = []
        m_vals = []

        for m in range(0, 7):
            config = f"{h}H{m}M"
            if config in results:
                acc = results[config]["accuracy"]
                deg = baseline - acc if baseline else 0
                degradation.append(deg)
                m_vals.append(m)

        if degradation:
            ax.plot(
                m_vals, degradation,
                marker='s', linewidth=2.5, markersize=9,
                color=HUMAN_COLORS[h],
                linestyle=LINE_STYLES[h],
                label=f'{h}H baseline',
            )

    title_parts = ["Accuracy Degradation with Model Distractors"]
    if dataset_type:
        title_parts.append(f"Dataset: {dataset_type}")
    if distractor_source:
        title_parts.append(f"Source: {distractor_source}")

    ax.set_xlabel('Number of Model Distractors Added', fontsize=14)
    ax.set_ylabel('Accuracy Drop from Baseline', fontsize=14)
    ax.set_title('\n'.join(title_parts), fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 6.5)
    plt.tight_layout()

    suffix_parts = [p for p in [dataset_type, distractor_source] if p]
    suffix = "_".join(suffix_parts) if suffix_parts else "all"
    output_path = output_dir / f"human_benefit_degradation_{suffix}.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Branching human distractor analysis")
    parser.add_argument("--results-dir", type=str, required=True, help="Results directory")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--dataset-type", type=str, help="Dataset type (e.g., mmlu_pro)")
    parser.add_argument("--distractor-source", type=str, help="Distractor source (e.g., scratch)")
    parser.add_argument("--output-dir", type=str, help="Output directory for plots")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    print("Creating branching analysis plots...")
    plot_human_distractor_branching(
        results_dir, args.model, args.dataset_type, args.distractor_source,
        output_dir, args.show,
    )

    print("\nCreating benefit comparison plot...")
    plot_human_benefit_comparison(
        results_dir, args.model, args.dataset_type, args.distractor_source,
        output_dir, args.show,
    )
