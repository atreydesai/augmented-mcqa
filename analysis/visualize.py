"""
Visualization module for Augmented MCQA.

Generates research question plots for the unified dataset with multiple
dataset_types (mmlu_pro, supergpqa, arc_easy, arc_challenge) and
distractor sources (scratch, dhuman, dmodel).

New directory structure:
  results/{model}_{dataset_type}_{distractor_source}/{nH}{mM}/results.json

Supports both the new ExperimentResults format and legacy summary files.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt


# Dataset type display names and colors
DATASET_TYPE_STYLES = {
    "mmlu_pro": {"label": "MMLU-Pro", "color": "#3498db", "marker": "o"},
    "supergpqa": {"label": "SuperGPQA", "color": "#e74c3c", "marker": "s"},
    "arc_easy": {"label": "ARC-Easy", "color": "#2ecc71", "marker": "^"},
    "arc_challenge": {"label": "ARC-Challenge", "color": "#9b59b6", "marker": "D"},
}

# Distractor source display names
DISTRACTOR_SOURCE_LABELS = {
    "scratch": "From Scratch",
    "dhuman": "Cond. Human",
    "dmodel": "Cond. Model",
}


def load_results_file(results_path: Path) -> Optional[Dict]:
    """
    Load a results.json file in the new ExperimentResults format.

    Returns dict with accuracy, correct, total, or None if file missing.
    """
    if not results_path.exists():
        return None

    with open(results_path, "r") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    return {
        "accuracy": summary.get("accuracy", 0),
        "correct": summary.get("correct", 0),
        "total": summary.get("total", 0),
    }


def load_summary_file(base_dir: Path, setting_name: str, summary_filename: str) -> Optional[Dict]:
    """Load a single summary file (legacy format) and return accuracy data."""
    summary_path = base_dir / setting_name / summary_filename
    if not summary_path.exists():
        return None

    with open(summary_path, "r") as f:
        summary = json.load(f)

    # Support both old and new formats
    return {
        "accuracy": summary.get("acc", summary.get("accuracy", 0)),
        "correct": summary.get("corr", summary.get("correct", 0)),
        "wrong": summary.get("wrong", summary.get("total", 0) - summary.get("correct", 0)),
    }


def _build_results_dir_name(model: str, dataset_type: str, distractor_source: str) -> str:
    """Build the directory name pattern for results."""
    return f"{model.replace('/', '_')}_{dataset_type}_{distractor_source}"


def load_3H_plus_M_results(
    base_dir: Path,
    model: Optional[str] = None,
    dataset_type: Optional[str] = None,
    distractor_source: Optional[str] = None,
) -> List[Dict]:
    """
    Load 3H + M results (3H0M through 3H6M).

    Supports new directory structure:
      base_dir/{model}_{dataset_type}_{distractor_source}/{nH}{mM}/results.json

    If model/dataset_type/distractor_source are None, searches directly under base_dir.
    """
    results = []

    for m in range(0, 7):
        config_str = f"3H{m}M"

        if model and dataset_type and distractor_source:
            dir_name = _build_results_dir_name(model, dataset_type, distractor_source)
            results_path = base_dir / dir_name / config_str / "results.json"
        else:
            results_path = base_dir / config_str / "results.json"

        data = load_results_file(results_path)
        if data:
            results.append({
                "setting": config_str,
                "num_human": 3,
                "num_model": m,
                "total_distractors": 3 + m,
                **data,
            })

    return results


def load_human_only_results(
    base_dir: Path,
    model: Optional[str] = None,
    dataset_type: Optional[str] = None,
    distractor_source: Optional[str] = None,
) -> List[Dict]:
    """Load Human Only results (1H0M, 2H0M, 3H0M)."""
    results = []

    for h in range(1, 4):
        config_str = f"{h}H0M"

        if model and dataset_type and distractor_source:
            dir_name = _build_results_dir_name(model, dataset_type, distractor_source)
            results_path = base_dir / dir_name / config_str / "results.json"
        else:
            results_path = base_dir / config_str / "results.json"

        data = load_results_file(results_path)
        if data:
            results.append({
                "setting": config_str,
                "num_human": h,
                "num_model": 0,
                "total_distractors": h,
                **data,
            })

    return results


def load_model_only_results(
    base_dir: Path,
    model: Optional[str] = None,
    dataset_type: Optional[str] = None,
    distractor_source: Optional[str] = None,
) -> List[Dict]:
    """Load Model Only results (0H1M through 0H6M)."""
    results = []

    for m in range(1, 7):
        config_str = f"0H{m}M"

        if model and dataset_type and distractor_source:
            dir_name = _build_results_dir_name(model, dataset_type, distractor_source)
            results_path = base_dir / dir_name / config_str / "results.json"
        else:
            results_path = base_dir / config_str / "results.json"

        data = load_results_file(results_path)
        if data:
            results.append({
                "setting": config_str,
                "num_human": 0,
                "num_model": m,
                "total_distractors": m,
                **data,
            })

    return results


def _detect_available_configs(
    base_dir: Path,
    model: str,
) -> Dict[str, List[str]]:
    """
    Auto-detect available dataset_type and distractor_source combinations.

    Returns dict mapping dataset_type -> list of distractor_sources found.
    """
    available = {}
    model_safe = model.replace("/", "_")
    prefix = f"{model_safe}_"

    if not base_dir.exists():
        return available

    for d in sorted(base_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        remainder = d.name[len(prefix):]
        # Parse: dataset_type_distractor_source
        for source in ["scratch", "dhuman", "dmodel"]:
            if remainder.endswith(f"_{source}"):
                dt = remainder[: -(len(source) + 1)]
                available.setdefault(dt, []).append(source)
                break

    return available


def plot_rq1_combined(
    base_dir: Path,
    model: str,
    distractor_source: str = "scratch",
    output_dir: Optional[Path] = None,
    show: bool = False,
):
    """
    RQ1: Combined plot showing 3H+M, Human-Only, Model-Only across dataset_types.

    One plot per distractor_source with lines per dataset_type.
    """
    base_dir = Path(base_dir)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    experiment_loaders = [
        ("3H + M Progressive", load_3H_plus_M_results, axes[0]),
        ("Human Only", load_human_only_results, axes[1]),
        ("Model Only", load_model_only_results, axes[2]),
    ]

    for title, loader, ax in experiment_loaders:
        has_data = False
        for dt, style in DATASET_TYPE_STYLES.items():
            results = loader(base_dir, model, dt, distractor_source)
            if results:
                has_data = True
                x = [r["total_distractors"] for r in results]
                y = [r["accuracy"] for r in results]
                ax.plot(
                    x, y,
                    marker=style["marker"], linewidth=2.5, markersize=9,
                    label=style["label"], color=style["color"],
                )

        ax.set_xlabel("Total Number of Distractors", fontsize=14)
        ax.set_ylabel("Accuracy", fontsize=14)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.legend(loc="best", fontsize=11, frameon=True)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=11)
        if not has_data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", fontsize=14)

    source_label = DISTRACTOR_SOURCE_LABELS.get(distractor_source, distractor_source)
    fig.suptitle(
        f"RQ1: Accuracy vs Distractors ({source_label})\nModel: {model}",
        fontsize=18, fontweight="bold", y=1.04,
    )
    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"rq1_{distractor_source}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()


def plot_rq2_human_distractors(
    base_dir: Path,
    model: str,
    output_dir: Optional[Path] = None,
    show: bool = False,
):
    """
    RQ2: Human-only distractor comparison across dataset_types.

    Shows how accuracy changes from 1H to 3H for each dataset_type.
    One subplot per distractor_source.
    """
    base_dir = Path(base_dir)
    sources = ["scratch", "dhuman", "dmodel"]

    fig, axes = plt.subplots(1, len(sources), figsize=(7 * len(sources), 7))
    if len(sources) == 1:
        axes = [axes]

    for ax, source in zip(axes, sources):
        has_data = False
        for dt, style in DATASET_TYPE_STYLES.items():
            results = load_human_only_results(base_dir, model, dt, source)
            if results:
                has_data = True
                x = [r["total_distractors"] for r in results]
                y = [r["accuracy"] for r in results]
                ax.plot(
                    x, y,
                    marker=style["marker"], linewidth=3, markersize=12,
                    label=style["label"], color=style["color"], alpha=0.8,
                )

        source_label = DISTRACTOR_SOURCE_LABELS.get(source, source)
        ax.set_xlabel("Number of Human Distractors", fontsize=14)
        ax.set_ylabel("Accuracy", fontsize=14)
        ax.set_title(f"Human Only ({source_label})", fontsize=15, fontweight="bold")
        ax.legend(loc="best", fontsize=12, frameon=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 3.5)
        ax.set_xticks([1, 2, 3])
        ax.tick_params(axis="both", labelsize=12)
        if not has_data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", fontsize=14)

    fig.suptitle(
        f"RQ2: Human-Only Distractor Analysis\nModel: {model}",
        fontsize=18, fontweight="bold", y=1.04,
    )
    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "rq2_human_only.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()


def plot_rq3_model_distractors(
    base_dir: Path,
    model: str,
    output_dir: Optional[Path] = None,
    show: bool = False,
):
    """
    RQ3: Model-only distractor comparison across dataset_types.

    Shows how accuracy changes from 0H+1M to 0H+6M for each dataset_type.
    One subplot per distractor_source.
    """
    base_dir = Path(base_dir)
    sources = ["scratch", "dhuman", "dmodel"]

    fig, axes = plt.subplots(1, len(sources), figsize=(7 * len(sources), 7))
    if len(sources) == 1:
        axes = [axes]

    for ax, source in zip(axes, sources):
        has_data = False
        for dt, style in DATASET_TYPE_STYLES.items():
            results = load_model_only_results(base_dir, model, dt, source)
            if results:
                has_data = True
                x = [r["total_distractors"] for r in results]
                y = [r["accuracy"] for r in results]
                ax.plot(
                    x, y,
                    marker=style["marker"], linewidth=3, markersize=12,
                    label=style["label"], color=style["color"], alpha=0.8,
                )

        source_label = DISTRACTOR_SOURCE_LABELS.get(source, source)
        ax.set_xlabel("Number of Model Distractors", fontsize=14)
        ax.set_ylabel("Accuracy", fontsize=14)
        ax.set_title(f"Model Only ({source_label})", fontsize=15, fontweight="bold")
        ax.legend(loc="best", fontsize=12, frameon=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 6.5)
        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.tick_params(axis="both", labelsize=12)
        if not has_data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", fontsize=14)

    fig.suptitle(
        f"RQ3: Model-Only Distractor Analysis\nModel: {model}",
        fontsize=18, fontweight="bold", y=1.04,
    )
    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "rq3_model_only.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()


def plot_all_rq(
    base_dir: Path,
    model: str = "",
    output_dir: Optional[Path] = None,
    show: bool = False,
):
    """
    Generate all RQ plots.

    Args:
        base_dir: Directory containing experiment results
        model: Model name used in directory structure. If empty, auto-detect.
        output_dir: Where to save plots (default: base_dir/plots)
        show: Whether to display plots interactively
    """
    base_dir = Path(base_dir)

    if output_dir is None:
        output_dir = base_dir / "plots"

    # Auto-detect model if not specified
    if not model:
        # Try to find model from directory names
        for d in base_dir.iterdir():
            if d.is_dir() and "_" in d.name:
                model = d.name.split("_")[0]
                break
        if not model:
            print("Could not auto-detect model name. Please provide --model.")
            return

    # Auto-detect available distractor sources
    available = _detect_available_configs(base_dir, model)
    if not available:
        print(f"No results found for model={model} in {base_dir}")
        return

    all_sources = set()
    for sources in available.values():
        all_sources.update(sources)

    print(f"Detected model: {model}")
    print(f"Detected dataset types: {list(available.keys())}")
    print(f"Detected distractor sources: {sorted(all_sources)}")

    for source in sorted(all_sources):
        print(f"\nGenerating RQ1 for distractor source: {source}...")
        plot_rq1_combined(base_dir, model, source, output_dir, show)

    print("\nGenerating RQ2: Human-Only Distractor Analysis...")
    plot_rq2_human_distractors(base_dir, model, output_dir, show)

    print("Generating RQ3: Model-Only Distractor Analysis...")
    plot_rq3_model_distractors(base_dir, model, output_dir, show)

    print(f"\nAll plots saved to: {output_dir}")
