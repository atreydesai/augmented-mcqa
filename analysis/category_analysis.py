import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR, DatasetType


# Category groupings for cleaner visualization
MMLU_PRO_CATEGORY_GROUPS = {
    "STEM": ["physics", "chemistry", "math", "computer science", "engineering"],
    "Social Sciences": ["economics", "psychology", "history", "law", "philosophy"],
    "Life Sciences": ["biology", "health"],
    "Business": ["business"],
    "Other": ["other"],
}

SUPERGPQA_DISCIPLINE_COLORS = {
    "Engineering": "#3498db",
    "Science": "#2ecc71",
    "Medicine": "#e74c3c",
    "Law": "#9b59b6",
    "Economics": "#f39c12",
    "Philosophy": "#1abc9c",
    "History": "#e67e22",
    "Other": "#95a5a6",
}

# Dataset type colors for breakdown plots
DATASET_TYPE_COLORS = {
    "mmlu_pro": "#3498db",
    "supergpqa": "#e74c3c",
    "arc_easy": "#2ecc71",
    "arc_challenge": "#9b59b6",
}


def load_results_with_categories(
    results_path: Path,
) -> Tuple[Dict[str, Any], List[Dict]]:
    with open(results_path) as f:
        data = json.load(f)

    summary = data.get("summary", {})
    entries = data.get("results", [])

    return summary, entries


def compute_accuracy_by_category(
    entries: List[Dict],
    category_key: str = "category",
) -> Dict[str, Dict[str, Any]]:
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})

    for entry in entries:
        cat = entry.get(category_key, "unknown")
        if not cat:
            cat = "unknown"

        by_category[cat]["total"] += 1
        if entry.get("is_correct", False):
            by_category[cat]["correct"] += 1

    # Compute accuracies
    results = {}
    for cat, data in by_category.items():
        if data["total"] > 0:
            results[cat] = {
                "accuracy": data["correct"] / data["total"],
                "correct": data["correct"],
                "total": data["total"],
            }

    return results


def compute_accuracy_by_dataset_type(
    base_dir: Path,
    model: str,
    config_str: str,
    distractor_sources: Optional[List[str]] = None,
    dataset_types: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if distractor_sources is None:
        distractor_sources = ["scratch", "dhuman", "dmodel"]
    if dataset_types is None:
        dataset_types = ["mmlu_pro", "supergpqa", "arc_easy", "arc_challenge"]

    results = {}
    model_safe = model.replace("/", "_")

    for dt in dataset_types:
        results[dt] = {}
        for source in distractor_sources:
            dir_name = f"{model_safe}_{dt}_{source}"
            results_path = base_dir / dir_name / config_str / "results.json"

            if not results_path.exists():
                continue

            with open(results_path) as f:
                data = json.load(f)

            summary = data.get("summary", {})
            results[dt][source] = {
                "accuracy": summary.get("accuracy", 0),
                "correct": summary.get("correct", 0),
                "total": summary.get("total", 0),
            }

    return results


def plot_category_breakdown(
    results_path: Path,
    category_key: str = "category",
    output_path: Optional[Path] = None,
    top_n: int = 15,
    min_samples: int = 10,
    figsize: Tuple[int, int] = (14, 8),
    show: bool = False,
) -> Optional[plt.Figure]:
    summary, entries = load_results_with_categories(results_path)
    by_cat = compute_accuracy_by_category(entries, category_key)

    # Filter and sort
    filtered = {
        k: v for k, v in by_cat.items()
        if v["total"] >= min_samples and k != "unknown"
    }
    sorted_cats = sorted(filtered.items(), key=lambda x: -x[1]["accuracy"])[:top_n]

    if not sorted_cats:
        print(f"No categories found with >= {min_samples} samples")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    categories = [c[0] for c in sorted_cats]
    accuracies = [c[1]["accuracy"] * 100 for c in sorted_cats]
    totals = [c[1]["total"] for c in sorted_cats]

    # Create bars with color gradient
    colors = plt.cm.RdYlGn(np.array(accuracies) / 100)
    bars = ax.barh(range(len(categories)), accuracies, color=colors, edgecolor='black')

    # Add sample counts
    for i, (bar, total) in enumerate(zip(bars, totals)):
        width = bar.get_width()
        ax.text(width + 1, i, f'n={total}', va='center', fontsize=10, color='gray')

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlabel('Accuracy (%)', fontsize=14)
    ax.set_title(f'Accuracy by {category_key.capitalize()}', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add overall accuracy line
    overall = summary.get("accuracy", 0) * 100
    if overall:
        ax.axvline(overall, color='red', linestyle='--', linewidth=2, label=f'Overall: {overall:.1f}%')
        ax.legend(loc='lower right', fontsize=12)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    return fig


def plot_category_heatmap(
    results_dir: Path,
    configs: List[str] = None,
    model: Optional[str] = None,
    dataset_type: Optional[str] = None,
    distractor_source: Optional[str] = None,
    category_key: str = "category",
    output_path: Optional[Path] = None,
    top_n: int = 12,
    show: bool = False,
) -> Optional[plt.Figure]:
    if configs is None:
        configs = ["3H0M", "3H3M", "0H3M"]

    # Load all configs
    all_by_cat = {}
    all_categories = set()

    for config in configs:
        if model and dataset_type and distractor_source:
            dir_name = f"{model.replace('/', '_')}_{dataset_type}_{distractor_source}"
            results_path = results_dir / dir_name / config / "results.json"
        else:
            results_path = results_dir / config / "results.json"

        if not results_path.exists():
            continue

        _, entries = load_results_with_categories(results_path)
        by_cat = compute_accuracy_by_category(entries, category_key)
        all_by_cat[config] = by_cat
        all_categories.update(by_cat.keys())

    if not all_by_cat:
        print("No results found")
        return None

    # Sort categories by average accuracy
    cat_avg = {}
    for cat in all_categories:
        if cat == "unknown":
            continue
        accs = [all_by_cat[cfg].get(cat, {}).get("accuracy", 0)
                for cfg in configs if cfg in all_by_cat]
        if accs:
            cat_avg[cat] = np.mean(accs)

    sorted_cats = sorted(cat_avg.items(), key=lambda x: -x[1])[:top_n]
    categories = [c[0] for c in sorted_cats]

    # Build matrix
    present_configs = [c for c in configs if c in all_by_cat]
    matrix = np.zeros((len(categories), len(present_configs)))
    for i, cat in enumerate(categories):
        for j, config in enumerate(present_configs):
            if config in all_by_cat and cat in all_by_cat[config]:
                matrix[i, j] = all_by_cat[config][cat]["accuracy"] * 100

    fig, ax = plt.subplots(figsize=(10, max(6, len(categories) * 0.5)))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=30, vmax=100)

    # Labels
    ax.set_xticks(range(len(present_configs)))
    ax.set_xticklabels(present_configs, fontsize=12)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=11)

    # Add text annotations
    for i in range(len(categories)):
        for j in range(len(present_configs)):
            val = matrix[i, j]
            color = 'white' if val < 60 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

    title = f'Accuracy by {category_key.capitalize()} x Configuration'
    if dataset_type:
        title += f'\n({dataset_type})'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Accuracy (%)', fontsize=12)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    return fig


def plot_dataset_type_breakdown(
    base_dir: Path,
    model: str,
    configs: Optional[List[str]] = None,
    distractor_sources: Optional[List[str]] = None,
    dataset_types: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> Optional[plt.Figure]:
    if configs is None:
        configs = ["3H0M", "3H3M", "3H6M", "0H3M", "0H6M"]
    if distractor_sources is None:
        distractor_sources = ["scratch"]
    if dataset_types is None:
        dataset_types = ["mmlu_pro", "supergpqa", "arc_easy", "arc_challenge"]

    # Collect data
    data = {}  # {config: {dataset_type: accuracy}}
    for config_str in configs:
        data[config_str] = {}
        for dt in dataset_types:
            for source in distractor_sources:
                result = compute_accuracy_by_dataset_type(
                    base_dir, model, config_str, [source], [dt]
                )
                if dt in result and source in result[dt]:
                    data[config_str][dt] = result[dt][source]["accuracy"]

    # Check we have data
    has_data = any(bool(v) for v in data.values())
    if not has_data:
        print("No data found for dataset type breakdown")
        return None

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(dataset_types))
    width = 0.8 / len(configs)

    for i, config_str in enumerate(configs):
        accuracies = [data[config_str].get(dt, 0) * 100 for dt in dataset_types]
        offset = (i - len(configs) / 2 + 0.5) * width
        bars = ax.bar(x + offset, accuracies, width, label=config_str, alpha=0.85)

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            if acc > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{acc:.0f}', ha='center', va='bottom', fontsize=9,
                )

    dt_labels = {
        "mmlu_pro": "MMLU-Pro",
        "supergpqa": "SuperGPQA",
        "arc_easy": "ARC-Easy",
        "arc_challenge": "ARC-Challenge",
    }

    ax.set_xlabel('Dataset Type', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title(f'Accuracy by Dataset Type\nModel: {model}', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([dt_labels.get(dt, dt) for dt in dataset_types], fontsize=12)
    ax.legend(fontsize=11, title="Config")
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    return fig


def plot_supergpqa_by_discipline(
    results_path: Path,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> Optional[plt.Figure]:
    summary, entries = load_results_with_categories(results_path)

    # By discipline
    by_discipline = compute_accuracy_by_category(entries, "discipline")

    # By field within discipline
    by_field = compute_accuracy_by_category(entries, "category")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: By Discipline
    if by_discipline:
        sorted_disc = sorted(by_discipline.items(), key=lambda x: -x[1]["accuracy"])
        disciplines = [d[0] for d in sorted_disc]
        accs = [d[1]["accuracy"] * 100 for d in sorted_disc]
        colors = [SUPERGPQA_DISCIPLINE_COLORS.get(d, "#95a5a6") for d in disciplines]

        ax1.barh(range(len(disciplines)), accs, color=colors, edgecolor='black')
        ax1.set_yticks(range(len(disciplines)))
        ax1.set_yticklabels(disciplines, fontsize=12)
        ax1.set_xlabel('Accuracy (%)', fontsize=13)
        ax1.set_title('Accuracy by Discipline', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 100)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)

    # Plot 2: By Field (top 10)
    if by_field:
        sorted_field = sorted(by_field.items(), key=lambda x: -x[1]["accuracy"])[:10]
        fields = [f[0] for f in sorted_field]
        accs = [f[1]["accuracy"] * 100 for f in sorted_field]

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(fields)))
        ax2.barh(range(len(fields)), accs, color=colors, edgecolor='black')
        ax2.set_yticks(range(len(fields)))
        ax2.set_yticklabels(fields, fontsize=11)
        ax2.set_xlabel('Accuracy (%)', fontsize=13)
        ax2.set_title('Top 10 Fields by Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 100)
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)

    plt.suptitle('SuperGPQA Performance Breakdown', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    return fig


def generate_category_report(
    results_path: Path,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    summary, entries = load_results_with_categories(results_path)

    report = {
        "overall_accuracy": summary.get("accuracy", 0),
        "total_entries": len(entries),
        "by_category": compute_accuracy_by_category(entries, "category"),
    }

    # Check for discipline (SuperGPQA)
    if any(e.get("discipline") for e in entries):
        report["by_discipline"] = compute_accuracy_by_category(entries, "discipline")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "category_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Saved report: {report_path}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Category/topic breakout analysis")
    parser.add_argument("--results", type=str, help="Path to results JSON (single file)")
    parser.add_argument("--results-dir", type=str, help="Results base directory (for multi-config)")
    parser.add_argument("--model", type=str, help="Model name (for dataset_type breakdown)")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--category-key", type=str, default="category")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.results:
        results_path = Path(args.results)
        output_dir = Path(args.output_dir) if args.output_dir else results_path.parent

        # Generate bar chart
        plot_category_breakdown(
            results_path,
            category_key=args.category_key,
            output_path=output_dir / "category_breakdown.png",
            show=args.show,
        )

        # Generate report
        generate_category_report(results_path, output_dir)

    if args.results_dir and args.model:
        base_dir = Path(args.results_dir)
        output_dir = Path(args.output_dir) if args.output_dir else base_dir / "plots"

        # Dataset type breakdown
        plot_dataset_type_breakdown(
            base_dir,
            model=args.model,
            output_path=output_dir / "dataset_type_breakdown.png",
            show=args.show,
        )
