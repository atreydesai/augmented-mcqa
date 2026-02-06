"""
Category/Topic Breakout Analysis.

Provides detailed visualizations broken down by:
- MMLU-Pro categories (business, physics, math, etc.)
- SuperGPQA disciplines and fields
- ARC question types

Supports both heatmaps and grouped bar charts.
"""

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


def load_results_with_categories(
    results_path: Path,
) -> Tuple[Dict[str, Any], List[Dict]]:
    """
    Load results file and extract per-entry category information.
    
    Args:
        results_path: Path to results JSON file
        
    Returns:
        Tuple of (summary dict, list of per-entry results)
    """
    with open(results_path) as f:
        data = json.load(f)
    
    summary = data.get("summary", {})
    entries = data.get("results", [])
    
    return summary, entries


def compute_accuracy_by_category(
    entries: List[Dict],
    category_key: str = "category",
) -> Dict[str, Dict[str, Any]]:
    """
    Compute accuracy statistics grouped by category.
    
    Args:
        entries: List of result entries
        category_key: Key to use for grouping ("category", "discipline", "field")
        
    Returns:
        Dict mapping category -> {accuracy, correct, total}
    """
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


def plot_category_breakdown(
    results_path: Path,
    category_key: str = "category",
    output_path: Optional[Path] = None,
    top_n: int = 15,
    min_samples: int = 10,
    figsize: Tuple[int, int] = (14, 8),
    show: bool = False,
) -> plt.Figure:
    """
    Create bar chart showing accuracy by category.
    
    Args:
        results_path: Path to results JSON
        category_key: Category field to group by
        output_path: Where to save figure
        top_n: Maximum categories to show
        min_samples: Minimum samples required per category
        figsize: Figure dimensions
        show: Show interactively
        
    Returns:
        Matplotlib figure
    """
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
    configs: List[str] = ["3H0M", "3H3M", "0H3M"],
    category_key: str = "category",
    output_path: Optional[Path] = None,
    top_n: int = 12,
    show: bool = False,
) -> plt.Figure:
    """
    Create heatmap showing accuracy across categories and configs.
    
    Args:
        results_dir: Directory containing results for different configs
        configs: Distractor configurations to compare
        category_key: Category field to group by
        output_path: Where to save figure
        top_n: Number of categories to show
        show: Show interactively
        
    Returns:
        Matplotlib figure
    """
    # Load all configs
    all_by_cat = {}
    all_categories = set()
    
    for config in configs:
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
    matrix = np.zeros((len(categories), len(configs)))
    for i, cat in enumerate(categories):
        for j, config in enumerate(configs):
            if config in all_by_cat and cat in all_by_cat[config]:
                matrix[i, j] = all_by_cat[config][cat]["accuracy"] * 100
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(categories) * 0.5)))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=30, vmax=100)
    
    # Labels
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, fontsize=12)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=11)
    
    # Add text annotations
    for i in range(len(categories)):
        for j in range(len(configs)):
            val = matrix[i, j]
            color = 'white' if val < 60 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', 
                    fontsize=10, color=color, fontweight='bold')
    
    ax.set_title(f'Accuracy by {category_key.capitalize()} × Configuration', 
                 fontsize=14, fontweight='bold')
    
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


def plot_supergpqa_by_discipline(
    results_path: Path,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Create discipline-specific visualization for SuperGPQA.
    
    Args:
        results_path: Path to SuperGPQA results
        output_path: Where to save
        show: Show interactively
        
    Returns:
        Matplotlib figure
    """
    summary, entries = load_results_with_categories(results_path)
    
    # By discipline
    by_discipline = compute_accuracy_by_category(entries, "discipline")
    
    # By field within discipline
    by_field = compute_accuracy_by_category(entries, "category")  # 'field' mapped to 'category'
    
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
    """
    Generate a comprehensive category analysis report.
    
    Args:
        results_path: Path to results JSON
        output_dir: Directory for output files
        
    Returns:
        Report dictionary
    """
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
    parser.add_argument("--results", type=str, required=True, help="Path to results JSON")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--category-key", type=str, default="category")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    
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
