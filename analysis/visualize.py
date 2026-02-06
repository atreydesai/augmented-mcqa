"""
Visualization module for Augmented MCQA.

Provides plotting utilities for experiment results.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json


def create_behavioral_bar_chart(
    experiments: Dict[str, Dict],
    title: str = "Behavioral Signature Comparison",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Create a bar chart comparing behavioral signatures across experiments.
    
    Args:
        experiments: Dict mapping experiment name to results with signature
        title: Chart title
        output_path: Path to save figure (optional)
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    names = list(experiments.keys())
    n_exp = len(names)
    
    # Extract rates
    g_rates = []
    h_rates = []
    m_rates = []
    
    for name in names:
        sig = experiments[name].get("signature", {})
        rates = sig.get("rates", {})
        g_rates.append(rates.get("G", 0) * 100)
        h_rates.append(rates.get("H", 0) * 100)
        m_rates.append(rates.get("M", 0) * 100)
    
    # Create plot
    x = np.arange(n_exp)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars1 = ax.bar(x - width, g_rates, width, label='Gold (G)', color='#2ecc71')
    bars2 = ax.bar(x, h_rates, width, label='Human (H)', color='#3498db')
    bars3 = ax.bar(x + width, m_rates, width, label='Model (M)', color='#e74c3c')
    
    ax.set_ylabel('Selection Rate (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig, ax


def create_accuracy_comparison(
    experiments: Dict[str, float],
    title: str = "Accuracy Comparison",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Create a bar chart comparing accuracies across experiments.
    
    Args:
        experiments: Dict mapping experiment name to accuracy (0-1)
        title: Chart title
        output_path: Path to save figure (optional)
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    names = list(experiments.keys())
    accuracies = [experiments[n] * 100 for n in names]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(range(len(names)), accuracies, color='#3498db')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f'{acc:.1f}%',
            ha='center',
            va='bottom',
            fontsize=9,
        )
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig, ax


def create_category_heatmap(
    category_results: Dict[str, Dict[str, float]],
    metric: str = "gold_rate",
    title: str = "Accuracy by Category",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
):
    """
    Create a heatmap of results by experiment and category.
    
    Args:
        category_results: Dict mapping experiment to category results
        metric: Which metric to display
        title: Chart title
        output_path: Path to save figure (optional)
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    experiments = list(category_results.keys())
    
    # Get all categories
    all_categories = set()
    for exp_results in category_results.values():
        all_categories.update(exp_results.keys())
    categories = sorted(all_categories)
    
    # Build matrix
    data = []
    for exp in experiments:
        row = []
        for cat in categories:
            val = category_results[exp].get(cat, {})
            if isinstance(val, dict):
                row.append(val.get(metric, 0))
            else:
                row.append(val)
        data.append(row)
    
    data = np.array(data) * 100  # Convert to percentage
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Labels
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_yticks(range(len(experiments)))
    ax.set_yticklabels(experiments)
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(f'{metric} (%)', rotation=-90, va="bottom")
    
    ax.set_title(title)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig, ax


def plot_results_summary(
    results_dir: Path,
    output_dir: Optional[Path] = None,
):
    """
    Generate standard plots from a results directory.
    
    Args:
        results_dir: Directory containing results.json
        output_dir: Where to save plots (default: results_dir/plots)
    """
    results_path = Path(results_dir) / "results.json"
    
    if not results_path.exists():
        print(f"No results.json found at {results_path}")
        return
    
    if output_dir is None:
        output_dir = Path(results_dir) / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Extract summary if available
    summary = data.get("summary", {})
    config = data.get("config", {})
    
    exp_name = config.get("name", "experiment")
    
    # Create behavioral chart if we have behavioral counts
    if "behavioral_counts" in summary:
        create_behavioral_bar_chart(
            {exp_name: {"signature": {"counts": summary["behavioral_counts"], "rates": {}}}},
            title=f"Behavioral Signature: {exp_name}",
            output_path=output_dir / "behavioral.png",
        )
    
    # Create accuracy by category chart
    if "accuracy_by_category" in summary:
        cat_data = summary["accuracy_by_category"]
        create_accuracy_comparison(
            cat_data,
            title=f"Accuracy by Category: {exp_name}",
            output_path=output_dir / "accuracy_by_category.png",
        )
    
    print(f"Plots saved to: {output_dir}")
