"""
Difficulty Scaling Visualization Module.

Provides visualization functions for comparing model performance
across difficulty levels (Easy, Medium, Hard).

Includes:
- Independent plots per dataset type
- Combined difficulty comparison plots
- Distractor effect scaling plots
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR, DatasetType
from experiments.difficulty import DifficultyLevel, DIFFICULTY_DATASETS


# Color schemes for each difficulty level
DIFFICULTY_COLORS = {
    DifficultyLevel.EASY: "#2ecc71",      # Green
    DifficultyLevel.MEDIUM: "#f39c12",    # Orange
    DifficultyLevel.HARD: "#e74c3c",      # Red
}

# Dataset-specific colors
DATASET_COLORS = {
    "arc_easy": "#27ae60",
    "arc_challenge": "#e67e22",
    "mmlu_pro": "#3498db",
    "supergpqa": "#9b59b6",
    "mmlu": "#1abc9c",
}

# Markers for different datasets
DATASET_MARKERS = {
    "arc_easy": "o",
    "arc_challenge": "s",
    "mmlu_pro": "^",
    "supergpqa": "D",
    "mmlu": "v",
}


def load_difficulty_results(
    results_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load all difficulty scaling results from a directory.
    
    Args:
        results_dir: Directory containing results (default: RESULTS_DIR/difficulty_scaling)
        
    Returns:
        Dict mapping dataset_name -> results
    """
    if results_dir is None:
        results_dir = RESULTS_DIR / "difficulty_scaling"
    
    results = {}
    if results_dir.exists():
        for f in results_dir.glob("*_results.json"):
            # Parse filename: {dataset}_{config}_results.json
            parts = f.stem.replace("_results", "").rsplit("_", 1)
            if len(parts) >= 1:
                dataset_name = parts[0]
                with open(f) as fp:
                    results[f.stem] = json.load(fp)
    
    return results


def plot_arc_comparison(
    results: Dict[str, Dict[str, Any]],
    distractor_configs: List[str] = ["3H0M", "0H3M", "3H3M"],
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot ARC-Easy vs ARC-Challenge comparison.
    
    Args:
        results: Results dict with arc_easy and arc_challenge data
        distractor_configs: Distractor configurations to plot
        output_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    datasets = ["arc_easy", "arc_challenge"]
    x = np.arange(len(distractor_configs))
    width = 0.35
    
    for i, dataset in enumerate(datasets):
        accuracies = []
        for config in distractor_configs:
            key = f"{dataset}_{config}"
            if key in results:
                accuracies.append(results[key].get("accuracy", 0))
            else:
                accuracies.append(0)
        
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            accuracies,
            width,
            label=DIFFICULTY_DATASETS[dataset].description.split(":")[0],
            color=DATASET_COLORS[dataset],
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
    
    ax.set_xlabel("Distractor Configuration")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("ARC-Easy vs ARC-Challenge Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(distractor_configs)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    
    return fig


def plot_supergpqa_by_difficulty(
    results: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot SuperGPQA results broken down by difficulty tag.
    
    Args:
        results: Results dict with supergpqa data
        output_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group by difficulty tag if available
    difficulties = ["middle", "hard"]
    configs = ["3H0M", "0H3M", "3H3M"]
    
    x = np.arange(len(configs))
    width = 0.35
    
    for i, diff in enumerate(difficulties):
        accuracies = []
        for config in configs:
            # Look for difficulty-specific results
            key = f"supergpqa_{diff}_{config}"
            if key in results:
                accuracies.append(results[key].get("accuracy", 0))
            else:
                # Fall back to overall supergpqa
                alt_key = f"supergpqa_{config}"
                if alt_key in results:
                    accuracies.append(results[alt_key].get("accuracy", 0))
                else:
                    accuracies.append(0)
        
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            accuracies,
            width,
            label=f"Difficulty: {diff.capitalize()}",
            color=DIFFICULTY_COLORS[DifficultyLevel.MEDIUM if diff == "middle" else DifficultyLevel.HARD],
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
    
    ax.set_xlabel("Distractor Configuration")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("SuperGPQA Performance by Difficulty Level")
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    
    return fig


def plot_difficulty_combined(
    results: Dict[str, Dict[str, Any]],
    distractor_config: str = "3H0M",
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Plot all datasets combined by difficulty level.
    
    Args:
        results: Results dict
        distractor_config: Distractor configuration to plot
        output_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group datasets by difficulty
    difficulty_order = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
    
    all_data = []
    for level in difficulty_order:
        level_datasets = [
            name for name, config in DIFFICULTY_DATASETS.items()
            if config.difficulty == level
        ]
        
        for dataset in level_datasets:
            key = f"{dataset}_{distractor_config}"
            if key in results:
                all_data.append({
                    "dataset": dataset,
                    "difficulty": level,
                    "accuracy": results[key].get("accuracy", 0),
                    "label": DIFFICULTY_DATASETS[dataset].description.split(":")[0],
                })
    
    if not all_data:
        ax.text(0.5, 0.5, "No results found", ha="center", va="center", transform=ax.transAxes)
        return fig
    
    # Create grouped bar chart
    x = np.arange(len(all_data))
    colors = [DIFFICULTY_COLORS[d["difficulty"]] for d in all_data]
    
    bars = ax.bar(x, [d["accuracy"] for d in all_data], color=colors, edgecolor="black")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    
    ax.set_xticks(x)
    ax.set_xticklabels([d["label"] for d in all_data], rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Performance Across Difficulty Levels ({distractor_config})")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    
    # Add legend for difficulty levels
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=DIFFICULTY_COLORS[level], label=level.value.capitalize())
        for level in difficulty_order
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    
    return fig


def plot_distractor_effect_scaling(
    results: Dict[str, Dict[str, Any]],
    datasets: List[str] = ["arc_easy", "arc_challenge", "supergpqa"],
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Plot how distractor effect varies with difficulty.
    
    Shows accuracy curves for 3H+M (varying number of model distractors)
    for each dataset.
    
    Args:
        results: Results dict
        datasets: Datasets to include
        output_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # X-axis: number of model distractors
    model_counts = [0, 1, 2, 3, 4, 5, 6]
    
    for dataset in datasets:
        accuracies = []
        for m in model_counts:
            config = f"3H{m}M"
            key = f"{dataset}_{config}"
            if key in results:
                accuracies.append(results[key].get("accuracy", 0))
            else:
                accuracies.append(None)
        
        # Plot line (skip None values)
        valid_x = [x for x, y in zip(model_counts, accuracies) if y is not None]
        valid_y = [y for y in accuracies if y is not None]
        
        if valid_y:
            ax.plot(
                valid_x,
                valid_y,
                marker=DATASET_MARKERS.get(dataset, "o"),
                color=DATASET_COLORS.get(dataset, "#666666"),
                label=DIFFICULTY_DATASETS[dataset].description.split(":")[0],
                linewidth=2,
                markersize=8,
            )
    
    ax.set_xlabel("Number of Model Distractors (3H + M)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Distractor Effect Across Difficulty Levels")
    ax.set_xticks(model_counts)
    ax.set_xticklabels([f"3H{m}M" for m in model_counts])
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    
    return fig


def plot_all_difficulty(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Generate all difficulty comparison plots.
    
    Args:
        results_dir: Directory containing results
        output_dir: Directory to save plots
        
    Returns:
        Dict mapping plot name -> output path
    """
    results = load_difficulty_results(results_dir)
    
    if output_dir is None:
        output_dir = RESULTS_DIR / "difficulty_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_paths = {}
    
    # ARC comparison
    arc_path = output_dir / "arc_comparison.png"
    plot_arc_comparison(results, output_path=arc_path)
    output_paths["arc_comparison"] = arc_path
    
    # SuperGPQA by difficulty
    supergpqa_path = output_dir / "supergpqa_difficulty.png"
    plot_supergpqa_by_difficulty(results, output_path=supergpqa_path)
    output_paths["supergpqa_difficulty"] = supergpqa_path
    
    # Combined difficulty
    combined_path = output_dir / "difficulty_combined.png"
    plot_difficulty_combined(results, output_path=combined_path)
    output_paths["difficulty_combined"] = combined_path
    
    # Distractor scaling
    scaling_path = output_dir / "distractor_scaling.png"
    plot_distractor_effect_scaling(results, output_path=scaling_path)
    output_paths["distractor_scaling"] = scaling_path
    
    print(f"\nGenerated {len(output_paths)} plots in {output_dir}")
    return output_paths


if __name__ == "__main__":
    # Test with any available results
    print("Testing difficulty visualization module...")
    
    results = load_difficulty_results()
    if results:
        print(f"Found {len(results)} result files")
        plot_all_difficulty()
    else:
        print("No results found. Run experiments first.")
