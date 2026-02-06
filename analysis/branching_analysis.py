"""
Branching Human Distractor Analysis.

Visualizes the benefit of adding 1 vs 2 vs 3 human distractors
with branching lines showing model distractor additions off each base.

Creates plots showing:
- Base lines: 1H, 2H, 3H (human-only baselines)
- Branching lines: 1H+1M..6M, 2H+1M..6M, 3H+1M..6M
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR


# Color scheme for human distractor baselines
HUMAN_COLORS = {
    1: "#e74c3c",  # Red for 1H
    2: "#f39c12",  # Orange for 2H
    3: "#27ae60",  # Green for 3H
}

# Line styles for readability
LINE_STYLES = {
    1: "-",
    2: "--",
    3: "-.",
}


def load_branching_results(
    base_dir: Path,
    dataset_type: str = "normal",
    is_choices_only: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Load results for branching analysis.
    
    Args:
        base_dir: Results base directory
        dataset_type: "normal" or "augmented"
        is_choices_only: Whether to load choices-only results
        
    Returns:
        Dict mapping config string -> results dict
    """
    results = {}
    suffix = "_choices_only" if is_choices_only else ""
    
    # Load all H+M configurations
    for h in range(1, 4):  # 1H, 2H, 3H
        for m in range(0, 7):  # 0M to 6M
            config = f"{h}H{m}M"
            setting_dir = base_dir / f"{config}_{dataset_type}{suffix}"
            
            # Try multiple summary file patterns
            patterns = [
                f"overall_summary_{h}H_{m}M{suffix}.json",
                f"overall_summary.json",
                "results.json",
            ]
            
            for pattern in patterns:
                summary_path = setting_dir / pattern
                if summary_path.exists():
                    with open(summary_path) as f:
                        data = json.load(f)
                    results[config] = {
                        "accuracy": data.get("acc", data.get("accuracy", 0)),
                        "correct": data.get("corr", data.get("correct", 0)),
                        "total": data.get("total", 0),
                    }
                    break
    
    return results


def plot_human_distractor_branching(
    base_dir: Path,
    dataset_type: str = "normal",
    output_dir: Optional[Path] = None,
    show: bool = False,
) -> Dict[str, Path]:
    """
    Create branching analysis plots.
    
    Creates two plots (Full Questions and Choices Only) showing:
    - Three baseline curves for 1H, 2H, 3H
    - Branching lines for adding model distractors
    
    Args:
        base_dir: Directory containing experiment results
        dataset_type: "normal" or "augmented"
        output_dir: Where to save plots
        show: Whether to display interactively
        
    Returns:
        Dict mapping plot name -> output path
    """
    base_dir = Path(base_dir)
    if output_dir is None:
        output_dir = base_dir / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_paths = {}
    
    for is_choices_only in [False, True]:
        suffix = "choices_only" if is_choices_only else "full"
        results = load_branching_results(base_dir, dataset_type, is_choices_only)
        
        if not results:
            print(f"No results found for {suffix}")
            continue
        
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
        title_suffix = "Choices Only" if is_choices_only else "Full Questions"
        ax.set_xlabel('Total Number of Distractors', fontsize=16)
        ax.set_ylabel('Accuracy', fontsize=16)
        ax.set_title(
            f'Benefit of Human Distractors with Model Additions\n({title_suffix}, {dataset_type.capitalize()})',
            fontsize=18, fontweight='bold'
        )
        ax.legend(loc='upper right', fontsize=12, frameon=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)
        
        # Adjust y-limits based on data
        if is_choices_only:
            ax.set_ylim(0.25, 0.75)
        else:
            ax.set_ylim(0.60, 0.95)
        
        ax.tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        
        # Save
        output_path = output_dir / f"branching_{dataset_type}_{suffix}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        output_paths[f"branching_{suffix}"] = output_path
        print(f"Saved: {output_path}")
        
        if show:
            plt.show()
        plt.close()
    
    return output_paths


def plot_human_benefit_comparison(
    base_dir: Path,
    output_dir: Optional[Path] = None,
    show: bool = False,
) -> Path:
    """
    Create side-by-side comparison of human distractor benefit.
    
    Shows how accuracy degrades with model distractors for different
    human distractor counts.
    
    Args:
        base_dir: Results directory
        output_dir: Output directory for plots
        show: Whether to display interactively
        
    Returns:
        Path to saved figure
    """
    base_dir = Path(base_dir)
    if output_dir is None:
        output_dir = base_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    for idx, (is_choices, ax) in enumerate(zip([False, True], axes)):
        results = load_branching_results(base_dir, "normal", is_choices)
        
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
        
        title = "Choices Only" if is_choices else "Full Questions"
        ax.set_xlabel('Number of Model Distractors Added', fontsize=14)
        ax.set_ylabel('Accuracy Drop from Baseline', fontsize=14)
        ax.set_title(f'Accuracy Degradation ({title})', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 6.5)
    
    plt.suptitle(
        'How Model Distractors Reduce Accuracy (by Human Base)',
        fontsize=18, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    
    output_path = output_dir / "human_benefit_degradation.png"
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
    parser.add_argument("--output-dir", type=str, help="Output directory for plots")
    parser.add_argument("--dataset-type", type=str, default="normal", choices=["normal", "augmented"])
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    print("Creating branching analysis plots...")
    plot_human_distractor_branching(results_dir, args.dataset_type, output_dir, args.show)
    
    print("\nCreating benefit comparison plot...")
    plot_human_benefit_comparison(results_dir, output_dir, args.show)
