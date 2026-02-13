"""
Visualization module for Augmented MCQA.

Generates the 6 RQ line plots:
- RQ1 (Full): 3H+M vs Human Only vs Model Only (Full Questions)
- RQ1 (Choices): 3H+M vs Human Only vs Model Only (Choices Only)
- RQ2 (Full): Human Only comparison (MMLU-Pro vs MMLU-Aug)
- RQ2 (Choices): Human Only comparison (Choices Only)
- RQ3 (Full): Model Only comparison (MMLU-Pro vs MMLU-Aug)
- RQ3 (Choices): Model Only comparison (Choices Only)

Based on overlay_full_vs_choices_only.py from the original repository.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt


def load_summary_file(base_dir: Path, setting_name: str) -> Optional[Dict]:
    """Load the new results.json format and return accuracy data."""
    summary_path = base_dir / setting_name / "results.json"
    if not summary_path.exists():
        # Fallback to overall_summary.json if results.json not found
        summary_path = base_dir / setting_name / "overall_summary.json"
        if not summary_path.exists():
            return None
    
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    # Handle new results.json structure
    if "summary" in data:
        summary = data["summary"]
        return {
            "accuracy": summary.get("accuracy", 0),
            "total": summary.get("total", 0),
            "correct": summary.get("correct", 0),
        }
    
    # Handle old overall_summary.json format
    return {
        "accuracy": data.get("acc", 0),
        "total": data.get("total", 0),
        "correct": data.get("corr", 0),
    }


def load_results_for_rq(
    base_dir: Path,
    dataset: str,
    model: str,
    prefix: str = "RQ",
) -> Dict[str, List[Dict]]:
    """
    Load all related results for an RQ combined analysis.
    
    Looks for directories matching the RQ pattern.
    """
    results = {
        "human_only": [],
        "human_only_choices": [],
        "dhuman_synthetic": [],
        "dhuman_synthetic_choices": [],
        "scratch_synthetic": [],
        "existing_synthetic": [],
        "dmodel_synthetic": [],
    }
    
    # Map from suffix to result key
    mappings = {
        f"human_only_{dataset}_{model}": "human_only",
        f"human_only_choices_{dataset}_{model}": "human_only_choices",
        f"dhuman_synthetic_{dataset}_{model}": "dhuman_synthetic",
        f"dhuman_synthetic_choices_{dataset}_{model}": "dhuman_synthetic_choices",
        f"scratch_synthetic_{dataset}_{model}": "scratch_synthetic",
        f"existing_synthetic_{dataset}_{model}": "existing_synthetic",
        f"dmodel_synthetic_{dataset}_{model}": "dmodel_synthetic",
    }
    
    for suffix, key in mappings.items():
        name = f"{prefix}_{suffix}"
        data = load_summary_file(base_dir, name)
        if data:
            results[key].append(data)
            
    return results


def load_3H_plus_M_results(
    base_dir: Path,
    dataset_type: str = "normal",
    is_choices_only: bool = False,
) -> List[Dict]:
    """Load 3H+M results (3H0M through 3H6M)."""
    results = []
    
    for m in range(7):  # 0M to 6M
        suffix = "_choices_only" if is_choices_only else ""
        setting_name = f"3H{m}M_{dataset_type}{suffix}"
        
        data = load_summary_file(base_dir, setting_name)
        if data:
            results.append({
                "setting": setting_name,
                "num_human": 3,
                "num_model": m,
                "total_distractors": 3 + m,
                **data
            })
    
    return results


def load_human_only_results(
    base_dir: Path,
    dataset_type: str = "normal",
    is_choices_only: bool = False,
) -> List[Dict]:
    """Load Human Only results (1H0M, 2H0M, 3H0M)."""
    results = []
    
    for h in range(1, 4):  # 1H to 3H
        suffix = "_choices_only" if is_choices_only else ""
        setting_name = f"{h}H0M_{dataset_type}{suffix}"
        
        data = load_summary_file(base_dir, setting_name)
        if data:
            results.append({
                "setting": setting_name,
                "num_human": h,
                "num_model": 0,
                "total_distractors": h,
                **data
            })
    
    return results


def load_model_only_results(
    base_dir: Path,
    dataset_type: str = "normal",
    is_choices_only: bool = False,
) -> List[Dict]:
    """Load Model Only results (0H1M through 0H6M)."""
    results = []
    
    for m in range(1, 7):  # 1M to 6M
        suffix = "_choices_only" if is_choices_only else ""
        setting_name = f"0H{m}M_{dataset_type}{suffix}"
        
        data = load_summary_file(base_dir, setting_name)
        if data:
            results.append({
                "setting": setting_name,
                "num_human": 0,
                "num_model": m,
                "total_distractors": m,
                **data
            })
    
    return results


def plot_rq1_combined(
    base_dir: Path,
    output_dir: Optional[Path] = None,
    show: bool = False,
):
    """
    RQ1: Combined plot showing all experiment types.
    
    Creates two plots:
    - Full Questions: 3H+M, Human Only, Model Only (MMLU-Pro and MMLU-Aug)
    - Choices Only: Same series for choices-only condition
    """
    base_dir = Path(base_dir)
    
    # Load all data variants
    print("Loading results...")
    
    # 3H + M results
    results_3H_normal_full = load_3H_plus_M_results(base_dir, "normal", False)
    results_3H_normal_choices = load_3H_plus_M_results(base_dir, "normal", True)
    results_3H_aug_full = load_3H_plus_M_results(base_dir, "augmented", False)
    results_3H_aug_choices = load_3H_plus_M_results(base_dir, "augmented", True)
    
    # Human Only results
    results_human_normal_full = load_human_only_results(base_dir, "normal", False)
    results_human_normal_choices = load_human_only_results(base_dir, "normal", True)
    results_human_aug_full = load_human_only_results(base_dir, "augmented", False)
    results_human_aug_choices = load_human_only_results(base_dir, "augmented", True)
    
    # Model Only results
    results_model_normal_full = load_model_only_results(base_dir, "normal", False)
    results_model_normal_choices = load_model_only_results(base_dir, "normal", True)
    results_model_aug_full = load_model_only_results(base_dir, "augmented", False)
    results_model_aug_choices = load_model_only_results(base_dir, "augmented", True)
    
    # ===== PLOT 1: Full Questions =====
    plt.figure(figsize=(14, 9))
    
    # 3H + M variants (Full)
    if results_3H_normal_full:
        x = [r['total_distractors'] for r in results_3H_normal_full]
        y = [r['accuracy'] for r in results_3H_normal_full]
        plt.plot(x, y, marker='o', linewidth=2.5, markersize=10, 
                 label='3H + M (MMLU-Pro)', color='blue', linestyle='-')
    
    if results_3H_aug_full:
        x = [r['total_distractors'] for r in results_3H_aug_full]
        y = [r['accuracy'] for r in results_3H_aug_full]
        plt.plot(x, y, marker='s', linewidth=2.5, markersize=10, 
                 label='3H + M (MMLU-Aug)', color='red', linestyle='-')
    
    # Human Only variants (Full)
    if results_human_normal_full:
        x = [r['total_distractors'] for r in results_human_normal_full]
        y = [r['accuracy'] for r in results_human_normal_full]
        plt.plot(x, y, marker='^', linewidth=2.5, markersize=10, 
                 label='Human Only (MMLU-Pro)', color='purple', linestyle='--', alpha=0.8)
    
    if results_human_aug_full:
        x = [r['total_distractors'] for r in results_human_aug_full]
        y = [r['accuracy'] for r in results_human_aug_full]
        plt.plot(x, y, marker='v', linewidth=2.5, markersize=10, 
                 label='Human Only (MMLU-Aug)', color='orange', linestyle='--', alpha=0.8)
    
    # Model Only variants (Full)
    if results_model_normal_full:
        x = [r['total_distractors'] for r in results_model_normal_full]
        y = [r['accuracy'] for r in results_model_normal_full]
        plt.plot(x, y, marker='D', linewidth=2.5, markersize=10, 
                 label='Model Only (MMLU-Pro)', color='green', linestyle=':', alpha=0.8)
    
    if results_model_aug_full:
        x = [r['total_distractors'] for r in results_model_aug_full]
        y = [r['accuracy'] for r in results_model_aug_full]
        plt.plot(x, y, marker='d', linewidth=2.5, markersize=10, 
                 label='Model Only (MMLU-Aug)', color='brown', linestyle=':', alpha=0.8)
    
    # Styling for Plot 1
    plt.xlabel('Total Number of Distractors', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Accuracy vs Total Number of Distractors\n(Full Questions)', fontsize=20, fontweight='bold')
    plt.legend(loc='upper right', fontsize=12, markerscale=1.0, frameon=True, 
               bbox_to_anchor=(1.0, 1.0), ncol=1)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 10)
    plt.ylim(0.65, 0.95)
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "rq1_full_questions.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()
    
    # ===== PLOT 2: Choices Only =====
    plt.figure(figsize=(14, 9))
    
    # 3H + M variants (Choices Only)
    if results_3H_normal_choices:
        x = [r['total_distractors'] for r in results_3H_normal_choices]
        y = [r['accuracy'] for r in results_3H_normal_choices]
        plt.plot(x, y, marker='o', linewidth=2.5, markersize=10, 
                 label='3H + M (MMLU-Pro)', color='blue', linestyle='-')
    
    if results_3H_aug_choices:
        x = [r['total_distractors'] for r in results_3H_aug_choices]
        y = [r['accuracy'] for r in results_3H_aug_choices]
        plt.plot(x, y, marker='s', linewidth=2.5, markersize=10, 
                 label='3H + M (MMLU-Aug)', color='red', linestyle='-')
    
    # Human Only variants (Choices Only)
    if results_human_normal_choices:
        x = [r['total_distractors'] for r in results_human_normal_choices]
        y = [r['accuracy'] for r in results_human_normal_choices]
        plt.plot(x, y, marker='^', linewidth=2.5, markersize=10, 
                 label='Human Only (MMLU-Pro)', color='purple', linestyle='--', alpha=0.8)
    
    if results_human_aug_choices:
        x = [r['total_distractors'] for r in results_human_aug_choices]
        y = [r['accuracy'] for r in results_human_aug_choices]
        plt.plot(x, y, marker='v', linewidth=2.5, markersize=10, 
                 label='Human Only (MMLU-Aug)', color='orange', linestyle='--', alpha=0.8)
    
    # Model Only variants (Choices Only)
    if results_model_normal_choices:
        x = [r['total_distractors'] for r in results_model_normal_choices]
        y = [r['accuracy'] for r in results_model_normal_choices]
        plt.plot(x, y, marker='D', linewidth=2.5, markersize=10, 
                 label='Model Only (MMLU-Pro)', color='green', linestyle=':', alpha=0.8)
    
    if results_model_aug_choices:
        x = [r['total_distractors'] for r in results_model_aug_choices]
        y = [r['accuracy'] for r in results_model_aug_choices]
        plt.plot(x, y, marker='d', linewidth=2.5, markersize=10, 
                 label='Model Only (MMLU-Aug)', color='brown', linestyle=':', alpha=0.8)
    
    # Styling for Plot 2
    plt.xlabel('Total Number of Distractors', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Accuracy vs Total Number of Distractors\n(Choices Only)', fontsize=20, fontweight='bold')
    plt.legend(loc='upper right', fontsize=12, markerscale=1.0, frameon=True, 
               bbox_to_anchor=(1.0, 1.0), ncol=1)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 10)
    plt.ylim(0.35, 0.8)
    plt.tight_layout()
    
    if output_dir:
        output_path = output_dir / "rq1_choices_only.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_rq2_human_distractors(
    base_dir: Path,
    output_dir: Optional[Path] = None,
    show: bool = False,
):
    """
    RQ2: Human-only distractor comparison (MMLU-Pro vs MMLU-Aug).
    
    Creates two plots: Full Questions and Choices Only.
    """
    base_dir = Path(base_dir)
    
    # Load Human Only results
    results_normal_full = load_human_only_results(base_dir, "normal", False)
    results_normal_choices = load_human_only_results(base_dir, "normal", True)
    results_aug_full = load_human_only_results(base_dir, "augmented", False)
    results_aug_choices = load_human_only_results(base_dir, "augmented", True)
    
    # ===== PLOT 1: Full Questions =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Full Questions
    if results_aug_full:
        x = [r['total_distractors'] for r in results_aug_full]
        y = [r['accuracy'] for r in results_aug_full]
        ax1.plot(x, y, marker='s', linewidth=4, markersize=14, 
                 label='MMLU-Aug (Augmented)', color='red', linestyle='-', alpha=0.7, zorder=2)
    
    if results_normal_full:
        x = [r['total_distractors'] + 0.05 for r in results_normal_full]  # Offset for visibility
        y = [r['accuracy'] for r in results_normal_full]
        ax1.plot(x, y, marker='o', linewidth=4, markersize=14, 
                 label='MMLU-Pro (Normal)', color='blue', linestyle='--', alpha=0.7, zorder=3,
                 markeredgewidth=2, markeredgecolor='darkblue', markerfacecolor='lightblue')
    
    ax1.set_xlabel('Number of Human Distractors', fontsize=18)
    ax1.set_ylabel('Accuracy', fontsize=18)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.set_title('Human-Only Distractors\n(Full Questions with Stem)', fontsize=20, fontweight='bold')
    ax1.legend(loc='best', fontsize=16, frameon=True)
    ax1.grid(True, alpha=0.3, linewidth=1.5)
    ax1.set_xlim(0.5, 3.5)
    ax1.set_xticks([1, 2, 3])
    
    # Choices Only
    if results_aug_choices:
        x = [r['total_distractors'] for r in results_aug_choices]
        y = [r['accuracy'] for r in results_aug_choices]
        ax2.plot(x, y, marker='s', linewidth=4, markersize=14, 
                 label='MMLU-Aug (Augmented)', color='red', linestyle='-', alpha=0.7, zorder=2)
    
    if results_normal_choices:
        x = [r['total_distractors'] + 0.05 for r in results_normal_choices]
        y = [r['accuracy'] for r in results_normal_choices]
        ax2.plot(x, y, marker='o', linewidth=4, markersize=14, 
                 label='MMLU-Pro (Normal)', color='blue', linestyle='--', alpha=0.7, zorder=3,
                 markeredgewidth=2, markeredgecolor='darkblue', markerfacecolor='lightblue')
    
    ax2.set_xlabel('Number of Human Distractors', fontsize=18)
    ax2.set_ylabel('Accuracy', fontsize=18)
    ax2.tick_params(axis='both', labelsize=16)
    ax2.set_title('Human-Only Distractors\n(Choices Only, No Stem)', fontsize=20, fontweight='bold')
    ax2.legend(loc='best', fontsize=16, frameon=True)
    ax2.grid(True, alpha=0.3, linewidth=1.5)
    ax2.set_xlim(0.5, 3.5)
    ax2.set_xticks([1, 2, 3])
    
    fig.suptitle('RQ2: Human-Only Distractor Analysis (1H-3H)', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "rq2_human_only.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_rq3_model_distractors(
    base_dir: Path,
    output_dir: Optional[Path] = None,
    show: bool = False,
):
    """
    RQ3: Model-only distractor comparison (MMLU-Pro vs MMLU-Aug).
    
    Creates two plots: Full Questions and Choices Only.
    """
    base_dir = Path(base_dir)
    
    # Load Model Only results
    results_normal_full = load_model_only_results(base_dir, "normal", False)
    results_normal_choices = load_model_only_results(base_dir, "normal", True)
    results_aug_full = load_model_only_results(base_dir, "augmented", False)
    results_aug_choices = load_model_only_results(base_dir, "augmented", True)
    
    # ===== Combined Plot =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Full Questions
    if results_aug_full:
        x = [r['total_distractors'] for r in results_aug_full]
        y = [r['accuracy'] for r in results_aug_full]
        ax1.plot(x, y, marker='s', linewidth=4, markersize=14, 
                 label='MMLU-Aug (Augmented)', color='red', linestyle='-', alpha=0.7, zorder=2)
    
    if results_normal_full:
        x = [r['total_distractors'] + 0.05 for r in results_normal_full]
        y = [r['accuracy'] for r in results_normal_full]
        ax1.plot(x, y, marker='o', linewidth=4, markersize=14, 
                 label='MMLU-Pro (Normal)', color='blue', linestyle='--', alpha=0.7, zorder=3,
                 markeredgewidth=2, markeredgecolor='darkblue', markerfacecolor='lightblue')
    
    ax1.set_xlabel('Number of Model Distractors', fontsize=18)
    ax1.set_ylabel('Accuracy', fontsize=18)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.set_title('Model-Only Distractors\n(Full Questions with Stem)', fontsize=20, fontweight='bold')
    ax1.legend(loc='best', fontsize=16, frameon=True)
    ax1.grid(True, alpha=0.3, linewidth=1.5)
    ax1.set_xlim(0.5, 6.5)
    ax1.set_xticks([1, 2, 3, 4, 5, 6])
    
    # Choices Only
    if results_aug_choices:
        x = [r['total_distractors'] for r in results_aug_choices]
        y = [r['accuracy'] for r in results_aug_choices]
        ax2.plot(x, y, marker='s', linewidth=4, markersize=14, 
                 label='MMLU-Aug (Augmented)', color='red', linestyle='-', alpha=0.7, zorder=2)
    
    if results_normal_choices:
        x = [r['total_distractors'] + 0.05 for r in results_normal_choices]
        y = [r['accuracy'] for r in results_normal_choices]
        ax2.plot(x, y, marker='o', linewidth=4, markersize=14, 
                 label='MMLU-Pro (Normal)', color='blue', linestyle='--', alpha=0.7, zorder=3,
                 markeredgewidth=2, markeredgecolor='darkblue', markerfacecolor='lightblue')
    
    ax2.set_xlabel('Number of Model Distractors', fontsize=18)
    ax2.set_ylabel('Accuracy', fontsize=18)
    ax2.tick_params(axis='both', labelsize=16)
    ax2.set_title('Model-Only Distractors\n(Choices Only, No Stem)', fontsize=20, fontweight='bold')
    ax2.legend(loc='best', fontsize=16, frameon=True)
    ax2.grid(True, alpha=0.3, linewidth=1.5)
    ax2.set_xlim(0.5, 6.5)
    ax2.set_xticks([1, 2, 3, 4, 5, 6])
    
    fig.suptitle('RQ3: Model-Only Distractor Analysis (1M-6M)', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "rq3_model_only.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_all_rq(
    base_dir: Path,
    output_dir: Optional[Path] = None,
    show: bool = False,
):
    """
    Generate all 6 RQ plots.
    
    Args:
        base_dir: Directory containing experiment results
        output_dir: Where to save plots (default: base_dir/plots)
        show: Whether to display plots interactively
    """
    base_dir = Path(base_dir)
    
    if output_dir is None:
        output_dir = base_dir / "plots"
    
    print("Generating RQ1: Combined Analysis...")
    plot_rq1_combined(base_dir, output_dir, show)
    
    print("Generating RQ2: Human-Only Distractor Analysis...")
    plot_rq2_human_distractors(base_dir, output_dir, show)
    
    print("Generating RQ3: Model-Only Distractor Analysis...")
    plot_rq3_model_distractors(base_dir, output_dir, show)
    
    print(f"\nAll plots saved to: {output_dir}")
