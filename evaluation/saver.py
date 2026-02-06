"""
Results saving module for Augmented MCQA.

Handles saving evaluation results to JSON and optionally to HuggingFace Hub.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from config import RESULTS_DIR, HF_SKIP_PUSH


def save_results_json(
    results: Dict[str, Any],
    output_path: Path,
    pretty: bool = True,
) -> Path:
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Output file path
        pretty: Whether to format JSON with indentation
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        if pretty:
            json.dump(results, f, indent=2, default=str)
        else:
            json.dump(results, f, default=str)
    
    return output_path


def save_results_csv(
    results: List[Dict[str, Any]],
    output_path: Path,
    columns: Optional[List[str]] = None,
) -> Path:
    """
    Save results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_path: Output file path
        columns: Column names to include (default: all)
        
    Returns:
        Path to saved file
    """
    import csv
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not results:
        # Create empty file
        output_path.touch()
        return output_path
    
    # Get columns from first result if not specified
    if columns is None:
        columns = list(results[0].keys())
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    return output_path


def create_results_summary(
    experiment_name: str,
    model_name: str,
    config: Dict[str, Any],
    accuracy: float,
    behavioral_counts: Optional[Dict[str, int]] = None,
    accuracy_by_category: Optional[Dict[str, float]] = None,
    total_entries: int = 0,
    duration_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Create a standardized results summary.
    
    Args:
        experiment_name: Name of the experiment
        model_name: Model identifier
        config: Experiment configuration
        accuracy: Overall accuracy
        behavioral_counts: G/H/M prediction counts
        accuracy_by_category: Accuracy per category
        total_entries: Total number of evaluated entries
        duration_seconds: Evaluation duration
        
    Returns:
        Standardized results summary dictionary
    """
    summary = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "accuracy": accuracy,
            "total_entries": total_entries,
        },
    }
    
    if behavioral_counts:
        summary["metrics"]["behavioral_counts"] = behavioral_counts
        total = sum(behavioral_counts.values())
        if total > 0:
            summary["metrics"]["behavioral_rates"] = {
                k: v / total for k, v in behavioral_counts.items()
            }
    
    if accuracy_by_category:
        summary["metrics"]["accuracy_by_category"] = accuracy_by_category
    
    if duration_seconds:
        summary["duration_seconds"] = duration_seconds
        summary["entries_per_second"] = total_entries / duration_seconds if duration_seconds > 0 else 0
    
    return summary


def save_experiment_results(
    experiment_name: str,
    model_name: str,
    config: Dict[str, Any],
    individual_results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Save all experiment results (summary + individual results).
    
    Args:
        experiment_name: Name of the experiment
        model_name: Model identifier
        config: Experiment configuration
        individual_results: List of per-entry results
        summary: Summary metrics
        output_dir: Output directory (default: RESULTS_DIR/experiment_name)
        
    Returns:
        Dict mapping result type to saved path
    """
    if output_dir is None:
        output_dir = RESULTS_DIR / experiment_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    
    # Save config
    config_path = save_results_json(config, output_dir / "config.json")
    saved_paths["config"] = config_path
    
    # Save summary
    summary_path = save_results_json(summary, output_dir / "summary.json")
    saved_paths["summary"] = summary_path
    
    # Save individual results
    results_path = save_results_json(
        {"results": individual_results},
        output_dir / "results.json",
    )
    saved_paths["results"] = results_path
    
    # Save CSV version
    if individual_results:
        csv_path = save_results_csv(
            individual_results,
            output_dir / "results.csv",
        )
        saved_paths["csv"] = csv_path
    
    return saved_paths


def push_to_hub(
    dataset_path: Path,
    repo_id: str,
    private: bool = True,
) -> bool:
    """
    Push results dataset to HuggingFace Hub.
    
    Args:
        dataset_path: Path to dataset directory
        repo_id: HuggingFace repository ID
        private: Whether to make the repo private
        
    Returns:
        True if successful, False if skipped or failed
    """
    if HF_SKIP_PUSH:
        print("Skipping HuggingFace push (HF_SKIP_PUSH=1)")
        return False
    
    try:
        from datasets import load_from_disk
        
        dataset = load_from_disk(str(dataset_path))
        dataset.push_to_hub(repo_id, private=private)
        print(f"Pushed to HuggingFace Hub: {repo_id}")
        return True
        
    except Exception as e:
        print(f"Failed to push to Hub: {e}")
        return False
