"""
Difficulty Scaling Experiments Module.

Supports cross-difficulty comparison experiments:
- ARC-Easy (easy, 4 options)
- ARC-Challenge / MMLU-Pro (medium, 4/10 options)
- SuperGPQA (hard, 10 options)

All loaders use exact column parsing based on analyzed HuggingFace structures.
"""

import json
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from config import (
    DATASETS_DIR,
    RESULTS_DIR,
    DatasetType,
    DATASET_SCHEMA,
    DistractorType,
)
from data.arc_processor import load_arc_dataset, get_arc_stats
from data.supergpqa_processor import (
    load_supergpqa_dataset,
    get_supergpqa_stats,
    filter_by_difficulty,
)


class DifficultyLevel(Enum):
    """Difficulty levels for experiments."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class DifficultyDataset:
    """Configuration for a difficulty-level dataset."""
    name: str
    dataset_type: DatasetType
    difficulty: DifficultyLevel
    num_options: int
    description: str
    
    # Loader arguments
    loader_kwargs: Dict[str, Any] = field(default_factory=dict)


# Pre-configured difficulty datasets
DIFFICULTY_DATASETS = {
    # Easy tier
    "arc_easy": DifficultyDataset(
        name="arc_easy",
        dataset_type=DatasetType.ARC_EASY,
        difficulty=DifficultyLevel.EASY,
        num_options=4,
        description="ARC-Easy: Elementary-level science questions",
        loader_kwargs={"difficulty": "easy"},
    ),
    
    # Medium tier
    "arc_challenge": DifficultyDataset(
        name="arc_challenge",
        dataset_type=DatasetType.ARC_CHALLENGE,
        difficulty=DifficultyLevel.MEDIUM,
        num_options=4,
        description="ARC-Challenge: Harder science questions requiring reasoning",
        loader_kwargs={"difficulty": "challenge"},
    ),
    "mmlu_pro": DifficultyDataset(
        name="mmlu_pro",
        dataset_type=DatasetType.MMLU_PRO,
        difficulty=DifficultyLevel.MEDIUM,
        num_options=10,
        description="MMLU-Pro: Extended MMLU with 10 options per question",
        loader_kwargs={},
    ),
    
    # Hard tier
    "supergpqa": DifficultyDataset(
        name="supergpqa",
        dataset_type=DatasetType.SUPERGPQA,
        difficulty=DifficultyLevel.HARD,
        num_options=10,
        description="SuperGPQA: Graduate-level expert questions",
        loader_kwargs={"filter_10_options": True},
    ),
}


def load_difficulty_dataset(
    dataset_name: str,
    split: str = "test",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load a dataset by name with proper column parsing.
    
    Args:
        dataset_name: One of the DIFFICULTY_DATASETS keys
        split: Dataset split to load
        limit: Optional limit on entries
        
    Returns:
        List of entries in unified format
    """
    if dataset_name not in DIFFICULTY_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DIFFICULTY_DATASETS.keys())}"
        )
    
    config = DIFFICULTY_DATASETS[dataset_name]
    
    # Route to appropriate loader
    if config.dataset_type in (DatasetType.ARC_EASY, DatasetType.ARC_CHALLENGE):
        return load_arc_dataset(
            difficulty=config.loader_kwargs.get("difficulty", "easy"),
            split=split,
            limit=limit,
        )
    
    elif config.dataset_type == DatasetType.SUPERGPQA:
        # SuperGPQA only has train split
        actual_split = "train" if split == "test" else split
        return load_supergpqa_dataset(
            split=actual_split,
            limit=limit,
            filter_10_options=config.loader_kwargs.get("filter_10_options", True),
        )
    
    elif config.dataset_type == DatasetType.MMLU_PRO:
        # Load from disk (Dataset object)
        from datasets import load_from_disk
        dataset = load_from_disk(str(DATASETS_DIR / "mmlu_pro_processed"))
        
        # Determine split
        if split in dataset:
            entries = list(dataset[split])
        else:
            entries = list(dataset.values())[0] if hasattr(dataset, "values") else list(dataset)
            
        if limit:
            entries = entries[:limit]
            
        # Add dataset_type
        for e in entries:
            e["dataset_type"] = DatasetType.MMLU_PRO.value
            
        return entries
    
    else:
        raise ValueError(f"No loader implemented for {config.dataset_type}")


def prepare_difficulty_evaluation(
    dataset_name: str,
    num_human: int = 3,
    num_model: int = 0,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Prepare a difficulty dataset for evaluation.
    
    Args:
        dataset_name: One of the DIFFICULTY_DATASETS keys
        num_human: Number of human/original distractors to include
        num_model: Number of model/synthetic distractors to include
        limit: Optional limit on entries
        
    Returns:
        List of entries ready for evaluation
    """
    entries = load_difficulty_dataset(dataset_name, limit=limit)
    config = DIFFICULTY_DATASETS[dataset_name]
    
    # Build options for each entry
    for entry in entries:
        # Get available distractors
        human_distractors = entry.get(DistractorType.COND_HUMAN_Q_A.value, [])
        model_distractors = entry.get(DistractorType.COND_MODEL_Q_A.value, [])
        
        # Select distractors
        selected_human = human_distractors[:num_human] if human_distractors else []
        selected_model = model_distractors[:num_model] if model_distractors else []
        
        # Build final options: gold + selected distractors
        gold = entry.get("gold_answer", "")
        entry["eval_options"] = [gold] + selected_human + selected_model
        entry["eval_num_options"] = len(entry["eval_options"])
        entry["distractor_config"] = f"{num_human}H{num_model}M"
    
    return entries


def compute_difficulty_comparison(
    results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute comparative statistics across difficulty levels.
    
    Args:
        results: Dict mapping dataset_name -> evaluation results
        
    Returns:
        Comparison statistics
    """
    comparison = {
        "by_difficulty": {},
        "accuracy_trend": [],
        "timestamp": datetime.now().isoformat(),
    }
    
    for diff_level in DifficultyLevel:
        level_results = [
            (name, res) for name, res in results.items()
            if DIFFICULTY_DATASETS.get(name, {}).difficulty == diff_level
        ]
        
        if level_results:
            accuracies = [res.get("accuracy", 0) for _, res in level_results]
            comparison["by_difficulty"][diff_level.value] = {
                "datasets": [name for name, _ in level_results],
                "mean_accuracy": sum(accuracies) / len(accuracies),
                "accuracies": dict(level_results),
            }
    
    # Build accuracy trend (easy -> medium -> hard)
    for level in [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]:
        level_data = comparison["by_difficulty"].get(level.value, {})
        if level_data:
            comparison["accuracy_trend"].append({
                "difficulty": level.value,
                "mean_accuracy": level_data.get("mean_accuracy", 0),
            })
    
    return comparison


def save_difficulty_results(
    results: Dict[str, Any],
    dataset_name: str,
    output_dir: Optional[Path] = None,
    distractor_config: str = "3H0M",
) -> Path:
    """
    Save difficulty experiment results with dataset type prefix.
    
    Args:
        results: Results dictionary
        dataset_name: Name of dataset (for prefix)
        output_dir: Output directory (default: RESULTS_DIR)
        distractor_config: Distractor configuration string
        
    Returns:
        Path to saved results
    """
    if output_dir is None:
        output_dir = RESULTS_DIR / "difficulty_scaling"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use dataset type prefix in filename
    filename = f"{dataset_name}_{distractor_config}_results.json"
    output_path = output_dir / filename
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Saved results to {output_path}")
    return output_path


def get_dataset_stats(dataset_name: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """Get statistics for a difficulty dataset."""
    config = DIFFICULTY_DATASETS[dataset_name]
    entries = load_difficulty_dataset(dataset_name, limit=limit)
    
    if config.dataset_type in (DatasetType.ARC_EASY, DatasetType.ARC_CHALLENGE):
        return get_arc_stats(entries)
    elif config.dataset_type == DatasetType.SUPERGPQA:
        return get_supergpqa_stats(entries)
    else:
        return {
            "total_entries": len(entries),
            "dataset_type": config.dataset_type.value,
            "difficulty": config.difficulty.value,
        }


if __name__ == "__main__":
    # Test all loaders
    print("Testing difficulty dataset loaders...\n")
    
    for name, config in DIFFICULTY_DATASETS.items():
        print(f"\n{'='*50}")
        print(f"{name} ({config.difficulty.value})")
        print(f"{'='*50}")
        
        try:
            entries = load_difficulty_dataset(name, limit=5)
            print(f"Loaded {len(entries)} entries")
            
            if entries:
                print(f"Sample question: {entries[0]['question'][:100]}...")
                print(f"Options count: {len(entries[0].get('options', []))}")
                print(f"Gold answer: {entries[0].get('gold_answer', '')[:50]}...")
                
        except Exception as e:
            print(f"Error: {e}")
