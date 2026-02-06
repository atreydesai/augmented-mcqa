"""
Difficulty scaling module for Augmented MCQA.

Provides utilities for evaluating model performance across difficulty levels:
- ARC: Easy (ARC-Easy) and Challenge (ARC-Challenge) splits
- SuperGPQA: Hard graduate-level questions
- MMLU-Pro: Standard difficulty

This enables RQ4: Does the distractor effect scale with question difficulty?
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass
from enum import Enum
import json

from config import DATASETS_DIR, RESULTS_DIR


class DifficultyLevel(Enum):
    """Question difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class DifficultyDataset:
    """Configuration for a difficulty-specific dataset."""
    name: str
    level: DifficultyLevel
    path: Path
    num_options: int  # ARC has 4, MMLU-Pro has 10, SuperGPQA varies
    source: str  # "arc", "mmlu_pro", "supergpqa"
    split: Optional[str] = None  # For datasets with multiple splits


# Standard difficulty scaling configuration
DIFFICULTY_DATASETS = {
    # Easy tier (ARC-Easy)
    "arc_easy": DifficultyDataset(
        name="ARC-Easy",
        level=DifficultyLevel.EASY,
        path=DATASETS_DIR / "arc" / "easy",
        num_options=4,
        source="arc",
        split="easy",
    ),
    # Medium tier (ARC-Challenge, MMLU-Pro)
    "arc_challenge": DifficultyDataset(
        name="ARC-Challenge",
        level=DifficultyLevel.MEDIUM,
        path=DATASETS_DIR / "arc" / "challenge",
        num_options=4,
        source="arc",
        split="challenge",
    ),
    "mmlu_pro": DifficultyDataset(
        name="MMLU-Pro",
        level=DifficultyLevel.MEDIUM,
        path=DATASETS_DIR / "mmlu_pro_sorted",
        num_options=10,
        source="mmlu_pro",
    ),
    # Hard tier (SuperGPQA)
    "supergpqa": DifficultyDataset(
        name="SuperGPQA",
        level=DifficultyLevel.HARD,
        path=DATASETS_DIR / "supergpqa",
        num_options=4,  # Typically 4 options
        source="supergpqa",
    ),
}


def load_arc_dataset(path: Path, split: str = "test") -> List[Dict]:
    """
    Load ARC dataset from disk.
    
    ARC format:
    - id: Question ID
    - question: Question text
    - choices: List of {"label": "A", "text": "..."} 
    - answerKey: Correct answer label (A/B/C/D)
    """
    from datasets import load_from_disk
    
    dataset = load_from_disk(str(path))
    
    entries = []
    for entry in dataset[split] if split in dataset else dataset:
        # Convert to unified format
        choices = entry.get("choices", {})
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        
        options = []
        for label, text in zip(labels, texts):
            options.append({"label": label, "text": text})
        
        # Find gold answer index
        answer_key = entry.get("answerKey", "A")
        gold_idx = labels.index(answer_key) if answer_key in labels else 0
        
        entries.append({
            "id": entry.get("id", ""),
            "question": entry.get("question", ""),
            "options": options,
            "gold_answer": texts[gold_idx] if gold_idx < len(texts) else "",
            "gold_index": gold_idx,
            "source": "arc",
        })
    
    return entries


def load_supergpqa_dataset(path: Path, split: str = "test") -> List[Dict]:
    """
    Load SuperGPQA dataset from disk.
    
    SuperGPQA format varies but typically includes:
    - question: Question text
    - choices/options: Answer choices
    - answer: Correct answer
    """
    from datasets import load_from_disk
    
    dataset = load_from_disk(str(path))
    
    entries = []
    for entry in dataset[split] if split in dataset else dataset:
        # Handle different possible formats
        question = entry.get("question", "")
        
        # Try different option field names
        options = []
        if "choices" in entry:
            choices = entry["choices"]
            if isinstance(choices, list):
                options = choices
            elif isinstance(choices, dict):
                options = list(choices.values())
        elif "options" in entry:
            options = entry["options"]
        elif "A" in entry:
            # Some formats use A, B, C, D directly
            options = [entry.get(k, "") for k in ["A", "B", "C", "D"] if k in entry]
        
        # Get gold answer
        answer = entry.get("answer", entry.get("correct_answer", "A"))
        
        # Convert to index
        if isinstance(answer, str) and len(answer) == 1:
            gold_idx = ord(answer.upper()) - ord("A")
        elif isinstance(answer, int):
            gold_idx = answer
        else:
            gold_idx = 0
        
        gold_idx = min(gold_idx, len(options) - 1) if options else 0
        
        entries.append({
            "id": entry.get("id", entry.get("question_id", "")),
            "question": question,
            "options": options,
            "gold_answer": options[gold_idx] if gold_idx < len(options) else "",
            "gold_index": gold_idx,
            "source": "supergpqa",
            "category": entry.get("category", entry.get("field", "general")),
        })
    
    return entries


def prepare_difficulty_evaluation(
    difficulty_level: DifficultyLevel,
    limit: Optional[int] = None,
) -> Dict[str, List[Dict]]:
    """
    Prepare datasets for a specific difficulty level.
    
    Args:
        difficulty_level: EASY, MEDIUM, or HARD
        limit: Optional limit on number of entries per dataset
        
    Returns:
        Dict mapping dataset name to list of entries
    """
    datasets = {}
    
    for key, config in DIFFICULTY_DATASETS.items():
        if config.level != difficulty_level:
            continue
        
        if not config.path.exists():
            print(f"Dataset not found: {config.path}")
            continue
        
        if config.source == "arc":
            entries = load_arc_dataset(config.path)
        elif config.source == "supergpqa":
            entries = load_supergpqa_dataset(config.path)
        else:
            # Use generic loader
            from datasets import load_from_disk
            dataset = load_from_disk(str(config.path))
            entries = [dict(e) for e in dataset]
        
        if limit:
            entries = entries[:limit]
        
        datasets[key] = entries
        print(f"Loaded {len(entries)} entries from {config.name}")
    
    return datasets


def compute_difficulty_comparison(
    results_by_difficulty: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Compute comparison metrics across difficulty levels.
    
    Args:
        results_by_difficulty: Dict mapping difficulty level to results dict
        
    Returns:
        Comparison analysis
    """
    analysis = {
        "by_difficulty": {},
        "trends": {},
    }
    
    for level, results in results_by_difficulty.items():
        accuracy = results.get("accuracy", 0)
        analysis["by_difficulty"][level] = {
            "accuracy": accuracy,
            "total": results.get("total", 0),
        }
    
    # Compute trends (accuracy drop from easy to hard)
    levels = ["easy", "medium", "hard"]
    accuracies = [
        analysis["by_difficulty"].get(l, {}).get("accuracy", 0)
        for l in levels
    ]
    
    if accuracies[0] > 0 and accuracies[-1] > 0:
        analysis["trends"]["accuracy_drop"] = accuracies[0] - accuracies[-1]
        analysis["trends"]["relative_drop"] = (accuracies[0] - accuracies[-1]) / accuracies[0]
    
    return analysis


def save_difficulty_results(
    results: Dict[str, Any],
    experiment_name: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Save difficulty scaling experiment results.
    
    Args:
        results: Results dictionary
        experiment_name: Name for this experiment
        output_dir: Output directory (default: RESULTS_DIR/difficulty)
        
    Returns:
        Path to saved results
    """
    if output_dir is None:
        output_dir = RESULTS_DIR / "difficulty" / experiment_name
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "difficulty_results.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Saved difficulty results to: {output_path}")
    return output_path
