"""
ARC Dataset Processor.

Processes ARC-Easy and ARC-Challenge datasets with:
1. Exact column parsing based on HuggingFace structure
2. Conversion to unified format
3. Synthetic distractor generation (cond_model_q_a)
4. Conditioned-on-synthetic distractor generation (cond_model_q_a_dmodel)
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from tqdm import tqdm

from config import (
    DATASETS_DIR,
    DatasetType,
    DATASET_SCHEMA,
    DistractorType,
    get_options_from_entry,
    get_answer_index,
)


def load_arc_dataset(
    difficulty: str = "easy",
    split: str = "test",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load ARC dataset with proper column parsing.
    
    Args:
        difficulty: "easy" for ARC-Easy, "challenge" for ARC-Challenge
        split: Dataset split to load
        limit: Optional limit on number of entries
        
    Returns:
        List of entries in unified format
    """
    dataset_type = DatasetType.ARC_EASY if difficulty == "easy" else DatasetType.ARC_CHALLENGE
    schema = DATASET_SCHEMA[dataset_type]
    
    # Load from HuggingFace
    ds = load_dataset(
        schema["hf_path"],
        schema["hf_config"],
        split=split,
        trust_remote_code=True,
    )
    
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    
    # Convert to unified format
    entries = []
    for entry in tqdm(ds, desc=f"Loading ARC-{difficulty.capitalize()}"):
        # Extract options from nested dict
        options = entry["choices"]["text"]
        labels = entry["choices"]["label"]
        
        # Get answer index from letter
        answer_letter = entry["answerKey"]
        try:
            answer_index = labels.index(answer_letter)
        except ValueError:
            # Handle edge case where answerKey might be numeric
            answer_index = int(answer_letter) - 1 if answer_letter.isdigit() else 0
        
        unified_entry = {
            "id": entry["id"],
            "question": entry["question"],
            "options": options,
            "labels": labels,
            "gold_answer": options[answer_index] if answer_index < len(options) else options[0],
            "answer_index": answer_index,
            "answer_letter": answer_letter,
            "dataset_type": dataset_type.value,
            # Original distractors (all non-gold options)
            DistractorType.COND_HUMAN_Q_A.value: [
                opt for i, opt in enumerate(options) if i != answer_index
            ],
        }
        entries.append(unified_entry)
    
    return entries


def process_arc_for_experiments(
    difficulty: str = "easy",
    split: str = "test",
    limit: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Process ARC dataset and save in unified format for experiments.
    
    Args:
        difficulty: "easy" or "challenge"
        split: Dataset split
        limit: Optional limit
        output_dir: Output directory (default: DATASETS_DIR)
        
    Returns:
        Path to saved dataset
    """
    entries = load_arc_dataset(difficulty, split, limit)
    
    if output_dir is None:
        output_dir = DATASETS_DIR
    
    # Use dataset type prefix in output
    prefix = f"arc_{difficulty}"
    output_path = output_dir / prefix / f"{split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)
    
    print(f"Saved {len(entries)} entries to {output_path}")
    return output_path


def add_synthetic_distractors_to_arc(
    entries: List[Dict[str, Any]],
    synthetic_distractors: List[List[str]],
    distractor_type: DistractorType = DistractorType.COND_MODEL_Q_A,
) -> List[Dict[str, Any]]:
    """
    Add synthetic distractors to ARC entries.
    
    Args:
        entries: List of ARC entries
        synthetic_distractors: List of distractor lists (one per entry)
        distractor_type: Type of distractors being added
        
    Returns:
        Updated entries with synthetic distractors
    """
    if len(synthetic_distractors) != len(entries):
        raise ValueError(
            f"Mismatch: {len(entries)} entries but {len(synthetic_distractors)} distractor lists"
        )
    
    for entry, distractors in zip(entries, synthetic_distractors):
        entry[distractor_type.value] = distractors
    
    return entries


def get_arc_stats(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get statistics about processed ARC entries."""
    stats = {
        "total_entries": len(entries),
        "has_human_distractors": sum(
            1 for e in entries if DistractorType.COND_HUMAN_Q_A.value in e
        ),
        "has_model_distractors": sum(
            1 for e in entries if DistractorType.COND_MODEL_Q_A.value in e
        ),
        "has_conditioned_distractors": sum(
            1 for e in entries if DistractorType.COND_MODEL_Q_A_DMODEL.value in e
        ),
    }
    return stats


if __name__ == "__main__":
    # Test loading
    print("Testing ARC-Easy loading...")
    easy_entries = load_arc_dataset("easy", limit=5)
    print(f"Loaded {len(easy_entries)} entries")
    print(f"Sample entry keys: {list(easy_entries[0].keys())}")
    print(f"Sample question: {easy_entries[0]['question'][:100]}...")
    print(f"Sample options: {easy_entries[0]['options']}")
    print(f"Gold answer: {easy_entries[0]['gold_answer']}")
    
    print("\nTesting ARC-Challenge loading...")
    challenge_entries = load_arc_dataset("challenge", limit=5)
    print(f"Loaded {len(challenge_entries)} entries")
