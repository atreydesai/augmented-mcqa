"""
SuperGPQA Dataset Processor.

Processes SuperGPQA dataset with:
1. Filtering to 10-option questions only (87.3% of dataset)
2. Exact column parsing based on HuggingFace structure
3. Conversion to unified format
4. Synthetic distractor generation support
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from config import (
    DATASETS_DIR,
    PROCESSED_DATASETS_DIR,
    DatasetType,
    DATASET_SCHEMA,
    DistractorType,
)


def load_supergpqa_dataset(
    split: str = "train",
    limit: Optional[int] = None,
    filter_10_options: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load SuperGPQA dataset with proper column parsing.
    
    Note: SuperGPQA only has a 'train' split on HuggingFace.
    
    Args:
        split: Dataset split (only 'train' is available)
        limit: Optional limit on number of entries
        filter_10_options: If True, only include questions with exactly 10 options
        
    Returns:
        List of entries in unified format
    """
    schema = DATASET_SCHEMA[DatasetType.SUPERGPQA]
    
    # Download JSONL directly to avoid HF cache issues
    file_path = hf_hub_download(
        repo_id="m-a-p/SuperGPQA",
        filename="SuperGPQA-all.jsonl",
        repo_type="dataset",
    )
    
    # Read and parse
    entries = []
    with open(file_path, "r") as f:
        for line in tqdm(f, desc="Loading SuperGPQA"):
            raw = json.loads(line)
            
            # Filter to 10 options if requested
            options = raw.get("options", [])
            if filter_10_options and len(options) != 10:
                continue
            
            # Get answer index from letter
            answer_letter = raw.get("answer_letter", "A")
            answer_index = ord(answer_letter.upper()) - ord('A') if answer_letter else 0
            
            # Build unified entry
            unified_entry = {
                "id": raw.get("uuid", ""),
                "question": raw.get("question", ""),
                "options": options,
                "answer": raw.get("answer", options[answer_index] if answer_index < len(options) else ""),
                "choices_answer": [raw.get("answer", options[answer_index] if answer_index < len(options) else "")],
                "answer_index": answer_index,
                "answer_letter": answer_letter,
                "dataset_type": DatasetType.SUPERGPQA.value,
                "category": raw.get("field", ""),
                "discipline": raw.get("discipline", ""),
                "subfield": raw.get("subfield", ""),
                "difficulty": raw.get("difficulty", "middle"),
                "is_calculation": raw.get("is_calculation", False),
                # Rename cond_human_q_a -> choices_human
                "choices_human": [
                    opt for i, opt in enumerate(options) if i != answer_index
                ],
            }
            entries.append(unified_entry)
            
            if limit and len(entries) >= limit:
                break
    
    return entries


def process_supergpqa_for_experiments(
    split: str = "train",
    limit: Optional[int] = None,
    filter_10_options: bool = True,
    output_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Any:
    """
    Process SuperGPQA dataset and save as HF Dataset format.
    
    Args:
        split: Dataset split
        limit: Optional limit
        filter_10_options: If True, only include 10-option questions
        output_dir: Output base directory
        output_path: Exact output directory path (overrides output_dir)
        
    Returns:
        Processed Dataset
    """
    from datasets import Dataset
    entries = load_supergpqa_dataset(split, limit, filter_10_options)
    
    # Convert to HF Dataset for standardization
    dataset = Dataset.from_list(entries)
    
    if output_path is None:
        if output_dir is None:
            output_dir = PROCESSED_DATASETS_DIR
        
        # Default path structure: output_dir/supergpqa_processed
        output_path = output_dir / "supergpqa_processed"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as HF Dataset
    dataset.save_to_disk(str(output_path))
    print(f"Saved {len(entries)} entries to {output_path}")
    
    # Push to Hugging Face
    from data.hub_utils import push_dataset_to_hub
    push_dataset_to_hub(dataset, repo_id="atreydesai/qgqa-supergpqa-processed")
    
    return dataset


def add_synthetic_distractors_to_supergpqa(
    entries: List[Dict[str, Any]],
    synthetic_distractors: List[List[str]],
    distractor_type: DistractorType = DistractorType.COND_MODEL_Q_A,
) -> List[Dict[str, Any]]:
    """
    Add synthetic distractors to SuperGPQA entries.
    
    Args:
        entries: List of SuperGPQA entries
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


def get_supergpqa_stats(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get statistics about processed SuperGPQA entries."""
    
    # Count by difficulty
    difficulty_counts = {}
    discipline_counts = {}
    
    for e in entries:
        diff = e.get("difficulty", "unknown")
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        
        disc = e.get("discipline", "unknown")
        discipline_counts[disc] = discipline_counts.get(disc, 0) + 1
    
    stats = {
        "total_entries": len(entries),
        "by_difficulty": difficulty_counts,
        "by_discipline": dict(sorted(
            discipline_counts.items(), 
            key=lambda x: -x[1]
        )[:10]),
        "has_human_distractors": sum(
            1 for e in entries if DistractorType.COND_HUMAN_Q_A.value in e
        ),
        "has_model_distractors": sum(
            1 for e in entries if DistractorType.COND_MODEL_Q_A.value in e
        ),
    }
    return stats


def filter_by_difficulty(
    entries: List[Dict[str, Any]],
    difficulty: str,
) -> List[Dict[str, Any]]:
    """Filter SuperGPQA entries by difficulty level."""
    return [e for e in entries if e.get("difficulty", "").lower() == difficulty.lower()]


def filter_by_discipline(
    entries: List[Dict[str, Any]],
    discipline: str,
) -> List[Dict[str, Any]]:
    """Filter SuperGPQA entries by discipline."""
    return [e for e in entries if e.get("discipline", "").lower() == discipline.lower()]


if __name__ == "__main__":
    # Test loading
    print("Testing SuperGPQA loading (10-option only)...")
    entries = load_supergpqa_dataset(limit=100, filter_10_options=True)
    print(f"Loaded {len(entries)} entries")
    
    if entries:
        print(f"\nSample entry keys: {list(entries[0].keys())}")
        print(f"Sample question: {entries[0]['question'][:150]}...")
        print(f"Sample options count: {len(entries[0]['options'])}")
        print(f"Gold answer: {entries[0]['gold_answer'][:100]}...")
        print(f"Difficulty: {entries[0]['difficulty']}")
        print(f"Discipline: {entries[0]['discipline']}")
        
        stats = get_supergpqa_stats(entries)
        print(f"\nStats: {json.dumps(stats, indent=2)}")
