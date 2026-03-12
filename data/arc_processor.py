"""ARC dataset loader and processor."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm

from config import DATASET_SCHEMA, PROCESSED_DATASETS_DIR, DatasetType


def load_arc_dataset(
    difficulty: str = "challenge",
    split: str = "test",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load ARC dataset with proper column parsing.
    
    Args:
        difficulty: Must be "challenge" (ARC-Easy is not active)
        split: Dataset split to load
        limit: Optional limit on number of entries
        
    Returns:
        List of entries in unified format
    """
    if difficulty != "challenge":
        raise ValueError("Only difficulty='challenge' is supported in active Final5 pipeline")

    dataset_type = DatasetType.ARC_CHALLENGE
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
    skipped_count = 0
    for entry in tqdm(ds, desc="Loading ARC-Challenge"):
        # Extract options from nested dict
        options = entry["choices"]["text"]
        labels = entry["choices"]["label"]
        
        # Filter: minimum 4 options required
        if len(options) < 4:
            skipped_count += 1
            continue
            
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
            "answer": options[answer_index] if answer_index < len(options) else options[0],
            "choices_answer": [options[answer_index] if answer_index < len(options) else options[0]],
            "answer_index": answer_index,
            "answer_letter": answer_letter,
            "dataset_type": dataset_type.value,
            "category": "", # ARC doesn't have categories
            # Human distractors are stored in choices_human
            "choices_human": [
                opt for i, opt in enumerate(options) if i != answer_index
            ],
        }
        entries.append(unified_entry)
        
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} entries with fewer than 4 options")
    
    return entries


def process_arc_for_experiments(
    difficulty: str = "challenge",
    split: str = "test",
    limit: Optional[int] = None,
    output_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Any:
    """
    Process ARC dataset and save as HF Dataset for experiments.
    
    Args:
        difficulty: Must be "challenge"
        split: Dataset split
        limit: Optional limit
        output_dir: Output base directory
        output_path: Exact output directory path (overrides output_dir)
        
    Returns:
        Processed Dataset
    """
    from datasets import Dataset
    entries = load_arc_dataset(difficulty, split, limit)
    
    # Convert to HF Dataset for standardization
    dataset = Dataset.from_list(entries)
    
    if output_path is None:
        if output_dir is None:
            output_dir = PROCESSED_DATASETS_DIR
        
        # Default path structure: output_dir/arc_processed/arc_challenge
        output_path = output_dir / "arc_processed" / "arc_challenge"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as HF Dataset
    dataset.save_to_disk(str(output_path))
    print(f"Saved {len(entries)} entries to {output_path}")
    
    return dataset
