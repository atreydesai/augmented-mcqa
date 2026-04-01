"""ARC dataset loader and processor."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm

from config import DATASET_SCHEMA, PROCESSED_DATASETS_DIR, DatasetType


ARC_OUTPUT_DIRNAME = "arc_processed"


def _answer_index(labels: list[str], answer_key: str) -> int | None:
    try:
        return labels.index(answer_key)
    except ValueError:
        if not answer_key.isdigit():
            return None
        index = int(answer_key) - 1
        return index if index >= 0 else None


def load_arc_dataset(
    split: str = "test",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load ARC dataset with proper column parsing.
    
    Args:
        split: Dataset split to load
        limit: Optional limit on number of entries
        
    Returns:
        List of entries in unified format
    """
    dataset_type = DatasetType.ARC_CHALLENGE
    schema = DATASET_SCHEMA[dataset_type]
    
    # Load from HuggingFace
    ds = load_dataset(
        schema["hf_path"],
        schema["hf_config"],
        split=split,
        trust_remote_code=True,
    )
    
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    
    # Convert to unified format
    entries = []
    skipped_option_count = 0
    skipped_answer_count = 0
    for entry in tqdm(ds, desc="Loading ARC-Challenge"):
        # Extract options from nested dict
        options = entry["choices"]["text"]
        labels = entry["choices"]["label"]
        
        # Filter: minimum 4 options required
        if len(options) < 4:
            skipped_option_count += 1
            continue
            
        # Get answer index from letter
        answer_letter = entry["answerKey"]
        answer_index = _answer_index(labels, answer_letter)
        if answer_index is None or answer_index >= len(options):
            skipped_answer_count += 1
            continue
        answer = options[answer_index]
        
        unified_entry = {
            "id": entry["id"],
            "question": entry["question"],
            "options": options,
            "labels": labels,
            "answer": answer,
            "choices_answer": [answer],
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
        
    if skipped_option_count > 0:
        print(f"  Skipped {skipped_option_count} entries with fewer than 4 options")
    if skipped_answer_count > 0:
        print(f"  Skipped {skipped_answer_count} entries with invalid answer keys")
    
    return entries


def process_arc_for_experiments(
    split: str = "test",
    limit: Optional[int] = None,
    output_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Any:
    """
    Process ARC dataset and save as HF Dataset for experiments.
    
    Args:
        split: Dataset split
        limit: Optional limit
        output_dir: Output base directory
        output_path: Exact output directory path (overrides output_dir)
        
    Returns:
        Processed Dataset
    """
    from datasets import Dataset
    entries = load_arc_dataset(split=split, limit=limit)
    
    # Convert to HF Dataset for standardization
    dataset = Dataset.from_list(entries)
    
    if output_path is None:
        if output_dir is None:
            output_dir = PROCESSED_DATASETS_DIR
        
        output_path = output_dir / ARC_OUTPUT_DIRNAME / DatasetType.ARC_CHALLENGE.value
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as HF Dataset
    dataset.save_to_disk(str(output_path))
    print(f"Saved {len(entries)} entries to {output_path}")
    
    return dataset
