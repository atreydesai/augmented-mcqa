"""
Dataset filtering and subset creation.

Creates simplified subsets of augmented datasets:
- 1 Gold + 3 Human distractors
- 1 Gold + 3 Synthetic distractors
- Configurable mix of human and synthetic
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm

from config import (
    DATASETS_DIR,
    RANDOM_SEED,
    DistractorType,
    get_distractor_column,
)


@dataclass
class FilterConfig:
    """Configuration for filtering a dataset."""
    num_human: int = 3
    num_model: int = 0
    model_distractor_type: DistractorType = DistractorType.COND_MODEL_Q_A
    shuffle_seed: int = RANDOM_SEED
    require_minimum: bool = True  # Skip entries that don't have enough distractors


def shuffle_options_deterministic(
    gold_answer: str,
    distractors: List[str],
    seed: int,
    question_idx: int,
) -> Tuple[List[str], int]:
    """
    Shuffle options deterministically and return new answer index.
    
    Uses a combination of seed and question index for reproducibility
    while ensuring different questions get different shuffles.
    
    Args:
        gold_answer: The correct answer text
        distractors: List of distractor texts
        seed: Base random seed
        question_idx: Index of the question (for deterministic variation)
        
    Returns:
        Tuple of (shuffled_options, new_answer_index)
    """
    # Create deterministic RNG for this question
    rng = random.Random(seed + question_idx)
    
    # Combine all options
    all_options = [gold_answer] + list(distractors)
    
    # Create shuffle order
    indices = list(range(len(all_options)))
    rng.shuffle(indices)
    
    # Apply shuffle
    shuffled = [all_options[i] for i in indices]
    new_answer_idx = indices.index(0)  # Original gold was at index 0
    
    return shuffled, new_answer_idx


CHOICE_LABELS = "ABCDEFGHIJ"


def get_answer_letter(index: int) -> str:
    """Convert answer index to letter (A, B, C, ...)."""
    if 0 <= index < len(CHOICE_LABELS):
        return CHOICE_LABELS[index]
    return "?"


def filter_dataset(
    dataset_path: Path,
    config: FilterConfig,
    output_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Dataset:
    """
    Filter a dataset to create a subset with specified number of distractors.
    
    Creates a new dataset with:
    - assembled_options: The filtered and shuffled options
    - answer: New answer letter (A, B, C, ...)
    - answer_index: New answer index
    
    Args:
        dataset_path: Path to input dataset
        config: Filter configuration
        output_path: Where to save filtered dataset
        limit: Limit number of entries (for testing)
        
    Returns:
        Filtered Dataset
    """
    # Load dataset
    dataset = load_from_disk(str(dataset_path))
    
    # Handle DatasetDict vs Dataset
    if isinstance(dataset, DatasetDict):
        if "test" in dataset:
            dataset = dataset["test"]
        else:
            dataset = list(dataset.values())[0]
    
    # Set up output path
    if output_path is None:
        name = f"filtered_{config.num_human}H_{config.num_model}M"
        output_path = DATASETS_DIR / name
    output_path = Path(output_path)
    
    # Process entries
    entries = list(dataset)
    if limit:
        entries = entries[:limit]
    
    filtered_entries = []
    skipped_count = 0
    
    for idx, entry in enumerate(tqdm(entries, desc="Filtering")):
        # Get gold answer
        gold_answers = entry.get("choices_answer", [])
        if not gold_answers:
            # Fallback
            options = entry.get("options", [])
            answer_idx = entry.get("answer_index", 0)
            gold_answer = options[answer_idx] if answer_idx < len(options) else ""
        else:
            gold_answer = gold_answers[0]
        
        if not gold_answer:
            skipped_count += 1
            continue
        
        # Get human distractors
        human_distractors = get_distractor_column(entry, DistractorType.COND_HUMAN_Q_A)
        
        # Get model distractors
        model_distractors = get_distractor_column(entry, config.model_distractor_type)
        
        # Check if we have enough
        if config.require_minimum:
            if len(human_distractors) < config.num_human:
                skipped_count += 1
                continue
            if len(model_distractors) < config.num_model:
                skipped_count += 1
                continue
        
        # Sample distractors deterministically
        rng = random.Random(config.shuffle_seed + idx)
        
        selected_human = rng.sample(
            human_distractors, 
            min(config.num_human, len(human_distractors))
        )
        selected_model = rng.sample(
            model_distractors,
            min(config.num_model, len(model_distractors))
        )
        
        # Combine distractors
        all_distractors = selected_human + selected_model
        
        # Shuffle options
        shuffled_options, new_answer_idx = shuffle_options_deterministic(
            gold_answer,
            all_distractors,
            config.shuffle_seed,
            idx,
        )
        
        # Create filtered entry
        filtered = {
            **entry,
            "assembled_options": shuffled_options,
            "answer": get_answer_letter(new_answer_idx),
            "answer_index": new_answer_idx,
            "num_human_distractors": len(selected_human),
            "num_model_distractors": len(selected_model),
            "filter_config": f"{config.num_human}H_{config.num_model}M",
        }
        filtered_entries.append(filtered)
    
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} entries (insufficient distractors)")
    
    # Create and save
    result = Dataset.from_list(filtered_entries)
    output_path.mkdir(parents=True, exist_ok=True)
    result.save_to_disk(str(output_path))
    
    # Push to Hugging Face
    from data.hub_utils import push_dataset_to_hub
    repo_id = f"atreydesai/qgqa-{Path(dataset_path).stem}-{config.num_human}H_{config.num_model}M-filtered"
    push_dataset_to_hub(result, repo_id=repo_id)
    
    print(f"\n✓ Saved filtered dataset to {output_path}")
    print(f"  Config: {config.num_human}H + {config.num_model}M")
    print(f"  Entries: {len(result)}")
    
    return result


def create_standard_subsets(
    dataset_path: Path,
    output_base: Optional[Path] = None,
    limit: Optional[int] = None,
) -> dict:
    """
    Create standard experiment subsets from a processed dataset.
    
    Creates:
    - 3H_0M: 3 human distractors only (baseline)
    - 0H_3M: 3 model distractors only
    - 1H_2M, 2H_1M: Mixed configurations
    - Variants for each model distractor type
    
    Args:
        dataset_path: Path to processed dataset with sorted distractors
        output_base: Base directory for outputs
        limit: Limit entries (for testing)
        
    Returns:
        Dict mapping subset name to Dataset
    """
    if output_base is None:
        output_base = DATASETS_DIR / "subsets"
    output_base = Path(output_base)
    
    results = {}
    
    # Standard configurations: (num_human, num_model)
    configs = [
        (3, 0),  # Human only
        (0, 3),  # Model only
        (1, 2),  # Mixed
        (2, 1),  # Mixed
        (3, 3),  # Full set
    ]
    
    for num_human, num_model in configs:
        name = f"{num_human}H_{num_model}M"
        
        config = FilterConfig(
            num_human=num_human,
            num_model=num_model,
            model_distractor_type=DistractorType.COND_MODEL_Q_A,
        )
        
        output_path = output_base / name
        
        try:
            result = filter_dataset(
                dataset_path=dataset_path,
                config=config,
                output_path=output_path,
                limit=limit,
            )
            results[name] = result
        except Exception as e:
            print(f"  Error creating {name}: {e}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter dataset to create subsets")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to input dataset",
    )
    parser.add_argument(
        "--num-human",
        type=int,
        default=3,
        help="Number of human distractors",
    )
    parser.add_argument(
        "--num-model",
        type=int,
        default=0,
        help="Number of model distractors",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["cond_model_q_a", "cond_model_q_a_dhuman", "cond_model_q_a_dmodel"],
        default="cond_model_q_a",
        help="Type of model distractors to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit entries (for testing)",
    )
    parser.add_argument(
        "--create-standard",
        action="store_true",
        help="Create all standard subsets instead of a single configuration",
    )
    args = parser.parse_args()
    
    if args.create_standard:
        create_standard_subsets(
            dataset_path=Path(args.dataset),
            output_base=Path(args.output) if args.output else None,
            limit=args.limit,
        )
    else:
        model_type = {
            "cond_model_q_a": DistractorType.COND_MODEL_Q_A,
            "cond_model_q_a_dhuman": DistractorType.COND_MODEL_Q_A_DHUMAN,
            "cond_model_q_a_dmodel": DistractorType.COND_MODEL_Q_A_DMODEL,
        }[args.model_type]
        
        config = FilterConfig(
            num_human=args.num_human,
            num_model=args.num_model,
            model_distractor_type=model_type,
        )
        
        filter_dataset(
            dataset_path=Path(args.dataset),
            config=config,
            output_path=Path(args.output) if args.output else None,
            limit=args.limit,
        )
