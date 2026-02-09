"""
Synthetic distractor generator for MCQA datasets.

Supports multiple generation modes:
- FROM_SCRATCH: Generate distractors from question + answer only
- CONDITIONED_HUMAN: Generate conditioned on human distractors
- CONDITIONED_SYNTHETIC: Generate conditioned on existing synthetic distractors

Supports multiple model providers:
- OpenAI (GPT-4, etc.)
- Anthropic (Claude)
- Google (Gemini)
"""

import time
import random
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass

from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm

from config import (
    DATASETS_DIR,
    AUGMENTED_DATASETS_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
    DistractorType,
    get_distractor_column,
    DISTRACTOR_GENERATION_PROMPT,
    DISTRACTOR_GENERATION_PROMPT_CONDITIONED,
    get_api_key,
)
from models import get_client


class AugmentorMode(Enum):
    """
    Generation mode for synthetic distractors.
    
    IMPORTANT: cond_model_q_a contains EXISTING synthetic distractors from MMLU-Pro.
    These modes generate NEW distractors:
    - FROM_SCRATCH: New distractors from Q+A only (no conditioning)
    - CONDITIONED_HUMAN: New distractors conditioned on 3 human distractors
    - CONDITIONED_SYNTHETIC: New distractors conditioned on 3 random existing synthetic
    """
    # Generate from question + answer only
    # Output: cond_model_q_a_scratch (NEW generated, not the existing MMLU-Pro ones)
    FROM_SCRATCH = "from_scratch"
    
    # Generate conditioned on 3 human distractors
    # Output: cond_model_q_a_dhuman
    CONDITIONED_HUMAN = "conditioned_human"
    
    # Generate conditioned on 3 RANDOMLY SELECTED existing synthetic distractors
    # Output: cond_model_q_a_dmodel
    CONDITIONED_SYNTHETIC = "conditioned_synthetic"


@dataclass
class GenerationConfig:
    """Configuration for distractor generation."""
    mode: AugmentorMode
    model_provider: str  # "openai", "anthropic", "google"
    model_name: str
    num_distractors: int = 9  # Default: generate 9 to make 10 total options
    temperature: float = 0.7
    max_retries: int = 3
    retry_delay: float = 1.0
    save_interval: int = 50  # Save intermediate results every N entries





def parse_generated_distractors(response: str, expected_count: int = 9) -> List[str]:
    """
    Parse generated distractors from model response.
    
    Expected format:
    B: <distractor>
    C: <distractor>
    ...
    
    Args:
        response: Raw model response
        expected_count: Expected number of distractors
        
    Returns:
        List of parsed distractor strings
    """
    distractors = []
    
    # Try to match pattern like "B: <text>" or "B. <text>" or just "B <text>"
    pattern = r'^([B-J])[:\.\s]+(.+)$'
    
    for line in response.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = re.match(pattern, line, re.IGNORECASE)
        if match:
            distractor = match.group(2).strip()
            # Clean up common formatting issues
            distractor = distractor.rstrip('.')
            if distractor:
                distractors.append(distractor)
    
    # If parsing failed, try alternative approaches
    if len(distractors) < expected_count // 2:
        # Try splitting by common separators
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        for line in lines:
            # Skip lines that are just letters
            if re.match(r'^[A-J]\.?$', line):
                continue
            # Remove leading letter markers
            cleaned = re.sub(r'^[B-J][:\.\s]+', '', line).strip()
            if cleaned and cleaned not in distractors:
                distractors.append(cleaned)
    
    return distractors[:expected_count]


def build_prompt(
    entry: Dict,
    mode: AugmentorMode,
    num_distractors: int = 6,
) -> str:
    """
    Build the generation prompt based on mode.
    
    Args:
        entry: Dataset entry with question, answer, and possibly existing distractors
        mode: Generation mode
        num_distractors: Number of distractors to generate (default 6)
        
    Returns:
        Formatted prompt string
    """
    question = entry["question"]
    
    # Get gold answer
    gold_answers = entry.get("choices_answer", [])
    if gold_answers:
        gold_answer = gold_answers[0]
    else:
        # Fallback: try to get from options using answer_index
        options = entry.get("options", [])
        answer_idx = entry.get("answer_index", 0)
        gold_answer = options[answer_idx] if answer_idx < len(options) else ""
    
    if mode == AugmentorMode.FROM_SCRATCH:
        return DISTRACTOR_GENERATION_PROMPT.format(
            question=question,
            gold_answer=gold_answer,
        )
    
    elif mode == AugmentorMode.CONDITIONED_HUMAN:
        # Get human distractors (up to 3 from original MMLU/ARC/SuperGPQA)
        distractors = get_distractor_column(entry, DistractorType.COND_HUMAN_Q_A)
        
        # Need exactly 3 distractors for conditioning
        if len(distractors) < 3:
            # Fall back to from_scratch if not enough human distractors
            return DISTRACTOR_GENERATION_PROMPT.format(
                question=question,
                gold_answer=gold_answer,
            )
        
        # Use first 3 human distractors
        return DISTRACTOR_GENERATION_PROMPT_CONDITIONED.format(
            question=question,
            gold_answer=gold_answer,
            distractor_1=distractors[0],
            distractor_2=distractors[1],
            distractor_3=distractors[2],
        )
    
    elif mode == AugmentorMode.CONDITIONED_SYNTHETIC:
        # Get existing synthetic distractors (from MMLU-Pro, up to 6)
        distractors = get_distractor_column(entry, DistractorType.COND_MODEL_Q_A)
        
        # Need at least 3 synthetic distractors to select from
        if len(distractors) < 3:
            # Fall back to from_scratch if not enough synthetic distractors
            return DISTRACTOR_GENERATION_PROMPT.format(
                question=question,
                gold_answer=gold_answer,
            )
        
        # RANDOMLY SELECT 3 from the available synthetic distractors
        selected_distractors = random.sample(distractors, 3)
        
        return DISTRACTOR_GENERATION_PROMPT_CONDITIONED.format(
            question=question,
            gold_answer=gold_answer,
            distractor_1=selected_distractors[0],
            distractor_2=selected_distractors[1],
            distractor_3=selected_distractors[2],
        )
    
    raise ValueError(f"Unknown mode: {mode}")


def generate_distractors(
    entry: Dict,
    config: GenerationConfig,
    client: Any,
) -> List[str]:
    """
    Generate synthetic distractors for a single entry.
    
    Args:
        entry: Dataset entry
        config: Generation configuration
        client: API client for the provider
        
    Returns:
        List of generated distractor strings
    """
    prompt = build_prompt(entry, config.mode, config.num_distractors)
    
    # Select generation function based on provider
    # NOTE: Functionality replaced by unified models.get_client()
    client = get_client(config.model_name)
    
    # Generate with retries
    for attempt in range(config.max_retries):
        try:

            response = client.generate(
                prompt=prompt,
                temperature=config.temperature,
                max_tokens=1000,
            )
            distractors = parse_generated_distractors(response.text, config.num_distractors)
            
            if len(distractors) >= config.num_distractors // 2:
                return distractors
            
            # Not enough distractors parsed, retry
            if attempt < config.max_retries - 1:
                time.sleep(config.retry_delay)
                
        except Exception as e:
            if attempt < config.max_retries - 1:
                time.sleep(config.retry_delay * (attempt + 1))
            else:
                raise
    
    return distractors  # Return whatever we got


def get_output_column(mode: AugmentorMode) -> str:
    """
    Get the output column name for a generation mode.
    
    Note: FROM_SCRATCH outputs to cond_model_q_a_scratch (NEW generated),
    NOT to cond_model_q_a (which holds EXISTING MMLU-Pro synthetic distractors).
    """
    mapping = {
        AugmentorMode.FROM_SCRATCH: DistractorType.COND_MODEL_Q_A_SCRATCH.value,
        AugmentorMode.CONDITIONED_HUMAN: DistractorType.COND_MODEL_Q_A_DHUMAN.value,
        AugmentorMode.CONDITIONED_SYNTHETIC: DistractorType.COND_MODEL_Q_A_DMODEL.value,
    }
    return mapping[mode]


def augment_dataset(
    dataset_path: Path,
    config: GenerationConfig,
    output_path: Optional[Path] = None,
    limit: Optional[int] = None,
    resume_from: Optional[Path] = None,
) -> Dataset:
    """
    Augment a dataset with synthetic distractors.
    
    Args:
        dataset_path: Path to input dataset
        config: Generation configuration
        output_path: Where to save augmented dataset
        limit: Limit number of entries to process (for testing)
        resume_from: Path to intermediate results to resume from
        
    Returns:
        Augmented Dataset
    """
    # Load dataset
    dataset = load_from_disk(str(dataset_path))
    
    # Handle DatasetDict vs Dataset
    if isinstance(dataset, DatasetDict):
        # Use test split by default
        if "test" in dataset:
            dataset = dataset["test"]
        else:
            dataset = list(dataset.values())[0]
    
    # Initialize client
    # The client type is inferred from the model name in the config
    client = get_client(config.model_name)
    
    # Load existing results if resuming
    processed = {}
    if resume_from and resume_from.exists():
        with open(resume_from, 'r') as f:
            processed = json.load(f)
        print(f"Resuming from {len(processed)} processed entries")
    
    # Set up output paths
    output_column = get_output_column(config.mode)
    
    if output_path is None:
        # Default path: augmented/{mode}/{dataset_name}_{model_name}.json
        dataset_name = Path(dataset_path).stem
        model_safe = config.model_name.replace("/", "_")
        
        output_dir = AUGMENTED_DATASETS_DIR / config.mode.value
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{dataset_name}_{model_safe}.json"
    output_path = Path(output_path)
    
    intermediate_path = output_path / "intermediate.json"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process entries
    entries = list(dataset)
    if limit:
        entries = entries[:limit]
    
    augmented_entries = []
    
    for i, entry in enumerate(tqdm(entries, desc=f"Generating ({config.mode.value})")):
        # Check if already processed
        entry_key = str(i)
        if entry_key in processed:
            augmented = dict(entry)
            augmented[output_column] = processed[entry_key]
            augmented_entries.append(augmented)
            continue
        
        try:
            distractors = generate_distractors(entry, config, client)
            processed[entry_key] = distractors
            
            augmented = dict(entry)
            augmented[output_column] = distractors
            augmented_entries.append(augmented)
            
            # Save intermediate results
            if (i + 1) % config.save_interval == 0:
                with open(intermediate_path, 'w') as f:
                    json.dump(processed, f)
                print(f"  Saved intermediate results ({i + 1} entries)")
                
        except Exception as e:
            print(f"  Error at entry {i}: {e}")
            augmented = dict(entry)
            augmented[output_column] = []
            augmented_entries.append(augmented)
    
    # Create and save final dataset
    result = Dataset.from_list(augmented_entries)
    result.save_to_disk(str(output_path))
    
    # Push to Hugging Face
    from data.hub_utils import push_dataset_to_hub
    repo_id = f"atreydesai/qgqa-{Path(dataset_path).stem}-{config.mode.value}-augmented"
    push_dataset_to_hub(result, repo_id=repo_id)
    
    # Clean up intermediate file
    if intermediate_path.exists():
        intermediate_path.unlink()
    
    print(f"\n✓ Saved augmented dataset to {output_path}")
    print(f"  Mode: {config.mode.value}")
    print(f"  Output column: {output_column}")
    print(f"  Entries: {len(result)}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic distractors")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to input dataset",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["from_scratch", "conditioned_human", "conditioned_synthetic"],
        default="from_scratch",
        help="Generation mode",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic", "google"],
        default="openai",
        help="Model provider",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="Model name",
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
        help="Limit number of entries (for testing)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to intermediate results to resume from",
    )
    args = parser.parse_args()
    
    config = GenerationConfig(
        mode=AugmentorMode(args.mode),
        model_provider=args.provider,
        model_name=args.model,
    )
    
    augment_dataset(
        dataset_path=Path(args.dataset),
        config=config,
        output_path=Path(args.output) if args.output else None,
        limit=args.limit,
        resume_from=Path(args.resume) if args.resume else None,
    )
