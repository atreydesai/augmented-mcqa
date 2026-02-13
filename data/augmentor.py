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





def parse_generated_distractors(response: str, expected_count: int = 6) -> List[str]:
    """
    Parse generated distractors from model response.
    Expected format: E: <text>, F: <text>, etc.
    """
    distractors = []
    
    # Try to match pattern like "E: <text>" or "E. <text>" or just "E <text>"
    pattern = r'^([E-J])[:\.\s]+(.+)$'
    
    for line in response.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = re.match(pattern, line, re.IGNORECASE)
        if match:
            distractor = match.group(2).strip()
            distractor = distractor.rstrip('.')
            if distractor:
                distractors.append(distractor)
    
    # Fallback: if we didn't get enough, just take any lines that look like options
    if len(distractors) < expected_count:
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        for line in lines:
            if ":" in line and line[0] in "ABCDEFGHIJ":
                cleaned = line.split(":", 1)[1].strip()
                if cleaned not in distractors:
                    distractors.append(cleaned)
    
    return distractors[:expected_count]


def build_prompt(
    entry: Dict,
    mode: AugmentorMode,
    num_distractors: int = 6,
) -> str:
    """Build the generation prompt based on mode."""
    question = entry["question"]
    gold_answer = entry.get("answer", "")
    if not gold_answer and entry.get("choices_answer"):
        gold_answer = entry["choices_answer"][0]

    if mode == AugmentorMode.FROM_SCRATCH:
        return DISTRACTOR_GENERATION_PROMPT.format(
            question=question,
            gold_answer=gold_answer,
        )
    
    elif mode == AugmentorMode.CONDITIONED_HUMAN:
        distractors = entry.get("choices_human", [])
        if len(distractors) < 3:
            # Fallback to 1 + enough human
            return DISTRACTOR_GENERATION_PROMPT.format(question=question, gold_answer=gold_answer)
        
        return DISTRACTOR_GENERATION_PROMPT_CONDITIONED.format(
            question=question,
            gold_answer=gold_answer,
            distractor_1=distractors[0],
            distractor_2=distractors[1],
            distractor_3=distractors[2],
        )
    
    elif mode == AugmentorMode.CONDITIONED_SYNTHETIC:
        # User instructions: pick 3 random from cond_model_q_a_scratch
        scratch_distractors = entry.get(DistractorType.COND_MODEL_Q_A_SCRATCH.value, [])
        if len(scratch_distractors) < 3:
            return DISTRACTOR_GENERATION_PROMPT.format(question=question, gold_answer=gold_answer)
        
        selected = random.sample(scratch_distractors, 3)
        return DISTRACTOR_GENERATION_PROMPT_CONDITIONED.format(
            question=question,
            gold_answer=gold_answer,
            distractor_1=selected[0],
            distractor_2=selected[1],
            distractor_3=selected[2],
        )
    
    raise ValueError(f"Unknown mode: {mode}")


def assemble_mcqa_options(
    answer: str,
    human_distractors: List[str],
    model_distractors: List[str],
    num_total: int = 10
) -> (List[str], str):
    """
    Assemble and randomize options.
    Returns: (list_of_options, correct_letter)
    """
    # 1. Answer
    # 2. Pick 3 human
    human_subset = human_distractors[:3]
    # 3. Model distractors (should be 6)
    
    all_options = [answer] + human_subset + model_distractors
    # Ensure exactly num_total by padding if needed (shouldn't happen with correct flow)
    all_options = all_options[:num_total]
    
    # Randomize
    indices = list(range(len(all_options)))
    random.shuffle(indices)
    
    shuffled_options = [all_options[i] for i in indices]
    answer_idx = indices.index(0) # 0 was the original answer index
    correct_letter = chr(ord('A') + answer_idx)
    
    return shuffled_options, correct_letter


def format_options_for_prompt(options: List[str]) -> str:
    """Format options as A: ..., B: ..."""
    formatted = []
    for i, opt in enumerate(options):
        letter = chr(ord('A') + i)
        formatted.append(f"{letter}: {opt}")
    return "\n".join(formatted)


def augment_dataset(
    dataset_path: Path,
    config: GenerationConfig,
    output_path: Optional[Path] = None,
    limit: Optional[int] = None,
    resume_from: Optional[Path] = None,
) -> Dataset:
    """
    Augment a dataset with synthetic distractors and full MCQA columns.
    """
    from config import MCQA_PROMPT_FULL
    
    # Load dataset
    dataset = load_from_disk(str(dataset_path))
    if isinstance(dataset, DatasetDict):
        # We process all splits in the dict if it's a Dict
        final_dataset_dict = {}
        for split_name, ds in dataset.items():
            print(f"\nProcessing split: {split_name}")
            final_dataset_dict[split_name] = augment_single_dataset(ds, config, limit)
        result = DatasetDict(final_dataset_dict)
    else:
        result = augment_single_dataset(dataset, config, limit)

    # Save
    if output_path:
        output_path = Path(output_path)
        result.save_to_disk(str(output_path))
        print(f"Saved to {output_path}")
        
    return result


def augment_single_dataset(dataset: Dataset, config: GenerationConfig, limit: Optional[int] = None) -> Dataset:
    """Process a single Dataset."""
    from config import MCQA_PROMPT_FULL
    client = get_client(config.model_name)
    
    # Prefixes for columns based on mode
    mode_suffixes = {
        AugmentorMode.FROM_SCRATCH: ("scratch", "qa"),
        AugmentorMode.CONDITIONED_HUMAN: ("dhuman", "qadh"),
        AugmentorMode.CONDITIONED_SYNTHETIC: ("dmodel", "qadm"),
    }
    suffix, prefix = mode_suffixes[config.mode]
    
    model_col = f"cond_model_q_a_{suffix}"
    options_col = f"{prefix}_options_randomized"
    letter_col = f"{prefix}_correct_answer_letter"
    question_col = f"{prefix}_full_question"
    input_col = f"{prefix}_model_input"
    output_col = f"{prefix}_model_output"
    
    entries = list(dataset)
    if limit:
        entries = entries[:limit]
        
    augmented_entries = []
    for i, entry in enumerate(tqdm(entries, desc=f"Augmenting ({config.mode.value})")):
        try:
            prompt = build_prompt(entry, config.mode)
            response = client.generate(prompt=prompt, temperature=config.temperature)
            raw_text = response.text
            distractors = parse_generated_distractors(raw_text, 6)
            
            # Assembly
            answer = entry.get("answer", "")
            if not answer and entry.get("choices_answer"):
                answer = entry["choices_answer"][0]
            
            human_distractors = entry.get("choices_human", [])
            
            # Prepare options
            options, correct_letter = assemble_mcqa_options(answer, human_distractors, distractors)
            
            # Formatting
            options_text = format_options_for_prompt(options)
            full_question = f"{entry['question']}\n{options_text}"
            model_input = MCQA_PROMPT_FULL.format(question=entry['question'], options=options_text)
            
            # Fill entry
            new_entry = dict(entry)
            new_entry[model_col] = distractors
            new_entry[output_col] = raw_text
            new_entry[options_col] = options
            new_entry[letter_col] = correct_letter
            new_entry[question_col] = full_question
            new_entry[input_col] = model_input
            
            augmented_entries.append(new_entry)
            
        except Exception as e:
            print(f"Error at {i}: {e}")
            augmented_entries.append(dict(entry))
            
    return Dataset.from_list(augmented_entries)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--mode", type=str, default="from_scratch")
    parser.add_argument("--model", type=str, default="gpt-4.1")
    parser.add_argument("--output", type=str)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    
    config = GenerationConfig(
        mode=AugmentorMode(args.mode),
        model_provider="openai", # Dummy, inferred from name
        model_name=args.model,
    )
    
    augment_dataset(
        dataset_path=Path(args.dataset),
        config=config,
        output_path=args.output,
        limit=args.limit
    )

