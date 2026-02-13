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
from data.hub_utils import push_dataset_to_hub


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
    Supports formats like B: <text> through J: <text>.
    """
    distractors = []
    
    # Match pattern like "B: <text>" through "J: <text>"
    pattern = r'^([B-J])[:\.\s]+(.+)$'
    
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
                parts = line.split(":", 1)
                if len(parts) > 1:
                    cleaned = parts[1].strip()
                    if cleaned and cleaned not in distractors:
                        distractors.append(cleaned)
    
    return distractors[:expected_count]


def build_prompt(
    entry: Dict,
    mode: AugmentorMode,
) -> (str, List[str]):
    """Build the generation prompt based on mode and return it along with selected distractors."""
    question = entry["question"]
    gold_answer = entry.get("answer", "")
    if not gold_answer and entry.get("choices_answer"):
        gold_answer = entry["choices_answer"][0]

    if mode == AugmentorMode.FROM_SCRATCH:
        return DISTRACTOR_GENERATION_PROMPT.format(
            question=question,
            gold_answer=gold_answer,
        ), []
    
    elif mode == AugmentorMode.CONDITIONED_HUMAN:
        distractors = entry.get("choices_human", [])
        if not distractors or len(distractors) < 3:
            # Fallback to scratch if no human distractors
            return DISTRACTOR_GENERATION_PROMPT.format(question=question, gold_answer=gold_answer), []
        
        selected = distractors[:3]
        return DISTRACTOR_GENERATION_PROMPT_CONDITIONED.format(
            question=question,
            gold_answer=gold_answer,
            distractor_1=selected[0],
            distractor_2=selected[1],
            distractor_3=selected[2],
        ), selected
    
    elif mode == AugmentorMode.CONDITIONED_SYNTHETIC:
        # User instructions: pick 3 from cond_model_q_a_scratch
        scratch_distractors = entry.get(DistractorType.COND_MODEL_Q_A_SCRATCH.value, [])
        if not scratch_distractors or len(scratch_distractors) < 3:
            return DISTRACTOR_GENERATION_PROMPT.format(question=question, gold_answer=gold_answer), []
        
        # We don't random sample here so it's deterministic and traceable if we just take first 3, 
        # or we random sample but return which ones we picked. User asked for 3 of the 6 previously generated.
        selected = random.sample(scratch_distractors, 3)
        return DISTRACTOR_GENERATION_PROMPT_CONDITIONED.format(
            question=question,
            gold_answer=gold_answer,
            distractor_1=selected[0],
            distractor_2=selected[1],
            distractor_3=selected[2],
        ), selected
    
    raise ValueError(f"Unknown mode: {mode}")


def assemble_mcqa_options(
    answer: str,
    human_distractors: List[str],
    model_distractors: List[str],
    mode: AugmentorMode,
    num_total: int = 10
) -> (List[str], str):
    """
    Assemble and randomize options.
    Returns: (list_of_options, correct_letter)
    """
    # 1. Answer
    # 2. Pick human distractors (always include up to 3 to reach 10 total)
    human_subset = human_distractors[:3]
    
    # 3. Model distractors (should be 6 per current prompts)
    all_options = [answer] + human_subset + model_distractors
    
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
    push_to_hub: bool = True,
) -> Dataset:
    """
    Augment a dataset with synthetic distractors and full MCQA columns.
    """
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
        
        if push_to_hub:
            dataset_name = output_path.name
            print(f"Pushing to Hub as {dataset_name}...")
            push_dataset_to_hub(result, dataset_name=dataset_name)
        
    return result


def augment_single_dataset(dataset: Dataset, config: GenerationConfig, limit: Optional[int] = None) -> Dataset:
    """Process a single Dataset through all augmentation modes (scratch, human-conditioned, model-conditioned)."""
    client = get_client(config.model_name)
    
    entries = list(dataset)
    if limit:
        entries = entries[:limit]
        
    augmented_entries = []
    
    # We run modes in sequence because qadm depends on qa (scratch)
    modes_to_run = [
        AugmentorMode.FROM_SCRATCH, 
        AugmentorMode.CONDITIONED_HUMAN, 
        AugmentorMode.CONDITIONED_SYNTHETIC
    ]
    
    mode_prefixes = {
        AugmentorMode.FROM_SCRATCH: "qa",
        AugmentorMode.CONDITIONED_HUMAN: "qadh",
        AugmentorMode.CONDITIONED_SYNTHETIC: "qadm",
    }
    
    mode_model_cols = {
        AugmentorMode.FROM_SCRATCH: "cond_model_q_a_scratch",
        AugmentorMode.CONDITIONED_HUMAN: "cond_model_q_a_dhuman",
        AugmentorMode.CONDITIONED_SYNTHETIC: "cond_model_q_a_dmodel",
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    temp_save_path = RESULTS_DIR / f"temp_augmented_{config.model_name}_{timestamp}.json"
    
    for i, entry in enumerate(tqdm(entries, desc="Augmenting multi-mode")):
        new_entry = dict(entry)
        
        # Core data needed for all modes
        answer = new_entry.get("answer", "")
        if not answer and new_entry.get("choices_answer"):
            answer = new_entry["choices_answer"][0]
        
        human_distractors = new_entry.get("choices_human", [])
        
        for mode in modes_to_run:
            try:
                prefix = mode_prefixes[mode]
                model_col = mode_model_cols[mode]
                
                # 1. Build Prompt (now returns distractors used for context)
                prompt, context_distractors = build_prompt(new_entry, mode)
                
                # 2. Generate
                response = client.generate(prompt=prompt, temperature=config.temperature)
                raw_text = response.text
                
                # 3. Parse Distractors (Always 6 per user prompt)
                distractors = parse_generated_distractors(raw_text, 6)
                
                # 4. Assembly (1 answer + 6 synthetic + 3 human = 10 total)
                options, correct_letter = assemble_mcqa_options(answer, human_distractors, distractors, mode)
                
                # 5. Column Population (Ordered as requested)
                # qa_full_question (Refined for conditioned modes)
                if mode == AugmentorMode.FROM_SCRATCH:
                    new_entry[f"{prefix}_full_question"] = f"Question: {new_entry['question']}\nAnswer: A: {answer}"
                else:
                    # Conditioned format: Question + Existing 4 options (A + B,C,D) + Answer
                    context_options = [f"A: {answer}"]
                    for j, d in enumerate(context_distractors):
                        letter = chr(ord('B') + j)
                        context_options.append(f"{letter}: {d}")
                    options_str = "\n".join(context_options)
                    new_entry[f"{prefix}_full_question"] = f"Question: {new_entry['question']}\nExisting 4 Options: {options_str}\nAnswer: A: {answer}"
                
                # qa_model_input
                new_entry[f"{prefix}_model_input"] = prompt
                
                # qa_model_output
                new_entry[f"{prefix}_model_output"] = raw_text
                
                # cond_model_q_a_scratch (or dhuman/dmodel)
                new_entry[model_col] = distractors
                
                # qa_options_randomized
                new_entry[f"{prefix}_options_randomized"] = options
                
                # qa_correct_answer_letter
                new_entry[f"{prefix}_correct_answer_letter"] = correct_letter
                
            except Exception as e:
                print(f"Error at entry {i}, mode {mode.value}: {e}")
                
        augmented_entries.append(new_entry)
        
        # Intermediate saving
        if (i + 1) % config.save_interval == 0:
            with open(temp_save_path, 'w') as f:
                json.dump(augmented_entries, f, indent=2)
            print(f"\n💾 Saved intermediate results ({i+1} entries) to {temp_save_path}")
            
    # Remove temp file on completion if we reached the end
    if temp_save_path.exists():
        temp_save_path.unlink()
        
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

