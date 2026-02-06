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
    RESULTS_DIR,
    RANDOM_SEED,
    DistractorType,
    get_distractor_column,
    DISTRACTOR_GENERATION_PROMPT,
    DISTRACTOR_GENERATION_PROMPT_CONDITIONED,
    get_api_key,
)


class AugmentorMode(Enum):
    """Generation mode for synthetic distractors."""
    # Generate from question + answer only (produces cond_model_q_a)
    FROM_SCRATCH = "from_scratch"
    
    # Generate conditioned on human distractors (produces cond_model_q_a_dhuman)
    CONDITIONED_HUMAN = "conditioned_human"
    
    # Generate conditioned on existing synthetic distractors (produces cond_model_q_a_dmodel)
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


def _get_openai_client():
    """Initialize OpenAI client."""
    import openai
    return openai.OpenAI(api_key=get_api_key("openai"))


def _get_anthropic_client():
    """Initialize Anthropic client."""
    import anthropic
    return anthropic.Anthropic(api_key=get_api_key("anthropic"))


def _get_google_client():
    """Initialize Google Gemini client."""
    from google import genai
    return genai.Client(api_key=get_api_key("google"))


def _generate_openai(
    client,
    prompt: str,
    model_name: str,
    temperature: float,
) -> str:
    """Generate using OpenAI API."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1000,
    )
    return response.choices[0].message.content


def _generate_anthropic(
    client,
    prompt: str,
    model_name: str,
    temperature: float,
) -> str:
    """Generate using Anthropic API."""
    response = client.messages.create(
        model=model_name,
        max_tokens=1000,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _generate_google(
    client,
    prompt: str,
    model_name: str,
    temperature: float,
) -> str:
    """Generate using Google Gemini API."""
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "temperature": temperature,
            "max_output_tokens": 1000,
        },
    )
    return response.text


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
    num_distractors: int = 9,
) -> str:
    """
    Build the generation prompt based on mode.
    
    Args:
        entry: Dataset entry with question, answer, and possibly existing distractors
        mode: Generation mode
        num_distractors: Number of distractors to generate
        
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
    
    elif mode in (AugmentorMode.CONDITIONED_HUMAN, AugmentorMode.CONDITIONED_SYNTHETIC):
        # Get conditioning distractors
        if mode == AugmentorMode.CONDITIONED_HUMAN:
            distractors = get_distractor_column(entry, DistractorType.COND_HUMAN_Q_A)
        else:
            distractors = get_distractor_column(entry, DistractorType.COND_MODEL_Q_A)
        
        # Need at least 3 distractors for conditioning
        if len(distractors) < 3:
            # Fall back to from_scratch if not enough distractors
            return DISTRACTOR_GENERATION_PROMPT.format(
                question=question,
                gold_answer=gold_answer,
            )
        
        return DISTRACTOR_GENERATION_PROMPT_CONDITIONED.format(
            question=question,
            gold_answer=gold_answer,
            distractor_1=distractors[0],
            distractor_2=distractors[1],
            distractor_3=distractors[2],
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
    generate_fn = {
        "openai": _generate_openai,
        "anthropic": _generate_anthropic,
        "google": _generate_google,
    }.get(config.model_provider)
    
    if not generate_fn:
        raise ValueError(f"Unknown provider: {config.model_provider}")
    
    # Generate with retries
    for attempt in range(config.max_retries):
        try:
            response = generate_fn(
                client,
                prompt,
                config.model_name,
                config.temperature,
            )
            distractors = parse_generated_distractors(response, config.num_distractors)
            
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
    """Get the output column name for a generation mode."""
    mapping = {
        AugmentorMode.FROM_SCRATCH: DistractorType.COND_MODEL_Q_A.value,
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
    client_init = {
        "openai": _get_openai_client,
        "anthropic": _get_anthropic_client,
        "google": _get_google_client,
    }.get(config.model_provider)
    
    if not client_init:
        raise ValueError(f"Unknown provider: {config.model_provider}")
    
    client = client_init()
    
    # Load existing results if resuming
    processed = {}
    if resume_from and resume_from.exists():
        with open(resume_from, 'r') as f:
            processed = json.load(f)
        print(f"Resuming from {len(processed)} processed entries")
    
    # Set up output paths
    output_column = get_output_column(config.mode)
    
    if output_path is None:
        output_path = DATASETS_DIR / f"augmented_{config.mode.value}"
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
