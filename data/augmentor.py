"""
Synthetic distractor generator for MCQA datasets.

Supports multiple generation modes:
- FROM_SCRATCH: Generate distractors from question + answer only
- CONDITIONED_HUMAN: Generate conditioned on human distractors
- CONDITIONED_SYNTHETIC: Generate conditioned on scratch distractors

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
    RESULTS_DIR,
    DistractorType,
    DISTRACTOR_GENERATION_PROMPT,
)
from models import get_client
from data.hub_utils import push_dataset_to_hub


class AugmentorMode(Enum):
    """
    Generation mode for synthetic distractors.
    
    These modes generate NEW distractors:
    - FROM_SCRATCH: New distractors from Q+A only (no conditioning)
    - CONDITIONED_HUMAN: New distractors conditioned on 3 human distractors
    - CONDITIONED_SYNTHETIC: New distractors conditioned on 3 random scratch distractors
    """
    # Generate from question + answer only
    # Output: cond_model_q_a_scratch (NEW generated, not the existing MMLU-Pro ones)
    FROM_SCRATCH = "from_scratch"
    
    # Generate conditioned on 3 human distractors
    # Output: cond_model_q_a_dhuman
    CONDITIONED_HUMAN = "conditioned_human"
    
    # Generate conditioned on 3 RANDOMLY SELECTED scratch distractors
    # Output: cond_model_q_a_dmodel
    CONDITIONED_SYNTHETIC = "conditioned_synthetic"


class NonRetryableAugmentationError(RuntimeError):
    """Raised for deterministic data/schema issues that should fail fast."""


def _require_column_list(entry: Dict[str, Any], column_name: str) -> List[str]:
    values = entry.get(column_name)
    if values is None:
        raise NonRetryableAugmentationError(
            f"Missing required column '{column_name}' in dataset entry"
        )
    return list(values)


@dataclass
class GenerationConfig:
    """Configuration for distractor generation."""
    mode: AugmentorMode
    model_provider: str  # "openai", "anthropic", "google"
    model_name: str
    num_distractors: int = 9  # Default: generate 9 to make 10 total options
    max_tokens: int = 2048  # Max output tokens per API call
    max_retries: int = 3
    retry_delay: float = 1.0
    save_interval: int = 50  # Save intermediate results every N entries
    reasoning_effort: Optional[str] = None
    generate_branching_prefix_columns: bool = False
    skip_failed_entries: bool = False


BRANCHING_PREFIX_MODEL_COLUMNS = {
    1: "cond_model_q_a_dhuman_h1",
    2: "cond_model_q_a_dhuman_h2",
    3: "cond_model_q_a_dhuman_h3",
}


def _build_conditioned_prompt(
    question: str,
    gold_answer: str,
    context_distractors: List[str],
    num_new_distractors: int,
) -> str:
    """Build a conditioned generation prompt for variable context/prefix lengths."""
    if num_new_distractors <= 0:
        raise ValueError(f"num_new_distractors must be > 0, got {num_new_distractors}")

    lines = [f"A: {gold_answer}"]
    for idx, distractor in enumerate(context_distractors, start=1):
        letter = chr(ord("A") + idx)
        lines.append(f"{letter}: {distractor}")

    start_letter_ord = ord("A") + len(context_distractors) + 1
    output_letters = [
        chr(start_letter_ord + offset)
        for offset in range(num_new_distractors)
    ]
    output_letters_str = ", ".join(output_letters)

    return (
        "I have a multiple-choice question with existing options where A is the correct answer.\n"
        "Please generate additional plausible but incorrect options.\n\n"
        f"Question: {question}\n"
        "Existing Options:\n"
        + "\n".join(lines)
        + "\n\n"
        f"Generate exactly {num_new_distractors} new incorrect options: {output_letters_str}.\n"
        "Output each option on a separate line in the format \"<LETTER>: <option>\"."
    )





def parse_generated_distractors(
    response: str,
    expected_count: int = 6,
    start_letter: str = "B",
) -> List[str]:
    """
    Parse generated distractors from model response using expected option letters.

    This parser is strict about required letters but tolerant of extra lines
    (e.g., model echoing existing options), which are ignored.
    """
    if expected_count <= 0:
        raise ValueError(f"expected_count must be > 0, got {expected_count}")
    if len(start_letter) != 1 or not start_letter.isalpha():
        raise ValueError(f"start_letter must be a single letter, got {start_letter!r}")

    start = start_letter.upper()
    start_ord = ord(start)
    expected_letters = [chr(start_ord + i) for i in range(expected_count)]
    expected_set = set(expected_letters)
    by_letter: Dict[str, str] = {}

    # Accept common separators: "E: text", "E. text", "E) text", "E - text"
    pattern = r"^\s*([A-Z])\s*(?:[:\.\)\-])\s*(.+?)\s*$"

    for line in response.strip().split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        match = re.match(pattern, stripped, re.IGNORECASE)
        if not match:
            continue

        letter = match.group(1).upper()
        if letter not in expected_set:
            continue

        text = match.group(2).strip().rstrip(".")
        if not text:
            continue

        # Keep first valid occurrence per expected letter for deterministic parsing.
        if letter not in by_letter:
            by_letter[letter] = text

    missing = [letter for letter in expected_letters if letter not in by_letter]
    if missing:
        raise ValueError(
            f"Expected exactly {expected_count} distractors ({expected_letters}), parsed "
            f"{len(by_letter)}. Missing letters: {missing}. Model output format is invalid."
        )

    return [by_letter[letter] for letter in expected_letters]


def _expected_letters(start_letter: str, expected_count: int) -> List[str]:
    start = start_letter.upper()
    return [chr(ord(start) + i) for i in range(expected_count)]


def _build_repair_prompt(
    raw_output: str,
    *,
    start_letter: str,
    expected_count: int,
) -> str:
    letters = _expected_letters(start_letter, expected_count)
    letters_str = ", ".join(letters)
    return (
        "Reformat the text below into strict MCQ distractor lines.\n"
        f"Output exactly {expected_count} lines with these letters only: {letters_str}.\n"
        "Each line must be in the exact format '<LETTER>: <option>'.\n"
        "Do not add explanations, markdown, numbering, or extra lines.\n\n"
        "Text to reformat:\n"
        f"{raw_output}"
    )


def _parse_or_repair_distractors(
    *,
    client,
    raw_text: str,
    expected_count: int,
    start_letter: str,
    max_tokens: int,
) -> (List[str], str):
    """
    Parse distractors; if parsing fails, request a strict reformat pass and parse again.
    """
    try:
        distractors = parse_generated_distractors(
            raw_text,
            expected_count=expected_count,
            start_letter=start_letter,
        )
        return distractors, raw_text
    except ValueError:
        repair_prompt = _build_repair_prompt(
            raw_text,
            start_letter=start_letter,
            expected_count=expected_count,
        )
        repaired = client.generate(prompt=repair_prompt, max_tokens=max_tokens)
        repaired_text = repaired.text
        distractors = parse_generated_distractors(
            repaired_text,
            expected_count=expected_count,
            start_letter=start_letter,
        )
        return distractors, repaired_text


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
        distractors = _require_column_list(entry, DistractorType.COND_HUMAN_Q_A.value)
        if not distractors or len(distractors) < 3:
            raise NonRetryableAugmentationError(
                f"CONDITIONED_HUMAN requires at least 3 human distractors in "
                f"{DistractorType.COND_HUMAN_Q_A.value}"
            )
        
        selected = distractors[:3]
        return _build_conditioned_prompt(
            question=question,
            gold_answer=gold_answer,
            context_distractors=selected,
            num_new_distractors=6,
        ), selected
    
    elif mode == AugmentorMode.CONDITIONED_SYNTHETIC:
        # User instructions: pick 3 from cond_model_q_a_scratch
        scratch_distractors = _require_column_list(
            entry, DistractorType.COND_MODEL_Q_A_SCRATCH.value
        )
        if not scratch_distractors or len(scratch_distractors) < 3:
            raise NonRetryableAugmentationError(
                "CONDITIONED_SYNTHETIC requires at least 3 distractors in cond_model_q_a_scratch"
            )
        
        # We don't random sample here so it's deterministic and traceable if we just take first 3, 
        # or we random sample but return which ones we picked. User asked for 3 of the 6 previously generated.
        selected = random.sample(scratch_distractors, 3)
        return _build_conditioned_prompt(
            question=question,
            gold_answer=gold_answer,
            context_distractors=selected,
            num_new_distractors=6,
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


def _generate_branching_prefix_columns(
    *,
    new_entry: Dict[str, Any],
    client,
    config: GenerationConfig,
    question: str,
    gold_answer: str,
    human_distractors: List[str],
    entry_idx: int,
) -> None:
    """
    Generate branching-specific model distractors for human-prefix conditions:
    - h=1 -> 5 model distractors
    - h=2 -> 4 model distractors
    - h=3 -> 3 model distractors
    """
    for human_prefix_count, column_name in BRANCHING_PREFIX_MODEL_COLUMNS.items():
        model_count = 6 - human_prefix_count
        if len(human_distractors) < human_prefix_count:
            raise NonRetryableAugmentationError(
                f"Branching generation requires at least {human_prefix_count} human distractors; "
                f"found {len(human_distractors)}"
            )

        prefix_humans = human_distractors[:human_prefix_count]
        prompt = _build_conditioned_prompt(
            question=question,
            gold_answer=gold_answer,
            context_distractors=prefix_humans,
            num_new_distractors=model_count,
        )

        success = False
        for attempt in range(config.max_retries):
            try:
                response = client.generate(prompt=prompt, max_tokens=config.max_tokens)
                start_letter = chr(ord("A") + human_prefix_count + 1)
                generated, model_output = _parse_or_repair_distractors(
                    client=client,
                    raw_text=response.text,
                    expected_count=model_count,
                    start_letter=start_letter,
                    max_tokens=config.max_tokens,
                )
                new_entry[column_name] = generated
                new_entry[f"{column_name}_model_input"] = prompt
                new_entry[f"{column_name}_model_output"] = model_output
                success = True
                break
            except Exception as exc:
                delay = config.retry_delay * (2 ** attempt)
                if attempt < config.max_retries - 1:
                    print(
                        f"⚠️ Entry {entry_idx}, branching h={human_prefix_count}, "
                        f"attempt {attempt + 1}/{config.max_retries}: {exc}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    print(
                        f"❌ Entry {entry_idx}, branching h={human_prefix_count}: "
                        f"Failed after {config.max_retries} attempts: {exc}"
                    )

        if not success:
            raise RuntimeError(
                f"Failed branching generation for entry={entry_idx}, h={human_prefix_count}, "
                f"column={column_name} after {config.max_retries} retries"
            )


def augment_dataset(
    dataset_path: Path,
    config: GenerationConfig,
    output_path: Optional[Path] = None,
    limit: Optional[int] = None,
    resume_from: Optional[Path] = None,
    push_to_hub: bool = True,
    splits: Optional[List[str]] = None,
) -> Dataset:
    """
    Augment a dataset with synthetic distractors and full MCQA columns.
    
    Args:
        splits: If provided, only process these specific splits (for parallelization).
    """
    # Load dataset
    dataset = load_from_disk(str(dataset_path))
    if isinstance(dataset, DatasetDict):
        # We process all splits in the dict if it's a Dict
        final_dataset_dict = {}
        items = dataset.items()
        if splits:
            missing_splits = sorted(set(splits) - set(dataset.keys()))
            if missing_splits:
                raise ValueError(f"Requested splits not found: {missing_splits}")
            items = [(k, v) for k, v in items if k in splits]
        if resume_from is not None and len(list(items)) != 1:
            raise ValueError(
                "resume_from is only supported when processing exactly one split"
            )
        for split_name, ds in items:
            print(f"\nProcessing split: {split_name}")
            final_dataset_dict[split_name] = augment_single_dataset(
                ds,
                config,
                limit,
                resume_from=resume_from,
            )
        result = DatasetDict(final_dataset_dict)
    else:
        result = augment_single_dataset(dataset, config, limit, resume_from=resume_from)

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


def augment_single_dataset(
    dataset: Dataset,
    config: GenerationConfig,
    limit: Optional[int] = None,
    resume_from: Optional[Path] = None,
) -> Dataset:
    """Process a single Dataset through all augmentation modes (scratch, human-conditioned, model-conditioned)."""
    client_kwargs: Dict[str, Any] = {}
    if config.model_provider == "openai" and config.reasoning_effort is not None:
        client_kwargs["reasoning_effort"] = config.reasoning_effort
    client = get_client(config.model_name, **client_kwargs)
    
    entries = list(dataset)
    if limit:
        entries = entries[:limit]

    augmented_entries: List[Dict[str, Any]] = []
    start_index = 0

    checkpoint_path: Optional[Path] = Path(resume_from) if resume_from else None
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        checkpoint_data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if not isinstance(checkpoint_data, list):
            raise ValueError(
                f"Resume checkpoint must be a list of entries: {checkpoint_path}"
            )
        if len(checkpoint_data) > len(entries):
            raise ValueError(
                "Resume checkpoint has more entries than target dataset: "
                f"{len(checkpoint_data)} > {len(entries)}"
            )
        for idx, row in enumerate(checkpoint_data):
            if not isinstance(row, dict):
                raise ValueError(
                    f"Resume checkpoint contains non-dict entry at position {idx}: {type(row)}"
                )
            row.setdefault("_source_index", idx)
        augmented_entries = list(checkpoint_data)
        start_index = (
            max(int(row["_source_index"]) for row in augmented_entries) + 1
            if augmented_entries
            else 0
        )
        if start_index > len(entries):
            raise ValueError(
                "Resume checkpoint source indices exceed dataset length: "
                f"start_index={start_index}, dataset_len={len(entries)}"
            )
        print(
            f"Resuming from checkpoint {checkpoint_path} "
            f"(resume_start_index={start_index}, kept_entries={len(augmented_entries)}, total={len(entries)})"
        )
    
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
    
    if checkpoint_path is not None:
        temp_save_path = checkpoint_path
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_save_path = RESULTS_DIR / f"temp_augmented_{config.model_name}_{timestamp}.json"
    temp_save_path.parent.mkdir(parents=True, exist_ok=True)

    if start_index >= len(entries):
        print("Checkpoint already covers all entries; nothing to generate.")
        return Dataset.from_list(augmented_entries)

    skipped_entries = 0
    iterator = tqdm(
        entries[start_index:],
        desc="Augmenting multi-mode",
        initial=start_index,
        total=len(entries),
    )
    for i, entry in enumerate(iterator, start=start_index):
        new_entry = dict(entry)
        new_entry["_source_index"] = i
        
        # Core data needed for all modes
        answer = new_entry.get("answer", "")
        if not answer and new_entry.get("choices_answer"):
            answer = new_entry["choices_answer"][0]
        
        human_distractors = _require_column_list(new_entry, DistractorType.COND_HUMAN_Q_A.value)
        
        skip_current_entry = False
        for mode in modes_to_run:
            prefix = mode_prefixes[mode]
            model_col = mode_model_cols[mode]
            success = False
            
            for attempt in range(config.max_retries):
                try:
                    # 1. Build Prompt (now returns distractors used for context)
                    prompt, context_distractors = build_prompt(new_entry, mode)
                    
                    # 2. Generate
                    response = client.generate(prompt=prompt, max_tokens=config.max_tokens)
                    raw_text = response.text
                    
                    # 3. Parse Distractors (Always 6 per user prompt)
                    start_letter = chr(ord("A") + len(context_distractors) + 1)
                    distractors, model_output = _parse_or_repair_distractors(
                        client=client,
                        raw_text=raw_text,
                        expected_count=6,
                        start_letter=start_letter,
                        max_tokens=config.max_tokens,
                    )
                    
                    # 4. Assembly (1 answer + 6 synthetic + 3 human = 10 total)
                    options, correct_letter = assemble_mcqa_options(answer, human_distractors, distractors, mode)
                    
                    # 5. Column Population (Ordered as requested)
                    if mode == AugmentorMode.FROM_SCRATCH:
                        new_entry[f"{prefix}_full_question"] = f"Question: {new_entry['question']}\nAnswer: A: {answer}"
                    else:
                        context_options = [f"A: {answer}"]
                        for j, d in enumerate(context_distractors):
                            letter = chr(ord('B') + j)
                            context_options.append(f"{letter}: {d}")
                        options_str = "\n".join(context_options)
                        new_entry[f"{prefix}_full_question"] = f"Question: {new_entry['question']}\nExisting 4 Options: {options_str}\nAnswer: A: {answer}"
                    
                    new_entry[f"{prefix}_model_input"] = prompt
                    new_entry[f"{prefix}_model_output"] = model_output
                    new_entry[model_col] = distractors
                    new_entry[f"{prefix}_options_randomized"] = options
                    new_entry[f"{prefix}_correct_answer_letter"] = correct_letter
                    
                    success = True
                    break
                    
                except Exception as e:
                    if isinstance(e, NonRetryableAugmentationError):
                        raise
                    delay = config.retry_delay * (2 ** attempt)
                    if attempt < config.max_retries - 1:
                        print(f"⚠️ Entry {i}, mode {mode.value}, attempt {attempt+1}/{config.max_retries}: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        print(f"❌ Entry {i}, mode {mode.value}: Failed after {config.max_retries} attempts: {e}")
            
            if not success:
                if config.skip_failed_entries:
                    print(
                        f"⏭️ Entry {i}: skipping question after {config.max_retries} "
                        f"failed attempts in mode={mode.value}"
                    )
                    skip_current_entry = True
                    skipped_entries += 1
                    break
                raise RuntimeError(
                    f"Failed generation for entry={i}, mode={mode.value} after {config.max_retries} retries"
                )

        if skip_current_entry:
            continue

        # Branching prefix columns are expensive to generate; only do this when explicitly requested.
        if config.generate_branching_prefix_columns:
            try:
                _generate_branching_prefix_columns(
                    new_entry=new_entry,
                    client=client,
                    config=config,
                    question=new_entry.get("question", ""),
                    gold_answer=answer,
                    human_distractors=human_distractors,
                    entry_idx=i,
                )
            except Exception as exc:
                if config.skip_failed_entries:
                    print(
                        f"⏭️ Entry {i}: skipping question due to branching generation failure: {exc}"
                    )
                    skipped_entries += 1
                    continue
                raise
                
        augmented_entries.append(new_entry)
        
        # Intermediate saving
        if (i + 1) % config.save_interval == 0:
            with open(temp_save_path, 'w') as f:
                json.dump(augmented_entries, f, indent=2)
            print(
                f"\n💾 Saved intermediate results ({len(augmented_entries)} kept entries, "
                f"processed={i+1}/{len(entries)}, skipped={skipped_entries}) to {temp_save_path}"
            )
            
    # Keep temp file as a recovery point (not deleted)
    with open(temp_save_path, "w") as f:
        json.dump(augmented_entries, f, indent=2)
    print(
        f"Completed split generation: kept={len(augmented_entries)}, "
        f"skipped={skipped_entries}, total={len(entries)}"
    )
        
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
