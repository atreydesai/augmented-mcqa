from typing import List, Dict, Optional, Tuple
import re

from config import MCQA_PROMPT_FULL, MCQA_PROMPT_CHOICES_ONLY

CHOICE_LABELS = "ABCDEFGHIJ"


def build_mcqa_prompt(
    question: str,
    options: List[str],
    choices_only: bool = False,
    template: str = MCQA_PROMPT_FULL,
) -> str:
    options_str = "\n".join(
        f"{CHOICE_LABELS[i]}. {opt.strip()}"
        for i, opt in enumerate(options)
    )
    
    if choices_only:
        return MCQA_PROMPT_CHOICES_ONLY.format(
            options=options_str,
        )
    else:
        return MCQA_PROMPT_FULL.format(
            question=question,
            options=options_str,
        )


def extract_answer(text: str) -> str:
    text = text.strip()
    
    # Pattern 1: "The answer is (X)" or "The answer is X"
    pattern1 = r"[Tt]he answer is \(?([A-J])\)?"
    match = re.search(pattern1, text)
    if match:
        return match.group(1).upper()
    
    # Pattern 2: Starts with a letter followed by period or colon
    pattern2 = r"^([A-J])[\.:\)]"
    match = re.match(pattern2, text)
    if match:
        return match.group(1).upper()
    
    # Pattern 3: Last standalone letter in the response
    pattern3 = r"\b([A-J])\b"
    matches = re.findall(pattern3, text)
    if matches:
        return matches[-1].upper()
    
    return ""


def check_correctness(
    prediction: str,
    gold_index: int,
) -> bool:
    if not prediction or gold_index < 0:
        return False
    
    gold_letter = CHOICE_LABELS[gold_index]
    return prediction.upper() == gold_letter


def get_prediction_type(
    prediction: str,
    gold_index: int,
    human_indices: List[int],
    model_indices: List[int],
) -> str:
    if not prediction or prediction not in CHOICE_LABELS:
        return "?"
    
    pred_idx = CHOICE_LABELS.index(prediction)
    
    if pred_idx == gold_index:
        return "G"
    elif pred_idx in human_indices:
        return "H"
    elif pred_idx in model_indices:
        return "M"
    else:
        return "?"


def compute_accuracy(
    predictions: List[str],
    gold_indices: List[int],
) -> float:
    if not predictions:
        return 0.0
    
    correct = sum(
        check_correctness(pred, gold)
        for pred, gold in zip(predictions, gold_indices)
    )
    
    return correct / len(predictions)


def compute_behavioral_signature(
    predictions: List[str],
    gold_indices: List[int],
    human_indices_list: List[List[int]],
    model_indices_list: List[List[int]],
) -> Dict[str, int]:
    counts = {"G": 0, "H": 0, "M": 0, "?": 0}
    
    for pred, gold, human_idx, model_idx in zip(
        predictions, gold_indices, human_indices_list, model_indices_list
    ):
        ptype = get_prediction_type(pred, gold, human_idx, model_idx)
        counts[ptype] = counts.get(ptype, 0) + 1
    
    return counts


def compute_gold_rate(predictions: List[str], gold_indices: List[int]) -> float:
    return compute_accuracy(predictions, gold_indices)


def compute_human_rate(
    predictions: List[str],
    human_indices_list: List[List[int]],
) -> float:
    if not predictions:
        return 0.0
    
    human_count = 0
    for pred, human_indices in zip(predictions, human_indices_list):
        if pred and pred in CHOICE_LABELS:
            pred_idx = CHOICE_LABELS.index(pred)
            if pred_idx in human_indices:
                human_count += 1
    
    return human_count / len(predictions)


def compute_model_rate(
    predictions: List[str],
    model_indices_list: List[List[int]],
) -> float:
    if not predictions:
        return 0.0
    
    model_count = 0
    for pred, model_indices in zip(predictions, model_indices_list):
        if pred and pred in CHOICE_LABELS:
            pred_idx = CHOICE_LABELS.index(pred)
            if pred_idx in model_indices:
                model_count += 1
    
    return model_count / len(predictions)
