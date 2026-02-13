"""
Central configuration module for Augmented MCQA.

This module provides:
- Environment variable loading
- Path configurations
- Unified distractor naming conventions
- Prompt templates
- Model configurations
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# =============================================================================
# Path Configuration
# =============================================================================

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = Path(os.getenv("DATASETS_DIR", PROJECT_ROOT / "datasets"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", PROJECT_ROOT / "results"))
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "/fs/nexus-scratch/adesai10/hub"))

# Additional Dataset Paths
RAW_DATASETS_DIR = DATASETS_DIR / "raw"
PROCESSED_DATASETS_DIR = DATASETS_DIR / "processed"
AUGMENTED_DATASETS_DIR = DATASETS_DIR / "augmented"
AUGMENTED_FROM_SCRATCH_DIR = AUGMENTED_DATASETS_DIR / "from_scratch"
AUGMENTED_CONDITIONED_HUMAN_DIR = AUGMENTED_DATASETS_DIR / "conditioned_human"
AUGMENTED_CONDITIONED_SYNTHETIC_DIR = AUGMENTED_DATASETS_DIR / "conditioned_synthetic"


# Ensure directories exist
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# API Configuration
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# HuggingFace settings
HF_HOME = os.getenv("HF_HOME", str(MODEL_CACHE_DIR))
HF_SKIP_PUSH = os.getenv("HF_SKIP_PUSH", "0").lower() in ("1", "true", "yes")

# Set HF environment variables
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = str(Path(HF_HOME) / "transformers")
os.environ["HF_DATASETS_CACHE"] = str(Path(HF_HOME) / "datasets")


# =============================================================================
# Experiment Configuration
# =============================================================================

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "12345"))
DEFAULT_LIMIT = os.getenv("DEFAULT_LIMIT")
if DEFAULT_LIMIT and DEFAULT_LIMIT.lower() != "none":
    DEFAULT_LIMIT = int(DEFAULT_LIMIT)
else:
    DEFAULT_LIMIT = None


# =============================================================================
# Unified Distractor Naming Convention
# =============================================================================

class DistractorType(Enum):
    """
    Unified naming convention for distractor types.
    
    These names are used consistently throughout the codebase:
    - In dataset columns
    - In experiment configurations
    - In results and analysis
    
    IMPORTANT DISTINCTION:
    - cond_human_q_a: 3 human distractors from original MMLU/ARC/SuperGPQA
    - cond_model_q_a: EXISTING 6 synthetic distractors from MMLU-Pro (NOT for ARC/SuperGPQA)
    - cond_model_q_a_scratch: NEWLY GENERATED from Q+A only (no conditioning)
    - cond_model_q_a_dhuman: NEWLY GENERATED conditioned on 3 human distractors
    - cond_model_q_a_dmodel: NEWLY GENERATED conditioned on 3 random existing synthetic
    """
    # Human distractors from original MMLU/ARC/SuperGPQA (up to 3)
    COND_HUMAN_Q_A = "cond_human_q_a"
    
    # EXISTING synthetic distractors from MMLU-Pro only (up to 6)
    # NOT available for ARC or SuperGPQA
    COND_MODEL_Q_A = "cond_model_q_a"
    
    # NEWLY GENERATED distractors from Q+A only (no conditioning)
    COND_MODEL_Q_A_SCRATCH = "cond_model_q_a_scratch"
    
    # NEWLY GENERATED distractors conditioned on Q+A + 3 human distractors
    COND_MODEL_Q_A_DHUMAN = "cond_model_q_a_dhuman"
    
    # NEWLY GENERATED distractors conditioned on Q+A + 3 random existing synthetic
    COND_MODEL_Q_A_DMODEL = "cond_model_q_a_dmodel"


# Mapping from legacy column names to unified names
LEGACY_COLUMN_MAPPING: Dict[str, DistractorType] = {
    # Human distractors
    "choices_human": DistractorType.COND_HUMAN_Q_A,
    
    # Model distractors (q+a conditioned)
    "choices_synthetic": DistractorType.COND_MODEL_Q_A,
    "choices_synthetic_conditioned_goldandstem": DistractorType.COND_MODEL_Q_A,
    
    # Model distractors (conditioned on human)
    "choices_newsynthetic": DistractorType.COND_MODEL_Q_A_DHUMAN,
    
    # Model distractors (conditioned on model)
    "choices_newsynthetic_conditioned_synthetic": DistractorType.COND_MODEL_Q_A_DMODEL,
}


# Reverse mapping: unified name to possible legacy column names
UNIFIED_TO_LEGACY: Dict[DistractorType, List[str]] = {
    DistractorType.COND_HUMAN_Q_A: ["choices_human"],
    DistractorType.COND_MODEL_Q_A: [
        "choices_synthetic",
        "choices_synthetic_conditioned_goldandstem",
    ],
    DistractorType.COND_MODEL_Q_A_DHUMAN: ["choices_newsynthetic"],
    DistractorType.COND_MODEL_Q_A_DMODEL: ["choices_newsynthetic_conditioned_synthetic"],
}


def get_distractor_column(entry: dict, distractor_type: DistractorType) -> List[str]:
    """
    Get distractor values from a dataset entry using the unified naming convention.
    
    Handles both legacy and new column names transparently.
    
    Args:
        entry: A dataset entry dictionary
        distractor_type: The type of distractors to retrieve
        
    Returns:
        List of distractor strings
    """
    # First try the unified name
    unified_name = distractor_type.value
    if unified_name in entry:
        return list(entry[unified_name])
    
    # Fall back to legacy names
    for legacy_name in UNIFIED_TO_LEGACY.get(distractor_type, []):
        if legacy_name in entry and entry[legacy_name]:
            return list(entry[legacy_name])
    
    return []


# =============================================================================
# Dataset Type and Schema Configuration
# =============================================================================

class DatasetType(Enum):
    """
    Enum for all supported dataset types.
    
    Each type has specific column mappings in DATASET_SCHEMA.
    """
    MMLU = "mmlu"
    MMLU_PRO = "mmlu_pro"
    ARC_EASY = "arc_easy"
    ARC_CHALLENGE = "arc_challenge"
    SUPERGPQA = "supergpqa"


# Exact column mappings for each dataset type (based on HuggingFace analysis)
DATASET_SCHEMA = {
    DatasetType.MMLU: {
        # cais/mmlu - Original MMLU dataset
        # columns: ['question', 'subject', 'choices', 'answer']
        "question": "question",
        "options": "choices",           # list of 4 strings
        "answer_index": "answer",       # int 0-3
        "answer_letter": None,          # not provided, compute from index
        "category": "subject",          # e.g., 'abstract_algebra'
        "num_options": 4,
        "hf_path": "cais/mmlu",
        "hf_config": "all",
        "splits": ["test", "validation", "dev"],
    },
    DatasetType.MMLU_PRO: {
        # TIGER-Lab/MMLU-Pro - Extended MMLU with 10 options
        # columns: ['question_id', 'question', 'options', 'answer', 'answer_index', 
        #           'cot_content', 'category', 'src']
        "question": "question",
        "options": "options",           # list of ~10 strings
        "answer_index": "answer_index", # int 0-9
        "answer_letter": "answer",      # 'A'-'J'
        "category": "category",         # e.g., 'business', 'physics'
        "src": "src",                   # original source dataset
        "num_options": 10,
        "hf_path": "TIGER-Lab/MMLU-Pro",
        "hf_config": None,
        "splits": ["test", "validation"],
    },
    DatasetType.ARC_EASY: {
        # allenai/ai2_arc (ARC-Easy config)
        # columns: ['id', 'question', 'choices', 'answerKey']
        # choices = {'text': ['opt1',...], 'label': ['A','B','C','D']}
        "question": "question",
        "options": "choices.text",      # nested: choices['text']
        "labels": "choices.label",      # nested: choices['label']
        "answer_letter": "answerKey",   # 'A'-'D'
        "answer_index": None,           # compute from answerKey
        "category": None,               # not provided
        "num_options": 4,
        "hf_path": "allenai/ai2_arc",
        "hf_config": "ARC-Easy",
        "splits": ["test", "validation", "train"],
    },
    DatasetType.ARC_CHALLENGE: {
        # allenai/ai2_arc (ARC-Challenge config)
        # Same structure as ARC-Easy but harder questions
        "question": "question",
        "options": "choices.text",
        "labels": "choices.label",
        "answer_letter": "answerKey",
        "answer_index": None,
        "category": None,
        "num_options": 4,
        "hf_path": "allenai/ai2_arc",
        "hf_config": "ARC-Challenge",
        "splits": ["test", "validation", "train"],
    },
    DatasetType.SUPERGPQA: {
        # m-a-p/SuperGPQA - Graduate-level questions
        # columns: ['uuid', 'question', 'options', 'answer', 'answer_letter',
        #           'discipline', 'field', 'subfield', 'difficulty', 'is_calculation']
        # Filter to 10-option questions only (87.3% of dataset)
        "question": "question",
        "options": "options",           # list of strings (filter to 10)
        "answer_text": "answer",        # full text of correct answer
        "answer_letter": "answer_letter", # 'A'-'J'
        "answer_index": None,           # compute from answer_letter
        "category": "field",            # e.g., 'Electronic Science'
        "discipline": "discipline",     # e.g., 'Engineering'
        "subfield": "subfield",
        "difficulty": "difficulty",     # 'middle' / 'hard'
        "num_options": 10,              # filter to 10 only
        "hf_path": "m-a-p/SuperGPQA",
        "hf_config": None,
        "splits": ["train"],            # only train split available
    },
}


def get_answer_index(entry: dict, dataset_type: DatasetType) -> int:
    """Get answer index from entry based on dataset type."""
    schema = DATASET_SCHEMA[dataset_type]
    
    # If answer_index is directly available
    if schema.get("answer_index") and schema["answer_index"] in entry:
        return int(entry[schema["answer_index"]])
    
    # Compute from answer_letter
    if schema.get("answer_letter") and schema["answer_letter"] in entry:
        letter = entry[schema["answer_letter"]]
        if letter and len(letter) == 1:
            return ord(letter.upper()) - ord('A')
    
    return 0


def get_options_from_entry(entry: dict, dataset_type: DatasetType) -> List[str]:
    """Get options list from entry based on dataset type."""
    schema = DATASET_SCHEMA[dataset_type]
    options_key = schema["options"]
    
    # Handle nested dict access (e.g., 'choices.text')
    if "." in options_key:
        parts = options_key.split(".")
        val = entry
        for part in parts:
            val = val.get(part, [])
        return list(val) if val else []
    
    return list(entry.get(options_key, []))


# =============================================================================
# Dataset Configuration
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    hf_path: str
    local_path: Optional[Path] = None
    splits: List[str] = field(default_factory=lambda: ["test"])
    answer_column: str = "answer"
    question_column: str = "question"
    options_column: str = "options"
    gold_column: str = "choices_answer"
    dataset_type: Optional[DatasetType] = None
    
    def __post_init__(self):
        if self.local_path is None:
            self.local_path = RAW_DATASETS_DIR / self.name


# Pre-configured datasets
DATASET_CONFIGS = {
    "mmlu": DatasetConfig(
        name="mmlu",
        hf_path="cais/mmlu",
        splits=["test"],
        dataset_type=DatasetType.MMLU,
    ),
    "mmlu_pro": DatasetConfig(
        name="mmlu_pro",
        hf_path="TIGER-Lab/MMLU-Pro",
        splits=["test", "validation"],
        dataset_type=DatasetType.MMLU_PRO,
    ),
    "arc_easy": DatasetConfig(
        name="arc_easy",
        hf_path="allenai/ai2_arc",
        splits=["test"],
        dataset_type=DatasetType.ARC_EASY,
    ),
    "arc_challenge": DatasetConfig(
        name="arc_challenge",
        hf_path="allenai/ai2_arc",
        splits=["test"],
        dataset_type=DatasetType.ARC_CHALLENGE,
    ),
    "supergpqa": DatasetConfig(
        name="supergpqa",
        hf_path="m-a-p/SuperGPQA",
        splits=["train"],
        dataset_type=DatasetType.SUPERGPQA,
    ),
}


# =============================================================================
# Prompt Templates
# =============================================================================

MCQA_PROMPT_FULL = """The following are multiple choice questions (with answers). Output the answer in the format of "The answer is (X)" at the end.

Question: {question}
{options}
Answer: """

MCQA_PROMPT_CHOICES_ONLY = """The following are multiple choice options. Output the answer in the format of "The answer is (X)" at the end.

{options}
Answer: """

DISTRACTOR_GENERATION_PROMPT = """I have a multiple-choice question with the single correct answer, and I need to expand it to a seven-option multiple-choice question. Please generate six additional plausible but incorrect options (B, C, D, E, F, G) to accompany the correct answer choice. Do not output anything except the incorrect options.

Input:

Question: {question}

Answer: A: {gold_answer}

Please generate only the six new incorrect options B, C, D, E, F, and G. Output each option on a separate line in the format "B: <option>", "C: <option>", etc."""

DISTRACTOR_GENERATION_PROMPT_CONDITIONED = """I have a multiple-choice question with four options, one of which is correct, and I need to expand it to a ten-option multiple-choice question. The original options are A, B, C, and D, with one of them being the correct answer. Please generate six additional plausible but incorrect options (E, F, G, H, I, J) to accompany the original four.

Input:
Question: {question}
Existing 4 Options: A: {gold_answer}
B: {distractor_1}
C: {distractor_2}
D: {distractor_3}
Answer: A: {gold_answer}

Please generate only the six new incorrect options E, F, G, H, I, and J. Output each option on a separate line in the format "E: <option>", "F: <option>", etc."""


# =============================================================================
# Utility Functions
# =============================================================================

def get_api_key(provider: str) -> str:
    """Get API key for a provider."""
    keys = {
        "openai": OPENAI_API_KEY,
        "anthropic": ANTHROPIC_API_KEY,
        "google": GOOGLE_API_KEY,
        "deepseek": DEEPSEEK_API_KEY,
    }
    key = keys.get(provider, "")
    if not key:
        raise ValueError(f"API key not configured for provider: {provider}. Check your .env file.")
    return key
