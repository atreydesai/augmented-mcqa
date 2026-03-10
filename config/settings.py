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

def _ensure_writable_dir(path: Path, fallback: Path, name: str) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError:
        print(f"⚠️ {name} not writable at {path}; falling back to {fallback}")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


DATASETS_DIR = _ensure_writable_dir(DATASETS_DIR, PROJECT_ROOT / "datasets", "DATASETS_DIR")
RESULTS_DIR = _ensure_writable_dir(RESULTS_DIR, PROJECT_ROOT / "results", "RESULTS_DIR")

# Additional Dataset Paths
RAW_DATASETS_DIR = DATASETS_DIR / "raw"
PROCESSED_DATASETS_DIR = DATASETS_DIR / "processed"
AUGMENTED_DATASETS_DIR = DATASETS_DIR / "augmented"


# =============================================================================
# API Configuration
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# HuggingFace settings
_hf_home_env = os.getenv("HF_HOME")
if _hf_home_env:
    HF_HOME = _hf_home_env
else:
    default_hf_home = Path(MODEL_CACHE_DIR)
    if str(default_hf_home).startswith("/fs"):
        HF_HOME = str(PROJECT_ROOT / ".hf_cache")
    else:
        HF_HOME = str(default_hf_home)
HF_SKIP_PUSH = os.getenv("HF_SKIP_PUSH", "0").lower() in ("1", "true", "yes")

# Set HF environment variables
try:
    Path(HF_HOME).mkdir(parents=True, exist_ok=True)
except OSError:
    fallback_hf_home = PROJECT_ROOT / ".hf_cache"
    print(f"⚠️ HF_HOME not writable at {HF_HOME}; falling back to {fallback_hf_home}")
    fallback_hf_home.mkdir(parents=True, exist_ok=True)
    HF_HOME = str(fallback_hf_home)
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = str(Path(HF_HOME) / "transformers")
os.environ["HF_DATASETS_CACHE"] = str(Path(HF_HOME) / "datasets")


# =============================================================================
# Shared Constants
# =============================================================================

CHOICE_LABELS = "ABCDEFGHIJ"


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
    - choices_human: 3 human distractors from original MMLU/ARC/GPQA
    - cond_model_q_a_scratch: NEWLY GENERATED from Q+A only (no conditioning)
    - cond_model_q_a_dhuman: NEWLY GENERATED conditioned on 3 human distractors
    - cond_model_q_a_dmodel: NEWLY GENERATED conditioned on 3 random scratch distractors
    """
    # Human distractors from original MMLU/ARC/GPQA (up to 3)
    COND_HUMAN_Q_A = "choices_human"
    
    # NEWLY GENERATED distractors from Q+A only (no conditioning)
    COND_MODEL_Q_A_SCRATCH = "cond_model_q_a_scratch"
    
    # NEWLY GENERATED distractors conditioned on Q+A + 3 human distractors
    COND_MODEL_Q_A_DHUMAN = "cond_model_q_a_dhuman"
    
    # NEWLY GENERATED distractors conditioned on Q+A + 3 random scratch distractors
    COND_MODEL_Q_A_DMODEL = "cond_model_q_a_dmodel"


# =============================================================================
# Dataset Type and Schema Configuration
# =============================================================================

class DatasetType(Enum):
    """
    Enum for all supported dataset types.
    
    Each type has specific column mappings in DATASET_SCHEMA.
    """
    # Internal support for MMLU is retained because MMLU-Pro filtering depends on it.
    MMLU = "mmlu"
    MMLU_PRO = "mmlu_pro"
    ARC_CHALLENGE = "arc_challenge"
    GPQA = "gpqa"


ACTIVE_DATASET_TYPES = [
    DatasetType.ARC_CHALLENGE.value,
    DatasetType.MMLU_PRO.value,
    DatasetType.GPQA.value,
]


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
    DatasetType.GPQA: {
        # Idavidrein/gpqa (subset=gpqa_main, split=train)
        # columns include:
        # ['Question', 'Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2',
        #  'Incorrect Answer 3', 'Subdomain', ...]
        "question": "Question",
        "options": None,                # no pre-bundled options column in source
        "answer_text": "Correct Answer",
        "answer_letter": None,
        "answer_index": None,           # not provided directly in source
        "category": None,
        "discipline": None,
        "subfield": "Subdomain",
        "difficulty": None,
        "num_options": 4,
        "hf_path": "Idavidrein/gpqa",
        "hf_config": "gpqa_main",
        "splits": ["train"],
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

    if not options_key:
        return []
    
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
    "arc_challenge": DatasetConfig(
        name="arc_challenge",
        hf_path="allenai/ai2_arc",
        splits=["test"],
        dataset_type=DatasetType.ARC_CHALLENGE,
    ),
    "gpqa": DatasetConfig(
        name="gpqa",
        hf_path="Idavidrein/gpqa",
        splits=["train"],
        dataset_type=DatasetType.GPQA,
    ),
}

