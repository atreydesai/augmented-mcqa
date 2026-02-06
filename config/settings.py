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
    """
    # Human distractors from original MMLU
    COND_HUMAN_Q_A = "cond_human_q_a"
    
    # Model distractors conditioned on question + answer only
    COND_MODEL_Q_A = "cond_model_q_a"
    
    # Model distractors conditioned on question + answer + human distractors
    COND_MODEL_Q_A_DHUMAN = "cond_model_q_a_dhuman"
    
    # Model distractors conditioned on question + answer + model distractors
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
    
    def __post_init__(self):
        if self.local_path is None:
            self.local_path = DATASETS_DIR / self.name


# Pre-configured datasets
DATASET_CONFIGS = {
    "mmlu_pro": DatasetConfig(
        name="mmlu_pro",
        hf_path="TIGER-Lab/MMLU-Pro",
        splits=["test", "validation"],
    ),
    "mmlu": DatasetConfig(
        name="mmlu",
        hf_path="cais/mmlu",
        splits=["test"],
    ),
    "arc_easy": DatasetConfig(
        name="arc_easy",
        hf_path="allenai/ai2_arc",
        splits=["test"],
    ),
    "supergpqa": DatasetConfig(
        name="supergpqa",
        hf_path="m-a-p/SuperGPQA",
        splits=["test"],
    ),
}


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    provider: str  # "openai", "anthropic", "google", "deepseek", "local"
    model_id: str
    supports_logprobs: bool = False
    max_tokens: int = 4000
    temperature: float = 0.0
    
    
# Pre-configured models
MODEL_CONFIGS = {
    # OpenAI
    "gpt-4.1": ModelConfig(
        name="gpt-4.1",
        provider="openai",
        model_id="gpt-4.1-2025-04-14",
        supports_logprobs=True,
    ),
    "gpt-4": ModelConfig(
        name="gpt-4",
        provider="openai",
        model_id="gpt-4",
        supports_logprobs=True,
    ),
    
    # Anthropic
    "claude-3.5-sonnet": ModelConfig(
        name="claude-3.5-sonnet",
        provider="anthropic",
        model_id="claude-3-5-sonnet-20241022",
        supports_logprobs=False,
    ),
    "claude-3-opus": ModelConfig(
        name="claude-3-opus",
        provider="anthropic",
        model_id="claude-3-opus-20240229",
        supports_logprobs=False,
    ),
    
    # Google Gemini
    "gemini-1.5-pro": ModelConfig(
        name="gemini-1.5-pro",
        provider="google",
        model_id="gemini-1.5-pro",
        supports_logprobs=False,
    ),
    "gemini-1.5-flash": ModelConfig(
        name="gemini-1.5-flash",
        provider="google",
        model_id="gemini-1.5-flash",
        supports_logprobs=False,
    ),
    
    # DeepSeek
    "deepseek-chat": ModelConfig(
        name="deepseek-chat",
        provider="deepseek",
        model_id="deepseek-chat",
        supports_logprobs=True,
    ),
    
    # Local models
    "qwen3-8b": ModelConfig(
        name="qwen3-8b",
        provider="local",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        supports_logprobs=True,
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

DISTRACTOR_GENERATION_PROMPT = """I have a multiple-choice question with the single correct answer, and I need to expand it to a ten-option multiple-choice question. Please generate nine additional plausible but incorrect options (B, C, D, E, F, G, H, I, J) to accompany the correct answer choice. Do not output anything except the incorrect options.

Input:

Question: {question}

Answer: A: {gold_answer}

Please generate only the nine new incorrect options B, C, D, E, F, G, H, I, and J. Output each option on a separate line in the format "B: <option>", "C: <option>", etc."""

DISTRACTOR_GENERATION_PROMPT_CONDITIONED = """Option Augmentation Prompt Instruction (One-shot)
I have a multiple-choice question with four options, one of which is correct, and I need to expand it to a ten-option multiple-choice question. The original options are A, B, C, and D, with one of them being the correct answer. Please generate six additional plausible but incorrect options (E, F, G, H, I, J) to accompany the original four.

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


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a model by name."""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    
    # Try to find by model_id
    for config in MODEL_CONFIGS.values():
        if config.model_id == model_name:
            return config
    
    raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
