from __future__ import annotations

from pathlib import Path

from config import ACTIVE_DATASET_TYPES, DATASETS_DIR, PROCESSED_DATASETS_DIR, RESULTS_DIR

CHOICE_LABELS = "ABCDEFGHIJ"
FINAL5_SETTINGS = (
    "human_from_scratch",
    "model_from_scratch",
    "augment_human",
    "augment_model",
    "augment_ablation",
)
MODE_CHOICES = ("full_question", "choices_only")
SETTING_SPECS: dict[str, dict[str, int]] = {
    "human_from_scratch": {"num_human": 3, "num_model": 0, "num_choices": 4},
    "model_from_scratch": {"num_human": 0, "num_model": 3, "num_choices": 4},
    "augment_human": {"num_human": 3, "num_model": 6, "num_choices": 10},
    "augment_model": {"num_human": 0, "num_model": 9, "num_choices": 10},
    "augment_ablation": {"num_human": 0, "num_model": 9, "num_choices": 10},
}
DATASET_ORDER = tuple(ACTIVE_DATASET_TYPES)
DEFAULT_GENERATION_MODELS = (
    "gpt-5.2-2025-12-11",
    "claude-opus-4-6",
    "gemini-3.1-pro-preview",
)
DEFAULT_EVALUATION_MODELS = (
    "Qwen/Qwen3-4B-Instruct-2507",
    "allenai/Olmo-3-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
)
DEFAULT_LOCAL_CLUSTER_MODELS = DEFAULT_EVALUATION_MODELS
DEFAULT_PROCESSED_DATASET = PROCESSED_DATASETS_DIR / "unified_processed_v2"
DEFAULT_INSPECT_RESULTS_DIR = RESULTS_DIR / "inspect"
DEFAULT_GENERATION_LOG_ROOT = DEFAULT_INSPECT_RESULTS_DIR / "generation"
DEFAULT_EVALUATION_LOG_ROOT = DEFAULT_INSPECT_RESULTS_DIR / "evaluation"
DEFAULT_AUGMENTED_CACHE_ROOT = DATASETS_DIR / "augmented"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
GENERATION_RETRY_LIMIT = 3
