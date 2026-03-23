from __future__ import annotations

from pathlib import Path

from config import ACTIVE_DATASET_TYPES, DATASETS_DIR, PROCESSED_DATASETS_DIR, RESULTS_DIR
from utils.recipes import load_setting_recipes, setting_specs

CHOICE_LABELS = "ABCDEFGHIJ"
FINAL5_SETTINGS = tuple(recipe.name for recipe in load_setting_recipes())
MODE_CHOICES = ("full_question", "choices_only")
SETTING_SPECS: dict[str, dict[str, int]] = setting_specs()
DATASET_ORDER = tuple(ACTIVE_DATASET_TYPES)
DEFAULT_GENERATION_MODELS = (
    "gpt-5.2-2025-12-11",
    "claude-opus-4-6",
    "gemini-3.1-pro-preview",
    "Qwen/Qwen3.5-397B-A17B",
    "Qwen/Qwen3.5-9B",
)
DEFAULT_LOCAL_GENERATION_MODELS = (
    "Qwen/Qwen3-4B-Instruct-2507",
    "allenai/Olmo-3-7B-Instruct",
)
DEFAULT_EVALUATION_MODELS = (
    "Qwen/Qwen3-4B-Instruct-2507",
    "allenai/Olmo-3-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    # "mistralai/Ministral-3-14B-Instruct-2512",
)
DEFAULT_LOCAL_EVALUATION_MODELS = DEFAULT_EVALUATION_MODELS
DEFAULT_PROCESSED_DATASET = PROCESSED_DATASETS_DIR / "unified_processed_v3"
DEFAULT_INSPECT_RESULTS_DIR = RESULTS_DIR / "inspect"
DEFAULT_GENERATION_LOG_ROOT = DEFAULT_INSPECT_RESULTS_DIR / "generation"
DEFAULT_EVALUATION_LOG_ROOT = DEFAULT_INSPECT_RESULTS_DIR / "evaluation"
DEFAULT_AUGMENTED_CACHE_ROOT = DATASETS_DIR / "augmented"
DEFAULT_EVALUATED_DATASET_ROOT = DATASETS_DIR / "evaluated"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
GENERATION_RETRY_LIMIT = 3
AUGMENTED_STORE_MANIFEST = "augmented_manifest.json"
AUGMENTED_STORE_SCHEMA_VERSION = "augmented_mcqa_setting_records_v2"
EVALUATED_STORE_MANIFEST = "evaluated_manifest.json"
EVALUATED_STORE_SCHEMA_VERSION = "evaluated_mcqa_setting_mode_records_v1"
