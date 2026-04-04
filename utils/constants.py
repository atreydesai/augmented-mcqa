from __future__ import annotations

from pathlib import Path

from config import CHOICE_LABELS, DATASETS_DIR, PROCESSED_DATASETS_DIR, RESULTS_DIR
from utils.recipes import load_setting_recipes, setting_specs

SETTING_NAMES = tuple(recipe.name for recipe in load_setting_recipes())
MODE_CHOICES = ("full_question", "choices_only")
SETTING_SPECS: dict[str, dict[str, int]] = setting_specs()
DEFAULT_LOCAL_GENERATION_MODELS = (
    "Qwen/Qwen3-4B-Instruct-2507",
    "allenai/Olmo-3-7B-Instruct",
)
DEFAULT_LOCAL_EVALUATION_MODELS = (
    "Qwen/Qwen3-4B-Instruct-2507",
    "allenai/Olmo-3-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
)
DEFAULT_PROCESSED_DATASET = PROCESSED_DATASETS_DIR / "unified_processed_v3"
DEFAULT_GENERATION_LOG_ROOT = RESULTS_DIR / "inspect" / "generation"
DEFAULT_EVALUATION_LOG_ROOT = RESULTS_DIR / "inspect" / "evaluation"
DEFAULT_AUGMENTED_CACHE_ROOT = DATASETS_DIR / "augmented"
DEFAULT_COLLECTED_DATASET_ROOT = DATASETS_DIR / "collected"
DEFAULT_SUPPORT_SET_ROOT = DATASETS_DIR / "support_sets"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
GENERATION_RETRY_LIMIT = 3
AUGMENTED_STORE_MANIFEST = "augmented_manifest.json"
AUGMENTED_STORE_SCHEMA_VERSION = "augmented_mcqa_setting_records_v2"
EVALUATED_STORE_MANIFEST = "evaluated_manifest.json"
EVALUATED_STORE_SCHEMA_VERSION = "evaluated_mcqa_setting_mode_records_v1"
SUPPORT_SET_MANIFEST = "support_manifest.json"
SUPPORT_SET_SCHEMA_VERSION = "augmented_mcqa_support_set_v1"
COLLECTED_STATE_FILENAME = "collected_state.json"
