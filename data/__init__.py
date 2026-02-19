"""Data processing module for Augmented MCQA.

Symbols are lazily imported to avoid loading optional dependencies at package
import time.
"""

from importlib import import_module


_SYMBOL_TO_MODULE = {
    # Downloader
    "download_dataset": "downloader",
    "download_mmlu_pro": "downloader",
    "download_mmlu_all_configs": "downloader",
    "download_arc": "downloader",
    "download_supergpqa": "downloader",
    "get_dataset_info": "downloader",
    "print_dataset_info": "downloader",
    # MMLU-Pro Processor
    "process_mmlu_pro": "mmlu_pro_processor",
    "sort_distractors": "mmlu_pro_processor",
    "build_mmlu_lookup": "mmlu_pro_processor",
    "clean_whitespace": "mmlu_pro_processor",
    "clean_options": "mmlu_pro_processor",
    "verify_sorting": "mmlu_pro_processor",
    "WHITESPACE_BUG_CATEGORIES": "mmlu_pro_processor",
    # Augmentor
    "AugmentorMode": "augmentor",
    "GenerationConfig": "augmentor",
    "augment_dataset": "augmentor",
    "build_prompt": "augmentor",
    "parse_generated_distractors": "augmentor",
    # Filter
    "FilterConfig": "filter",
    "filter_dataset": "filter",
    "create_standard_subsets": "filter",
    "shuffle_options_deterministic": "filter",
    "get_answer_letter": "filter",
    "CHOICE_LABELS": "filter",
    # ARC Processor
    "load_arc_dataset": "arc_processor",
    "process_arc_for_experiments": "arc_processor",
    "add_synthetic_distractors_to_arc": "arc_processor",
    "get_arc_stats": "arc_processor",
    # SuperGPQA Processor
    "load_supergpqa_dataset": "supergpqa_processor",
    "process_supergpqa_for_experiments": "supergpqa_processor",
    "add_synthetic_distractors_to_supergpqa": "supergpqa_processor",
    "get_supergpqa_stats": "supergpqa_processor",
    "filter_by_difficulty": "supergpqa_processor",
    "filter_by_discipline": "supergpqa_processor",
    # Hub Utils
    "push_dataset_to_hub": "hub_utils",
}

__all__ = list(_SYMBOL_TO_MODULE.keys())


def __getattr__(name: str):
    if name not in _SYMBOL_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name = _SYMBOL_TO_MODULE[name]
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)

    # Cache resolved symbol for subsequent lookups.
    globals()[name] = value
    return value
