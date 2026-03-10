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
    "download_gpqa": "downloader",
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
    # ARC Processor
    "load_arc_dataset": "arc_processor",
    "process_arc_for_experiments": "arc_processor",
    "get_arc_stats": "arc_processor",
    # GPQA Processor
    "load_gpqa_dataset": "gpqa_processor",
    "process_gpqa_for_experiments": "gpqa_processor",
    "get_gpqa_stats": "gpqa_processor",
    # Benchmarker export
    "export_benchmarker_items": "benchmarker_export",
    # Hub Utils
    "push_dataset_to_hub": "hub_utils",
    # Inspect-native Final5 store
    "build_generation_dataset": "final5_store",
    "build_evaluation_dataset": "final5_store",
    "ensure_augmented_dataset": "final5_store",
    "materialize_augmented_dataset": "final5_store",
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
