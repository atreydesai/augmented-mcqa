"""Data processing module for Augmented MCQA."""

from .downloader import (
    download_dataset,
    download_mmlu_pro,
    download_mmlu_all_configs,
    download_arc,
    download_supergpqa,
    get_dataset_info,
    print_dataset_info,
)

from .mmlu_pro_processor import (
    process_mmlu_pro,
    sort_distractors,
    build_mmlu_lookup,
    clean_whitespace,
    clean_options,
    verify_sorting,
    WHITESPACE_BUG_CATEGORIES,
)

from .augmentor import (
    AugmentorMode,
    GenerationConfig,
    augment_dataset,
    build_prompt,
    parse_generated_distractors,
)

from .filter import (
    FilterConfig,
    filter_dataset,
    create_standard_subsets,
    shuffle_options_deterministic,
    get_answer_letter,
    CHOICE_LABELS,
)

from .arc_processor import (
    load_arc_dataset,
    process_arc_for_experiments,
    add_synthetic_distractors_to_arc,
    get_arc_stats,
)

from .supergpqa_processor import (
    load_supergpqa_dataset,
    process_supergpqa_for_experiments,
    add_synthetic_distractors_to_supergpqa,
    get_supergpqa_stats,
    filter_by_difficulty,
    filter_by_discipline,
)

from .hub_utils import (
    push_dataset_to_hub,
)

__all__ = [
    # Downloader
    "download_dataset",
    "download_mmlu_pro",
    "download_mmlu_all_configs",
    "download_arc",
    "download_supergpqa",
    "get_dataset_info",
    "print_dataset_info",
    # MMLU-Pro Processor
    "process_mmlu_pro",
    "sort_distractors",
    "build_mmlu_lookup",
    "clean_whitespace",
    "clean_options",
    "verify_sorting",
    "WHITESPACE_BUG_CATEGORIES",
    # Augmentor
    "AugmentorMode",
    "GenerationConfig",
    "augment_dataset",
    "build_prompt",
    "parse_generated_distractors",
    # Filter
    "FilterConfig",
    "filter_dataset",
    "create_standard_subsets",
    "shuffle_options_deterministic",
    "get_answer_letter",
    "CHOICE_LABELS",
    # ARC Processor
    "load_arc_dataset",
    "process_arc_for_experiments",
    "add_synthetic_distractors_to_arc",
    "get_arc_stats",
    # SuperGPQA Processor
    "load_supergpqa_dataset",
    "process_supergpqa_for_experiments",
    "add_synthetic_distractors_to_supergpqa",
    "get_supergpqa_stats",
    "filter_by_difficulty",
    "filter_by_discipline",
    # Hub Utils
    "push_dataset_to_hub",
]
