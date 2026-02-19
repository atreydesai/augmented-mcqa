"""Experiments module for Augmented MCQA."""

from .config import (
    ExperimentConfig,
    create_batch_configs,
    save_batch_configs,
    load_batch_configs,
)

from .runner import (
    EvalResult,
    ExperimentResults,
    ExperimentRunner,
    run_experiment,
    run_batch,
    build_mcqa_prompt,
    determine_prediction_type,
    CHOICE_LABELS,
)

from .matrix import (
    MatrixPreset,
    ALL_DATASET_TYPES,
    DISTRACTOR_SOURCE_MAP,
    MATRIX_PRESETS,
    get_preset_distractor_configs,
    build_matrix_configs,
    sort_configs_for_sharding,
    select_shard,
    maybe_select_shard,
    summarize_configs,
    build_manifest,
    save_manifest,
    load_manifest,
    load_configs_from_manifest,
)


__all__ = [
    # Config
    "ExperimentConfig",
    "create_batch_configs",
    "save_batch_configs",
    "load_batch_configs",
    # Runner
    "EvalResult",
    "ExperimentResults",
    "ExperimentRunner",
    "run_experiment",
    "run_batch",
    "build_mcqa_prompt",
    "determine_prediction_type",
    "CHOICE_LABELS",
    # Matrix
    "MatrixPreset",
    "ALL_DATASET_TYPES",
    "DISTRACTOR_SOURCE_MAP",
    "MATRIX_PRESETS",
    "get_preset_distractor_configs",
    "build_matrix_configs",
    "sort_configs_for_sharding",
    "select_shard",
    "maybe_select_shard",
    "summarize_configs",
    "build_manifest",
    "save_manifest",
    "load_manifest",
    "load_configs_from_manifest",
]
