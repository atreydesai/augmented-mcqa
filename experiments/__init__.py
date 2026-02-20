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
from .defaults import (
    DEFAULT_MATRIX_PRESET,
    DEFAULT_EVAL_MODE,
    DEFAULT_EVAL_SEED,
    DEFAULT_EVAL_TEMPERATURE,
    DEFAULT_EVAL_MAX_TOKENS,
    DEFAULT_EVAL_SAVE_INTERVAL,
    DEFAULT_EVAL_KEEP_CHECKPOINTS,
    DEFAULT_GENERATOR_DATASET_LABEL,
    DEFAULT_NUM_HUMAN_DISTRACTORS,
    DEFAULT_NUM_MODEL_DISTRACTORS,
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
    # Shared defaults
    "DEFAULT_MATRIX_PRESET",
    "DEFAULT_EVAL_MODE",
    "DEFAULT_EVAL_SEED",
    "DEFAULT_EVAL_TEMPERATURE",
    "DEFAULT_EVAL_MAX_TOKENS",
    "DEFAULT_EVAL_SAVE_INTERVAL",
    "DEFAULT_EVAL_KEEP_CHECKPOINTS",
    "DEFAULT_GENERATOR_DATASET_LABEL",
    "DEFAULT_NUM_HUMAN_DISTRACTORS",
    "DEFAULT_NUM_MODEL_DISTRACTORS",
]
