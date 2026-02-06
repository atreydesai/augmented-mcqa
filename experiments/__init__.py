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
]
