"""Shared defaults for evaluation workflows.

Centralizing these values prevents drift across config dataclasses, matrix
builders, and CLI entrypoints.
"""

from __future__ import annotations

import os
from typing import List, Optional

from config import RANDOM_SEED


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _get_optional_float_env(name: str, default: Optional[float]) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


DEFAULT_MATRIX_PRESET = os.getenv("AUGMCQA_DEFAULT_MATRIX_PRESET", "final5")
DEFAULT_EVAL_MODE = os.getenv("AUGMCQA_DEFAULT_EVAL_MODE", "behavioral")
DEFAULT_EVAL_SEED = _get_int_env("AUGMCQA_DEFAULT_EVAL_SEED", RANDOM_SEED)
DEFAULT_EVAL_TEMPERATURE = _get_optional_float_env("AUGMCQA_DEFAULT_EVAL_TEMPERATURE", None)
DEFAULT_EVAL_MAX_TOKENS = _get_int_env("AUGMCQA_DEFAULT_EVAL_MAX_TOKENS", 100)

# Stop sequences: halt generation when the model starts a new question block or
# a repetitive pattern.  These apply to local (vLLM) inference only.
_stop_env = os.getenv("AUGMCQA_DEFAULT_EVAL_STOP", "")
DEFAULT_EVAL_STOP: List[str] = (
    [s for s in _stop_env.split("|||") if s]
    if _stop_env.strip()
    else [
        "\n\nQuestion:",
        "\n\nThe following",
        "\n\nAnswer:",
    ]
)
DEFAULT_EVAL_SAVE_INTERVAL = _get_int_env("AUGMCQA_DEFAULT_EVAL_SAVE_INTERVAL", 50)
DEFAULT_EVAL_KEEP_CHECKPOINTS = _get_int_env("AUGMCQA_DEFAULT_EVAL_KEEP_CHECKPOINTS", 2)
DEFAULT_GENERATOR_DATASET_LABEL = os.getenv("AUGMCQA_DEFAULT_GENERATOR_LABEL", "manual")
DEFAULT_NUM_HUMAN_DISTRACTORS = _get_int_env("AUGMCQA_DEFAULT_NUM_HUMAN_DISTRACTORS", 3)
DEFAULT_NUM_MODEL_DISTRACTORS = _get_int_env("AUGMCQA_DEFAULT_NUM_MODEL_DISTRACTORS", 0)


if DEFAULT_NUM_HUMAN_DISTRACTORS < 0:
    raise ValueError("AUGMCQA_DEFAULT_NUM_HUMAN_DISTRACTORS must be >= 0")
if DEFAULT_NUM_MODEL_DISTRACTORS < 0:
    raise ValueError("AUGMCQA_DEFAULT_NUM_MODEL_DISTRACTORS must be >= 0")
if DEFAULT_EVAL_MAX_TOKENS <= 0:
    raise ValueError("AUGMCQA_DEFAULT_EVAL_MAX_TOKENS must be > 0")
if DEFAULT_EVAL_SAVE_INTERVAL <= 0:
    raise ValueError("AUGMCQA_DEFAULT_EVAL_SAVE_INTERVAL must be > 0")
if DEFAULT_EVAL_KEEP_CHECKPOINTS < 0:
    raise ValueError("AUGMCQA_DEFAULT_EVAL_KEEP_CHECKPOINTS must be >= 0")
