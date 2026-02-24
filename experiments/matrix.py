"""Matrix utilities for deterministic evaluation config generation and sharding."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from config import DistractorType, RESULTS_DIR
from .config import ExperimentConfig
from .defaults import (
    DEFAULT_EVAL_MAX_TOKENS,
    DEFAULT_EVAL_MODE,
    DEFAULT_EVAL_SAVE_INTERVAL,
    DEFAULT_EVAL_SEED,
    DEFAULT_EVAL_TEMPERATURE,
    DEFAULT_MATRIX_PRESET,
)


MatrixPreset = Literal["core16", "branching21"]

ALL_DATASET_TYPES = ["mmlu_pro", "gpqa", "arc_easy", "arc_challenge"]

DISTRACTOR_SOURCE_MAP = {
    "scratch": DistractorType.COND_MODEL_Q_A_SCRATCH,
    "dhuman": DistractorType.COND_MODEL_Q_A_DHUMAN,
    "dmodel": DistractorType.COND_MODEL_Q_A_DMODEL,
}


def _dedupe_ordered(items: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    deduped: list[tuple[int, int]] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


CORE16_DISTRACTOR_CONFIGS = _dedupe_ordered(
    [(3, m) for m in range(0, 7)]
    + [(h, 0) for h in range(1, 4)]
    + [(0, m) for m in range(1, 7)]
)

# Branching layout (21 configs total):
# - 0H + (1..6)M
# - 1H + (0..5)M
# - 2H + (0..4)M
# - 3H + (0..3)M
BRANCHING21_DISTRACTOR_CONFIGS = (
    [(0, m) for m in range(1, 7)]
    + [(1, m) for m in range(0, 6)]
    + [(2, m) for m in range(0, 5)]
    + [(3, m) for m in range(0, 4)]
)

MATRIX_PRESETS: dict[MatrixPreset, list[tuple[int, int]]] = {
    "core16": CORE16_DISTRACTOR_CONFIGS,
    "branching21": BRANCHING21_DISTRACTOR_CONFIGS,
}


def get_preset_distractor_configs(preset: MatrixPreset) -> list[tuple[int, int]]:
    if preset not in MATRIX_PRESETS:
        valid = ", ".join(sorted(MATRIX_PRESETS.keys()))
        raise ValueError(f"Unknown preset: {preset}. Valid presets: {valid}")
    return list(MATRIX_PRESETS[preset])


def build_matrix_configs(
    model: str,
    dataset_path: Path,
    generator_dataset_label: str,
    dataset_types: list[str],
    distractor_sources: list[str],
    preset: MatrixPreset = cast(MatrixPreset, DEFAULT_MATRIX_PRESET),
    output_base: Path | None = None,
    limit: int | None = None,
    eval_mode: str = DEFAULT_EVAL_MODE,
    choices_only: bool = False,
    seed: int = DEFAULT_EVAL_SEED,
    reasoning_effort: str | None = None,
    thinking_level: str | None = None,
    temperature: float | None = DEFAULT_EVAL_TEMPERATURE,
    max_tokens: int = DEFAULT_EVAL_MAX_TOKENS,
    save_interval: int = DEFAULT_EVAL_SAVE_INTERVAL,
) -> list[ExperimentConfig]:
    """Build a deterministic experiment matrix."""
    generator_label = str(generator_dataset_label).strip()
    if not generator_label:
        raise ValueError("generator_dataset_label is required and cannot be blank")

    if output_base is None:
        output_base = RESULTS_DIR

    unknown_datasets = sorted(set(dataset_types) - set(ALL_DATASET_TYPES))
    if unknown_datasets:
        valid = ", ".join(ALL_DATASET_TYPES)
        raise ValueError(f"Unknown dataset types: {unknown_datasets}. Valid dataset types: {valid}")

    unknown_sources = sorted(set(distractor_sources) - set(DISTRACTOR_SOURCE_MAP.keys()))
    if unknown_sources:
        valid = ", ".join(sorted(DISTRACTOR_SOURCE_MAP.keys()))
        raise ValueError(f"Unknown distractor sources: {unknown_sources}. Valid distractor sources: {valid}")

    configs: list[ExperimentConfig] = []
    model_safe = model.replace("/", "_")
    sampling_strategy = "branching_cumulative" if preset == "branching21" else "independent"
    branching_mode = "human_prefix" if preset == "branching21" else "shuffled_prefix"
    label_root = output_base if output_base.name == generator_label else output_base / generator_label

    for dataset_type in dataset_types:
        human_only_emitted: set[int] = set()
        for source_name in distractor_sources:
            distractor_type = DISTRACTOR_SOURCE_MAP[source_name]

            for num_human, num_model in get_preset_distractor_configs(preset):
                # Configs with num_model=0 are identical across all distractor
                # sources: no model distractors are selected regardless of
                # generation strategy.  Emit only once per (dataset_type, nH).
                if num_model == 0:
                    if num_human in human_only_emitted:
                        continue
                    human_only_emitted.add(num_human)
                config_str = f"{num_human}H{num_model}M"
                name = f"{generator_label}_{model_safe}_{dataset_type}_{source_name}_{config_str}"
                output_dir = (
                    label_root
                    / f"{model_safe}_{dataset_type}_{source_name}"
                    / config_str
                )

                config = ExperimentConfig(
                    name=name,
                    dataset_path=dataset_path,
                    model_name=model,
                    generator_dataset_label=generator_label,
                    num_human=num_human,
                    num_model=num_model,
                    model_distractor_type=distractor_type,
                    eval_mode=eval_mode,
                    sampling_strategy=sampling_strategy,
                    branching_mode=branching_mode,
                    choices_only=choices_only,
                    limit=limit,
                    seed=seed,
                    reasoning_effort=reasoning_effort,
                    thinking_level=thinking_level,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    save_interval=save_interval,
                    output_dir=output_dir,
                    dataset_type_filter=dataset_type,
                    distractor_source=source_name,
                )
                configs.append(config)

    return configs


def sort_configs_for_sharding(configs: list[ExperimentConfig]) -> list[ExperimentConfig]:
    """Sort by config_id for deterministic shard assignment and execution order."""
    return sorted(configs, key=lambda cfg: cfg.config_id)


def select_shard(
    configs: list[ExperimentConfig],
    num_shards: int,
    shard_index: int,
) -> list[ExperimentConfig]:
    """Deterministic round-robin sharding over sorted config IDs."""
    if num_shards <= 0:
        raise ValueError(f"num_shards must be > 0, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards - 1}], got {shard_index}"
        )

    ordered = sort_configs_for_sharding(configs)
    return [cfg for idx, cfg in enumerate(ordered) if idx % num_shards == shard_index]


def maybe_select_shard(
    configs: list[ExperimentConfig],
    num_shards: int | None,
    shard_index: int | None,
) -> list[ExperimentConfig]:
    """Optionally apply shard filtering; always returns deterministically ordered configs."""
    if num_shards is None and shard_index is None:
        return sort_configs_for_sharding(configs)
    if num_shards is None or shard_index is None:
        raise ValueError("Both num_shards and shard_index must be provided together")
    return select_shard(configs, num_shards, shard_index)


def summarize_configs(configs: list[ExperimentConfig]) -> dict[str, Any]:
    """Build a compact summary for logs/manifests."""
    by_dataset: dict[str, int] = {}
    by_source: dict[str, int] = {}

    for cfg in configs:
        dataset_type = cfg.dataset_type_filter or "unknown"
        source = cfg.distractor_source or "unknown"
        by_dataset[dataset_type] = by_dataset.get(dataset_type, 0) + 1
        by_source[source] = by_source.get(source, 0) + 1

    return {
        "total": len(configs),
        "by_dataset_type": dict(sorted(by_dataset.items())),
        "by_distractor_source": dict(sorted(by_source.items())),
    }


def build_manifest(
    configs: list[ExperimentConfig],
    *,
    preset: MatrixPreset,
    model: str,
    dataset_path: Path,
    generator_dataset_label: str,
    dataset_types: list[str],
    distractor_sources: list[str],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a JSON-serializable manifest for reproducible reruns."""
    generator_label = str(generator_dataset_label).strip()
    if not generator_label:
        raise ValueError("generator_dataset_label is required and cannot be blank")
    config_labels = {
        str(cfg.generator_dataset_label).strip() for cfg in configs if str(cfg.generator_dataset_label).strip()
    }
    if not config_labels:
        raise ValueError("Configs are missing generator_dataset_label")
    if len(config_labels) > 1:
        raise ValueError(f"Mixed generator_dataset_label values in configs: {sorted(config_labels)}")
    config_label = next(iter(config_labels))
    if config_label != generator_label:
        raise ValueError(
            f"generator_dataset_label mismatch: configs='{config_label}' manifest='{generator_label}'"
        )
    return {
        "manifest_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preset": preset,
        "model": model,
        "dataset_path": str(dataset_path),
        "generator_dataset_label": generator_label,
        "dataset_types": list(dataset_types),
        "distractor_sources": list(distractor_sources),
        "summary": summarize_configs(configs),
        "metadata": metadata or {},
        "configs": [cfg.to_dict() for cfg in sort_configs_for_sharding(configs)],
    }


def save_manifest(manifest: dict[str, Any], path: Path) -> Path:
    """Write manifest JSON to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a manifest JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def load_configs_from_manifest(path: Path) -> list[ExperimentConfig]:
    """Load configs from a saved manifest file."""
    manifest = load_manifest(path)
    raw_configs = manifest.get("configs", [])
    if not isinstance(raw_configs, list):
        raise ValueError(f"Invalid manifest at {path}: 'configs' must be a list")
    return [ExperimentConfig.from_dict(c) for c in raw_configs]
