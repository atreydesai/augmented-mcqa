"""Matrix utilities for deterministic evaluation config generation and sharding."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from config import DistractorType, RESULTS_DIR
from .config import ExperimentConfig


MatrixPreset = Literal["core16", "branching21"]

ALL_DATASET_TYPES = ["mmlu_pro", "supergpqa", "arc_easy", "arc_challenge"]

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

BRANCHING21_DISTRACTOR_CONFIGS = [(h, m) for h in range(1, 4) for m in range(0, 7)]

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
    dataset_types: list[str],
    distractor_sources: list[str],
    preset: MatrixPreset = "core16",
    output_base: Path | None = None,
    limit: int | None = None,
    eval_mode: str = "behavioral",
    choices_only: bool = False,
    seed: int = 42,
    reasoning_effort: str | None = None,
    thinking_level: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 100,
) -> list[ExperimentConfig]:
    """Build a deterministic experiment matrix."""
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

    for dataset_type in dataset_types:
        for source_name in distractor_sources:
            distractor_type = DISTRACTOR_SOURCE_MAP[source_name]

            for num_human, num_model in get_preset_distractor_configs(preset):
                config_str = f"{num_human}H{num_model}M"
                name = f"{model_safe}_{dataset_type}_{source_name}_{config_str}"
                output_dir = output_base / f"{model_safe}_{dataset_type}_{source_name}" / config_str

                config = ExperimentConfig(
                    name=name,
                    dataset_path=dataset_path,
                    model_name=model,
                    num_human=num_human,
                    num_model=num_model,
                    model_distractor_type=distractor_type,
                    eval_mode=eval_mode,
                    sampling_strategy=sampling_strategy,
                    choices_only=choices_only,
                    limit=limit,
                    seed=seed,
                    reasoning_effort=reasoning_effort,
                    thinking_level=thinking_level,
                    temperature=temperature,
                    max_tokens=max_tokens,
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
    dataset_types: list[str],
    distractor_sources: list[str],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a JSON-serializable manifest for reproducible reruns."""
    return {
        "manifest_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preset": preset,
        "model": model,
        "dataset_path": str(dataset_path),
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
