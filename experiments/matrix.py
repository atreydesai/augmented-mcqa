"""Final5 matrix utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from config import RESULTS_DIR
from .config import ExperimentConfig
from .defaults import (
    DEFAULT_EVAL_MAX_TOKENS,
    DEFAULT_EVAL_MODE,
    DEFAULT_EVAL_SAVE_INTERVAL,
    DEFAULT_EVAL_SEED,
    DEFAULT_EVAL_TEMPERATURE,
)


MatrixPreset = Literal["final5"]

ALL_DATASET_TYPES = ["arc_challenge", "mmlu_pro", "gpqa"]
DISTRACTOR_SOURCE_MAP: dict[str, str] = {}
MATRIX_PRESETS: dict[MatrixPreset, list[str]] = {
    "final5": [
        "human_from_scratch",
        "model_from_scratch",
        "augment_human",
        "augment_model",
        "augment_ablation",
    ]
}

SETTING_SPECS: dict[str, dict[str, Any]] = {
    "human_from_scratch": {
        "num_human": 3,
        "num_model": 0,
    },
    "model_from_scratch": {
        "num_human": 0,
        "num_model": 3,
    },
    "augment_human": {
        "num_human": 3,
        "num_model": 6,
    },
    "augment_model": {
        "num_human": 0,
        "num_model": 9,
    },
    "augment_ablation": {
        "num_human": 0,
        "num_model": 9,
    },
}


def get_preset_setting_ids(preset: MatrixPreset) -> list[str]:
    if preset not in MATRIX_PRESETS:
        valid = ", ".join(sorted(MATRIX_PRESETS.keys()))
        raise ValueError(f"Unknown preset: {preset}. Valid presets: {valid}")
    return list(MATRIX_PRESETS[preset])


def build_matrix_configs(
    model: str,
    dataset_path: Path,
    generator_dataset_label: str,
    dataset_types: list[str],
    distractor_sources: list[str] | None = None,
    preset: MatrixPreset = cast(MatrixPreset, "final5"),
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
    entry_shards: int = 1,
    entry_shard_index: int = 0,
    entry_shard_strategy: str = "contiguous",
) -> list[ExperimentConfig]:
    del distractor_sources

    generator_label = str(generator_dataset_label).strip()
    if not generator_label:
        raise ValueError("generator_dataset_label is required and cannot be blank")

    unknown_datasets = sorted(set(dataset_types) - set(ALL_DATASET_TYPES))
    if unknown_datasets:
        valid = ", ".join(ALL_DATASET_TYPES)
        raise ValueError(f"Unknown dataset types: {unknown_datasets}. Valid dataset types: {valid}")

    if output_base is None:
        output_base = RESULTS_DIR

    mode_name = "choices_only" if choices_only else "full_question"
    model_safe = model.replace("/", "_")

    configs: list[ExperimentConfig] = []
    for dataset_type in dataset_types:
        for setting_id in get_preset_setting_ids(preset):
            spec = SETTING_SPECS[setting_id]
            output_dir = (
                Path(output_base)
                / generator_label
                / model_safe
                / mode_name
                / dataset_type
                / setting_id
            )
            name = f"{generator_label}_{model_safe}_{mode_name}_{dataset_type}_{setting_id}"

            cfg = ExperimentConfig(
                name=name,
                dataset_path=Path(dataset_path),
                model_name=model,
                generator_dataset_label=generator_label,
                setting_id=setting_id,
                num_human=int(spec["num_human"]),
                num_model=int(spec["num_model"]),
                eval_mode=eval_mode,
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
                distractor_source=setting_id,
                sampling_strategy="independent",
                branching_mode="shuffled_prefix",
                entry_shards=entry_shards,
                entry_shard_index=entry_shard_index,
                entry_shard_strategy=entry_shard_strategy,
            )
            configs.append(cfg)

    return configs


def sort_configs_for_sharding(configs: list[ExperimentConfig]) -> list[ExperimentConfig]:
    return sorted(configs, key=lambda cfg: cfg.config_id)


def select_shard(configs: list[ExperimentConfig], num_shards: int, shard_index: int) -> list[ExperimentConfig]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be > 0, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards - 1}], got {shard_index}")
    ordered = sort_configs_for_sharding(configs)
    return [cfg for idx, cfg in enumerate(ordered) if idx % num_shards == shard_index]


def maybe_select_shard(
    configs: list[ExperimentConfig],
    num_shards: int | None,
    shard_index: int | None,
) -> list[ExperimentConfig]:
    if num_shards is None and shard_index is None:
        return sort_configs_for_sharding(configs)
    if num_shards is None or shard_index is None:
        raise ValueError("Both num_shards and shard_index must be provided together")
    return select_shard(configs, num_shards, shard_index)


def summarize_configs(configs: list[ExperimentConfig]) -> dict[str, Any]:
    by_dataset: dict[str, int] = {}
    by_setting: dict[str, int] = {}
    by_mode: dict[str, int] = {}

    for cfg in configs:
        dataset_type = cfg.dataset_type_filter or "unknown"
        setting_id = cfg.setting_id
        mode_name = "choices_only" if cfg.choices_only else "full_question"
        by_dataset[dataset_type] = by_dataset.get(dataset_type, 0) + 1
        by_setting[setting_id] = by_setting.get(setting_id, 0) + 1
        by_mode[mode_name] = by_mode.get(mode_name, 0) + 1

    return {
        "total": len(configs),
        "by_dataset_type": dict(sorted(by_dataset.items())),
        "by_setting": dict(sorted(by_setting.items())),
        "by_mode": dict(sorted(by_mode.items())),
    }


def build_manifest(
    configs: list[ExperimentConfig],
    *,
    preset: MatrixPreset,
    model: str,
    dataset_path: Path,
    generator_dataset_label: str,
    dataset_types: list[str],
    distractor_sources: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del distractor_sources

    generator_label = str(generator_dataset_label).strip()
    if not generator_label:
        raise ValueError("generator_dataset_label is required and cannot be blank")

    return {
        "manifest_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preset": preset,
        "model": model,
        "dataset_path": str(dataset_path),
        "generator_dataset_label": generator_label,
        "dataset_types": list(dataset_types),
        "summary": summarize_configs(configs),
        "metadata": metadata or {},
        "configs": [cfg.to_dict() for cfg in sort_configs_for_sharding(configs)],
    }


def save_manifest(manifest: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_configs_from_manifest(path: Path) -> list[ExperimentConfig]:
    manifest = load_manifest(path)
    raw = manifest.get("configs", [])
    if not isinstance(raw, list):
        raise ValueError(f"Invalid manifest at {path}: 'configs' must be a list")
    return [ExperimentConfig.from_dict(c) for c in raw]
