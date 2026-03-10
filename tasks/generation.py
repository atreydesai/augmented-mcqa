from __future__ import annotations

from inspect_ai import Task

from data.final5_store import build_generation_dataset
from scorers.generation import final5_generation_scorer
from solvers.final5_generation import final5_generation_solver
from utils.constants import FINAL5_SETTINGS


def build_generation_tasks(
    *,
    processed_dataset_path,
    dataset_types,
    shard_count,
    shard_index,
    shard_strategy,
    limit,
    run_name,
    generation_model,
) -> list[Task]:
    dataset = build_generation_dataset(
        processed_dataset_path,
        dataset_types=dataset_types,
        limit=limit,
        shard_count=shard_count,
        shard_index=shard_index,
        shard_strategy=shard_strategy,
    )
    if len(dataset) == 0:
        return []
    return [
        Task(
            name="final5_generate",
            dataset=dataset,
            solver=final5_generation_solver(),
            scorer=final5_generation_scorer(),
            metadata={
                "kind": "generation",
                "run_name": run_name,
                "generation_model": generation_model,
                "dataset_types": list(dataset_types),
                "settings": list(FINAL5_SETTINGS),
                "shard_count": shard_count,
                "shard_index": shard_index,
                "shard_strategy": shard_strategy,
            },
        )
    ]
