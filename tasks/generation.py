from __future__ import annotations

from inspect_ai import Task

from data.final5_store import build_generation_dataset
from scorers.generation import final5_generation_scorer
from solvers.final5_generation import final5_generation_solver
from utils.constants import FINAL5_SETTINGS


def build_generation_tasks(
    *,
    processed_dataset_path,
    strategies,
    dataset_types,
    question_start,
    shard_count,
    shard_index,
    shard_strategy,
    limit,
    run_name,
    generation_model,
    generation_log_dir=None,
    task_metadata_by_strategy=None,
) -> list[Task]:
    tasks: list[Task] = []
    task_metadata_by_strategy = task_metadata_by_strategy or {}
    for strategy in strategies:
        dataset = build_generation_dataset(
            processed_dataset_path,
            strategy=strategy,
            dataset_types=dataset_types,
            question_start=question_start,
            limit=limit,
            generation_log_dir=generation_log_dir,
            shard_count=shard_count,
            shard_index=shard_index,
            shard_strategy=shard_strategy,
        )
        if len(dataset) == 0:
            continue
        task_metadata = {
            "kind": "generation",
            "run_name": run_name,
            "generation_model": generation_model,
            "dataset_types": list(dataset_types),
            "settings": list(FINAL5_SETTINGS),
            "generation_strategy": strategy,
            "question_start": question_start,
            "limit": limit,
            "shard_count": shard_count,
            "shard_index": shard_index,
            "shard_strategy": shard_strategy,
        }
        task_metadata.update(dict(task_metadata_by_strategy.get(strategy, {}) or {}))
        tasks.append(
            Task(
                name=f"final5_generate_{strategy}",
                dataset=dataset,
                solver=final5_generation_solver(strategy),
                scorer=final5_generation_scorer(),
                metadata=task_metadata,
            )
        )
    return tasks
