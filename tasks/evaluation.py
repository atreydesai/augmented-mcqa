from __future__ import annotations

from inspect_ai import Task

from data.final5_store import build_evaluation_dataset
from scorers.evaluation import final5_evaluation_scorer
from solvers.final5_evaluation import final5_evaluation_solver


def build_evaluation_tasks(
    *,
    augmented_dataset_path,
    dataset_types,
    settings,
    modes,
    shard_count,
    shard_index,
    shard_strategy,
    limit,
    run_name,
    generation_run_name,
    generation_model,
    evaluation_model,
) -> list[Task]:
    tasks: list[Task] = []
    for setting in settings:
        for mode in modes:
            dataset = build_evaluation_dataset(
                augmented_dataset_path,
                setting=setting,
                mode=mode,
                dataset_types=dataset_types,
                limit=limit,
                shard_count=shard_count,
                shard_index=shard_index,
                shard_strategy=shard_strategy,
            )
            if len(dataset) == 0:
                continue
            tasks.append(
                Task(
                    name=f"final5_eval_{setting}_{mode}",
                    dataset=dataset,
                    solver=final5_evaluation_solver(mode),
                    scorer=final5_evaluation_scorer(),
                    metadata={
                        "kind": "evaluation",
                        "run_name": run_name,
                        "generation_run_name": generation_run_name,
                        "generation_model": generation_model,
                        "evaluation_model": evaluation_model,
                        "setting": setting,
                        "mode": mode,
                        "dataset_types": list(dataset_types),
                        "shard_count": shard_count,
                        "shard_index": shard_index,
                        "shard_strategy": shard_strategy,
                    },
                )
            )
    return tasks
