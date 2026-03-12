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
    question_start=0,
    shard_count,
    shard_index,
    shard_strategy,
    limit,
    run_name,
    generation_run_name,
    generation_model,
    evaluation_model,
    task_metadata_by_setting_mode=None,
) -> list[Task]:
    tasks: list[Task] = []
    task_metadata_by_setting_mode = task_metadata_by_setting_mode or {}
    for setting in settings:
        for mode in modes:
            dataset = build_evaluation_dataset(
                augmented_dataset_path,
                setting=setting,
                mode=mode,
                dataset_types=dataset_types,
                question_start=question_start,
                limit=limit,
                shard_count=shard_count,
                shard_index=shard_index,
                shard_strategy=shard_strategy,
            )
            if len(dataset) == 0:
                continue
            task_metadata = {
                "kind": "evaluation",
                "run_name": run_name,
                "generation_run_name": generation_run_name,
                "generation_model": generation_model,
                "evaluation_model": evaluation_model,
                "setting": setting,
                "mode": mode,
                "dataset_types": list(dataset_types),
                "question_start": question_start,
                "limit": limit,
                "shard_count": shard_count,
                "shard_index": shard_index,
                "shard_strategy": shard_strategy,
            }
            task_metadata.update(dict(task_metadata_by_setting_mode.get((setting, mode), {}) or {}))
            tasks.append(
                Task(
                    name=f"final5_eval_{setting}_{mode}",
                    dataset=dataset,
                    solver=final5_evaluation_solver(mode),
                    scorer=final5_evaluation_scorer(),
                    metadata=task_metadata,
                )
            )
    return tasks
