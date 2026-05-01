from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cli.runtime import evaluation_logs_completed, generation_logs_succeeded, inspect_eval
from config import ACTIVE_DATASET_TYPES
from data.store import (
    _load_dataset_dict,
    build_evaluation_dataset,
    build_generation_support_manifest,
    build_generation_dataset,
    combined_support_ids,
    ensure_augmented_dataset,
    materialize_evaluated_datasets,
)
from utils.cluster_submit import (
    ClusterTask,
    FINALIZER_WRAPPER_TEMPLATE,
    WRAPPER_TEMPLATE,
    build_bundle_paths,
    render_manifest,
    render_submit_script,
    submit_bundle,
    write_bundle,
)
from utils.constants import (
    DEFAULT_AUGMENTED_CACHE_ROOT,
    DEFAULT_COLLECTED_DATASET_ROOT,
    DEFAULT_EVALUATION_LOG_ROOT,
    DEFAULT_GENERATION_LOG_ROOT,
    DEFAULT_LOCAL_EVALUATION_MODELS,
    DEFAULT_LOCAL_GENERATION_MODELS,
    DEFAULT_PROCESSED_DATASET,
    DEFAULT_SUPPORT_SET_ROOT,
    COLLECTED_STATE_FILENAME,
    EVALUATED_STORE_MANIFEST,
    SETTING_NAMES,
    MODE_CHOICES,
)
from utils.modeling import resolve_model_name, safe_name
from utils.scheduler_state import (
    EVALUATION_SETTING_DEPENDENCIES,
    GENERATION_STRATEGY_DEPENDENCIES,
    SCHEDULABLE_GENERATION_STRATEGIES,
    build_scheduler_state,
    build_scheduler_state_from_tasks,
    chunk_ranges,
    collect_slice_attempts,
    evaluation_slice_ref,
    generation_slice_ref,
    iso_now,
    load_scheduler_manifests,
    resource_class_for_model,
    task_slug,
)

prepare_data = None
DEFAULT_DATASET_TYPES = list(ACTIVE_DATASET_TYPES)


def _csv_list(raw: str | None, *, default: list[str]) -> list[str]:
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _select_values(
    raw: str | None,
    *,
    default: list[str],
    allowed: list[str],
    label: str,
) -> list[str]:
    values = _csv_list(raw, default=default)
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise ValueError(f"Unsupported {label}: " + ", ".join(invalid))
    return values


def _slice_task_metadata(
    *,
    stage: str,
    run_name: str,
    model: str,
    dataset_type: str,
    question_start: int,
    question_end: int,
    strategy: str | None = None,
    setting: str | None = None,
    mode: str | None = None,
) -> dict[str, object]:
    if stage == "generate":
        slice_ref = generation_slice_ref(
            run_name=run_name,
            model=model,
            dataset_type=dataset_type,
            strategy=str(strategy or ""),
            question_start=question_start,
            question_end=question_end,
        )
    else:
        slice_ref = evaluation_slice_ref(
            run_name=run_name,
            model=model,
            dataset_type=dataset_type,
            setting=str(setting or ""),
            mode=str(mode or ""),
            question_start=question_start,
            question_end=question_end,
        )
    return {
        "slice_ref": slice_ref,
        "question_end": question_end,
        "task_slug": task_slug(
            stage=stage,
            model=model,
            dataset_type=dataset_type,
            strategy=strategy,
            setting=setting,
            mode=mode,
            question_start=question_start,
            question_end=question_end,
        ),
    }


def _planned_task_entry(
    *,
    stage: str,
    run_name: str,
    model: str,
    dataset_type: str,
    question_start: int,
    question_end: int,
    task_slug_value: str,
    slice_ref: str,
    force: bool,
    dataset_types: list[str],
    state_dependency_refs: list[str],
    submit_dependency_refs: list[str],
    strategy: str | None = None,
    setting: str | None = None,
    mode: str | None = None,
    generation_run_name: str = "",
    generation_model: str = "",
) -> dict[str, object]:
    return {
        "slice_ref": slice_ref,
        "stage": stage,
        "run_name": run_name,
        "model": model,
        "dataset_type": dataset_type,
        "dataset_types": dataset_types,
        "strategy": strategy or "",
        "setting": setting or "",
        "mode": mode or "",
        "task_slug": task_slug_value,
        "question_start": question_start,
        "question_end": question_end,
        "state_dependency_refs": state_dependency_refs,
        "submit_dependency_refs": submit_dependency_refs,
        "force": force,
        "generation_run_name": generation_run_name,
        "generation_model": generation_model,
    }


def _run_model_dir(root: Path, run_name: str, model: str) -> Path:
    return root / safe_name(run_name) / safe_name(model)


def _generation_log_dir(root: Path, run_name: str, model: str) -> Path:
    return _run_model_dir(root, run_name, model)


def _augmented_cache_dir(root: Path, run_name: str, model: str) -> Path:
    return _run_model_dir(root, run_name, model)


def _support_manifest_dir(root: Path, run_name: str, model: str) -> Path:
    return _run_model_dir(root, run_name, model)


def _evaluation_log_dir(root: Path, run_name: str, generator_run: str, generator_model: str, eval_model: str) -> Path:
    return root / safe_name(run_name) / safe_name(generator_run) / safe_name(generator_model) / safe_name(eval_model)


def _cluster_dataset_types(dataset_dict, dataset_types: list[str]) -> list[str]:
    sizes = {dataset_type: len(dataset_dict[dataset_type]) if dataset_type in dataset_dict else 0 for dataset_type in dataset_types}
    indexed = {dataset_type: index for index, dataset_type in enumerate(dataset_types)}
    return sorted(dataset_types, key=lambda dataset_type: (-sizes.get(dataset_type, 0), indexed[dataset_type]))


def _dataset_sizes(
    dataset_dict,
    dataset_types: list[str],
    *,
    limit: int | None = None,
) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for dataset_type in dataset_types:
        size = len(dataset_dict[dataset_type]) if dataset_type in dataset_dict else 0
        if limit is not None and limit >= 0:
            size = min(size, limit)
        sizes[dataset_type] = size
    return sizes


def _cluster_dataset_context(args: argparse.Namespace) -> tuple[Path, list[str], dict[str, int]]:
    processed_dataset = Path(args.processed_dataset)
    dataset_dict = _load_dataset_dict(processed_dataset)
    dataset_types = _cluster_dataset_types(
        dataset_dict,
        _select_values(
            args.dataset_types,
            default=args.default_dataset_types,
            allowed=list(dataset_dict.keys()),
            label="dataset types",
        ),
    )
    return processed_dataset, dataset_types, _dataset_sizes(dataset_dict, dataset_types, limit=args.limit)


def _cluster_models(raw: str | None, *, default: list[str], backend: str | None = None) -> list[str]:
    models = [resolve_model_name(model, backend) for model in _csv_list(raw, default=default)]
    if not models:
        raise ValueError("No models selected.")
    return models


def _cluster_resources(args: argparse.Namespace) -> dict[str, object]:
    return {
        "partition": args.partition,
        "account": args.account,
        "qos": args.qos,
        "time_limit": args.time_limit,
        "memory": args.mem,
        "cpus_per_task": args.cpus_per_task,
        "gpu_type": args.gpu_type,
    }


def _runtime_argv(args: argparse.Namespace) -> list[str]:
    argv: list[str] = []
    if getattr(args, "model_base_url", None):
        argv.extend(["--model-base-url", args.model_base_url])
    if getattr(args, "max_connections", None) is not None:
        argv.extend(["--max-connections", str(args.max_connections)])
    if getattr(args, "max_tokens", None) is not None:
        argv.extend(["--max-tokens", str(args.max_tokens)])
    if getattr(args, "temperature", None) is not None:
        argv.extend(["--temperature", str(args.temperature)])
    if getattr(args, "reasoning_effort", None):
        argv.extend(["--reasoning-effort", str(args.reasoning_effort)])
    if getattr(args, "retry_on_error", None) is not None:
        argv.extend(["--retry-on-error", str(args.retry_on_error)])
    stop_seqs = getattr(args, "stop_seqs", None) or []
    if stop_seqs:
        argv.append("--stop-seqs")
        argv.extend(list(stop_seqs))
    return argv


def _strategy_phases(strategies: list[str]) -> list[list[str]]:
    ordered = [strategy for strategy in ("model_from_scratch", "augment_human", "augment_ablation") if strategy in strategies]
    phases = [ordered] if ordered else []
    if "augment_model" in strategies:
        phases.append(["augment_model"])
    return phases


def _resolved_generation_artifacts(
    *,
    processed_dataset: Path,
    dataset_types: list[str],
    run_name: str,
    model: str,
    backend: str | None,
    generation_log_dir: str | None,
    generation_log_root: str | Path | None,
    cache_root: str | Path,
    augmented_dataset: str | None,
    rebuild: bool,
    ensure: bool,
) -> tuple[str, Path, Path]:
    resolved_model = resolve_model_name(model, backend)
    log_dir = (
        Path(generation_log_dir)
        if generation_log_dir
        else _generation_log_dir(Path(generation_log_root or DEFAULT_GENERATION_LOG_ROOT), run_name, resolved_model)
    )
    explicit_augmented_dataset = Path(augmented_dataset) if augmented_dataset else None
    cache_dir = explicit_augmented_dataset or _augmented_cache_dir(Path(cache_root), run_name, resolved_model)
    if ensure and explicit_augmented_dataset is None:
        ensure_augmented_dataset(
            processed_dataset_path=processed_dataset,
            generation_log_dir=log_dir,
            output_path=cache_dir,
            dataset_types=dataset_types,
            rebuild=rebuild,
        )
    elif ensure and explicit_augmented_dataset is not None and not cache_dir.exists():
        raise FileNotFoundError(
            f"Explicit augmented dataset path does not exist: {cache_dir}. "
            "Omit --augmented-dataset to derive and materialize the cache automatically."
        )
    return resolved_model, log_dir, cache_dir


def _combined_support_ids_for_run(
    *,
    run_name: str,
    support_root: Path | str,
    cache_root: Path | str,
) -> dict[str, set[str]]:
    return combined_support_ids(
        run_name=run_name,
        support_root=Path(support_root),
        augmented_root=Path(cache_root),
    )


def _current_stage_state(*, stage: str, run_name: str, output_dir: str | None = None) -> dict[str, object]:
    paths = build_bundle_paths(stage=stage, run_name=run_name, output_dir=output_dir)
    manifests = load_scheduler_manifests(paths.run_dir)
    log_root = Path(DEFAULT_GENERATION_LOG_ROOT if stage == "generate" else DEFAULT_EVALUATION_LOG_ROOT) / safe_name(run_name)
    kind = "generation" if stage == "generate" else "evaluation"
    attempts = collect_slice_attempts(log_root, kind=kind) if log_root.exists() else {}
    return build_scheduler_state(manifests=manifests, attempts_by_slice=attempts)


def _write_scheduler_outputs(*, stage: str, run_name: str, output_dir: str | None) -> Path:
    paths = build_bundle_paths(stage=stage, run_name=run_name, output_dir=output_dir)
    state = _current_stage_state(stage=stage, run_name=run_name, output_dir=output_dir)
    paths.state_path.parent.mkdir(parents=True, exist_ok=True)
    paths.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return paths.state_path


def _slice_status_lookup(state: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(entry["slice_ref"]): entry for entry in list(state.get("slices", []))}


def _evaluation_generation_dependencies(
    *,
    setting: str,
    generator_run_name: str,
    generator_model: str,
    dataset_type: str,
    question_start: int,
    question_end: int,
    generation_state: dict[str, dict[str, object]],
) -> list[str]:
    if setting == "human_from_scratch":
        failed_candidate: str | None = None
        for strategy in SCHEDULABLE_GENERATION_STRATEGIES:
            candidate_ref = generation_slice_ref(
                run_name=generator_run_name,
                model=generator_model,
                dataset_type=dataset_type,
                strategy=strategy,
                question_start=question_start,
                question_end=question_end,
            )
            candidate_status = str((generation_state.get(candidate_ref) or {}).get("status", ""))
            if candidate_status == "current":
                return [candidate_ref]
            if candidate_status == "failed" and failed_candidate is None:
                failed_candidate = candidate_ref
        if failed_candidate is not None:
            return [failed_candidate]
        raise ValueError(
            "Missing current generation prerequisite for "
            f"human_from_scratch on {dataset_type} {question_start}:{question_end}. "
            "At least one successful generation slice for the same model, dataset, and chunk is required."
        )

    return [
        generation_slice_ref(
            run_name=generator_run_name,
            model=generator_model,
            dataset_type=dataset_type,
            strategy=dependency,
            question_start=question_start,
            question_end=question_end,
        )
        for dependency in EVALUATION_SETTING_DEPENDENCIES.get(setting, ())
    ]


def _cluster_task(
    *,
    stage: str,
    run_name: str,
    model: str,
    dataset_type: str,
    resource_class: str,
    slice_ref: str,
    task_slug_value: str,
    question_start: int,
    question_end: int,
    chunk_index: int,
    argv: list[str],
    resources: dict[str, object],
    force: bool,
    **extra: object,
) -> ClusterTask:
    return ClusterTask(
        stage=stage,
        run_name=run_name,
        model=model,
        model_slug=safe_name(model),
        dataset_type=dataset_type,
        dataset_slug=safe_name(dataset_type),
        resource_class=resource_class,
        slice_ref=slice_ref,
        task_slug=task_slug_value,
        question_start=question_start,
        question_end=question_end,
        chunk_index=chunk_index,
        argv=argv,
        resources=resources,
        submit_dependency_refs=[],
        force=force,
        **extra,
    )


def _build_generation_cluster_tasks(args: argparse.Namespace) -> tuple[list[ClusterTask], dict[str, int | None]]:
    processed_dataset, dataset_types, dataset_sizes = _cluster_dataset_context(args)
    augmented_cache_root = Path(DEFAULT_AUGMENTED_CACHE_ROOT)
    models = _cluster_models(args.models, default=list(args.default_models), backend=args.backend)
    strategies = _select_values(
        args.generation_strategies,
        default=list(SCHEDULABLE_GENERATION_STRATEGIES),
        allowed=list(SCHEDULABLE_GENERATION_STRATEGIES),
        label="generation strategies",
    )
    resources = _cluster_resources(args)
    force = bool(args.force)
    generation_artifact_base = {
        "processed_dataset": processed_dataset,
        "run_name": args.run_name,
        "backend": None,
        "generation_log_dir": None,
        "generation_log_root": DEFAULT_GENERATION_LOG_ROOT,
        "cache_root": augmented_cache_root,
        "augmented_dataset": None,
        "rebuild": False,
        "ensure": False,
    }
    existing = _slice_status_lookup(_current_stage_state(stage="generate", run_name=args.run_name, output_dir=args.output_dir))
    tasks_by_ref: dict[str, ClusterTask] = {}
    for dataset_type in dataset_types:
        for model in models:
            resource_class = resource_class_for_model(model)
            for chunk_index, question_start, question_end in chunk_ranges(dataset_sizes.get(dataset_type, 0), args.questions_per_job):
                question_limit = question_end - question_start
                for strategy in strategies:
                    metadata = _slice_task_metadata(
                        stage="generate",
                        run_name=args.run_name,
                        model=model,
                        dataset_type=dataset_type,
                        strategy=strategy,
                        question_start=question_start,
                        question_end=question_end,
                    )
                    slice_ref = str(metadata["slice_ref"])
                    existing_slice = existing.get(slice_ref, {})
                    if not args.force and str(existing_slice.get("status", "")) in {"current", "pending"}:
                        continue

                    state_dependency_refs = [
                        generation_slice_ref(
                            run_name=args.run_name,
                            model=model,
                            dataset_type=dataset_type,
                            strategy=dependency,
                            question_start=question_start,
                            question_end=question_end,
                        )
                        for dependency in GENERATION_STRATEGY_DEPENDENCIES.get(strategy, ())
                    ]
                    argv = [
                        "generate",
                        "--model",
                        model,
                        "--run-name",
                        args.run_name,
                        "--processed-dataset",
                        str(processed_dataset),
                        "--dataset-types",
                        dataset_type,
                        "--generation-strategies",
                        strategy,
                        "--question-start",
                        str(question_start),
                        "--limit",
                        str(question_limit),
                    ]
                    argv.extend(_runtime_argv(args))
                    tasks_by_ref[slice_ref] = _cluster_task(
                        stage="generate",
                        run_name=args.run_name,
                        model=model,
                        dataset_type=dataset_type,
                        resource_class=resource_class,
                        slice_ref=slice_ref,
                        task_slug_value=str(metadata["task_slug"]),
                        question_start=question_start,
                        question_end=question_end,
                        chunk_index=chunk_index,
                        strategy=strategy,
                        state_dependency_refs=state_dependency_refs,
                        argv=argv,
                        resources=resources,
                        force=force,
                    )

    runnable_counts: dict[tuple[str, str, str, int, int], int] = {}

    def runnable_generation_sample_count(task: ClusterTask) -> int:
        key = (task.model, task.dataset_type, str(task.strategy or ""), task.question_start, task.question_end)
        if key not in runnable_counts:
            _, generation_log_dir, augmented_cache_dir = _resolved_generation_artifacts(
                dataset_types=[task.dataset_type],
                model=task.model,
                **generation_artifact_base,
            )
            dataset = build_generation_dataset(
                processed_dataset,
                strategy=str(task.strategy or ""),
                dataset_types=[task.dataset_type],
                question_start=task.question_start,
                limit=task.question_end - task.question_start,
                generation_log_dir=generation_log_dir,
                augmented_dataset_path=augmented_cache_dir,
                shard_count=1,
                shard_index=0,
                shard_strategy="contiguous",
            )
            runnable_counts[key] = len(dataset)
        return runnable_counts[key]

    filtered_tasks: list[ClusterTask] = []
    for task in tasks_by_ref.values():
        submit_dependency_refs: list[str] = []
        skip_task = False
        for dependency_ref in task.state_dependency_refs or []:
            if dependency_ref in tasks_by_ref:
                submit_dependency_refs.append(dependency_ref)
            else:
                dependency_state = existing.get(dependency_ref)
                dependency_status = str((dependency_state or {}).get("status", ""))
                if dependency_status == "current":
                    continue
                if task.strategy == "augment_model":
                    if runnable_generation_sample_count(task) > 0:
                        continue
                    skip_task = True
                    break
                if dependency_status != "current":
                    raise ValueError(
                        f"Missing current prerequisite for {task.slice_ref}: {dependency_ref}. "
                        "Select the prerequisite slice in this submission or rerun it first."
                    )
        if skip_task:
            continue
        task.submit_dependency_refs = submit_dependency_refs
        filtered_tasks.append(task)

    return filtered_tasks, {"local": args.gpu_count, "api": args.gpu_count}


def _build_evaluation_cluster_tasks(args: argparse.Namespace) -> tuple[list[ClusterTask], dict[str, int | None]]:
    processed_dataset, dataset_types, dataset_sizes = _cluster_dataset_context(args)
    resources = _cluster_resources(args)
    force = bool(args.force)
    generation_artifact_base = {
        "processed_dataset": processed_dataset,
        "dataset_types": dataset_types,
        "run_name": args.generator_run_name,
        "model": args.generator_model,
        "backend": args.generator_backend,
        "generation_log_dir": getattr(args, "generation_log_dir", None),
        "generation_log_root": getattr(args, "generation_log_root", DEFAULT_GENERATION_LOG_ROOT),
        "cache_root": getattr(args, "cache_root", DEFAULT_AUGMENTED_CACHE_ROOT),
        "augmented_dataset": args.augmented_dataset,
        "rebuild": True,
    }
    generation_model, _generation_log_dir, resolved_augmented_cache_dir = _resolved_generation_artifacts(
        ensure=False,
        **generation_artifact_base,
    )
    models = _cluster_models(args.models, default=list(args.default_models), backend=args.backend)
    settings = _select_values(args.settings, default=list(SETTING_NAMES), allowed=list(SETTING_NAMES), label="settings")
    modes = _select_values(args.modes, default=list(MODE_CHOICES), allowed=list(MODE_CHOICES), label="modes")
    explicit_augmented_dataset = resolved_augmented_cache_dir if getattr(args, "augmented_dataset", None) else None

    existing = _slice_status_lookup(_current_stage_state(stage="evaluate", run_name=args.run_name, output_dir=args.output_dir))
    generation_state = (
        {}
        if explicit_augmented_dataset is not None
        else _slice_status_lookup(_current_stage_state(stage="generate", run_name=args.generator_run_name))
    )
    evaluation_counts: dict[tuple[str, str, str, int, int], int] = {}
    augmented_cache_dir: Path | None = explicit_augmented_dataset
    support_sample_ids: dict[str, set[str]] | None = None

    def runnable_evaluation_sample_count(
        *,
        dataset_type: str,
        setting: str,
        mode: str,
        question_start: int,
        question_end: int,
    ) -> int:
        nonlocal augmented_cache_dir, support_sample_ids
        key = (dataset_type, setting, mode, question_start, question_end)
        if key not in evaluation_counts:
            if augmented_cache_dir is None:
                _resolved_model, _generation_log_dir, augmented_cache_dir = _resolved_generation_artifacts(
                    ensure=True,
                    **generation_artifact_base,
                )
            if support_sample_ids is None:
                support_sample_ids = _combined_support_ids_for_run(
                    run_name=args.generator_run_name,
                    support_root=Path(args.support_root),
                    cache_root=Path(str(generation_artifact_base["cache_root"])),
                )
            dataset = build_evaluation_dataset(
                augmented_cache_dir,
                setting=setting,
                mode=mode,
                dataset_types=[dataset_type],
                support_sample_ids=support_sample_ids,
                question_start=question_start,
                limit=question_end - question_start,
                shard_count=1,
                shard_index=0,
                shard_strategy="contiguous",
            )
            evaluation_counts[key] = len(dataset)
        return evaluation_counts[key]

    tasks: list[ClusterTask] = []
    for dataset_type in dataset_types:
        for model in models:
            resource_class = resource_class_for_model(model)
            for chunk_index, question_start, question_end in chunk_ranges(dataset_sizes.get(dataset_type, 0), args.questions_per_job):
                question_limit = question_end - question_start
                for setting in settings:
                    for mode in modes:
                        metadata = _slice_task_metadata(
                            stage="evaluate",
                            run_name=args.run_name,
                            model=model,
                            dataset_type=dataset_type,
                            setting=setting,
                            mode=mode,
                            question_start=question_start,
                            question_end=question_end,
                        )
                        slice_ref = str(metadata["slice_ref"])
                        existing_slice = existing.get(slice_ref, {})
                        if not args.force and str(existing_slice.get("status", "")) in {"current", "pending"}:
                            continue

                        if explicit_augmented_dataset is not None:
                            if runnable_evaluation_sample_count(
                                dataset_type=dataset_type,
                                setting=setting,
                                mode=mode,
                                question_start=question_start,
                                question_end=question_end,
                            ) <= 0:
                                continue
                            state_dependency_refs: list[str] = []
                        else:
                            state_dependency_refs = _evaluation_generation_dependencies(
                                setting=setting,
                                generator_run_name=args.generator_run_name,
                                generator_model=generation_model,
                                dataset_type=dataset_type,
                                question_start=question_start,
                                question_end=question_end,
                                generation_state=generation_state,
                            )
                            for dependency_ref in state_dependency_refs:
                                dependency_state = generation_state.get(dependency_ref)
                                dependency_status = str((dependency_state or {}).get("status", ""))
                                if dependency_status == "current":
                                    continue
                                if dependency_status == "failed":
                                    if runnable_evaluation_sample_count(
                                        dataset_type=dataset_type,
                                        setting=setting,
                                        mode=mode,
                                        question_start=question_start,
                                        question_end=question_end,
                                    ) > 0:
                                        continue
                                    state_dependency_refs = []
                                    break
                                if dependency_status != "current":
                                    raise ValueError(
                                        f"Missing current generation prerequisite for {slice_ref}: {dependency_ref}. "
                                        "Rerun or complete the required generation slice before scheduling evaluation."
                                    )
                            if not state_dependency_refs:
                                continue

                        argv = [
                            "evaluate",
                            "--model",
                            model,
                            "--run-name",
                            args.run_name,
                            "--generator-run-name",
                            args.generator_run_name,
                            "--generator-model",
                            generation_model,
                            "--processed-dataset",
                            str(processed_dataset),
                            "--dataset-types",
                            dataset_type,
                            "--settings",
                            setting,
                            "--modes",
                            mode,
                            "--question-start",
                            str(question_start),
                            "--limit",
                            str(question_limit),
                        ]
                        if explicit_augmented_dataset is not None:
                            argv.extend(["--augmented-dataset", str(explicit_augmented_dataset)])
                        argv.extend(["--support-root", str(args.support_root)])
                        argv.append("--skip-collect-evaluated")
                        argv.extend(_runtime_argv(args))
                        tasks.append(
                            _cluster_task(
                                stage="evaluate",
                                run_name=args.run_name,
                                model=model,
                                dataset_type=dataset_type,
                                resource_class=resource_class,
                                slice_ref=slice_ref,
                                task_slug_value=str(metadata["task_slug"]),
                                question_start=question_start,
                                question_end=question_end,
                                chunk_index=chunk_index,
                                setting=setting,
                                mode=mode,
                                state_dependency_refs=state_dependency_refs,
                                argv=argv,
                                resources=resources,
                                force=force,
                                generation_run_name=args.generator_run_name,
                                generation_model=generation_model,
                                collected_root=str(args.collected_root),
                            )
                        )
    return tasks, {"local": args.gpu_count, "api": args.gpu_count}


def _run_cluster_submit(
    *,
    stage: str,
    run_name: str,
    tasks: list[ClusterTask],
    resources: dict[str, object],
    concurrency_caps: dict[str, int | None],
    output_dir: str | None,
    submit: bool,
    dry_run: bool,
) -> int:
    if not tasks:
        print("No cluster tasks selected.")
        if dry_run:
            return 0
        state_path = _write_scheduler_outputs(stage=stage, run_name=run_name, output_dir=output_dir)
        print(state_path)
        return 0

    paths = build_bundle_paths(stage=stage, run_name=run_name, output_dir=output_dir)
    manifest_text = render_manifest(
        stage=stage,
        run_name=run_name,
        resources=resources,
        tasks=tasks,
        paths=paths,
        concurrency_caps=concurrency_caps,
    )
    submit_text = render_submit_script(paths)
    wrapper_text = WRAPPER_TEMPLATE
    finalizer_wrapper_text = FINALIZER_WRAPPER_TEMPLATE

    if dry_run:
        print(f"Cluster stage: {stage}")
        print(f"Task count: {len(tasks)}")
        print(f"Run dir: {paths.run_dir}")
        print(f"Submission dir: {paths.submission_dir}")
        print(f"Manifest: {paths.manifest_path}")
        print(f"Submit: bash {paths.submit_path.name}")
        return 0

    write_bundle(
        paths=paths,
        manifest_text=manifest_text,
        submit_text=submit_text,
        local_wrapper_text=wrapper_text,
        api_wrapper_text=wrapper_text,
        finalizer_wrapper_text=finalizer_wrapper_text,
    )
    print(paths.manifest_path)
    print(paths.local_wrapper_path)
    print(paths.api_wrapper_path)
    print(paths.finalizer_wrapper_path)
    print(paths.submit_path)

    state_path = _write_scheduler_outputs(stage=stage, run_name=run_name, output_dir=output_dir)
    print(state_path)

    if not submit:
        return 0

    try:
        result = submit_bundle(paths)
    except OSError as exc:
        print(str(exc))
        return 1
    _write_scheduler_outputs(stage=stage, run_name=run_name, output_dir=output_dir)
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr.strip())
        return int(result.returncode)
    return 0


def _prepare_data(args: argparse.Namespace) -> int:
    prepare_fn = prepare_data
    if prepare_fn is None:
        from data import prepare_data as prepare_fn

    download_all = bool(args.all or args.step == "all")
    return prepare_fn(
        step=args.step,
        dataset=args.dataset,
        download_all=download_all,
        output_dir=args.output_dir,
        output_path=args.output_path,
        limit=args.limit,
    )


def _evaluation_collection_state(
    *,
    run_name: str,
    evaluation_log_root: Path | str,
    scheduler_output_dir: str | None = None,
    planned_tasks: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    attempts = collect_slice_attempts(evaluation_log_root, kind="evaluation")
    if planned_tasks is not None:
        state = build_scheduler_state_from_tasks(
            stage="evaluate",
            run_name=run_name,
            planned_tasks=planned_tasks,
            attempts_by_slice=attempts,
        )
    else:
        manifests = load_scheduler_manifests(Path(scheduler_output_dir)) if scheduler_output_dir else []
        state = build_scheduler_state(manifests=manifests, attempts_by_slice=attempts)
    state["run_name"] = run_name
    return state


def _local_evaluation_task_plans(tasks: list[object], *, fallback_model: str) -> list[dict[str, object]]:
    planned: list[dict[str, object]] = []
    for task in tasks:
        metadata = dict(getattr(task, "metadata", {}) or {})
        setting = str(metadata.get("setting", "") or "")
        mode = str(metadata.get("mode", "") or "")
        dataset_types = [str(value) for value in list(metadata.get("dataset_types", []) or []) if str(value)]
        dataset_label = dataset_types[0] if len(dataset_types) == 1 else "__".join(dataset_types) or "dataset_bundle"
        question_start = int(metadata.get("question_start", 0) or 0)
        raw_limit = metadata.get("limit")
        question_span = (
            int(raw_limit)
            if raw_limit not in (None, "")
            else len(getattr(task, "dataset", []) or [])
        )
        question_end = question_start + max(0, question_span)
        eval_model = str(metadata.get("evaluation_model") or fallback_model or "")
        slice_metadata = _slice_task_metadata(
            stage="evaluate",
            run_name=str(metadata.get("run_name", "") or ""),
            model=eval_model,
            dataset_type=dataset_label,
            setting=setting or None,
            mode=mode or None,
            question_start=question_start,
            question_end=question_end,
        )
        slice_ref = str(metadata.get("slice_ref", "") or slice_metadata["slice_ref"])
        task_slug_value = str(metadata.get("task_slug", "") or slice_metadata["task_slug"])
        metadata["slice_ref"] = slice_ref
        metadata["task_slug"] = task_slug_value
        if isinstance(getattr(task, "metadata", None), dict):
            task.metadata.update(metadata)
        else:
            try:
                setattr(task, "metadata", metadata)
            except (AttributeError, TypeError):
                pass
        planned.append(
            _planned_task_entry(
                stage="evaluate",
                run_name=str(metadata.get("run_name", "") or ""),
                model=eval_model,
                dataset_type=dataset_label,
                question_start=question_start,
                question_end=question_end,
                task_slug_value=task_slug_value,
                slice_ref=slice_ref,
                force=False,
                dataset_types=dataset_types,
                state_dependency_refs=list(metadata.get("state_dependency_refs", []) or []),
                submit_dependency_refs=list(metadata.get("submit_dependency_refs", []) or []),
                setting=setting,
                mode=mode,
                generation_run_name=str(metadata.get("generation_run_name", "") or ""),
                generation_model=str(metadata.get("generation_model", "") or ""),
            )
        )
    return planned


def _filtered_collection_state(
    state: dict[str, object],
    *,
    generation_run_name: str,
    generation_model: str,
    evaluation_model: str,
) -> dict[str, object]:
    slices = list(state.get("slices", []))
    filtered = [
        dict(entry)
        for entry in slices
        if (
            not str(entry.get("generation_run_name", "") or "")
            or str(entry.get("generation_run_name", "") or "") == generation_run_name
        )
        and (
            not str(entry.get("generation_model", "") or "")
            or str(entry.get("generation_model", "") or "") == generation_model
        )
        and (
            not str(entry.get("model", "") or "")
            or str(entry.get("model", "") or "") == evaluation_model
        )
    ]
    if not filtered and slices and all(not str(entry.get("model", "") or "") for entry in slices):
        filtered = [dict(entry) for entry in slices]
    status_counts: dict[str, int] = {}
    for entry in filtered:
        status = str(entry.get("status", "") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "schema_version": "augmented_mcqa_collected_state_v1",
        "storage_kind": "evaluation_slice_state",
        "generated_at": str(state.get("generated_at") or iso_now()),
        "run_name": str(state.get("run_name", "") or ""),
        "generation_run_name": generation_run_name,
        "generation_model": generation_model,
        "evaluation_model": evaluation_model,
        "slice_count": len(filtered),
        "status_counts": status_counts,
        "slices": filtered,
    }


def _write_collected_state(
    output_paths: list[Path],
    *,
    state: dict[str, object],
) -> None:
    for output_path in output_paths:
        manifest_path = output_path / EVALUATED_STORE_MANIFEST
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = _filtered_collection_state(
            state,
            generation_run_name=str(manifest.get("generation_run_name", "") or ""),
            generation_model=str(manifest.get("generation_model", "") or ""),
            evaluation_model=str(manifest.get("evaluation_model", "") or ""),
        )
        (output_path / COLLECTED_STATE_FILENAME).write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )


def _materialize_collected_evaluation(
    *,
    run_name: str,
    evaluation_log_root: Path | str,
    collected_root: Path | str,
    augmented_dataset: Path | str,
    support_root: Path | str,
    generation_run_name: str,
    generation_model: str,
    evaluation_model: str,
    dataset_types: list[str],
    settings: list[str],
    modes: list[str],
    scheduler_output_dir: str | None = None,
    planned_tasks: list[dict[str, object]] | None = None,
) -> list[Path]:
    outputs = materialize_evaluated_datasets(
        evaluation_log_root=evaluation_log_root,
        output_root=collected_root,
        augmented_root=augmented_dataset,
        support_root=support_root,
        expected_dataset_types=dataset_types,
        expected_settings=settings,
        expected_modes=modes,
        generation_run_name=generation_run_name,
        generation_model=generation_model,
        evaluation_model=evaluation_model,
    )
    state = _evaluation_collection_state(
        run_name=run_name,
        evaluation_log_root=evaluation_log_root,
        scheduler_output_dir=scheduler_output_dir,
        planned_tasks=planned_tasks,
    )
    _write_collected_state(outputs, state=state)
    return outputs


def _run_generate(args: argparse.Namespace) -> int:
    from tasks import build_generation_tasks

    dataset_types = _select_values(
        args.dataset_types,
        default=args.default_dataset_types,
        allowed=list(ACTIVE_DATASET_TYPES),
        label="dataset types",
    )
    generation_model, log_dir, cache_dir = _resolved_generation_artifacts(
        processed_dataset=Path(args.processed_dataset),
        dataset_types=dataset_types,
        run_name=args.run_name,
        model=args.model,
        backend=args.backend,
        generation_log_dir=None,
        generation_log_root=args.log_root,
        cache_root=args.cache_root,
        augmented_dataset=args.augmented_dataset,
        rebuild=args.rebuild_cache,
        ensure=False,
    )
    strategies = _select_values(
        args.generation_strategies,
        default=list(SCHEDULABLE_GENERATION_STRATEGIES),
        allowed=list(SCHEDULABLE_GENERATION_STRATEGIES),
        label="generation strategies",
    )

    task_metadata_by_strategy: dict[str, dict[str, object]] = {}
    if len(dataset_types) == 1:
        dataset_type = dataset_types[0]
        question_start = int(getattr(args, "question_start", 0) or 0)
        question_limit = int(args.limit or 0)
        question_end = question_start + question_limit if question_limit > 0 else question_start
        for strategy in strategies:
            if question_limit <= 0:
                continue
            task_metadata_by_strategy[strategy] = _slice_task_metadata(
                stage="generate",
                run_name=args.run_name,
                model=generation_model,
                dataset_type=dataset_type,
                strategy=strategy,
                question_start=question_start,
                question_end=question_end,
            )

    any_tasks = False
    for phase in _strategy_phases(strategies):
        tasks = build_generation_tasks(
            processed_dataset_path=Path(args.processed_dataset),
            strategies=phase,
            dataset_types=dataset_types,
            question_start=args.question_start,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
            shard_strategy=args.shard_strategy,
            limit=args.limit,
            run_name=args.run_name,
            generation_model=generation_model,
            generation_log_dir=log_dir,
            augmented_dataset_path=cache_dir,
            task_metadata_by_strategy=task_metadata_by_strategy,
        )
        if not tasks:
            continue
        any_tasks = True
        eval_logs = inspect_eval(tasks, model=generation_model, log_dir=log_dir, args=args)
        if not generation_logs_succeeded(eval_logs):
            return 1

    if not any_tasks:
        if any(GENERATION_STRATEGY_DEPENDENCIES.get(strategy) for strategy in strategies):
            print("No generation samples selected. Missing successful prerequisite generation outputs.")
            return 1
        print("No generation samples selected.")
        return 0
    print(f"Generation logs: {log_dir}")
    if args.materialize_cache:
        ensure_augmented_dataset(
            processed_dataset_path=Path(args.processed_dataset),
            generation_log_dir=log_dir,
            output_path=cache_dir,
            dataset_types=dataset_types,
            rebuild=args.rebuild_cache,
        )
        build_generation_support_manifest(
            processed_dataset_path=Path(args.processed_dataset),
            generation_log_dir=log_dir,
            run_name=args.run_name,
            generation_model=generation_model,
            output_root=Path(args.support_root),
            dataset_types=dataset_types,
        )
        print(f"Augmented dataset cache: {cache_dir}")
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    from tasks import build_evaluation_tasks

    dataset_types = _select_values(
        args.dataset_types,
        default=args.default_dataset_types,
        allowed=list(ACTIVE_DATASET_TYPES),
        label="dataset types",
    )
    settings = _select_values(args.settings, default=list(SETTING_NAMES), allowed=list(SETTING_NAMES), label="settings")
    modes = _select_values(args.modes, default=list(MODE_CHOICES), allowed=list(MODE_CHOICES), label="modes")
    eval_model = resolve_model_name(args.model, args.backend)
    generation_model, _generation_log_dir, cache_dir = _resolved_generation_artifacts(
        processed_dataset=Path(args.processed_dataset),
        dataset_types=dataset_types,
        run_name=args.generator_run_name,
        model=args.generator_model,
        backend=args.generator_backend,
        generation_log_dir=args.generation_log_dir,
        generation_log_root=args.generation_log_root,
        cache_root=args.cache_root,
        augmented_dataset=args.augmented_dataset,
        rebuild=args.rebuild_cache,
        ensure=True,
    )
    log_dir = _evaluation_log_dir(
        Path(args.log_root),
        args.run_name,
        args.generator_run_name,
        generation_model,
        eval_model,
    )
    support_sample_ids = _combined_support_ids_for_run(
        run_name=args.generator_run_name,
        support_root=Path(args.support_root),
        cache_root=Path(args.cache_root),
    )
    tasks = build_evaluation_tasks(
        augmented_dataset_path=cache_dir,
        support_sample_ids=support_sample_ids,
        dataset_types=dataset_types,
        settings=settings,
        modes=modes,
        question_start=args.question_start,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        shard_strategy=args.shard_strategy,
        limit=args.limit,
        run_name=args.run_name,
        generation_run_name=args.generator_run_name,
        generation_model=generation_model,
        evaluation_model=eval_model,
        task_metadata_by_setting_mode={
            (setting, mode): _slice_task_metadata(
                stage="evaluate",
                run_name=args.run_name,
                model=eval_model,
                dataset_type=dataset_types[0],
                setting=setting,
                mode=mode,
                question_start=args.question_start,
                question_end=args.question_start + int(args.limit or 0),
            )
            for setting in settings
            for mode in modes
            if len(dataset_types) == 1 and args.limit
        },
    )
    if not tasks:
        print("No evaluation samples selected.")
        return 0
    planned_tasks = _local_evaluation_task_plans(tasks, fallback_model=eval_model)
    eval_logs = inspect_eval(tasks, model=eval_model, log_dir=log_dir, args=args)
    print(f"Evaluation logs: {log_dir}")
    if args.collect_evaluated:
        collected_outputs = _materialize_collected_evaluation(
            run_name=args.run_name,
            evaluation_log_root=log_dir,
            collected_root=Path(args.collected_root),
            augmented_dataset=cache_dir,
            support_root=Path(args.support_root),
            generation_run_name=args.generator_run_name,
            generation_model=generation_model,
            evaluation_model=eval_model,
            dataset_types=dataset_types,
            settings=settings,
            modes=modes,
            planned_tasks=planned_tasks,
        )
        for output in collected_outputs:
            print(output)
    return 0 if evaluation_logs_completed(eval_logs) else 1


def _run_analyze(args: argparse.Namespace) -> int:
    from analysis.visualize import (
        load_analysis_frames,
        plot_pairwise_accuracy,
        write_results_summary_table,
    )

    analysis_root = Path(args.collected_root)
    if not analysis_root.exists():
        print(f"Missing collected dataset root: {analysis_root}")
        return 1

    row_df, summary_df = load_analysis_frames(analysis_root)
    if args.table_output:
        df = write_results_summary_table(
            analysis_root,
            args.table_output,
            summary_df=summary_df,
        )
        print(f"Wrote {len(df)} summary rows to {args.table_output}")
    outputs = plot_pairwise_accuracy(
        results_root=analysis_root,
        output_dir=Path(args.output_dir),
        include_tables=not args.skip_tables,
        row_df=row_df,
        summary_df=summary_df,
    )
    if not outputs:
        print("No evaluated rows found.")
        return 1
    for output in outputs:
        print(output)
    return 0


def _run_analyze_irt(args: argparse.Namespace) -> int:
    from analysis.irt import run_cli

    return run_cli(args)


def _run_collect_evaluated(args: argparse.Namespace) -> int:
    outputs: list[Path] = []
    if args.collection_spec:
        for raw_spec in args.collection_spec:
            spec = json.loads(raw_spec)
            model_outputs = _materialize_collected_evaluation(
                run_name=args.run_name,
                evaluation_log_root=Path(spec["evaluation_log_root"]),
                collected_root=Path(args.collected_root),
                augmented_dataset=Path(spec["augmented_dataset"]),
                support_root=Path(args.support_root),
                generation_run_name=str(spec["generator_run_name"]),
                generation_model=str(spec["generator_model"]),
                evaluation_model=str(spec["evaluation_model"]),
                dataset_types=list(spec["dataset_types"]),
                settings=list(spec["settings"]),
                modes=list(spec["modes"]),
                scheduler_output_dir=args.scheduler_output_dir,
            )
            outputs.extend(model_outputs)
    else:
        outputs = _materialize_collected_evaluation(
            run_name=args.run_name,
            evaluation_log_root=Path(args.evaluation_log_root),
            collected_root=Path(args.collected_root),
            augmented_dataset=Path(args.augmented_dataset),
            support_root=Path(args.support_root),
            generation_run_name=args.generator_run_name,
            generation_model=args.generator_model,
            evaluation_model=args.model,
            dataset_types=_select_values(args.dataset_types, default=args.default_dataset_types, allowed=list(ACTIVE_DATASET_TYPES), label="dataset types"),
            settings=_select_values(args.settings, default=list(SETTING_NAMES), allowed=list(SETTING_NAMES), label="settings"),
            modes=_select_values(args.modes, default=list(MODE_CHOICES), allowed=list(MODE_CHOICES), label="modes"),
            scheduler_output_dir=args.scheduler_output_dir,
        )
    for output in outputs:
        print(output)
    return 0


def _run_export(args: argparse.Namespace) -> int:
    from data import export_benchmarker_items

    dataset_path = Path(args.input)
    summary_path = export_benchmarker_items(dataset_path, args.output_root)
    print(summary_path)
    return 0


def _run_materialize_store(args: argparse.Namespace) -> int:
    raw_model = resolve_model_name(args.model, args.backend)
    log_dir = _generation_log_dir(Path(args.generation_log_root), args.run_name, raw_model)
    output_path = (
        Path(args.output_path)
        if args.output_path
        else _augmented_cache_dir(Path(args.cache_root), args.run_name, raw_model)
    )
    dataset_types = _select_values(
        args.dataset_types,
        default=args.default_dataset_types,
        allowed=list(ACTIVE_DATASET_TYPES),
        label="dataset types",
    )
    ensure_augmented_dataset(
        processed_dataset_path=Path(args.processed_dataset),
        generation_log_dir=log_dir,
        output_path=output_path,
        dataset_types=dataset_types,
        rebuild=args.rebuild_cache,
    )
    support_path = build_generation_support_manifest(
        processed_dataset_path=Path(args.processed_dataset),
        generation_log_dir=log_dir,
        run_name=args.run_name,
        generation_model=raw_model,
        output_root=Path(args.support_root),
        dataset_types=dataset_types,
    )
    print(output_path)
    print(support_path)
    return 0


def _add_materialize_store_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    formatter,
) -> argparse.ArgumentParser:
    parser = sub.add_parser(
        "build-augmented-dataset",
        prog="main.py build-augmented-dataset",
        description="Build the setting-scoped augmented dataset for one generation run/model directly from Inspect generation logs.",
        formatter_class=formatter,
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Generation run name whose Inspect logs should be materialized.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Generation model whose Inspect logs should be materialized.",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Situational: provider prefix to apply to an unqualified model name.",
    )
    parser.add_argument(
        "--generation-log-root",
        default=str(DEFAULT_GENERATION_LOG_ROOT),
        help="Advanced override: root directory containing generation Inspect logs.",
    )
    parser.add_argument(
        "--processed-dataset",
        default=str(DEFAULT_PROCESSED_DATASET),
        help="Processed dataset source used to rebuild the augmented store from logs.",
    )
    parser.add_argument(
        "--cache-root",
        default=str(DEFAULT_AUGMENTED_CACHE_ROOT),
        help="Advanced override: root directory where augmented caches are stored.",
    )
    parser.add_argument(
        "--support-root",
        default=str(DEFAULT_SUPPORT_SET_ROOT),
        help="Advanced override: root directory where generation support manifests should be written.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Advanced override: exact output path for the rebuilt augmented cache.",
    )
    parser.add_argument(
        "--dataset-types",
        default=None,
        help="Optional subset: comma-separated subset of dataset splits to materialize.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Advanced override: force regeneration even if the cache appears up to date.",
    )
    parser.set_defaults(default_dataset_types=list(DEFAULT_DATASET_TYPES))
    parser.set_defaults(handler=_run_materialize_store)
    return parser


def _add_collect_evaluated_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    formatter,
) -> argparse.ArgumentParser:
    parser = sub.add_parser(
        "build-collected-dataset",
        prog="main.py build-collected-dataset",
        description="Build collected evaluation datasets directly from evaluation logs.",
        formatter_class=formatter,
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--generator-run-name", required=False)
    parser.add_argument("--generator-model", required=False)
    parser.add_argument("--model", required=False, help="Evaluation model whose collected dataset should be materialized.")
    parser.add_argument("--evaluation-log-root", required=False)
    parser.add_argument("--augmented-dataset", required=False)
    parser.add_argument(
        "--collected-root",
        default=str(DEFAULT_COLLECTED_DATASET_ROOT),
        help="Root directory where collected evaluation datasets should be written.",
    )
    parser.add_argument(
        "--support-root",
        default=str(DEFAULT_SUPPORT_SET_ROOT),
        help="Advanced override: root directory where generation support manifests are stored.",
    )
    parser.add_argument(
        "--scheduler-output-dir",
        default=None,
        help="Optional scheduler bundle root used to capture slice state inside the collected dataset.",
    )
    parser.add_argument(
        "--dataset-types",
        default=None,
        help="Comma-separated subset of dataset splits expected in the collected dataset.",
    )
    parser.add_argument(
        "--settings",
        default=None,
        help="Comma-separated subset of settings expected in the collected dataset.",
    )
    parser.add_argument(
        "--modes",
        default=None,
        help="Comma-separated subset of modes expected in the collected dataset.",
    )
    parser.add_argument(
        "--collection-spec",
        action="append",
        default=[],
        help="JSON-encoded per-model collection spec emitted by the cluster scheduler.",
    )
    parser.set_defaults(default_dataset_types=list(DEFAULT_DATASET_TYPES))
    parser.set_defaults(handler=_run_collect_evaluated)
    return parser


def _run_submit_generate_cluster(args: argparse.Namespace) -> int:
    return _run_submit_cluster(args, stage="generate", builder=_build_generation_cluster_tasks)


def _run_submit_evaluate_cluster(args: argparse.Namespace) -> int:
    return _run_submit_cluster(args, stage="evaluate", builder=_build_evaluation_cluster_tasks)


def _run_submit_cluster(
    args: argparse.Namespace,
    *,
    stage: str,
    builder,
) -> int:
    try:
        tasks, concurrency_caps = builder(args)
    except ValueError as exc:
        print(str(exc))
        return 1
    return _run_cluster_submit(
        stage=stage,
        run_name=args.run_name,
        tasks=tasks,
        resources=_cluster_resources(args),
        concurrency_caps=concurrency_caps,
        output_dir=args.output_dir,
        submit=args.submit,
        dry_run=args.dry_run,
    )


def add_runtime_flags(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--backend",
        default=None,
        help="Situational: provider prefix to apply to an unqualified model name, such as openai or vllm.",
    )
    command.add_argument(
        "--model-base-url",
        default=None,
        help="Situational: base URL for OpenAI-compatible model endpoints or custom provider endpoints.",
    )
    command.add_argument(
        "--max-connections",
        type=int,
        default=None,
        help="Advanced tuning: maximum concurrent model connections Inspect may open for this run.",
    )
    command.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Advanced tuning: maximum tokens requested from the model for each generation or evaluation call.",
    )
    command.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Advanced tuning: sampling temperature forwarded to the model backend.",
    )
    command.add_argument(
        "--reasoning-effort",
        default="medium",
        help="Advanced tuning: optional reasoning-effort hint for models/providers that support it.",
    )
    command.add_argument(
        "--retry-on-error",
        type=int,
        default=2,
        help="Advanced tuning: how many times Inspect should retry a failed model call.",
    )
    command.add_argument(
        "--stop-seqs",
        nargs="*",
        default=None,
        help="Advanced tuning: optional stop sequences forwarded to the model backend.",
    )


def add_shard_flags(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Manual fallback control: number of deterministic shards to split the selected samples into. Most users should use the cluster submit commands instead.",
    )
    command.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Manual fallback control: zero-based shard index to run from the selected shard count.",
    )
    command.add_argument(
        "--shard-strategy",
        choices=["contiguous", "modulo"],
        default="contiguous",
        help="Manual fallback control: how samples are partitioned across shards when using explicit shard flags.",
    )


def add_cluster_submit_flags(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of models to schedule. Models can be local vllm/... or hosted/API providers.",
    )
    command.add_argument(
        "--processed-dataset",
        default=str(DEFAULT_PROCESSED_DATASET),
        help="Unified processed DatasetDict to use when building scheduler slices.",
    )
    command.add_argument(
        "--dataset-types",
        default=None,
        help="Comma-separated subset of dataset splits to schedule, such as arc_challenge,gpqa.",
    )
    command.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Advanced/debug option: optional per-dataset cap on the number of samples to schedule before chunking.",
    )
    command.add_argument(
        "--questions-per-job",
        type=int,
        default=None,
        help="Optional contiguous question-chunk size per scheduled slice. If omitted, one chunk is used per model×dataset unit.",
    )
    command.add_argument(
        "--gpu-count",
        type=int,
        default=None,
        help="Optional scheduler concurrency cap applied per resource class when submitting per-slice jobs.",
    )
    command.add_argument(
        "--output-dir",
        default=None,
        help="Advanced override: directory where generated manifests, wrappers, state files, and helper scripts should be written.",
    )
    command.add_argument(
        "--submit",
        dest="submit",
        action="store_true",
        default=True,
        help="Advanced control: submit the generated sbatch array after writing bundle files.",
    )
    command.add_argument(
        "--write-only",
        dest="submit",
        action="store_false",
        help="Advanced control: write manifests and submit helpers but do not call sbatch.",
    )
    command.add_argument(
        "--dry-run",
        action="store_true",
        help="Advanced control: print the planned scheduler details without writing or submitting anything.",
    )
    command.add_argument(
        "--force",
        action="store_true",
        help="Resubmit the selected slices even if they are already current or pending in this run.",
    )
    command.add_argument(
        "--partition",
        default="clip",
        help="Advanced cluster override: SLURM partition to request for each generated slice job.",
    )
    command.add_argument(
        "--account",
        default="clip",
        help="Advanced cluster override: SLURM account to charge for each generated slice job.",
    )
    command.add_argument(
        "--qos",
        default="high",
        help="Advanced cluster override: SLURM quality-of-service value to set on generated jobs.",
    )
    command.add_argument(
        "--time-limit",
        default="12:00:00",
        help="Advanced cluster override: wall-clock time limit for each generated job.",
    )
    command.add_argument(
        "--mem",
        default="32G",
        help="Advanced cluster override: memory request for each generated job.",
    )
    command.add_argument(
        "--cpus-per-task",
        type=int,
        default=4,
        help="Advanced cluster override: CPU cores requested per generated job.",
    )
    command.add_argument(
        "--gpu-type",
        default="rtxa6000",
        help="Advanced cluster override: GPU type to request for local-model jobs.",
    )


def _add_prepare_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser], formatter) -> argparse.ArgumentParser:
    prepare = sub.add_parser(
        "prepare-data",
        help="Download raw datasets and/or build the unified processed dataset.",
        description="Download raw source datasets and/or process them into the unified Augmented MCQA dataset.",
        formatter_class=formatter,
    )
    prepare.add_argument("--step", choices=["download", "process", "all"], default="all", help="Which stage of data preparation to run.")
    prepare.add_argument("--dataset", choices=["mmlu_pro", "mmlu", "arc", "gpqa"], default=None, help="Specific raw dataset to download when not using --all.")
    prepare.add_argument("--all", action="store_true", help="Download every supported raw dataset instead of a single dataset.")
    prepare.add_argument("--output-dir", default="datasets/raw", help="Advanced override: directory where raw downloaded datasets should be stored.")
    prepare.add_argument("--output-path", default=str(DEFAULT_PROCESSED_DATASET), help="Directory where the processed unified DatasetDict should be written.")
    prepare.add_argument("--limit", type=int, default=None, help="Advanced/debug option: optional per-dataset cap when building the processed unified dataset.")
    prepare.set_defaults(handler=_prepare_data)
    return prepare


def _add_generate_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser], formatter) -> argparse.ArgumentParser:
    generate = sub.add_parser(
        "generate",
        help="Run Augmented MCQA distractor generation for one model.",
        description="Generate Augmented MCQA distractor variants for one model over the processed dataset.",
        formatter_class=formatter,
    )
    generate.add_argument("--model", required=True, help="Model name or alias to use for generation.")
    generate.add_argument("--run-name", required=True, help="Logical run name used to organize logs and caches.")
    generate.add_argument("--processed-dataset", default=str(DEFAULT_PROCESSED_DATASET), help="Processed dataset source to read input questions from. Accepts the unified DatasetDict or a dataset manifest JSON.")
    generate.add_argument("--dataset-types", default=None, help="Optional subset: comma-separated subset of dataset splits to generate for.")
    generate.add_argument("--generation-strategies", default=None, help="Advanced subset override: comma-separated subset of schedulable generation strategies to run.")
    generate.add_argument("--question-start", type=int, default=0, help="Advanced/debug option: zero-based per-dataset starting row for generation.")
    generate.add_argument("--limit", type=int, default=None, help="Advanced/debug option: optional per-dataset cap on the number of samples to generate.")
    generate.add_argument("--log-root", default=str(DEFAULT_GENERATION_LOG_ROOT), help="Advanced override: root directory for Inspect generation logs.")
    generate.add_argument("--cache-root", default=str(DEFAULT_AUGMENTED_CACHE_ROOT), help="Advanced override: root directory where derived augmented dataset caches should be stored.")
    generate.add_argument("--support-root", default=str(DEFAULT_SUPPORT_SET_ROOT), help="Advanced override: root directory where generation support manifests are stored.")
    generate.add_argument("--augmented-dataset", default=None, help="Advanced override: exact output path for the setting-scoped augmented store produced from generation logs.")
    generate.add_argument("--materialize-cache", action="store_true", help="Rebuild the setting-scoped augmented store immediately after generation completes.")
    generate.add_argument("--rebuild-cache", action="store_true", help="Advanced override: force regeneration of the augmented cache even if it already exists.")
    generate.set_defaults(default_dataset_types=list(DEFAULT_DATASET_TYPES))
    add_runtime_flags(generate)
    add_shard_flags(generate)
    generate.set_defaults(handler=_run_generate)
    return generate


def _add_evaluate_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser], formatter) -> argparse.ArgumentParser:
    evaluate = sub.add_parser(
        "evaluate",
        help="Evaluate one model against one generation run.",
        description="Evaluate a single model across the requested Augmented MCQA settings and modes.",
        formatter_class=formatter,
    )
    evaluate.add_argument("--model", required=True, help="Model name or alias to use for evaluation.")
    evaluate.add_argument("--run-name", required=True, help="Logical run name used to organize evaluation logs.")
    evaluate.add_argument("--generator-run-name", required=True, help="Generation run name whose augmented cache or logs should be evaluated.")
    evaluate.add_argument("--generator-model", required=True, help="Generation model whose outputs should be evaluated.")
    evaluate.add_argument("--generator-backend", default=None, help="Situational: backend prefix to apply when resolving --generator-model.")
    evaluate.add_argument("--generation-log-dir", default=None, help="Advanced override: exact generation log directory to read instead of deriving one from run name and model.")
    evaluate.add_argument("--generation-log-root", default=str(DEFAULT_GENERATION_LOG_ROOT), help="Advanced override: root directory for generation Inspect logs when deriving inputs automatically.")
    evaluate.add_argument("--processed-dataset", default=str(DEFAULT_PROCESSED_DATASET), help="Processed dataset source used only when deriving and materializing the augmented store automatically from generation logs.")
    evaluate.add_argument("--augmented-dataset", default=None, help="Advanced override: exact read-only augmented store path to evaluate instead of deriving one from generation artifacts.")
    evaluate.add_argument("--cache-root", default=str(DEFAULT_AUGMENTED_CACHE_ROOT), help="Advanced override: root directory where augmented dataset caches are stored.")
    evaluate.add_argument("--support-root", default=str(DEFAULT_SUPPORT_SET_ROOT), help="Advanced override: root directory where generation support manifests are stored.")
    evaluate.add_argument("--dataset-types", default=None, help="Optional subset: comma-separated subset of dataset splits to evaluate.")
    evaluate.add_argument("--question-start", type=int, default=0, help="Advanced/debug option: zero-based per-dataset starting row for evaluation.")
    evaluate.add_argument("--settings", default=None, help="Advanced subset override: comma-separated subset of Augmented MCQA settings to evaluate.")
    evaluate.add_argument("--modes", default=None, help="Advanced subset override: comma-separated subset of evaluation modes to run.")
    evaluate.add_argument("--limit", type=int, default=None, help="Advanced/debug option: optional per-dataset cap on the number of evaluation samples.")
    evaluate.add_argument("--log-root", default=str(DEFAULT_EVALUATION_LOG_ROOT), help="Advanced override: root directory for Inspect evaluation logs.")
    evaluate.add_argument(
        "--collected-root",
        default=str(DEFAULT_COLLECTED_DATASET_ROOT),
        help="Root directory where the collected evaluation dataset should be written when collection is enabled.",
    )
    evaluate.add_argument(
        "--collect-evaluated",
        dest="collect_evaluated",
        action="store_true",
        default=True,
        help="Materialize the collected evaluation dataset immediately after this evaluate command finishes.",
    )
    evaluate.add_argument(
        "--skip-collect-evaluated",
        dest="collect_evaluated",
        action="store_false",
        help="Skip collected dataset materialization for this evaluate command. Intended for cluster slice jobs that defer collection to a finalizer.",
    )
    evaluate.add_argument("--rebuild-cache", action="store_true", help="Advanced override: force regeneration of the augmented cache before evaluation.")
    evaluate.set_defaults(default_dataset_types=list(DEFAULT_DATASET_TYPES))
    add_runtime_flags(evaluate)
    add_shard_flags(evaluate)
    evaluate.set_defaults(handler=_run_evaluate)
    return evaluate


def _add_analyze_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser], formatter) -> argparse.ArgumentParser:
    analyze = sub.add_parser(
        "analyze",
        help="Build Augmented MCQA plots and summary tables from collected datasets.",
        description="Analyze collected evaluation datasets directly, without rebuilding them from Inspect evaluation logs.",
        formatter_class=formatter,
    )
    analyze.add_argument("--output-dir", default="results/augmented_mcqa_plots", help="Situational output override: directory where plots and optional tables should be written.")
    analyze.add_argument("--table-output", default="results/augmented_mcqa_plots/tables/augmented_mcqa_results_summary.csv", help="Advanced output override: CSV path for the flat summary table.")
    analyze.add_argument("--skip-tables", action="store_true", help="Advanced output option: write plots only and skip the pairwise comparison CSV tables.")
    analyze.add_argument("--collected-root", default=str(DEFAULT_COLLECTED_DATASET_ROOT), help="Root directory or specific collected dataset folder to analyze.")
    analyze.set_defaults(handler=_run_analyze)
    return analyze


def _add_analyze_irt_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser], formatter) -> argparse.ArgumentParser:
    analyze = sub.add_parser(
        "analyze-irt",
        help="Fit a decomposed 3PL IRT model over collected Augmented MCQA evaluations.",
        description="Run the custom SciPy 3PL IRT analysis directly from collected full-question evaluation datasets.",
        formatter_class=formatter,
    )
    analyze.add_argument("--collected-root", default=str(DEFAULT_COLLECTED_DATASET_ROOT), help="Root directory or specific collected dataset folder to analyze.")
    analyze.add_argument("--output-dir", default="results/augmented_mcqa_irt", help="Directory where IRT tables, plots, and summary JSON should be written.")
    analyze.add_argument("--generators", default=None, help="Optional comma-separated subset of generator model names to include.")
    analyze.add_argument("--evaluators", default=None, help="Optional comma-separated subset of evaluator model names to include.")
    analyze.add_argument("--datasets", default=None, help="Optional comma-separated subset of dataset names to include.")
    analyze.add_argument("--settings", default=None, help="Optional comma-separated subset of settings to include.")
    analyze.add_argument("--modes", default=None, help=argparse.SUPPRESS)
    analyze.add_argument("--maxiter", type=int, default=2000, help="Maximum optimizer iterations per fitted model.")
    analyze.add_argument("--maxfun", type=int, default=50000, help="Maximum objective evaluations per fitted model.")
    analyze.add_argument("--gtol", type=float, default=1e-5, help="Gradient tolerance for the L-BFGS-B optimizer.")
    analyze.add_argument("--item-prior-sd", type=float, default=3.0, help=argparse.SUPPRESS)
    analyze.set_defaults(handler=_run_analyze_irt)
    return analyze


def _add_export_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser], formatter) -> argparse.ArgumentParser:
    export = sub.add_parser(
        "export",
        help="Export an augmented store or collected group root to benchmarker JSONL files.",
        description="Export a setting-scoped augmented store, or a collected evaluated group root, into benchmarker-compatible JSONL files.",
        formatter_class=formatter,
    )
    export.add_argument("--input", required=True, help="Exact augmented store root, or exact collected group root, to export.")
    export.add_argument("--output-root", default="datasets/benchmarker_items", help="Situational output override: root directory where benchmarker JSONL outputs should be written.")
    export.set_defaults(handler=_run_export)
    return export


def _add_submit_generate_cluster_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser], formatter) -> argparse.ArgumentParser:
    parser = sub.add_parser(
        "submit-generate-cluster",
        help="Submit dependency-aware generation jobs over model×dataset×strategy×chunk slices.",
        description="Generate per-slice SLURM submissions for local and API-backed generation, with exact dependency wiring where needed.",
        formatter_class=formatter,
    )
    parser.add_argument("--run-name", required=True, help="Logical run name used to organize generated manifests, logs, and output caches.")
    parser.add_argument("--generation-strategies", default=None, help="Comma-separated subset of schedulable generation strategies to submit. human_from_scratch remains implicit.")
    add_cluster_submit_flags(parser)
    add_runtime_flags(parser)
    parser.set_defaults(default_models=list(DEFAULT_LOCAL_GENERATION_MODELS))
    parser.set_defaults(default_dataset_types=list(DEFAULT_DATASET_TYPES))
    parser.set_defaults(handler=_run_submit_generate_cluster)
    return parser


def _add_submit_evaluate_cluster_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser], formatter) -> argparse.ArgumentParser:
    parser = sub.add_parser(
        "submit-evaluate-cluster",
        help="Submit dependency-aware evaluation jobs over model×dataset×setting×mode×chunk slices.",
        description="Generate per-slice SLURM submissions for local and API-backed evaluation, keyed to exact generation prerequisites.",
        formatter_class=formatter,
    )
    parser.add_argument("--run-name", required=True, help="Logical run name used to organize generated manifests and evaluation logs.")
    parser.add_argument("--generator-run-name", required=True, help="Generation run name whose augmented outputs the cluster jobs should evaluate.")
    parser.add_argument("--generator-model", required=True, help="Generation model whose outputs the cluster jobs should evaluate.")
    parser.add_argument("--generator-backend", default=None, help="Situational: backend prefix to apply when resolving --generator-model.")
    parser.add_argument("--settings", default=None, help="Comma-separated subset of Augmented MCQA settings to schedule.")
    parser.add_argument("--modes", default=None, help="Comma-separated subset of evaluation modes to schedule.")
    parser.add_argument("--augmented-dataset", default=None, help="Advanced override: exact setting-scoped augmented store path to schedule directly instead of deriving one from generation logs.")
    parser.add_argument("--support-root", default=str(DEFAULT_SUPPORT_SET_ROOT), help="Advanced override: root directory where generation support manifests are stored.")
    parser.add_argument("--collected-root", default=str(DEFAULT_COLLECTED_DATASET_ROOT), help="Root directory where collected evaluation datasets should be refreshed after cluster evaluation completes.")
    add_cluster_submit_flags(parser)
    add_runtime_flags(parser)
    parser.set_defaults(default_models=list(DEFAULT_LOCAL_EVALUATION_MODELS))
    parser.set_defaults(default_dataset_types=list(DEFAULT_DATASET_TYPES))
    parser.set_defaults(handler=_run_submit_evaluate_cluster)
    return parser


def build_parser() -> argparse.ArgumentParser:
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(
        description="Inspect-first Augmented MCQA pipeline",
        formatter_class=formatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_prepare_parser(sub, formatter)
    _add_materialize_store_parser(sub, formatter)
    _add_collect_evaluated_parser(sub, formatter)
    _add_generate_parser(sub, formatter)
    _add_evaluate_parser(sub, formatter)
    _add_analyze_parser(sub, formatter)
    _add_analyze_irt_parser(sub, formatter)
    _add_export_parser(sub, formatter)
    _add_submit_generate_cluster_parser(sub, formatter)
    _add_submit_evaluate_cluster_parser(sub, formatter)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
