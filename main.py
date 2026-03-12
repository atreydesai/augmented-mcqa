from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspect_ai import eval as inspect_eval

from analysis.analyzer import analyze_experiment, format_signature_table
from analysis.visualize import plot_final5_pairwise, write_final5_summary_table
from data import export_benchmarker_items, prepare_data
from data.final5_store import _load_dataset_dict, build_evaluation_dataset, build_generation_dataset, ensure_augmented_dataset
from tasks import build_evaluation_tasks, build_generation_tasks
from utils.cluster_submit import (
    ClusterTask,
    build_bundle_paths,
    render_finalizer_wrapper_script,
    render_manifest,
    render_submit_script,
    render_wrapper_script,
    submit_bundle,
    write_bundle,
)
from utils.constants import (
    DEFAULT_AUGMENTED_CACHE_ROOT,
    DEFAULT_EVALUATION_LOG_ROOT,
    DEFAULT_EVALUATION_MODELS,
    DEFAULT_GENERATION_LOG_ROOT,
    DEFAULT_GENERATION_MODELS,
    DEFAULT_LOCAL_EVALUATION_MODELS,
    DEFAULT_LOCAL_GENERATION_MODELS,
    DEFAULT_PROCESSED_DATASET,
    FINAL5_SETTINGS,
    MODE_CHOICES,
)
from utils.logs import iter_eval_logs
from utils.modeling import resolve_model_name, safe_name
from utils.scheduler_state import (
    EVALUATION_SETTING_DEPENDENCIES,
    GENERATION_STRATEGY_DEPENDENCIES,
    SCHEDULABLE_GENERATION_STRATEGIES,
    build_scheduler_state,
    chunk_ranges,
    collect_slice_attempts,
    evaluation_slice_ref,
    generation_slice_ref,
    load_scheduler_manifests,
    render_scheduler_dashboard,
    resource_class_for_model,
    task_slug,
)

REQUIRED_GENERATION_COLUMNS = [
    ("model_from_scratch", 3),
    ("augment_human", 6),
    ("augment_model", 9),
    ("augment_ablation", 9),
]


def _csv_list(raw: str | None, *, default: list[str]) -> list[str]:
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _generation_log_dir(root: Path, run_name: str, model: str) -> Path:
    return root / safe_name(run_name) / safe_name(model)


def _augmented_cache_dir(root: Path, run_name: str, model: str) -> Path:
    return root / safe_name(run_name) / safe_name(model)


def _evaluation_log_dir(root: Path, run_name: str, generator_run: str, generator_model: str, eval_model: str) -> Path:
    return root / safe_name(run_name) / safe_name(generator_run) / safe_name(generator_model) / safe_name(eval_model)


def _cluster_augmented_dataset_dir(
    root: Path,
    run_name: str,
    model: str,
    dataset_type: str,
    *,
    strategy: str,
    question_start: int,
    question_end: int,
) -> Path:
    return (
        root
        / safe_name(run_name)
        / safe_name(model)
        / "_cluster_slices"
        / safe_name(dataset_type)
        / safe_name(strategy)
        / f"{question_start}-{question_end}"
    )


def _cluster_dataset_types(processed_dataset_path: Path, dataset_types: list[str]) -> list[str]:
    dataset_dict = _load_dataset_dict(processed_dataset_path)
    sizes = {dataset_type: len(dataset_dict[dataset_type]) if dataset_type in dataset_dict else 0 for dataset_type in dataset_types}
    indexed = {dataset_type: index for index, dataset_type in enumerate(dataset_types)}
    return sorted(dataset_types, key=lambda dataset_type: (-sizes.get(dataset_type, 0), indexed[dataset_type]))


def _dataset_sizes(
    processed_dataset_path: Path,
    dataset_types: list[str],
    *,
    limit: int | None = None,
) -> dict[str, int]:
    dataset_dict = _load_dataset_dict(processed_dataset_path)
    sizes: dict[str, int] = {}
    for dataset_type in dataset_types:
        size = len(dataset_dict[dataset_type]) if dataset_type in dataset_dict else 0
        if limit is not None and limit >= 0:
            size = min(size, limit)
        sizes[dataset_type] = size
    return sizes


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


def _current_stage_state(*, stage: str, run_name: str, output_dir: str | None = None) -> dict[str, object]:
    paths = build_bundle_paths(stage=stage, run_name=run_name, output_dir=output_dir)
    manifests = load_scheduler_manifests(paths.run_dir)
    log_root = Path(DEFAULT_GENERATION_LOG_ROOT if stage == "generate" else DEFAULT_EVALUATION_LOG_ROOT) / safe_name(run_name)
    kind = "generation" if stage == "generate" else "evaluation"
    attempts = collect_slice_attempts(log_root, kind=kind) if log_root.exists() else {}
    return build_scheduler_state(manifests=manifests, attempts_by_slice=attempts)


def _write_scheduler_outputs(*, stage: str, run_name: str, output_dir: str | None, render_status: bool) -> tuple[Path, Path]:
    paths = build_bundle_paths(stage=stage, run_name=run_name, output_dir=output_dir)
    state = _current_stage_state(stage=stage, run_name=run_name, output_dir=output_dir)
    paths.state_path.parent.mkdir(parents=True, exist_ok=True)
    paths.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    if render_status:
        paths.dashboard_path.write_text(render_scheduler_dashboard(state), encoding="utf-8")
    return paths.state_path, paths.dashboard_path


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


def _build_generation_cluster_tasks(args: argparse.Namespace) -> tuple[list[ClusterTask], dict[str, int | None]]:
    processed_dataset = Path(args.processed_dataset)
    dataset_types = _cluster_dataset_types(
        processed_dataset,
        _csv_list(args.dataset_types, default=args.default_dataset_types),
    )
    dataset_sizes = _dataset_sizes(processed_dataset, dataset_types, limit=args.limit)
    models = _cluster_models(args.models, default=list(args.default_models), backend=args.backend)
    strategies = _csv_list(args.generation_strategies, default=list(SCHEDULABLE_GENERATION_STRATEGIES))
    invalid = [strategy for strategy in strategies if strategy not in SCHEDULABLE_GENERATION_STRATEGIES]
    if invalid:
        raise ValueError("Unsupported generation strategies: " + ", ".join(invalid))

    existing = _slice_status_lookup(_current_stage_state(stage="generate", run_name=args.run_name, output_dir=args.output_dir))
    tasks_by_ref: dict[str, ClusterTask] = {}
    for dataset_type in dataset_types:
        for model in models:
            resource_class = resource_class_for_model(model)
            for chunk_index, question_start, question_end in chunk_ranges(dataset_sizes.get(dataset_type, 0), args.questions_per_job):
                question_limit = question_end - question_start
                for strategy in strategies:
                    slice_ref = generation_slice_ref(
                        run_name=args.run_name,
                        model=model,
                        dataset_type=dataset_type,
                        strategy=strategy,
                        question_start=question_start,
                        question_end=question_end,
                    )
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
                        "--augmented-dataset",
                        str(
                            _cluster_augmented_dataset_dir(
                                Path(DEFAULT_AUGMENTED_CACHE_ROOT),
                                args.run_name,
                                model,
                                dataset_type,
                                strategy=strategy,
                                question_start=question_start,
                                question_end=question_end,
                            )
                        ),
                    ]
                    argv.extend(_runtime_argv(args))
                    tasks_by_ref[slice_ref] = ClusterTask(
                        stage="generate",
                        run_name=args.run_name,
                        model=model,
                        model_slug=safe_name(model),
                        dataset_type=dataset_type,
                        dataset_slug=safe_name(dataset_type),
                        resource_class=resource_class,
                        slice_ref=slice_ref,
                        task_slug=task_slug(
                            stage="generate",
                            model=model,
                            dataset_type=dataset_type,
                            strategy=strategy,
                            question_start=question_start,
                            question_end=question_end,
                        ),
                        question_start=question_start,
                        question_end=question_end,
                        chunk_index=chunk_index,
                        strategy=strategy,
                        state_dependency_refs=state_dependency_refs,
                        submit_dependency_refs=[],
                        argv=argv,
                        resources=_cluster_resources(args),
                        force=bool(args.force),
                    )

    runnable_counts: dict[tuple[str, str, str, int, int], int] = {}

    def runnable_generation_sample_count(task: ClusterTask) -> int:
        key = (task.model, task.dataset_type, str(task.strategy or ""), task.question_start, task.question_end)
        if key not in runnable_counts:
            dataset = build_generation_dataset(
                processed_dataset,
                strategy=str(task.strategy or ""),
                dataset_types=[task.dataset_type],
                question_start=task.question_start,
                limit=task.question_end - task.question_start,
                generation_log_dir=_generation_log_dir(Path(DEFAULT_GENERATION_LOG_ROOT), args.run_name, task.model),
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
                if dependency_status == "failed" and task.strategy == "augment_model":
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
    processed_dataset = Path(args.processed_dataset)
    dataset_types = _cluster_dataset_types(
        processed_dataset,
        _csv_list(args.dataset_types, default=args.default_dataset_types),
    )
    dataset_sizes = _dataset_sizes(processed_dataset, dataset_types, limit=args.limit)
    generation_model = resolve_model_name(args.generator_model, args.generator_backend)
    models = _cluster_models(args.models, default=list(args.default_models), backend=args.backend)
    settings = _csv_list(args.settings, default=list(FINAL5_SETTINGS))
    modes = _csv_list(args.modes, default=list(MODE_CHOICES))

    existing = _slice_status_lookup(_current_stage_state(stage="evaluate", run_name=args.run_name, output_dir=args.output_dir))
    generation_state = _slice_status_lookup(_current_stage_state(stage="generate", run_name=args.generator_run_name))
    evaluation_counts: dict[tuple[str, str, str, int, int], int] = {}
    augmented_cache_dir: Path | None = None

    def runnable_evaluation_sample_count(
        *,
        dataset_type: str,
        setting: str,
        mode: str,
        question_start: int,
        question_end: int,
    ) -> int:
        nonlocal augmented_cache_dir
        key = (dataset_type, setting, mode, question_start, question_end)
        if key not in evaluation_counts:
            if augmented_cache_dir is None:
                generation_log_dir = _generation_log_dir(Path(DEFAULT_GENERATION_LOG_ROOT), args.generator_run_name, generation_model)
                augmented_cache_dir = _augmented_cache_dir(Path(DEFAULT_AUGMENTED_CACHE_ROOT), args.generator_run_name, generation_model)
                ensure_augmented_dataset(
                    processed_dataset_path=processed_dataset,
                    generation_log_dir=generation_log_dir,
                    output_path=augmented_cache_dir,
                    rebuild=True,
                )
            dataset = build_evaluation_dataset(
                augmented_cache_dir,
                setting=setting,
                mode=mode,
                dataset_types=[dataset_type],
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
                        slice_ref = evaluation_slice_ref(
                            run_name=args.run_name,
                            model=model,
                            dataset_type=dataset_type,
                            setting=setting,
                            mode=mode,
                            question_start=question_start,
                            question_end=question_end,
                        )
                        existing_slice = existing.get(slice_ref, {})
                        if not args.force and str(existing_slice.get("status", "")) in {"current", "pending"}:
                            continue

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
                        argv.extend(_runtime_argv(args))
                        tasks.append(
                            ClusterTask(
                                stage="evaluate",
                                run_name=args.run_name,
                                model=model,
                                model_slug=safe_name(model),
                                dataset_type=dataset_type,
                                dataset_slug=safe_name(dataset_type),
                                resource_class=resource_class,
                                slice_ref=slice_ref,
                                task_slug=task_slug(
                                    stage="evaluate",
                                    model=model,
                                    dataset_type=dataset_type,
                                    setting=setting,
                                    mode=mode,
                                    question_start=question_start,
                                    question_end=question_end,
                                ),
                                question_start=question_start,
                                question_end=question_end,
                                chunk_index=chunk_index,
                                setting=setting,
                                mode=mode,
                                state_dependency_refs=state_dependency_refs,
                                submit_dependency_refs=[],
                                argv=argv,
                                resources=_cluster_resources(args),
                                force=bool(args.force),
                                generation_run_name=args.generator_run_name,
                                generation_model=generation_model,
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
    render_status: bool,
) -> int:
    if not tasks:
        print("No cluster tasks selected.")
        if dry_run:
            return 0
        state_path, dashboard_path = _write_scheduler_outputs(
            stage=stage,
            run_name=run_name,
            output_dir=output_dir,
            render_status=render_status,
        )
        print(state_path)
        if render_status:
            print(dashboard_path)
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
    wrapper_text = render_wrapper_script()
    finalizer_wrapper_text = render_finalizer_wrapper_script()

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

    state_path, dashboard_path = _write_scheduler_outputs(
        stage=stage,
        run_name=run_name,
        output_dir=output_dir,
        render_status=render_status,
    )
    print(state_path)
    if render_status:
        print(dashboard_path)

    if not submit:
        return 0

    try:
        result = submit_bundle(paths)
    except OSError as exc:
        print(str(exc))
        return 1
    _write_scheduler_outputs(
        stage=stage,
        run_name=run_name,
        output_dir=output_dir,
        render_status=render_status,
    )
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr.strip())
        return int(result.returncode)
    return 0


def _inspect_eval(tasks, *, model: str, log_dir: Path, args: argparse.Namespace):
    log_dir.mkdir(parents=True, exist_ok=True)
    return inspect_eval(
        tasks,
        model=model,
        model_base_url=args.model_base_url,
        log_dir=str(log_dir),
        display="plain",
        fail_on_error=False,
        retry_on_error=args.retry_on_error,
        max_connections=args.max_connections,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        stop_seqs=args.stop_seqs,
    )


def _prepare_data(args: argparse.Namespace) -> int:
    download_all = bool(args.all or args.step == "all")
    return prepare_data(
        step=args.step,
        dataset=args.dataset,
        download_all=download_all,
        output_dir=args.output_dir,
        output_path=args.output_path,
        limit=args.limit,
    )


def _run_generate(args: argparse.Namespace) -> int:
    dataset_types = _csv_list(args.dataset_types, default=args.default_dataset_types)
    raw_model = resolve_model_name(args.model, args.backend)
    log_dir = _generation_log_dir(Path(args.log_root), args.run_name, raw_model)
    strategies = _csv_list(args.generation_strategies, default=list(SCHEDULABLE_GENERATION_STRATEGIES))
    for strategy in strategies:
        if strategy not in SCHEDULABLE_GENERATION_STRATEGIES:
            raise ValueError(f"Unsupported generation strategy: {strategy}")

    task_metadata_by_strategy: dict[str, dict[str, object]] = {}
    if len(dataset_types) == 1:
        dataset_type = dataset_types[0]
        question_start = int(getattr(args, "question_start", 0) or 0)
        question_limit = int(args.limit or 0)
        question_end = question_start + question_limit if question_limit > 0 else question_start
        for strategy in strategies:
            if question_limit <= 0:
                continue
            task_metadata_by_strategy[strategy] = {
                "slice_ref": generation_slice_ref(
                    run_name=args.run_name,
                    model=raw_model,
                    dataset_type=dataset_type,
                    strategy=strategy,
                    question_start=question_start,
                    question_end=question_end,
                ),
                "question_end": question_end,
                "task_slug": task_slug(
                    stage="generate",
                    model=raw_model,
                    dataset_type=dataset_type,
                    strategy=strategy,
                    question_start=question_start,
                    question_end=question_end,
                ),
            }

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
            generation_model=raw_model,
            generation_log_dir=log_dir,
            task_metadata_by_strategy=task_metadata_by_strategy,
        )
        if not tasks:
            continue
        any_tasks = True
        _inspect_eval(tasks, model=raw_model, log_dir=log_dir, args=args)

    if not any_tasks:
        print("No generation samples selected.")
        return 0
    print(f"Generation logs: {log_dir}")
    if args.materialize_cache or args.shard_count == 1:
        cache_dir = (
            Path(args.augmented_dataset)
            if args.augmented_dataset
            else _augmented_cache_dir(Path(args.cache_root), args.run_name, raw_model)
        )
        ensure_augmented_dataset(
            processed_dataset_path=Path(args.processed_dataset),
            generation_log_dir=log_dir,
            output_path=cache_dir,
            dataset_types=dataset_types,
            rebuild=args.rebuild_cache,
        )
        print(f"Augmented dataset cache: {cache_dir}")
    return 0


def _run_generate_all(args: argparse.Namespace) -> int:
    for model in _csv_list(args.models, default=list(DEFAULT_GENERATION_MODELS)):
        child = argparse.Namespace(**vars(args))
        child.model = model
        rc = _run_generate(child)
        if rc != 0:
            return rc
    return 0


def _resolve_generation_artifacts(args: argparse.Namespace) -> tuple[Path, Path]:
    generation_model = resolve_model_name(args.generator_model, args.generator_backend)
    generation_log_dir = (
        Path(args.generation_log_dir)
        if args.generation_log_dir
        else _generation_log_dir(Path(args.generation_log_root), args.generator_run_name, generation_model)
    )
    cache_dir = (
        Path(args.augmented_dataset)
        if args.augmented_dataset
        else _augmented_cache_dir(Path(args.cache_root), args.generator_run_name, generation_model)
    )
    ensure_augmented_dataset(
        processed_dataset_path=Path(args.processed_dataset),
        generation_log_dir=generation_log_dir,
        output_path=cache_dir,
        dataset_types=_csv_list(args.dataset_types, default=args.default_dataset_types),
        rebuild=args.rebuild_cache,
    )
    return generation_log_dir, cache_dir


def _run_evaluate(args: argparse.Namespace) -> int:
    dataset_types = _csv_list(args.dataset_types, default=args.default_dataset_types)
    settings = _csv_list(args.settings, default=list(FINAL5_SETTINGS))
    modes = _csv_list(args.modes, default=list(MODE_CHOICES))
    eval_model = resolve_model_name(args.model, args.backend)
    _generation_log_dir, cache_dir = _resolve_generation_artifacts(args)
    log_dir = _evaluation_log_dir(
        Path(args.log_root),
        args.run_name,
        args.generator_run_name,
        resolve_model_name(args.generator_model, args.generator_backend),
        eval_model,
    )
    tasks = build_evaluation_tasks(
        augmented_dataset_path=cache_dir,
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
        generation_model=resolve_model_name(args.generator_model, args.generator_backend),
        evaluation_model=eval_model,
        task_metadata_by_setting_mode={
            (setting, mode): {
                "slice_ref": evaluation_slice_ref(
                    run_name=args.run_name,
                    model=eval_model,
                    dataset_type=dataset_types[0],
                    setting=setting,
                    mode=mode,
                    question_start=args.question_start,
                    question_end=args.question_start + int(args.limit or 0),
                ),
                "question_end": args.question_start + int(args.limit or 0),
                "task_slug": task_slug(
                    stage="evaluate",
                    model=eval_model,
                    dataset_type=dataset_types[0],
                    setting=setting,
                    mode=mode,
                    question_start=args.question_start,
                    question_end=args.question_start + int(args.limit or 0),
                ),
            }
            for setting in settings
            for mode in modes
            if len(dataset_types) == 1 and args.limit
        },
    )
    if not tasks:
        print("No evaluation samples selected.")
        return 0
    _inspect_eval(tasks, model=eval_model, log_dir=log_dir, args=args)
    print(f"Evaluation logs: {log_dir}")
    return 0


def _run_evaluate_all(args: argparse.Namespace) -> int:
    for model in _csv_list(args.models, default=list(DEFAULT_EVALUATION_MODELS)):
        child = argparse.Namespace(**vars(args))
        child.model = model
        rc = _run_evaluate(child)
        if rc != 0:
            return rc
    return 0


def _run_analyze(args: argparse.Namespace) -> int:
    if args.table_output:
        df = write_final5_summary_table(args.results_root, args.table_output)
        print(f"Wrote {len(df)} summary rows to {args.table_output}")
    outputs = plot_final5_pairwise(
        results_root=Path(args.results_root),
        output_dir=Path(args.output_dir),
        include_tables=not args.skip_tables,
    )
    if not outputs:
        print("No evaluation logs found.")
        return 1
    for output in outputs:
        print(output)
    return 0


def _run_signature_table(args: argparse.Namespace) -> int:
    base_dir = Path(args.dir)
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        return 1

    exp_files = sorted(base_dir.glob("**/*.eval"))
    if not exp_files:
        print(f"No experiment results found in {base_dir}")
        return 0

    results = {}
    for exp_file in exp_files:
        try:
            analysis = analyze_experiment(exp_file)
            results[exp_file.parent.name] = {
                "signature": analysis["overall"]["signature"],
                "gold_rate": analysis["overall"]["gold_rate"],
            }
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Failed to analyze {exp_file}: {exc}")

    table = format_signature_table(results, include_counts=True)
    print(table)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table + "\n", encoding="utf-8")
        print(output_path)
    return 0


def _run_export(args: argparse.Namespace) -> int:
    if args.input:
        dataset_path = Path(args.input)
    else:
        _generation_log_dir, dataset_path = _resolve_generation_artifacts(args)
    summary_path = export_benchmarker_items(dataset_path, args.output_root)
    print(summary_path)
    return 0


def _run_materialize_generation_cache(args: argparse.Namespace) -> int:
    raw_model = resolve_model_name(args.model, args.backend)
    log_dir = _generation_log_dir(Path(args.generation_log_root), args.run_name, raw_model)
    output_path = (
        Path(args.output_path)
        if args.output_path
        else _augmented_cache_dir(Path(args.cache_root), args.run_name, raw_model)
    )
    ensure_augmented_dataset(
        processed_dataset_path=Path(args.processed_dataset),
        generation_log_dir=log_dir,
        output_path=output_path,
        dataset_types=_csv_list(args.dataset_types, default=args.default_dataset_types),
        rebuild=args.rebuild_cache,
    )
    print(output_path)
    return 0


def _run_submit_generate_cluster(args: argparse.Namespace) -> int:
    try:
        tasks, concurrency_caps = _build_generation_cluster_tasks(args)
    except ValueError as exc:
        print(str(exc))
        return 1
    return _run_cluster_submit(
        stage="generate",
        run_name=args.run_name,
        tasks=tasks,
        resources=_cluster_resources(args),
        concurrency_caps=concurrency_caps,
        output_dir=args.output_dir,
        submit=args.submit,
        dry_run=args.dry_run,
        render_status=args.render_status,
    )


def _run_submit_evaluate_cluster(args: argparse.Namespace) -> int:
    try:
        tasks, concurrency_caps = _build_evaluation_cluster_tasks(args)
    except ValueError as exc:
        print(str(exc))
        return 1
    return _run_cluster_submit(
        stage="evaluate",
        run_name=args.run_name,
        tasks=tasks,
        resources=_cluster_resources(args),
        concurrency_caps=concurrency_caps,
        output_dir=args.output_dir,
        submit=args.submit,
        dry_run=args.dry_run,
        render_status=args.render_status,
    )


def _run_diagnose_failures(args: argparse.Namespace) -> int:
    ds = _load_dataset_dict(Path(args.dataset_path))
    total_failed = 0
    for split in ds.keys():
        failed = []
        for i, row in enumerate(ds[split]):
            missing = [key for key, count in REQUIRED_GENERATION_COLUMNS if len(row.get(key) or []) < count]
            if missing:
                failed.append(
                    {
                        "idx": i,
                        "id": row.get("id"),
                        "question_id": row.get("question_id"),
                        "missing": missing,
                        "question": str(row.get("question", ""))[:140],
                    }
                )
        print(f"\n{split}: failed_rows={len(failed)}")
        for row in failed:
            print(json.dumps(row, ensure_ascii=True))
        total_failed += len(failed)
    return 1 if total_failed > 0 else 0


def _generation_trace_records(
    *,
    log_dir: str,
    sample_id: str | None,
    only_errors: bool,
    limit: int | None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for log_path, log in iter_eval_logs(log_dir, kind="generation"):
        eval_metadata = dict(getattr(log.eval, "metadata", {}) or {})
        for sample in getattr(log, "samples", []):
            if not sample.scores:
                continue
            score = next(iter(sample.scores.values()))
            metadata = dict(getattr(score, "metadata", {}) or {})
            if not metadata:
                continue
            current_sample_id = str(getattr(sample, "id", ""))
            status = str(metadata.get("status", ""))
            if sample_id and current_sample_id != sample_id:
                continue
            if only_errors and status != "error":
                continue
            records.append(
                {
                    "log_path": str(log_path),
                    "run_name": eval_metadata.get("run_name"),
                    "generation_model": eval_metadata.get("generation_model"),
                    "sample_id": current_sample_id,
                    "status": status,
                    "error": metadata.get("error"),
                    "dataset_type": metadata.get("dataset_type"),
                    "row_index": metadata.get("row_index"),
                    "question": metadata.get("question"),
                    "answer": metadata.get("answer"),
                    "category": metadata.get("category"),
                    "traces": metadata.get("traces", {}),
                    "human_from_scratch": metadata.get("human_from_scratch", []),
                    "model_from_scratch": metadata.get("model_from_scratch", []),
                    "augment_human": metadata.get("augment_human", []),
                    "augment_model": metadata.get("augment_model", []),
                    "augment_ablation": metadata.get("augment_ablation", []),
                }
            )
            if limit is not None and len(records) >= limit:
                return records
    return records


def _run_diagnose_trace(args: argparse.Namespace) -> int:
    records = _generation_trace_records(
        log_dir=args.log_dir,
        sample_id=args.sample_id,
        only_errors=args.only_errors,
        limit=args.limit,
    )
    if not records:
        print("No generation traces matched the requested filters.")
        return 1

    if args.summary:
        for record in records:
            print(
                json.dumps(
                    {
                        "sample_id": record["sample_id"],
                        "status": record["status"],
                        "dataset_type": record["dataset_type"],
                        "error": record["error"],
                        "log_path": record["log_path"],
                    },
                    ensure_ascii=True,
                )
            )
        return 0

    payload = json.dumps(records, indent=2, ensure_ascii=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
        print(output_path)
    else:
        print(payload)
    return 0


def _run_smoke_generate(args: argparse.Namespace) -> int:
    dataset_path = Path(args.processed_dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}")

    for model in args.models:
        argv = [
            "generate",
            "--model",
            model,
            "--run-name",
            args.run_name,
            "--processed-dataset",
            str(dataset_path),
            "--limit",
            str(args.limit),
            "--log-root",
            args.log_root,
            "--cache-root",
            args.cache_root,
            "--materialize-cache",
            "--max-tokens",
            str(args.max_tokens),
        ]
        if args.dataset_types:
            argv.extend(["--dataset-types", args.dataset_types])
        if args.backend:
            argv.extend(["--backend", args.backend])
        if args.model_base_url:
            argv.extend(["--model-base-url", args.model_base_url])
        if args.reasoning_effort:
            argv.extend(["--reasoning-effort", args.reasoning_effort])
        print("python main.py " + " ".join(argv))
        if args.dry_run:
            continue
        rc = main(argv)
        if rc != 0:
            return rc
    return 0


def _run_smoke_evaluate(args: argparse.Namespace) -> int:
    for model in args.models:
        argv = [
            "evaluate",
            "--model",
            model,
            "--run-name",
            args.run_name,
            "--generator-run-name",
            args.generator_run_name,
            "--generator-model",
            args.generator_model,
            "--processed-dataset",
            args.processed_dataset,
            "--generation-log-root",
            args.generation_log_root,
            "--cache-root",
            args.cache_root,
            "--log-root",
            args.log_root,
            "--limit",
            str(args.limit),
            "--max-tokens",
            str(args.max_tokens),
        ]
        if args.settings:
            argv.extend(["--settings", args.settings])
        if args.modes:
            argv.extend(["--modes", args.modes])
        if args.dataset_types:
            argv.extend(["--dataset-types", args.dataset_types])
        if args.generator_backend:
            argv.extend(["--generator-backend", args.generator_backend])
        if args.backend:
            argv.extend(["--backend", args.backend])
        if args.model_base_url:
            argv.extend(["--model-base-url", args.model_base_url])
        if args.reasoning_effort:
            argv.extend(["--reasoning-effort", args.reasoning_effort])
        print("python main.py " + " ".join(argv))
        if args.dry_run:
            continue
        rc = main(argv)
        if rc != 0:
            return rc
    return 0


def build_parser() -> argparse.ArgumentParser:
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(
        description="Inspect-first Final5 pipeline",
        formatter_class=formatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser(
        "prepare-data",
        help="Download raw datasets and/or build the unified processed dataset.",
        description="Download raw source datasets and/or process them into the unified Final5 dataset.",
        formatter_class=formatter,
    )
    prepare.add_argument(
        "--step",
        choices=["download", "process", "all"],
        default="all",
        help="Which stage of data preparation to run.",
    )
    prepare.add_argument(
        "--dataset",
        choices=["mmlu_pro", "mmlu", "arc", "gpqa"],
        default=None,
        help="Specific raw dataset to download when not using --all.",
    )
    prepare.add_argument(
        "--all",
        action="store_true",
        help="Download every supported raw dataset instead of a single dataset.",
    )
    prepare.add_argument(
        "--output-dir",
        default="datasets/raw",
        help="Advanced override: directory where raw downloaded datasets should be stored.",
    )
    prepare.add_argument(
        "--output-path",
        default=str(DEFAULT_PROCESSED_DATASET),
        help="Directory where the processed unified DatasetDict should be written.",
    )
    prepare.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Advanced/debug option: optional per-dataset cap when building the processed unified dataset.",
    )
    prepare.set_defaults(handler=_prepare_data)

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
            default=2048,
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
            "--render-status",
            action="store_true",
            help="Write a scheduler HTML dashboard for this run after generating the new submission bundle.",
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

    generate = sub.add_parser(
        "generate",
        help="Run Final5 distractor generation for one model.",
        description="Generate Final5 distractor variants for one model over the processed dataset.",
        formatter_class=formatter,
    )
    generate.add_argument("--model", required=True, help="Model name or alias to use for generation.")
    generate.add_argument("--run-name", required=True, help="Logical run name used to organize logs and caches.")
    generate.add_argument(
        "--processed-dataset",
        default=str(DEFAULT_PROCESSED_DATASET),
        help="Processed unified DatasetDict to read input questions from.",
    )
    generate.add_argument(
        "--dataset-types",
        default=None,
        help="Optional subset: comma-separated subset of dataset splits to generate for.",
    )
    generate.add_argument(
        "--generation-strategies",
        default=None,
        help="Advanced subset override: comma-separated subset of schedulable generation strategies to run.",
    )
    generate.add_argument(
        "--question-start",
        type=int,
        default=0,
        help="Advanced/debug option: zero-based per-dataset starting row for generation.",
    )
    generate.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Advanced/debug option: optional per-dataset cap on the number of samples to generate.",
    )
    generate.add_argument(
        "--log-root",
        default=str(DEFAULT_GENERATION_LOG_ROOT),
        help="Advanced override: root directory for Inspect generation logs.",
    )
    generate.add_argument(
        "--cache-root",
        default=str(DEFAULT_AUGMENTED_CACHE_ROOT),
        help="Advanced override: root directory where derived augmented dataset caches should be stored.",
    )
    generate.add_argument(
        "--augmented-dataset",
        default=None,
        help="Advanced override: exact output path for the augmented cache produced from generation logs.",
    )
    generate.add_argument(
        "--materialize-cache",
        action="store_true",
        help="Rebuild the augmented DatasetDict cache immediately after generation completes.",
    )
    generate.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Advanced override: force regeneration of the augmented cache even if it already exists.",
    )
    generate.set_defaults(default_dataset_types=["arc_challenge", "mmlu_pro", "gpqa"])
    add_runtime_flags(generate)
    add_shard_flags(generate)
    generate.set_defaults(handler=_run_generate)

    generate_all = sub.add_parser(
        "generate-all",
        help="Run generation for the default generator model set.",
        description="Run Final5 distractor generation for every default generation model.",
        formatter_class=formatter,
    )
    generate_all.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of generation models to override the default API generation model set.",
    )
    generate_all.add_argument("--run-name", required=True, help="Logical run name used to organize logs and caches.")
    generate_all.add_argument(
        "--processed-dataset",
        default=str(DEFAULT_PROCESSED_DATASET),
        help="Processed unified DatasetDict to read input questions from.",
    )
    generate_all.add_argument(
        "--dataset-types",
        default=None,
        help="Optional subset: comma-separated subset of dataset splits to generate for.",
    )
    generate_all.add_argument(
        "--generation-strategies",
        default=None,
        help="Advanced subset override: comma-separated subset of schedulable generation strategies to run.",
    )
    generate_all.add_argument(
        "--question-start",
        type=int,
        default=0,
        help="Advanced/debug option: zero-based per-dataset starting row for generation.",
    )
    generate_all.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Advanced/debug option: optional per-dataset cap on the number of samples for each model.",
    )
    generate_all.add_argument(
        "--log-root",
        default=str(DEFAULT_GENERATION_LOG_ROOT),
        help="Advanced override: root directory for Inspect generation logs.",
    )
    generate_all.add_argument(
        "--cache-root",
        default=str(DEFAULT_AUGMENTED_CACHE_ROOT),
        help="Advanced override: root directory where derived augmented dataset caches should be stored.",
    )
    generate_all.add_argument(
        "--materialize-cache",
        action="store_true",
        help="Rebuild each augmented DatasetDict cache immediately after generation completes.",
    )
    generate_all.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Advanced override: force regeneration of each augmented cache even if it already exists.",
    )
    generate_all.set_defaults(default_dataset_types=["arc_challenge", "mmlu_pro", "gpqa"])
    add_runtime_flags(generate_all)
    add_shard_flags(generate_all)
    generate_all.set_defaults(handler=_run_generate_all)

    evaluate = sub.add_parser(
        "evaluate",
        help="Evaluate one model against one generation run.",
        description="Evaluate a single model across the requested Final5 settings and modes.",
        formatter_class=formatter,
    )
    evaluate.add_argument("--model", required=True, help="Model name or alias to use for evaluation.")
    evaluate.add_argument("--run-name", required=True, help="Logical run name used to organize evaluation logs.")
    evaluate.add_argument(
        "--generator-run-name",
        required=True,
        help="Generation run name whose augmented cache or logs should be evaluated.",
    )
    evaluate.add_argument(
        "--generator-model",
        required=True,
        help="Generation model whose outputs should be evaluated.",
    )
    evaluate.add_argument(
        "--generator-backend",
        default=None,
        help="Situational: backend prefix to apply when resolving --generator-model.",
    )
    evaluate.add_argument(
        "--generation-log-dir",
        default=None,
        help="Advanced override: exact generation log directory to read instead of deriving one from run name and model.",
    )
    evaluate.add_argument(
        "--generation-log-root",
        default=str(DEFAULT_GENERATION_LOG_ROOT),
        help="Advanced override: root directory for generation Inspect logs when deriving inputs automatically.",
    )
    evaluate.add_argument(
        "--processed-dataset",
        default=str(DEFAULT_PROCESSED_DATASET),
        help="Processed unified DatasetDict used if the augmented cache must be rebuilt from generation logs.",
    )
    evaluate.add_argument(
        "--augmented-dataset",
        default=None,
        help="Advanced override: exact augmented DatasetDict path to evaluate instead of deriving one from generation artifacts.",
    )
    evaluate.add_argument(
        "--cache-root",
        default=str(DEFAULT_AUGMENTED_CACHE_ROOT),
        help="Advanced override: root directory where augmented dataset caches are stored.",
    )
    evaluate.add_argument(
        "--dataset-types",
        default=None,
        help="Optional subset: comma-separated subset of dataset splits to evaluate.",
    )
    evaluate.add_argument(
        "--question-start",
        type=int,
        default=0,
        help="Advanced/debug option: zero-based per-dataset starting row for evaluation.",
    )
    evaluate.add_argument(
        "--settings",
        default=None,
        help="Advanced subset override: comma-separated subset of Final5 settings to evaluate.",
    )
    evaluate.add_argument(
        "--modes",
        default=None,
        help="Advanced subset override: comma-separated subset of evaluation modes to run.",
    )
    evaluate.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Advanced/debug option: optional per-dataset cap on the number of evaluation samples.",
    )
    evaluate.add_argument(
        "--log-root",
        default=str(DEFAULT_EVALUATION_LOG_ROOT),
        help="Advanced override: root directory for Inspect evaluation logs.",
    )
    evaluate.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Advanced override: force regeneration of the augmented cache before evaluation.",
    )
    evaluate.set_defaults(default_dataset_types=["arc_challenge", "mmlu_pro", "gpqa"])
    add_runtime_flags(evaluate)
    add_shard_flags(evaluate)
    evaluate.set_defaults(handler=_run_evaluate)

    evaluate_all = sub.add_parser(
        "evaluate-all",
        help="Evaluate the default local evaluation model set against one generation run.",
        description="Evaluate every default local evaluation model across the requested Final5 settings and modes.",
        formatter_class=formatter,
    )
    evaluate_all.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of evaluation models to override the default local evaluation model set.",
    )
    evaluate_all.add_argument("--run-name", required=True, help="Logical run name used to organize evaluation logs.")
    evaluate_all.add_argument(
        "--generator-run-name",
        required=True,
        help="Generation run name whose augmented cache or logs should be evaluated.",
    )
    evaluate_all.add_argument(
        "--generator-model",
        required=True,
        help="Generation model whose outputs should be evaluated.",
    )
    evaluate_all.add_argument(
        "--generator-backend",
        default=None,
        help="Situational: backend prefix to apply when resolving --generator-model.",
    )
    evaluate_all.add_argument(
        "--generation-log-dir",
        default=None,
        help="Advanced override: exact generation log directory to read instead of deriving one from run name and model.",
    )
    evaluate_all.add_argument(
        "--generation-log-root",
        default=str(DEFAULT_GENERATION_LOG_ROOT),
        help="Advanced override: root directory for generation Inspect logs when deriving inputs automatically.",
    )
    evaluate_all.add_argument(
        "--processed-dataset",
        default=str(DEFAULT_PROCESSED_DATASET),
        help="Processed unified DatasetDict used if the augmented cache must be rebuilt from generation logs.",
    )
    evaluate_all.add_argument(
        "--augmented-dataset",
        default=None,
        help="Advanced override: exact augmented DatasetDict path to evaluate instead of deriving one from generation artifacts.",
    )
    evaluate_all.add_argument(
        "--cache-root",
        default=str(DEFAULT_AUGMENTED_CACHE_ROOT),
        help="Advanced override: root directory where augmented dataset caches are stored.",
    )
    evaluate_all.add_argument(
        "--dataset-types",
        default=None,
        help="Optional subset: comma-separated subset of dataset splits to evaluate.",
    )
    evaluate_all.add_argument(
        "--question-start",
        type=int,
        default=0,
        help="Advanced/debug option: zero-based per-dataset starting row for evaluation.",
    )
    evaluate_all.add_argument(
        "--settings",
        default=None,
        help="Advanced subset override: comma-separated subset of Final5 settings to evaluate.",
    )
    evaluate_all.add_argument(
        "--modes",
        default=None,
        help="Advanced subset override: comma-separated subset of evaluation modes to run.",
    )
    evaluate_all.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Advanced/debug option: optional per-dataset cap on the number of evaluation samples for each model.",
    )
    evaluate_all.add_argument(
        "--log-root",
        default=str(DEFAULT_EVALUATION_LOG_ROOT),
        help="Advanced override: root directory for Inspect evaluation logs.",
    )
    evaluate_all.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Advanced override: force regeneration of the augmented cache before evaluation.",
    )
    evaluate_all.set_defaults(default_dataset_types=["arc_challenge", "mmlu_pro", "gpqa"])
    add_runtime_flags(evaluate_all)
    add_shard_flags(evaluate_all)
    evaluate_all.set_defaults(handler=_run_evaluate_all)

    analyze = sub.add_parser(
        "analyze",
        help="Aggregate evaluation logs into Final5 plots and summary tables.",
        description="Read Inspect evaluation logs and produce Final5 CSV summaries and comparison plots.",
        formatter_class=formatter,
    )
    analyze.add_argument(
        "--results-root",
        default=str(DEFAULT_EVALUATION_LOG_ROOT),
        help="Usually leave alone: directory containing Inspect evaluation logs to analyze.",
    )
    analyze.add_argument(
        "--output-dir",
        default="results/final5_plots",
        help="Situational output override: directory where plots and optional tables should be written.",
    )
    analyze.add_argument(
        "--table-output",
        default="results/final5_plots/tables/final5_results_summary.csv",
        help="Advanced output override: CSV path for the flat summary table.",
    )
    analyze.add_argument(
        "--skip-tables",
        action="store_true",
        help="Advanced output option: write plots only and skip the pairwise comparison CSV tables.",
    )
    analyze.set_defaults(handler=_run_analyze)

    signatures = sub.add_parser(
        "signature-table",
        help="Build a compact behavioral-signature table from .eval logs.",
        description="Read one or more Inspect .eval logs and print a markdown signature table.",
        formatter_class=formatter,
    )
    signatures.add_argument("--dir", required=True, help="Directory or .eval file to summarize.")
    signatures.add_argument(
        "--output",
        default=None,
        help="Advanced output override: optional file path to write the rendered signature table.",
    )
    signatures.set_defaults(handler=_run_signature_table)

    export = sub.add_parser(
        "export",
        help="Export an augmented DatasetDict to benchmarker JSONL files.",
        description="Export a derived augmented dataset cache into benchmarker-compatible JSONL files.",
        formatter_class=formatter,
    )
    export.add_argument(
        "--input",
        default=None,
        help="Advanced override: exact augmented DatasetDict path to export. If omitted, generation artifacts are resolved automatically.",
    )
    export.add_argument(
        "--output-root",
        default="datasets/benchmarker_items",
        help="Situational output override: root directory where benchmarker JSONL outputs should be written.",
    )
    export.add_argument(
        "--generator-run-name",
        default=None,
        help="Generation run name to use when deriving the augmented cache automatically.",
    )
    export.add_argument(
        "--generator-model",
        default=None,
        help="Generation model to use when deriving the augmented cache automatically.",
    )
    export.add_argument(
        "--generator-backend",
        default=None,
        help="Situational: backend prefix to apply when resolving --generator-model.",
    )
    export.add_argument(
        "--generation-log-dir",
        default=None,
        help="Advanced override: exact generation log directory to read instead of deriving one from run name and model.",
    )
    export.add_argument(
        "--generation-log-root",
        default=str(DEFAULT_GENERATION_LOG_ROOT),
        help="Advanced override: root directory for generation Inspect logs when deriving inputs automatically.",
    )
    export.add_argument(
        "--processed-dataset",
        default=str(DEFAULT_PROCESSED_DATASET),
        help="Processed unified DatasetDict used if the augmented cache must be rebuilt from generation logs.",
    )
    export.add_argument(
        "--augmented-dataset",
        default=None,
        help="Advanced override: exact augmented DatasetDict path to use instead of deriving one from generation artifacts.",
    )
    export.add_argument(
        "--cache-root",
        default=str(DEFAULT_AUGMENTED_CACHE_ROOT),
        help="Advanced override: root directory where augmented dataset caches are stored.",
    )
    export.add_argument(
        "--dataset-types",
        default=None,
        help="Advanced subset override: comma-separated subset of dataset splits to include when rebuilding the augmented cache.",
    )
    export.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Advanced override: force regeneration of the augmented cache before export.",
    )
    export.set_defaults(default_dataset_types=["arc_challenge", "mmlu_pro", "gpqa"])
    export.set_defaults(handler=_run_export)

    materialize_generation_cache = sub.add_parser(
        "materialize-generation-cache",
        help="Rebuild or refresh an augmented cache from generation logs.",
        description="Materialize the merged augmented DatasetDict for one generation run/model directly from Inspect generation logs.",
        formatter_class=formatter,
    )
    materialize_generation_cache.add_argument(
        "--run-name",
        required=True,
        help="Generation run name whose Inspect logs should be merged.",
    )
    materialize_generation_cache.add_argument(
        "--model",
        required=True,
        help="Generation model whose Inspect logs should be merged.",
    )
    materialize_generation_cache.add_argument(
        "--backend",
        default=None,
        help="Situational: provider prefix to apply to an unqualified model name.",
    )
    materialize_generation_cache.add_argument(
        "--generation-log-root",
        default=str(DEFAULT_GENERATION_LOG_ROOT),
        help="Advanced override: root directory containing generation Inspect logs.",
    )
    materialize_generation_cache.add_argument(
        "--processed-dataset",
        default=str(DEFAULT_PROCESSED_DATASET),
        help="Processed unified DatasetDict used to rebuild the augmented cache from logs.",
    )
    materialize_generation_cache.add_argument(
        "--cache-root",
        default=str(DEFAULT_AUGMENTED_CACHE_ROOT),
        help="Advanced override: root directory where augmented caches are stored.",
    )
    materialize_generation_cache.add_argument(
        "--output-path",
        default=None,
        help="Advanced override: exact output path for the rebuilt augmented cache.",
    )
    materialize_generation_cache.add_argument(
        "--dataset-types",
        default=None,
        help="Optional subset: comma-separated subset of dataset splits to materialize.",
    )
    materialize_generation_cache.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Advanced override: force regeneration even if the cache appears up to date.",
    )
    materialize_generation_cache.set_defaults(default_dataset_types=["arc_challenge", "mmlu_pro", "gpqa"])
    materialize_generation_cache.set_defaults(handler=_run_materialize_generation_cache)

    submit_generate_cluster = sub.add_parser(
        "submit-generate-cluster",
        help="Submit dependency-aware generation jobs over model×dataset×strategy×chunk slices.",
        description="Generate per-slice SLURM submissions for local and API-backed generation, with exact dependency wiring where needed.",
        formatter_class=formatter,
    )
    submit_generate_cluster.add_argument(
        "--run-name",
        required=True,
        help="Logical run name used to organize generated manifests, logs, and output caches.",
    )
    submit_generate_cluster.add_argument(
        "--generation-strategies",
        default=None,
        help="Comma-separated subset of schedulable generation strategies to submit. human_from_scratch remains implicit.",
    )
    add_cluster_submit_flags(submit_generate_cluster)
    add_runtime_flags(submit_generate_cluster)
    submit_generate_cluster.set_defaults(default_models=list(DEFAULT_LOCAL_GENERATION_MODELS))
    submit_generate_cluster.set_defaults(default_dataset_types=["arc_challenge", "mmlu_pro", "gpqa"])
    submit_generate_cluster.set_defaults(handler=_run_submit_generate_cluster)

    submit_evaluate_cluster = sub.add_parser(
        "submit-evaluate-cluster",
        help="Submit dependency-aware evaluation jobs over model×dataset×setting×mode×chunk slices.",
        description="Generate per-slice SLURM submissions for local and API-backed evaluation, keyed to exact generation prerequisites.",
        formatter_class=formatter,
    )
    submit_evaluate_cluster.add_argument(
        "--run-name",
        required=True,
        help="Logical run name used to organize generated manifests and evaluation logs.",
    )
    submit_evaluate_cluster.add_argument(
        "--generator-run-name",
        required=True,
        help="Generation run name whose augmented outputs the cluster jobs should evaluate.",
    )
    submit_evaluate_cluster.add_argument(
        "--generator-model",
        required=True,
        help="Generation model whose outputs the cluster jobs should evaluate.",
    )
    submit_evaluate_cluster.add_argument(
        "--generator-backend",
        default=None,
        help="Situational: backend prefix to apply when resolving --generator-model.",
    )
    submit_evaluate_cluster.add_argument(
        "--settings",
        default=None,
        help="Comma-separated subset of Final5 settings to schedule.",
    )
    submit_evaluate_cluster.add_argument(
        "--modes",
        default=None,
        help="Comma-separated subset of evaluation modes to schedule.",
    )
    add_cluster_submit_flags(submit_evaluate_cluster)
    add_runtime_flags(submit_evaluate_cluster)
    submit_evaluate_cluster.set_defaults(default_models=list(DEFAULT_LOCAL_EVALUATION_MODELS))
    submit_evaluate_cluster.set_defaults(default_dataset_types=["arc_challenge", "mmlu_pro", "gpqa"])
    submit_evaluate_cluster.set_defaults(handler=_run_submit_evaluate_cluster)

    diagnose_failures = sub.add_parser(
        "diagnose-failures",
        help="Report rows in an augmented cache that are missing required generated columns.",
        description="Inspect an augmented DatasetDict and print rows with missing Final5 generation outputs.",
        formatter_class=formatter,
    )
    diagnose_failures.add_argument(
        "--dataset-path",
        required=True,
        help="Augmented DatasetDict path to inspect for missing generation outputs.",
    )
    diagnose_failures.set_defaults(handler=_run_diagnose_failures)

    diagnose_trace = sub.add_parser(
        "diagnose-trace",
        help="Extract generation traces from Inspect logs.",
        description="Dump generation prompt/output traces from Inspect generation logs for debugging.",
        formatter_class=formatter,
    )
    diagnose_trace.add_argument("--log-dir", required=True, help="Generation log directory or .eval file to inspect.")
    diagnose_trace.add_argument(
        "--sample-id",
        default=None,
        help="Advanced/debug filter: sample id so only one question's traces are returned.",
    )
    diagnose_trace.add_argument(
        "--only-errors",
        action="store_true",
        help="Restrict output to generation samples whose status is error.",
    )
    diagnose_trace.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Advanced/debug option: maximum number of matching trace records to emit.",
    )
    diagnose_trace.add_argument(
        "--summary",
        action="store_true",
        help="Print a compact one-line summary per trace instead of full JSON payloads.",
    )
    diagnose_trace.add_argument(
        "--output",
        default=None,
        help="Advanced output override: optional JSON file path to write the extracted trace records.",
    )
    diagnose_trace.set_defaults(handler=_run_diagnose_trace)

    smoke_generate = sub.add_parser(
        "smoke-generate",
        help="Run a tiny generation smoke test across one or more models.",
        description="Launch small generation runs that are useful for validating credentials, prompts, and outputs.",
        formatter_class=formatter,
    )
    smoke_generate.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_GENERATION_MODELS),
        help="Space-separated list of generation models to smoke test.",
    )
    smoke_generate.add_argument("--run-name", default="smoke-generate", help="Run name to use for the smoke test.")
    smoke_generate.add_argument(
        "--processed-dataset",
        default=str(DEFAULT_PROCESSED_DATASET),
        help="Processed unified DatasetDict to draw smoke-test samples from.",
    )
    smoke_generate.add_argument(
        "--dataset-types",
        default=None,
        help="Optional subset: comma-separated subset of dataset splits to sample from.",
    )
    smoke_generate.add_argument("--limit", type=int, default=2, help="Number of samples per selected dataset split to run.")
    smoke_generate.add_argument(
        "--log-root",
        default=str(DEFAULT_GENERATION_LOG_ROOT),
        help="Advanced override: root directory for Inspect generation logs.",
    )
    smoke_generate.add_argument(
        "--cache-root",
        default=str(DEFAULT_AUGMENTED_CACHE_ROOT),
        help="Advanced override: root directory where augmented dataset caches should be stored.",
    )
    smoke_generate.add_argument(
        "--backend",
        default=None,
        help="Situational: backend prefix to apply to unqualified smoke-test model names.",
    )
    smoke_generate.add_argument(
        "--model-base-url",
        default=None,
        help="Situational: base URL for OpenAI-compatible local or remote model endpoints.",
    )
    smoke_generate.add_argument(
        "--reasoning-effort",
        default=None,
        help="Advanced tuning: optional reasoning-effort hint forwarded to the model backend.",
    )
    smoke_generate.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum output tokens per smoke-test generation call.",
    )
    smoke_generate.add_argument(
        "--dry-run",
        action="store_true",
        help="Advanced/debug option: print the commands that would be run without actually executing them.",
    )
    smoke_generate.set_defaults(handler=_run_smoke_generate)

    smoke_evaluate = sub.add_parser(
        "smoke-evaluate",
        help="Run a tiny evaluation smoke test across one or more models.",
        description="Launch small evaluation runs that are useful for validating caches, prompts, and model access.",
        formatter_class=formatter,
    )
    smoke_evaluate.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_EVALUATION_MODELS),
        help="Space-separated list of evaluation models to smoke test.",
    )
    smoke_evaluate.add_argument("--run-name", default="smoke-evaluate", help="Run name to use for the smoke test.")
    smoke_evaluate.add_argument(
        "--generator-run-name",
        required=True,
        help="Generation run name whose outputs should be used for the smoke test.",
    )
    smoke_evaluate.add_argument(
        "--generator-model",
        required=True,
        help="Generation model whose outputs should be used for the smoke test.",
    )
    smoke_evaluate.add_argument(
        "--generator-backend",
        default=None,
        help="Situational: backend prefix to apply when resolving --generator-model.",
    )
    smoke_evaluate.add_argument(
        "--processed-dataset",
        default=str(DEFAULT_PROCESSED_DATASET),
        help="Processed unified DatasetDict used if the augmented cache must be rebuilt from generation logs.",
    )
    smoke_evaluate.add_argument(
        "--dataset-types",
        default=None,
        help="Optional subset: comma-separated subset of dataset splits to evaluate.",
    )
    smoke_evaluate.add_argument(
        "--settings",
        default=None,
        help="Advanced subset override: comma-separated subset of Final5 settings to evaluate in the smoke test.",
    )
    smoke_evaluate.add_argument(
        "--modes",
        default=None,
        help="Advanced subset override: comma-separated subset of evaluation modes to run in the smoke test.",
    )
    smoke_evaluate.add_argument("--limit", type=int, default=2, help="Number of samples per selected dataset split to run.")
    smoke_evaluate.add_argument(
        "--generation-log-root",
        default=str(DEFAULT_GENERATION_LOG_ROOT),
        help="Advanced override: root directory for generation Inspect logs when deriving inputs automatically.",
    )
    smoke_evaluate.add_argument(
        "--cache-root",
        default=str(DEFAULT_AUGMENTED_CACHE_ROOT),
        help="Advanced override: root directory where augmented dataset caches are stored.",
    )
    smoke_evaluate.add_argument(
        "--log-root",
        default=str(DEFAULT_EVALUATION_LOG_ROOT),
        help="Advanced override: root directory for Inspect evaluation logs.",
    )
    smoke_evaluate.add_argument(
        "--backend",
        default=None,
        help="Situational: backend prefix to apply to unqualified smoke-test model names.",
    )
    smoke_evaluate.add_argument(
        "--model-base-url",
        default=None,
        help="Situational: base URL for OpenAI-compatible local or remote model endpoints.",
    )
    smoke_evaluate.add_argument(
        "--reasoning-effort",
        default=None,
        help="Advanced tuning: optional reasoning-effort hint forwarded to the model backend.",
    )
    smoke_evaluate.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum output tokens per smoke-test evaluation call.",
    )
    smoke_evaluate.add_argument(
        "--dry-run",
        action="store_true",
        help="Advanced/debug option: print the commands that would be run without actually executing them.",
    )
    smoke_evaluate.set_defaults(handler=_run_smoke_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
