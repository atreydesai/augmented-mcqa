from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspect_ai import eval as inspect_eval

from analysis.analyzer import analyze_experiment, format_signature_table
from analysis.visualize import plot_final5_pairwise, write_final5_summary_table
from data import export_benchmarker_items, prepare_data
from data.final5_store import _load_dataset_dict, ensure_augmented_dataset
from tasks import build_evaluation_tasks, build_generation_tasks
from utils.cluster_submit import (
    ClusterTask,
    build_bundle_paths,
    render_manifest,
    render_sbatch,
    render_submit_script,
    submit_bundle,
    write_bundle,
)
from utils.constants import (
    DEFAULT_AUGMENTED_CACHE_ROOT,
    DEFAULT_EVALUATION_LOG_ROOT,
    DEFAULT_EVALUATION_MODELS,
    DEFAULT_GENERATION_LOG_ROOT,
    DEFAULT_GENERATION_MODELS,
    DEFAULT_LOCAL_CLUSTER_MODELS,
    DEFAULT_PROCESSED_DATASET,
    FINAL5_SETTINGS,
    MODE_CHOICES,
)
from utils.logs import iter_eval_logs
from utils.modeling import resolve_model_name, safe_name

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


def _cluster_augmented_dataset_dir(root: Path, run_name: str, model: str, dataset_type: str) -> Path:
    return root / safe_name(run_name) / safe_name(model) / safe_name(dataset_type)


def _cluster_dataset_types(processed_dataset_path: Path, dataset_types: list[str]) -> list[str]:
    dataset_dict = _load_dataset_dict(processed_dataset_path)
    sizes = {dataset_type: len(dataset_dict[dataset_type]) if dataset_type in dataset_dict else 0 for dataset_type in dataset_types}
    indexed = {dataset_type: index for index, dataset_type in enumerate(dataset_types)}
    return sorted(dataset_types, key=lambda dataset_type: (-sizes.get(dataset_type, 0), indexed[dataset_type]))


def _cluster_models(raw: str | None) -> list[str]:
    models = [resolve_model_name(model) for model in _csv_list(raw, default=list(DEFAULT_LOCAL_CLUSTER_MODELS))]
    if not models:
        raise ValueError("No local models selected.")
    invalid = [model for model in models if not str(model).startswith("vllm/")]
    if invalid:
        raise ValueError(
            "Cluster submit commands only support local vllm/... models. "
            "Use main.py generate/evaluate directly for hosted/API models: "
            + ", ".join(invalid)
        )
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
        "gpu_count": args.gpu_count,
    }


def _build_generation_cluster_tasks(args: argparse.Namespace) -> list[ClusterTask]:
    processed_dataset = Path(args.processed_dataset)
    dataset_types = _cluster_dataset_types(
        processed_dataset,
        _csv_list(args.dataset_types, default=args.default_dataset_types),
    )
    tasks: list[ClusterTask] = []
    for dataset_type in dataset_types:
        for model in _cluster_models(args.models):
            tasks.append(
                ClusterTask(
                    stage="generate",
                    model=model,
                    model_slug=safe_name(model),
                    dataset_type=dataset_type,
                    dataset_slug=safe_name(dataset_type),
                    argv=[
                        "generate",
                        "--model",
                        model,
                        "--run-name",
                        args.run_name,
                        "--processed-dataset",
                        str(processed_dataset),
                        "--dataset-types",
                        dataset_type,
                        "--augmented-dataset",
                        str(
                            _cluster_augmented_dataset_dir(
                                Path(DEFAULT_AUGMENTED_CACHE_ROOT),
                                args.run_name,
                                model,
                                dataset_type,
                            )
                        ),
                        "--materialize-cache",
                    ],
                )
            )
    return tasks


def _build_evaluation_cluster_tasks(args: argparse.Namespace) -> list[ClusterTask]:
    processed_dataset = Path(args.processed_dataset)
    dataset_types = _cluster_dataset_types(
        processed_dataset,
        _csv_list(args.dataset_types, default=args.default_dataset_types),
    )
    generation_model = resolve_model_name(args.generator_model)
    tasks: list[ClusterTask] = []
    for dataset_type in dataset_types:
        dataset_cache = _cluster_augmented_dataset_dir(
            Path(DEFAULT_AUGMENTED_CACHE_ROOT),
            args.generator_run_name,
            generation_model,
            dataset_type,
        )
        for model in _cluster_models(args.models):
            tasks.append(
                ClusterTask(
                    stage="evaluate",
                    model=model,
                    model_slug=safe_name(model),
                    dataset_type=dataset_type,
                    dataset_slug=safe_name(dataset_type),
                    argv=[
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
                        str(processed_dataset),
                        "--dataset-types",
                        dataset_type,
                        "--augmented-dataset",
                        str(dataset_cache),
                    ],
                )
            )
    return tasks


def _run_cluster_submit(
    *,
    stage: str,
    run_name: str,
    tasks: list[ClusterTask],
    resources: dict[str, object],
    output_dir: str | None,
    submit: bool,
    dry_run: bool,
) -> int:
    if not tasks:
        print("No cluster tasks selected.")
        return 1

    paths = build_bundle_paths(stage=stage, run_name=run_name, output_dir=output_dir)
    manifest_text = render_manifest(stage=stage, run_name=run_name, resources=resources, tasks=tasks)
    sbatch_text = render_sbatch(
        paths=paths,
        stage=stage,
        run_name=run_name,
        resources=resources,
        task_count=len(tasks),
        gpu_count=resources["gpu_count"],
    )
    submit_text = render_submit_script(paths)

    if dry_run:
        print(f"Cluster stage: {stage}")
        print(f"Task count: {len(tasks)}")
        print(f"Bundle dir: {paths.bundle_dir}")
        print(f"SBATCH: {paths.sbatch_path}")
        print(f"Manifest: {paths.manifest_path}")
        print(f"Submit: sbatch {paths.sbatch_path.name}")
        return 0

    write_bundle(paths=paths, manifest_text=manifest_text, sbatch_text=sbatch_text, submit_text=submit_text)
    print(paths.manifest_path)
    print(paths.sbatch_path)
    print(paths.submit_path)

    if not submit:
        return 0

    try:
        result = submit_bundle(paths)
    except OSError as exc:
        print(str(exc))
        return 1
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
    return prepare_data(
        step=args.step,
        dataset=args.dataset,
        download_all=args.all,
        output_dir=args.output_dir,
        output_path=args.output_path,
        limit=args.limit,
    )


def _run_generate(args: argparse.Namespace) -> int:
    dataset_types = _csv_list(args.dataset_types, default=args.default_dataset_types)
    raw_model = resolve_model_name(args.model, args.backend)
    log_dir = _generation_log_dir(Path(args.log_root), args.run_name, raw_model)
    tasks = build_generation_tasks(
        processed_dataset_path=Path(args.processed_dataset),
        dataset_types=dataset_types,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        shard_strategy=args.shard_strategy,
        limit=args.limit,
        run_name=args.run_name,
        generation_model=raw_model,
    )
    if not tasks:
        print("No generation samples selected.")
        return 0
    _inspect_eval(tasks, model=raw_model, log_dir=log_dir, args=args)
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
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        shard_strategy=args.shard_strategy,
        limit=args.limit,
        run_name=args.run_name,
        generation_run_name=args.generator_run_name,
        generation_model=resolve_model_name(args.generator_model, args.generator_backend),
        evaluation_model=eval_model,
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


def _run_submit_generate_cluster(args: argparse.Namespace) -> int:
    try:
        tasks = _build_generation_cluster_tasks(args)
    except ValueError as exc:
        print(str(exc))
        return 1
    return _run_cluster_submit(
        stage="generate",
        run_name=args.run_name,
        tasks=tasks,
        resources=_cluster_resources(args),
        output_dir=args.output_dir,
        submit=args.submit,
        dry_run=args.dry_run,
    )


def _run_submit_evaluate_cluster(args: argparse.Namespace) -> int:
    try:
        tasks = _build_evaluation_cluster_tasks(args)
    except ValueError as exc:
        print(str(exc))
        return 1
    return _run_cluster_submit(
        stage="evaluate",
        run_name=args.run_name,
        tasks=tasks,
        resources=_cluster_resources(args),
        output_dir=args.output_dir,
        submit=args.submit,
        dry_run=args.dry_run,
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
    parser = argparse.ArgumentParser(description="Inspect-first Final5 pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-data")
    prepare.add_argument("--step", choices=["download", "process", "all"], default="all")
    prepare.add_argument("--dataset", choices=["mmlu_pro", "mmlu", "arc", "gpqa"], default=None)
    prepare.add_argument("--all", action="store_true")
    prepare.add_argument("--output-dir", default="datasets/raw")
    prepare.add_argument("--output-path", default=str(DEFAULT_PROCESSED_DATASET))
    prepare.add_argument("--limit", type=int, default=None)
    prepare.set_defaults(handler=_prepare_data)

    def add_runtime_flags(command: argparse.ArgumentParser) -> None:
        command.add_argument("--backend", default=None)
        command.add_argument("--model-base-url", default=None)
        command.add_argument("--max-connections", type=int, default=None)
        command.add_argument("--max-tokens", type=int, default=512)
        command.add_argument("--temperature", type=float, default=None)
        command.add_argument("--reasoning-effort", default=None)
        command.add_argument("--retry-on-error", type=int, default=2)
        command.add_argument("--stop-seqs", nargs="*", default=None)

    def add_shard_flags(command: argparse.ArgumentParser) -> None:
        command.add_argument("--shard-count", type=int, default=1)
        command.add_argument("--shard-index", type=int, default=0)
        command.add_argument("--shard-strategy", choices=["contiguous", "modulo"], default="contiguous")

    def add_cluster_submit_flags(command: argparse.ArgumentParser) -> None:
        command.add_argument("--models", default=None)
        command.add_argument("--processed-dataset", default=str(DEFAULT_PROCESSED_DATASET))
        command.add_argument("--dataset-types", default=None)
        command.add_argument("--gpu-count", type=int, default=None)
        command.add_argument("--output-dir", default=None)
        command.add_argument("--submit", dest="submit", action="store_true", default=True)
        command.add_argument("--write-only", dest="submit", action="store_false")
        command.add_argument("--dry-run", action="store_true")
        command.add_argument("--partition", default="clip")
        command.add_argument("--account", default="clip")
        command.add_argument("--qos", default="high")
        command.add_argument("--time-limit", default="12:00:00")
        command.add_argument("--mem", default="32G")
        command.add_argument("--cpus-per-task", type=int, default=4)
        command.add_argument("--gpu-type", default="rtxa6000")

    generate = sub.add_parser("generate")
    generate.add_argument("--model", required=True)
    generate.add_argument("--run-name", required=True)
    generate.add_argument("--processed-dataset", default=str(DEFAULT_PROCESSED_DATASET))
    generate.add_argument("--dataset-types", default=None)
    generate.add_argument("--limit", type=int, default=None)
    generate.add_argument("--log-root", default=str(DEFAULT_GENERATION_LOG_ROOT))
    generate.add_argument("--cache-root", default=str(DEFAULT_AUGMENTED_CACHE_ROOT))
    generate.add_argument("--augmented-dataset", default=None)
    generate.add_argument("--materialize-cache", action="store_true")
    generate.add_argument("--rebuild-cache", action="store_true")
    generate.set_defaults(default_dataset_types=["arc_challenge", "mmlu_pro", "gpqa"])
    add_runtime_flags(generate)
    add_shard_flags(generate)
    generate.set_defaults(handler=_run_generate)

    generate_all = sub.add_parser("generate-all")
    generate_all.add_argument("--models", default=None)
    generate_all.add_argument("--run-name", required=True)
    generate_all.add_argument("--processed-dataset", default=str(DEFAULT_PROCESSED_DATASET))
    generate_all.add_argument("--dataset-types", default=None)
    generate_all.add_argument("--limit", type=int, default=None)
    generate_all.add_argument("--log-root", default=str(DEFAULT_GENERATION_LOG_ROOT))
    generate_all.add_argument("--cache-root", default=str(DEFAULT_AUGMENTED_CACHE_ROOT))
    generate_all.add_argument("--materialize-cache", action="store_true")
    generate_all.add_argument("--rebuild-cache", action="store_true")
    generate_all.set_defaults(default_dataset_types=["arc_challenge", "mmlu_pro", "gpqa"])
    add_runtime_flags(generate_all)
    add_shard_flags(generate_all)
    generate_all.set_defaults(handler=_run_generate_all)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--run-name", required=True)
    evaluate.add_argument("--generator-run-name", required=True)
    evaluate.add_argument("--generator-model", required=True)
    evaluate.add_argument("--generator-backend", default=None)
    evaluate.add_argument("--generation-log-dir", default=None)
    evaluate.add_argument("--generation-log-root", default=str(DEFAULT_GENERATION_LOG_ROOT))
    evaluate.add_argument("--processed-dataset", default=str(DEFAULT_PROCESSED_DATASET))
    evaluate.add_argument("--augmented-dataset", default=None)
    evaluate.add_argument("--cache-root", default=str(DEFAULT_AUGMENTED_CACHE_ROOT))
    evaluate.add_argument("--dataset-types", default=None)
    evaluate.add_argument("--settings", default=None)
    evaluate.add_argument("--modes", default=None)
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.add_argument("--log-root", default=str(DEFAULT_EVALUATION_LOG_ROOT))
    evaluate.add_argument("--rebuild-cache", action="store_true")
    evaluate.set_defaults(default_dataset_types=["arc_challenge", "mmlu_pro", "gpqa"])
    add_runtime_flags(evaluate)
    add_shard_flags(evaluate)
    evaluate.set_defaults(handler=_run_evaluate)

    evaluate_all = sub.add_parser("evaluate-all")
    evaluate_all.add_argument("--models", default=None)
    evaluate_all.add_argument("--run-name", required=True)
    evaluate_all.add_argument("--generator-run-name", required=True)
    evaluate_all.add_argument("--generator-model", required=True)
    evaluate_all.add_argument("--generator-backend", default=None)
    evaluate_all.add_argument("--generation-log-dir", default=None)
    evaluate_all.add_argument("--generation-log-root", default=str(DEFAULT_GENERATION_LOG_ROOT))
    evaluate_all.add_argument("--processed-dataset", default=str(DEFAULT_PROCESSED_DATASET))
    evaluate_all.add_argument("--augmented-dataset", default=None)
    evaluate_all.add_argument("--cache-root", default=str(DEFAULT_AUGMENTED_CACHE_ROOT))
    evaluate_all.add_argument("--dataset-types", default=None)
    evaluate_all.add_argument("--settings", default=None)
    evaluate_all.add_argument("--modes", default=None)
    evaluate_all.add_argument("--limit", type=int, default=None)
    evaluate_all.add_argument("--log-root", default=str(DEFAULT_EVALUATION_LOG_ROOT))
    evaluate_all.add_argument("--rebuild-cache", action="store_true")
    evaluate_all.set_defaults(default_dataset_types=["arc_challenge", "mmlu_pro", "gpqa"])
    add_runtime_flags(evaluate_all)
    add_shard_flags(evaluate_all)
    evaluate_all.set_defaults(handler=_run_evaluate_all)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--results-root", default=str(DEFAULT_EVALUATION_LOG_ROOT))
    analyze.add_argument("--output-dir", default="results/final5_plots")
    analyze.add_argument("--table-output", default="results/final5_plots/tables/final5_results_summary.csv")
    analyze.add_argument("--skip-tables", action="store_true")
    analyze.set_defaults(handler=_run_analyze)

    signatures = sub.add_parser("signature-table")
    signatures.add_argument("--dir", required=True)
    signatures.add_argument("--output", default=None)
    signatures.set_defaults(handler=_run_signature_table)

    export = sub.add_parser("export")
    export.add_argument("--input", default=None)
    export.add_argument("--output-root", default="datasets/benchmarker_items")
    export.add_argument("--generator-run-name", default=None)
    export.add_argument("--generator-model", default=None)
    export.add_argument("--generator-backend", default=None)
    export.add_argument("--generation-log-dir", default=None)
    export.add_argument("--generation-log-root", default=str(DEFAULT_GENERATION_LOG_ROOT))
    export.add_argument("--processed-dataset", default=str(DEFAULT_PROCESSED_DATASET))
    export.add_argument("--augmented-dataset", default=None)
    export.add_argument("--cache-root", default=str(DEFAULT_AUGMENTED_CACHE_ROOT))
    export.add_argument("--dataset-types", default=None)
    export.add_argument("--rebuild-cache", action="store_true")
    export.set_defaults(default_dataset_types=["arc_challenge", "mmlu_pro", "gpqa"])
    export.set_defaults(handler=_run_export)

    submit_generate_cluster = sub.add_parser("submit-generate-cluster")
    submit_generate_cluster.add_argument("--run-name", required=True)
    add_cluster_submit_flags(submit_generate_cluster)
    submit_generate_cluster.set_defaults(default_dataset_types=["arc_challenge", "mmlu_pro", "gpqa"])
    submit_generate_cluster.set_defaults(handler=_run_submit_generate_cluster)

    submit_evaluate_cluster = sub.add_parser("submit-evaluate-cluster")
    submit_evaluate_cluster.add_argument("--run-name", required=True)
    submit_evaluate_cluster.add_argument("--generator-run-name", required=True)
    submit_evaluate_cluster.add_argument("--generator-model", required=True)
    add_cluster_submit_flags(submit_evaluate_cluster)
    submit_evaluate_cluster.set_defaults(default_dataset_types=["arc_challenge", "mmlu_pro", "gpqa"])
    submit_evaluate_cluster.set_defaults(handler=_run_submit_evaluate_cluster)

    diagnose_failures = sub.add_parser("diagnose-failures")
    diagnose_failures.add_argument("--dataset-path", required=True)
    diagnose_failures.set_defaults(handler=_run_diagnose_failures)

    diagnose_trace = sub.add_parser("diagnose-trace")
    diagnose_trace.add_argument("--log-dir", required=True)
    diagnose_trace.add_argument("--sample-id", default=None)
    diagnose_trace.add_argument("--only-errors", action="store_true")
    diagnose_trace.add_argument("--limit", type=int, default=None)
    diagnose_trace.add_argument("--summary", action="store_true")
    diagnose_trace.add_argument("--output", default=None)
    diagnose_trace.set_defaults(handler=_run_diagnose_trace)

    smoke_generate = sub.add_parser("smoke-generate")
    smoke_generate.add_argument("--models", nargs="+", default=list(DEFAULT_GENERATION_MODELS))
    smoke_generate.add_argument("--run-name", default="smoke-generate")
    smoke_generate.add_argument("--processed-dataset", default=str(DEFAULT_PROCESSED_DATASET))
    smoke_generate.add_argument("--dataset-types", default=None)
    smoke_generate.add_argument("--limit", type=int, default=2)
    smoke_generate.add_argument("--log-root", default=str(DEFAULT_GENERATION_LOG_ROOT))
    smoke_generate.add_argument("--cache-root", default=str(DEFAULT_AUGMENTED_CACHE_ROOT))
    smoke_generate.add_argument("--backend", default=None)
    smoke_generate.add_argument("--model-base-url", default=None)
    smoke_generate.add_argument("--reasoning-effort", default=None)
    smoke_generate.add_argument("--max-tokens", type=int, default=256)
    smoke_generate.add_argument("--dry-run", action="store_true")
    smoke_generate.set_defaults(handler=_run_smoke_generate)

    smoke_evaluate = sub.add_parser("smoke-evaluate")
    smoke_evaluate.add_argument("--models", nargs="+", default=list(DEFAULT_EVALUATION_MODELS))
    smoke_evaluate.add_argument("--run-name", default="smoke-evaluate")
    smoke_evaluate.add_argument("--generator-run-name", required=True)
    smoke_evaluate.add_argument("--generator-model", required=True)
    smoke_evaluate.add_argument("--generator-backend", default=None)
    smoke_evaluate.add_argument("--processed-dataset", default=str(DEFAULT_PROCESSED_DATASET))
    smoke_evaluate.add_argument("--dataset-types", default=None)
    smoke_evaluate.add_argument("--settings", default=None)
    smoke_evaluate.add_argument("--modes", default=None)
    smoke_evaluate.add_argument("--limit", type=int, default=2)
    smoke_evaluate.add_argument("--generation-log-root", default=str(DEFAULT_GENERATION_LOG_ROOT))
    smoke_evaluate.add_argument("--cache-root", default=str(DEFAULT_AUGMENTED_CACHE_ROOT))
    smoke_evaluate.add_argument("--log-root", default=str(DEFAULT_EVALUATION_LOG_ROOT))
    smoke_evaluate.add_argument("--backend", default=None)
    smoke_evaluate.add_argument("--model-base-url", default=None)
    smoke_evaluate.add_argument("--reasoning-effort", default=None)
    smoke_evaluate.add_argument("--max-tokens", type=int, default=128)
    smoke_evaluate.add_argument("--dry-run", action="store_true")
    smoke_evaluate.set_defaults(handler=_run_smoke_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
