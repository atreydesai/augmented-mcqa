from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

from inspect_ai import eval as inspect_eval

from analysis.visualize import plot_final5_pairwise, write_final5_summary_table
from data import export_benchmarker_items
from data.final5_store import ensure_augmented_dataset
from tasks import build_evaluation_tasks, build_generation_tasks
from utils.constants import (
    DEFAULT_AUGMENTED_CACHE_ROOT,
    DEFAULT_EVALUATION_LOG_ROOT,
    DEFAULT_EVALUATION_MODELS,
    DEFAULT_GENERATION_LOG_ROOT,
    DEFAULT_GENERATION_MODELS,
    DEFAULT_PROCESSED_DATASET,
    FINAL5_SETTINGS,
    MODE_CHOICES,
)
from utils.modeling import resolve_model_name, safe_name


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
    mod = importlib.import_module("scripts.01_data_pipeline")
    namespace = argparse.Namespace(
        dataset=args.dataset,
        all=args.all,
        output_dir=args.output_dir,
        output_path=args.output_path,
        limit=args.limit,
    )
    if args.step == "download":
        return int(mod.cmd_download(namespace))
    if args.step == "process":
        return int(mod.cmd_process(namespace))
    return int(mod.cmd_all(namespace))


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
    models = _csv_list(args.models, default=list(DEFAULT_GENERATION_MODELS))
    for model in models:
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
    generation_log_dir, cache_dir = _resolve_generation_artifacts(args)
    _ = generation_log_dir
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
    models = _csv_list(args.models, default=list(DEFAULT_EVALUATION_MODELS))
    for model in models:
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


def _run_export(args: argparse.Namespace) -> int:
    if args.input:
        dataset_path = Path(args.input)
    else:
        _generation_log_dir, dataset_path = _resolve_generation_artifacts(args)
    summary_path = export_benchmarker_items(dataset_path, args.output_root)
    print(summary_path)
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
