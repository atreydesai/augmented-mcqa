#!/usr/bin/env python3
"""Primary matrix CLI for deterministic, sequential evaluation runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RESULTS_DIR
from experiments import run_experiment
from models import get_client
from experiments.matrix import (
    ALL_DATASET_TYPES,
    DISTRACTOR_SOURCE_MAP,
    MATRIX_PRESETS,
    MatrixPreset,
    build_manifest,
    build_matrix_configs,
    load_configs_from_manifest,
    maybe_select_shard,
    save_manifest,
    summarize_configs,
)
from experiments.defaults import (
    DEFAULT_EVAL_KEEP_CHECKPOINTS,
    DEFAULT_EVAL_MAX_TOKENS,
    DEFAULT_EVAL_MODE,
    DEFAULT_EVAL_SAVE_INTERVAL,
    DEFAULT_EVAL_SEED,
    DEFAULT_EVAL_TEMPERATURE,
    DEFAULT_MATRIX_PRESET,
)


def _require_generator_label(raw: str | None) -> str:
    label = (raw or "").strip()
    if not label:
        raise ValueError("generator_dataset_label is required and cannot be blank")
    return label


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _env_bool(name: str, default: bool | None = None) -> bool | None:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {value}")


def _add_common_build_args(parser: argparse.ArgumentParser, required_inputs: bool) -> None:
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(MATRIX_PRESETS.keys()),
        default=DEFAULT_MATRIX_PRESET,
        help="Matrix preset to build",
    )
    parser.add_argument("--model", type=str, required=required_inputs, help="Model name")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=required_inputs,
        help="Path to unified processed/augmented dataset",
    )
    parser.add_argument(
        "--distractor-source",
        "--distractor-sources",
        dest="distractor_sources",
        type=str,
        nargs="+",
        choices=sorted(DISTRACTOR_SOURCE_MAP.keys()),
        default=sorted(DISTRACTOR_SOURCE_MAP.keys()),
        help="Distractor source(s) to run",
    )
    parser.add_argument(
        "--dataset-types",
        type=str,
        nargs="+",
        choices=ALL_DATASET_TYPES,
        default=ALL_DATASET_TYPES,
        help="Dataset type(s) to run",
    )
    parser.add_argument(
        "--generator-dataset-label",
        type=str,
        required=True,
        help="Required generator dataset label used to isolate output paths",
    )

    parser.add_argument("--limit", type=int, help="Limit entries per config")
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["accuracy", "behavioral"],
        default=DEFAULT_EVAL_MODE,
    )
    parser.add_argument("--choices-only", action="store_true", help="Use choices-only prompt")
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED)

    parser.add_argument("--reasoning-effort", type=str, help="OpenAI reasoning effort")
    parser.add_argument("--thinking-level", type=str, help="Anthropic/Gemini thinking level")
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_EVAL_TEMPERATURE,
        help="Sampling temperature (provider default if omitted)",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_EVAL_MAX_TOKENS)
    parser.add_argument(
        "--save-interval",
        type=int,
        default=DEFAULT_EVAL_SAVE_INTERVAL,
        help=(
            "Checkpoint interval in processed entries "
            f"(default: {DEFAULT_EVAL_SAVE_INTERVAL})"
        ),
    )

    parser.add_argument("--output-dir", type=str, help="Base output directory")


def _add_shard_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-shards", type=int, help="Total shard count")
    parser.add_argument("--shard-index", type=int, help="0-based shard index")


def _resolve_configs(args: argparse.Namespace) -> list:
    def _apply_runtime_overrides(cfg) -> None:
        cfg.save_interval = args.save_interval
        cfg.inference_batch_size = int(getattr(args, "eval_batch_size", cfg.inference_batch_size))
        cfg.workpack_format = str(getattr(args, "workpack_format", cfg.workpack_format))
        cfg.vllm_max_num_batched_tokens = getattr(
            args, "vllm_max_num_batched_tokens", cfg.vllm_max_num_batched_tokens
        )
        cfg.vllm_max_num_seqs = getattr(args, "vllm_max_num_seqs", cfg.vllm_max_num_seqs)
        cfg.vllm_enable_chunked_prefill = getattr(
            args, "vllm_enable_chunked_prefill", cfg.vllm_enable_chunked_prefill
        )

    if getattr(args, "manifest", None):
        configs = load_configs_from_manifest(Path(args.manifest))
        for cfg in configs:
            _apply_runtime_overrides(cfg)
        return configs

    output_base = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    configs = build_matrix_configs(
        model=args.model,
        dataset_path=Path(args.dataset_path),
        generator_dataset_label=args.generator_dataset_label,
        dataset_types=args.dataset_types,
        distractor_sources=args.distractor_sources,
        preset=args.preset,
        output_base=output_base,
        limit=args.limit,
        eval_mode=args.eval_mode,
        choices_only=args.choices_only,
        seed=args.seed,
        reasoning_effort=args.reasoning_effort,
        thinking_level=args.thinking_level,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        save_interval=args.save_interval,
    )
    for cfg in configs:
        _apply_runtime_overrides(cfg)
    return configs


def _print_summary(label: str, configs: list) -> None:
    summary = summarize_configs(configs)
    print(f"\n=== {label} ===")
    print(f"Total configs: {summary['total']}")
    print(f"By dataset type: {summary['by_dataset_type']}")
    print(f"By distractor source: {summary['by_distractor_source']}")


def _default_manifest_path(model: str, preset: str, generator_label: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    model_safe = model.replace("/", "_")
    label_safe = generator_label.replace("/", "_")
    return RESULTS_DIR / "manifests" / f"eval_matrix_{label_safe}_{model_safe}_{preset}_{ts}.json"


def cmd_plan(args: argparse.Namespace) -> int:
    generator_label = _require_generator_label(args.generator_dataset_label)
    configs = _resolve_configs(args)
    _print_summary("Matrix Plan", configs)

    selected = maybe_select_shard(configs, args.num_shards, args.shard_index)
    if args.num_shards is not None:
        print(
            f"Shard selection: index={args.shard_index}/{args.num_shards - 1} "
            f"-> {len(selected)} configs"
        )

    manifest = build_manifest(
        configs,
        preset=args.preset,
        model=args.model,
        dataset_path=Path(args.dataset_path),
        generator_dataset_label=generator_label,
        dataset_types=args.dataset_types,
        distractor_sources=args.distractor_sources,
        metadata={
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "choices_only": args.choices_only,
            "save_interval": args.save_interval,
        },
    )

    manifest_out = (
        Path(args.manifest_out)
        if args.manifest_out
        else _default_manifest_path(args.model, args.preset, generator_label)
    )
    save_manifest(manifest, manifest_out)
    print(f"Manifest written to: {manifest_out}")

    if args.print_configs:
        print("\nConfigs in run order:")
        for idx, cfg in enumerate(selected, start=1):
            print(
                f"  [{idx:3d}] {cfg.name} | {cfg.distractor_config_str} | "
                f"dataset={cfg.dataset_type_filter} | source={cfg.distractor_source}"
            )

    return 0


def _summary_path(
    model: str,
    generator_dataset_label: str,
    output_base: Path,
    num_shards: int | None,
    shard_index: int | None,
) -> Path:
    safe_model = model.replace("/", "_")
    safe_label = generator_dataset_label.replace("/", "_")
    if num_shards is not None and shard_index is not None:
        filename = (
            f"batch_summary_{safe_label}_{safe_model}_shard_{shard_index}_of_{num_shards}.json"
        )
    else:
        filename = f"batch_summary_{safe_label}_{safe_model}.json"
    return output_base / filename


def _label_from_configs(configs: list) -> str:
    labels = {str(getattr(cfg, "generator_dataset_label", "")).strip() for cfg in configs}
    labels = {label for label in labels if label}
    if not labels:
        raise ValueError(
            "No generator_dataset_label found in configs. Rebuild manifest/configs with required label."
        )
    if len(labels) > 1:
        raise ValueError(f"Mixed generator_dataset_label values in configs: {sorted(labels)}")
    return next(iter(labels))


def _checkpoint_root_from_output_dir(output_dir: Path) -> Path:
    try:
        return output_dir.parent.parent
    except IndexError:
        return output_dir.parent


def _prune_checkpoint_files(checkpoint_roots: set[Path], keep: int) -> dict[str, Any]:
    if keep < 0:
        raise ValueError(f"keep_checkpoints must be >= 0, got {keep}")

    total_found = 0
    total_deleted = 0
    per_root: dict[str, dict[str, int]] = {}

    for root in sorted(checkpoint_roots):
        files = sorted(
            root.rglob("eval_checkpoint_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        total_found += len(files)
        to_delete = files[keep:] if keep < len(files) else []
        deleted = 0
        for path in to_delete:
            try:
                path.unlink()
                deleted += 1
            except FileNotFoundError:
                continue
        total_deleted += deleted
        per_root[str(root)] = {
            "found": len(files),
            "deleted": deleted,
            "kept": len(files) - deleted,
        }

    return {
        "keep": keep,
        "roots": per_root,
        "total_found": total_found,
        "total_deleted": total_deleted,
        "total_kept": total_found - total_deleted,
    }


def _is_local_model_name(model_name: str) -> bool:
    return model_name == "local" or "/" in model_name


def _client_cache_key(
    config,
) -> tuple[str, str | None, str | None, int | None, int | None, bool | None]:
    return (
        config.model_name,
        config.reasoning_effort,
        config.thinking_level,
        config.vllm_max_num_batched_tokens,
        config.vllm_max_num_seqs,
        config.vllm_enable_chunked_prefill,
    )


def _client_kwargs(config) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if config.reasoning_effort:
        kwargs["reasoning_effort"] = config.reasoning_effort
    if config.thinking_level:
        kwargs["thinking_level"] = config.thinking_level
    if _is_local_model_name(config.model_name):
        if config.vllm_max_num_batched_tokens is not None:
            kwargs["max_num_batched_tokens"] = config.vllm_max_num_batched_tokens
        if config.vllm_max_num_seqs is not None:
            kwargs["max_num_seqs"] = config.vllm_max_num_seqs
        if config.vllm_enable_chunked_prefill is not None:
            kwargs["enable_chunked_prefill"] = config.vllm_enable_chunked_prefill
    return kwargs


def _get_or_create_shared_client(
    config,
    client_cache: dict[
        tuple[str, str | None, str | None, int | None, int | None, bool | None], Any
    ],
):
    key = _client_cache_key(config)
    if key in client_cache:
        return client_cache[key]

    client = get_client(config.model_name, **_client_kwargs(config))
    client_cache[key] = client
    print(f"  Shared client initialized for: {config.model_name}")
    return client


def _unload_shared_clients(client_cache: dict[tuple[str, str | None, str | None], Any]) -> None:
    for client in client_cache.values():
        unload = getattr(client, "unload", None)
        if callable(unload):
            try:
                unload()
            except Exception as exc:
                print(f"Warning: failed to unload client {client}: {exc}")


def _run_single_config(
    config,
    *,
    client_cache: dict[
        tuple[str, str | None, str | None, int | None, int | None, bool | None], Any
    ],
    dataset_cache: dict[tuple[Any, ...], Any],
) -> dict[str, Any]:
    try:
        client = _get_or_create_shared_client(config, client_cache)
        results = run_experiment(config, client=client, dataset_cache=dataset_cache)
        if results.attempted_entries > 0 and results.successful_entries == 0:
            return {
                "name": config.name,
                "config": config.distractor_config_str,
                "accuracy": 0.0,
                "total": len(results.results),
                "attempted_entries": results.attempted_entries,
                "successful_entries": results.successful_entries,
                "failed_entries": results.failed_entries,
                "accuracy_success_only": 0.0,
                "entry_failure_count": len(results.entry_failures),
                "resumed_from_checkpoint": results.resumed_from_checkpoint,
                "status": "error: zero successful entries",
                "output_dir": str(config.output_dir),
            }
        return {
            "name": config.name,
            "config": config.distractor_config_str,
            "accuracy": results.accuracy,
            "total": len(results.results),
            "attempted_entries": results.attempted_entries,
            "successful_entries": results.successful_entries,
            "failed_entries": results.failed_entries,
            "accuracy_success_only": results.accuracy,
            "entry_failure_count": len(results.entry_failures),
            "resumed_from_checkpoint": results.resumed_from_checkpoint,
            "status": "success",
            "output_dir": str(config.output_dir),
        }
    except Exception as exc:
        return {
            "name": config.name,
            "config": config.distractor_config_str,
            "accuracy": 0.0,
            "total": 0,
            "attempted_entries": 0,
            "successful_entries": 0,
            "failed_entries": 0,
            "accuracy_success_only": 0.0,
            "entry_failure_count": 0,
            "resumed_from_checkpoint": False,
            "status": f"error: {exc}",
            "output_dir": str(config.output_dir),
        }


def _required_workpack_columns(configs: list) -> list[str]:
    columns = {
        "question",
        "choices_answer",
        "category",
        "choices_human",
        "dataset_type",
    }
    for cfg in configs:
        columns.add(str(cfg.model_distractor_type.value))
        if cfg.branching_mode == "human_prefix":
            columns.update(
                {
                    "cond_model_q_a_dhuman_h1",
                    "cond_model_q_a_dhuman_h2",
                    "cond_model_q_a_dhuman_h3",
                }
            )
    return sorted(columns)


def _load_split_for_dataset_type(dataset: Any, dataset_type: str | None, dataset_path: str) -> Any:
    if dataset_type:
        if hasattr(dataset, "keys") and dataset_type in dataset:
            data = dataset[dataset_type]
        elif hasattr(dataset, "column_names") and "dataset_type" in dataset.column_names:
            data = dataset.filter(lambda x: x["dataset_type"] == dataset_type)
            if len(data) == 0:
                raise ValueError(
                    f"dataset_type_filter='{dataset_type}' matched zero rows in {dataset_path}"
                )
        else:
            available_splits = list(dataset.keys()) if hasattr(dataset, "keys") else []
            raise ValueError(
                f"dataset_type_filter='{dataset_type}' not found in {dataset_path}. "
                f"Available splits: {available_splits}"
            )
    else:
        if hasattr(dataset, "keys"):
            if "test" in dataset:
                data = dataset["test"]
            elif len(dataset.keys()) == 1:
                data = next(iter(dataset.values()))
            else:
                raise ValueError(
                    f"Dataset has multiple splits {list(dataset.keys())}; "
                    "set dataset_type_filter explicitly."
                )
        else:
            data = dataset
    return data


def _prepare_workpacks(
    configs: list,
    workpack_format: str,
    output_base: Path,
    *,
    workpack_root: Path | None = None,
) -> dict[tuple[str, str], Path]:
    if workpack_format == "none":
        return {}

    from datasets import load_from_disk

    grouped: dict[tuple[str, str], list] = {}
    for cfg in configs:
        key = (str(cfg.dataset_path), cfg.dataset_type_filter or "__all__")
        grouped.setdefault(key, []).append(cfg)

    effective_workpack_root = workpack_root or (output_base / "_workpacks")
    effective_workpack_root.mkdir(parents=True, exist_ok=True)

    raw_dataset_cache: dict[str, Any] = {}
    produced: dict[tuple[str, str], Path] = {}
    created_count = 0
    reused_count = 0

    for (dataset_path, dataset_type_key), group in sorted(grouped.items()):
        dataset_type = None if dataset_type_key == "__all__" else dataset_type_key
        required_cols = _required_workpack_columns(group)
        cache_key = f"{dataset_path}::{dataset_type_key}::{','.join(required_cols)}"
        suffix = ".parquet" if workpack_format == "parquet" else ".arrow"
        stem = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:16]
        out_name = f"{stem}_{dataset_type_key}{suffix}"
        out_path = effective_workpack_root / out_name
        lock_path = effective_workpack_root / f".{out_name}.lock"

        if out_path.exists():
            reused_count += 1
            produced[(dataset_path, dataset_type_key)] = out_path
            for cfg in group:
                cfg.workpack_format = workpack_format
                cfg.workpack_path = out_path
            continue

        lock_acquired = False
        wait_deadline = time.monotonic() + 600.0
        while True:
            try:
                lock_path.mkdir(parents=False, exist_ok=False)
                lock_acquired = True
                break
            except FileExistsError:
                if out_path.exists():
                    break
                if time.monotonic() >= wait_deadline:
                    raise TimeoutError(
                        f"Timed out waiting for workpack lock: {lock_path}"
                    )
                time.sleep(0.25)

        if lock_acquired:
            try:
                if not out_path.exists():
                    dataset = raw_dataset_cache.get(dataset_path)
                    if dataset is None:
                        dataset = load_from_disk(dataset_path)
                        raw_dataset_cache[dataset_path] = dataset

                    split = _load_split_for_dataset_type(dataset, dataset_type, dataset_path)
                    existing_cols = set(split.column_names)
                    selected_cols = [col for col in required_cols if col in existing_cols]
                    if selected_cols and len(selected_cols) < len(split.column_names):
                        split = split.select_columns(selected_cols)

                    if workpack_format == "parquet":
                        tmp_path = out_path.with_suffix(f"{out_path.suffix}.tmp.{os.getpid()}")
                        if tmp_path.exists():
                            tmp_path.unlink()
                        split.to_parquet(str(tmp_path))
                        tmp_path.replace(out_path)
                    else:
                        tmp_path = out_path.parent / f"{out_path.name}.tmp.{os.getpid()}"
                        if tmp_path.exists():
                            shutil.rmtree(tmp_path)
                        split.save_to_disk(str(tmp_path))
                        if out_path.exists():
                            shutil.rmtree(out_path)
                        tmp_path.rename(out_path)
                    created_count += 1
                else:
                    reused_count += 1
            finally:
                try:
                    lock_path.rmdir()
                except OSError:
                    pass
        else:
            reused_count += 1

        produced[(dataset_path, dataset_type_key)] = out_path
        for cfg in group:
            cfg.workpack_format = workpack_format
            cfg.workpack_path = out_path

    print(
        "Prepared workpacks: "
        f"format={workpack_format} groups={len(grouped)} "
        f"created={created_count} reused={reused_count} root={effective_workpack_root}"
    )
    return produced


def cmd_run(args: argparse.Namespace) -> int:
    requested_label = _require_generator_label(args.generator_dataset_label)
    configs = _resolve_configs(args)
    if not configs:
        print("No configs available to run.")
        return 0

    config_label = _label_from_configs(configs)
    if config_label != requested_label:
        raise ValueError(
            "generator_dataset_label mismatch: "
            f"CLI='{requested_label}' vs configs='{config_label}'"
        )

    configs = maybe_select_shard(configs, args.num_shards, args.shard_index)

    if args.skip_existing:
        before = len(configs)
        configs = [cfg for cfg in configs if not (cfg.output_dir / "results.json").exists()]
        skipped = before - len(configs)
        if skipped:
            print(f"Skipping {skipped} configs with existing results.json")

    if not configs:
        print("No configs to run after filters/shard/skip-existing.")
        return 0

    output_base = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_base.mkdir(parents=True, exist_ok=True)
    workpack_root = Path(args.workpack_root).expanduser() if args.workpack_root else None
    _prepare_workpacks(
        configs,
        args.workpack_format,
        output_base,
        workpack_root=workpack_root,
    )

    _print_summary("Run Set", configs)

    summaries = []
    checkpoint_roots: set[Path] = set()
    client_cache: dict[
        tuple[str, str | None, str | None, int | None, int | None, bool | None], Any
    ] = {}
    dataset_cache: dict[tuple[Any, ...], Any] = {}
    try:
        for idx, config in enumerate(configs, start=1):
            print(f"\n[{idx}/{len(configs)}] {config.name}")
            checkpoint_roots.add(_checkpoint_root_from_output_dir(config.output_dir))
            summary = _run_single_config(
                config,
                client_cache=client_cache,
                dataset_cache=dataset_cache,
            )
            summaries.append(summary)
            if summary["status"] == "success":
                print(f"  Accuracy: {summary['accuracy']:.2%}")
            else:
                print(f"  FAILED: {summary['status']}")
    finally:
        _unload_shared_clients(client_cache)

    model_name = configs[0].model_name
    summary_path = _summary_path(
        model_name,
        requested_label,
        output_base,
        args.num_shards,
        args.shard_index,
    )

    prune_stats = _prune_checkpoint_files(checkpoint_roots, args.keep_checkpoints)
    print(
        "Checkpoint prune complete: "
        f"found={prune_stats['total_found']} deleted={prune_stats['total_deleted']} "
        f"kept={prune_stats['total_kept']}"
    )

    payload = {
        "model": model_name,
        "generator_dataset_label": requested_label,
        "preset": args.preset,
        "manifest": args.manifest,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "workpack_format": args.workpack_format,
        "workpack_root": args.workpack_root,
        "eval_batch_size": args.eval_batch_size,
        "vllm_max_num_batched_tokens": args.vllm_max_num_batched_tokens,
        "vllm_max_num_seqs": args.vllm_max_num_seqs,
        "vllm_enable_chunked_prefill": args.vllm_enable_chunked_prefill,
        "total_configs": len(summaries),
        "successful": sum(1 for s in summaries if s["status"] == "success"),
        "failed": sum(1 for s in summaries if s["status"] != "success"),
        "entry_failures_total": sum(s["entry_failure_count"] for s in summaries),
        "configs_with_entry_failures": sum(1 for s in summaries if s["entry_failure_count"] > 0),
        "fatal_config_failures": sum(1 for s in summaries if s["status"] != "success"),
        "checkpoint_prune": prune_stats,
        "results": summaries,
    }

    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nBatch summary saved to: {summary_path}")

    failed = [s for s in summaries if s["status"] != "success"]
    if failed:
        print("\nFailures:")
        for item in failed:
            print(f"  - {item['name']}: {item['status']}")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic matrix planner/runner for MCQA evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    plan_parser = subparsers.add_parser("plan", help="Build matrix and write manifest")
    _add_common_build_args(plan_parser, required_inputs=True)
    _add_shard_args(plan_parser)
    plan_parser.add_argument("--manifest-out", type=str, help="Output path for manifest JSON")
    plan_parser.add_argument("--print-configs", action="store_true", help="Print per-config run list")
    plan_parser.set_defaults(handler=cmd_plan)

    run_parser = subparsers.add_parser("run", help="Run matrix sequentially")
    _add_common_build_args(run_parser, required_inputs=False)
    _add_shard_args(run_parser)
    run_parser.add_argument("--manifest", type=str, help="Load configs from existing manifest")
    run_parser.add_argument("--skip-existing", action="store_true", help="Skip configs with results.json")
    run_parser.add_argument(
        "--workpack-format",
        type=str,
        choices=["none", "parquet", "arrow"],
        default=os.getenv("EVAL_WORKPACK_FORMAT", "parquet"),
        help="Precompute/read dataset workpacks for faster repeated loads",
    )
    run_parser.add_argument(
        "--workpack-root",
        type=str,
        default=os.getenv("EVAL_WORKPACK_ROOT", ""),
        help="Shared directory for reusable workpacks (default: output-dir/_workpacks)",
    )
    run_parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=_env_int("EVAL_BATCH_SIZE", 8),
        help="Inference batch size used by runner when calling generate_batch",
    )
    run_parser.add_argument(
        "--vllm-max-num-batched-tokens",
        type=int,
        default=_env_int("VLLM_MAX_NUM_BATCHED_TOKENS", None),
        help="Local vLLM max_num_batched_tokens override",
    )
    run_parser.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=_env_int("VLLM_MAX_NUM_SEQS", None),
        help="Local vLLM max_num_seqs override",
    )
    run_parser.add_argument(
        "--vllm-enable-chunked-prefill",
        type=lambda v: bool(int(v)),
        default=_env_bool("VLLM_ENABLE_CHUNKED_PREFILL", None),
        help="Local vLLM enable_chunked_prefill override (0/1)",
    )
    run_parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=DEFAULT_EVAL_KEEP_CHECKPOINTS,
        help="After run completion, keep only this many newest checkpoint files per output root",
    )
    run_parser.set_defaults(handler=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "run" and args.keep_checkpoints < 0:
        parser.error("--keep-checkpoints must be >= 0")
    if args.subcommand == "run" and args.eval_batch_size <= 0:
        parser.error("--eval-batch-size must be > 0")

    if args.subcommand == "run" and not args.manifest:
        missing = [flag for flag in ["model", "dataset_path"] if getattr(args, flag) is None]
        if missing:
            parser.error("run requires --manifest OR both --model and --dataset-path")

    if args.subcommand == "run" and args.manifest:
        # Manifest-driven runs ignore matrix-construction fields.
        if args.model or args.dataset_path:
            print("Note: --manifest provided; ignoring matrix build inputs and using manifest configs.")

    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
