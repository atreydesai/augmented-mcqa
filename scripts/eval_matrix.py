#!/usr/bin/env python3
"""Final5 matrix planner/runner."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RESULTS_DIR
from experiments import run_experiment
from experiments.defaults import (
    DEFAULT_EVAL_KEEP_CHECKPOINTS,
    DEFAULT_EVAL_MAX_TOKENS,
    DEFAULT_EVAL_MODE,
    DEFAULT_EVAL_SAVE_INTERVAL,
    DEFAULT_EVAL_SEED,
    DEFAULT_EVAL_TEMPERATURE,
    DEFAULT_MATRIX_PRESET,
)
from experiments.matrix import (
    ALL_DATASET_TYPES,
    MATRIX_PRESETS,
    MatrixPreset,
    build_manifest,
    build_matrix_configs,
    load_configs_from_manifest,
    maybe_select_shard,
    save_manifest,
    summarize_configs,
)
from models import get_client


LEGACY_PRESETS = {"core16", "branching21"}


def _require_generator_label(raw: str | None) -> str:
    label = (raw or "").strip()
    if not label:
        raise ValueError("generator_dataset_label is required and cannot be blank")
    return label


def _validate_preset(preset: str) -> str:
    if preset in LEGACY_PRESETS:
        raise ValueError(
            f"Preset '{preset}' is archived. Use 'final5'. Legacy code is in archive/legacy_experiments/."
        )
    if preset not in MATRIX_PRESETS:
        valid = ", ".join(sorted(MATRIX_PRESETS.keys()))
        raise ValueError(f"Unknown preset '{preset}'. Valid presets: {valid}")
    return preset


def _results_path(cfg) -> Path:
    if cfg.entry_shards <= 1:
        return cfg.output_dir / "results.json"
    return (
        cfg.output_dir
        / "_partials"
        / f"entry_shard_{cfg.entry_shard_index}_of_{cfg.entry_shards}"
        / "results.json"
    )


def _resolve_configs(args: argparse.Namespace) -> list:
    if args.manifest:
        configs = load_configs_from_manifest(Path(args.manifest))
    else:
        output_base = Path(args.output_dir) if args.output_dir else RESULTS_DIR
        configs = build_matrix_configs(
            model=args.model,
            dataset_path=Path(args.dataset_path),
            generator_dataset_label=args.generator_dataset_label,
            dataset_types=args.dataset_types,
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
            entry_shards=args.entry_shards,
            entry_shard_index=args.entry_shard_index,
        )

    for cfg in configs:
        cfg.save_interval = args.save_interval
        cfg.entry_shards = args.entry_shards
        cfg.entry_shard_index = args.entry_shard_index
        if hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
            cfg.inference_batch_size = int(args.eval_batch_size)
        if hasattr(args, "vllm_max_num_batched_tokens"):
            cfg.vllm_max_num_batched_tokens = args.vllm_max_num_batched_tokens
        if hasattr(args, "vllm_max_num_seqs"):
            cfg.vllm_max_num_seqs = args.vllm_max_num_seqs
        if hasattr(args, "vllm_enable_chunked_prefill"):
            cfg.vllm_enable_chunked_prefill = args.vllm_enable_chunked_prefill

    return configs


def _print_summary(label: str, configs: list) -> None:
    summary = summarize_configs(configs)
    print(f"\n=== {label} ===")
    print(f"Total configs: {summary['total']}")
    print(f"By dataset type: {summary['by_dataset_type']}")
    print(f"By setting: {summary['by_setting']}")
    print(f"By mode: {summary['by_mode']}")


def _default_manifest_path(model: str, preset: str, generator_label: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    model_safe = model.replace("/", "_")
    label_safe = generator_label.replace("/", "_")
    return RESULTS_DIR / "manifests" / f"eval_matrix_{label_safe}_{model_safe}_{preset}_{ts}.json"


def cmd_plan(args: argparse.Namespace) -> int:
    args.preset = _validate_preset(args.preset)
    generator_label = _require_generator_label(args.generator_dataset_label)
    configs = _resolve_configs(args)
    _print_summary("Matrix Plan", configs)

    selected = maybe_select_shard(configs, args.num_shards, args.shard_index)
    if args.num_shards is not None:
        print(
            f"Shard selection: index={args.shard_index}/{args.num_shards - 1} -> {len(selected)} configs"
        )

    manifest = build_manifest(
        configs,
        preset=args.preset,
        model=args.model,
        dataset_path=Path(args.dataset_path),
        generator_dataset_label=generator_label,
        dataset_types=args.dataset_types,
        metadata={
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "choices_only": args.choices_only,
            "entry_shards": args.entry_shards,
            "entry_shard_index": args.entry_shard_index,
            "save_interval": args.save_interval,
        },
    )

    manifest_out = Path(args.manifest_out) if args.manifest_out else _default_manifest_path(args.model, args.preset, generator_label)
    save_manifest(manifest, manifest_out)
    print(f"Manifest written to: {manifest_out}")

    if args.print_configs:
        print("\nConfigs in run order:")
        for idx, cfg in enumerate(selected, start=1):
            print(
                f"  [{idx:3d}] {cfg.name} | setting={cfg.setting_id} | "
                f"dataset={cfg.dataset_type_filter} | mode={'choices_only' if cfg.choices_only else 'full_question'}"
            )
    return 0


def _client_cache_key(cfg) -> tuple:
    return (
        cfg.model_name,
        cfg.reasoning_effort,
        cfg.thinking_level,
        cfg.vllm_max_num_batched_tokens,
        cfg.vllm_max_num_seqs,
        cfg.vllm_enable_chunked_prefill,
    )


def _client_kwargs(cfg) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if cfg.reasoning_effort:
        kwargs["reasoning_effort"] = cfg.reasoning_effort
    if cfg.thinking_level:
        kwargs["thinking_level"] = cfg.thinking_level
    if cfg.model_name == "local" or "/" in cfg.model_name:
        if cfg.vllm_max_num_batched_tokens is not None:
            kwargs["max_num_batched_tokens"] = cfg.vllm_max_num_batched_tokens
        if cfg.vllm_max_num_seqs is not None:
            kwargs["max_num_seqs"] = cfg.vllm_max_num_seqs
        if cfg.vllm_enable_chunked_prefill is not None:
            kwargs["enable_chunked_prefill"] = cfg.vllm_enable_chunked_prefill
    return kwargs


def _load_or_create_client(cfg, cache: dict[tuple, Any]):
    key = _client_cache_key(cfg)
    if key in cache:
        return cache[key]
    client = get_client(cfg.model_name, **_client_kwargs(cfg))
    cache[key] = client
    return client


def _unload_clients(cache: dict[tuple, Any]) -> None:
    for client in cache.values():
        unload = getattr(client, "unload", None)
        if callable(unload):
            try:
                unload()
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: failed to unload client: {exc}")


def cmd_run(args: argparse.Namespace) -> int:
    args.preset = _validate_preset(args.preset)
    requested_label = _require_generator_label(args.generator_dataset_label)

    configs = _resolve_configs(args)
    configs = maybe_select_shard(configs, args.num_shards, args.shard_index)

    if args.skip_existing:
        before = len(configs)
        configs = [cfg for cfg in configs if not _results_path(cfg).exists()]
        skipped = before - len(configs)
        if skipped:
            print(f"Skipping {skipped} configs with existing results")

    if not configs:
        print("No configs to run.")
        return 0

    _print_summary("Run Set", configs)

    cache: dict[tuple, Any] = {}
    dataset_cache: dict[tuple[Any, ...], Any] = {}
    summaries = []

    try:
        for idx, cfg in enumerate(configs, start=1):
            print(f"\n[{idx}/{len(configs)}] {cfg.name}")
            try:
                client = _load_or_create_client(cfg, cache)
                result = run_experiment(cfg, client=client, dataset_cache=dataset_cache)
                random_baseline = 1.0 / (cfg.num_human + cfg.num_model + 1)
                summary = {
                    "name": cfg.name,
                    "setting_id": cfg.setting_id,
                    "config": cfg.distractor_config_str,
                    "dataset_type": cfg.dataset_type_filter,
                    "mode": "choices_only" if cfg.choices_only else "full_question",
                    "accuracy": result.accuracy,
                    "random_baseline": random_baseline,
                    "delta_over_random": result.accuracy - random_baseline,
                    "total": len(result.results),
                    "attempted_entries": result.attempted_entries,
                    "successful_entries": result.successful_entries,
                    "failed_entries": result.failed_entries,
                    "entry_failure_count": len(result.entry_failures),
                    "status": "success",
                    "output_path": str(_results_path(cfg)),
                }
            except Exception as exc:  # noqa: BLE001
                summary = {
                    "name": cfg.name,
                    "setting_id": cfg.setting_id,
                    "config": cfg.distractor_config_str,
                    "dataset_type": cfg.dataset_type_filter,
                    "mode": "choices_only" if cfg.choices_only else "full_question",
                    "accuracy": 0.0,
                    "random_baseline": 1.0 / (cfg.num_human + cfg.num_model + 1),
                    "delta_over_random": None,
                    "total": 0,
                    "attempted_entries": 0,
                    "successful_entries": 0,
                    "failed_entries": 0,
                    "entry_failure_count": 0,
                    "status": f"error: {exc}",
                    "output_path": str(_results_path(cfg)),
                }
            summaries.append(summary)
    finally:
        _unload_clients(cache)

    output_base = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_base.mkdir(parents=True, exist_ok=True)
    model_safe = (args.model or configs[0].model_name).replace("/", "_")
    label_safe = requested_label.replace("/", "_")
    mode_suffix = "choices_only" if args.choices_only else "full_question"

    summary_name = f"batch_summary_{label_safe}_{model_safe}_{mode_suffix}"
    if args.num_shards is not None and args.shard_index is not None:
        summary_name += f"_cfgshard_{args.shard_index}_of_{args.num_shards}"
    if args.entry_shards > 1:
        summary_name += f"_entryshard_{args.entry_shard_index}_of_{args.entry_shards}"
    summary_path = output_base / f"{summary_name}.json"

    payload = {
        "manifest_version": 2,
        "preset": args.preset,
        "model": args.model or configs[0].model_name,
        "generator_dataset_label": requested_label,
        "choices_only": args.choices_only,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "entry_shards": args.entry_shards,
        "entry_shard_index": args.entry_shard_index,
        "total_configs": len(summaries),
        "successful": sum(1 for x in summaries if x["status"] == "success"),
        "failed": sum(1 for x in summaries if x["status"] != "success"),
        "results": summaries,
    }

    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Batch summary saved: {summary_path}")

    failed = [x for x in summaries if x["status"] != "success"]
    if failed:
        print("Failures:")
        for x in failed:
            print(f"  - {x['name']}: {x['status']}")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Final5 matrix planner/runner")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    def add_common(p: argparse.ArgumentParser, required_inputs: bool) -> None:
        p.add_argument("--preset", type=str, default=DEFAULT_MATRIX_PRESET)
        p.add_argument("--model", type=str, required=required_inputs)
        p.add_argument("--dataset-path", type=str, required=required_inputs)
        p.add_argument(
            "--dataset-types",
            type=str,
            nargs="+",
            choices=ALL_DATASET_TYPES,
            default=ALL_DATASET_TYPES,
        )
        p.add_argument(
            "--distractor-source",
            "--distractor-sources",
            dest="distractor_sources",
            type=str,
            nargs="+",
            default=[],
            help="Deprecated/ignored in final5",
        )
        p.add_argument("--generator-dataset-label", type=str, required=True)
        p.add_argument("--limit", type=int)
        p.add_argument("--eval-mode", type=str, choices=["accuracy", "behavioral"], default=DEFAULT_EVAL_MODE)
        p.add_argument("--choices-only", action="store_true")
        p.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED)
        p.add_argument("--reasoning-effort", type=str)
        p.add_argument("--thinking-level", type=str)
        p.add_argument("--temperature", type=float, default=DEFAULT_EVAL_TEMPERATURE)
        p.add_argument("--max-tokens", type=int, default=DEFAULT_EVAL_MAX_TOKENS)
        p.add_argument("--save-interval", type=int, default=DEFAULT_EVAL_SAVE_INTERVAL)
        p.add_argument("--output-dir", type=str)
        p.add_argument("--num-shards", type=int)
        p.add_argument("--shard-index", type=int)
        p.add_argument("--entry-shards", type=int, default=1)
        p.add_argument("--entry-shard-index", type=int, default=0)

    plan = sub.add_parser("plan", help="Build final5 manifest")
    add_common(plan, required_inputs=True)
    plan.add_argument("--manifest-out", type=str)
    plan.add_argument("--print-configs", action="store_true")
    plan.set_defaults(handler=cmd_plan)

    run = sub.add_parser("run", help="Run final5 configs")
    add_common(run, required_inputs=False)
    run.add_argument("--manifest", type=str)
    run.add_argument("--skip-existing", action="store_true")
    run.add_argument("--eval-batch-size", type=int, default=8)
    run.add_argument("--vllm-max-num-batched-tokens", type=int, default=None)
    run.add_argument("--vllm-max-num-seqs", type=int, default=None)
    run.add_argument("--vllm-enable-chunked-prefill", type=lambda v: bool(int(v)), default=None)
    run.add_argument("--keep-checkpoints", type=int, default=DEFAULT_EVAL_KEEP_CHECKPOINTS)
    run.set_defaults(handler=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.entry_shards <= 0:
        parser.error("--entry-shards must be > 0")
    if args.entry_shard_index < 0 or args.entry_shard_index >= args.entry_shards:
        parser.error("--entry-shard-index must be in [0, entry_shards-1]")

    if args.subcommand == "run" and not args.manifest:
        missing = [name for name in ["model", "dataset_path"] if getattr(args, name) is None]
        if missing:
            parser.error("run requires --manifest OR both --model and --dataset-path")

    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
