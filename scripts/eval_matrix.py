#!/usr/bin/env python3
"""Primary matrix CLI for deterministic, sequential evaluation runs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RESULTS_DIR
from experiments import run_experiment
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


def _add_common_build_args(parser: argparse.ArgumentParser, required_inputs: bool) -> None:
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(MATRIX_PRESETS.keys()),
        default="core16",
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

    parser.add_argument("--limit", type=int, help="Limit entries per config")
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["accuracy", "behavioral"],
        default="behavioral",
    )
    parser.add_argument("--choices-only", action="store_true", help="Use choices-only prompt")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--reasoning-effort", type=str, help="OpenAI reasoning effort")
    parser.add_argument("--thinking-level", type=str, help="Anthropic/Gemini thinking level")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=100)

    parser.add_argument("--output-dir", type=str, help="Base output directory")


def _add_shard_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-shards", type=int, help="Total shard count")
    parser.add_argument("--shard-index", type=int, help="0-based shard index")


def _resolve_configs(args: argparse.Namespace) -> list:
    if getattr(args, "manifest", None):
        configs = load_configs_from_manifest(Path(args.manifest))
        return configs

    output_base = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    return build_matrix_configs(
        model=args.model,
        dataset_path=Path(args.dataset_path),
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
    )


def _print_summary(label: str, configs: list) -> None:
    summary = summarize_configs(configs)
    print(f"\n=== {label} ===")
    print(f"Total configs: {summary['total']}")
    print(f"By dataset type: {summary['by_dataset_type']}")
    print(f"By distractor source: {summary['by_distractor_source']}")


def _default_manifest_path(model: str, preset: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    model_safe = model.replace("/", "_")
    return RESULTS_DIR / "manifests" / f"eval_matrix_{model_safe}_{preset}_{ts}.json"


def cmd_plan(args: argparse.Namespace) -> int:
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
        dataset_types=args.dataset_types,
        distractor_sources=args.distractor_sources,
        metadata={
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "choices_only": args.choices_only,
        },
    )

    manifest_out = Path(args.manifest_out) if args.manifest_out else _default_manifest_path(args.model, args.preset)
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


def _summary_path(model: str, output_base: Path, num_shards: int | None, shard_index: int | None) -> Path:
    safe_model = model.replace("/", "_")
    if num_shards is not None and shard_index is not None:
        filename = f"batch_summary_{safe_model}_shard_{shard_index}_of_{num_shards}.json"
    else:
        filename = f"batch_summary_{safe_model}.json"
    return output_base / filename


def _run_single_config(config) -> dict[str, Any]:
    try:
        results = run_experiment(config)
        return {
            "name": config.name,
            "config": config.distractor_config_str,
            "accuracy": results.accuracy,
            "total": len(results.results),
            "status": "success",
            "output_dir": str(config.output_dir),
        }
    except Exception as exc:
        return {
            "name": config.name,
            "config": config.distractor_config_str,
            "accuracy": 0.0,
            "total": 0,
            "status": f"error: {exc}",
            "output_dir": str(config.output_dir),
        }


def cmd_run(args: argparse.Namespace) -> int:
    configs = _resolve_configs(args)
    if not configs:
        print("No configs available to run.")
        return 0

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

    _print_summary("Run Set", configs)

    summaries = []
    for idx, config in enumerate(configs, start=1):
        print(f"\n[{idx}/{len(configs)}] {config.name}")
        summary = _run_single_config(config)
        summaries.append(summary)
        if summary["status"] == "success":
            print(f"  Accuracy: {summary['accuracy']:.2%}")
        else:
            print(f"  FAILED: {summary['status']}")

    model_name = configs[0].model_name
    output_base = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_base.mkdir(parents=True, exist_ok=True)
    summary_path = _summary_path(model_name, output_base, args.num_shards, args.shard_index)

    payload = {
        "model": model_name,
        "preset": args.preset,
        "manifest": args.manifest,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "total_configs": len(summaries),
        "successful": sum(1 for s in summaries if s["status"] == "success"),
        "failed": sum(1 for s in summaries if s["status"] != "success"),
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
    run_parser.set_defaults(handler=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

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
