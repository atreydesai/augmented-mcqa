#!/usr/bin/env python3
"""Build a Final5 evaluation SLURM bundle with config sharding + entry sub-sharding."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVAL_MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "allenai/Olmo-3-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]

MODES = ["full_question", "choices_only"]
ACTIVE_DATASETS = ["arc_challenge", "mmlu_pro", "gpqa"]
SETTING_IDS = [
    "human_from_scratch",
    "model_from_scratch",
    "augment_human",
    "augment_model",
    "augment_ablation",
]


def _sanitize(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return cleaned.strip("_") or "x"


def _model_dir_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_generators(regen_manifest: dict[str, Any]) -> list[dict[str, str]]:
    generators: list[dict[str, str]] = []
    for item in regen_manifest.get("generators", []):
        if int(item.get("returncode", 1)) != 0:
            continue
        dataset_path = str(item.get("dataset_path", "")).strip()
        model = str(item.get("model", "")).strip()
        if not model or not dataset_path:
            continue
        generators.append({"model": model, "dataset_path": dataset_path})

    if not generators:
        raise ValueError("No successful generator runs found in manifest")

    return sorted(generators, key=lambda x: x["model"])


def _render_sbatch(
    *,
    job_name: str,
    num_gpus: int,
    generator_label: str,
    generator_dataset_path: str,
    eval_model: str,
    mode: str,
    entry_shards: int,
    entry_shard_index: int,
    output_base: str,
    save_interval: int,
    max_tokens: int,
) -> str:
    use_choices_only = "1" if mode == "choices_only" else "0"
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=${{LOG_DIR:-logs/final5_eval}}/{job_name}_%A_%a.out
#SBATCH --error=${{LOG_DIR:-logs/final5_eval}}/{job_name}_%A_%a.err
#SBATCH --array=0-{num_gpus - 1}
#SBATCH --partition=${{SLURM_PARTITION_OVERRIDE:-clip}}
#SBATCH --account=${{SLURM_ACCOUNT_OVERRIDE:-clip}}
#SBATCH --qos=${{SLURM_QOS_OVERRIDE:-high}}
#SBATCH --time=${{SLURM_TIME_OVERRIDE:-12:00:00}}
#SBATCH --mem=${{SLURM_MEM_OVERRIDE:-32G}}
#SBATCH --cpus-per-task=${{SLURM_CPUS_OVERRIDE:-4}}
#SBATCH --gres=${{SLURM_GRES_OVERRIDE:-gpu:rtxa6000:1}}

set -euo pipefail

PROJECT_ROOT="${{PROJECT_ROOT:-$SLURM_SUBMIT_DIR}}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
mkdir -p "${{LOG_DIR:-logs/final5_eval}}"

GENERATOR_LABEL="{generator_label}"
GENERATOR_DATASET_PATH="{generator_dataset_path}"
EVAL_MODEL="{eval_model}"
MODE="{mode}"
ENTRY_SHARDS="{entry_shards}"
ENTRY_SHARD_INDEX="{entry_shard_index}"
NUM_SHARDS="{num_gpus}"
SHARD_INDEX="$SLURM_ARRAY_TASK_ID"
OUTPUT_BASE="${{OUTPUT_BASE:-{output_base}}}"
SAVE_INTERVAL="${{SAVE_INTERVAL:-{save_interval}}}"
MAX_TOKENS="${{MAX_TOKENS:-{max_tokens}}}"

CMD=(
  uv run python scripts/eval_matrix.py run
  --preset final5
  --model "$EVAL_MODEL"
  --dataset-path "$GENERATOR_DATASET_PATH"
  --generator-dataset-label "$GENERATOR_LABEL"
  --dataset-types arc_challenge mmlu_pro gpqa
  --output-dir "$OUTPUT_BASE"
  --num-shards "$NUM_SHARDS"
  --shard-index "$SHARD_INDEX"
  --entry-shards "$ENTRY_SHARDS"
  --entry-shard-index "$ENTRY_SHARD_INDEX"
  --save-interval "$SAVE_INTERVAL"
  --max-tokens "$MAX_TOKENS"
  --skip-existing
)

if [[ "{use_choices_only}" == "1" ]]; then
  CMD+=(--choices-only)
fi

printf 'Running: %q ' "${{CMD[@]}}"
printf '\n'
"${{CMD[@]}}"
"""


def _render_submit_all(sbatch_files: list[Path]) -> str:
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        "DRY_RUN=0",
        "if [[ \"${1:-}\" == \"--dry-run\" ]]; then",
        "  DRY_RUN=1",
        "fi",
        "",
    ]

    for path in sorted(sbatch_files):
        rel = path.name
        lines.append(f'echo "Submitting {rel}"')
        lines.append('if [[ "$DRY_RUN" == "1" ]]; then')
        lines.append(f'  echo "  sbatch $SCRIPT_DIR/{rel}"')
        lines.append("else")
        lines.append(f'  sbatch "$SCRIPT_DIR/{rel}"')
        lines.append("fi")
        lines.append("")

    return "\n".join(lines) + "\n"


def _render_readme(
    *,
    bundle_manifest_path: Path,
    submit_path: Path,
    total_base_groups: int,
    entry_shards: int,
    num_gpus: int,
    total_job_groups: int,
    total_array_tasks: int,
) -> str:
    return f"""# Final5 Eval SLURM Bundle

Generated from regeneration manifest.

## Shard model

- Base groups = generator model x eval model x mode = {total_base_groups}
- Entry sub-shards = {entry_shards}
- Config shards per group (`--num-gpus`) = {num_gpus}

Formulas:

- Job groups = `base_groups * entry_shards` = `{total_base_groups} * {entry_shards} = {total_job_groups}`
- Array tasks submitted = `base_groups * entry_shards * num_gpus` = `{total_base_groups} * {entry_shards} * {num_gpus} = {total_array_tasks}`

## Files

- Bundle manifest: `{bundle_manifest_path.name}`
- Submit script: `{submit_path.name}`
- SBATCH scripts: one per `(generator, eval_model, mode, entry_shard_index)`

## Submit everything

```bash
bash {submit_path.name}
```

Dry run:

```bash
bash {submit_path.name} --dry-run
```

## Re-run failed array tasks

```bash
sbatch --array=1,4,7 <one-of-the-generated>.sbatch
```

## Merge entry sub-shards to canonical results

```bash
uv run python scripts/merge_eval_subshards.py --bundle-manifest {bundle_manifest_path} --strict
```
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Final5 eval SLURM bundle")
    parser.add_argument("--manifest", type=str, required=True, help="Path to regeneration manifest JSON")
    parser.add_argument(
        "--output-root",
        type=str,
        default="jobs/generated",
        help="Directory under which timestamped bundle dirs are created",
    )
    parser.add_argument("--output-dir", type=str, default="", help="Optional exact bundle output directory")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of config shards per job")
    parser.add_argument("--entry-shards", type=int, default=1, help="Row sub-shards per config")
    parser.add_argument("--output-base", type=str, default="results", help="Root results output directory")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.num_gpus <= 0:
        raise ValueError("--num-gpus must be > 0")
    if args.entry_shards <= 0:
        raise ValueError("--entry-shards must be > 0")

    regen_manifest_path = Path(args.manifest)
    regen_manifest = _load_manifest(regen_manifest_path)
    generators = _select_generators(regen_manifest)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    if args.output_dir:
        bundle_dir = Path(args.output_dir)
    else:
        bundle_dir = Path(args.output_root) / timestamp
    bundle_dir.mkdir(parents=True, exist_ok=True)

    sbatch_files: list[Path] = []
    jobs: list[dict[str, Any]] = []

    for gen in generators:
        generator_label = gen["model"]
        generator_dataset_path = gen["dataset_path"]

        for eval_model in EVAL_MODELS:
            for mode in MODES:
                for entry_shard_index in range(args.entry_shards):
                    name_parts = [
                        "final5",
                        _sanitize(generator_label),
                        _sanitize(eval_model),
                        mode,
                        f"entry{entry_shard_index}of{args.entry_shards}",
                    ]
                    job_name = "__".join(name_parts)
                    sbatch_name = f"{job_name}.sbatch"
                    sbatch_path = bundle_dir / sbatch_name

                    sbatch_text = _render_sbatch(
                        job_name=job_name,
                        num_gpus=args.num_gpus,
                        generator_label=generator_label,
                        generator_dataset_path=generator_dataset_path,
                        eval_model=eval_model,
                        mode=mode,
                        entry_shards=args.entry_shards,
                        entry_shard_index=entry_shard_index,
                        output_base=args.output_base,
                        save_interval=args.save_interval,
                        max_tokens=args.max_tokens,
                    )

                    sbatch_path.write_text(sbatch_text, encoding="utf-8")
                    sbatch_path.chmod(0o755)
                    sbatch_files.append(sbatch_path)

                    jobs.append(
                        {
                            "job_name": job_name,
                            "sbatch_file": str(sbatch_path),
                            "generator_label": generator_label,
                            "generator_dataset_path": generator_dataset_path,
                            "eval_model": eval_model,
                            "mode": mode,
                            "entry_shards": args.entry_shards,
                            "entry_shard_index": entry_shard_index,
                            "num_shards": args.num_gpus,
                        }
                    )

    submit_all_path = bundle_dir / "submit_all.sh"
    submit_all_path.write_text(_render_submit_all(sbatch_files), encoding="utf-8")
    submit_all_path.chmod(0o755)

    total_base_groups = len(generators) * len(EVAL_MODELS) * len(MODES)
    total_job_groups = total_base_groups * args.entry_shards
    total_array_tasks = total_job_groups * args.num_gpus
    limit_per_dataset = int(regen_manifest.get("limit_per_dataset", 1000) or 1000)
    expected_eval_rows = (
        len(generators)
        * len(EVAL_MODELS)
        * len(MODES)
        * len(ACTIVE_DATASETS)
        * len(SETTING_IDS)
        * limit_per_dataset
    )

    config_roots = []
    for gen in generators:
        generator_label = gen["model"]
        for eval_model in EVAL_MODELS:
            eval_dir = _model_dir_name(eval_model)
            for mode in MODES:
                for dataset in ACTIVE_DATASETS:
                    for setting in SETTING_IDS:
                        config_roots.append(
                            str(Path(args.output_base) / generator_label / eval_dir / mode / dataset / setting)
                        )

    bundle_manifest = {
        "bundle_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "regeneration_manifest": str(regen_manifest_path),
        "output_base": args.output_base,
        "num_gpus": args.num_gpus,
        "entry_shards": args.entry_shards,
        "eval_models": list(EVAL_MODELS),
        "modes": list(MODES),
        "datasets": list(ACTIVE_DATASETS),
        "setting_ids": list(SETTING_IDS),
        "generators": generators,
        "total_base_groups": total_base_groups,
        "total_job_groups": total_job_groups,
        "total_array_tasks": total_array_tasks,
        "limit_per_dataset": limit_per_dataset,
        "expected_eval_rows": expected_eval_rows,
        "jobs": jobs,
        "config_roots": sorted(set(config_roots)),
    }

    bundle_manifest_path = bundle_dir / "bundle_manifest.json"
    bundle_manifest_path.write_text(json.dumps(bundle_manifest, indent=2), encoding="utf-8")

    readme_path = bundle_dir / "README.md"
    readme_path.write_text(
        _render_readme(
            bundle_manifest_path=bundle_manifest_path,
            submit_path=submit_all_path,
            total_base_groups=total_base_groups,
            entry_shards=args.entry_shards,
            num_gpus=args.num_gpus,
            total_job_groups=total_job_groups,
            total_array_tasks=total_array_tasks,
        ),
        encoding="utf-8",
    )

    print(f"Bundle directory: {bundle_dir}")
    print(f"SBATCH files: {len(sbatch_files)}")
    print(f"Base groups: {total_base_groups}")
    print(f"Job groups (with entry shards): {total_job_groups}")
    print(f"Array tasks submitted: {total_array_tasks}")
    print(f"Submit all: {submit_all_path}")
    print(f"Manifest: {bundle_manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
