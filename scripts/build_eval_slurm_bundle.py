#!/usr/bin/env python3
"""Build a Final5 evaluation SLURM bundle with balanced per-pair work units."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_from_disk

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


def _load_dataset_row_counts(dataset_path: str) -> dict[str, int]:
    ds = load_from_disk(dataset_path)
    if not hasattr(ds, "keys"):
        raise ValueError(f"Expected DatasetDict at {dataset_path}")

    counts = {}
    for dataset in ACTIVE_DATASETS:
        counts[dataset] = int(len(ds[dataset])) if dataset in ds else 0
    return counts


def _build_part_counts(row_counts: dict[str, int], target_rows_per_subsplit: int) -> dict[str, int]:
    if target_rows_per_subsplit <= 0:
        raise ValueError("target_rows_per_subsplit must be > 0")

    parts: dict[str, int] = {}
    for dataset in ACTIVE_DATASETS:
        rows = int(row_counts.get(dataset, 0))
        parts[dataset] = max(1, int(math.ceil(rows / target_rows_per_subsplit)))
    return parts


def _contiguous_shard_size(total_rows: int, shard_total: int, shard_idx: int) -> int:
    base = total_rows // shard_total
    remainder = total_rows % shard_total
    return base + (1 if shard_idx < remainder else 0)


def _build_work_units(
    *,
    row_counts: dict[str, int],
    part_counts: dict[str, int],
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    idx = 0
    for mode in MODES:
        for dataset in ACTIVE_DATASETS:
            entry_shards = int(part_counts[dataset])
            total_rows = int(row_counts[dataset])
            for entry_shard_index in range(entry_shards):
                expected_rows = _contiguous_shard_size(total_rows, entry_shards, entry_shard_index)
                units.append(
                    {
                        "array_index": idx,
                        "mode": mode,
                        "dataset": dataset,
                        "entry_shards": entry_shards,
                        "entry_shard_index": entry_shard_index,
                        "entry_shard_strategy": "contiguous",
                        "choices_only": bool(mode == "choices_only"),
                        "expected_rows": expected_rows,
                    }
                )
                idx += 1
    return units


def _render_sbatch(
    *,
    job_name: str,
    generator_label: str,
    generator_dataset_path: str,
    eval_model: str,
    output_base: str,
    save_interval: int,
    max_tokens: int,
    work_units_filename: str,
    num_work_units: int,
) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=${{LOG_DIR:-logs/final5_eval}}/{job_name}_%A_%a.out
#SBATCH --error=${{LOG_DIR:-logs/final5_eval}}/{job_name}_%A_%a.err
#SBATCH --array=0-{num_work_units - 1}
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
DEFAULT_VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
if [[ ! -f "$DEFAULT_VENV_ACTIVATE" ]]; then
  DEFAULT_VENV_ACTIVATE="/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/.venv/bin/activate"
fi
VENV_ACTIVATE="${{VENV_ACTIVATE:-$DEFAULT_VENV_ACTIVATE}}"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Error: venv activate script not found: $VENV_ACTIVATE"
  exit 1
fi
source "$VENV_ACTIVATE"
PYTHON_BIN="${{PYTHON_BIN:-$(dirname "$VENV_ACTIVATE")/python}}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Error: python executable not found after activating venv."
  exit 1
fi
mkdir -p "${{LOG_DIR:-logs/final5_eval}}"

GENERATOR_LABEL="{generator_label}"
GENERATOR_DATASET_PATH="{generator_dataset_path}"
EVAL_MODEL="{eval_model}"
OUTPUT_BASE="${{OUTPUT_BASE:-{output_base}}}"
SAVE_INTERVAL="${{SAVE_INTERVAL:-{save_interval}}}"
MAX_TOKENS="${{MAX_TOKENS:-{max_tokens}}}"
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
WORK_UNITS_FILE="${{WORK_UNITS_FILE:-$SCRIPT_DIR/{work_units_filename}}}"

readarray -t UNIT_EXPORTS < <("$PYTHON_BIN" - "$WORK_UNITS_FILE" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json
import shlex
import sys

units = json.loads(open(sys.argv[1], encoding='utf-8').read())
idx = int(sys.argv[2])
if idx < 0 or idx >= len(units):
    raise SystemExit(f"Invalid SLURM_ARRAY_TASK_ID={{idx}} for work_units size={{len(units)}}")
unit = units[idx]
for key in ("mode", "dataset", "entry_shards", "entry_shard_index", "entry_shard_strategy", "choices_only", "expected_rows"):
    val = unit[key]
    print(f"{{key.upper()}}={{shlex.quote(str(val))}}")
PY
)

for kv in "${{UNIT_EXPORTS[@]}}"; do
  eval "$kv"
done

CMD=(
  "$PYTHON_BIN" scripts/eval_matrix.py run
  --preset final5
  --model "$EVAL_MODEL"
  --dataset-path "$GENERATOR_DATASET_PATH"
  --generator-dataset-label "$GENERATOR_LABEL"
  --dataset-types "$DATASET"
  --output-dir "$OUTPUT_BASE"
  --entry-shards "$ENTRY_SHARDS"
  --entry-shard-index "$ENTRY_SHARD_INDEX"
  --entry-shard-strategy "$ENTRY_SHARD_STRATEGY"
  --save-interval "$SAVE_INTERVAL"
  --max-tokens "$MAX_TOKENS"
  --skip-existing
)

if [[ "$CHOICES_ONLY" == "True" || "$CHOICES_ONLY" == "true" || "$CHOICES_ONLY" == "1" ]]; then
  CMD+=(--choices-only)
fi

printf 'Running task %s | mode=%s dataset=%s part=%s/%s expected_rows=%s\\n' \
  "$SLURM_ARRAY_TASK_ID" "$MODE" "$DATASET" "$ENTRY_SHARD_INDEX" "$ENTRY_SHARDS" "$EXPECTED_ROWS"
printf 'Running: %q ' "${{CMD[@]}}"
printf '\\n'
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
    total_pairs: int,
    total_array_tasks: int,
    target_rows_per_subsplit: int,
) -> str:
    return f"""# Final5 Eval SLURM Bundle (Per-Pair Work Units)

Generated from regeneration manifest.

## Shard model

- Pair groups = generator model x eval model = {total_pairs}
- Modes per pair = 2 (`full_question`, `choices_only`)
- Per-dataset part counts are dynamic from row counts with target rows/subsplit = {target_rows_per_subsplit}
- One sbatch file per pair, with array indexes mapped via `work_units.json`

## Files

- Bundle manifest: `{bundle_manifest_path.name}`
- Submit script: `{submit_path.name}`
- SBATCH scripts: one per `(generator, eval_model)`
- Work unit maps: one JSON per sbatch file

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

## Quick stats

- Pair groups: {total_pairs}
- Total array tasks across all pair sbatch files: {total_array_tasks}
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
    parser.add_argument(
        "--target-rows-per-subsplit",
        type=int,
        default=500,
        help="Target rows per dataset sub-split used for balanced work-unit sizing",
    )
    parser.add_argument("--output-base", type=str, default="results", help="Root results output directory")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.target_rows_per_subsplit <= 0:
        raise ValueError("--target-rows-per-subsplit must be > 0")

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

    dataset_row_counts: dict[str, dict[str, int]] = {}
    dataset_part_counts: dict[str, dict[str, int]] = {}

    for gen in generators:
        label = gen["model"]
        row_counts = _load_dataset_row_counts(gen["dataset_path"])
        part_counts = _build_part_counts(row_counts, args.target_rows_per_subsplit)
        dataset_row_counts[label] = row_counts
        dataset_part_counts[label] = part_counts

    expected_entry_shards_by_config_root: dict[str, int] = {}
    config_roots: list[str] = []

    for gen in generators:
        generator_label = gen["model"]
        generator_dataset_path = gen["dataset_path"]
        row_counts = dataset_row_counts[generator_label]
        part_counts = dataset_part_counts[generator_label]
        work_units = _build_work_units(row_counts=row_counts, part_counts=part_counts)
        num_work_units = len(work_units)

        for eval_model in EVAL_MODELS:
            job_name = "__".join(
                [
                    "final5_pair",
                    _sanitize(generator_label),
                    _sanitize(eval_model),
                ]
            )
            sbatch_name = f"{job_name}.sbatch"
            work_units_name = f"{job_name}.work_units.json"
            sbatch_path = bundle_dir / sbatch_name
            work_units_path = bundle_dir / work_units_name

            work_units_path.write_text(json.dumps(work_units, indent=2), encoding="utf-8")

            sbatch_text = _render_sbatch(
                job_name=job_name,
                generator_label=generator_label,
                generator_dataset_path=generator_dataset_path,
                eval_model=eval_model,
                output_base=args.output_base,
                save_interval=args.save_interval,
                max_tokens=args.max_tokens,
                work_units_filename=work_units_name,
                num_work_units=num_work_units,
            )

            sbatch_path.write_text(sbatch_text, encoding="utf-8")
            sbatch_path.chmod(0o755)
            sbatch_files.append(sbatch_path)

            jobs.append(
                {
                    "job_name": job_name,
                    "sbatch_file": str(sbatch_path),
                    "work_units_file": str(work_units_path),
                    "generator_label": generator_label,
                    "generator_dataset_path": generator_dataset_path,
                    "eval_model": eval_model,
                    "num_work_units": num_work_units,
                    "dataset_row_counts": row_counts,
                    "dataset_part_counts": part_counts,
                    "work_units": work_units,
                }
            )

            eval_dir = _model_dir_name(eval_model)
            for mode in MODES:
                for dataset in ACTIVE_DATASETS:
                    expected_shards = int(part_counts[dataset])
                    for setting in SETTING_IDS:
                        root = Path(args.output_base) / generator_label / eval_dir / mode / dataset / setting
                        key = str(root)
                        expected_entry_shards_by_config_root[key] = expected_shards
                        config_roots.append(key)

    submit_all_path = bundle_dir / "submit_all.sh"
    submit_all_path.write_text(_render_submit_all(sbatch_files), encoding="utf-8")
    submit_all_path.chmod(0o755)

    total_pairs = len(generators) * len(EVAL_MODELS)
    total_array_tasks = sum(int(job["num_work_units"]) for job in jobs)
    expected_eval_rows = 0
    for gen in generators:
        generator_label = gen["model"]
        rows_total = sum(int(dataset_row_counts[generator_label].get(ds, 0)) for ds in ACTIVE_DATASETS)
        expected_eval_rows += rows_total * len(EVAL_MODELS) * len(MODES) * len(SETTING_IDS)

    bundle_manifest = {
        "bundle_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "regeneration_manifest": str(regen_manifest_path),
        "output_base": args.output_base,
        "target_rows_per_subsplit": args.target_rows_per_subsplit,
        "eval_models": list(EVAL_MODELS),
        "modes": list(MODES),
        "datasets": list(ACTIVE_DATASETS),
        "setting_ids": list(SETTING_IDS),
        "generators": generators,
        "dataset_row_counts": dataset_row_counts,
        "dataset_part_counts": dataset_part_counts,
        "total_pairs": total_pairs,
        "total_sbatch_files": len(sbatch_files),
        "total_array_tasks": total_array_tasks,
        "expected_eval_rows": expected_eval_rows,
        "jobs": jobs,
        "config_roots": sorted(set(config_roots)),
        "expected_entry_shards_by_config_root": expected_entry_shards_by_config_root,
    }

    bundle_manifest_path = bundle_dir / "bundle_manifest.json"
    bundle_manifest_path.write_text(json.dumps(bundle_manifest, indent=2), encoding="utf-8")

    readme_path = bundle_dir / "README.md"
    readme_path.write_text(
        _render_readme(
            bundle_manifest_path=bundle_manifest_path,
            submit_path=submit_all_path,
            total_pairs=total_pairs,
            total_array_tasks=total_array_tasks,
            target_rows_per_subsplit=args.target_rows_per_subsplit,
        ),
        encoding="utf-8",
    )

    print(f"Bundle directory: {bundle_dir}")
    print(f"SBATCH files: {len(sbatch_files)}")
    print(f"Total pairs: {total_pairs}")
    print(f"Total array tasks: {total_array_tasks}")
    print(f"Submit all: {submit_all_path}")
    print(f"Manifest: {bundle_manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
