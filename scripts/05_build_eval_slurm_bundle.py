#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.constants import DEFAULT_EVALUATION_MODELS
from utils.modeling import resolve_model_name, safe_name


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}_%A_%a.out
#SBATCH --error={log_dir}/{job_name}_%A_%a.err
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --time={time_limit}
#SBATCH --mem={memory}
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres={gres}
#SBATCH --array=0-{array_end}

set -euo pipefail

PROJECT_ROOT="${{PROJECT_ROOT:-$SLURM_SUBMIT_DIR}}"
cd "$PROJECT_ROOT"
PYTHON_BIN="${{PYTHON_BIN:-python}}"

"$PYTHON_BIN" main.py evaluate \\
  --model "{eval_model}" \\
  --run-name "{run_name}" \\
  --generator-run-name "{generator_run_name}" \\
  --generator-model "{generator_model}" \\
  --processed-dataset "{processed_dataset}" \\
  --shard-count {shard_count} \\
  --shard-index "${{SLURM_ARRAY_TASK_ID}}" \\
  --shard-strategy "{shard_strategy}" \\
  --generation-log-root "{generation_log_root}" \\
  --cache-root "{cache_root}" \\
  --log-root "{evaluation_log_root}"{extra_args}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build thin SLURM wrappers for Inspect-first evaluation")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--generator-run-name", required=True)
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--processed-dataset", default="datasets/processed/unified_processed_v2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evaluation-models", default=",".join(DEFAULT_EVALUATION_MODELS))
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--shard-strategy", choices=["contiguous", "modulo"], default="contiguous")
    parser.add_argument("--settings", default=None)
    parser.add_argument("--modes", default=None)
    parser.add_argument("--generation-log-root", default="results/inspect/generation")
    parser.add_argument("--evaluation-log-root", default="results/inspect/evaluation")
    parser.add_argument("--cache-root", default="datasets/augmented")
    parser.add_argument("--partition", default="clip")
    parser.add_argument("--account", default="clip")
    parser.add_argument("--time-limit", default="12:00:00")
    parser.add_argument("--memory", default="32G")
    parser.add_argument("--cpus", default="4")
    parser.add_argument("--gres", default="gpu:1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    models = [resolve_model_name(item.strip()) for item in args.evaluation_models.split(",") if item.strip()]
    sbatch_files: list[Path] = []

    for model in models:
        extra_args = []
        if args.settings:
            extra_args.extend(["--settings", args.settings])
        if args.modes:
            extra_args.extend(["--modes", args.modes])
        extra_blob = "".join(f" \\\n  {item}" for item in extra_args)
        job_name = f"final5_eval_{safe_name(model)}"
        content = SBATCH_TEMPLATE.format(
            job_name=job_name,
            log_dir=str(log_dir),
            partition=args.partition,
            account=args.account,
            time_limit=args.time_limit,
            memory=args.memory,
            cpus=args.cpus,
            gres=args.gres,
            array_end=args.shard_count - 1,
            eval_model=model,
            run_name=args.run_name,
            generator_run_name=args.generator_run_name,
            generator_model=args.generator_model,
            processed_dataset=args.processed_dataset,
            shard_count=args.shard_count,
            shard_strategy=args.shard_strategy,
            generation_log_root=args.generation_log_root,
            cache_root=args.cache_root,
            evaluation_log_root=args.evaluation_log_root,
            extra_args=extra_blob,
        )
        path = out_dir / f"{job_name}.sbatch"
        path.write_text(content, encoding="utf-8")
        sbatch_files.append(path)

    submit_lines = ["#!/bin/bash", "set -euo pipefail"]
    for path in sbatch_files:
        submit_lines.append(f"sbatch {path.name}")
    submit_path = out_dir / "submit_all.sh"
    submit_path.write_text("\n".join(submit_lines) + "\n", encoding="utf-8")
    print(submit_path)
    for path in sbatch_files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
