# Sharding and Aggregation Guide

The Inspect-first refactor keeps cluster sharding, but removes the old merge architecture.

## Core Idea

- Stable sample ids are computed across the unified dataset.
- `--shard-count` and `--shard-index` deterministically partition those ids.
- Inspect `.eval` logs are written per shard.
- Analysis aggregates shard logs directly.

There is no canonical recombination step that rebuilds `summary.json` and `rows/` trees anymore.

## Shard Controls

Both `generate` and `evaluate` accept:

- `--shard-count`
- `--shard-index`
- `--shard-strategy`

Supported strategies:

- `contiguous`
- `modulo`

## Generation Example

```bash
uv run python main.py generate \
  --model gpt-5.2-2025-12-11 \
  --run-name gen_cluster \
  --processed-dataset datasets/processed/unified_processed_v2 \
  --shard-count 8 \
  --shard-index 0
```

## Evaluation Example

```bash
uv run python main.py evaluate \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --run-name eval_cluster \
  --generator-run-name gen_cluster \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v2 \
  --shard-count 8 \
  --shard-index 0
```

## Thin SLURM Helpers

Single-shard launchers:

- [`jobs/run_generate_shard.sh`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/jobs/run_generate_shard.sh)
- [`jobs/run_evaluate_shard.sh`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/jobs/run_evaluate_shard.sh)

Array templates:

- [`jobs/generate_array.sbatch`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/jobs/generate_array.sbatch)
- [`jobs/evaluate_array.sbatch`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/jobs/evaluate_array.sbatch)

Per-eval-model bundle generator:

```bash
uv run python scripts/05_build_eval_slurm_bundle.py \
  --run-name eval_cluster \
  --generator-run-name gen_cluster \
  --generator-model gpt-5.2-2025-12-11 \
  --output-dir jobs/generated/eval_cluster \
  --shard-count 8
```

Submit:

```bash
bash jobs/generated/eval_cluster/submit_all.sh
```

## Canonical Output Layout

Generation logs:

```text
results/inspect/generation/<run_name>/<generator_model>/**/*.eval
```

Evaluation logs:

```text
results/inspect/evaluation/<run_name>/<generator_run_name>/<generator_model>/<eval_model>/**/*.eval
```

Derived augmented cache:

```text
datasets/augmented/<generator_run_name>/<generator_model>/
```

## Aggregation

Use:

```bash
uv run python main.py analyze --results-root results/inspect/evaluation
```

The plotting and summary code walks every shard log. If all shards finish, the analysis sees the full run without a separate merge command.

`scripts/06_merge_eval_subshards.py` remains only as a compatibility message and intentionally performs no work.
