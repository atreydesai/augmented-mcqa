# Cluster Scheduling and Low-Level Sharding

The Inspect-first refactor keeps low-level sharding, but the primary cluster path is now dataset-aware SLURM submit commands.

## Primary Cluster Shape

Use one login-node command per stage:

```bash
uv run python main.py submit-generate-cluster --run-name gen_cluster
uv run python main.py submit-evaluate-cluster --run-name eval_cluster --generator-run-name gen_cluster --generator-model Qwen/Qwen3-4B-Instruct-2507
```

Each command:

- only supports local `vllm/...` models
- creates one SLURM array task per `model × dataset`
- keeps settings and modes grouped inside the task
- avoids repeated cold starts within that job
- still pays one cold start per `model × dataset` task
- writes bundle files to `jobs/generated/<stage>/<run>/`
- writes logs to `logs/slurm/<stage>/<run>/`

If you pass `--gpu-count 4`, the sbatch array renders `%4`. If you omit it, there is no concurrency cap in the array specification.

## Why There Is No Recombination Step

Inspect `.eval` logs are the canonical artifact. Analysis walks all logs directly, so there is no merge command that rebuilds `summary.json` and `rows/`.

## Low-Level Shard Controls

Both direct `generate` and `evaluate` still accept:

- `--shard-count`
- `--shard-index`
- `--shard-strategy`

Supported strategies:

- `contiguous`
- `modulo`

Use those flags only if one `model × dataset` task is still too large and you need to split it manually.

Example:

```bash
uv run python main.py evaluate \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --run-name eval_sharded \
  --generator-run-name gen_cluster \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v2 \
  --dataset-types mmlu_pro \
  --shard-count 8 \
  --shard-index 3
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

Cluster submit generation writes dataset-scoped caches under `datasets/augmented/<generator_run_name>/<generator_model>/<dataset>/`.

## Aggregation

Use:

```bash
uv run python main.py analyze --results-root results/inspect/evaluation
```

The plotting and summary code walks every log. If all tasks finish, the analysis sees the full run without a separate merge command.
