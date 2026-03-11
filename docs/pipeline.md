# Pipeline Guide

This repo now runs an Inspect-first Final5 workflow. The old manifest/merge pipeline is not canonical anymore.

## Flow

```text
prepare-data -> generate -> evaluate -> analyze/export
```

Inspect `.eval` logs are the source of truth for both generation and evaluation. The augmented dataset under `datasets/augmented/` is rebuilt from generation logs when needed.

## Primary CLI

Use [`main.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/main.py) directly.

### 1. Prepare data

```bash
uv run python main.py prepare-data --step all --output-path datasets/processed/unified_processed_v2
```

### 2. Generate Final5 distractors

Single model:

```bash
uv run python main.py generate \
  --model gpt-5.2-2025-12-11 \
  --run-name gen_openai \
  --processed-dataset datasets/processed/unified_processed_v2 \
  --materialize-cache
```

All default generators:

```bash
uv run python main.py generate-all \
  --run-name gen_all \
  --processed-dataset datasets/processed/unified_processed_v2 \
  --materialize-cache
```

Generation is plain-text only. The solver expects labeled lines like `B. ...`, `C. ...`, `D. ...` and retries on parse failure.

### 3. Evaluate

Single evaluation model:

```bash
uv run python main.py evaluate \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --run-name eval_local \
  --generator-run-name gen_openai \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v2
```

All default evaluation models:

```bash
uv run python main.py evaluate-all \
  --run-name eval_all \
  --generator-run-name gen_openai \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v2
```

### 3a. Submit local cluster generation

```bash
uv run python main.py submit-generate-cluster \
  --run-name gen_cluster \
  --processed-dataset datasets/processed/unified_processed_v2
```

### 3b. Submit local cluster evaluation

```bash
uv run python main.py submit-evaluate-cluster \
  --run-name eval_cluster \
  --generator-run-name gen_openai \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v2 \
  --gpu-count 4
```

Both cluster commands:

- are for local `vllm/...` models only
- create one SLURM array task per `model × dataset`
- default to 9 jobs with the current 3 local models and 3 datasets
- avoid repeated cold starts between configs and modes inside a task
- still incur one cold start per `model × dataset` task
- write manifests and sbatch scripts under `jobs/generated/<stage>/<run>/`
- write SLURM logs under `logs/slurm/<stage>/<run>/`
- use dataset-scoped augmented caches under `datasets/augmented/<run>/<model>/<dataset>/`

If `--gpu-count` is omitted, the array is submitted with no `%N` concurrency cap.

### 4. Analyze

```bash
uv run python main.py analyze \
  --results-root results/inspect/evaluation \
  --output-dir results/final5_plots
```

### 5. Export

```bash
uv run python main.py export \
  --generator-run-name gen_openai \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v2
```

### 6. Benchmarker Writing-Flaw Analysis

```bash
uv run python analysis/benchmarker_analysis.py \
  --writing-flaw-jsonl datasets/benchmarker_results/atrey_writing_flaw_rows.jsonl.zip \
  --results-root results/inspect/evaluation \
  --cache-root datasets/augmented \
  --generator-run-name gen_openai \
  --generator-model gpt-5.2-2025-12-11 \
  --output-dir analysis/figures/benchmarker
```

## API and Local Backends

Hosted APIs:

```bash
uv run python main.py generate \
  --model openai/gpt-5.2-2025-12-11 \
  --run-name gen_api \
  --processed-dataset datasets/processed/unified_processed_v2
```

OpenAI-compatible local server:

```bash
uv run python main.py evaluate \
  --model my-local-model \
  --backend openai \
  --model-base-url http://localhost:8000/v1 \
  --run-name eval_local_api \
  --generator-run-name gen_api \
  --generator-model openai/gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v2
```

vLLM alias:

```bash
uv run python main.py evaluate \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --run-name eval_vllm \
  --generator-run-name gen_api \
  --generator-model openai/gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v2
```

## Sharding

Both generation and evaluation support:

- `--shard-count`
- `--shard-index`
- `--shard-strategy`

Example:

```bash
uv run python main.py evaluate \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --run-name eval_sharded \
  --generator-run-name gen_openai \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v2 \
  --shard-count 8 \
  --shard-index 3
```

Those flags are now the low-level escape hatch. The preferred login-node path is `submit-generate-cluster` / `submit-evaluate-cluster`.

## Additional CLI Utilities

The same CLI also provides:

- `main.py submit-generate-cluster`
- `main.py submit-evaluate-cluster`
- `main.py signature-table`
- `main.py diagnose-failures`
- `main.py diagnose-trace`
- `main.py smoke-generate`
- `main.py smoke-evaluate`

## Repo Structure

- [`tasks/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/tasks): Inspect task factories
- [`solvers/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/solvers): generation/evaluation prompt logic
- [`scorers/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/scorers): Final5 score metadata
- [`data/final5_store.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/data/final5_store.py): dataset adapters and cache materialization
- [`utils/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/utils): parsing, sharding, model aliasing, log helpers
- [`prompts/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/prompts): plain-text templates
