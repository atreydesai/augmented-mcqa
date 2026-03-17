# CLI Reference

This page is for day-to-day command usage.

Use `uv` for all commands:

```bash
uv run python main.py --help
```

## Core Inputs

### Processed dataset input

`--processed-dataset` accepts either:

- a processed Hugging Face `DatasetDict`, usually `datasets/processed/unified_processed_v3`
- a dataset manifest JSON for custom datasets

### Model input

`--model`, `--models`, and `--generator-model` must resolve through `config/model_registry.json`.

### Augmented output

Generated data is stored at:

```text
datasets/augmented/<run>/<model>/<dataset>/<setting>/
```

The store root also contains:

```text
datasets/augmented/<run>/<model>/augmented_manifest.json
```

## Commands

### `prepare-data`

Build the standard processed dataset:

```bash
uv run python main.py prepare-data \
  --step all \
  --output-path datasets/processed/unified_processed_v3
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--step` | `download`, `process`, or `all` |
| `--dataset` | limit download to one raw dataset |
| `--all` | download every supported raw dataset |
| `--output-dir` | override the raw-data directory |
| `--output-path` | processed dataset output path |
| `--limit` | cap rows per dataset for debugging |

### `generate`

Run one generation model directly:

```bash
uv run python main.py generate \
  --model gpt-5.2-2025-12-11 \
  --run-name gen_gpt52 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --materialize-cache
```

Realistic variants:

Generate one dataset only:

```bash
uv run python main.py generate \
  --model gpt-5.2-2025-12-11 \
  --run-name gen_gpqa \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --dataset-types gpqa \
  --materialize-cache
```

Generate one strategy only:

```bash
uv run python main.py generate \
  --model claude-opus-4-6 \
  --run-name gen_claude_model_only \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --generation-strategies model_from_scratch \
  --materialize-cache
```

Generate a small smoke slice:

```bash
uv run python main.py generate \
  --model gemini-3.1-pro-preview \
  --run-name gen_gemini_smoke \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --limit 5 \
  --materialize-cache
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--model` | generation model from the registry |
| `--run-name` | run name used in log and store paths |
| `--processed-dataset` | processed dataset dir or dataset manifest JSON |
| `--dataset-types` | subset of datasets |
| `--generation-strategies` | subset of schedulable generation settings |
| `--question-start` | per-dataset starting row |
| `--limit` | per-dataset row limit |
| `--log-root` | generation log root |
| `--cache-root` | augmented store root |
| `--augmented-dataset` | exact augmented store path |
| `--materialize-cache` | rebuild the store after generation |
| `--rebuild-cache` | force store regeneration |

Also supports runtime flags such as `--max-tokens`, `--temperature`, `--reasoning-effort`, and sharding flags such as `--shard-count`.

### `generate-all`

Run the default generation model set:

```bash
uv run python main.py generate-all \
  --run-name gen_all_defaults \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --materialize-cache
```

Override the model list:

```bash
uv run python main.py generate-all \
  --run-name gen_custom_set \
  --models gpt-5.2-2025-12-11,claude-opus-4-6 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --materialize-cache
```

### `materialize-generation-cache`

Rebuild the setting-scoped augmented store from existing generation logs:

```bash
uv run python main.py materialize-generation-cache \
  --run-name gen_gpt52 \
  --model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3
```

Use this when:

- generation already finished
- you edited store-building logic
- you want to refresh the derived store without rerunning generation

### `evaluate`

Run evaluation for one model directly:

```bash
uv run python main.py evaluate \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --run-name eval_qwen4b \
  --generator-run-name gen_gpt52 \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3
```

Realistic variants:

Evaluate one setting only:

```bash
uv run python main.py evaluate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --run-name eval_aug_model \
  --generator-run-name gen_gpt52 \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --settings augment_model
```

Evaluate one mode only:

```bash
uv run python main.py evaluate \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --run-name eval_choices \
  --generator-run-name gen_gpt52 \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --modes choices_only
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--model` | evaluation model from the registry |
| `--run-name` | evaluation run name |
| `--generator-run-name` | generation run to read |
| `--generator-model` | generation model to read |
| `--processed-dataset` | dataset source used if the store must be rebuilt |
| `--augmented-dataset` | exact store path |
| `--dataset-types` | subset of datasets |
| `--settings` | subset of settings |
| `--modes` | `full_question` and/or `choices_only` |
| `--limit` | per-dataset row limit |
| `--rebuild-cache` | rebuild the store before evaluating |

### `evaluate-all`

Run the default evaluation model set directly:

```bash
uv run python main.py evaluate-all \
  --run-name eval_all_defaults \
  --generator-run-name gen_gpt52 \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3
```

### `submit-generate-cluster`

Create dependency-aware SLURM generation slices.

Each slice is:

- model
- dataset
- generation strategy
- question chunk

Example:

```bash
uv run python main.py submit-generate-cluster \
  --run-name gen_cluster \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --models Qwen/Qwen3-4B-Instruct-2507,allenai/Olmo-3-7B-Instruct,gpt-5.2-2025-12-11 \
  --generation-strategies model_from_scratch,augment_human,augment_model,augment_ablation \
  --questions-per-job 200 \
  --gpu-count 4 \
  --render-status
```

Preview only:

```bash
uv run python main.py submit-generate-cluster \
  --run-name gen_preview \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --models Qwen/Qwen3-4B-Instruct-2507 \
  --generation-strategies model_from_scratch,augment_model \
  --questions-per-job 100 \
  --write-only \
  --render-status
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--models` | comma-separated generation model list |
| `--processed-dataset` | dataset source used to build slices |
| `--dataset-types` | subset of datasets |
| `--generation-strategies` | subset of schedulable strategies |
| `--questions-per-job` | chunk size |
| `--gpu-count` | concurrency cap per resource class |
| `--write-only` | write bundle but do not submit |
| `--dry-run` | print plan only |
| `--force` | ignore current/pending slice state |
| `--render-status` | write HTML dashboard |

### `submit-evaluate-cluster`

Create dependency-aware SLURM evaluation slices.

Each slice is:

- model
- dataset
- setting
- mode
- question chunk

Example:

```bash
uv run python main.py submit-evaluate-cluster \
  --run-name eval_cluster \
  --generator-run-name gen_gpt52 \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --models Qwen/Qwen3-4B-Instruct-2507,allenai/Olmo-3-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct \
  --settings human_from_scratch,model_from_scratch,augment_human,augment_model,augment_ablation \
  --modes full_question,choices_only \
  --questions-per-job 200 \
  --gpu-count 3 \
  --render-status
```

Rerun just one family:

```bash
uv run python main.py submit-evaluate-cluster \
  --run-name eval_cluster \
  --generator-run-name gen_gpt52 \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --dataset-types gpqa \
  --settings augment_model \
  --modes full_question \
  --models Qwen/Qwen3-4B-Instruct-2507 \
  --questions-per-job 200 \
  --force \
  --render-status
```

### `export`

Export benchmarker JSONL files from the setting-scoped store:

```bash
uv run python main.py export \
  --generator-run-name gen_gpt52 \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3
```

Export from an explicit store path:

```bash
uv run python main.py export \
  --input datasets/augmented/gen_gpt52/openai_gpt-5.2-2025-12-11 \
  --output-root datasets/benchmarker_items
```

### `analyze`

Build plots and summary tables:

```bash
uv run python main.py analyze \
  --results-root results/inspect/evaluation \
  --output-dir results/final5_plots
```

### `diagnose-trace`

Dump generation traces for debugging:

```bash
uv run python main.py diagnose-trace \
  --log-dir results/inspect/generation/gen_gpt52/openai_gpt-5.2-2025-12-11 \
  --sample-id gpqa:gpqa-1
```

### `diagnose-failures`

Inspect a setting-scoped store for missing outputs:

```bash
uv run python main.py diagnose-failures \
  --dataset-path datasets/augmented/gen_gpt52/openai_gpt-5.2-2025-12-11
```

## Shared Runtime Flags

These apply to direct generation and evaluation commands.

| Flag | Meaning |
|---|---|
| `--backend` | provider prefix for a model name |
| `--model-base-url` | custom OpenAI-compatible endpoint |
| `--max-connections` | Inspect concurrency |
| `--max-tokens` | max output tokens |
| `--temperature` | sampling temperature |
| `--reasoning-effort` | reasoning hint for supported providers |
| `--retry-on-error` | retry count for model errors |
| `--stop-seqs` | optional stop sequences |

## Shared Shard Flags

These apply to direct generation and evaluation commands.

| Flag | Meaning |
|---|---|
| `--shard-count` | number of deterministic shards |
| `--shard-index` | zero-based shard to run |
| `--shard-strategy` | `contiguous` or `modulo` |

## Custom Dataset Manifest

Example:

```json
{
  "schema_version": "augmented_mcqa_dataset_manifest_v1",
  "datasets": {
    "custom_benchmark": {
      "path": "datasets/custom/questions.jsonl",
      "format": "jsonl",
      "question_key": "question",
      "answer_key": "answer",
      "choices_human_key": "choices_human",
      "category_key": "category",
      "question_id_key": "question_id"
    }
  }
}
```

## Custom Model Registry Entry

Example:

```json
{
  "name": "my-local-model",
  "resolved": "vllm/my-org/my-local-model"
}
```

## Custom Recipe Changes

Edit `config/generation_recipes.json` when you want to change:

- prompt template file
- prompt mode
- prerequisite setting
- distractor counts
- total choice counts

Then rerun the affected setting and rebuild the store:

```bash
uv run python main.py generate \
  --model gpt-5.2-2025-12-11 \
  --run-name gen_gpt52 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --generation-strategies augment_model \
  --rebuild-cache \
  --materialize-cache
```
