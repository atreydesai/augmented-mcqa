# CLI Reference

This page is the exhaustive flag reference.
For the exact copy-paste commands for the current workflow, start with the top-level README.

Reading guide:

- plain description: normal workflow flag
- `Situational:` useful in specific setups, but not part of the default path
- `Advanced tuning:` performance or model-call tuning that most runs should leave alone
- `Advanced override:` path or behavior override that most runs should leave alone
- `Advanced/debug option:` mainly for debugging or tiny smoke runs
- `Manual fallback control:` low-level control that is usually superseded by the scheduler path

## Shared Direct-Run Flags

These flags apply to `generate`, `generate-all`, `evaluate`, and `evaluate-all`.

| Flag | Meaning |
|---|---|
| `--backend` | Situational: provider prefix to apply to an unqualified model name, such as `openai`, `google`, `together`, or `vllm`. |
| `--model-base-url` | Situational: base URL for an OpenAI-compatible endpoint or other custom provider endpoint. |
| `--max-connections` | Advanced tuning: maximum concurrent model connections Inspect may open for the run. |
| `--max-tokens` | Advanced tuning: maximum output tokens requested per model call. |
| `--temperature` | Advanced tuning: sampling temperature forwarded to the backend. |
| `--reasoning-effort` | Advanced tuning: reasoning-effort hint for providers that support it. |
| `--retry-on-error` | Advanced tuning: how many times Inspect should retry a failed model call. |
| `--stop-seqs` | Advanced tuning: optional stop sequences forwarded to the backend. |

## Shared Shard Flags

These flags apply to `generate`, `generate-all`, `evaluate`, and `evaluate-all`.

Most users should not need them. They are manual fallback controls for cases where you want to split a direct run yourself instead of using the scheduler path (`submit-generate-cluster` / `submit-evaluate-cluster`).

| Flag | Meaning |
|---|---|
| `--shard-count` | Manual fallback control: number of deterministic shards to split the selected samples into. |
| `--shard-index` | Manual fallback control: zero-based shard index to run. |
| `--shard-strategy` | Manual fallback control: shard partitioning strategy, `contiguous` or `modulo`. |

## Shared Cluster-Submit Flags

These flags apply to `submit-generate-cluster` and `submit-evaluate-cluster`.

| Flag | Meaning |
|---|---|
| `--models` | Comma-separated list of models to schedule. Models can be local `vllm/...` or hosted/API models. |
| `--processed-dataset` | Processed unified `DatasetDict` used to build scheduler slices. |
| `--dataset-types` | Comma-separated subset of dataset splits to schedule. |
| `--limit` | Advanced/debug option: optional per-dataset cap on the number of samples to schedule before chunking. |
| `--questions-per-job` | Optional contiguous question-chunk size per scheduled slice. If omitted, each slice family stays in one chunk. |
| `--gpu-count` | Optional concurrency cap applied per resource class by the master submit script. |
| `--output-dir` | Advanced override: run directory where manifests, wrappers, state, dashboard, and submit helpers are written. |
| `--submit` | Advanced control: submit the generated per-slice bundle after writing it. |
| `--write-only` | Advanced control: write the bundle but do not run the master submit script. |
| `--dry-run` | Advanced control: print the planned bundle details without writing or submitting anything. |
| `--force` | Resubmit slices even if they are already current or pending. |
| `--render-status` | Write the HTML dashboard for the run. `scheduler_state.json` is refreshed whenever a bundle is written. |
| `--partition` | Advanced cluster override: SLURM partition to request for each task. |
| `--account` | Advanced cluster override: SLURM account to charge for each task. |
| `--qos` | Advanced cluster override: SLURM QoS value for each task. |
| `--time-limit` | Advanced cluster override: wall-clock time limit for each task. |
| `--mem` | Advanced cluster override: memory request per task. |
| `--cpus-per-task` | Advanced cluster override: CPU core request per task. |
| `--gpu-type` | Advanced cluster override: GPU type to request for local GPU-backed slices, such as `rtxa6000`. |

## Step 1: Prepare Data

Recommended command:

```bash
uv run python main.py prepare-data \
  --step all \
  --output-path datasets/processed/unified_processed_v3
```

| Flag | Meaning |
|---|---|
| `--step` | Which phase to run: `download`, `process`, or `all`. |
| `--dataset` | Specific raw dataset to download when not using `--all`. |
| `--all` | Download every supported raw dataset instead of one dataset. |
| `--output-dir` | Advanced override: directory where raw downloaded datasets should be stored. |
| `--output-path` | Directory where the processed unified `DatasetDict` should be written. |
| `--limit` | Advanced/debug option: optional per-dataset cap when building the processed dataset. |

## Step 2: Generate Distractors

The README contains the full copy-paste commands for all recommended generation models.

Recommended API generator command:

```bash
uv run python main.py generate \
  --model gpt-5.2-2025-12-11 \
  --run-name gen_gpt52 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --materialize-cache
```

Recommended scheduler generator command:

```bash
uv run python main.py submit-generate-cluster \
  --run-name gen_scheduler_all \
  --models Qwen/Qwen3-4B-Instruct-2507,allenai/Olmo-3-7B-Instruct,gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --generation-strategies model_from_scratch,augment_human,augment_model,augment_ablation \
  --questions-per-job 200 \
  --gpu-count 4 \
  --render-status
```

Default model sets:

- `generate-all` defaults to the five API generation models:
  - `gpt-5.2-2025-12-11`
  - `claude-opus-4-6`
  - `gemini-3.1-pro-preview`
  - `Qwen/Qwen3.5-397B-A17B`
  - `Qwen/Qwen3.5-9B`
- `submit-generate-cluster` defaults to the two local generation models:
  - `Qwen/Qwen3-4B-Instruct-2507`
  - `allenai/Olmo-3-7B-Instruct`

### `main.py generate`

| Flag | Meaning |
|---|---|
| `--model` | Generation model name, alias, or provider-qualified Inspect model id. |
| `--run-name` | Logical run name used to organize generation logs and caches. |
| `--processed-dataset` | Processed unified `DatasetDict` to read questions from. |
| `--dataset-types` | Optional subset: comma-separated subset of dataset splits to generate for. |
| `--limit` | Advanced/debug option: optional per-dataset cap on the number of samples to generate. |
| `--log-root` | Advanced override: root directory for Inspect generation logs. |
| `--cache-root` | Advanced override: root directory where derived augmented caches are stored. |
| `--augmented-dataset` | Advanced override: exact output path for the augmented cache. |
| `--materialize-cache` | Rebuild the augmented `DatasetDict` cache immediately after generation completes. |
| `--rebuild-cache` | Advanced override: force regeneration of the augmented cache even if it already exists. |

Also supports all flags from `Shared Direct-Run Flags` and `Shared Shard Flags`. In normal use, the shard flags should be left alone.

### `main.py generate-all`

Use this when you want to run the full default API generator set instead of one model.

| Flag | Meaning |
|---|---|
| `--models` | Comma-separated list of generation models to override the default API generation set. |
| `--run-name` | Logical run name used to organize generation logs and caches. |
| `--processed-dataset` | Processed unified `DatasetDict` to read questions from. |
| `--dataset-types` | Optional subset: comma-separated subset of dataset splits to generate for. |
| `--limit` | Advanced/debug option: optional per-dataset cap on the number of samples for each model. |
| `--log-root` | Advanced override: root directory for Inspect generation logs. |
| `--cache-root` | Advanced override: root directory where derived augmented caches are stored. |
| `--materialize-cache` | Rebuild each augmented cache immediately after generation completes. |
| `--rebuild-cache` | Advanced override: force regeneration of each augmented cache even if it already exists. |

Also supports all flags from `Shared Direct-Run Flags` and `Shared Shard Flags`. In normal use, the shard flags should be left alone.

### `main.py submit-generate-cluster`

Use this when generation should be scheduled as dependency-aware SLURM slices.

Each slice is one:

- model
- dataset
- generation strategy
- question chunk

Supported schedulable generation strategies:

- `model_from_scratch`
- `augment_human`
- `augment_model`
- `augment_ablation`

`augment_model` depends on the matching `model_from_scratch` slice for the same model, dataset, and question chunk. `human_from_scratch` remains implicit and is not scheduled as its own slice.

`--write-only` bundles remain `planned` in `scheduler_state.json` until `submit_all.sh` is actually run.

| Flag | Meaning |
|---|---|
| `--run-name` | Logical run name used to organize the generated bundle, logs, and output caches. |
| `--generation-strategies` | Comma-separated subset of schedulable generation strategies to submit. |

Also supports all flags from `Shared Cluster-Submit Flags`.

## Step 3: Evaluate

The scheduler can now fan out both local and hosted/API evaluation jobs.

Recommended cluster command:

```bash
uv run python main.py submit-evaluate-cluster \
  --run-name eval_scheduler_all \
  --generator-run-name gen_scheduler_all \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --models Qwen/Qwen3-4B-Instruct-2507,allenai/Olmo-3-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct \
  --settings human_from_scratch,model_from_scratch,augment_human,augment_model,augment_ablation \
  --modes full_question,choices_only \
  --questions-per-job 200 \
  --gpu-count 3 \
  --render-status
```

The README contains the exact evaluation commands for each recommended generation run.

Default model sets:

- `evaluate-all` defaults to the three local evaluation models:
  - `Qwen/Qwen3-4B-Instruct-2507`
  - `allenai/Olmo-3-7B-Instruct`
  - `meta-llama/Llama-3.1-8B-Instruct`
- `submit-evaluate-cluster` defaults to the same three local evaluation models

### `main.py evaluate`

| Flag | Meaning |
|---|---|
| `--model` | Evaluation model name, alias, or provider-qualified Inspect model id. |
| `--run-name` | Logical run name used to organize evaluation logs. |
| `--generator-run-name` | Generation run whose outputs should be evaluated. |
| `--generator-model` | Generation model whose outputs should be evaluated. |
| `--generator-backend` | Situational: backend prefix to apply when resolving `--generator-model`. |
| `--generation-log-dir` | Advanced override: exact generation log directory to read instead of deriving one automatically. |
| `--generation-log-root` | Advanced override: root directory for generation logs when deriving inputs automatically. |
| `--processed-dataset` | Processed unified `DatasetDict` used if the augmented cache must be rebuilt from generation logs. |
| `--augmented-dataset` | Advanced override: exact augmented cache path to evaluate instead of deriving one automatically. |
| `--cache-root` | Advanced override: root directory where augmented caches are stored. |
| `--dataset-types` | Optional subset: comma-separated subset of dataset splits to evaluate. |
| `--settings` | Advanced subset override: comma-separated subset of Final5 settings to evaluate. |
| `--modes` | Advanced subset override: comma-separated subset of evaluation modes to run. |
| `--limit` | Advanced/debug option: optional per-dataset cap on the number of evaluation samples. |
| `--log-root` | Advanced override: root directory for Inspect evaluation logs. |
| `--rebuild-cache` | Advanced override: force regeneration of the augmented cache before evaluation. |

Also supports all flags from `Shared Direct-Run Flags` and `Shared Shard Flags`. In normal use, the shard flags should be left alone.

### `main.py evaluate-all`

Use this when you want to run the full default local evaluation model set directly instead of one model.

| Flag | Meaning |
|---|---|
| `--models` | Comma-separated list of evaluation models to override the default local evaluation set. |
| `--run-name` | Logical run name used to organize evaluation logs. |
| `--generator-run-name` | Generation run whose outputs should be evaluated. |
| `--generator-model` | Generation model whose outputs should be evaluated. |
| `--generator-backend` | Situational: backend prefix to apply when resolving `--generator-model`. |
| `--generation-log-dir` | Advanced override: exact generation log directory to read instead of deriving one automatically. |
| `--generation-log-root` | Advanced override: root directory for generation logs when deriving inputs automatically. |
| `--processed-dataset` | Processed unified `DatasetDict` used if the augmented cache must be rebuilt from generation logs. |
| `--augmented-dataset` | Advanced override: exact augmented cache path to evaluate instead of deriving one automatically. |
| `--cache-root` | Advanced override: root directory where augmented caches are stored. |
| `--dataset-types` | Optional subset: comma-separated subset of dataset splits to evaluate. |
| `--settings` | Advanced subset override: comma-separated subset of Final5 settings to evaluate. |
| `--modes` | Advanced subset override: comma-separated subset of evaluation modes to run. |
| `--limit` | Advanced/debug option: optional per-dataset cap on the number of evaluation samples for each model. |
| `--log-root` | Advanced override: root directory for Inspect evaluation logs. |
| `--rebuild-cache` | Advanced override: force regeneration of the augmented cache before evaluation. |

Also supports all flags from `Shared Direct-Run Flags` and `Shared Shard Flags`. In normal use, the shard flags should be left alone.

### `main.py submit-evaluate-cluster`

Use this when evaluation should be scheduled as dependency-aware SLURM slices.

Each slice is one:

- model
- dataset
- Final5 setting
- evaluation mode
- question chunk

Each evaluation slice depends on the exact generation slice or slices required by its setting for the same dataset chunk.

`--write-only` bundles remain `planned` in `scheduler_state.json` until `submit_all.sh` is actually run.

| Flag | Meaning |
|---|---|
| `--run-name` | Logical run name used to organize the generated bundle and evaluation logs. |
| `--generator-run-name` | Generation run whose outputs the cluster jobs should evaluate. |
| `--generator-model` | Generation model whose outputs the cluster jobs should evaluate. |
| `--settings` | Comma-separated subset of Final5 settings to schedule. |
| `--modes` | Comma-separated subset of evaluation modes to schedule. |

Also supports all flags from `Shared Cluster-Submit Flags`.

## Step 4: Analyze

Recommended command:

```bash
uv run python main.py analyze \
  --results-root results/inspect/evaluation \
  --output-dir results/final5_plots
```

| Flag | Meaning |
|---|---|
| `--results-root` | Usually leave alone: directory containing Inspect evaluation logs to analyze. |
| `--output-dir` | Situational output override: directory where plots and optional tables should be written. |
| `--table-output` | Advanced output override: CSV path for the flat summary table. |
| `--skip-tables` | Advanced output option: write plots only and skip the pairwise comparison CSV tables. |

## Step 5: Export Benchmarker Files

Recommended command:

```bash
uv run python main.py export \
  --generator-run-name gen_gpt52 \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3
```

| Flag | Meaning |
|---|---|
| `--input` | Advanced override: exact augmented cache path to export. If omitted, generation artifacts are resolved automatically. |
| `--output-root` | Situational output override: root directory where benchmarker JSONL files should be written. |
| `--generator-run-name` | Generation run name used when deriving the augmented cache automatically. |
| `--generator-model` | Generation model used when deriving the augmented cache automatically. |
| `--generator-backend` | Situational: backend prefix to apply when resolving `--generator-model`. |
| `--generation-log-dir` | Advanced override: exact generation log directory to read instead of deriving one automatically. |
| `--generation-log-root` | Advanced override: root directory for generation Inspect logs when deriving inputs automatically. |
| `--processed-dataset` | Processed unified `DatasetDict` used if the augmented cache must be rebuilt from generation logs. |
| `--augmented-dataset` | Advanced override: exact augmented cache path to use instead of deriving one automatically. |
| `--cache-root` | Advanced override: root directory where augmented caches are stored. |
| `--dataset-types` | Advanced subset override: comma-separated subset of dataset splits to include when rebuilding the augmented cache. |
| `--rebuild-cache` | Advanced override: force regeneration of the augmented cache before export. |

## Step 6: Optional Benchmarker Writing-Flaw Analysis

Recommended command:

```bash
uv run python analysis/benchmarker_analysis.py \
  --writing-flaw-jsonl datasets/benchmarker_results/atrey_writing_flaw_rows.jsonl.zip \
  --results-root results/inspect/evaluation \
  --cache-root datasets/augmented \
  --generator-run-name gen_gpt52 \
  --generator-model gpt-5.2-2025-12-11 \
  --output-dir analysis/figures/benchmarker
```

| Flag | Meaning |
|---|---|
| `--writing-flaw-jsonl` | Writing-flaw rows JSONL or zipped JSONL file to analyze. |
| `--results-root` | Evaluation log directory to read Inspect `.eval` files from. |
| `--augmented-dataset` | Advanced override: exact augmented cache path to use. |
| `--cache-root` | Advanced override: root directory to search for augmented caches if `--augmented-dataset` is not provided. |
| `--output-dir` | Situational output override: directory where the benchmarker figures should be written. |
| `--generator-model` | Generation model filter used to select matching evaluation logs and caches. |
| `--generator-run-name` | Generation run-name filter used to select matching evaluation logs and caches. |
| `--eval-models` | Advanced subset override: comma-separated list of evaluation models to include. |
