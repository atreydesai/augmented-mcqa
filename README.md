# Augmented MCQA

Minimal pipeline for:

- preparing processed datasets
- generating augmented multiple-choice datasets
- evaluating models against them
- exporting benchmarker items
- analyzing collected evaluation datasets

Run everything with `uv`:

```bash
uv run python main.py --help
```

## Workflow

```text
prepare-data -> generate -> evaluate -> analyze
                           \-> export
```

The important artifacts are:

- processed dataset: `datasets/processed/unified_processed_v3`
- augmented store: `datasets/augmented/<generation_run>/<generation_model>/`
- collected dataset: `datasets/collected/<generation_run>/<generation_model>/<evaluation_model>/`

## Quick Start

Build processed data:

```bash
bash scripts/prepare-data.sh
```

Generate a full augmented store with the main generator model:

```bash
uv run python main.py generate \
  --run-name gen_gpt52_v2 \
  --model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --dataset-types arc_challenge,mmlu_pro,gpqa \
  --generation-strategies model_from_scratch,augment_human,augment_model,augment_ablation \
  --materialize-cache
```

Submit the same generation run as a cluster bundle:

```bash
uv run python main.py submit-generate-cluster \
  --run-name gen_gpt52_v2 \
  --models gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --generation-strategies model_from_scratch,augment_human,augment_model,augment_ablation \
  --questions-per-job 200 \
  --write-only
```

Evaluate a local model against that run. This will also write the collected dataset automatically:

```bash
uv run python main.py evaluate \
  --run-name eval_qwen4b_on_gen_gpt52_v2 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --generator-run-name gen_gpt52_v2 \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3
```

Submit the main evaluation sweep as a cluster bundle:

```bash
uv run python main.py submit-evaluate-cluster \
  --run-name eval_qwen4b_on_gen_gpt52_v2 \
  --generator-run-name gen_gpt52_v2 \
  --generator-model gpt-5.2-2025-12-11 \
  --models Qwen/Qwen3-4B-Instruct-2507,allenai/Olmo-3-7B-Instruct \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --settings human_from_scratch,model_from_scratch,augment_human,augment_model,augment_ablation \
  --modes full_question,choices_only \
  --questions-per-job 200 \
  --write-only
```

Analyze from collected data:

```bash
uv run python main.py analyze \
  --collected-root datasets/collected/gen_gpt52_v2/openai_gpt-5.2-2025-12-11/vllm_Qwen_Qwen3-4B-Instruct-2507
```

Export benchmarker items from the augmented store:

```bash
uv run python main.py export \
  --input datasets/augmented/gen_gpt52_v2/openai_gpt-5.2-2025-12-11
```

## Scripts

Use the scripts for the common local workflows.

Prepare data:

```bash
bash scripts/prepare-data.sh
```

Local generation:

```bash
bash scripts/local-generate.sh
```

Local evaluation:

```bash
bash scripts/local-evaluate.sh
```

Local end-to-end smoke:

```bash
bash scripts/local-smoke.sh
```

Minimal API smoke:

```bash
bash scripts/api-smoke.sh
```

Cluster generation bundle:

```bash
bash scripts/submit-generate-cluster.sh
```

Cluster evaluation bundle:

```bash
bash scripts/submit-evaluate-cluster.sh
```

Useful script overrides:

- `DATASET_TYPES`
- `GENERATION_STRATEGIES`
- `SETTINGS`
- `MODES`
- `LIMIT`
- `QUESTIONS_PER_JOB`
- `GPU_COUNT`
- `SUBMIT=1`

## Sharing Results

For analysis, the shareable unit is one collected dataset folder:

```text
datasets/collected/<generation_run>/<generation_model>/<evaluation_model>/
```

That folder includes both:

- the evaluated rows needed for plots and summary tables
- the `collected_state.json` needed to understand missing, planned, failed, or partial slices

You do not need to send separate `.eval` logs just to rerun analysis.

## Validation

Run the test suite:

```bash
uv run pytest -q
```

Check the CLI surface:

```bash
uv run python main.py --help
```

Run the local smoke:

```bash
bash scripts/local-smoke.sh
```

## CLI Reference

These are the top-level `main.py` commands. The examples below are intentionally practical: they describe when you would actually reach for each command.

### `prepare-data`

Build or refresh the processed benchmark dataset used by generation and evaluation.

Use this when:
- you are setting up the repo for the first time
- the raw benchmark inputs changed
- you want to rebuild `datasets/processed/...` before any downstream pipeline step

Example:

```bash
uv run python main.py prepare-data
```

### `materialize-store`

Rebuild one augmented dataset store from generation logs for a specific run and model.

Use this when:
- generation logs already exist but the augmented store is missing or stale
- you want to recover a single generation run without rerunning generation
- you are debugging whether generation logs were written correctly

Example:

```bash
uv run python main.py materialize-store \
  --run-name gen_gpt52_v2 \
  --model gpt-5.2-2025-12-11
```

### `collect-evaluated`

Materialize collected evaluation datasets from evaluation logs. In normal scheduled runs this now happens once at the end of the run; this command remains useful for manual recovery or ad hoc refreshes.

Use this when:
- evaluation logs already exist but `datasets/collected/...` needs to be rebuilt
- a scheduled collect job failed and you want to rerun just collection
- you want to refresh analysis inputs without rerunning evaluation

Example:

```bash
uv run python main.py collect-evaluated \
  --run-name eval_qwen4b_on_gen_gpt52_v2 \
  --generator-run-name gen_gpt52_v2 \
  --generator-model gpt-5.2-2025-12-11 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --evaluation-log-root results/inspect/evaluation/eval_qwen4b_on_gen_gpt52_v2/gen_gpt52_v2/openai_gpt-5.2-2025-12-11/vllm_Qwen_Qwen3-4B-Instruct-2507 \
  --augmented-dataset datasets/augmented/gen_gpt52_v2/openai_gpt-5.2-2025-12-11
```

### `generate`

Run generation locally for one model over one or more dataset types and strategies.

Use this when:
- you are iterating locally on prompts or generation settings
- you want a small or medium run without going through cluster submission
- you are smoke-testing a single model before launching a larger sweep

Example:

```bash
uv run python main.py generate \
  --run-name gen_gpt52_v2 \
  --model gpt-5.2-2025-12-11 \
  --dataset-types arc_challenge \
  --generation-strategies model_from_scratch,augment_human \
  --limit 50
```

### `evaluate`

Run evaluation locally for one evaluation model over one generation run.

Use this when:
- you want to inspect failures locally before scheduling a large cluster run
- you are testing one evaluation model against an existing augmented dataset
- you want a small local end-to-end check

Example:

```bash
uv run python main.py evaluate \
  --run-name eval_qwen4b_on_gen_gpt52_v2 \
  --generator-run-name gen_gpt52_v2 \
  --generator-model gpt-5.2-2025-12-11 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-types arc_challenge \
  --settings model_from_scratch,augment_human \
  --modes full_question,choices_only \
  --limit 50
```

### `analyze`

Build plots and summary tables from collected evaluation datasets.

Use this when:
- collection is complete and you want the final figures/tables
- you want to rerun analysis after changing visualization code
- someone shared a collected dataset folder and you want to analyze it directly

Example:

```bash
uv run python main.py analyze \
  --collected-root datasets/collected/gen_gpt52_v2/openai_gpt-5.2-2025-12-11/vllm_Qwen_Qwen3-4B-Instruct-2507
```

### `export`

Export an augmented dataset store into benchmarker-compatible JSONL outputs.

Use this when:
- you need to hand off generated items to another evaluation stack
- you want to inspect or share the augmented store outside this repo

Example:

```bash
uv run python main.py export \
  --input datasets/augmented/gen_gpt52_v2/openai_gpt-5.2-2025-12-11
```

### `submit-generate-cluster`

Build and optionally submit a dependency-aware Slurm bundle for generation.

Use this when:
- local generation is too large or slow
- you need to shard generation across many jobs
- you want a reproducible submission manifest before actually submitting

Example:

```bash
uv run python main.py submit-generate-cluster \
  --run-name gen_gpt52_v2 \
  --models gpt-5.2-2025-12-11,Qwen/Qwen3-4B-Instruct-2507 \
  --generation-strategies model_from_scratch,augment_human,augment_model,augment_ablation \
  --questions-per-job 200 \
  --write-only
```

### `submit-evaluate-cluster`

Build and optionally submit a dependency-aware Slurm bundle for evaluation. This command now schedules one run-level collection job after all eval slices finish, rather than collecting inside each eval job.

Use this when:
- you want to evaluate many model x dataset x setting x mode slices in parallel
- local evaluation would be too slow or too fragile
- you want the cluster bundle to manage dependency wiring and final collection for you

Example:

```bash
uv run python main.py submit-evaluate-cluster \
  --run-name eval_qwen4b_on_gen_gpt52_v2 \
  --generator-run-name gen_gpt52_v2 \
  --generator-model gpt-5.2-2025-12-11 \
  --models Qwen/Qwen3-4B-Instruct-2507,allenai/Olmo-3-7B-Instruct \
  --settings human_from_scratch,model_from_scratch,augment_human,augment_model,augment_ablation \
  --modes full_question,choices_only \
  --questions-per-job 200 \
  --write-only
```
