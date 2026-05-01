# Augmented MCQA

Minimal pipeline for:

- preparing processed datasets
- generating augmented multiple-choice datasets
- evaluating models against them
- exporting benchmarker items
- analyzing collected evaluation datasets
- fitting many-facet IRT models over collected evaluations

Run everything with `uv`:

```bash
uv run python main.py --help
```

## Workflow

```text
prepare-data -> generate -> evaluate -> analyze
                           \-> analyze-irt
                           \-> export
```

The important artifacts are:

- processed dataset: `datasets/processed/unified_processed_v3`
- augmented store: `datasets/augmented/<generation_run>/<generation_model>/`
- collected dataset: `datasets/collected/<generation_run>/<generation_model>/<evaluation_model>/`
- IRT outputs: `results/augmented_mcqa_irt/`

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

Fit many-facet IRT models across collected evaluation folders:

```bash
uv run python main.py analyze-irt \
  --collected-root datasets/collected \
  --output-dir results/augmented_mcqa_irt
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

For IRT analysis, share the common collected root that contains every generator/evaluator folder you want in the same model. The IRT command discovers `evaluated_manifest.json` files recursively under that root and fits from the materialized collected datasets, not from `.eval` logs.

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

### `build-augmented-dataset`

Rebuild one augmented dataset store from generation logs for a specific run and model.

Note:
- this command now also writes generation-support metadata under `datasets/support_sets/...`
- cluster and local evaluation use that metadata automatically to define the shared benchmark subset before evaluation runs

Use this when:
- generation logs already exist but the augmented store is missing or stale
- you want to recover a single generation run without rerunning generation
- you are debugging whether generation logs were written correctly

Example:

```bash
uv run python main.py build-augmented-dataset \
  --run-name gen_gpt52_v2 \
  --model gpt-5.2-2025-12-11
```

### `build-collected-dataset`

Materialize collected evaluation datasets from evaluation logs. In normal scheduled runs this now happens once at the end of the run; this command remains useful for manual recovery or ad hoc refreshes.

Note:
- this command trusts the precomputed generation-support subset
- if an evaluation output is missing or unusable, collection keeps the row and fills a deterministic random fallback prediction while preserving the underlying eval status/raw output

Use this when:
- evaluation logs already exist but `datasets/collected/...` needs to be rebuilt
- a scheduled collect job failed and you want to rerun just collection
- you want to refresh analysis inputs without rerunning evaluation

Example:

```bash
uv run python main.py build-collected-dataset \
  --run-name eval_qwen4b_on_gen_gpt52_v2 \
  --generator-run-name gen_gpt52_v2 \
  --generator-model gpt-5.2-2025-12-11 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --evaluation-log-root results/inspect/evaluation/eval_qwen4b_on_gen_gpt52_v2/gen_gpt52_v2/openai_gpt-5.2-2025-12-11/vllm_Qwen_Qwen3-4B-Instruct-2507 \
  --augmented-dataset datasets/augmented/gen_gpt52_v2/openai_gpt-5.2-2025-12-11
```

### Support Sets

Support sets live under `datasets/support_sets/<generation-run>/<generation-model>/support_manifest.json`.

They store only the `sample_id`s that are eligible for downstream evaluation, plus counts by dataset. They do not store question text, answer text, choices, randomized options, generations, or evaluator outputs. If canonical question text changes, repair the augmented and collected datasets; the support manifest only needs to change when eligibility changes.

Use support sets when:
- comparing generator models on the same question IDs
- rebuilding collected datasets from existing evaluation logs
- launching evaluation jobs after generation has been materialized

The normal flow is:

```bash
uv run python main.py build-augmented-dataset \
  --run-name gen_gpt52_v2 \
  --model gpt-5.2-2025-12-11
```

That command writes the support manifest automatically. Evaluation and collection commands use `--support-root datasets/support_sets` by default, so you usually do not need to pass it explicitly. Pass `--support-root` only when reading or writing support manifests in a non-default location.

Before rerunning evaluation or collection for a generation run, make sure every generator model in that run has a support manifest. The evaluation pipeline intersects the manifests for the run so all generator models are evaluated on the same supported sample IDs.

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

### `analyze-irt`

Fit fixed-guessing many-facet IRT models from collected evaluation datasets.

Use this when:
- you have collected results for multiple generator or evaluator models
- you want estimates for item difficulty, setting difficulty, generator ability, and evaluator severity
- you want model-based diagnostics such as item anomaly plots and residual summaries

By default this command analyzes `full_question` rows. Pass `--modes full_question,choices_only` when you want both modes included. Filter with comma-separated `--generators`, `--evaluators`, `--datasets`, or `--settings` values when you want a smaller identified design. Each included facet must have enough overlap for the model to be identified.

Example:

```bash
uv run python main.py analyze-irt \
  --collected-root datasets/collected \
  --output-dir results/augmented_mcqa_irt \
  --modes full_question
```

Typical outputs include:
- `tables/setting_difficulty.csv`
- `tables/evaluator_severity.csv`
- `tables/generator_ability.csv`
- `tables/item_difficulty.csv`
- `tables/residual_summary.csv`
- `figures/setting_difficulty_forest.png`
- `figures/evaluator_severity_forest.png`
- `figures/item_anomalies.png`
- `fit_summary.json`

### Benchmarker Writing-Flaw Analysis

`analysis/benchmarker_analysis.py` joins benchmarker writing-flaw outputs, evaluation logs, and augmented rows. It writes figures to `results/benchmarker_analysis` by default. It reads Inspect `.eval` archives through the tolerant log-payload helpers, so corrupt or partial archives are skipped instead of stopping the run. Use `--generator-run-name`, `--generator-model`, and `--eval-models` to restrict the log scan when `results/inspect/evaluation` contains multiple sweeps.

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
