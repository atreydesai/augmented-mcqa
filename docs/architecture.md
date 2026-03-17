# Architecture

This repo has one CLI entrypoint:

- `main.py`

## Runtime Shape

```text
prepare-data -> generate -> setting-scoped augmented store -> evaluate -> analyze/export
```

Rules:

- Inspect `.eval` logs are the source of truth.
- The augmented store is derived from generation logs.
- Evaluation and export read the setting-scoped store.
- Cluster jobs are just dependency-aware wrappers around `python main.py ...`.

## Main Pieces

- `main.py`
  CLI orchestration, Inspect launch, cache materialization, and cluster submission.
- `data/`
  Processed dataset loading, setting-scoped store materialization, export, and migration.
- `tasks/`
  Inspect task builders for generation and evaluation.
- `solvers/`
  Prompt rendering, model calls, and parser wiring.
- `scorers/`
  Inspect score metadata for generation and evaluation.
- `analysis/`
  Plotting and benchmarker analysis.
- `utils/`
  Model resolution, recipe loading, sharding, log helpers, and scheduler logic.

## Data Model

### Processed dataset input

`--processed-dataset` can be:

- the unified processed `DatasetDict`
- a dataset manifest JSON

The standard processed dataset has splits:

- `arc_challenge`
- `mmlu_pro`
- `gpqa`

### Model registry

`config/model_registry.json` is the source of truth for model names.

Examples:

- `gpt-5.2-2025-12-11` -> `openai/gpt-5.2-2025-12-11`
- `Qwen/Qwen3-4B-Instruct-2507` -> `vllm/Qwen/Qwen3-4B-Instruct-2507`

Unregistered models are rejected.

### Recipe config

`config/generation_recipes.json` defines:

- setting names
- schedulable strategies
- prompt template files
- prompt mode
- prerequisite settings
- distractor counts
- final choice counts

Public settings stay:

- `human_from_scratch`
- `model_from_scratch`
- `augment_human`
- `augment_model`
- `augment_ablation`

Schedulable generation strategies are:

- `model_from_scratch`
- `augment_human`
- `augment_model`
- `augment_ablation`

`augment_model` depends on `model_from_scratch`.

## Store Layout

The augmented store lives at:

```text
datasets/augmented/<run>/<model>/
```

It contains:

- `augmented_manifest.json`
- one saved dataset per `dataset/setting`

Example:

```text
datasets/augmented/gen_gpt52/openai_gpt-5.2-2025-12-11/
  augmented_manifest.json
  gpqa/
    human_from_scratch/
    model_from_scratch/
    augment_human/
    augment_model/
    augment_ablation/
```

Each stored row contains:

- stable row identity
- original question/answer/category fields
- human and model distractor lists
- randomized options
- correct answer letter
- counts and traces metadata

The repo no longer relies on one wide row with fixed columns for every setting.

## Generation Flow

1. `tasks/generation.py` builds one Inspect task per requested strategy.
2. `solvers/final5_generation.py` loads the recipe for that strategy.
3. The solver renders the configured prompt and parses distractors.
4. Generation logs are written to `results/inspect/generation/<run>/<model>/`.
5. `data/final5_store.py` materializes the setting-scoped store from those logs.

`human_from_scratch` is still implicit for scheduling, but it is stored as a first-class setting in the augmented store.

## Evaluation Flow

1. `tasks/evaluation.py` builds one Inspect task per `setting x mode`.
2. `data/final5_store.py` loads the saved dataset for that setting.
3. `solvers/final5_evaluation.py` prompts the evaluator in:
   - `full_question`
   - `choices_only`
4. `scorers/evaluation.py` records correctness and option-source metadata.

Evaluation no longer reads wide `*_options_randomized` columns from one merged dataset root.

## Export Flow

`data/benchmarker_export.py` exports:

- `original`
- `human_from_scratch`
- `model_from_scratch`
- `augment_human`
- `augment_model`
- `augment_ablation`

It reads each setting dataset directly from the store.

## Cluster Flow

Supported cluster commands:

- `main.py submit-generate-cluster`
- `main.py submit-evaluate-cluster`

Generation slice shape:

- model
- dataset
- strategy
- question chunk

Evaluation slice shape:

- model
- dataset
- setting
- mode
- question chunk

Scheduler logic:

- exact prerequisite wiring between matching slices
- `augment_model` waits on `model_from_scratch`
- stale/current/failed tracking in `scheduler_state.json`
- optional HTML dashboard

Local `vllm/...` slices request GPUs. Hosted/API slices do not.

## Artifact Layout

- processed dataset: `datasets/processed/unified_processed_v3`
- generation logs: `results/inspect/generation/<run>/<model>/`
- evaluation logs: `results/inspect/evaluation/<run>/<generator_run>/<generator_model>/<eval_model>/`
- augmented store root: `datasets/augmented/<run>/<model>/`
- per-setting datasets: `datasets/augmented/<run>/<model>/<dataset>/<setting>/`
- benchmarker export root: `datasets/benchmarker_items/<store-name>/`
- cluster bundles: `jobs/generated/<stage>/<run>/`
- scheduler state: `jobs/generated/<stage>/<run>/scheduler_state.json`
- scheduler dashboard: `jobs/generated/<stage>/<run>/scheduler_status.html`
- SLURM logs: `logs/slurm/<stage>/<run>/`

## Tests

Run:

```bash
uv run pytest -q
```

Pytest redirects artifacts into `test-artifacts/pytest/` so normal runtime directories stay clean.
