# Pipeline Guide

This document explains the end-to-end Final5 pipeline from dataset download to analysis.

## Pipeline Flow

```
Download → Process → Augment → Evaluate → Merge → Analyze
  (01)      (01)      (02/03)    (04/05)   (06)    (08)
```

## Scripts (in pipeline order)

### Step 1: Data Preparation — `scripts/01_data_pipeline.py`

Downloads raw datasets from HuggingFace and processes them into the unified Final5 schema.

```bash
# Download all raw datasets
uv run python scripts/01_data_pipeline.py download --all

# Process into unified schema
uv run python scripts/01_data_pipeline.py process --output-path datasets/processed/unified_processed_v2

# Or do both at once
uv run python scripts/01_data_pipeline.py all
```

Active datasets: `arc_challenge`, `mmlu_pro`, `gpqa`

### Step 2: Distractor Generation — `scripts/02_generate_distractors.py`

Generates Final5 distractor columns for a processed dataset using LLM APIs. Supports parallel per-split generation.

```bash
uv run python scripts/02_generate_distractors.py \
  --input datasets/processed/unified_processed_v2 \
  --output datasets/augmented/final5_full_<timestamp>_<model> \
  --model gpt-5.2-2025-12-11
```

### Step 3: Orchestrated Regeneration — `scripts/03_regenerate_experiments.py`

Runs step 2 across all three generator models concurrently and writes a regeneration manifest.

```bash
uv run python scripts/03_regenerate_experiments.py \
  --processed-dataset datasets/processed/unified_processed_v2 \
  --output-root datasets/augmented
```

### Step 4: Evaluation — `scripts/04_eval_matrix.py`

Plans and runs evaluation configs. Has two subcommands: `plan` (builds manifests) and `run` (executes).

```bash
# Plan configs
uv run python scripts/04_eval_matrix.py plan \
  --preset final5 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-path datasets/augmented/<generator_dataset> \
  --generator-dataset-label <generator>

# Run configs
uv run python scripts/04_eval_matrix.py run \
  --manifest <manifest.json> \
  --generator-dataset-label <generator>
```

### Step 5: SLURM Bundle — `scripts/05_build_eval_slurm_bundle.py`

Builds balanced SLURM job bundles for distributed evaluation on GPU clusters.

```bash
uv run python scripts/05_build_eval_slurm_bundle.py \
  --manifest datasets/augmented/<manifest>.json \
  --target-rows-per-subsplit 500
```

### Step 6: Merge Sub-shards — `scripts/06_merge_eval_subshards.py`

Merges partial evaluation results from entry sub-shards into canonical Arrow + summary outputs.

```bash
uv run python scripts/06_merge_eval_subshards.py \
  --bundle-manifest jobs/generated/<timestamp>/bundle_manifest.json \
  --strict
```

### Step 7: Export — `scripts/07_export_benchmarker_items.py`

Exports augmented datasets into benchmarker JSONL files for external tools.

```bash
uv run python scripts/07_export_benchmarker_items.py \
  --input datasets/augmented/<dataset> \
  --output-root datasets/benchmarker_items
```

### Step 8: Analysis — `scripts/08_analyze.py`

Analyzes results and generates plots. Has two subcommands: `table` and `plot`.

```bash
# Generate behavioral signature tables
uv run python scripts/08_analyze.py table --dir results/

# Generate Final5 pairwise plots
uv run python scripts/08_analyze.py plot --results-root results --output-dir results/final5_plots
```

## Utility Scripts

### `scripts/diagnose.py`

Debugging tools for generation issues:

```bash
# Check for missing distractor columns
uv run python scripts/diagnose.py failures --dataset-path datasets/augmented/...

# Run structured-generation trace
uv run python scripts/diagnose.py trace --model gemini-3.1-pro-preview
```

### `scripts/smoke_test.py`

Live API smoke tests:

```bash
# Test structured outputs across all providers
uv run python scripts/smoke_test.py clients

# End-to-end generation smoke with real datasets
uv run python scripts/smoke_test.py generate --limit 2 --dry-run
```

## Quick Start (Minimal Pipeline)

For a new user, the core happy path is:

```bash
# 1. Download and process datasets
uv run python scripts/01_data_pipeline.py all

# 2. Generate distractors (single model)
uv run python scripts/02_generate_distractors.py \
  --input datasets/processed/unified_processed_v2 \
  --output datasets/augmented/my_run \
  --model gpt-5.2-2025-12-11

# 3. Run evaluation
uv run python scripts/04_eval_matrix.py run \
  --preset final5 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-path datasets/augmented/my_run \
  --generator-dataset-label gpt-5.2-2025-12-11

# 4. Analyze results
uv run python scripts/08_analyze.py plot --results-root results
```

## Project Structure

```
config/        Central configuration (paths, API keys, constants, prompts)
data/          Dataset downloading, processing, augmentation
models/        LLM client abstraction (OpenAI, Anthropic, Gemini, local/vLLM)
evaluation/    MCQA evaluation logic (prompt building, answer extraction, scoring)
experiments/   Experiment configuration, matrix operations, runner
analysis/      Post-hoc analysis and visualization
scripts/       CLI entry points (numbered by pipeline order)
tests/         Unit tests
docs/          Documentation
jobs/          SLURM job scripts and templates
```
