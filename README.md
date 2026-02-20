# Augmented MCQA

A research toolkit for generating synthetic distractors for multiple-choice questions and evaluating language models across distractor configurations. Datasets are augmented with AI-generated distractors; models are then evaluated across a matrix of distractor counts and sources (human vs. model).

## Architecture

| Module | Description |
|---|---|
| `data/` | Dataset downloading, processing, and augmentation utilities |
| `models/` | Multi-provider model clients (OpenAI, Anthropic, Gemini, DeepSeek, local/vLLM) |
| `experiments/` | Experiment configuration, matrix definitions, and runner |
| `evaluation/` | MCQA prompt building and answer extraction |
| `analysis/` | Result aggregation, category breakdown, and visualization |
| `scripts/` | CLI entry points for each pipeline stage |
| `jobs/` | SLURM job scripts for running on the cluster |
| `config/` | Settings and declarative model aliases (`model_aliases.toml`) |

## Setup

```bash
uv sync
```

Copy `.env.example` to `.env` and fill in API keys for the providers you plan to use:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=...
```

For local model evaluation, install vLLM separately after `uv sync`:

```bash
uv pip install --no-build-isolation 'vllm==0.11.2' 'transformers<5' 'numpy<2.3'
```

Stage model weights before running local eval:

```bash
huggingface-cli download <model_id> --local-dir /path/to/cache
```

## End-to-End Workflow

### 1. Download datasets

```bash
uv run python scripts/download_datasets.py
```

### 2. Process datasets

```bash
uv run python scripts/process_all.py --dataset all
```

### 3. Generate distractors

```bash
uv run python scripts/generate_distractors.py \
  --model gpt-4.1 \
  --input datasets/processed/unified_processed \
  --output datasets/finished_sets/gpt-4.1 \
  --parallel
```

### 4. Plan evaluation matrix (optional preview)

```bash
uv run python scripts/eval_matrix.py plan \
  --preset core16 \
  --model gpt-4.1 \
  --generator-dataset-label gpt-4.1 \
  --dataset-path datasets/augmented/unified_processed_gpt-4.1_20260213_033916 \
  --print-configs
```

### 5. Run evaluation matrix

```bash
uv run python scripts/eval_matrix.py run \
  --preset core16 \
  --model gpt-4.1 \
  --generator-dataset-label gpt-4.1 \
  --dataset-path datasets/augmented/unified_processed_gpt-4.1_20260213_033916 \
  --save-interval 50 \
  --keep-checkpoints 2 \
  --skip-existing
```

### 6. Analyze results

```bash
uv run python scripts/analyze_all.py --dir results
```

## SLURM: API Models

Submit a sharded array job using `jobs/submit_eval_array.sh` + `jobs/eval_matrix_array.sbatch`:

```bash
jobs/submit_eval_array.sh \
  gpt-4.1 \
  datasets/augmented/unified_processed_gpt-4.1_20260213_033916 \
  gpt-4.1 \
  8 \
  --dataset-types mmlu_pro,gpqa \
  --distractor-source scratch,dhuman
```

Arguments: `<model> <dataset_path> <generator_dataset_label> <num_shards> [options]`

Submit from repo root. The sbatch script uses `$SLURM_SUBMIT_DIR` to set the working directory.

## SLURM: Local Models

Use `jobs/run_local_eval.sh`, which submits array jobs via `jobs/local_model_eval.sbatch`.

Smoke run (2 entries per config, to verify model loads correctly):

```bash
jobs/run_local_eval.sh \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --generator-dataset-label gpt-4.1 \
  --dataset-path datasets/augmented/unified_processed_gpt-4.1_20260213_033916 \
  --phase smoke
```

Full run:

```bash
jobs/run_local_eval.sh \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --generator-dataset-label gpt-4.1 \
  --dataset-path datasets/augmented/unified_processed_gpt-4.1_20260213_033916 \
  --phase main \
  --num-shards 8
```

Both phases in sequence:

```bash
jobs/run_local_eval.sh \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --generator-dataset-label gpt-4.1 \
  --dataset-path datasets/augmented/unified_processed_gpt-4.1_20260213_033916 \
  --phase both \
  --num-shards 8
```

See `jobs/README_local_eval.md` for full argument reference and log locations.

**vLLM prerequisite:** Install vLLM before submitting local eval jobs (see Setup above). The job script does not install it automatically.

## Distractor Generation on SLURM

Use `jobs/aug.sh` for CPU-only distractor generation jobs:

```bash
sbatch jobs/aug.sh \
  --model gpt-4.1 \
  --input datasets/processed/unified_processed \
  --output datasets/finished_sets/gpt-4.1
```

Optional flags: `--limit N`, `--save-interval N`.

## Matrix Presets

| Preset | Description |
|---|---|
| `core16` | Core evaluation matrix; 15 unique configs after deduplicating overlap (`3H0M` appears in two groups) |
| `branching21` | Human-prefix branching layout: `0H+1..6M`, `1H+0..5M`, `2H+0..4M`, `3H+0..3M`. Requires branching generation columns (`cond_model_q_a_dhuman_h1/h2/h3`). |

Difficulty is controlled via `--dataset-types` (`arc_easy`, `arc_challenge`, `mmlu_pro`, `gpqa`).

Results are stored at:

```
results/<generator_dataset_label>/<model>_<dataset_type>_<distractor_source>/<nHnM>/results.json
```

## Adding Models

Model aliases are defined in `config/model_aliases.toml`. Add an entry:

```toml
[aliases."my-model"]
provider = "openai"
model_id = "gpt-4.1"
```

For local models with vLLM-specific defaults:

```toml
[aliases."Qwen/Qwen3-4B-Instruct-2507"]
provider = "local"
model_id = "Qwen/Qwen3-4B-Instruct-2507"

[aliases."Qwen/Qwen3-4B-Instruct-2507".defaults]
dtype = "bfloat16"
max_model_len = 32768
```

Supported providers: `openai`, `anthropic`, `gemini`, `deepseek`, `local`.

## Additional Docs

- `docs/evaluation.md` — evaluation matrix details, sharding semantics, preset definitions
- `docs/models.md` — model registry, alias schema, resolution order
- `jobs/README_local_eval.md` — local model SLURM workflow reference
