# Augmented MCQA

LLM generation and evaluation tooling for multiple-choice distractor experiments.

Use `uv` for everything:

```bash
uv run python main.py --help
```

## What This Repo Does

The normal flow is:

```text
prepare-data -> generate -> materialize augmented store -> evaluate -> analyze/export
```

Important rules:

- Inspect `.eval` logs are the source of truth.
- Generated datasets are stored under `datasets/augmented/<run>/<model>/`.
- That store is setting-scoped, not one wide merged table.
- Models must exist in `config/model_registry.json`.
- Setting definitions, prompt templates, and distractor counts live in `config/generation_recipes.json`.
- `--processed-dataset` accepts either:
  - the unified processed dataset directory
  - a dataset manifest JSON for a custom dataset

## Repo Layout

- processed dataset: `datasets/processed/unified_processed_v3`
- generation logs: `results/inspect/generation/<run>/<model>/`
- evaluation logs: `results/inspect/evaluation/<run>/<generator_run>/<generator_model>/<eval_model>/`
- augmented store root: `datasets/augmented/<run>/<model>/`
- per-setting records: `datasets/augmented/<run>/<model>/<dataset>/<setting>/`
- benchmarker exports: `datasets/benchmarker_items/<store-name>/`
- model registry: `config/model_registry.json`
- recipe config: `config/generation_recipes.json`
- prompts: `prompts/`

Cluster note:

- `--gpu-count` on `submit-generate-cluster` and `submit-evaluate-cluster` is a per-resource-class concurrency cap for scheduler submission. It is not a "GPUs per job" flag.

## Quick Start

### 1. Build the processed benchmark dataset

```bash
uv run python main.py prepare-data \
  --step all \
  --output-path datasets/processed/unified_processed_v3
```

### 2. Run generation for one model

```bash
uv run python main.py generate \
  --model gpt-5.2-2025-12-11 \
  --run-name gen_gpt52 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --dataset-types arc_challenge,mmlu_pro,gpqa \
  --materialize-cache
```

### 3. Evaluate one model against that generation run

```bash
uv run python main.py evaluate \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --run-name eval_qwen4b_on_gpt52 \
  --generator-run-name gen_gpt52 \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3
```

### 4. Export benchmarker JSONL files

```bash
uv run python main.py export \
  --generator-run-name gen_gpt52 \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3
```

### 5. Make plots and summary tables

```bash
uv run python main.py analyze \
  --results-root results/inspect/evaluation \
  --output-dir results/final5_plots
```

## Commands You Will Actually Use

### Prepare data

Build everything:

```bash
uv run python main.py prepare-data \
  --step all \
  --output-path datasets/processed/unified_processed_v3
```

Build a smaller processed dataset for debugging:

```bash
uv run python main.py prepare-data \
  --step all \
  --limit 20 \
  --output-path datasets/processed/unified_processed_debug20
```

### Direct generation

Generate only `model_from_scratch` for GPQA:

```bash
uv run python main.py generate \
  --model gpt-5.2-2025-12-11 \
  --run-name gen_gpqa_model_only \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --dataset-types gpqa \
  --generation-strategies model_from_scratch \
  --materialize-cache
```

Generate only the first 10 rows per dataset:

```bash
uv run python main.py generate \
  --model claude-opus-4-6 \
  --run-name gen_claude_small \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --limit 10 \
  --materialize-cache
```

Rerun only `augment_model` after changing its prompt or recipe:

```bash
uv run python main.py generate \
  --model gpt-5.2-2025-12-11 \
  --run-name gen_gpt52 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --dataset-types gpqa \
  --generation-strategies augment_model \
  --rebuild-cache \
  --materialize-cache
```

Rebuild the setting-scoped augmented store from existing logs:

```bash
uv run python main.py materialize-generation-cache \
  --run-name gen_gpt52 \
  --model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3
```

### Cluster generation

Write a preview submission bundle without calling `sbatch`:

```bash
uv run python main.py submit-generate-cluster \
  --run-name gen_preview \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --models Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-types arc_challenge,gpqa \
  --generation-strategies model_from_scratch,augment_model \
  --questions-per-job 100 \
  --write-only \
  --render-status
```

Mix local and API models in one submission:

```bash
uv run python main.py submit-generate-cluster \
  --run-name gen_mixed \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --models Qwen/Qwen3-4B-Instruct-2507,allenai/Olmo-3-7B-Instruct,gpt-5.2-2025-12-11 \
  --generation-strategies model_from_scratch,augment_human,augment_model,augment_ablation \
  --questions-per-job 200 \
  --gpu-count 4 \
  --render-status
```

### Direct evaluation

Evaluate only `choices_only` mode:

```bash
uv run python main.py evaluate \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --run-name eval_choices_only \
  --generator-run-name gen_gpt52 \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --modes choices_only
```

Evaluate only `augment_model` on GPQA:

```bash
uv run python main.py evaluate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --run-name eval_gpqa_aug_model \
  --generator-run-name gen_gpt52 \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --dataset-types gpqa \
  --settings augment_model
```

### Cluster evaluation

Schedule all five settings across two modes:

```bash
uv run python main.py submit-evaluate-cluster \
  --run-name eval_all \
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

Rerun one evaluation family after generation changed:

```bash
uv run python main.py submit-evaluate-cluster \
  --run-name eval_all \
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

### Export and analysis

Export from an explicit augmented store path:

```bash
uv run python main.py export \
  --input datasets/augmented/gen_gpt52/openai_gpt-5.2-2025-12-11 \
  --output-root datasets/benchmarker_items
```

Run standalone benchmarker writing-flaw analysis:

```bash
uv run python analysis/benchmarker_analysis.py \
  --writing-flaw-jsonl datasets/benchmarker_results/atrey_writing_flaw_rows.jsonl.zip \
  --results-root results/inspect/evaluation \
  --cache-root datasets/augmented \
  --generator-run-name gen_gpt52 \
  --generator-model gpt-5.2-2025-12-11 \
  --output-dir analysis/figures/benchmarker
```

## Custom Datasets

If you want to use your own dataset, point `--processed-dataset` at a dataset manifest JSON instead of a Hugging Face `DatasetDict`.

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

Then run generation against it:

```bash
uv run python main.py generate \
  --model gpt-5.2-2025-12-11 \
  --run-name gen_custom \
  --processed-dataset datasets/custom/dataset_manifest.json \
  --dataset-types custom_benchmark \
  --materialize-cache
```

## Custom Models

All generation and evaluation models must be registered in `config/model_registry.json`.

Example entry:

```json
{
  "name": "my-local-model",
  "resolved": "vllm/my-org/my-local-model"
}
```

Then use:

```bash
uv run python main.py generate \
  --model my-local-model \
  --run-name gen_my_local \
  --processed-dataset datasets/processed/unified_processed_v3
```

## Custom Prompts and Choice Counts

Edit `config/generation_recipes.json`.

That file controls:

- setting name
- prompt template file
- prompt mode
- prerequisite setting
- number of human distractors
- number of model distractors
- total number of choices
- how many new distractors generation should produce

Typical uses:

- change `augment_model` prompt template
- increase `augment_ablation` from 9 distractors to another value
- run a smaller `model_from_scratch` ablation

After changing recipes, rerun generation for the affected setting and rebuild the augmented store:

```bash
uv run python main.py generate \
  --model gpt-5.2-2025-12-11 \
  --run-name gen_gpt52 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --generation-strategies augment_model \
  --rebuild-cache \
  --materialize-cache
```

## Schedulable Generation Strategies

- `model_from_scratch`
- `augment_human`
- `augment_model`
- `augment_ablation`

`human_from_scratch` is stored and evaluated like the others, but it is not scheduled as its own generation slice.

## Evaluation Settings

- `human_from_scratch`
- `model_from_scratch`
- `augment_human`
- `augment_model`
- `augment_ablation`

## More Detail

- [CLI reference](docs/cli-reference.md)
- [Architecture](docs/architecture.md)
