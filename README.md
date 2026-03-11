# Augmented MCQA
![diagram](diagram-flowchart.png)
## Setup

```bash
cp .env.example .env
uv sync
```

If you will run local `vllm/...` models:

```bash
uv pip install --no-build-isolation 'vllm==0.11.2' 'transformers<5' 'numpy<2.3'
```

Set the provider keys you actually need in `.env`:

- `OPENAI_API_KEY` for GPT-5.2
- `ANTHROPIC_API_KEY` for Claude Opus
- `GOOGLE_API_KEY` for Gemini
- `TOGETHER_API_KEY` for Together-hosted Qwen 3.5 models

## The Normal Pipeline

This is the path the repo is now optimized for:

1. prepare `datasets/processed/unified_processed_v3`
2. generate distractors with one generator model
3. evaluate those generated distractors on the local cluster with `vllm/...` evaluation models
4. analyze the evaluation logs
5. export benchmarker files if needed

API evaluation is not the recommended workflow here. The normal evaluation path is local `vllm/...` only.

## Step 1: Prepare Data

```bash
uv run python main.py prepare-data \
  --step all \
  --output-path datasets/processed/unified_processed_v3
```

## Step 2: Generate Distractors

Pick one of the generator commands below.

### API generators

GPT-5.2:

```bash
uv run python main.py generate \
  --model gpt-5.2-2025-12-11 \
  --run-name gen_gpt52 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --materialize-cache
```

Claude Opus 4.6:

```bash
uv run python main.py generate \
  --model claude-opus-4-6 \
  --run-name gen_claude_opus46 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --materialize-cache
```

Gemini 3.1 Pro:

```bash
uv run python main.py generate \
  --model gemini-3.1-pro-preview \
  --run-name gen_gemini31pro \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --materialize-cache
```

TogetherAI Qwen/Qwen3.5-397B-A17B:

```bash
uv run python main.py generate \
  --model Qwen/Qwen3.5-397B-A17B \
  --run-name gen_together_qwen397b \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --materialize-cache
```

TogetherAI Qwen/Qwen3.5-9B:

```bash
uv run python main.py generate \
  --model Qwen/Qwen3.5-9B \
  --run-name gen_together_qwen9b \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --materialize-cache
```

### Local generators on the cluster

Qwen3-4B:

```bash
uv run python main.py submit-generate-cluster \
  --run-name gen_local_qwen3_4b \
  --models Qwen/Qwen3-4B-Instruct-2507 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --gpu-count 1
```

Olmo3-7B:

```bash
uv run python main.py submit-generate-cluster \
  --run-name gen_local_olmo3_7b \
  --models allenai/Olmo-3-7B-Instruct \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --gpu-count 1
```

If you want to launch both local generation models at once:

```bash
uv run python main.py submit-generate-cluster \
  --run-name gen_local_all \
  --models Qwen/Qwen3-4B-Instruct-2507,allenai/Olmo-3-7B-Instruct \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --gpu-count 2
```

## Step 3: Evaluate On The Local Cluster

The normal evaluation models are:

- `Qwen/Qwen3-4B-Instruct-2507`
- `allenai/Olmo-3-7B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`

The recommended command is always `submit-evaluate-cluster`. That one command schedules all three local evaluation models across the dataset splits.

If you generated with GPT-5.2:

```bash
uv run python main.py submit-evaluate-cluster \
  --run-name eval_gpt52 \
  --generator-run-name gen_gpt52 \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --models Qwen/Qwen3-4B-Instruct-2507,allenai/Olmo-3-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct \
  --gpu-count 3
```

If you generated with Claude Opus 4.6:

```bash
uv run python main.py submit-evaluate-cluster \
  --run-name eval_claude_opus46 \
  --generator-run-name gen_claude_opus46 \
  --generator-model claude-opus-4-6 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --models Qwen/Qwen3-4B-Instruct-2507,allenai/Olmo-3-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct \
  --gpu-count 3
```

If you generated with Gemini 3.1 Pro:

```bash
uv run python main.py submit-evaluate-cluster \
  --run-name eval_gemini31pro \
  --generator-run-name gen_gemini31pro \
  --generator-model gemini-3.1-pro-preview \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --models Qwen/Qwen3-4B-Instruct-2507,allenai/Olmo-3-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct \
  --gpu-count 3
```

If you generated with TogetherAI Qwen/Qwen3.5-397B-A17B:

```bash
uv run python main.py submit-evaluate-cluster \
  --run-name eval_together_qwen397b \
  --generator-run-name gen_together_qwen397b \
  --generator-model Qwen/Qwen3.5-397B-A17B \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --models Qwen/Qwen3-4B-Instruct-2507,allenai/Olmo-3-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct \
  --gpu-count 3
```

If you generated with TogetherAI Qwen/Qwen3.5-9B:

```bash
uv run python main.py submit-evaluate-cluster \
  --run-name eval_together_qwen9b \
  --generator-run-name gen_together_qwen9b \
  --generator-model Qwen/Qwen3.5-9B \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --models Qwen/Qwen3-4B-Instruct-2507,allenai/Olmo-3-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct \
  --gpu-count 3
```

If you generated with local Qwen3-4B:

```bash
uv run python main.py submit-evaluate-cluster \
  --run-name eval_local_qwen3_4b \
  --generator-run-name gen_local_qwen3_4b \
  --generator-model Qwen/Qwen3-4B-Instruct-2507 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --models Qwen/Qwen3-4B-Instruct-2507,allenai/Olmo-3-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct \
  --gpu-count 3
```

If you generated with local Olmo3-7B:

```bash
uv run python main.py submit-evaluate-cluster \
  --run-name eval_local_olmo3_7b \
  --generator-run-name gen_local_olmo3_7b \
  --generator-model allenai/Olmo-3-7B-Instruct \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --models Qwen/Qwen3-4B-Instruct-2507,allenai/Olmo-3-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct \
  --gpu-count 3
```

Notes:

- `submit-generate-cluster` defaults to the two local generation models, so with the default three dataset splits it creates `2 × 3 = 6` tasks.
- `submit-evaluate-cluster` defaults to the three local evaluation models, so with the default three dataset splits it creates `3 × 3 = 9` tasks.
- `--gpu-count` is a concurrency cap on the SLURM array. If you omit it, the full array is submitted without a `%N` cap.

## Step 4: Analyze

```bash
uv run python main.py analyze \
  --results-root results/inspect/evaluation \
  --output-dir results/final5_plots
```

## Step 5: Export Benchmarker Items

Example for the GPT-5.2 generation run:

```bash
uv run python main.py export \
  --generator-run-name gen_gpt52 \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3
```

If you used a different generator, keep the same command shape and change `--generator-run-name` and `--generator-model` to match Step 2.

## Optional: Standalone Benchmarker Writing-Flaw Analysis

Example for the GPT-5.2 generation run:

```bash
uv run python analysis/benchmarker_analysis.py \
  --writing-flaw-jsonl datasets/benchmarker_results/atrey_writing_flaw_rows.jsonl.zip \
  --results-root results/inspect/evaluation \
  --cache-root datasets/augmented \
  --generator-run-name gen_gpt52 \
  --generator-model gpt-5.2-2025-12-11 \
  --output-dir analysis/figures/benchmarker
```

## Canonical Artifacts

- processed dataset: `datasets/processed/unified_processed_v3`
- generation logs: `results/inspect/generation/<run>/<model>/`
- evaluation logs: `results/inspect/evaluation/<run>/<generator_run>/<generator_model>/<eval_model>/`
- augmented cache: `datasets/augmented/<run>/<model>/`
- cluster bundles: `jobs/generated/<stage>/<run>/`
- cluster logs: `logs/slurm/<stage>/<run>/`

Cluster generation uses dataset-scoped caches under `datasets/augmented/<run>/<model>/<dataset>/` so concurrent jobs do not overwrite each other.

## More Detail

- [`docs/cli-reference.md`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/docs/cli-reference.md)
- [`docs/architecture.md`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/docs/architecture.md)
