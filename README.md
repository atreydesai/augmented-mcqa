# Augmented MCQA

Inspect-first Final5 generation and evaluation for `arc_challenge`, `mmlu_pro`, and `gpqa`.

The canonical interface is [`main.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/main.py). The old numbered script wrappers have been removed.

## What Changed

- Generation and evaluation are both built on [Inspect AI](https://inspect.aisi.org.uk/).
- Structured outputs are gone. Final5 generation now uses plain-text labeled lines with strict parsing and retry.
- Inspect `.eval` logs are the canonical run artifact.
- Augmented Hugging Face datasets under `datasets/augmented/` are derived caches for reuse and export.
- SLURM support now goes through dataset-aware submit commands that generate one job per `model × dataset`.

## Core Commands

| Command | Purpose |
|---|---|
| `main.py prepare-data` | Download and process raw datasets |
| `main.py generate` | Run one generation model over the processed dataset |
| `main.py generate-all` | Run the full generator model list |
| `main.py evaluate` | Evaluate one model against one generation run |
| `main.py evaluate-all` | Evaluate the full local/API eval model list |
| `main.py submit-generate-cluster` | Submit local generation jobs as one SLURM array over `model × dataset` |
| `main.py submit-evaluate-cluster` | Submit local evaluation jobs as one SLURM array over `model × dataset` |
| `main.py analyze` | Aggregate Inspect logs into plots and summary tables |
| `main.py signature-table` | Build a behavioral-signature table from Inspect logs |
| `main.py export` | Export a derived augmented dataset to benchmarker JSONL |
| `main.py diagnose-failures` | Find incomplete rows in an augmented cache |
| `main.py diagnose-trace` | Dump generation traces from Inspect logs |
| `main.py smoke-generate` | Run small generation smoke jobs |
| `main.py smoke-evaluate` | Run small evaluation smoke jobs |

## Setup

```bash
cp .env.example .env
uv sync
```

For local vLLM-backed runs:

```bash
uv pip install --no-build-isolation 'vllm==0.11.2' 'transformers<5' 'numpy<2.3'
```

## Quick Start

Prepare the unified processed dataset:

```bash
uv run python main.py prepare-data --step all --output-path datasets/processed/unified_processed_v2
```

Run API-backed generation:

```bash
uv run python main.py generate \
  --model gpt-5.2-2025-12-11 \
  --run-name gen_openai \
  --processed-dataset datasets/processed/unified_processed_v2 \
  --materialize-cache
```

Run local generation against an OpenAI-compatible server:

```bash
uv run python main.py generate \
  --model my-local-model \
  --backend openai \
  --model-base-url http://localhost:8000/v1 \
  --run-name gen_local \
  --processed-dataset datasets/processed/unified_processed_v2 \
  --materialize-cache
```

Evaluate with a local vLLM model:

```bash
uv run python main.py evaluate \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --run-name eval_local \
  --generator-run-name gen_openai \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v2
```

Submit all local generation jobs from the login node:

```bash
uv run python main.py submit-generate-cluster \
  --run-name gen_cluster \
  --processed-dataset datasets/processed/unified_processed_v2
```

Submit all local evaluation jobs with a 4-GPU concurrency cap:

```bash
uv run python main.py submit-evaluate-cluster \
  --run-name eval_cluster \
  --generator-run-name gen_openai \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v2 \
  --gpu-count 4
```

Analyze Inspect logs:

```bash
uv run python main.py analyze \
  --results-root results/inspect/evaluation \
  --output-dir results/final5_plots
```

Export benchmarker items from the derived augmented cache:

```bash
uv run python main.py export \
  --generator-run-name gen_openai \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v2
```

Run the standalone benchmarker writing-flaw analysis against Inspect eval logs plus the derived augmented cache:

```bash
uv run python analysis/benchmarker_analysis.py \
  --writing-flaw-jsonl datasets/benchmarker_results/atrey_writing_flaw_rows.jsonl.zip \
  --results-root results/inspect/evaluation \
  --cache-root datasets/augmented \
  --generator-run-name gen_openai \
  --generator-model gpt-5.2-2025-12-11 \
  --output-dir analysis/figures/benchmarker
```

## Execution Modes

API models:
- Pass a provider-qualified Inspect model id directly, such as `openai/gpt-5.2-2025-12-11`.
- Or use the short aliases in [`utils/modeling.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/utils/modeling.py).

Local models:
- Use the built-in aliases such as `Qwen/Qwen3-4B-Instruct-2507`, which resolve to `vllm/...`.
- Or point at an OpenAI-compatible local endpoint with `--backend openai --model-base-url ...`.

Sharded cluster runs:
- `submit-generate-cluster` and `submit-evaluate-cluster` are for local `vllm/...` models only.
- Each array task owns one local model and one dataset split, then runs all work for that pair before exiting.
- With the default 3 local models and 3 datasets, each command writes 9 tasks.
- `--gpu-count` adds an array cap like `%4`. If omitted, the full array is submitted without a concurrency cap.
- Cold starts are avoided within a job because settings and modes stay grouped inside that task. There is still one cold start per `model × dataset` job.
- Low-level `--shard-count` / `--shard-index` remain available on direct `generate` and `evaluate`, but they are no longer the primary documented cluster interface.

## Canonical Artifacts

- Generation logs: `results/inspect/generation/<run>/<model>/`
- Evaluation logs: `results/inspect/evaluation/<run>/<generator_run>/<generator_model>/<eval_model>/`
- Derived augmented dataset cache: `datasets/augmented/<run>/<model>/`
- Cluster bundle artifacts: `jobs/generated/<stage>/<run>/`
- Cluster runtime logs: `logs/slurm/<stage>/<run>/`

Cluster submit generation uses dataset-scoped cache directories under `datasets/augmented/<run>/<model>/<dataset>/` so concurrent jobs do not overwrite each other.

Analysis reads Inspect logs directly. There is no merge stage in the canonical workflow anymore.

## Repo Layout

- [`main.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/main.py): primary CLI
- [`tasks/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/tasks): Inspect task factories
- [`solvers/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/solvers): prompt and parsing logic
- [`scorers/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/scorers): Final5 scoring and metadata emission
- [`data/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/data): processed dataset loading, cache materialization, export
- [`utils/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/utils): constants, model aliasing, sharding, parsing, log helpers
- [`prompts/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/prompts): plain-text prompt templates

## Additional Docs

- [`docs/pipeline.md`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/docs/pipeline.md)
- [`docs/evaluation.md`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/docs/evaluation.md)
- [`docs/models.md`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/docs/models.md)
- [`docs/sharding_and_recombination.md`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/docs/sharding_and_recombination.md)
- [`jobs/README_local_eval.md`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/jobs/README_local_eval.md)
