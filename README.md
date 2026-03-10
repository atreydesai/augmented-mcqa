# Augmented MCQA

Inspect-first Final5 generation and evaluation for `arc_challenge`, `mmlu_pro`, and `gpqa`.

The canonical interface is now [`main.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/main.py). Numbered scripts in [`scripts/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/scripts) remain as thin compatibility wrappers where that helps with existing workflows.

## What Changed

- Generation and evaluation are both built on [Inspect AI](https://inspect.aisi.org.uk/).
- Structured outputs are gone. Final5 generation now uses plain-text labeled lines with strict parsing and retry.
- Inspect `.eval` logs are the canonical run artifact.
- Augmented Hugging Face datasets under `datasets/augmented/` are derived caches for reuse and export.
- SLURM support remains, but only as thin shard launchers over `main.py`.

## Core Commands

| Command | Purpose |
|---|---|
| `main.py prepare-data` | Download and process raw datasets |
| `main.py generate` | Run one generation model over the processed dataset |
| `main.py generate-all` | Run the full generator model list |
| `main.py evaluate` | Evaluate one model against one generation run |
| `main.py evaluate-all` | Evaluate the full local/API eval model list |
| `main.py analyze` | Aggregate Inspect logs into plots and summary tables |
| `main.py export` | Export a derived augmented dataset to benchmarker JSONL |

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

## Execution Modes

API models:
- Pass a provider-qualified Inspect model id directly, such as `openai/gpt-5.2-2025-12-11`.
- Or use the short aliases in [`utils/modeling.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/utils/modeling.py).

Local models:
- Use the built-in aliases such as `Qwen/Qwen3-4B-Instruct-2507`, which resolve to `vllm/...`.
- Or point at an OpenAI-compatible local endpoint with `--backend openai --model-base-url ...`.

Sharded cluster runs:
- Every `generate` and `evaluate` command accepts `--shard-count`, `--shard-index`, and `--shard-strategy`.
- Stable sample ids keep shard membership deterministic for resume and retry.
- Thin launch helpers live in [`jobs/run_generate_shard.sh`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/jobs/run_generate_shard.sh), [`jobs/run_evaluate_shard.sh`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/jobs/run_evaluate_shard.sh), [`jobs/generate_array.sbatch`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/jobs/generate_array.sbatch), and [`jobs/evaluate_array.sbatch`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/jobs/evaluate_array.sbatch).

## Canonical Artifacts

- Generation logs: `results/inspect/generation/<run>/<model>/`
- Evaluation logs: `results/inspect/evaluation/<run>/<generator_run>/<generator_model>/<eval_model>/`
- Derived augmented dataset cache: `datasets/augmented/<run>/<model>/`

Analysis reads Inspect logs directly. There is no merge stage in the canonical workflow anymore.

## Repo Layout

- [`main.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/main.py): primary CLI
- [`tasks/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/tasks): Inspect task factories
- [`solvers/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/solvers): prompt and parsing logic
- [`scorers/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/scorers): Final5 scoring and metadata emission
- [`data/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/data): processed dataset loading, cache materialization, export
- [`utils/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/utils): constants, model aliasing, sharding, parsing, log helpers
- [`prompts/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/prompts): plain-text prompt templates

## Compatibility Scripts

These still exist, but they now just forward into the Inspect-first core:

- [`scripts/02_generate_distractors.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/scripts/02_generate_distractors.py)
- [`scripts/03_regenerate_experiments.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/scripts/03_regenerate_experiments.py)
- [`scripts/04_eval_matrix.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/scripts/04_eval_matrix.py)
- [`scripts/07_export_benchmarker_items.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/scripts/07_export_benchmarker_items.py)
- [`scripts/08_analyze.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/scripts/08_analyze.py)

## Additional Docs

- [`docs/pipeline.md`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/docs/pipeline.md)
- [`docs/models.md`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/docs/models.md)
- [`docs/sharding_and_recombination.md`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/docs/sharding_and_recombination.md)
