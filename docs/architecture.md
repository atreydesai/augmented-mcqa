# Architecture

This repo has one canonical entrypoint: [`main.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/main.py).

## Runtime Shape

```text
prepare-data -> generate -> augmented cache -> evaluate -> analyze/export
```

- Inspect `.eval` logs are the source of truth.
- The augmented Hugging Face dataset cache is derived from generation logs.
- Cluster runs are just generated SLURM arrays that call back into `python main.py ...`.

## Main Modules

- [`main.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/main.py)
  Orchestrates the CLI, launches Inspect, materializes caches, and generates/submits SLURM bundles.
- [`data/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/data)
  Raw-data download, dataset processing, augmented-cache materialization, and benchmarker export.
- [`tasks/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/tasks)
  Builds Inspect `Task` objects for generation and evaluation.
- [`solvers/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/solvers)
  Prompt construction, model calls, and response parsing.
- [`scorers/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/scorers)
  Converts solver outputs into Inspect scores and metadata.
- [`analysis/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/analysis)
  Reads Inspect logs and produces plots, summary tables, and the standalone benchmarker analysis.
- [`utils/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/utils)
  Shared constants, model aliasing, parsing, sharding, log helpers, and cluster bundle rendering.

## Data Flow

### Processed dataset

[`data/pipeline.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/data/pipeline.py) builds a unified `DatasetDict` with splits:

- `arc_challenge`
- `mmlu_pro`
- `gpqa`

The per-dataset processors are:

- [`data/arc_processor.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/data/arc_processor.py)
- [`data/mmlu_pro_processor.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/data/mmlu_pro_processor.py)
- [`data/gpqa_processor.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/data/gpqa_processor.py)

### Generation

[`tasks/generation.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/tasks/generation.py) builds one Inspect task over the processed dataset. The solver in [`solvers/final5_generation.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/solvers/final5_generation.py) creates the five Final5 settings:

- `human_from_scratch`
- `model_from_scratch`
- `augment_human`
- `augment_model`
- `augment_ablation`

Generation prompts use XML-style sections and ask for JSON keyed by distractor label. The parser in [`utils/parsing.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/utils/parsing.py) validates that JSON first, and still accepts the older labeled-line format as a backward-compatible fallback during retries.

### Augmented cache

[`data/final5_store.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/data/final5_store.py) reads generation `.eval` logs and rebuilds an augmented `DatasetDict` containing:

- generated distractor columns
- randomized options per setting
- correct-answer letters per setting

### Evaluation

[`tasks/evaluation.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/tasks/evaluation.py) builds one Inspect task per `setting × mode`. The evaluation solver in [`solvers/final5_evaluation.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/solvers/final5_evaluation.py) prompts for a single answer letter in:

- `full_question`
- `choices_only`

The evaluation prompts also use XML-style sections and require a small JSON object with an `"answer"` key in both modes. The evaluation scorer in [`scorers/evaluation.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/scorers/evaluation.py) records correctness plus metadata needed by analysis.

## Model Resolution

[`utils/modeling.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/utils/modeling.py) maps short aliases to Inspect model ids.

Examples:

- `gpt-5.2-2025-12-11` -> `openai/gpt-5.2-2025-12-11`
- `claude-opus-4-6` -> `anthropic/claude-opus-4-6`
- `Qwen/Qwen3.5-397B-A17B` -> `together/Qwen/Qwen3.5-397B-A17B`
- `Qwen/Qwen3-4B-Instruct-2507` -> `vllm/Qwen/Qwen3-4B-Instruct-2507`

You can also pass provider-qualified model ids directly.

## Cluster Submit Flow

The supported cluster interface is:

- `main.py submit-generate-cluster`
- `main.py submit-evaluate-cluster`

Those commands:

- only support local `vllm/...` models
- create one SLURM array task per `model × dataset`
- optionally cap array concurrency with `--gpu-count`
- keep settings and modes grouped inside a task to avoid repeated cold starts within that job

The SLURM rendering logic lives in [`utils/cluster_submit.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/utils/cluster_submit.py).

## Artifact Layout

- processed dataset: `datasets/processed/unified_processed_v3`
- generation logs: `results/inspect/generation/<run>/<model>/`
- evaluation logs: `results/inspect/evaluation/<run>/<generator_run>/<generator_model>/<eval_model>/`
- augmented cache: `datasets/augmented/<run>/<model>/`
- cluster bundles: `jobs/generated/<stage>/<run>/`
- cluster logs: `logs/slurm/<stage>/<run>/`

Cluster generation uses dataset-scoped caches under `datasets/augmented/<run>/<model>/<dataset>/` so concurrent jobs do not overwrite each other.

## Tests

The test suite lives under [`tests/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/tests). Pytest output is redirected into `test-artifacts/pytest/` by [`tests/conftest.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/tests/conftest.py) so normal runtime directories stay clean.
