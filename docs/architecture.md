# Architecture

This repo has one canonical entrypoint: [`main.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/main.py).

## Runtime Shape

```text
prepare-data -> generate -> augmented cache -> evaluate -> analyze/export
```

- Inspect `.eval` logs are the source of truth.
- The augmented Hugging Face dataset cache is derived from generation logs.
- Cluster runs are dependency-aware SLURM bundles that call back into `python main.py ...` for each schedulable slice.

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

[`tasks/generation.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/tasks/generation.py) now builds one Inspect task per requested generation strategy over the selected processed-dataset slice. The solver in [`solvers/final5_generation.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/solvers/final5_generation.py) fills the Final5 settings:

- `human_from_scratch`
- `model_from_scratch`
- `augment_human`
- `augment_model`
- `augment_ablation`

`human_from_scratch` is implicit. The schedulable strategies are:

- `model_from_scratch`
- `augment_human`
- `augment_model`
- `augment_ablation`

`augment_model` uses `model_from_scratch` from prior logs as its prerequisite.

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

- support both local `vllm/...` models and hosted/API models
- create one schedulable slice per:
  - generation: `model × dataset × strategy × question_chunk`
  - evaluation: `model × dataset × setting × mode × question_chunk`
- write one manifest per submission plus a master `submit_all.sh`
- submit one `sbatch` job per slice instead of one array element per slice family
- attach exact `afterok` prerequisites between matching slices, for example `augment_model` after `model_from_scratch`
- use `afterany` slot throttles when `--gpu-count` is acting as a concurrency cap
- keep a merged run state in `scheduler_state.json` and optionally render `scheduler_status.html`

Local slices request one GPU. API slices request no GPU.

The SLURM rendering logic lives in [`utils/cluster_submit.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/utils/cluster_submit.py).
The run-state and dashboard logic lives in [`utils/scheduler_state.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/utils/scheduler_state.py).

## Artifact Layout

- processed dataset: `datasets/processed/unified_processed_v3`
- generation logs: `results/inspect/generation/<run>/<model>/`
- evaluation logs: `results/inspect/evaluation/<run>/<generator_run>/<generator_model>/<eval_model>/`
- augmented cache: `datasets/augmented/<run>/<model>/`
- cluster bundles: `jobs/generated/<stage>/<run>/`
- submission manifests: `jobs/generated/<stage>/<run>/submissions/<submission_id>/manifest.json`
- master submit scripts: `jobs/generated/<stage>/<run>/submissions/<submission_id>/submit_all.sh`
- scheduler state: `jobs/generated/<stage>/<run>/scheduler_state.json`
- scheduler dashboard: `jobs/generated/<stage>/<run>/scheduler_status.html`
- cluster logs: `logs/slurm/<stage>/<run>/`

## Tests

The test suite lives under [`tests/`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/tests). Pytest output is redirected into `test-artifacts/pytest/` by [`tests/conftest.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/tests/conftest.py) so normal runtime directories stay clean.
