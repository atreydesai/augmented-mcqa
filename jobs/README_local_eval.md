# Local Model Evaluation (SLURM)

This documents the workflow for running local vLLM model evaluations on the cluster using `jobs/run_local_eval.sh` and `jobs/local_model_eval.sbatch`.

## Prerequisites

One-time setup from repo root:

```bash
uv sync
uv pip install --no-build-isolation 'vllm==0.11.2' 'transformers<5' 'numpy<2.3'
```

Stage model weights:

```bash
huggingface-cli download <model_id> --local-dir /path/to/cache
```

## Usage

```
jobs/run_local_eval.sh --model <model> --generator-dataset-label <label> \
  --dataset-path <path> [options]

Required:
  --model <model>                    Model alias (must match config/model_aliases.toml)
  --generator-dataset-label <label>  Generator label (e.g. gpt-4.1, opus)
  --dataset-path <path>              Path to augmented dataset directory

Options:
  --num-shards <int>         SLURM array shards (default: 8)
  --phase <smoke|main|both>  Which phase(s) to run (default: both)
  --smoke-limit <int>        Entries per config in smoke phase (default: 2)
  --preset <name>            Matrix preset: core16 or branching21 (default: core16)
  --dataset-types <csv>      Comma-separated dataset types (default: all)
  --distractor-source <csv>  Comma-separated distractor sources (default: all)
  --output-dir <path>        Results directory (default: results)
  --save-interval <int>      Checkpoint save interval (default: 200)
  --keep-checkpoints <int>   Checkpoints to keep per root (default: 2)
  --max-tokens <int>         Max generation tokens (default: 150)
```

## Smoke + Main Workflow

### Step 1: Smoke run

Run a small subset (default 2 entries per config) to verify the model loads and produces valid output:

```bash
jobs/run_local_eval.sh \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --generator-dataset-label gpt-4.1 \
  --dataset-path datasets/augmented/unified_processed_gpt-4.1_20260213_033916 \
  --phase smoke \
  --num-shards 3
```

Check the smoke logs before proceeding:

```
logs/local_eval/slurm_<job_id>_<array_task>.out
logs/local_eval/slurm_<job_id>_<array_task>.err
```

### Step 2: Main run

After smoke looks good, submit the full evaluation:

```bash
jobs/run_local_eval.sh \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --generator-dataset-label gpt-4.1 \
  --dataset-path datasets/augmented/unified_processed_gpt-4.1_20260213_033916 \
  --phase main \
  --num-shards 8
```

Or run both phases in one call (smoke first, then main):

```bash
jobs/run_local_eval.sh \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --generator-dataset-label gpt-4.1 \
  --dataset-path datasets/augmented/unified_processed_gpt-4.1_20260213_033916 \
  --phase both \
  --num-shards 8
```

## Log Locations

| Item | Location |
|---|---|
| SLURM stdout | `logs/local_eval/slurm_<A>_<a>.out` |
| SLURM stderr | `logs/local_eval/slurm_<A>_<a>.err` |
| Results | `results/<generator_dataset_label>/<model>_<dataset_type>_<distractor_source>/<nHnM>/results.json` |

## Restart / Resume

All runs use `--skip-existing` by default. To resume a partial run, re-submit with the same arguments and SLURM will skip already-completed entries.

If individual shards failed, re-submit targeting only failed shards using `--array` override:

```bash
sbatch --array=2,5,7 \
  --export=ALL,MODEL=...,DATASET_PATH=...,GENERATOR_DATASET_LABEL=...,NUM_SHARDS=8,... \
  jobs/local_model_eval.sbatch
```

## Default Models

Models with pre-configured aliases in `config/model_aliases.toml`:

- `Nanbeige/Nanbeige4.1-3B`
- `Qwen/Qwen3-4B-Instruct-2507`
- `allenai/Olmo-3-7B-Instruct`

See `docs/models.md` for alias configuration details.
