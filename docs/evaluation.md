# Evaluation Matrix Guide

## Primary CLI

Use `scripts/eval_matrix.py`.

```bash
uv run python scripts/eval_matrix.py plan ...
uv run python scripts/eval_matrix.py run ...
```

## Subcommands

## `plan`

Builds deterministic configs and writes a manifest.

```bash
uv run python scripts/eval_matrix.py plan \
  --preset core16 \
  --model gpt-4.1 \
  --dataset-path datasets/augmented/unified_processed_gpt-4.1_20260213_033916 \
  --manifest-out results/manifests/my_plan.json \
  --print-configs
```

## `run`

Runs sequentially in deterministic order.

```bash
uv run python scripts/eval_matrix.py run \
  --preset core16 \
  --model gpt-4.1 \
  --dataset-path datasets/augmented/unified_processed_gpt-4.1_20260213_033916 \
  --skip-existing
```

Or run from manifest:

```bash
uv run python scripts/eval_matrix.py run --manifest results/manifests/my_plan.json --skip-existing
```

## Presets

- `core16`
  - Historical label for the core matrix
  - 15 unique configs after deduplicating overlap (`3H0M` appears in two conceptual groups)
- `branching21`
  - Full grid `1H..3H` x `0M..6M`
  - Cumulative branching semantics per question:
    - Example: `1H0M -> 1H1M -> 1H2M` carries the same selected `H` distractor forward and adds `M` distractors by deterministic prefix expansion
    - Option order is re-shuffled deterministically for each config

Difficulty is represented by dataset type selection, not a separate pipeline:

- `arc_easy`
- `arc_challenge`
- `mmlu_pro`
- `supergpqa`

## Sharding Semantics

Sharding uses deterministic round-robin over configs sorted by `config_id`.

- Config index `i` goes to shard `i % num_shards`
- Run shard `k` with:

```bash
--num-shards N --shard-index k
```

Example:

```bash
uv run python scripts/eval_matrix.py run \
  --preset core16 \
  --model gpt-4.1 \
  --dataset-path datasets/augmented/unified_processed_gpt-4.1_20260213_033916 \
  --num-shards 8 \
  --shard-index 0 \
  --skip-existing
```

## SLURM Array Workflow

Use:

- `jobs/eval_matrix_array.sbatch`
- `jobs/submit_eval_array.sh`

Submit example:

```bash
jobs/submit_eval_array.sh \
  gpt-4.1 \
  datasets/augmented/unified_processed_gpt-4.1_20260213_033916 \
  8 \
  --dataset-types mmlu_pro,supergpqa \
  --distractor-source scratch,dhuman
```

## Outputs

Result paths are unchanged:

```text
results/<model>_<dataset_type>_<distractor_source>/<nHnM>/results.json
```

Batch summary files:

- Non-sharded: `results/batch_summary_<model>.json`
- Sharded: `results/batch_summary_<model>_shard_<i>_of_<n>.json`

Per-question rows in `results.json` now also include evaluation trace fields aligned with generation-side transparency:

- `eval_options_randomized` / `options_randomized`
- `eval_correct_answer_letter` / `correct_answer_letter`
- `eval_full_question` / `full_question`
- `eval_model_input` / `model_input`
- `eval_model_output` / `model_output`
- `selected_human_distractors`, `selected_model_distractors`
- `human_option_indices`, `model_option_indices`

## Failure/Restart Pattern

Recommended restart strategy:

1. Always run with `--skip-existing`
2. Re-run failed shard(s) only
3. Keep shard count fixed for reproducibility
4. Use manifests for exact reproducible reruns
