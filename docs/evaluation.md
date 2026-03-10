# Evaluation

Final5 evaluation is now Inspect-native.

## Active Preset

`final5` is the only active evaluation shape:

- `human_from_scratch`
- `model_from_scratch`
- `augment_human`
- `augment_model`
- `augment_ablation`

Modes:

- `full_question`
- `choices_only`

## Canonical Execution

Run evaluation with [`main.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/main.py):

```bash
uv run python main.py evaluate \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --run-name eval_local \
  --generator-run-name gen_openai \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v2
```

Or run the default eval matrix:

```bash
uv run python main.py evaluate-all \
  --run-name eval_all \
  --generator-run-name gen_openai \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v2
```

## Output Layout

Inspect logs:

```text
results/inspect/evaluation/<run_name>/<generator_run_name>/<generator_model>/<eval_model>/**/*.eval
```

Derived generation cache used as evaluation input:

```text
datasets/augmented/<generator_run_name>/<generator_model>/
```

There is no canonical `summary.json` + `rows/` output tree anymore. Analysis reads the `.eval` logs directly.

## SLURM

Per-model bundle generation:

```bash
uv run python scripts/05_build_eval_slurm_bundle.py \
  --run-name eval_cluster \
  --generator-run-name gen_openai \
  --generator-model gpt-5.2-2025-12-11 \
  --output-dir jobs/generated/eval_cluster \
  --shard-count 8
```

Submit:

```bash
bash jobs/generated/eval_cluster/submit_all.sh
```

Each sbatch file launches `main.py evaluate` with one evaluation model and one shard index per array task.

## Sharding

Evaluation supports:

- `--shard-count`
- `--shard-index`
- `--shard-strategy`

Sharding is deterministic because task construction uses stable sample ids over the unified dataset.

## Notes

- `mmlu_pro` preprocessing still keeps the existing exact-match filter against raw `mmlu`.
- The compatibility script [`scripts/04_eval_matrix.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/scripts/04_eval_matrix.py) now forwards straight into `main.py evaluate`.
- [`scripts/06_merge_eval_subshards.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/scripts/06_merge_eval_subshards.py) intentionally does nothing beyond printing that Inspect logs are canonical.
