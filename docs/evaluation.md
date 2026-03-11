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

Cluster submit jobs use dataset-scoped cache paths under `datasets/augmented/<generator_run_name>/<generator_model>/<dataset>/`.

There is no canonical `summary.json` + `rows/` output tree anymore. Analysis reads the `.eval` logs directly.

## Login-Node Cluster Submit

Submit all local evaluation jobs in one command:

```bash
uv run python main.py submit-evaluate-cluster \
  --run-name eval_cluster \
  --generator-run-name gen_openai \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v2
```

This command:

- only accepts local models that resolve to `vllm/...`
- writes one array task per `eval model × dataset`
- runs all five Final5 settings and both modes inside each task
- keeps one model loaded for the lifetime of that task
- writes the manifest and sbatch script under `jobs/generated/evaluate/<run>/`
- writes bootstrap and per-task logs under `logs/slurm/evaluate/<run>/`

Limit concurrent jobs to the number of GPUs you want SLURM to use:

```bash
uv run python main.py submit-evaluate-cluster \
  --run-name eval_cluster \
  --generator-run-name gen_openai \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v2 \
  --gpu-count 4
```

If `--gpu-count` is omitted, the array is submitted without a `%N` cap and SLURM schedules tasks as resources become available.

Default fanout is `3 local eval models × 3 datasets = 9 jobs`.

## Low-Level Sharding

Direct `evaluate` still supports:

- `--shard-count`
- `--shard-index`
- `--shard-strategy`

Use those only if a single `model × dataset` job is too large for your time limit. They are no longer the primary cluster orchestration path.

## Notes

- `mmlu_pro` preprocessing still keeps the existing exact-match filter against raw `mmlu`.
- There is no merge stage anymore; analysis reads Inspect logs directly.
- Cluster submit is for local models only. Hosted/API models should use direct `evaluate` or `evaluate-all`.
