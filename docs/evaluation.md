# Evaluation (Final5)

## Preset

`final5` is the only active preset.

Legacy presets (`core16`, `branching21`) are migration-blocked and return explicit errors.

## Settings

- `human_from_scratch`: evaluate `3H`
- `model_from_scratch`: evaluate `3M`
- `augment_human`: evaluate `3H + 6M` (A + C)
- `augment_model`: evaluate `3M + 6M` combined as `9M` (B + D)
- `augment_ablation`: evaluate direct `9M` (E)

## Modes

- `full_question`
- `choices_only`

## Dataset Processing Note

`mmlu_pro` preprocessing keeps the existing exact-match filtering against raw `mmlu` unchanged (`data/mmlu_pro_processor.py`).

## Result Paths

Canonical result path format:

```text
results/<generator>/<eval_model>/<mode>/<dataset>/<setting>/results.json
```

If entry sub-sharding is enabled (`entry_shards > 1`), partial files are written to:

```text
results/<generator>/<eval_model>/<mode>/<dataset>/<setting>/_partials/entry_shard_<i>_of_<n>/results.json
```

Then recombined with `scripts/merge_eval_subshards.py`.

## CLI

### Build bundle for SLURM

```bash
uv run python scripts/build_eval_slurm_bundle.py \
  --manifest datasets/augmented/<final5_regeneration_manifest>.json \
  --num-gpus 8 \
  --entry-shards 4
```

### Plan configs

```bash
uv run python scripts/eval_matrix.py plan \
  --preset final5 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-path datasets/augmented/gpt-5.2-2025-12-11 \
  --generator-dataset-label gpt-5.2-2025-12-11
```

### Run configs

```bash
uv run python scripts/eval_matrix.py run \
  --preset final5 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-path datasets/augmented/gpt-5.2-2025-12-11 \
  --generator-dataset-label gpt-5.2-2025-12-11 \
  --num-shards 8 \
  --shard-index 0 \
  --entry-shards 4 \
  --entry-shard-index 0 \
  --skip-existing
```

## Baselines and Deltas

Batch summaries include:

- `random_baseline`
- `delta_over_random`

Random baseline is `1 / (#choices)` for each setting.

## Strict Final5 Columns

Runner uses new Final5 columns only (no legacy fallback):

- `human_from_scratch`
- `model_from_scratch`
- `augment_human`
- `augment_model`
- `augment_ablation`
