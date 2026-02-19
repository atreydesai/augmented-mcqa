# Augmented MCQA

Tools for generating synthetic distractors and running MCQA evaluation experiments across dataset/distractor settings.

## What Changed

Evaluation orchestration is now centered on a deterministic matrix runner:

- Primary CLI: `scripts/eval_matrix.py`
- No in-process evaluation parallelization
- Deterministic sharding for SLURM arrays (`--num-shards`, `--shard-index`)
- Declarative model aliases in `config/model_aliases.toml`

## Environment

Canonical workflow uses `uv`:

```bash
uv sync
```

Then run commands with `uv run`.

## End-to-End Workflow

### 1. Download/process datasets

```bash
uv run python scripts/download_datasets.py
uv run python scripts/process_all.py --dataset all
```

If a dataset is missing locally, you can pull from Hugging Face (for this project: `atreydesai/*` as needed).

### 2. Generate distractors

```bash
uv run python scripts/generate_distractors.py \
  --input datasets/processed/unified_processed \
  --model gpt-4.1 \
  --output datasets/finished_sets/gpt-4.1
```

### 3. Plan evaluation matrix

```bash
uv run python scripts/eval_matrix.py plan \
  --preset core16 \
  --model gpt-4.1 \
  --dataset-path datasets/augmented/unified_processed_gpt-4.1_20260213_033916 \
  --print-configs
```

### 4. Run evaluation matrix (sequential within job)

```bash
uv run python scripts/eval_matrix.py run \
  --preset core16 \
  --model gpt-4.1 \
  --dataset-path datasets/augmented/unified_processed_gpt-4.1_20260213_033916 \
  --skip-existing
```

### 5. Run sharded matrix for SLURM

Single shard (manual):

```bash
uv run python scripts/eval_matrix.py run \
  --preset core16 \
  --model gpt-4.1 \
  --dataset-path datasets/augmented/unified_processed_gpt-4.1_20260213_033916 \
  --num-shards 8 \
  --shard-index 3 \
  --skip-existing
```

Array helper:

```bash
jobs/submit_eval_array.sh \
  gpt-4.1 \
  datasets/augmented/unified_processed_gpt-4.1_20260213_033916 \
  8 \
  --dataset-types mmlu_pro,supergpqa \
  --distractor-source scratch,dhuman
```

### 6. Analyze

```bash
uv run python scripts/analyze_all.py --dir results
```

## Matrix Presets

- `core16`: historical label for the core matrix (15 unique configs after overlap dedupe)
- `branching21`: full 1H..3H crossed with 0M..6M (21 configs per dataset/source pair)
  - Uses cumulative branching sampling: per question, distractor selections are carried forward as `H`/`M` increase (prefix behavior), then options are re-shuffled per config.

Difficulty is controlled by `--dataset-types` (`arc_easy`, `arc_challenge`, `mmlu_pro`, `supergpqa`), not a separate intrinsic-difficulty pipeline.

Results layout remains unchanged:

```text
results/<model>_<dataset_type>_<distractor_source>/<nHnM>/results.json
```

## Adding/Extending Models

Model aliases are defined in `config/model_aliases.toml`.

1. Add an alias entry:

```toml
[aliases."my-new-model"]
provider = "openai"
model_id = "gpt-4.1"
```

2. Use it anywhere a model name is accepted:

```bash
uv run python scripts/eval_matrix.py run ... --model my-new-model
```

Supported providers in registry:

- `openai`
- `anthropic`
- `gemini`
- `deepseek`
- `local`

## Additional Docs

- `docs/evaluation.md`
- `docs/models.md`
