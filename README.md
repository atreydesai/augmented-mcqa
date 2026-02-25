# Augmented MCQA

Final5 pipeline for distractor generation and evaluation.

## Active Scope

- Datasets: `arc_challenge`, `mmlu_pro`, `gpqa`
- Generator models:
  - `gpt-5.2-2025-12-11`
  - `claude-opus-4-6`
  - `gemini-3.1-pro-preview`
- Local evaluation models:
  - `Qwen/Qwen3-4B-Instruct-2507`
  - `allenai/Olmo-3-7B-Instruct`
  - `meta-llama/Llama-3.1-8B-Instruct`
- Eval preset: `final5` only

Legacy experiment surfaces are archived in `archive/legacy_experiments/`.

## Final5 Settings

Generation/evaluation setting IDs:

1. `human_from_scratch` (A): `Q + A -> 3H` (passthrough)
2. `model_from_scratch` (B): `Q + A -> 3M`
3. `augment_human` (A + C): `Q + A + 3H -> +6M`
4. `augment_model` (B + D): `Q + A + 3M -> +6M` (stored as `augment_model_delta_6m` + combined `augment_model`)
5. `augment_ablation` (E): `Q + A -> 9M`

## Setup

```bash
cp .env.example .env
# fill API keys and path values in .env
uv sync
```

For local-model eval, install vLLM after sync:

```bash
uv pip install --no-build-isolation 'vllm==0.11.2' 'transformers<5' 'numpy<2.3'
```

Stage local model weights (script reads `.env` and targets scratch cache):

```bash
jobs/install_local_model_weights.sh --dry-run
# then run without --dry-run on remote GPU host
```

## End-to-End Commands

1. Download raw datasets:

```bash
uv run python scripts/download_datasets.py --all
```

If you prefer the module entrypoint, run:

```bash
uv run python -m data.downloader --dataset all
```

2. Process to Final5 unified schema (deterministic first 1000 per dataset):

```bash
uv run python scripts/process_all.py --output-path datasets/processed/unified_processed_v2
```

`mmlu_pro` keeps the existing exact-match filtering behavior against `mmlu` (unchanged).

3. Regenerate all generator datasets and write manifest:

```bash
uv run python scripts/regenerate_experiments.py \
  --processed-dataset datasets/processed/unified_processed_v2 \
  --output-root datasets/augmented
```

4. Build eval SLURM bundle (per-pair balanced work units):

```bash
uv run python scripts/build_eval_slurm_bundle.py \
  --manifest datasets/augmented/<manifest>.json \
  --target-rows-per-subsplit 500
```

5. Submit jobs:

```bash
bash jobs/generated/<timestamp>/submit_all.sh
```

6. Merge entry sub-shards into canonical results:

```bash
uv run python scripts/merge_eval_subshards.py \
  --bundle-manifest jobs/generated/<timestamp>/bundle_manifest.json \
  --strict
```

Canonical outputs are now:

- `.../summary.json` (lightweight metrics + metadata)
- `.../rows/` (HuggingFace Arrow row store)

7. Plot required Final5 pairwise comparisons:

```bash
uv run python scripts/plot_final5.py --results-root results --output-dir results/final5_plots
```

## Sanity Counts

- Ideal generation target rows (if every dataset has 1000 rows): `3 * 4 * 3 * 1000 = 36,000`
- Ideal eval target rows (if every dataset has 1000 rows): `3 * 5 * 3 * 2 * 1000 * 3 = 270,000`

Actual totals can be lower when a source dataset has fewer than `limit` rows after filtering/validation
(for example, GPQA may be `<1000` rows in some runs).

## Live API Smoke

Runs 1-2 rows per split through all 3 generator APIs:

```bash
uv run python scripts/live_api_smoke.py --limit 2 --dry-run
```

## Remote Eval Sharding Smoke

Runs a minimum-question end-to-end Final5 eval sharding test on one GPU host
with all eval models and all 5 settings.

```bash
scripts/run_final5_remote_smoke.sh
```

Useful overrides:

```bash
scripts/run_final5_remote_smoke.sh \
  --gen-full-ds datasets/augmented/final5_full_<timestamp>_<generator> \
  --target-rows 3 \
  --save-interval 2 \
  --max-tokens 32
```

## Documentation

- `docs/models.md`
- `docs/evaluation.md`
- `docs/sharding_and_recombination.md`
- `jobs/README_local_eval.md`
