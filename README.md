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

## Remote Full Eval Runbook (clip + A6000)

Assumes:

- repo path: `/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa`
- venv path: `/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/.venv/bin/activate`
- model cache: `/fs/nexus-scratch/adesai10/hub`
- one generated Final5 dataset as eval input

1. Environment setup:

```bash
cd /fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa
source /fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/.venv/bin/activate
set -eo pipefail

export HF_HOME=/fs/nexus-scratch/adesai10/hub
export MODEL_CACHE_DIR=/fs/nexus-scratch/adesai10/hub
```

2. Create regeneration manifest and paths:

```bash
GEN_FULL_DS="/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/datasets/augmented/final5_full_20260225_004316_gpt-5.2-2025-12-11"
GENERATOR_LABEL="$(basename "$GEN_FULL_DS" | sed -E 's/^final5_full_[0-9]{8}_[0-9]{6}_//')"
TS="$(date +%Y%m%d_%H%M%S)"

REGEN_MANIFEST="datasets/augmented/final5_regeneration_manifest_${GENERATOR_LABEL}_${TS}.json"
BUNDLE="jobs/generated/final5_full_${GENERATOR_LABEL}_${TS}"
OUT="results/final5_full_${GENERATOR_LABEL}_${TS}"

cat > "$REGEN_MANIFEST" <<JSON
{
  "manifest_version": 1,
  "schema_version": "final5_v1",
  "generators": [
    {"model": "${GENERATOR_LABEL}", "dataset_path": "${GEN_FULL_DS}", "returncode": 0}
  ]
}
JSON
```

3. Build bundle:

```bash
python scripts/build_eval_slurm_bundle.py \
  --manifest "$REGEN_MANIFEST" \
  --output-dir "$BUNDLE" \
  --output-base "$OUT" \
  --target-rows-per-subsplit 500 \
  --save-interval 50 \
  --max-tokens 100
```

4. Preflight checks:

```bash
jq '.total_sbatch_files, .total_work_units, .dataset_part_counts' "$BUNDLE/bundle_manifest.json"
for f in "$BUNDLE"/final5_pair__*.sbatch; do sbatch --test-only "$f"; done
```

5. Submit full evaluation:

```bash
bash "$BUNDLE/submit_all.sh" | tee "$BUNDLE/submit.log"
awk '/Submitted batch job/{print $4}' "$BUNDLE/submit.log" > "$BUNDLE/job_ids.txt"
cat "$BUNDLE/job_ids.txt"
```

6. Monitor:

```bash
JOB_CSV="$(paste -sd, "$BUNDLE/job_ids.txt")"
while squeue -h -j "$JOB_CSV" | grep -q .; do
  date
  squeue -j "$JOB_CSV" -o "%.18i %.9P %.30j %.8T %.10M %.20R"
  sleep 60
done

while read -r j; do
  sacct -j "$j" --format=JobID,State,ExitCode,Elapsed,NodeList -n -P
done < "$BUNDLE/job_ids.txt"
```

7. Merge shards:

```bash
python scripts/merge_eval_subshards.py \
  --bundle-manifest "$BUNDLE/bundle_manifest.json" \
  --strict | tee "$BUNDLE/merge.log"
```

8. Validate summary counts (for 1 generator x 3 eval models, with part counts `2/2/1`):

```bash
CANONICAL=$(find "$OUT" -path '*/summary.json' | grep -E '/(human_from_scratch|model_from_scratch|augment_human|augment_model|augment_ablation)/summary.json$' | wc -l)
PARTIAL=$(find "$OUT" -path '*/_partials/entry_shard_*_of_*/summary.json' | wc -l)
echo "canonical=$CANONICAL expected=90"
echo "partial=$PARTIAL expected=150"
```

If your `dataset_part_counts` differ, recompute expected partial count from the bundle manifest.

9. Generate visuals:

```bash
python scripts/plot_final5.py \
  --results-root "$OUT" \
  --output-dir "$OUT/final5_plots"

find "$OUT/final5_plots" -maxdepth 2 -type f | sort
```

## Documentation

- `docs/models.md`
- `docs/evaluation.md`
- `docs/sharding_and_recombination.md`
- `jobs/README_local_eval.md`
