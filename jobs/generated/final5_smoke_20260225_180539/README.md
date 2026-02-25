# Final5 Eval SLURM Bundle (Per-Pair Work Units)

Generated from regeneration manifest.

## Shard model

- Pair groups = generator model x eval model = 3
- Modes per pair = 2 (`full_question`, `choices_only`)
- Per-dataset part counts are dynamic from row counts with target rows/subsplit = 3
- One sbatch file per pair, with array indexes mapped via `work_units.json`

## Files

- Bundle manifest: `bundle_manifest.json`
- Submit script: `submit_all.sh`
- SBATCH scripts: one per `(generator, eval_model)`
- Work unit maps: one JSON per sbatch file

## Submit everything

```bash
bash submit_all.sh
```

Dry run:

```bash
bash submit_all.sh --dry-run
```

## Re-run failed array tasks

```bash
sbatch --array=1,4,7 <one-of-the-generated>.sbatch
```

## Merge entry sub-shards to canonical results

```bash
uv run python scripts/merge_eval_subshards.py --bundle-manifest /fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/jobs/generated/final5_smoke_20260225_180539/bundle_manifest.json --strict
```

## Quick stats

- Pair groups: 3
- Total array tasks across all pair sbatch files: 30
