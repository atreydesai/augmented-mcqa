# Final5 Eval SLURM Bundle (Per-Pair Work Units)

Generated from regeneration manifest.

## Shard model

- Pair groups = generator model x eval model = 3
- Modes per pair = 2 (`full_question`, `choices_only`)
- Per-dataset part counts are dynamic from row counts with target rows/subsplit = 500
- One sbatch file per pair, each running a per-pair manifest with all work units

## Files

- Bundle manifest: `bundle_manifest.json`
- Submit script: `submit_all.sh`
- SBATCH scripts: one per `(generator, eval_model)`
- Work unit maps: one JSON per sbatch file
- Per-pair run manifests: one JSON per sbatch file

## Submit everything

```bash
bash submit_all.sh
```

Dry run:

```bash
bash submit_all.sh --dry-run
```

## Re-run failed pair jobs

```bash
sbatch <one-of-the-generated>.sbatch
```

## Merge entry sub-shards to canonical results

```bash
python scripts/merge_eval_subshards.py --bundle-manifest jobs/generated/final5_full_gpt-5.2-2025-12-11_20260225_210132/bundle_manifest.json --strict
```

## Quick stats

- Pair groups: 3
- Total work units across all pair jobs: 30
