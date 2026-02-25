# Local Eval on SLURM (Final5)

Primary flow is bundle-based generation of sbatch files, then manual submission.

## 1) Stage local model weights

```bash
jobs/install_local_model_weights.sh --dry-run
# run without --dry-run on remote GPU server
```

## 2) Build eval bundle

```bash
uv run python scripts/build_eval_slurm_bundle.py \
  --manifest datasets/augmented/<final5_regeneration_manifest>.json \
  --target-rows-per-subsplit 500
```

Default output:

```text
jobs/generated/<timestamp>/
```

Contains:

- per-group `.sbatch` files
- `submit_all.sh`
- `bundle_manifest.json`
- README

## 3) Submit jobs manually

```bash
bash jobs/generated/<timestamp>/submit_all.sh
```

## 4) Re-run failed array IDs

```bash
sbatch --array=1,4,7 jobs/generated/<timestamp>/<specific>.sbatch
```

## 5) Recombine sub-shards

```bash
uv run python scripts/merge_eval_subshards.py \
  --bundle-manifest jobs/generated/<timestamp>/bundle_manifest.json \
  --strict
```

## Optional legacy wrappers

`jobs/run_local_eval.sh` and `jobs/local_model_eval.sbatch` remain available for direct array submission, but the bundle path above is the recommended Final5 workflow.
