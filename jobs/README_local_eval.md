# Local Eval on SLURM

The only supported SLURM path is the Inspect-first shard launcher flow.

## 1. Stage model weights

```bash
jobs/install_local_model_weights.sh --dry-run
```

## 2. Build per-model sbatch wrappers

```bash
uv run python scripts/05_build_eval_slurm_bundle.py \
  --run-name eval_cluster \
  --generator-run-name gen_cluster \
  --generator-model gpt-5.2-2025-12-11 \
  --output-dir jobs/generated/eval_cluster \
  --shard-count 8
```

## 3. Submit

```bash
bash jobs/generated/eval_cluster/submit_all.sh
```

## 4. Re-run failed array ids

```bash
sbatch --array=1,4,7 jobs/generated/eval_cluster/<one-file>.sbatch
```

## 5. Direct launch helpers

- [`jobs/run_generate_shard.sh`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/jobs/run_generate_shard.sh)
- [`jobs/run_evaluate_shard.sh`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/jobs/run_evaluate_shard.sh)
- [`jobs/generate_array.sbatch`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/jobs/generate_array.sbatch)
- [`jobs/evaluate_array.sbatch`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/jobs/evaluate_array.sbatch)

There are no remaining legacy array wrappers in this repo.
