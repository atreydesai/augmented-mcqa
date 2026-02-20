# Local-Model SLURM Orchestrator (Smoke + Main)

This companion note is for:

- `/Users/ndesai-air/Documents/GitHub/augmented-mcqa/jobs/clip_local_eval_master.sh`

The script orchestrates remote eval for local models across:

- Eval models: `Nanbeige/Nanbeige4.1-3B`, `Qwen/Qwen3-4B-Instruct-2507`, `allenai/Olmo-3-7B-Instruct`
- Generator datasets: `opus`, `gpt-4.1`, `gpt-5.2`
- Dataset types: `mmlu_pro`, `gpqa`, `arc_easy`, `arc_challenge`
- Prompt modes: full-question and `--choices-only` matrix variants
- Preset: `core16` (non-branching)

## Remote Setup

```bash
cd /fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa
source ~/.bashrc
export HF_TOKEN="<your_hf_token>"
uv sync --inexact
```

The orchestrator defaults to `uv sync --inexact` (so extra runtime deps like `vllm` are not removed each run).
If you need exact lockfile sync behavior, set `UV_SYNC_INEXACT=0`.

The orchestrator checks for `vllm` and installs a wheel-only build if missing (default: `vllm==0.11.2`).
Override with `VLLM_INSTALL_SPEC`, for example:

```bash
VLLM_INSTALL_SPEC='vllm==0.10.2' jobs/clip_local_eval_master.sh --phase smoke ...
```

## Per-Model srun Smoke Tests (Recommended Before Full Matrix)

Use single-model smoke tests first to confirm model load + one short generation works on your SLURM setup.

Single model:

```bash
jobs/srun_local_model_smoke.sh --model Nanbeige/Nanbeige4.1-3B --tokenizer-mode auto
jobs/srun_local_model_smoke.sh --model Qwen/Qwen3-4B-Instruct-2507
jobs/srun_local_model_smoke.sh --model allenai/Olmo-3-7B-Instruct
```

All three (sequentially):

```bash
jobs/srun_all_local_models_smoke.sh
```

The smoke runner uses:
- `/Users/ndesai-air/Documents/GitHub/augmented-mcqa/scripts/smoke_local_model.py`
- local snapshot resolution from cache by default (`--no-local-snapshot` to disable)

Logs:
- `/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/logs/slurm/local_model_smoke/<run_tag>/`

## Smoke Run (recommended first)

```bash
jobs/clip_local_eval_master.sh \
  --phase smoke \
  --run-tag local_eval_smoke_$(date -u +%Y%m%d_%H%M%S) \
  --max-concurrent-jobs 12 \
  --num-shards-smoke 3 \
  --save-interval 50 \
  --keep-checkpoints 2 \
  --max-tokens 2048
```

## Main Run (after smoke looks good)

Reuse the same run tag if you want smoke/main under one run root:

```bash
RUN_TAG="local_eval_prod_$(date -u +%Y%m%d_%H%M%S)"

jobs/clip_local_eval_master.sh \
  --phase smoke \
  --run-tag "$RUN_TAG" \
  --max-concurrent-jobs 12 \
  --num-shards-smoke 3

jobs/clip_local_eval_master.sh \
  --phase main \
  --run-tag "$RUN_TAG" \
  --max-concurrent-jobs 12 \
  --num-shards-main 12 \
  --save-interval 50 \
  --keep-checkpoints 2 \
  --max-tokens 2048
```

Or run both phases in one call:

```bash
jobs/clip_local_eval_master.sh --phase both --max-concurrent-jobs 12 --num-shards-main 12
```

## Logs and Artifacts

Given `RUN_TAG=<tag>`:

- SLURM logs: `/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/logs/slurm/local_eval/<tag>/`
- Results root: `/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/results/local_eval/<tag>/`
- Heartbeat: `/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/results/local_eval/<tag>/status/heartbeat.log`
- Dashboard JSON: `/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/results/local_eval/<tag>/status/dashboard.json`
- Dashboard TSV: `/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/results/local_eval/<tag>/status/dashboard.tsv`
- Final manifest (includes HF URLs for successful pushes): `/fs/nexus-projects/rlab/atrey/qgqa/augmented-mcqa/results/local_eval/<tag>/status/final_manifest.json`

## Notes

- One shard job = one SBATCH job = one A6000 GPU.
- Global active shard-job cap is controlled by `--max-concurrent-jobs`.
- Incomplete shard sets are retried once automatically.
- Plots generated are `RQ1`, `RQ2`, `RQ3` only (no branching plots).
- Use `--skip-push` to keep outputs local only.
