# Local Eval on SLURM

The supported SLURM path is the dataset-aware submit flow in [`main.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/main.py).

## 1. Stage model weights

```bash
jobs/install_local_model_weights.sh --dry-run
```

## 2. Submit generation

```bash
uv run python main.py submit-generate-cluster \
  --run-name gen_cluster \
  --processed-dataset datasets/processed/unified_processed_v2
```

## 3. Submit evaluation

```bash
uv run python main.py submit-evaluate-cluster \
  --run-name eval_cluster \
  --generator-run-name gen_cluster \
  --generator-model Qwen/Qwen3-4B-Instruct-2507 \
  --processed-dataset datasets/processed/unified_processed_v2 \
  --gpu-count 4
```

## 4. What gets written

- bundle files under `jobs/generated/<stage>/<run>/`
- bootstrap logs under `logs/slurm/<stage>/<run>/_bootstrap/`
- per-task logs under `logs/slurm/<stage>/<run>/`

## 5. Notes

- Cluster submit commands only support local `vllm/...` models.
- Each job is one `model × dataset` pair on one GPU.
- If `--gpu-count` is omitted, the array is submitted with no concurrency cap.
- Settings and modes stay grouped inside each evaluation job, so there is no repeated cold start between configs inside that task.
