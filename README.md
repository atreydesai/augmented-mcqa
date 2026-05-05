# Augmented MCQA

Pipeline for generating augmented multiple-choice QA datasets, evaluating models on the strict filtered benchmark, collecting results, and fitting IRT figures.

Run CLI help with:

```bash
uv run python main.py --help
```

## Current Evaluation Workflow

The normal evaluation path uses prebuilt strict filtered augmented datasets:

```text
datasets/augmented_filtered/strict/gemini
datasets/augmented_filtered/strict/gpt
datasets/augmented_filtered/strict/qwen
```

These stores already apply the shared strict filter across all 3 generators and all 5 settings. Evaluation no longer uses support manifests or rebuilt augmented caches on the normal path.

Strict filtered question counts:

| Dataset | Questions |
|---|---:|
| ARC Challenge | 995 |
| MMLU Pro | 954 |
| GPQA | 415 |

Each generator evaluates the same question stems for all settings and modes:

```text
settings: human_from_scratch, model_from_scratch, augment_human, augment_model, augment_ablation
modes: full_question, choices_only
datasets: arc_challenge, mmlu_pro, gpqa
```

## Run All Evaluations

Use `submit-evaluate-cluster` with a generator alias. This writes the per-slice jobs and a final collection job. The slice jobs use `--skip-collect-evaluated` internally; collection is deferred to the finalizer.

Evaluator models:

```text
vllm/allenai/Olmo-3-7B-Instruct,
vllm/meta-llama/Llama-3.1-8B-Instruct,
vllm/meta-llama/Llama-3.2-3B-Instruct,
vllm/nvidia/NVIDIA-Nemotron-Nano-9B-v2,
vllm/Qwen/Qwen3-4B-Instruct-2507
```

Qwen-generated data:

```bash
MODEL_CACHE_DIR=/fs/nexus-scratch/adesai10/hub HF_HOME=/fs/nexus-scratch/adesai10/hub \
  ./.venv/bin/python main.py submit-evaluate-cluster \
  --run-name eval_together \
  --generator qwen \
  --models vllm/allenai/Olmo-3-7B-Instruct,vllm/meta-llama/Llama-3.1-8B-Instruct,vllm/meta-llama/Llama-3.2-3B-Instruct,vllm/nvidia/NVIDIA-Nemotron-Nano-9B-v2,vllm/Qwen/Qwen3-4B-Instruct-2507 \
  --max-tokens 32768
```

Gemini-generated data:

```bash
MODEL_CACHE_DIR=/fs/nexus-scratch/adesai10/hub HF_HOME=/fs/nexus-scratch/adesai10/hub \
  ./.venv/bin/python main.py submit-evaluate-cluster \
  --run-name eval_gemini \
  --generator gemini \
  --models vllm/allenai/Olmo-3-7B-Instruct,vllm/meta-llama/Llama-3.1-8B-Instruct,vllm/meta-llama/Llama-3.2-3B-Instruct,vllm/nvidia/NVIDIA-Nemotron-Nano-9B-v2,vllm/Qwen/Qwen3-4B-Instruct-2507 \
  --max-tokens 32768
```

GPT-5.2-generated data:

```bash
MODEL_CACHE_DIR=/fs/nexus-scratch/adesai10/hub HF_HOME=/fs/nexus-scratch/adesai10/hub \
  ./.venv/bin/python main.py submit-evaluate-cluster \
  --run-name eval_gpt52 \
  --generator gpt \
  --models vllm/allenai/Olmo-3-7B-Instruct,vllm/meta-llama/Llama-3.1-8B-Instruct,vllm/meta-llama/Llama-3.2-3B-Instruct,vllm/nvidia/NVIDIA-Nemotron-Nano-9B-v2,vllm/Qwen/Qwen3-4B-Instruct-2507 \
  --max-tokens 32768
```

Generator aliases:

| Alias | Generation run | Generator model | Filtered dataset |
|---|---|---|---|
| `gemini` | `gemini_from_scratch_testing` | `google/gemini-3.1-pro-preview` | `datasets/augmented_filtered/strict/gemini` |
| `gpt` | `gen_gpt52_v2` | `openai/gpt-5.2-2025-12-11` | `datasets/augmented_filtered/strict/gpt` |
| `gpt52` | `gen_gpt52_v2` | `openai/gpt-5.2-2025-12-11` | `datasets/augmented_filtered/strict/gpt` |
| `qwen` | `together_from_scratch_testing` | `together/Qwen/Qwen3.5-397B-A17B` | `datasets/augmented_filtered/strict/qwen` |

## Useful Options

Use these only when needed:

```bash
--write-only          # write job files without submitting
--questions-per-job  # change slice size, default 200
--gpu-count          # GPUs per evaluation job
--settings           # restrict settings
--modes              # restrict modes
--dataset-types      # restrict datasets
--limit              # debug with fewer questions per dataset
```

The old eval flags for generation log roots, processed dataset fallback, augmented cache rebuilds, support roots, current/pending skipping, manual collection roots, and force resubmission are intentionally removed from the public evaluation commands.

## Local Debug Evaluation

For a tiny local/debug run, call `evaluate` directly. This uses the same generator alias and filtered augmented store:

```bash
./.venv/bin/python main.py evaluate \
  --run-name debug_qwen_eval \
  --generator qwen \
  --model vllm/Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-types arc_challenge \
  --settings human_from_scratch \
  --modes full_question \
  --limit 5
```

## Collected Data

Collected datasets are written under:

```text
datasets/collected/<generation_run>/<generation_model>/<evaluation_model>/
```

The scheduled finalizer materializes collected datasets from the `.eval` logs after the slice jobs finish. Analysis should read collected datasets, not raw `.eval` logs.

## Analysis

Analyze one collected evaluator folder:

```bash
uv run python main.py analyze \
  --collected-root datasets/collected/<generation_run>/<generation_model>/<evaluation_model>
```

Fit IRT models across all collected evaluation folders:

```bash
uv run python main.py analyze-irt \
  --collected-root datasets/collected \
  --output-dir results/augmented_mcqa_irt
```

Regenerate the IRT figures:

```bash
./.venv/bin/python analysis/figures/regenerate_plots.py
```

## Generation

The main evaluation workflow assumes generation is already materialized into `datasets/augmented_filtered/strict/*`.

Generation commands are still available for producing new augmented stores:

```bash
uv run python main.py generate --help
uv run python main.py submit-generate-cluster --help
uv run python main.py build-augmented-dataset --help
```

If the source augmented datasets change, rebuild the strict filtered stores before submitting evaluation.

## Validation

Check the CLI surface:

```bash
uv run python main.py evaluate --help
uv run python main.py submit-evaluate-cluster --help
```

Run tests:

```bash
uv run pytest -q
```
