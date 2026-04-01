# Augmented MCQA

Minimal pipeline for:

- preparing processed datasets
- generating augmented multiple-choice datasets
- evaluating models against them
- exporting benchmarker items
- analyzing collected evaluation datasets

Run everything with `uv`:

```bash
uv run python main.py --help
```

## Workflow

```text
prepare-data -> generate -> evaluate -> analyze
                           \-> export
```

The important artifacts are:

- processed dataset: `datasets/processed/unified_processed_v3`
- augmented store: `datasets/augmented/<generation_run>/<generation_model>/`
- collected dataset: `datasets/collected/<generation_run>/<generation_model>/<evaluation_model>/`

## Quick Start

Build processed data:

```bash
bash scripts/prepare-data.sh
```

Generate a full augmented store with the main generator model:

```bash
uv run python main.py generate \
  --run-name gen_gpt52_v2 \
  --model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --dataset-types arc_challenge,mmlu_pro,gpqa \
  --generation-strategies model_from_scratch,augment_human,augment_model,augment_ablation \
  --materialize-cache
```

Submit the same generation run as a cluster bundle:

```bash
uv run python main.py submit-generate-cluster \
  --run-name gen_gpt52_v2 \
  --models gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --generation-strategies model_from_scratch,augment_human,augment_model,augment_ablation \
  --questions-per-job 200 \
  --write-only
```

Evaluate a local model against that run. This will also write the collected dataset automatically:

```bash
uv run python main.py evaluate \
  --run-name eval_qwen4b_on_gen_gpt52_v2 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --generator-run-name gen_gpt52_v2 \
  --generator-model gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v3
```

Submit the main evaluation sweep as a cluster bundle:

```bash
uv run python main.py submit-evaluate-cluster \
  --run-name eval_qwen4b_on_gen_gpt52_v2 \
  --generator-run-name gen_gpt52_v2 \
  --generator-model gpt-5.2-2025-12-11 \
  --models Qwen/Qwen3-4B-Instruct-2507,allenai/Olmo-3-7B-Instruct \
  --processed-dataset datasets/processed/unified_processed_v3 \
  --settings human_from_scratch,model_from_scratch,augment_human,augment_model,augment_ablation \
  --modes full_question,choices_only \
  --questions-per-job 200 \
  --write-only
```

Analyze from collected data:

```bash
uv run python main.py analyze \
  --collected-root datasets/collected/gen_gpt52_v2/openai_gpt-5.2-2025-12-11/vllm_Qwen_Qwen3-4B-Instruct-2507
```

Export benchmarker items from the augmented store:

```bash
uv run python main.py export \
  --input datasets/augmented/gen_gpt52_v2/openai_gpt-5.2-2025-12-11
```

## Scripts

Use the scripts for the common local workflows.

Prepare data:

```bash
bash scripts/prepare-data.sh
```

Local generation:

```bash
bash scripts/local-generate.sh
```

Local evaluation:

```bash
bash scripts/local-evaluate.sh
```

Local end-to-end smoke:

```bash
bash scripts/local-smoke.sh
```

Minimal API smoke:

```bash
bash scripts/api-smoke.sh
```

Cluster generation bundle:

```bash
bash scripts/submit-generate-cluster.sh
```

Cluster evaluation bundle:

```bash
bash scripts/submit-evaluate-cluster.sh
```

Useful script overrides:

- `DATASET_TYPES`
- `GENERATION_STRATEGIES`
- `SETTINGS`
- `MODES`
- `LIMIT`
- `QUESTIONS_PER_JOB`
- `GPU_COUNT`
- `SUBMIT=1`

## Sharing Results

For analysis, the shareable unit is one collected dataset folder:

```text
datasets/collected/<generation_run>/<generation_model>/<evaluation_model>/
```

That folder includes both:

- the evaluated rows needed for plots and summary tables
- the `collected_state.json` needed to understand missing, planned, failed, or partial slices

You do not need to send separate `.eval` logs just to rerun analysis.

## Validation

Run the test suite:

```bash
uv run pytest -q
```

Check the CLI surface:

```bash
uv run python main.py --help
```

Run the local smoke:

```bash
bash scripts/local-smoke.sh
```
