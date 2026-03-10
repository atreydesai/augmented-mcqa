# Models

Model execution is Inspect-native.

## Resolution Rules

Short aliases are resolved in [`utils/modeling.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/utils/modeling.py):

- `gpt-5.2-2025-12-11` -> `openai/gpt-5.2-2025-12-11`
- `claude-opus-4-6` -> `anthropic/claude-opus-4-6`
- `gemini-3.1-pro-preview` -> `google/gemini-3.1-pro-preview`
- `Qwen/Qwen3-4B-Instruct-2507` -> `vllm/Qwen/Qwen3-4B-Instruct-2507`
- `allenai/Olmo-3-7B-Instruct` -> `vllm/allenai/Olmo-3-7B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct` -> `vllm/meta-llama/Llama-3.1-8B-Instruct`

You can also pass a provider-qualified Inspect model id directly.

## Hosted API Models

Examples:

```bash
uv run python main.py generate \
  --model openai/gpt-5.2-2025-12-11 \
  --run-name gen_api \
  --processed-dataset datasets/processed/unified_processed_v2
```

```bash
uv run python main.py generate \
  --model anthropic/claude-opus-4-6 \
  --run-name gen_claude \
  --processed-dataset datasets/processed/unified_processed_v2
```

## Local Models

Use a vLLM-qualified Inspect id directly, or use the short alias:

```bash
uv run python main.py evaluate \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --run-name eval_vllm \
  --generator-run-name gen_api \
  --generator-model openai/gpt-5.2-2025-12-11 \
  --processed-dataset datasets/processed/unified_processed_v2
```

For an OpenAI-compatible local server:

```bash
uv run python main.py generate \
  --model my-local-model \
  --backend openai \
  --model-base-url http://localhost:8000/v1 \
  --run-name gen_local \
  --processed-dataset datasets/processed/unified_processed_v2
```

## Runtime Flags

Shared generation/evaluation runtime flags:

- `--backend`
- `--model-base-url`
- `--max-connections`
- `--max-tokens`
- `--temperature`
- `--reasoning-effort`
- `--retry-on-error`
- `--stop-seqs`

These flow straight through to Inspect's model execution layer.

## Prompting Contract

Generation and evaluation both use plain text now.

Generation:
- No JSON mode
- No structured outputs
- Strict labeled-line parsing for distractors

Evaluation:
- Model is prompted to return a single answer letter
- Answer extraction is tolerant of short surrounding prose, but the intended output is still one letter

## Cluster Usage

Sharded local-cluster runs are still supported. Use:

- [`jobs/run_generate_shard.sh`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/jobs/run_generate_shard.sh)
- [`jobs/run_evaluate_shard.sh`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/jobs/run_evaluate_shard.sh)
- [`scripts/05_build_eval_slurm_bundle.py`](/Users/ndesai-air/Documents/GitHub/augmented-mcqa/scripts/05_build_eval_slurm_bundle.py)

The SLURM layer only launches `main.py` with deterministic shard arguments. There is no custom model client layer underneath it anymore.
