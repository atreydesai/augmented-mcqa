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

Use the high-level cluster submit commands for local GPU-backed models:

```bash
uv run python main.py submit-generate-cluster \
  --run-name gen_cluster \
  --processed-dataset datasets/processed/unified_processed_v2
```

```bash
uv run python main.py submit-evaluate-cluster \
  --run-name eval_cluster \
  --generator-run-name gen_cluster \
  --generator-model Qwen/Qwen3-4B-Instruct-2507 \
  --processed-dataset datasets/processed/unified_processed_v2 \
  --gpu-count 4
```

Notes:

- These commands only accept models that resolve to `vllm/...`.
- Each task is one `model × dataset` pair on one GPU.
- If `--gpu-count` is omitted, the array is submitted without a concurrency cap.
- Hosted/API models are out of scope for the cluster submit commands and should be run with direct `generate` / `evaluate`.
