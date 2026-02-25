# Models

## Active Models

### Generator models (API)

- `gpt-5.2-2025-12-11`
- `claude-opus-4-6`
- `gemini-3.1-pro-preview`

### Evaluation models (local/vLLM)

- `Qwen/Qwen3-4B-Instruct-2507`
- `allenai/Olmo-3-7B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`

## Registry and Aliases

Aliases are defined in `config/model_aliases.toml`.

Provider registry is in `models/registry.py`.

Supported active providers:

- `openai`
- `anthropic`
- `gemini`
- `local`

## Generation Reasoning Policy (Hardcoded)

- GPT-5.2 generation requests use `reasoning_effort=medium`
- Claude Opus 4.6 generation requests use `thinking={"type":"adaptive"}`
- Gemini 3.1 generation uses provider defaults (no thinking knobs)

Implementation points:

- `scripts/generate_distractors.py` (`_model_policy`)
- `data/augmentor.py` (`GenerationConfig` + provider-specific kwargs)
- `models/gemini_client.py` (OpenAI-compatible transport)

## Gemini 3 Transport

Gemini requests use OpenAI-compatible Chat Completions transport:

- base URL: `https://generativelanguage.googleapis.com/v1beta/openai/`
- model: `gemini-3.1-pro-preview`
- default generation policy: no explicit thinking knobs

Reference: [Gemini 3 docs](https://ai.google.dev/gemini-api/docs/gemini-3).

## Claude Adaptive Thinking

Anthropic thinking payload for Opus is explicit:

- `thinking={"type":"adaptive"}`

Reference: [Anthropic thinking docs](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking).

## Local Weight Staging

Use:

```bash
jobs/install_local_model_weights.sh --dry-run
```

Script behavior:

- reads `.env`
- resolves `MODEL_CACHE_DIR`/`HF_HOME`
- stages all 3 local eval models into scratch cache
- supports execute mode (default) and `--dry-run`
