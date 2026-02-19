# Model Registry and Aliases

## Overview

Model resolution now uses:

- Provider registry: `models/registry.py`
- Declarative aliases: `config/model_aliases.toml`

`models.get_client()` remains the stable public API.

## Provider Registry Contract

Providers currently supported:

- `openai`
- `anthropic`
- `gemini`
- `deepseek`
- `local`

Each provider maps to a client class implementing `ModelClient`.

## Alias Schema

Aliases are defined under `[aliases]`.

Minimal alias:

```toml
[aliases."my-alias"]
provider = "openai"
```

Alias with explicit model ID:

```toml
[aliases."my-eval-model"]
provider = "openai"
model_id = "gpt-4.1"
```

Alias with defaults:

```toml
[aliases."my-reasoning-model"]
provider = "openai"
model_id = "gpt-5.2-2025-12-11"

[aliases."my-reasoning-model".defaults]
reasoning_effort = "high"
```

Notes:

- Explicit kwargs passed to `get_client()` override alias defaults.
- Alias provider must exist in provider registry.

## Resolution Order

`get_client(model_name, **kwargs)` resolves in this order:

1. Exact alias from `config/model_aliases.toml`
2. Provider shortcut (`openai`, `anthropic`, etc.)
3. Heuristic inference from model name
   - `gpt*` -> `openai`
   - `claude*` -> `anthropic`
   - `gemini*` -> `gemini`
   - `deepseek*` -> `deepseek`
   - names containing `/` -> `local`

If unresolved, an explicit error is raised with known aliases/providers.

## Listing Available Models

Use public helper APIs:

```python
from models import list_available_models

print(list_available_models())
```

And from CLI:

```bash
uv run python scripts/generate_distractors.py --list-models
```
