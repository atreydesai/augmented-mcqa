"""Provider registry and heuristic model name resolution."""

from __future__ import annotations

from typing import Any

from .anthropic_client import AnthropicClient
from .gemini_client import GeminiClient
from .local_client import LocalClient
from .openai_client import OpenAIClient


PROVIDER_REGISTRY = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "gemini": GeminiClient,
    "local": LocalClient,
}


def list_model_aliases(include_providers: bool = True) -> list[str]:
    """List known provider names."""
    if include_providers:
        return sorted(PROVIDER_REGISTRY.keys())
    return []


def _resolve_by_heuristic(model_name: str) -> tuple[str, str | None, dict[str, Any]]:
    model_lower = model_name.lower()

    if model_name == "local":
        return "local", None, {}
    if "/" in model_name:
        return "local", model_name, {}

    if "gpt" in model_lower or "openai" in model_lower:
        return "openai", model_name, {}
    if "claude" in model_lower or "anthropic" in model_lower:
        return "anthropic", model_name, {}
    if "gemini" in model_lower:
        return "gemini", model_name, {}
    available = ", ".join(sorted(PROVIDER_REGISTRY.keys()))
    raise ValueError(f"Unknown model: {model_name}. Recognized providers: {available}")


def resolve_model(model_name: str) -> tuple[str, str | None, dict[str, Any]]:
    """Resolve model name to (provider, model_id, defaults)."""
    if model_name in PROVIDER_REGISTRY:
        return model_name, None, {}
    return _resolve_by_heuristic(model_name)


def get_provider_registry() -> dict[str, type]:
    """Return provider registry mapping provider -> client class."""
    return dict(PROVIDER_REGISTRY)


def create_client(model_name: str, **kwargs):
    """Create a model client from provider/heuristic resolution."""
    provider, model_id, defaults = resolve_model(model_name)
    client_class = PROVIDER_REGISTRY[provider]

    init_kwargs: dict[str, Any] = dict(defaults)
    init_kwargs.update(kwargs)

    if model_id is not None and "model_id" not in init_kwargs:
        init_kwargs["model_id"] = model_id

    return client_class(**init_kwargs)
