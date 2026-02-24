"""Models module for Augmented MCQA."""

from .base import ModelClient, GenerationResult
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .gemini_client import GeminiClient
from .local_client import LocalClient
from .registry import (
    AliasSpec,
    create_client,
    resolve_model,
    list_model_aliases,
    load_model_aliases,
    clear_model_alias_cache,
    get_provider_registry,
)


def get_client(model_name: str, **kwargs) -> ModelClient:
    """Factory function to get a model client.

    Args:
        model_name: Alias, provider name, or direct model identifier.
        **kwargs: Additional constructor arguments. Explicit kwargs override
            alias defaults from config/model_aliases.toml.

    Returns:
        Configured ModelClient instance.
    """
    return create_client(model_name, **kwargs)


def list_available_models(include_providers: bool = True) -> list[str]:
    """List declared model aliases (and providers by default)."""
    return list_model_aliases(include_providers=include_providers)


__all__ = [
    # Base classes
    "ModelClient",
    "GenerationResult",
    # Clients
    "OpenAIClient",
    "AnthropicClient",
    "GeminiClient",
    "LocalClient",
    # Alias/registry types
    "AliasSpec",
    # Factory and helpers
    "get_client",
    "list_available_models",
    "resolve_model",
    "list_model_aliases",
    "load_model_aliases",
    "clear_model_alias_cache",
    "get_provider_registry",
]
