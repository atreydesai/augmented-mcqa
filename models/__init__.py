"""Models module for Augmented MCQA."""

from .base import ModelClient, GenerationResult, ReasoningEffort
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .gemini_client import GeminiClient
from .local_client import LocalClient
from .registry import (
    create_client,
    resolve_model,
    list_model_aliases,
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
    "ReasoningEffort",
    # Clients
    "OpenAIClient",
    "AnthropicClient",
    "GeminiClient",
    "LocalClient",
    # Factory and helpers
    "get_client",
    "list_available_models",
    "resolve_model",
    "list_model_aliases",
    "get_provider_registry",
]
