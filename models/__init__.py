"""Models module for Augmented MCQA."""

from .base import ModelClient, GenerationResult
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .gemini_client import GeminiClient
from .deepseek_client import DeepSeekClient
from .local_client import LocalClient


# Client registry for factory function
_CLIENT_REGISTRY = {
    # OpenAI models
    "gpt-4": OpenAIClient,
    "gpt-4.1": OpenAIClient,
    "gpt-4.1-2025-04-14": OpenAIClient,
    "openai": OpenAIClient,
    
    # Anthropic models
    "claude-3.5-sonnet": AnthropicClient,
    "claude-3-5-sonnet-20241022": AnthropicClient,
    "claude-3-opus": AnthropicClient,
    "claude-3-opus-20240229": AnthropicClient,
    "anthropic": AnthropicClient,
    
    # Google Gemini models
    "gemini-1.5-pro": GeminiClient,
    "gemini-1.5-flash": GeminiClient,
    "gemini-2.0-flash": GeminiClient,
    "gemini": GeminiClient,
    
    # DeepSeek models
    "deepseek-chat": DeepSeekClient,
    "deepseek": DeepSeekClient,
    
    # Local models
    "local": LocalClient,
    "qwen3-8b": LocalClient,
    "Qwen/Qwen2.5-7B-Instruct": LocalClient,
}


def get_client(model_name: str, **kwargs) -> ModelClient:
    """
    Factory function to get a model client.
    
    Args:
        model_name: Model name or provider identifier
        **kwargs: Additional arguments passed to client constructor
        
    Returns:
        Configured ModelClient instance
        
    Examples:
        >>> client = get_client("gpt-4.1")
        >>> client = get_client("claude-3.5-sonnet")
        >>> client = get_client("gemini-1.5-flash")
        >>> client = get_client("local", model_id="Qwen/Qwen2.5-7B-Instruct")
    """
    # Look up in registry
    if model_name in _CLIENT_REGISTRY:
        client_class = _CLIENT_REGISTRY[model_name]
        
        # For specific model names, pass as model_id
        if "model_id" not in kwargs and model_name not in ["openai", "anthropic", "gemini", "deepseek", "local"]:
            kwargs["model_id"] = model_name
        
        return client_class(**kwargs)
    
    # Try to infer provider from model name
    model_lower = model_name.lower()
    
    if "gpt" in model_lower or "openai" in model_lower:
        return OpenAIClient(model_id=model_name, **kwargs)
    
    if "claude" in model_lower or "anthropic" in model_lower:
        return AnthropicClient(model_id=model_name, **kwargs)
    
    if "gemini" in model_lower:
        return GeminiClient(model_id=model_name, **kwargs)
    
    if "deepseek" in model_lower:
        return DeepSeekClient(model_id=model_name, **kwargs)
    
    # Default to local for HuggingFace model IDs (contain /)
    if "/" in model_name:
        return LocalClient(model_id=model_name, **kwargs)
    
    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Available: {list(_CLIENT_REGISTRY.keys())}"
    )


__all__ = [
    # Base classes
    "ModelClient",
    "GenerationResult",
    # Clients
    "OpenAIClient",
    "AnthropicClient",
    "GeminiClient",
    "DeepSeekClient",
    "LocalClient",
    # Factory
    "get_client",
]
