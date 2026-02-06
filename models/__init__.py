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
    "gpt-4.1": OpenAIClient,
    "gpt-4.1-2025-04-14": OpenAIClient,
    "gpt-5-mini": OpenAIClient,
    "gpt-5-mini-2025-08-07": OpenAIClient,
    "gpt-5.2": OpenAIClient,
    "gpt-5.2-2025-12-11": OpenAIClient,
    "openai": OpenAIClient,
    
    # Anthropic models
    "claude-opus-4-6": AnthropicClient,
    "claude-sonnet-4-5": AnthropicClient,
    "claude-sonnet-4-5-20250929": AnthropicClient,
    "claude-haiku-4-5": AnthropicClient,
    "claude-haiku-4-5-20251001": AnthropicClient,
    "anthropic": AnthropicClient,
    
    # Google Gemini models
    "gemini-3-pro-preview": GeminiClient,
    "gemini-3-flash-preview": GeminiClient,
    "gemini-2.5-flash-lite": GeminiClient,
    "gemini": GeminiClient,
    
    # DeepSeek models
    "deepseek-chat": DeepSeekClient,
    "deepseek-reasoner": DeepSeekClient,
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
            - reasoning_effort: For OpenAI GPT-5 models ("minimal"/"low"/"medium"/"high"/"none")
            - thinking_level: For Anthropic/Gemini ("off"/"low"/"medium"/"high")
        
    Returns:
        Configured ModelClient instance
        
    Examples:
        >>> client = get_client("gpt-5.2-2025-12-11")
        >>> client = get_client("gpt-5.2-2025-12-11", reasoning_effort="high")
        >>> client = get_client("claude-sonnet-4-5-20250929", thinking_level="medium")
        >>> client = get_client("gemini-3-flash-preview", thinking_level="high")
        >>> client = get_client("deepseek-reasoner")
        >>> client = get_client("Qwen/Qwen2.5-7B-Instruct")
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
