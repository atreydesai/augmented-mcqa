"""
Anthropic Claude model client for Augmented MCQA.

Supports Claude 4.x models with extended thinking capability.
"""

from typing import Optional, Literal

from .base import ModelClient, GenerationResult
from config import get_api_key


# Valid thinking budget presets
ThinkingLevel = Literal["off", "low", "medium", "high"]


class AnthropicClient(ModelClient):
    """
    Anthropic Claude model client using Messages API.
    
    Supports:
    - Claude Opus 4, Sonnet 4, Haiku 4.5 with extended thinking
    - Extended thinking allows deeper reasoning with configurable budget
    """
    
    # Models that support extended thinking
    THINKING_MODELS = {
        "claude-opus-4", "claude-sonnet-4", "claude-haiku-4",
        "claude-opus-4-6", "claude-sonnet-4-5", "claude-haiku-4-5"
    }
    
    # Approximate thinking budgets (in tokens)
    THINKING_BUDGETS = {
        "off": 0,
        "low": 1024,
        "medium": 4096,
        "high": 16384,
    }
    
    def __init__(
        self,
        model_id: str = "claude-sonnet-4-5-20250929",
        api_key: Optional[str] = None,
        thinking_level: ThinkingLevel = "off",
    ):
        """
        Initialize Anthropic client.
        
        Args:
            model_id: Model identifier
                - "claude-opus-4-6": Claude Opus 4.6
                - "claude-sonnet-4-5-20250929": Claude Sonnet 4.5
                - "claude-haiku-4-5-20251001": Claude Haiku 4.5
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
            thinking_level: Extended thinking budget level:
                - "off": No extended thinking
                - "low": Light reasoning (~1K tokens)
                - "medium": Moderate reasoning (~4K tokens)
                - "high": Deep reasoning (~16K tokens)
        """
        import anthropic
        
        self._model_id = model_id
        self._api_key = api_key or get_api_key("anthropic")
        self._client = anthropic.Anthropic(api_key=self._api_key)
        self._thinking_level = thinking_level
        
        # Check if model supports thinking
        self._supports_thinking = any(
            prefix in self._model_id for prefix in self.THINKING_MODELS
        )
    
    @property
    def name(self) -> str:
        return f"Anthropic ({self._model_id})"
    
    @property
    def model_id(self) -> str:
        return self._model_id
    
    @property
    def supports_thinking(self) -> bool:
        """Check if this model supports extended thinking."""
        return self._supports_thinking
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        thinking_level: Optional[ThinkingLevel] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate using Anthropic Messages API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum output tokens
            thinking_level: Override thinking level for this call
            **kwargs: Additional API parameters
        """
        # Build request parameters
        params = {
            "model": self._model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        # Add extended thinking for supported models
        level = thinking_level or self._thinking_level
        if self._supports_thinking and level != "off":
            budget = self.THINKING_BUDGETS.get(level, 4096)
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget,
            }
        
        params.update(kwargs)
        
        response = self._client.messages.create(**params)
        
        # Extract text from content blocks
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
        
        usage = None
        if response.usage:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        
        return GenerationResult(
            text=text,
            model=response.model,
            finish_reason=response.stop_reason,
            usage=usage,
            raw_response=response,
        )
