"""
OpenAI model client for Augmented MCQA.

Supports GPT-4.1, GPT-5, and GPT-5-mini with reasoning effort control.
"""

import os
from typing import Optional, Literal

from .base import ModelClient, GenerationResult
from config import get_api_key


# Valid reasoning effort levels for GPT-5 models
ReasoningEffort = Literal["minimal", "low", "medium", "high", "none"]


class OpenAIClient(ModelClient):
    """
    OpenAI model client using Chat Completions API.
    
    Supports:
    - GPT-4.1: Standard chat model (no reasoning effort)
    - GPT-5, GPT-5-mini: Support reasoning_effort parameter for controlling
      computational depth (minimal, low, medium, high, none)
    """
    
    # Models that support reasoning_effort parameter
    REASONING_MODELS = {"gpt-5", "gpt-5-mini", "gpt-5.1", "gpt-5.2","gpt-5.2-2025-12-11"}
    
    def __init__(
        self,
        model_id: str = "gpt-4.1-2025-04-14",
        api_key: Optional[str] = None,
        reasoning_effort: Optional[ReasoningEffort] = None,
    ):
        """
        Initialize OpenAI client.
        
        Args:
            model_id: Model identifier
                - "gpt-4.1-2025-04-14": Standard GPT-4.1
                - "gpt-5-mini-2025-08-07": GPT-5 Mini with reasoning
                - "gpt-5.2-2025-12-11": Latest GPT-5
            api_key: API key (defaults to OPENAI_API_KEY env var)
            reasoning_effort: For GPT-5 models only. Controls reasoning depth:
                - "minimal": Few/no reasoning tokens (fastest)
                - "low": Favors speed
                - "medium": Balanced (default for GPT-5)
                - "high": More complete reasoning
                - "none": Disable reasoning
        """
        import openai
        
        self._model_id = model_id
        self._api_key = api_key or get_api_key("openai")
        
        # Regional fix for some environments
        base_url = os.getenv("OPENAI_BASE_URL", "https://us.api.openai.com/v1")
        self._client = openai.OpenAI(api_key=self._api_key, base_url=base_url)
        self._reasoning_effort = reasoning_effort
        
        # Check if model supports reasoning
        self._supports_reasoning = any(
            self._model_id.startswith(prefix) for prefix in self.REASONING_MODELS
        )
    
    @property
    def name(self) -> str:
        return f"OpenAI ({self._model_id})"
    
    @property
    def model_id(self) -> str:
        return self._model_id
    
    @property
    def supports_reasoning(self) -> bool:
        """Check if this model supports reasoning_effort parameter."""
        return self._supports_reasoning
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        reasoning_effort: Optional[ReasoningEffort] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate using OpenAI Chat Completions API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum completion tokens
            reasoning_effort: Override reasoning effort for this call (GPT-5 only)
            **kwargs: Additional API parameters
        """
        # Build request parameters
        params = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,
        }
        
        # Add reasoning effort for GPT-5 models
        if self._supports_reasoning:
            effort = reasoning_effort or self._reasoning_effort
            if effort is not None:
                params["reasoning_effort"] = effort
        
        params.update(kwargs)
        
        response = self._client.chat.completions.create(**params)
        
        choice = response.choices[0]
        
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        return GenerationResult(
            text=choice.message.content or "",
            model=response.model,
            finish_reason=choice.finish_reason,
            usage=usage,
            raw_response=response,
        )
