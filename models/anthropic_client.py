"""
Anthropic Claude model client for Augmented MCQA.

Uses the Messages API (current as of 2025).
"""

from typing import Optional

from .base import ModelClient, GenerationResult
from config import get_api_key


class AnthropicClient(ModelClient):
    """
    Anthropic Claude model client using Messages API.
    
    Supports Claude 3.5 Sonnet, Claude 3 Opus, Claude 4, etc.
    """
    
    def __init__(
        self,
        model_id: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
    ):
        """
        Initialize Anthropic client.
        
        Args:
            model_id: Model identifier (e.g., "claude-3-5-sonnet-20241022")
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
        """
        import anthropic
        
        self._model_id = model_id
        self._api_key = api_key or get_api_key("anthropic")
        self._client = anthropic.Anthropic(api_key=self._api_key)
    
    @property
    def name(self) -> str:
        return f"Anthropic ({self._model_id})"
    
    @property
    def model_id(self) -> str:
        return self._model_id
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate using Anthropic Messages API.
        """
        response = self._client.messages.create(
            model=self._model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        
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
