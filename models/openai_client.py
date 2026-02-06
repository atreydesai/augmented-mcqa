"""
OpenAI model client for Augmented MCQA.

Uses the Chat Completions API (still fully supported as of 2025).
"""

from typing import Optional

from .base import ModelClient, GenerationResult
from config import get_api_key


class OpenAIClient(ModelClient):
    """
    OpenAI model client using Chat Completions API.
    
    Supports GPT-4, GPT-4.1, and other chat models.
    """
    
    def __init__(
        self,
        model_id: str = "gpt-4.1-2025-04-14",
        api_key: Optional[str] = None,
    ):
        """
        Initialize OpenAI client.
        
        Args:
            model_id: Model identifier (e.g., "gpt-4.1-2025-04-14", "gpt-4")
            api_key: API key (defaults to OPENAI_API_KEY env var)
        """
        import openai
        
        self._model_id = model_id
        self._api_key = api_key or get_api_key("openai")
        self._client = openai.OpenAI(api_key=self._api_key)
    
    @property
    def name(self) -> str:
        return f"OpenAI ({self._model_id})"
    
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
        Generate using OpenAI Chat Completions API.
        
        Note: Uses max_completion_tokens (max_tokens deprecated Sept 2024).
        """
        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            **kwargs,
        )
        
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
