"""
DeepSeek model client for Augmented MCQA.

Uses OpenAI-compatible API endpoint.
"""

from typing import Optional

from .base import ModelClient, GenerationResult
from config import get_api_key


class DeepSeekClient(ModelClient):
    """
    DeepSeek model client using OpenAI-compatible API.
    
    DeepSeek provides an OpenAI-compatible endpoint.
    """
    
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    
    def __init__(
        self,
        model_id: str = "deepseek-chat",
        api_key: Optional[str] = None,
    ):
        """
        Initialize DeepSeek client.
        
        Args:
            model_id: Model identifier (e.g., "deepseek-chat", "deepseek-coder")
            api_key: API key (defaults to DEEPSEEK_API_KEY env var)
        """
        import openai
        
        self._model_id = model_id
        self._api_key = api_key or get_api_key("deepseek")
        self._client = openai.OpenAI(
            api_key=self._api_key,
            base_url=self.DEEPSEEK_BASE_URL,
        )
    
    @property
    def name(self) -> str:
        return f"DeepSeek ({self._model_id})"
    
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
        Generate using DeepSeek API (OpenAI-compatible).
        """
        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
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
