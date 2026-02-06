"""
DeepSeek model client for Augmented MCQA.

Supports deepseek-chat and deepseek-reasoner models.
"""

from typing import Optional

from .base import ModelClient, GenerationResult
from config import get_api_key


class DeepSeekClient(ModelClient):
    """
    DeepSeek model client using OpenAI-compatible API.
    
    Supports:
    - deepseek-chat: Standard chat model
    - deepseek-reasoner: Reasoning model that returns chain-of-thought
      in reasoning_content field
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
            model_id: Model identifier
                - "deepseek-chat": Standard chat model
                - "deepseek-reasoner": Reasoning model with CoT
            api_key: API key (defaults to DEEPSEEK_API_KEY env var)
        """
        import openai
        
        self._model_id = model_id
        self._api_key = api_key or get_api_key("deepseek")
        self._client = openai.OpenAI(
            api_key=self._api_key,
            base_url=self.DEEPSEEK_BASE_URL,
        )
        
        self._is_reasoner = "reasoner" in model_id.lower()
    
    @property
    def name(self) -> str:
        return f"DeepSeek ({self._model_id})"
    
    @property
    def model_id(self) -> str:
        return self._model_id
    
    @property
    def is_reasoner(self) -> bool:
        """Check if this is the reasoner model that returns CoT."""
        return self._is_reasoner
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate using DeepSeek API.
        
        For deepseek-reasoner, the response will contain reasoning_content
        in the raw_response.
        """
        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        
        choice = response.choices[0]
        
        # Extract text content
        text = choice.message.content or ""
        
        # For reasoner model, check for reasoning_content
        reasoning_content = None
        if self._is_reasoner and hasattr(choice.message, "reasoning_content"):
            reasoning_content = choice.message.reasoning_content
        
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            # Add reasoning tokens if available
            if hasattr(response.usage, "reasoning_tokens"):
                usage["reasoning_tokens"] = response.usage.reasoning_tokens
        
        result = GenerationResult(
            text=text,
            model=response.model,
            finish_reason=choice.finish_reason,
            usage=usage,
            raw_response=response,
        )
        
        # Store reasoning content for access
        if reasoning_content:
            result.reasoning_content = reasoning_content
        
        return result
