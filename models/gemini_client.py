"""
Google Gemini model client for Augmented MCQA.

Supports Gemini 2.5 and Gemini 3 models with thinking level control.
"""

from typing import Optional, Literal

from .base import ModelClient, GenerationResult
from config import get_api_key


# Valid thinking levels for Gemini 3 models
ThinkingLevel = Literal["off", "low", "medium", "high"]


class GeminiClient(ModelClient):
    """
    Google Gemini model client using google-genai SDK.
    
    Supports:
    - Gemini 2.5 Flash Lite: Fast, lightweight
    - Gemini 3 Pro/Flash Preview: With thinking_level for reasoning control
    """
    
    # Models that support thinking_level parameter
    THINKING_MODELS = {"gemini-3"}
    
    def __init__(
        self,
        model_id: str = "gemini-3-flash-preview",
        api_key: Optional[str] = None,
        thinking_level: ThinkingLevel = "off",
    ):
        """
        Initialize Gemini client.
        
        Args:
            model_id: Model identifier
                - "gemini-2.5-flash-lite": Fast lightweight model
                - "gemini-3-pro-preview": Gemini 3 Pro with thinking
                - "gemini-3-flash-preview": Gemini 3 Flash with thinking
            api_key: API key (defaults to GOOGLE_API_KEY env var)
            thinking_level: For Gemini 3 models. Controls thinking depth:
                - "off": No extended thinking
                - "low": Light reasoning
                - "medium": Moderate reasoning
                - "high": Deep reasoning (default for Gemini 3)
        """
        from google import genai
        
        self._model_id = model_id
        self._api_key = api_key or get_api_key("google")
        self._client = genai.Client(api_key=self._api_key)
        self._thinking_level = thinking_level
        
        # Check if model supports thinking
        self._supports_thinking = any(
            self._model_id.startswith(prefix) for prefix in self.THINKING_MODELS
        )
    
    @property
    def name(self) -> str:
        return f"Gemini ({self._model_id})"
    
    @property
    def model_id(self) -> str:
        return self._model_id
    
    @property
    def supports_thinking(self) -> bool:
        """Check if this model supports thinking_level parameter."""
        return self._supports_thinking
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        thinking_level: Optional[ThinkingLevel] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate using Google Gemini API.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            thinking_level: Override thinking level for this call (Gemini 3 only)
            **kwargs: Additional API parameters
        """
        from google.genai import types
        
        # Build generation config
        config_params = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        # Add thinking level for Gemini 3 models
        level = thinking_level or self._thinking_level
        if self._supports_thinking and level != "off":
            config_params["thinking_level"] = level
        
        config = types.GenerateContentConfig(**config_params)
        
        response = self._client.models.generate_content(
            model=self._model_id,
            contents=prompt,
            config=config,
        )
        
        # Extract text
        text = response.text if response.text else ""
        
        # Extract usage if available
        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }
        
        # Get finish reason
        finish_reason = None
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, "finish_reason"):
                finish_reason = str(candidate.finish_reason)
        
        return GenerationResult(
            text=text,
            model=self._model_id,
            finish_reason=finish_reason,
            usage=usage,
            raw_response=response,
        )
