"""
Google Gemini model client for Augmented MCQA.

Uses the google-genai SDK (current as of 2025).
"""

from typing import Optional

from .base import ModelClient, GenerationResult
from config import get_api_key


class GeminiClient(ModelClient):
    """
    Google Gemini model client using google-genai SDK.
    
    Supports Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 2.0, etc.
    """
    
    def __init__(
        self,
        model_id: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
    ):
        """
        Initialize Gemini client.
        
        Args:
            model_id: Model identifier (e.g., "gemini-1.5-flash", "gemini-1.5-pro")
            api_key: API key (defaults to GOOGLE_API_KEY env var)
        """
        from google import genai
        
        self._model_id = model_id
        self._api_key = api_key or get_api_key("google")
        self._client = genai.Client(api_key=self._api_key)
    
    @property
    def name(self) -> str:
        return f"Gemini ({self._model_id})"
    
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
        Generate using Google Gemini API.
        """
        from google.genai import types
        
        # Build generation config
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
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
