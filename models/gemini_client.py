"""Google Gemini model client for Augmented MCQA via OpenAI-compatible API."""

from __future__ import annotations

import os
from typing import Optional, Literal

from .base import GenerationResult, ModelClient
from config import get_api_key


ReasoningEffort = Literal["minimal", "low", "medium", "high", "none"]


class GeminiClient(ModelClient):
    """Gemini client using OpenAI-compatible chat completions."""

    def __init__(
        self,
        model_id: str = "gemini-3.1-pro-preview",
        api_key: Optional[str] = None,
        reasoning_effort: Optional[ReasoningEffort] = None,
    ):
        import openai

        self._model_id = model_id
        self._api_key = api_key or get_api_key("google")
        self._reasoning_effort = reasoning_effort
        self._base_url = os.getenv(
            "GEMINI_OPENAI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self._client = openai.OpenAI(api_key=self._api_key, base_url=self._base_url)

    @property
    def name(self) -> str:
        return f"Gemini ({self._model_id})"

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        reasoning_effort: Optional[ReasoningEffort] = None,
        **kwargs,
    ) -> GenerationResult:
        params = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,
        }

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
