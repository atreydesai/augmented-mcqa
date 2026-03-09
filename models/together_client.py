"""Together AI model client for Augmented MCQA via OpenAI-compatible API."""

from __future__ import annotations

import os
import sys
from typing import Optional

from .base import GenerationResult, ModelClient
from config import get_api_key


class TogetherClient(ModelClient):
    """Together AI client using OpenAI-compatible chat completions."""

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        api_key: Optional[str] = None,
    ):
        import openai

        self._model_id = model_id
        self._api_key = api_key or get_api_key("together")
        self._base_url = os.getenv(
            "TOGETHER_BASE_URL",
            "https://api.together.xyz/v1",
        )
        self._client = openai.OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )

    @property
    def name(self) -> str:
        return f"Together ({self._model_id})"

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        **kwargs,
    ) -> GenerationResult:
        params = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
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

        result = GenerationResult(
            text=choice.message.content or "",
            model=response.model,
            finish_reason=choice.finish_reason,
            usage=usage,
            raw_response=response,
        )

        return result
