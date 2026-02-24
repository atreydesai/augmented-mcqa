"""Google Gemini model client for Augmented MCQA."""

from __future__ import annotations

from typing import Any, Optional

from .base import GenerationResult, ModelClient
from config import get_api_key


class GeminiClient(ModelClient):
    """Gemini client using google-genai SDK.

    Guardrail: this client never allows both thinking_level and thinking_budget
    in the same request.
    """

    def __init__(
        self,
        model_id: str = "gemini-3.1-pro-preview",
        api_key: Optional[str] = None,
        thinking_level: Optional[str] = None,
        thinking_budget: Optional[int] = None,
    ):
        from google import genai

        if thinking_level is not None and thinking_budget is not None:
            raise ValueError(
                "Gemini requests cannot specify both thinking_level and thinking_budget"
            )

        self._model_id = model_id
        self._api_key = api_key or get_api_key("google")
        self._client = genai.Client(api_key=self._api_key)
        self._default_thinking_level = thinking_level
        self._default_thinking_budget = thinking_budget

    @property
    def name(self) -> str:
        return f"Gemini ({self._model_id})"

    @property
    def model_id(self) -> str:
        return self._model_id

    @staticmethod
    def _normalize_thinking_level(level: Optional[str]) -> Optional[str]:
        if level is None:
            return None
        normalized = str(level).strip().upper()
        if not normalized:
            return None
        allowed = {"LOW", "MEDIUM", "HIGH"}
        if normalized not in allowed:
            raise ValueError(f"Unsupported Gemini thinking_level={level!r}. Allowed: {sorted(allowed)}")
        return normalized

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        thinking_level: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        **kwargs,
    ) -> GenerationResult:
        from google.genai import types

        level = thinking_level if thinking_level is not None else self._default_thinking_level
        budget = thinking_budget if thinking_budget is not None else self._default_thinking_budget

        if level is not None and budget is not None:
            raise ValueError(
                "Gemini request invalid: cannot use both thinking_level and thinking_budget"
            )

        config_params: dict[str, Any] = {
            "max_output_tokens": max_tokens,
        }

        # Pass through selected decoding options when provided.
        passthrough = [
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "candidate_count",
            "seed",
        ]
        for key in passthrough:
            if key in kwargs and kwargs[key] is not None:
                config_params[key] = kwargs.pop(key)

        if "thinking_config" in kwargs and kwargs["thinking_config"] is not None:
            if level is not None or budget is not None:
                raise ValueError(
                    "Gemini request invalid: explicit thinking_config cannot be combined with "
                    "thinking_level or thinking_budget"
                )
            config_params["thinking_config"] = kwargs.pop("thinking_config")
        else:
            normalized_level = self._normalize_thinking_level(level)
            if normalized_level is not None:
                config_params["thinking_config"] = types.ThinkingConfig(
                    thinking_level=normalized_level
                )
            elif budget is not None:
                config_params["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=int(budget)
                )

        config = types.GenerateContentConfig(**config_params)

        response = self._client.models.generate_content(
            model=self._model_id,
            contents=prompt,
            config=config,
        )

        text = ""
        try:
            text = response.text or ""
        except Exception:
            text = ""

        if not text and response.candidates:
            parts = response.candidates[0].content.parts
            text = "\n".join(
                p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False)
            )

        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", None),
                "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", None),
                "total_tokens": getattr(response.usage_metadata, "total_token_count", None),
            }

        finish_reason = None
        if response.candidates:
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
