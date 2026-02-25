"""
Anthropic Claude model client for Augmented MCQA.

Supports Claude 4.x models with extended thinking capability.
"""

from typing import Optional, Literal

from .base import ModelClient, GenerationResult
from config import get_api_key


# Valid thinking budget presets
ThinkingLevel = Literal["off", "low", "medium", "high", "adaptive"]


class AnthropicClient(ModelClient):
    """
    Anthropic Claude model client using Messages API.
    
    Supports:
    - Claude Opus 4, Sonnet 4, Haiku 4.5 with extended thinking
    - Extended thinking allows deeper reasoning with configurable budget
    """
    
    # Models that support extended thinking
    THINKING_MODELS = {
        "claude-opus-4", "claude-sonnet-4", "claude-haiku-4",
        "claude-opus-4-6", "claude-sonnet-4-5", "claude-haiku-4-5"
    }
    
    # Approximate thinking budgets (in tokens)
    THINKING_BUDGETS = {
        "off": 0,
        "low": 1024,
        "medium": 4096,
        "high": 16384,
    }
    
    def __init__(
        self,
        model_id: str = "claude-sonnet-4-5-20250929",
        api_key: Optional[str] = None,
        thinking_level: ThinkingLevel = "off",
        request_timeout: Optional[float] = 60.0,
    ):
        """
        Initialize Anthropic client.
        
        Args:
            model_id: Model identifier
                - "claude-opus-4-6": Claude Opus 4.6
                - "claude-sonnet-4-5-20250929": Claude Sonnet 4.5
                - "claude-haiku-4-5-20251001": Claude Haiku 4.5
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
            thinking_level: Extended thinking budget level:
                - "off": No extended thinking
                - "low": Light reasoning (~1K tokens)
                - "medium": Moderate reasoning (~4K tokens)
                - "high": Deep reasoning (~16K tokens)
        """
        import anthropic
        
        self._model_id = model_id
        self._api_key = api_key or get_api_key("anthropic")
        self._client = anthropic.Anthropic(api_key=self._api_key)
        self._thinking_level = thinking_level
        self._request_timeout = request_timeout
        
        # Check if model supports thinking
        self._supports_thinking = any(
            prefix in self._model_id for prefix in self.THINKING_MODELS
        )
    
    @property
    def name(self) -> str:
        return f"Anthropic ({self._model_id})"
    
    @property
    def model_id(self) -> str:
        return self._model_id
    
    @property
    def supports_thinking(self) -> bool:
        """Check if this model supports extended thinking."""
        return self._supports_thinking
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        thinking_level: Optional[ThinkingLevel] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate using Anthropic Messages API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum output tokens
            thinking_level: Override thinking level for this call
            **kwargs: Additional API parameters
        """
        # Build request parameters
        params = {
            "model": self._model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        timeout = kwargs.pop("timeout", getattr(self, "_request_timeout", 120.0))
        if timeout is not None:
            params["timeout"] = timeout
        
        # Add extended thinking for supported models, unless caller provided an
        # explicit `thinking` payload.
        level = thinking_level or self._thinking_level
        if self._supports_thinking and "thinking" not in kwargs and level != "off":
            if level == "adaptive":
                params["thinking"] = {"type": "adaptive"}
            else:
                budget = self.THINKING_BUDGETS.get(level, 4096)
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget,
                }
        
        params.update(kwargs)
        
        response = self._client.messages.create(**params)
        
        # Extract text from content blocks. With adaptive thinking enabled,
        # some responses may return thinking blocks without a text block.
        text_parts = []
        thinking_parts = []

        def _append_thinking_fragment(value):
            if value is None:
                return
            if isinstance(value, str):
                if value.strip():
                    thinking_parts.append(value)
                return
            if isinstance(value, list):
                for item in value:
                    _append_thinking_fragment(item)
                return
            if isinstance(value, dict):
                if value.get("text"):
                    thinking_parts.append(str(value["text"]))
                elif value.get("thinking"):
                    _append_thinking_fragment(value.get("thinking"))
                return
            if hasattr(value, "text") and getattr(value, "text"):
                thinking_parts.append(str(getattr(value, "text")))

        for block in response.content:
            block_type = getattr(block, "type", None)
            block_text = getattr(block, "text", None)
            block_thinking = getattr(block, "thinking", None)

            if block_type == "text" and block_text:
                text_parts.append(block_text)
                continue
            if block_text:
                text_parts.append(block_text)
                continue
            if block_thinking:
                _append_thinking_fragment(block_thinking)

        text = "".join(text_parts).strip()
        if not text and thinking_parts:
            text = "\n".join(x for x in thinking_parts if x.strip())

        if not text:
            # Fallback through model dump for SDK variants with dict-like blocks.
            try:
                dumped = response.model_dump()
                dumped_content = dumped.get("content", []) if isinstance(dumped, dict) else []
                for block in dumped_content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text" and block.get("text"):
                        text += str(block["text"])
                    elif block.get("thinking"):
                        thinking_val = block["thinking"]
                        if isinstance(thinking_val, str):
                            text += f"\n{thinking_val}"
                        elif isinstance(thinking_val, list):
                            for item in thinking_val:
                                if isinstance(item, dict) and item.get("text"):
                                    text += f"\n{item['text']}"
                                elif isinstance(item, str) and item.strip():
                                    text += f"\n{item}"
                text = text.strip()
            except Exception:  # noqa: BLE001
                pass
        
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
