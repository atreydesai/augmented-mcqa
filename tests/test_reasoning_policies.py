from unittest.mock import MagicMock

from models.anthropic_client import AnthropicClient
from models.gemini_client import GeminiClient
from scripts.generate_distractors import _model_policy


def test_generation_model_policy_mapping():
    provider, client_kwargs, generate_kwargs = _model_policy("gpt-5.2-2025-12-11")
    assert provider == "openai"
    assert client_kwargs == {"reasoning_effort": "medium"}
    assert generate_kwargs == {}

    provider, client_kwargs, generate_kwargs = _model_policy("claude-opus-4-6")
    assert provider == "anthropic"
    assert client_kwargs == {}
    assert generate_kwargs == {"thinking": {"type": "adaptive"}}

    provider, client_kwargs, generate_kwargs = _model_policy("gemini-3.1-pro-preview")
    assert provider == "gemini"
    assert client_kwargs == {}
    assert generate_kwargs == {}


def test_gemini_guard_rejects_mixed_thinking_level_and_budget_in_generate():
    client = GeminiClient.__new__(GeminiClient)
    client._model_id = "gemini-3.1-pro-preview"
    client._default_thinking_level = None
    client._default_thinking_budget = None

    try:
        client.generate(
            prompt="hello",
            max_tokens=10,
            thinking_level="LOW",
            thinking_budget=128,
        )
    except ValueError as exc:
        assert "cannot use both thinking_level and thinking_budget" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mixed Gemini thinking params")


def test_anthropic_adaptive_thinking_payload_is_explicit():
    client = AnthropicClient.__new__(AnthropicClient)
    client._model_id = "claude-opus-4-6"
    client._thinking_level = "adaptive"
    client._supports_thinking = True

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="The answer is A")]
    mock_response.usage = None
    mock_response.model = "claude-opus-4-6"
    mock_response.stop_reason = "end_turn"

    captured = {}

    class _Messages:
        @staticmethod
        def create(**params):
            captured.update(params)
            return mock_response

    class _DummyAnthropic:
        messages = _Messages()

    client._client = _DummyAnthropic()

    out = client.generate("prompt", max_tokens=32)
    assert out.text == "The answer is A"
    assert captured["thinking"] == {"type": "adaptive"}


def test_gemini_init_rejects_mixed_thinking_knobs():
    try:
        GeminiClient(model_id="gemini-3.1-pro-preview", api_key="x", thinking_level="LOW", thinking_budget=64)
    except ValueError as exc:
        assert "cannot specify both thinking_level and thinking_budget" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
