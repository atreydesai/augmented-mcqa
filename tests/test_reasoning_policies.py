from unittest.mock import MagicMock

from models.anthropic_client import AnthropicClient
from models.gemini_client import GeminiClient
from scripts.generate_distractors import _build_config, _model_policy


def test_generation_model_policy_mapping():
    provider, client_kwargs, generate_kwargs = _model_policy("gpt-5.2-2025-12-11")
    assert provider == "openai"
    assert client_kwargs == {"reasoning_effort": "medium"}
    assert generate_kwargs == {}


def test_build_config_preserves_anthropic_timeout_and_thinking():
    cfg = _build_config(
        model_name="claude-opus-4-6",
        save_interval=25,
        force_overwrite=False,
        skip_failed_entries=False,
        max_retries=3,
        retry_delay=1.0,
        request_log=None,
        slow_call_seconds=45.0,
    )
    assert cfg.model_provider == "anthropic"
    assert cfg.anthropic_thinking == {"type": "adaptive"}
    assert cfg.generate_kwargs.get("timeout") == 60.0
    assert cfg.max_tokens == 2048

    provider, client_kwargs, generate_kwargs = _model_policy("claude-opus-4-6")
    assert provider == "anthropic"
    assert client_kwargs == {}
    assert generate_kwargs == {"thinking": {"type": "adaptive"}, "timeout": 60.0}

    provider, client_kwargs, generate_kwargs = _model_policy("gemini-3.1-pro-preview")
    assert provider == "gemini"
    assert client_kwargs == {}
    assert generate_kwargs == {}


def test_build_config_uses_uniform_2048_output_cap():
    for model_name in [
        "gpt-5.2-2025-12-11",
        "claude-opus-4-6",
        "gemini-3.1-pro-preview",
    ]:
        cfg = _build_config(
            model_name=model_name,
            save_interval=25,
            force_overwrite=False,
            skip_failed_entries=False,
            max_retries=3,
            retry_delay=1.0,
            request_log=None,
            slow_call_seconds=45.0,
        )
        assert cfg.max_tokens == 2048


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


def test_anthropic_falls_back_to_thinking_text_when_text_block_missing():
    client = AnthropicClient.__new__(AnthropicClient)
    client._model_id = "claude-opus-4-6"
    client._thinking_level = "adaptive"
    client._supports_thinking = True

    thinking_block = MagicMock()
    thinking_block.type = "thinking"
    thinking_block.text = None
    thinking_block.thinking = "E: wrong one\nF: wrong two\nG: wrong three"

    mock_response = MagicMock()
    mock_response.content = [thinking_block]
    mock_response.usage = None
    mock_response.model = "claude-opus-4-6"
    mock_response.stop_reason = "end_turn"

    class _Messages:
        @staticmethod
        def create(**_params):
            return mock_response

    class _DummyAnthropic:
        messages = _Messages()

    client._client = _DummyAnthropic()

    out = client.generate("prompt", max_tokens=32)
    assert "E: wrong one" in out.text


def test_gemini_client_uses_openai_compat_base_url(monkeypatch):
    captured = {}

    class _DummyOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("openai.OpenAI", _DummyOpenAI)

    client = GeminiClient(model_id="gemini-3.1-pro-preview", api_key="x")

    assert client.model_id == "gemini-3.1-pro-preview"
    assert captured["api_key"] == "x"
    assert captured["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai/"


def test_gemini_generate_passes_reasoning_effort():
    client = GeminiClient.__new__(GeminiClient)
    client._model_id = "gemini-3.1-pro-preview"
    client._reasoning_effort = "medium"

    captured = {}
    mock_response = MagicMock()
    mock_response.model = "gemini-3.1-pro-preview"
    mock_response.usage = None
    choice = MagicMock()
    choice.message.content = '{"distractors":["x","y","z"]}'
    choice.finish_reason = "stop"
    mock_response.choices = [choice]

    class _Completions:
        @staticmethod
        def create(**params):
            captured.update(params)
            return mock_response

    class _Chat:
        completions = _Completions()

    class _DummyClient:
        chat = _Chat()

    client._client = _DummyClient()

    out = client.generate(
        prompt="hello",
        max_tokens=8,
        response_format={"type": "json_schema"},
    )
    assert out.text == '{"distractors":["x","y","z"]}'
    assert captured["reasoning_effort"] == "medium"
    assert captured["max_completion_tokens"] == 8
    assert captured["response_format"] == {"type": "json_schema"}
