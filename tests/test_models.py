import pytest
from unittest.mock import MagicMock, patch
from models import get_client, ModelClient, GenerationResult
from models.openai_client import OpenAIClient
from models.anthropic_client import AnthropicClient
from models.gemini_client import GeminiClient

def test_generation_result_extract_answer():
    """Test answer extraction logic."""
    # Pattern 1: "The answer is (A)"
    res = GenerationResult(text="The answer is (A)", model="test")
    assert res.extract_answer() == "A"
    
    # Pattern 1 variation: "The answer is B"
    res = GenerationResult(text="The answer is B", model="test")
    assert res.extract_answer() == "B"
    
    # Pattern 2: Last capital letter
    res = GenerationResult(text="Reasoning... So the answer is C.", model="test")
    assert res.extract_answer() == "C"

@patch("openai.OpenAI")
def test_get_client_openai(mock_openai):
    """Test getting OpenAI client."""
    client = get_client("gpt-4")
    assert isinstance(client, OpenAIClient)
    assert client.model_id == "gpt-4"

@patch("anthropic.Anthropic")
def test_get_client_anthropic(mock_anthropic):
    """Test getting Anthropic client."""
    client = get_client("claude-3-opus-20240229")
    assert isinstance(client, AnthropicClient)
    # Note: factory might perform some mapping or use exact ID

@patch("openai.OpenAI")
def test_get_client_registry_lookup(mock_openai):
    """Test getting client from registry alias."""
    client = get_client("gpt-4.1")
    assert isinstance(client, OpenAIClient)

def test_get_client_unknown():
    """Test unknown client raises error."""
    with pytest.raises(ValueError):
        get_client("unknown-model-123")

def test_model_client_interface():
    """Test that ModelClient cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ModelClient()
