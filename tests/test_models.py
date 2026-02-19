import pytest

from models import get_client, ModelClient, GenerationResult
from models import registry as model_registry


class DummyClient:
    def __init__(self, model_id=None, **kwargs):
        self._model_id = model_id
        self.kwargs = kwargs

    @property
    def model_id(self):
        return self._model_id


def test_generation_result_extract_answer():
    """Test answer extraction logic."""
    res = GenerationResult(text="The answer is (A)", model="test")
    assert res.extract_answer() == "A"

    res = GenerationResult(text="The answer is B", model="test")
    assert res.extract_answer() == "B"

    res = GenerationResult(text="Reasoning... So the answer is C.", model="test")
    assert res.extract_answer() == "C"


def test_get_client_openai(monkeypatch):
    """Heuristic OpenAI resolution should route to openai provider."""
    monkeypatch.setitem(model_registry.PROVIDER_REGISTRY, "openai", DummyClient)
    client = get_client("gpt-4")
    assert isinstance(client, DummyClient)
    assert client.model_id == "gpt-4"


def test_get_client_anthropic(monkeypatch):
    """Heuristic Anthropic resolution should route to anthropic provider."""
    monkeypatch.setitem(model_registry.PROVIDER_REGISTRY, "anthropic", DummyClient)
    client = get_client("claude-3-opus-20240229")
    assert isinstance(client, DummyClient)
    assert client.model_id == "claude-3-opus-20240229"


def test_get_client_registry_lookup(monkeypatch):
    """Alias lookup should route through registry TOML."""
    monkeypatch.setitem(model_registry.PROVIDER_REGISTRY, "openai", DummyClient)
    client = get_client("gpt-4.1")
    assert isinstance(client, DummyClient)
    assert client.model_id == "gpt-4.1"


def test_get_client_unknown():
    """Unknown client should raise a clear error."""
    with pytest.raises(ValueError):
        get_client("unknown-model-123")


def test_model_client_interface():
    """ModelClient is abstract and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ModelClient()
