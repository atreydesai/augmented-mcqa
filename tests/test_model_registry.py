from models.registry import list_model_aliases, resolve_model


def test_alias_resolution_from_toml():
    provider, model_id, defaults = resolve_model("gpt-4.1")
    assert provider == "openai"
    assert model_id == "gpt-4.1"
    assert defaults == {}


def test_heuristic_resolution_when_alias_missing():
    provider, model_id, defaults = resolve_model("gpt-4")
    assert provider == "openai"
    assert model_id == "gpt-4"
    assert defaults == {}


def test_alias_list_includes_provider_shortcuts():
    names = list_model_aliases(include_providers=True)
    assert "openai" in names
    assert "anthropic" in names
    assert "gpt-4.1" in names
