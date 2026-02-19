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


def test_local_alias_defaults_are_loaded():
    provider, model_id, defaults = resolve_model("Qwen/Qwen3-4B-Instruct-2507")
    assert provider == "local"
    assert model_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert defaults["dtype"] == "bfloat16"
    assert defaults["max_model_len"] == 32768
    assert defaults["trust_remote_code"] is True


def test_nanbeige_alias_uses_slow_tokenizer_default():
    provider, model_id, defaults = resolve_model("Nanbeige/Nanbeige4.1-3B")
    assert provider == "local"
    assert model_id == "Nanbeige/Nanbeige4.1-3B"
    assert defaults["tokenizer_mode"] == "slow"
    assert defaults["stop_token_ids"] == [166101]
