"""Provider registry and declarative model alias resolution."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import tomllib

from config import PROJECT_ROOT

from .anthropic_client import AnthropicClient
from .gemini_client import GeminiClient
from .local_client import LocalClient
from .openai_client import OpenAIClient


ALIASES_PATH = PROJECT_ROOT / "config" / "model_aliases.toml"

PROVIDER_REGISTRY = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "gemini": GeminiClient,
    "local": LocalClient,
}


@dataclass(frozen=True)
class AliasSpec:
    """Alias configuration loaded from TOML."""

    name: str
    provider: str
    model_id: str | None = None
    defaults: dict[str, Any] | None = None


@lru_cache(maxsize=1)
def load_model_aliases(path: Path | None = None) -> dict[str, AliasSpec]:
    """Load aliases from TOML and cache for process lifetime."""
    alias_path = path or ALIASES_PATH
    if not alias_path.exists():
        return {}

    with open(alias_path, "rb") as f:
        raw = tomllib.load(f)

    aliases_data = raw.get("aliases", {})
    if not isinstance(aliases_data, dict):
        raise ValueError(f"Invalid aliases file {alias_path}: [aliases] must be a table")

    aliases: dict[str, AliasSpec] = {}
    for alias_name, alias_cfg in aliases_data.items():
        if not isinstance(alias_cfg, dict):
            raise ValueError(
                f"Invalid alias '{alias_name}' in {alias_path}: alias value must be a table"
            )

        provider = alias_cfg.get("provider")
        if not provider or not isinstance(provider, str):
            raise ValueError(
                f"Invalid alias '{alias_name}' in {alias_path}: provider is required"
            )

        model_id = alias_cfg.get("model_id")
        if model_id is not None and not isinstance(model_id, str):
            raise ValueError(
                f"Invalid alias '{alias_name}' in {alias_path}: model_id must be a string"
            )

        defaults = alias_cfg.get("defaults", {})
        if defaults is None:
            defaults = {}
        if not isinstance(defaults, dict):
            raise ValueError(
                f"Invalid alias '{alias_name}' in {alias_path}: defaults must be a table"
            )

        aliases[alias_name] = AliasSpec(
            name=alias_name,
            provider=provider,
            model_id=model_id,
            defaults=dict(defaults),
        )

    return aliases


def clear_model_alias_cache() -> None:
    """Clear alias cache for tests or live reload scenarios."""
    load_model_aliases.cache_clear()


def list_model_aliases(include_providers: bool = True) -> list[str]:
    """List known model aliases (plus provider shortcuts by default)."""
    names = set(load_model_aliases().keys())
    if include_providers:
        names.update(PROVIDER_REGISTRY.keys())
    return sorted(names)


def _resolve_by_heuristic(model_name: str) -> tuple[str, str | None, dict[str, Any]]:
    model_lower = model_name.lower()

    if model_name == "local":
        return "local", None, {}
    if "/" in model_name:
        return "local", model_name, {}

    if "gpt" in model_lower or "openai" in model_lower:
        return "openai", model_name, {}
    if "claude" in model_lower or "anthropic" in model_lower:
        return "anthropic", model_name, {}
    if "gemini" in model_lower:
        return "gemini", model_name, {}
    available = ", ".join(list_model_aliases(include_providers=True))
    raise ValueError(f"Unknown model: {model_name}. Available aliases/providers: {available}")


def resolve_model(model_name: str) -> tuple[str, str | None, dict[str, Any]]:
    """Resolve model name to (provider, model_id, defaults)."""
    aliases = load_model_aliases()

    if model_name in aliases:
        spec = aliases[model_name]
        if spec.provider not in PROVIDER_REGISTRY:
            valid = ", ".join(sorted(PROVIDER_REGISTRY.keys()))
            raise ValueError(
                f"Alias '{model_name}' maps to unknown provider '{spec.provider}'. "
                f"Valid providers: {valid}"
            )
        return spec.provider, spec.model_id, dict(spec.defaults or {})

    if model_name in PROVIDER_REGISTRY:
        return model_name, None, {}

    return _resolve_by_heuristic(model_name)


def get_provider_registry() -> dict[str, type]:
    """Return provider registry mapping provider -> client class."""
    return dict(PROVIDER_REGISTRY)


def create_client(model_name: str, **kwargs):
    """Create a model client from alias/provider/heuristic resolution."""
    provider, model_id, defaults = resolve_model(model_name)
    client_class = PROVIDER_REGISTRY[provider]

    init_kwargs: dict[str, Any] = dict(defaults)
    init_kwargs.update(kwargs)

    if model_id is not None and "model_id" not in init_kwargs:
        init_kwargs["model_id"] = model_id

    return client_class(**init_kwargs)
