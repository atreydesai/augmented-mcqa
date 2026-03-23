from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

NEMOTRON_MODEL_IDS = (
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    "vllm/nvidia/NVIDIA-Nemotron-Nano-9B-v2",
)
MINISTRAL_MODEL_IDS = (
    "mistralai/Ministral-3-14B-Instruct-2512",
    "vllm/mistralai/Ministral-3-14B-Instruct-2512",
)
TRUST_REMOTE_CODE_MODEL_IDS = (
    "allenai/Olmo-3-7B-Instruct",
    "vllm/allenai/Olmo-3-7B-Instruct",
    *NEMOTRON_MODEL_IDS,
)
NO_REASONING_EFFORT_MODEL_IDS = (
    "together/Qwen/Qwen3.5-397B-A17B",
    "together/Qwen/Qwen3.5-9B",
)
VLLM_STARTUP_MAX_MODEL_LEN = 8192
VLLM_STARTUP_SERVER_ARGS = {
    "enforce_eager": True,
    "max_model_len": VLLM_STARTUP_MAX_MODEL_LEN,
}

MODEL_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "config" / "model_registry.json"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "x"


def is_nemotron_model(model: str) -> bool:
    model = str(model or "").strip()
    return model in NEMOTRON_MODEL_IDS


@lru_cache(maxsize=1)
def _model_aliases() -> dict[str, str]:
    payload = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    aliases: dict[str, str] = {}
    for entry in payload.get("models", []):
        name = str(entry.get("name", "")).strip()
        resolved = str(entry.get("resolved", "")).strip()
        if name and resolved:
            aliases[name] = resolved
            aliases.setdefault(resolved, resolved)
    return aliases


def resolve_model_name(model: str, backend: str | None = None) -> str:
    model = str(model or "").strip()
    if not model:
        raise ValueError("model is required")
    aliases = _model_aliases()
    if model in aliases:
        return aliases[model]
    if backend:
        backend = str(backend).strip().lower()
        if not backend:
            raise ValueError("backend cannot be blank")
        candidate = model if model.startswith(f"{backend}/") else f"{backend}/{model}"
        if candidate in aliases:
            return aliases[candidate]
        return candidate
    raise ValueError(
        f"Unknown model {model!r}. Add it to {MODEL_REGISTRY_PATH} before using it."
    )


def vllm_server_args(model: str) -> dict[str, object]:
    raw_model = str(model or "").strip()
    if not raw_model:
        return {}
    try:
        resolved = resolve_model_name(raw_model)
    except ValueError:
        resolved = raw_model
    if not str(resolved).startswith("vllm/"):
        return {}
    args = dict(VLLM_STARTUP_SERVER_ARGS)
    if resolved in TRUST_REMOTE_CODE_MODEL_IDS:
        args["trust_remote_code"] = True
    if resolved in NEMOTRON_MODEL_IDS:
        args["mamba_ssm_cache_dtype"] = "float32"
    if resolved in MINISTRAL_MODEL_IDS:
        args.update(
            tokenizer_mode="mistral",
            config_format="mistral",
            load_format="mistral",
        )
    return args


def reasoning_effort_for_model(model: str, reasoning_effort: str | None) -> str | None:
    effort = str(reasoning_effort or "").strip() or None
    if effort is None:
        return None
    raw_model = str(model or "").strip()
    if not raw_model:
        return effort
    try:
        resolved = resolve_model_name(raw_model)
    except ValueError:
        resolved = raw_model
    if resolved in NO_REASONING_EFFORT_MODEL_IDS:
        return None
    return effort
