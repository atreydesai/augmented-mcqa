from __future__ import annotations

import re

NEMOTRON_MODEL_IDS = (
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    "vllm/nvidia/NVIDIA-Nemotron-Nano-9B-v2",
)

MODEL_ALIASES = {
    "gpt-5.2-2025-12-11": "openai/gpt-5.2-2025-12-11",
    "claude-opus-4-6": "anthropic/claude-opus-4-6",
    "gemini-3.1-pro-preview": "google/gemini-3.1-pro-preview",
    "Qwen/Qwen3.5-397B-A17B": "together/Qwen/Qwen3.5-397B-A17B",
    "Qwen/Qwen3.5-9B": "together/Qwen/Qwen3.5-9B",
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2": "vllm/nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    "Qwen/Qwen3-4B-Instruct-2507": "vllm/Qwen/Qwen3-4B-Instruct-2507",
    "allenai/Olmo-3-7B-Instruct": "vllm/allenai/Olmo-3-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "vllm/meta-llama/Llama-3.1-8B-Instruct",
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "x"


def is_nemotron_model(model: str) -> bool:
    model = str(model or "").strip()
    return model in NEMOTRON_MODEL_IDS


def resolve_model_name(model: str, backend: str | None = None) -> str:
    model = str(model or "").strip()
    if not model:
        raise ValueError("model is required")
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    if backend:
        backend = str(backend).strip().lower()
        if not backend:
            raise ValueError("backend cannot be blank")
        if model.startswith(f"{backend}/"):
            return model
        return f"{backend}/{model}"
    return model
