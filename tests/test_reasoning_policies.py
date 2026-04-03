import json

import utils.modeling as modeling
from utils.modeling import is_nemotron_model, reasoning_effort_for_model, resolve_model_name, vllm_server_args


def test_provider_qualified_aliases_are_preserved():
    assert resolve_model_name("openai/gpt-5.2-2025-12-11") == "openai/gpt-5.2-2025-12-11"
    assert resolve_model_name("vllm/meta-llama/Llama-3.1-8B-Instruct") == "vllm/meta-llama/Llama-3.1-8B-Instruct"


def test_backend_prefix_is_applied_for_unqualified_models():
    assert resolve_model_name("my-local-model", backend="openai") == "openai/my-local-model"
    assert resolve_model_name("openai/my-local-model", backend="openai") == "openai/my-local-model"


def test_nemotron_alias_and_detection_are_supported():
    resolved = resolve_model_name("nvidia/NVIDIA-Nemotron-Nano-9B-v2")
    assert resolved == "vllm/nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    assert is_nemotron_model("nvidia/NVIDIA-Nemotron-Nano-9B-v2") is True
    assert is_nemotron_model(resolved) is True


def test_registered_resolved_ids_are_valid_inputs(tmp_path, monkeypatch):
    registry_path = tmp_path / "model_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "name": "tiny-local",
                        "resolved": "vllm/acme/Tiny-Instruct",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(modeling, "MODEL_REGISTRY_PATH", registry_path)
    modeling._model_aliases.cache_clear()
    try:
        assert modeling.resolve_model_name("tiny-local") == "vllm/acme/Tiny-Instruct"
        assert modeling.resolve_model_name("vllm/acme/Tiny-Instruct") == "vllm/acme/Tiny-Instruct"
    finally:
        modeling._model_aliases.cache_clear()


def test_vllm_server_args_use_eval_friendly_defaults():
    assert vllm_server_args("Qwen/Qwen3-4B-Instruct-2507") == {
        "enforce_eager": True,
        "max_model_len": 8192,
    }
    assert vllm_server_args("nvidia/NVIDIA-Nemotron-Nano-9B-v2") == {
        "enforce_eager": True,
        "max_model_len": 8192,
        "trust_remote_code": True,
        "mamba_ssm_cache_dtype": "float32",
    }
    assert vllm_server_args("allenai/Olmo-3-7B-Instruct") == {
        "enforce_eager": True,
        "max_model_len": 8192,
        "trust_remote_code": True,
    }
    assert vllm_server_args("gpt-5.2-2025-12-11") == {}


def test_reasoning_effort_is_disabled_for_together_qwen_models():
    assert reasoning_effort_for_model("Qwen/Qwen3.5-397B-A17B", "medium") is None
    assert reasoning_effort_for_model("together/Qwen/Qwen3.5-397B-A17B", "medium") is None
    assert reasoning_effort_for_model("gpt-5.2-2025-12-11", "medium") == "medium"
