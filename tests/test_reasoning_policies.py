from utils.modeling import is_nemotron_model, resolve_model_name


def test_aliases_preserve_provider_prefixes_when_already_qualified():
    assert resolve_model_name("openai/gpt-5.2-2025-12-11") == "openai/gpt-5.2-2025-12-11"
    assert resolve_model_name("vllm/meta-llama/Llama-3.1-8B-Instruct") == "vllm/meta-llama/Llama-3.1-8B-Instruct"


def test_backend_prefix_is_applied_for_unqualified_models():
    assert resolve_model_name("my-local-model", backend="openai") == "openai/my-local-model"


def test_nemotron_alias_and_detection_are_supported():
    resolved = resolve_model_name("nvidia/NVIDIA-Nemotron-Nano-9B-v2")
    assert resolved == "vllm/nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    assert is_nemotron_model("nvidia/NVIDIA-Nemotron-Nano-9B-v2") is True
    assert is_nemotron_model(resolved) is True
