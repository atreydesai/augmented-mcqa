import main as app_main
from utils.modeling import resolve_model_name


def test_main_parser_generate_defaults_use_inspect_first_shape():
    parser = app_main.build_parser()
    args = parser.parse_args(["generate", "--model", "gpt-5.2-2025-12-11", "--run-name", "demo"])
    assert args.processed_dataset.endswith("unified_processed_v2")
    assert args.shard_count == 1
    assert args.shard_strategy == "contiguous"
    assert args.log_root.endswith("results/inspect/generation")


def test_main_parser_evaluate_defaults_use_inspect_first_shape():
    parser = app_main.build_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "--model",
            "Qwen/Qwen3-4B-Instruct-2507",
            "--run-name",
            "eval",
            "--generator-run-name",
            "gen",
            "--generator-model",
            "gpt-5.2-2025-12-11",
        ]
    )
    assert args.cache_root.endswith("datasets/augmented")
    assert args.log_root.endswith("results/inspect/evaluation")
    assert args.shard_count == 1


def test_model_alias_resolution_covers_api_and_local_defaults():
    assert resolve_model_name("gpt-5.2-2025-12-11") == "openai/gpt-5.2-2025-12-11"
    assert resolve_model_name("Qwen/Qwen3-4B-Instruct-2507") == "vllm/Qwen/Qwen3-4B-Instruct-2507"
    assert resolve_model_name("custom-model", "openai") == "openai/custom-model"
