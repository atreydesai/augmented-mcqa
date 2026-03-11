import os
from pathlib import Path

import main as app_main
from utils.modeling import resolve_model_name


def test_main_parser_generate_defaults_use_inspect_first_shape():
    parser = app_main.build_parser()
    args = parser.parse_args(["generate", "--model", "gpt-5.2-2025-12-11", "--run-name", "demo"])
    assert args.processed_dataset.endswith("unified_processed_v3")
    assert args.shard_count == 1
    assert args.shard_strategy == "contiguous"
    assert Path(args.log_root).relative_to(Path(os.environ["RESULTS_DIR"])) == Path("inspect/generation")


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
    assert Path(args.cache_root).relative_to(Path(os.environ["DATASETS_DIR"])) == Path("augmented")
    assert Path(args.log_root).relative_to(Path(os.environ["RESULTS_DIR"])) == Path("inspect/evaluation")
    assert args.shard_count == 1


def test_main_parser_submit_generate_cluster_defaults_use_local_cluster_shape():
    parser = app_main.build_parser()
    args = parser.parse_args(["submit-generate-cluster", "--run-name", "cluster-gen"])
    assert args.gpu_count is None
    assert args.partition == "clip"
    assert args.account == "clip"
    assert args.qos == "high"
    assert args.gpu_type == "rtxa6000"
    assert args.submit is True


def test_main_parser_submit_evaluate_cluster_defaults_use_local_cluster_shape():
    parser = app_main.build_parser()
    args = parser.parse_args(
        [
            "submit-evaluate-cluster",
            "--run-name",
            "cluster-eval",
            "--generator-run-name",
            "gen",
            "--generator-model",
            "gpt-5.2-2025-12-11",
        ]
    )
    assert args.gpu_count is None
    assert args.partition == "clip"
    assert args.account == "clip"
    assert args.qos == "high"
    assert args.gpu_type == "rtxa6000"
    assert args.submit is True


def test_supported_main_subcommands_match_the_inspect_first_cli():
    parser = app_main.build_parser()
    subparser_action = next(action for action in parser._actions if getattr(action, "choices", None))
    assert set(subparser_action.choices) == {
        "prepare-data",
        "generate",
        "generate-all",
        "evaluate",
        "evaluate-all",
        "analyze",
        "signature-table",
        "export",
        "submit-generate-cluster",
        "submit-evaluate-cluster",
        "diagnose-failures",
        "diagnose-trace",
        "smoke-generate",
        "smoke-evaluate",
    }


def test_generate_help_describes_materialize_cache_flag(capsys):
    parser = app_main.build_parser()
    try:
        parser.parse_args(["generate", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    output = capsys.readouterr().out
    assert "--materialize-cache" in output
    assert "augmented DatasetDict cache immediately" in output


def test_cluster_help_mentions_gpu_count_and_write_only(capsys):
    parser = app_main.build_parser()
    try:
        parser.parse_args(["submit-generate-cluster", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    output = capsys.readouterr().out
    assert "--gpu-count" in output
    assert "concurrency cap" in output
    assert "--write-only" in output


def test_prepare_data_step_all_implies_download_all(monkeypatch):
    captured = {}

    def fake_prepare_data(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(app_main, "prepare_data", fake_prepare_data)

    rc = app_main.main(
        [
            "prepare-data",
            "--step",
            "all",
            "--output-path",
            "datasets/processed/unified_processed_v3",
        ]
    )

    assert rc == 0
    assert captured["step"] == "all"
    assert captured["download_all"] is True
    assert captured["dataset"] is None


def test_model_alias_resolution_covers_api_and_local_defaults():
    assert resolve_model_name("gpt-5.2-2025-12-11") == "openai/gpt-5.2-2025-12-11"
    assert resolve_model_name("Qwen/Qwen3.5-397B-A17B") == "together/Qwen/Qwen3.5-397B-A17B"
    assert resolve_model_name("Qwen/Qwen3-4B-Instruct-2507") == "vllm/Qwen/Qwen3-4B-Instruct-2507"
    assert resolve_model_name("custom-model", "openai") == "openai/custom-model"
