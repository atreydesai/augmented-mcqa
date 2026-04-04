import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli.app as app_main
from utils.constants import DEFAULT_LOCAL_EVALUATION_MODELS
from utils.modeling import resolve_model_name, vllm_server_args


def test_main_parser_generate_defaults_use_inspect_first_shape():
    parser = app_main.build_parser()
    args = parser.parse_args(["generate", "--model", "gpt-5.2-2025-12-11", "--run-name", "demo"])
    assert args.processed_dataset.endswith("unified_processed_v3")
    assert args.shard_count == 1
    assert args.shard_strategy == "contiguous"
    assert Path(args.log_root).parts[-2:] == ("inspect", "generation")


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
    assert Path(args.cache_root).parts[-1] == "augmented"
    assert Path(args.log_root).parts[-2:] == ("inspect", "evaluation")
    assert args.shard_count == 1
    assert args.collect_evaluated is True


def test_main_parser_submit_generate_cluster_defaults_use_local_cluster_shape():
    parser = app_main.build_parser()
    args = parser.parse_args(["submit-generate-cluster", "--run-name", "cluster-gen"])
    assert args.gpu_count is None
    assert args.limit is None
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
    assert args.limit is None
    assert args.partition == "clip"
    assert args.account == "clip"
    assert args.qos == "high"
    assert args.gpu_type == "rtxa6000"
    assert args.submit is True
    assert args.augmented_dataset is None


def test_main_parser_export_requires_explicit_input():
    parser = app_main.build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["export"])

    assert excinfo.value.code == 2


def test_main_parser_export_rejects_removed_generation_resolution_flags():
    parser = app_main.build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["export", "--generator-run-name", "gen"])

    assert excinfo.value.code == 2


def test_supported_main_subcommands_match_the_inspect_first_cli():
    parser = app_main.build_parser()
    subparser_action = next(action for action in parser._actions if getattr(action, "choices", None))
    assert set(subparser_action.choices) == {
        "prepare-data",
        "build-augmented-dataset",
        "build-collected-dataset",
        "generate",
        "evaluate",
        "analyze",
        "export",
        "submit-generate-cluster",
        "submit-evaluate-cluster",
    }


def test_build_augmented_dataset_remains_callable_via_main_parser(monkeypatch):
    captured = {}

    def fake_run_materialize_store(args):
        captured["run_name"] = args.run_name
        captured["model"] = args.model
        return 0

    monkeypatch.setattr(app_main, "_run_materialize_store", fake_run_materialize_store)

    rc = app_main.main(["build-augmented-dataset", "--run-name", "gen", "--model", "Qwen/Qwen3-4B-Instruct-2507"])

    assert rc == 0
    assert captured == {
        "run_name": "gen",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
    }


def test_generate_help_describes_materialize_cache_flag(capsys):
    parser = app_main.build_parser()
    try:
        parser.parse_args(["generate", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    output = capsys.readouterr().out
    assert "--materialize-cache" in output
    assert "setting-scoped augmented store immediately" in output


def test_cluster_help_mentions_gpu_count_and_write_only(capsys):
    parser = app_main.build_parser()
    try:
        parser.parse_args(["submit-generate-cluster", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    output = capsys.readouterr().out
    assert "--gpu-count" in output
    assert "--limit" in output
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


def test_all_default_local_evaluation_models_resolve_and_provide_vllm_runtime_config():
    resolved = [resolve_model_name(model) for model in DEFAULT_LOCAL_EVALUATION_MODELS]

    assert resolved
    assert all(model.startswith("vllm/") for model in resolved)
    assert all(isinstance(vllm_server_args(model), dict) for model in resolved)


def test_inspect_eval_uses_current_venv_bin_on_path(monkeypatch, tmp_path):
    captured = {}
    fake_python = tmp_path / "venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True, exist_ok=True)
    fake_python.write_text("", encoding="utf-8")

    def fake_eval(tasks, **kwargs):  # noqa: ANN001
        captured["path"] = os.environ["PATH"]
        captured["server_args"] = os.environ.get("VLLM_DEFAULT_SERVER_ARGS", "")
        return []

    monkeypatch.setattr("inspect_ai.eval", fake_eval)
    monkeypatch.setattr(sys, "executable", str(fake_python))

    args = SimpleNamespace(
        model_base_url=None,
        retry_on_error=0,
        max_connections=1,
        max_tokens=32,
        temperature=None,
        reasoning_effort=None,
        stop_seqs=None,
    )

    app_main.inspect_eval([], model="Qwen/Qwen3-4B-Instruct-2507", log_dir=tmp_path / "logs", args=args)

    current_bin = str(fake_python.parent)
    assert captured["path"].split(os.pathsep)[0] == current_bin
    assert captured["server_args"]
