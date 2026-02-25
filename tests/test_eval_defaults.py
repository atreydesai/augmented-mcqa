from pathlib import Path

from experiments.config import ExperimentConfig
from experiments.defaults import (
    DEFAULT_EVAL_KEEP_CHECKPOINTS,
    DEFAULT_EVAL_MAX_TOKENS,
    DEFAULT_EVAL_MODE,
    DEFAULT_EVAL_SAVE_INTERVAL,
    DEFAULT_EVAL_SEED,
    DEFAULT_EVAL_TEMPERATURE,
    DEFAULT_MATRIX_PRESET,
)
from experiments.matrix import build_matrix_configs
from scripts import eval_matrix


def test_experiment_config_defaults_use_shared_constants(tmp_path):
    cfg = ExperimentConfig(
        name="defaults",
        dataset_path=Path("datasets/augmented/final5"),
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        generator_dataset_label="gpt-5.2-2025-12-11",
        output_dir=tmp_path / "defaults",
    )
    assert cfg.eval_mode == DEFAULT_EVAL_MODE
    assert cfg.seed == DEFAULT_EVAL_SEED
    assert cfg.temperature == DEFAULT_EVAL_TEMPERATURE
    assert cfg.max_tokens == DEFAULT_EVAL_MAX_TOKENS
    assert cfg.save_interval == DEFAULT_EVAL_SAVE_INTERVAL


def test_build_matrix_configs_defaults_use_shared_constants(tmp_path):
    cfg = build_matrix_configs(
        model="Qwen/Qwen3-4B-Instruct-2507",
        dataset_path=Path("datasets/augmented/final5"),
        generator_dataset_label="gpt-5.2-2025-12-11",
        dataset_types=["mmlu_pro"],
        output_base=tmp_path,
    )[0]
    assert cfg.eval_mode == DEFAULT_EVAL_MODE
    assert cfg.seed == DEFAULT_EVAL_SEED
    assert cfg.temperature == DEFAULT_EVAL_TEMPERATURE
    assert cfg.max_tokens == DEFAULT_EVAL_MAX_TOKENS
    assert cfg.save_interval == DEFAULT_EVAL_SAVE_INTERVAL


def test_eval_matrix_parser_defaults_use_shared_constants():
    parser = eval_matrix.build_parser()
    args = parser.parse_args(
        [
            "plan",
            "--model",
            "Qwen/Qwen3-4B-Instruct-2507",
            "--dataset-path",
            "datasets/augmented/final5",
            "--generator-dataset-label",
            "gpt-5.2-2025-12-11",
        ]
    )
    assert args.preset == DEFAULT_MATRIX_PRESET
    assert args.eval_mode == DEFAULT_EVAL_MODE
    assert args.seed == DEFAULT_EVAL_SEED
    assert args.temperature == DEFAULT_EVAL_TEMPERATURE
    assert args.max_tokens == DEFAULT_EVAL_MAX_TOKENS
    assert args.save_interval == DEFAULT_EVAL_SAVE_INTERVAL
    assert args.entry_shard_strategy == "contiguous"

    run_args = parser.parse_args(
        [
            "run",
            "--manifest",
            "manifest.json",
            "--generator-dataset-label",
            "gpt-5.2-2025-12-11",
        ]
    )
    assert run_args.keep_checkpoints == DEFAULT_EVAL_KEEP_CHECKPOINTS
    assert run_args.entry_shard_strategy == "contiguous"


def test_legacy_presets_are_rejected():
    for preset in ["core16", "branching21"]:
        try:
            eval_matrix._validate_preset(preset)
        except ValueError as exc:
            assert "archived" in str(exc)
        else:
            raise AssertionError(f"Expected ValueError for preset={preset}")


def test_eval_matrix_plan_no_manifest_attribute_crash(tmp_path):
    out_manifest = tmp_path / "plan_manifest.json"
    rc = eval_matrix.main(
        [
            "plan",
            "--model",
            "Qwen/Qwen3-4B-Instruct-2507",
            "--dataset-path",
            "datasets/augmented/final5",
            "--generator-dataset-label",
            "gpt-5.2-2025-12-11",
            "--manifest-out",
            str(out_manifest),
        ]
    )
    assert rc == 0
    assert out_manifest.exists()
