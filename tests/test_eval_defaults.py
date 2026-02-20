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
        dataset_path=Path("datasets/augmented/unified_processed_example"),
        model_name="gpt-5-mini-2025-08-07",
        generator_dataset_label="unit",
        output_dir=tmp_path / "defaults",
    )
    assert cfg.eval_mode == DEFAULT_EVAL_MODE
    assert cfg.seed == DEFAULT_EVAL_SEED
    assert cfg.temperature == DEFAULT_EVAL_TEMPERATURE
    assert cfg.max_tokens == DEFAULT_EVAL_MAX_TOKENS
    assert cfg.save_interval == DEFAULT_EVAL_SAVE_INTERVAL


def test_build_matrix_configs_defaults_use_shared_constants(tmp_path):
    cfg = build_matrix_configs(
        model="gpt-5-mini-2025-08-07",
        dataset_path=Path("datasets/augmented/unified_processed_example"),
        generator_dataset_label="unit",
        dataset_types=["mmlu_pro"],
        distractor_sources=["scratch"],
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
            "gpt-5-mini-2025-08-07",
            "--dataset-path",
            "datasets/augmented/unified_processed_example",
            "--generator-dataset-label",
            "unit",
        ]
    )
    assert args.preset == DEFAULT_MATRIX_PRESET
    assert args.eval_mode == DEFAULT_EVAL_MODE
    assert args.seed == DEFAULT_EVAL_SEED
    assert args.temperature == DEFAULT_EVAL_TEMPERATURE
    assert args.max_tokens == DEFAULT_EVAL_MAX_TOKENS
    assert args.save_interval == DEFAULT_EVAL_SAVE_INTERVAL

    run_args = parser.parse_args(
        [
            "run",
            "--manifest",
            "manifest.json",
            "--generator-dataset-label",
            "unit",
        ]
    )
    assert run_args.keep_checkpoints == DEFAULT_EVAL_KEEP_CHECKPOINTS
