from pathlib import Path

from experiments.matrix import build_matrix_configs


def test_final5_matrix_has_five_settings_per_dataset(tmp_path):
    configs = build_matrix_configs(
        model="Qwen/Qwen3-4B-Instruct-2507",
        dataset_path=Path("datasets/augmented/final5"),
        generator_dataset_label="gpt-5.2-2025-12-11",
        dataset_types=["arc_challenge", "mmlu_pro", "gpqa"],
        output_base=tmp_path,
        choices_only=False,
    )

    # 3 datasets * 5 settings
    assert len(configs) == 15

    setting_ids = sorted({cfg.setting_id for cfg in configs})
    assert setting_ids == [
        "augment_ablation",
        "augment_human",
        "augment_model",
        "human_from_scratch",
        "model_from_scratch",
    ]

    sample = next(c for c in configs if c.setting_id == "augment_model")
    assert sample.num_human == 0
    assert sample.num_model == 9
