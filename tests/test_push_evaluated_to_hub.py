from __future__ import annotations

from pathlib import Path

from datasets import Dataset

from data.push_evaluated_to_hub import build_subset_dataset
from utils.constants import SETTING_NAMES


def _write_leaf(root: Path, dataset_type: str, setting: str, mode: str, sample_id: str) -> None:
    path = root / dataset_type / setting / mode
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(
        [
            {
                "id": sample_id,
                "sample_id": sample_id,
                "dataset_type": dataset_type,
                "setting": setting,
            }
        ]
    ).save_to_disk(str(path))


def test_build_subset_dataset_splits_modes_at_repo_level(tmp_path):
    eval_root = tmp_path / "vllm_Qwen_Qwen3-4B-Instruct-2507"
    eval_root.mkdir(parents=True)
    (eval_root / "evaluated_manifest.json").write_text("{}", encoding="utf-8")

    _write_leaf(eval_root, "arc_challenge", "human_from_scratch", "full_question", "arc-full")
    _write_leaf(eval_root, "gpqa", "human_from_scratch", "full_question", "gpqa-full")
    _write_leaf(eval_root, "arc_challenge", "human_from_scratch", "choices_only", "arc-choices")
    _write_leaf(eval_root, "arc_challenge", "augment_ablation", "choices_only", "ablation-choices")
    _write_leaf(eval_root, "arc_challenge", "model_from_scratch", "full_question", "model-full")

    full_question = build_subset_dataset(eval_root, "full_question")
    choices_only = build_subset_dataset(eval_root, "choices_only")

    assert list(full_question) == [setting for setting in SETTING_NAMES if setting in {"human_from_scratch", "model_from_scratch"}]
    assert full_question["human_from_scratch"].num_rows == 2
    assert full_question["model_from_scratch"].num_rows == 1
    assert "mode" not in full_question["human_from_scratch"].column_names
    assert set(full_question["human_from_scratch"]["sample_id"]) == {"arc-full", "gpqa-full"}

    assert list(choices_only) == [setting for setting in SETTING_NAMES if setting in {"human_from_scratch", "augment_ablation"}]
    assert choices_only["human_from_scratch"].num_rows == 1
    assert choices_only["augment_ablation"].num_rows == 1
    assert set(choices_only["human_from_scratch"]["sample_id"]) == {"arc-choices"}
