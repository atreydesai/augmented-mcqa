import json

from datasets import Dataset

from data.store import build_evaluation_dataset
from utils.constants import AUGMENTED_STORE_MANIFEST, AUGMENTED_STORE_SCHEMA_VERSION


def _augmented_dataset(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / AUGMENTED_STORE_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": AUGMENTED_STORE_SCHEMA_VERSION,
                "storage_kind": "setting_records",
                "dataset_types": ["arc_challenge", "mmlu_pro", "gpqa"],
                "settings": ["human_from_scratch"],
            }
        ),
        encoding="utf-8",
    )
    Dataset.from_list(
        [
            {
                "id": "arc-1",
                "dataset_type": "arc_challenge",
                "sample_id": "arc_challenge:arc-1",
                "row_index": 0,
                "question": "Q1",
                "answer": "Gold",
                "category": "cat",
                "human_distractors": ["H1", "H2", "H3"],
                "model_distractors": [],
                "options_randomized": ["Gold", "H1", "H2", "H3"],
                "correct_answer_letter": "A",
            },
            {
                "id": "arc-2",
                "dataset_type": "arc_challenge",
                "sample_id": "arc_challenge:arc-2",
                "row_index": 1,
                "question": "Q2",
                "answer": "Gold2",
                "category": "cat",
                "human_distractors": ["legacy-1", "legacy-2", "legacy-3"],
                "model_distractors": [],
                "options_randomized": [],
                "correct_answer_letter": "",
            },
        ]
    ).save_to_disk(str(path / "arc_challenge" / "human_from_scratch"))


def test_build_evaluation_dataset_requires_new_randomized_columns_without_legacy_fallback(tmp_path):
    path = tmp_path / "augmented"
    _augmented_dataset(path)

    dataset = build_evaluation_dataset(path, setting="human_from_scratch", mode="full_question")
    assert len(dataset) == 1
    assert dataset[0].id == "arc_challenge:arc-1"
