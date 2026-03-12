from datasets import Dataset, DatasetDict

from data.final5_store import build_evaluation_dataset


def _augmented_dataset(path):
    dataset = DatasetDict(
        {
            "arc_challenge": Dataset.from_list(
                [
                    {
                        "id": "arc-1",
                        "question": "Q1",
                        "answer": "Gold",
                        "category": "cat",
                        "human_from_scratch": ["H1", "H2", "H3"],
                        "human_from_scratch_options_randomized": ["Gold", "H1", "H2", "H3"],
                        "human_from_scratch_correct_answer_letter": "A",
                        "model_from_scratch": [],
                        "model_from_scratch_options_randomized": [],
                        "model_from_scratch_correct_answer_letter": "",
                        "augment_human": [],
                        "augment_human_options_randomized": [],
                        "augment_human_correct_answer_letter": "",
                        "augment_model": [],
                        "augment_model_options_randomized": [],
                        "augment_model_correct_answer_letter": "",
                        "augment_ablation": [],
                        "augment_ablation_options_randomized": [],
                        "augment_ablation_correct_answer_letter": "",
                    },
                    {
                        "id": "arc-2",
                        "question": "Q2",
                        "answer": "Gold2",
                        "category": "cat",
                        "choices_human": ["legacy-1", "legacy-2", "legacy-3"],
                        "human_from_scratch": ["legacy-1", "legacy-2", "legacy-3"],
                        "human_from_scratch_options_randomized": [],
                        "human_from_scratch_correct_answer_letter": "",
                        "model_from_scratch": [],
                        "model_from_scratch_options_randomized": [],
                        "model_from_scratch_correct_answer_letter": "",
                        "augment_human": [],
                        "augment_human_options_randomized": [],
                        "augment_human_correct_answer_letter": "",
                        "augment_model": [],
                        "augment_model_options_randomized": [],
                        "augment_model_correct_answer_letter": "",
                        "augment_ablation": [],
                        "augment_ablation_options_randomized": [],
                        "augment_ablation_correct_answer_letter": "",
                    },
                ]
            ),
            "mmlu_pro": Dataset.from_list([]),
            "gpqa": Dataset.from_list([]),
        }
    )
    dataset.save_to_disk(str(path))


def test_build_evaluation_dataset_requires_new_randomized_columns_without_legacy_fallback(tmp_path):
    path = tmp_path / "augmented"
    _augmented_dataset(path)

    dataset = build_evaluation_dataset(path, setting="human_from_scratch", mode="full_question")
    assert len(dataset) == 1
    assert dataset[0].id == "arc_challenge:arc-1"
