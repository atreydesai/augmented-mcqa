from datasets import Dataset

from data.pipeline import _exclude_known_bad_questions


def test_exclude_known_bad_questions_uses_dataset_specific_ids():
    gpqa = Dataset.from_list(
        [
            {"id": "keep-me", "question": "ok"},
            {"id": "recDjE01bu72pPUU2", "question": "bad gpqa 1"},
            {"id": "recZSGUkn56v9kEp1", "question": "bad gpqa 2"},
            {"id": "recnGEpF1srQpaqWq", "question": "bad gpqa 3"},
            {"id": "recnTTKdBzfuoZ7w7", "question": "bad gpqa"},
        ]
    )
    mmlu_pro = Dataset.from_list(
        [
            {"question_id": 995, "question": "ok"},
            {"question_id": 996, "question": "bad mmlu"},
        ]
    )

    filtered_gpqa = _exclude_known_bad_questions(gpqa, "gpqa")
    filtered_mmlu = _exclude_known_bad_questions(mmlu_pro, "mmlu_pro")

    assert filtered_gpqa.num_rows == 1
    assert filtered_gpqa[0]["id"] == "keep-me"

    assert filtered_mmlu.num_rows == 1
    assert filtered_mmlu[0]["question_id"] == 995
