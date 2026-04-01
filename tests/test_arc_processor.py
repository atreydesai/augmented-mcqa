from datasets import Dataset

from data.arc_processor import load_arc_dataset


def test_load_arc_dataset_skips_invalid_answer_keys(monkeypatch):
    raw = Dataset.from_list(
        [
            {
                "id": "valid",
                "question": "Valid question",
                "choices": {"text": ["A1", "A2", "A3", "A4"], "label": ["A", "B", "C", "D"]},
                "answerKey": "B",
            },
            {
                "id": "invalid",
                "question": "Invalid question",
                "choices": {"text": ["B1", "B2", "B3", "B4"], "label": ["A", "B", "C", "D"]},
                "answerKey": "0",
            },
        ]
    )

    monkeypatch.setattr("data.arc_processor.load_dataset", lambda *args, **kwargs: raw)

    rows = load_arc_dataset()

    assert len(rows) == 1
    assert rows[0]["id"] == "valid"
    assert rows[0]["answer"] == "A2"
    assert rows[0]["choices_answer"] == ["A2"]


def test_load_arc_dataset_respects_zero_limit(monkeypatch):
    raw = Dataset.from_list(
        [
            {
                "id": "row-1",
                "question": "Question",
                "choices": {"text": ["A1", "A2", "A3", "A4"], "label": ["A", "B", "C", "D"]},
                "answerKey": "A",
            }
        ]
    )

    monkeypatch.setattr("data.arc_processor.load_dataset", lambda *args, **kwargs: raw)

    assert load_arc_dataset(limit=0) == []
