from datasets import Dataset

from data.gpqa_processor import load_gpqa_dataset


def test_load_gpqa_dataset_mapping(monkeypatch):
    raw = Dataset.from_list(
        [
            {
                "Question": "What is H2O?",
                "Correct Answer": "Water",
                "Incorrect Answer 1": "Hydrogen",
                "Incorrect Answer 2": "Oxygen",
                "Incorrect Answer 3": "Helium",
                "Subdomain": "Chemistry",
                "High-level domain": "Science",
                "Record ID": "abc123",
            }
        ]
    )

    monkeypatch.setattr("data.gpqa_processor.load_dataset", lambda *args, **kwargs: raw)

    rows = load_gpqa_dataset(limit=1)
    assert len(rows) == 1
    row = rows[0]
    assert row["question"] == "What is H2O?"
    assert row["answer"] == "Water"
    assert row["choices_answer"] == ["Water"]
    assert row["choices_human"] == ["Hydrogen", "Oxygen", "Helium"]
    assert row["subfield"] == "Chemistry"
    assert row["dataset_type"] == "gpqa"
