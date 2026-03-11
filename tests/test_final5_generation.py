from inspect_ai.model import ChatMessageUser, ModelOutput
from inspect_ai.solver import TaskState

from datasets import Dataset, DatasetDict

from data.final5_store import build_generation_dataset
from solvers.final5_generation import _fresh_state
from utils.parsing import LabeledParseError, parse_labeled_distractors


def _processed_dataset(path):
    ds = DatasetDict(
        {
            "arc_challenge": Dataset.from_list(
                [
                    {
                        "id": "arc-1",
                        "question": "Q1",
                        "answer": "Gold 1",
                        "choices_human": ["H1", "H2", "H3"],
                        "category": "science",
                    },
                    {
                        "id": "arc-2",
                        "question": "Q2",
                        "answer": "Gold 2",
                        "choices_human": ["A", "B", "C"],
                        "category": "science",
                    },
                ]
            ),
            "mmlu_pro": Dataset.from_list([]),
            "gpqa": Dataset.from_list([]),
        }
    )
    ds.save_to_disk(str(path))


def test_parse_labeled_distractors_accepts_exact_plain_text_labels():
    parsed = parse_labeled_distractors("B. One\nC. Two\nD. Three", ["B", "C", "D"], forbidden=["Gold"])
    assert parsed == ["One", "Two", "Three"]


def test_parse_labeled_distractors_rejects_extra_or_mislabeled_lines():
    try:
        parse_labeled_distractors("B. One\nD. Two\nE. Three", ["B", "C", "D"])
    except LabeledParseError as exc:
        assert "Expected label C" in str(exc)
    else:
        raise AssertionError("Expected strict label validation error")


def test_parse_labeled_distractors_rejects_duplicates_and_forbidden_answers():
    for payload in ("B. Same\nC. Same\nD. Different", "B. Gold\nC. Two\nD. Three"):
        try:
            parse_labeled_distractors(payload, ["B", "C", "D"], forbidden=["Gold"])
        except LabeledParseError:
            pass
        else:
            raise AssertionError("Expected parser failure")


def test_build_generation_dataset_flattens_processed_rows_with_stable_ids(tmp_path):
    path = tmp_path / "processed"
    _processed_dataset(path)

    dataset = build_generation_dataset(path)
    assert len(dataset) == 2
    assert dataset[0].id == "arc_challenge:arc-1"
    assert dataset[1].metadata["choices_human"] == ["A", "B", "C"]


def test_fresh_state_clones_task_state_without_model_copy():
    state = TaskState(
        model="openai/test",
        sample_id="sample-1",
        epoch=1,
        input="Original prompt",
        messages=[ChatMessageUser(content="Original prompt")],
        output=ModelOutput(model="openai/test"),
        metadata={"sample_id": "sample-1"},
        store={},
    )
    state.output.completion = "previous output"

    fresh = _fresh_state(state, "New prompt")

    assert fresh is not state
    assert fresh.user_prompt.text == "New prompt"
    assert fresh.output.completion == ""
    assert state.user_prompt.text == "Original prompt"
    assert state.output.completion == "previous output"
