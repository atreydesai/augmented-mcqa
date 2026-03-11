from pathlib import Path

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ModelOutput
from inspect_ai.scorer import Score, scorer
from inspect_ai.solver import TaskState
from inspect_ai.solver import solver as inspect_solver

from datasets import Dataset, DatasetDict

from data.final5_store import build_evaluation_dataset, build_generation_dataset, materialize_augmented_dataset
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


def _processed_dataset_three_splits(path: Path):
    ds = DatasetDict(
        {
            "arc_challenge": Dataset.from_list(
                [
                    {"id": "arc-1", "question": "ARC 1", "answer": "Gold ARC 1", "choices_human": ["A1", "A2", "A3"]},
                    {"id": "arc-2", "question": "ARC 2", "answer": "Gold ARC 2", "choices_human": ["A4", "A5", "A6"]},
                ]
            ),
            "mmlu_pro": Dataset.from_list(
                [
                    {"question_id": 101, "question": "MMLU 1", "answer": "Gold MMLU 1", "choices_human": ["M1", "M2", "M3"]},
                    {"question_id": 102, "question": "MMLU 2", "answer": "Gold MMLU 2", "choices_human": ["M4", "M5", "M6"]},
                ]
            ),
            "gpqa": Dataset.from_list(
                [
                    {"id": "gpqa-1", "question": "GPQA 1", "answer": "Gold GPQA 1", "choices_human": ["G1", "G2", "G3"]},
                    {"id": "gpqa-2", "question": "GPQA 2", "answer": "Gold GPQA 2", "choices_human": ["G4", "G5", "G6"]},
                ]
            ),
        }
    )
    ds.save_to_disk(str(path))


@inspect_solver
def _noop_solver():
    async def solve(state, generate):  # noqa: ANN001
        state.output.completion = "ok"
        return state

    return solve


@scorer(metrics=[])
def _generation_payload_scorer():
    async def score(state, target):  # noqa: ANN001
        return Score(value=1.0, metadata=dict(state.metadata["generation_payload"]))

    return score


def _write_generation_log(root: Path, samples: list[Sample]):
    inspect_eval(
        Task(
            name="final5_generate_test",
            dataset=MemoryDataset(samples),
            solver=_noop_solver(),
            scorer=_generation_payload_scorer(),
            metadata={"kind": "generation"},
        ),
        log_dir=str(root),
        display="none",
    )


def test_parse_labeled_distractors_accepts_exact_plain_text_labels():
    parsed = parse_labeled_distractors("B. One\nC. Two\nD. Three", ["B", "C", "D"], forbidden=["Gold"])
    assert parsed == ["One", "Two", "Three"]


def test_parse_labeled_distractors_accepts_exact_json_labels():
    payload = '{"B": "One", "C": "Two", "D": "Three"}'
    parsed = parse_labeled_distractors(payload, ["B", "C", "D"], forbidden=["Gold"])
    assert parsed == ["One", "Two", "Three"]


def test_parse_labeled_distractors_recovers_last_valid_json_object_from_messy_response():
    payload = """{
  "B": "broken start",
  "C": "

Let me redo that properly.

```json
{
  "B": "still broken",
  "C": "also broken",
  "D": "

I need to start over cleanly.

{
  "B": "One",
  "C": "Two",
  "D": "Three"
}
"""
    parsed = parse_labeled_distractors(payload, ["B", "C", "D"], forbidden=["Gold"])
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


def test_parse_labeled_distractors_rejects_missing_json_keys():
    try:
        parse_labeled_distractors('{"B": "One", "D": "Three"}', ["B", "C", "D"], forbidden=["Gold"])
    except LabeledParseError as exc:
        assert "Missing distractor keys: C" in str(exc)
    else:
        raise AssertionError("Expected parser failure")


def test_build_generation_dataset_flattens_processed_rows_with_stable_ids(tmp_path):
    path = tmp_path / "processed"
    _processed_dataset(path)

    dataset = build_generation_dataset(path)
    assert len(dataset) == 2
    assert dataset[0].id == "arc_challenge:arc-1"
    assert dataset[1].metadata["choices_human"] == ["A", "B", "C"]


def test_build_generation_dataset_limit_applies_per_dataset_split(tmp_path):
    path = tmp_path / "processed"
    _processed_dataset_three_splits(path)

    dataset = build_generation_dataset(path, limit=1)

    assert len(dataset) == 3
    assert [sample.id for sample in dataset] == [
        "arc_challenge:arc-1",
        "mmlu_pro:101",
        "gpqa:gpqa-1",
    ]


def test_materialized_augmented_cache_only_keeps_rows_present_in_generation_logs(tmp_path):
    processed_path = tmp_path / "processed"
    log_dir = tmp_path / "logs"
    cache_path = tmp_path / "augmented"
    _processed_dataset_three_splits(processed_path)

    samples = [
        Sample(
            input="ARC 1",
            target="",
            id="arc_challenge:arc-1",
            metadata={
                "generation_payload": {
                    "status": "success",
                    "human_from_scratch": ["A1", "A2", "A3"],
                    "human_from_scratch_options_randomized": ["Gold ARC 1", "A1", "A2", "A3"],
                    "human_from_scratch_correct_answer_letter": "A",
                }
            },
        ),
        Sample(
            input="MMLU 1",
            target="",
            id="mmlu_pro:101",
            metadata={
                "generation_payload": {
                    "status": "success",
                    "human_from_scratch": ["M1", "M2", "M3"],
                    "human_from_scratch_options_randomized": ["Gold MMLU 1", "M1", "M2", "M3"],
                    "human_from_scratch_correct_answer_letter": "A",
                }
            },
        ),
        Sample(
            input="GPQA 1",
            target="",
            id="gpqa:gpqa-1",
            metadata={
                "generation_payload": {
                    "status": "error",
                    "error": "test failure",
                    "human_from_scratch": ["G1", "G2", "G3"],
                    "human_from_scratch_options_randomized": ["Gold GPQA 1", "G1", "G2", "G3"],
                    "human_from_scratch_correct_answer_letter": "A",
                }
            },
        ),
    ]
    _write_generation_log(log_dir, samples)

    materialize_augmented_dataset(processed_path, log_dir, cache_path)
    dataset = DatasetDict.load_from_disk(str(cache_path))

    assert len(dataset["arc_challenge"]) == 1
    assert len(dataset["mmlu_pro"]) == 1
    assert len(dataset["gpqa"]) == 1
    assert dataset["arc_challenge"][0]["sample_id"] == "arc_challenge:arc-1"
    assert dataset["mmlu_pro"][0]["sample_id"] == "mmlu_pro:101"
    assert dataset["gpqa"][0]["sample_id"] == "gpqa:gpqa-1"


def test_build_evaluation_dataset_limit_applies_per_dataset_split(tmp_path):
    path = tmp_path / "augmented"
    dataset = DatasetDict(
        {
            "arc_challenge": Dataset.from_list(
                [
                    {
                        "id": "arc-1",
                        "sample_id": "arc_challenge:arc-1",
                        "row_index": 0,
                        "question": "ARC 1",
                        "answer": "Gold ARC 1",
                        "category": "",
                        "human_from_scratch": ["A1", "A2", "A3"],
                        "human_from_scratch_options_randomized": ["Gold ARC 1", "A1", "A2", "A3"],
                        "human_from_scratch_correct_answer_letter": "A",
                    },
                    {
                        "id": "arc-2",
                        "sample_id": "arc_challenge:arc-2",
                        "row_index": 1,
                        "question": "ARC 2",
                        "answer": "Gold ARC 2",
                        "category": "",
                        "human_from_scratch": ["A4", "A5", "A6"],
                        "human_from_scratch_options_randomized": ["Gold ARC 2", "A4", "A5", "A6"],
                        "human_from_scratch_correct_answer_letter": "A",
                    },
                ]
            ),
            "mmlu_pro": Dataset.from_list(
                [
                    {
                        "question_id": 101,
                        "sample_id": "mmlu_pro:101",
                        "row_index": 0,
                        "question": "MMLU 1",
                        "answer": "Gold MMLU 1",
                        "category": "",
                        "human_from_scratch": ["M1", "M2", "M3"],
                        "human_from_scratch_options_randomized": ["Gold MMLU 1", "M1", "M2", "M3"],
                        "human_from_scratch_correct_answer_letter": "A",
                    },
                    {
                        "question_id": 102,
                        "sample_id": "mmlu_pro:102",
                        "row_index": 1,
                        "question": "MMLU 2",
                        "answer": "Gold MMLU 2",
                        "category": "",
                        "human_from_scratch": ["M4", "M5", "M6"],
                        "human_from_scratch_options_randomized": ["Gold MMLU 2", "M4", "M5", "M6"],
                        "human_from_scratch_correct_answer_letter": "A",
                    },
                ]
            ),
            "gpqa": Dataset.from_list(
                [
                    {
                        "id": "gpqa-1",
                        "sample_id": "gpqa:gpqa-1",
                        "row_index": 0,
                        "question": "GPQA 1",
                        "answer": "Gold GPQA 1",
                        "category": "",
                        "human_from_scratch": ["G1", "G2", "G3"],
                        "human_from_scratch_options_randomized": ["Gold GPQA 1", "G1", "G2", "G3"],
                        "human_from_scratch_correct_answer_letter": "A",
                    },
                    {
                        "id": "gpqa-2",
                        "sample_id": "gpqa:gpqa-2",
                        "row_index": 1,
                        "question": "GPQA 2",
                        "answer": "Gold GPQA 2",
                        "category": "",
                        "human_from_scratch": ["G4", "G5", "G6"],
                        "human_from_scratch_options_randomized": ["Gold GPQA 2", "G4", "G5", "G6"],
                        "human_from_scratch_correct_answer_letter": "A",
                    },
                ]
            ),
        }
    )
    dataset.save_to_disk(str(path))

    eval_dataset = build_evaluation_dataset(path, setting="human_from_scratch", mode="full_question", limit=1)

    assert len(eval_dataset) == 3
    assert [sample.id for sample in eval_dataset] == [
        "arc_challenge:arc-1",
        "mmlu_pro:101",
        "gpqa:gpqa-1",
    ]


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
