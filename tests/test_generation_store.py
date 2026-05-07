import asyncio
from dataclasses import replace
import json
from pathlib import Path
import os
from types import SimpleNamespace

import pytest
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ModelOutput
from inspect_ai.scorer import Score, scorer
from inspect_ai.solver import TaskState
from inspect_ai.solver import solver as inspect_solver

from datasets import Dataset, DatasetDict, load_from_disk

from data.store import _generation_payloads, _load_setting_dataset, build_evaluation_dataset, build_generation_dataset, materialize_augmented_dataset
from data.store import ensure_augmented_dataset
from solvers.generation import _fresh_state, generation_solver
from utils.constants import AUGMENTED_STORE_MANIFEST, AUGMENTED_STORE_SCHEMA_VERSION
from utils.parsing import LabeledParseError, parse_distractors
from utils.recipes import get_setting_recipe


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
            name="augmented_mcqa_generate_test",
            dataset=MemoryDataset(samples),
            solver=_noop_solver(),
            scorer=_generation_payload_scorer(),
            metadata={"kind": "generation"},
        ),
        log_dir=str(root),
        display="none",
    )


def _setting_sample_ids(path: Path, dataset_type: str, setting: str = "human_from_scratch") -> list[str]:
    return [row["sample_id"] for row in _load_setting_dataset(path, dataset_type, setting)]


def test_build_generation_dataset_accepts_dataset_manifest_json(tmp_path):
    source_path = tmp_path / "custom.jsonl"
    source_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "qid": "custom-1",
                        "prompt": "Custom question 1",
                        "gold": "Custom gold 1",
                        "human": ["d1", "d2", "d3"],
                        "group": "bio",
                    }
                ),
                json.dumps(
                    {
                        "qid": "custom-2",
                        "prompt": "Custom question 2",
                        "gold": "Custom gold 2",
                        "human": ["e1", "e2", "e3"],
                        "group": "chem",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "datasets_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "augmented_mcqa_dataset_manifest_v1",
                "datasets": {
                    "custom_benchmark": {
                        "path": str(source_path),
                        "format": "jsonl",
                        "question_key": "prompt",
                        "answer_key": "gold",
                        "choices_human_key": "human",
                        "category_key": "group",
                        "question_id_key": "qid",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    dataset = build_generation_dataset(manifest_path, dataset_types=["custom_benchmark"], limit=1)

    assert len(dataset) == 1
    assert dataset[0].id == "custom_benchmark:custom-1"
    assert dataset[0].metadata["category"] == "bio"


def test_generate_qa_prompt_uses_single_distractors_list_contract():
    prompt = (Path("prompts") / "generate_qa.txt").read_text(encoding="utf-8")
    assert 'exactly one key: "distractors"' in prompt
    assert '"distractors"' in prompt
    assert "JSON keys" not in prompt
    assert "forbidden value" not in prompt


def test_generate_conditioned_prompt_uses_choices_block_and_single_distractors_list_contract():
    prompt = (Path("prompts") / "generate_conditioned.txt").read_text(encoding="utf-8")
    assert "{old_count}" in prompt
    assert "{choices}" in prompt
    assert 'exactly one key: "distractors"' in prompt
    assert "forbidden value" not in prompt


def test_parse_distractors_accepts_exact_json_list():
    payload = '{"distractors": ["One", "Two", "Three"]}'
    parsed = parse_distractors(payload, 3, forbidden=["Gold"])
    assert parsed == ["One", "Two", "Three"]


def test_parse_distractors_recovers_last_valid_json_object_from_messy_response():
    payload = """{
  "distractors": "

Let me redo that properly.

```json
{
  "distractors": "still broken"

I need to start over cleanly.

{
  "distractors": ["One", "Two", "Three"]
}
"""
    parsed = parse_distractors(payload, 3, forbidden=["Gold"])
    assert parsed == ["One", "Two", "Three"]


@pytest.mark.parametrize(
    ("payload", "count", "forbidden", "message"),
    [
        ('{"wrong_key": ["One", "Two", "Three"]}', 3, None, 'Missing required key: "distractors"'),
        ('{"distractors": ["One", "Two", "Three"], "extra": 1}', 3, None, "Unexpected distractor keys: extra"),
        ('{"distractors": "One"}', 1, None, 'Expected "distractors" to be a list'),
        ('{"distractors": ["One", "Two"]}', 3, None, "Expected 3 distractors, got 2"),
        ('{"distractors": ["One", "   ", "Three"]}', 3, None, "Distractor 2 is empty"),
        ('{"distractors": ["Same", "Same", "Different"]}', 3, ["Gold"], "Duplicate distractor at position 2"),
        ('{"distractors": ["Gold", "Two", "Three"]}', 3, ["Gold"], "Forbidden distractor at position 1"),
        ('{"distractors": ["One", 2, "Three"]}', 3, None, "Distractor 2 must be a string"),
    ],
)
def test_parse_distractors_rejects_invalid_payloads(payload, count, forbidden, message):
    with pytest.raises(LabeledParseError) as exc_info:
        parse_distractors(payload, count, forbidden=forbidden)

    assert message in str(exc_info.value)


def test_generation_solver_preserves_failed_parse_attempts_in_traces():
    state = TaskState(
        model="together/Qwen/Qwen3.5-397B-A17B",
        sample_id="arc_challenge:arc-1",
        epoch=1,
        input="Q1",
        messages=[ChatMessageUser(content="Q1")],
        output=ModelOutput(model="together/Qwen/Qwen3.5-397B-A17B"),
        metadata={
            "sample_id": "arc_challenge:arc-1",
            "dataset_type": "arc_challenge",
            "row_index": 0,
            "question": "Q1",
            "answer": "Gold 1",
            "choices_human": ["H1", "H2", "H3"],
            "category": "science",
        },
        store={},
    )

    async def fake_generate(current_state: TaskState) -> TaskState:
        current_state.output.completion = "not valid json"
        return current_state

    solved = asyncio.run(generation_solver("model_from_scratch")(state, fake_generate))

    generation = solved.metadata["generation"]
    assert generation["status"] == "error"
    assert "Response does not contain a valid JSON object" in generation["error"]
    assert generation["traces"]["model_from_scratch"]["output"] == "not valid json"
    assert generation["traces"]["model_from_scratch"]["attempts"][-1]["output"] == "not valid json"


def test_generation_solver_fails_fast_when_recipe_prompt_template_is_missing(monkeypatch):
    human_recipe = get_setting_recipe("human_from_scratch")
    broken_recipe = replace(get_setting_recipe("model_from_scratch"), prompt_template=None)

    def fake_get_setting_recipe(name: str):
        if name == "human_from_scratch":
            return human_recipe
        if name == "model_from_scratch":
            return broken_recipe
        raise AssertionError(f"unexpected recipe lookup: {name}")

    monkeypatch.setattr("solvers.generation.get_setting_recipe", fake_get_setting_recipe)

    state = TaskState(
        model="together/Qwen/Qwen3.5-397B-A17B",
        sample_id="arc_challenge:arc-1",
        epoch=1,
        input="Q1",
        messages=[ChatMessageUser(content="Q1")],
        output=ModelOutput(model="together/Qwen/Qwen3.5-397B-A17B"),
        metadata={
            "sample_id": "arc_challenge:arc-1",
            "dataset_type": "arc_challenge",
            "row_index": 0,
            "question": "Q1",
            "answer": "Gold 1",
            "choices_human": ["H1", "H2", "H3"],
            "category": "science",
        },
        store={},
    )

    async def fake_generate(current_state: TaskState) -> TaskState:
        raise AssertionError("generation should not run without a prompt template")

    solved = asyncio.run(generation_solver("model_from_scratch")(state, fake_generate))

    generation = solved.metadata["generation"]
    assert generation["status"] == "error"
    assert generation["error"] == "Generation recipe 'model_from_scratch' is missing a prompt template"


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


def test_build_generation_dataset_augment_model_skips_rows_missing_prerequisites(tmp_path):
    processed_path = tmp_path / "processed"
    log_dir = tmp_path / "logs"
    _processed_dataset(processed_path)

    samples = [
        Sample(
            input="Q1",
            target="",
            id="arc_challenge:arc-1",
            metadata={
                "generation_payload": {
                    "status": "success",
                    "sample_id": "arc_challenge:arc-1",
                    "dataset_type": "arc_challenge",
                    "row_index": 0,
                    "question": "Q1",
                    "answer": "Gold 1",
                    "human_from_scratch": ["H1", "H2", "H3"],
                    "human_from_scratch_options_randomized": ["Gold 1", "H1", "H2", "H3"],
                    "human_from_scratch_correct_answer_letter": "A",
                    "model_from_scratch": ["B1", "C1", "D1"],
                    "model_from_scratch_options_randomized": ["Gold 1", "B1", "C1", "D1"],
                    "model_from_scratch_correct_answer_letter": "A",
                }
            },
        ),
        Sample(
            input="Q2",
            target="",
            id="arc_challenge:arc-2",
            metadata={
                "generation_payload": {
                    "status": "error",
                    "sample_id": "arc_challenge:arc-2",
                    "dataset_type": "arc_challenge",
                    "row_index": 1,
                    "question": "Q2",
                    "answer": "Gold 2",
                    "error": "missing labeled output",
                }
            },
        ),
    ]
    _write_generation_log(log_dir, samples)

    dataset = build_generation_dataset(
        processed_path,
        strategy="augment_model",
        dataset_types=["arc_challenge"],
        generation_log_dir=log_dir,
    )

    assert len(dataset) == 1
    assert dataset[0].id == "arc_challenge:arc-1"
    assert dataset[0].metadata["existing_model_from_scratch"] == ["B1", "C1", "D1"]


def test_generation_payloads_prefers_augmented_generation_score_when_multiple_scores(monkeypatch):
    preferred = SimpleNamespace(
        metadata={
            "status": "success",
            "sample_id": "arc_challenge:arc-1",
            "dataset_type": "arc_challenge",
            "row_index": 0,
            "question": "Q1",
            "answer": "Gold 1",
            "human_from_scratch": ["H1", "H2", "H3"],
            "human_from_scratch_options_randomized": ["Gold 1", "H1", "H2", "H3"],
            "human_from_scratch_correct_answer_letter": "A",
        }
    )
    other = SimpleNamespace(
        metadata={
            "status": "error",
            "sample_id": "wrong",
            "dataset_type": "wrong",
            "row_index": 9,
            "question": "Wrong",
            "answer": "Wrong",
        }
    )
    log = SimpleNamespace(
        samples=[
            SimpleNamespace(
                id="arc_challenge:arc-1",
                scores={
                    "other_metric": other,
                    "augmented_mcqa_generation": preferred,
                },
            )
        ]
    )

    monkeypatch.setattr("data.store.iter_eval_logs", lambda *args, **kwargs: [("sample.eval", log)])

    payloads = _generation_payloads("logs")

    assert payloads["arc_challenge:arc-1"]["dataset_type"] == "arc_challenge"
    assert payloads["arc_challenge:arc-1"]["answer"] == "Gold 1"
    assert payloads["arc_challenge:arc-1"]["human_from_scratch"] == ["H1", "H2", "H3"]


def test_build_generation_dataset_augment_model_uses_materialized_cache_prerequisites(tmp_path):
    processed_path = tmp_path / "processed"
    cache_path = tmp_path / "augmented"
    _processed_dataset(processed_path)

    cache_path.mkdir(parents=True, exist_ok=True)
    (cache_path / AUGMENTED_STORE_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": AUGMENTED_STORE_SCHEMA_VERSION,
                "storage_kind": "setting_records",
                "dataset_types": ["arc_challenge"],
                "settings": [
                    "human_from_scratch",
                    "model_from_scratch",
                    "augment_human",
                    "augment_model",
                    "augment_ablation",
                ],
            }
        ),
        encoding="utf-8",
    )
    Dataset.from_list(
        [
            {
                "id": "arc-1",
                "question_id": None,
                "dataset_type": "arc_challenge",
                "row_index": 0,
                "sample_id": "arc_challenge:arc-1",
                "question": "Q1",
                "answer": "Gold 1",
                "category": "science",
                "options": ["Gold 1", "B1", "C1", "D1"],
                "answer_index": 0,
                "choices_human": ["H1", "H2", "H3"],
                "setting": "model_from_scratch",
                "generation_strategy": "model_from_scratch",
                "status": "success",
                "num_human": 0,
                "num_model": 3,
                "num_choices": 4,
                "human_distractors": [],
                "model_distractors": ["B1", "C1", "D1"],
                "distractors": ["B1", "C1", "D1"],
                "options_randomized": ["Gold 1", "B1", "C1", "D1"],
                "correct_answer_letter": "A",
                "traces": {},
            }
        ]
    ).save_to_disk(str(cache_path / "arc_challenge" / "model_from_scratch"))

    dataset = build_generation_dataset(
        processed_path,
        strategy="augment_model",
        dataset_types=["arc_challenge"],
        augmented_dataset_path=cache_path,
    )

    assert len(dataset) == 1
    assert dataset[0].id == "arc_challenge:arc-1"
    assert dataset[0].metadata["existing_model_from_scratch"] == ["B1", "C1", "D1"]


def test_build_generation_dataset_rejects_non_manifest_json_processed_input(tmp_path):
    processed_path = tmp_path / "processed.json"
    processed_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="DatasetDict root or a dataset manifest JSON"):
        build_generation_dataset(processed_path)


def test_build_generation_dataset_rejects_manifest_dataset_source_format(tmp_path):
    source_path = tmp_path / "custom"
    Dataset.from_list([{"qid": "custom-1", "prompt": "Q1", "gold": "A", "human": ["d1", "d2", "d3"]}]).save_to_disk(
        str(source_path)
    )
    manifest_path = tmp_path / "datasets_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "augmented_mcqa_dataset_manifest_v1",
                "datasets": {
                    "custom_benchmark": {
                        "path": str(source_path),
                        "format": "dataset",
                        "question_key": "prompt",
                        "answer_key": "gold",
                        "choices_human_key": "human",
                        "question_id_key": "qid",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported dataset manifest format: dataset"):
        build_generation_dataset(manifest_path)


def test_build_generation_dataset_rejects_single_dataset_processed_root(tmp_path):
    processed_path = tmp_path / "processed"
    Dataset.from_list([{"id": "arc-1", "question": "Q1", "answer": "A"}]).save_to_disk(str(processed_path))

    with pytest.raises(TypeError, match="Expected DatasetDict"):
        build_generation_dataset(processed_path)


def test_materialized_augmented_cache_preserves_rows_without_successful_generations(tmp_path):
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
    assert _setting_sample_ids(cache_path, "arc_challenge") == ["arc_challenge:arc-1", "arc_challenge:arc-2"]
    assert _setting_sample_ids(cache_path, "mmlu_pro") == ["mmlu_pro:101", "mmlu_pro:102"]
    assert _setting_sample_ids(cache_path, "gpqa") == ["gpqa:gpqa-1", "gpqa:gpqa-2"]

    gpqa_row = dict(_load_setting_dataset(cache_path, "gpqa", "human_from_scratch")[0])
    assert gpqa_row["sample_id"] == "gpqa:gpqa-1"
    assert gpqa_row["options_randomized"]
    assert gpqa_row["correct_answer_letter"] in {"A", "B", "C", "D"}

    gpqa_missing_row = dict(_load_setting_dataset(cache_path, "gpqa", "human_from_scratch")[1])
    assert gpqa_missing_row["sample_id"] == "gpqa:gpqa-2"
    assert gpqa_missing_row["options_randomized"] == []


def test_ensure_augmented_dataset_refreshes_existing_cache_when_new_shard_logs_arrive(tmp_path):
    processed_path = tmp_path / "processed"
    log_dir = tmp_path / "logs"
    cache_path = tmp_path / "augmented"
    _processed_dataset(processed_path)

    shard0 = [
        Sample(
            input="Q1",
            target="",
            id="arc_challenge:arc-1",
            metadata={
                "generation_payload": {
                    "status": "success",
                    "human_from_scratch": ["H1", "H2", "H3"],
                    "human_from_scratch_options_randomized": ["Gold 1", "H1", "H2", "H3"],
                    "human_from_scratch_correct_answer_letter": "A",
                }
            },
        )
    ]
    _write_generation_log(log_dir, shard0)
    first_log = max(log_dir.glob("*.eval"), key=lambda path: path.stat().st_mtime)
    os.utime(first_log, (1000, 1000))

    ensure_augmented_dataset(processed_path, log_dir, cache_path)
    assert _setting_sample_ids(cache_path, "arc_challenge") == [
        "arc_challenge:arc-1",
        "arc_challenge:arc-2",
    ]

    cache_mtime = max(path.stat().st_mtime for path in cache_path.rglob("*") if path.is_file())

    shard1 = [
        Sample(
            input="Q2",
            target="",
            id="arc_challenge:arc-2",
            metadata={
                "generation_payload": {
                    "status": "success",
                    "human_from_scratch": ["A", "B", "C"],
                    "human_from_scratch_options_randomized": ["Gold 2", "A", "B", "C"],
                    "human_from_scratch_correct_answer_letter": "A",
                }
            },
        )
    ]
    _write_generation_log(log_dir, shard1)
    second_log = max(log_dir.glob("*.eval"), key=lambda path: path.stat().st_mtime)
    os.utime(second_log, (cache_mtime + 10, cache_mtime + 10))

    ensure_augmented_dataset(processed_path, log_dir, cache_path)
    assert _setting_sample_ids(cache_path, "arc_challenge") == [
        "arc_challenge:arc-1",
        "arc_challenge:arc-2",
    ]


def test_ensure_augmented_dataset_rejects_cluster_slice_staging_root(tmp_path):
    processed_path = tmp_path / "processed"
    log_dir = tmp_path / "logs"
    cache_path = tmp_path / "augmented"
    _processed_dataset(processed_path)

    shard0 = [
        Sample(
            input="Q1",
            target="",
            id="arc_challenge:arc-1",
            metadata={
                "generation_payload": {
                    "status": "success",
                    "human_from_scratch": ["H1", "H2", "H3"],
                    "human_from_scratch_options_randomized": ["Gold 1", "H1", "H2", "H3"],
                    "human_from_scratch_correct_answer_letter": "A",
                }
            },
        )
    ]
    _write_generation_log(log_dir, shard0)

    staging_path = cache_path / "_cluster_slices" / "arc_challenge" / "model_from_scratch" / "0-1"
    DatasetDict({"arc_challenge": Dataset.from_list([{"sample_id": "arc_challenge:arc-1"}])}).save_to_disk(str(staging_path))

    with pytest.raises(ValueError, match="unsupported augmented cache layout"):
        ensure_augmented_dataset(processed_path, log_dir, cache_path)

    assert not (cache_path / AUGMENTED_STORE_MANIFEST).exists()
    assert staging_path.exists()


def test_ensure_augmented_dataset_rejects_legacy_cache_without_overwriting(tmp_path):
    processed_path = tmp_path / "processed"
    cache_path = tmp_path / "augmented"
    _processed_dataset(processed_path)

    DatasetDict(
        {
            "arc_challenge": Dataset.from_list(
                [
                    {"sample_id": "arc_challenge:arc-1", "status": "success"},
                    {"sample_id": "arc_challenge:arc-2", "status": "success"},
                ]
            )
        }
    ).save_to_disk(str(cache_path))

    with pytest.raises(ValueError, match="unsupported augmented cache layout"):
        ensure_augmented_dataset(processed_path, tmp_path / "logs", cache_path)

    assert (cache_path / "dataset_dict.json").exists()
    preserved = load_from_disk(str(cache_path))
    assert isinstance(preserved, DatasetDict)
    assert [row["sample_id"] for row in preserved["arc_challenge"]] == [
        "arc_challenge:arc-1",
        "arc_challenge:arc-2",
    ]
    assert not (cache_path / AUGMENTED_STORE_MANIFEST).exists()


def test_build_evaluation_dataset_rejects_unsupported_cache_layout(tmp_path):
    cache_path = tmp_path / "augmented"
    DatasetDict(
        {
            "arc_challenge": Dataset.from_list([{"sample_id": "arc_challenge:arc-1", "status": "success"}]),
            "mmlu_pro": Dataset.from_list([]),
            "gpqa": Dataset.from_list([]),
        }
    ).save_to_disk(str(cache_path))

    with pytest.raises(ValueError, match="unsupported augmented cache layout"):
        build_evaluation_dataset(
            cache_path,
            setting="human_from_scratch",
            mode="full_question",
        )


def test_augmented_cache_rejects_paths_overlapping_processed_dataset(tmp_path):
    processed_path = tmp_path / "processed"
    _processed_dataset(processed_path)

    with pytest.raises(ValueError, match="must not overlap processed dataset path"):
        ensure_augmented_dataset(processed_path, tmp_path / "logs", processed_path)

    with pytest.raises(ValueError, match="must not overlap processed dataset path"):
        materialize_augmented_dataset(
            processed_path,
            tmp_path / "logs",
            processed_path / "nested-output",
        )


def _setting_record(
    *,
    dataset_type: str,
    sample_id: str,
    row_index: int,
    question: str,
    answer: str,
    human_distractors: list[str],
    options_randomized: list[str],
    correct_answer_letter: str,
) -> dict[str, object]:
    return {
        "id": sample_id.split(":")[-1],
        "question_id": None,
        "dataset_type": dataset_type,
        "row_index": row_index,
        "sample_id": sample_id,
        "question": question,
        "answer": answer,
        "category": "",
        "options": [],
        "answer_index": None,
        "choices_human": [],
        "setting": "human_from_scratch",
        "generation_strategy": "human_from_scratch",
        "status": "success",
        "num_human": len(human_distractors),
        "num_model": 0,
        "num_choices": len(options_randomized),
        "human_distractors": list(human_distractors),
        "model_distractors": [],
        "distractors": list(human_distractors),
        "options_randomized": list(options_randomized),
        "correct_answer_letter": correct_answer_letter,
        "traces": {},
    }


def _write_human_from_scratch_store(path: Path, rows_by_dataset: dict[str, list[dict[str, object]]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / AUGMENTED_STORE_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": AUGMENTED_STORE_SCHEMA_VERSION,
                "storage_kind": "setting_records",
                "dataset_types": list(rows_by_dataset.keys()),
                "settings": ["human_from_scratch"],
            }
        ),
        encoding="utf-8",
    )
    for dataset_type, rows in rows_by_dataset.items():
        Dataset.from_list(rows).save_to_disk(str(path / dataset_type / "human_from_scratch"))


def test_build_evaluation_dataset_limit_applies_per_dataset_split(tmp_path):
    path = tmp_path / "augmented"
    _write_human_from_scratch_store(
        path,
        {
            "arc_challenge": [
                _setting_record(
                    dataset_type="arc_challenge",
                    sample_id="arc_challenge:arc-1",
                    row_index=0,
                    question="ARC 1",
                    answer="Gold ARC 1",
                    human_distractors=["A1", "A2", "A3"],
                    options_randomized=["Gold ARC 1", "A1", "A2", "A3"],
                    correct_answer_letter="A",
                ),
                _setting_record(
                    dataset_type="arc_challenge",
                    sample_id="arc_challenge:arc-2",
                    row_index=1,
                    question="ARC 2",
                    answer="Gold ARC 2",
                    human_distractors=["A4", "A5", "A6"],
                    options_randomized=["Gold ARC 2", "A4", "A5", "A6"],
                    correct_answer_letter="A",
                ),
            ],
            "mmlu_pro": [
                _setting_record(
                    dataset_type="mmlu_pro",
                    sample_id="mmlu_pro:101",
                    row_index=0,
                    question="MMLU 1",
                    answer="Gold MMLU 1",
                    human_distractors=["M1", "M2", "M3"],
                    options_randomized=["Gold MMLU 1", "M1", "M2", "M3"],
                    correct_answer_letter="A",
                ),
                _setting_record(
                    dataset_type="mmlu_pro",
                    sample_id="mmlu_pro:102",
                    row_index=1,
                    question="MMLU 2",
                    answer="Gold MMLU 2",
                    human_distractors=["M4", "M5", "M6"],
                    options_randomized=["Gold MMLU 2", "M4", "M5", "M6"],
                    correct_answer_letter="A",
                ),
            ],
            "gpqa": [
                _setting_record(
                    dataset_type="gpqa",
                    sample_id="gpqa:gpqa-1",
                    row_index=0,
                    question="GPQA 1",
                    answer="Gold GPQA 1",
                    human_distractors=["G1", "G2", "G3"],
                    options_randomized=["Gold GPQA 1", "G1", "G2", "G3"],
                    correct_answer_letter="A",
                ),
                _setting_record(
                    dataset_type="gpqa",
                    sample_id="gpqa:gpqa-2",
                    row_index=1,
                    question="GPQA 2",
                    answer="Gold GPQA 2",
                    human_distractors=["G4", "G5", "G6"],
                    options_randomized=["Gold GPQA 2", "G4", "G5", "G6"],
                    correct_answer_letter="A",
                ),
            ],
        },
    )

    eval_dataset = build_evaluation_dataset(path, setting="human_from_scratch", mode="full_question", limit=1)

    assert len(eval_dataset) == 3
    assert [sample.id for sample in eval_dataset] == [
        "arc_challenge:arc-1",
        "mmlu_pro:101",
        "gpqa:gpqa-1",
    ]


def test_build_evaluation_dataset_respects_filtered_question_chunk_bounds(tmp_path):
    path = tmp_path / "augmented"
    _write_human_from_scratch_store(
        path,
        {
            "arc_challenge": [
                _setting_record(
                    dataset_type="arc_challenge",
                    sample_id="arc_challenge:arc-0",
                    row_index=0,
                    question="ARC 0",
                    answer="Gold ARC 0",
                    human_distractors=["A1", "A2", "A3"],
                    options_randomized=["Gold ARC 0", "A1", "A2", "A3"],
                    correct_answer_letter="A",
                ),
                _setting_record(
                    dataset_type="arc_challenge",
                    sample_id="arc_challenge:arc-1",
                    row_index=1,
                    question="ARC 1",
                    answer="Gold ARC 1",
                    human_distractors=["B1", "B2", "B3"],
                    options_randomized=[],
                    correct_answer_letter="",
                ),
                _setting_record(
                    dataset_type="arc_challenge",
                    sample_id="arc_challenge:arc-2",
                    row_index=2,
                    question="ARC 2",
                    answer="Gold ARC 2",
                    human_distractors=["C1", "C2", "C3"],
                    options_randomized=["Gold ARC 2", "C1", "C2", "C3"],
                    correct_answer_letter="A",
                ),
                _setting_record(
                    dataset_type="arc_challenge",
                    sample_id="arc_challenge:arc-3",
                    row_index=3,
                    question="ARC 3",
                    answer="Gold ARC 3",
                    human_distractors=["D1", "D2", "D3"],
                    options_randomized=["Gold ARC 3", "D1", "D2", "D3"],
                    correct_answer_letter="A",
                ),
            ],
            "mmlu_pro": [],
            "gpqa": [],
        },
    )

    eval_dataset = build_evaluation_dataset(
        path,
        setting="human_from_scratch",
        mode="full_question",
        dataset_types=["arc_challenge"],
        question_start=1,
        limit=2,
    )

    assert [sample.id for sample in eval_dataset] == ["arc_challenge:arc-2"]


def test_build_evaluation_dataset_chunks_sparse_filtered_tail_by_position(tmp_path):
    path = tmp_path / "augmented"
    _write_human_from_scratch_store(
        path,
        {
            "arc_challenge": [
                _setting_record(
                    dataset_type="arc_challenge",
                    sample_id="arc_challenge:arc-0",
                    row_index=0,
                    question="ARC 0",
                    answer="Gold ARC 0",
                    human_distractors=["A1", "A2", "A3"],
                    options_randomized=["Gold ARC 0", "A1", "A2", "A3"],
                    correct_answer_letter="A",
                ),
                _setting_record(
                    dataset_type="arc_challenge",
                    sample_id="arc_challenge:arc-tail",
                    row_index=999,
                    question="ARC tail",
                    answer="Gold ARC tail",
                    human_distractors=["T1", "T2", "T3"],
                    options_randomized=["Gold ARC tail", "T1", "T2", "T3"],
                    correct_answer_letter="A",
                ),
            ],
            "mmlu_pro": [],
            "gpqa": [],
        },
    )

    eval_dataset = build_evaluation_dataset(
        path,
        setting="human_from_scratch",
        mode="full_question",
        dataset_types=["arc_challenge"],
        question_start=1,
        limit=1,
    )

    assert [sample.id for sample in eval_dataset] == ["arc_challenge:arc-tail"]
    assert eval_dataset[0].metadata["row_index"] == 999


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
