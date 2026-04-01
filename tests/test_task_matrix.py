import json

from datasets import Dataset

from tasks.evaluation import build_evaluation_tasks
from utils.constants import AUGMENTED_STORE_MANIFEST, AUGMENTED_STORE_SCHEMA_VERSION


def _augmented_dataset(path):
    path.mkdir(parents=True, exist_ok=True)
    settings = ["human_from_scratch", "model_from_scratch", "augment_human", "augment_model", "augment_ablation"]
    (path / AUGMENTED_STORE_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": AUGMENTED_STORE_SCHEMA_VERSION,
                "storage_kind": "setting_records",
                "dataset_types": ["arc_challenge", "mmlu_pro", "gpqa"],
                "settings": settings,
            }
        ),
        encoding="utf-8",
    )

    def setting_row(setting: str, options: list[str], human: list[str], model: list[str], correct: str) -> dict[str, object]:
        return {
            "id": "row-1",
            "question_id": None,
            "dataset_type": "arc_challenge",
            "row_index": 0,
            "sample_id": "arc_challenge:row-1",
            "question": "Q1",
            "answer": "Gold",
            "category": "cat",
            "options": [],
            "answer_index": None,
            "choices_human": [],
            "setting": setting,
            "generation_strategy": setting,
            "status": "success",
            "num_human": len(human),
            "num_model": len(model),
            "num_choices": len(options),
            "human_distractors": human,
            "model_distractors": model,
            "distractors": [*human, *model],
            "options_randomized": options,
            "correct_answer_letter": correct,
            "traces": {},
        }

    rows_by_setting = {
        "human_from_scratch": setting_row("human_from_scratch", ["Gold", "H1", "H2", "H3"], ["H1", "H2", "H3"], [], "A"),
        "model_from_scratch": setting_row("model_from_scratch", ["M1", "Gold", "M2", "M3"], [], ["M1", "M2", "M3"], "B"),
        "augment_human": setting_row(
            "augment_human",
            ["Gold", "H1", "H2", "H3", "C1", "C2", "C3", "C4", "C5", "C6"],
            ["H1", "H2", "H3"],
            ["C1", "C2", "C3", "C4", "C5", "C6"],
            "A",
        ),
        "augment_model": setting_row(
            "augment_model",
            ["M1", "Gold", "M2", "M3", "D1", "D2", "D3", "D4", "D5", "D6"],
            [],
            ["M1", "M2", "M3", "D1", "D2", "D3", "D4", "D5", "D6"],
            "B",
        ),
        "augment_ablation": setting_row(
            "augment_ablation",
            ["Gold", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"],
            [],
            ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"],
            "A",
        ),
    }

    for setting, row in rows_by_setting.items():
        Dataset.from_list([row]).save_to_disk(str(path / "arc_challenge" / setting))


def test_build_evaluation_tasks_creates_five_settings_times_two_modes(tmp_path):
    path = tmp_path / "augmented"
    _augmented_dataset(path)
    tasks = build_evaluation_tasks(
        augmented_dataset_path=path,
        dataset_types=["arc_challenge"],
        settings=["human_from_scratch", "model_from_scratch", "augment_human", "augment_model", "augment_ablation"],
        modes=["full_question", "choices_only"],
        shard_count=1,
        shard_index=0,
        shard_strategy="contiguous",
        limit=None,
        run_name="eval-run",
        generation_run_name="gen-run",
        generation_model="openai/gpt",
        evaluation_model="vllm/qwen",
    )
    assert len(tasks) == 10
