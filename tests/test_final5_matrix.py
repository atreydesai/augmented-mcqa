from datasets import Dataset, DatasetDict

from tasks.evaluation import build_evaluation_tasks


def _augmented_dataset(path):
    row = {
        "id": "row-1",
        "question": "Q1",
        "answer": "Gold",
        "category": "cat",
        "human_from_scratch": ["H1", "H2", "H3"],
        "human_from_scratch_options_randomized": ["Gold", "H1", "H2", "H3"],
        "human_from_scratch_correct_answer_letter": "A",
        "model_from_scratch": ["M1", "M2", "M3"],
        "model_from_scratch_options_randomized": ["M1", "Gold", "M2", "M3"],
        "model_from_scratch_correct_answer_letter": "B",
        "augment_human": ["C1", "C2", "C3", "C4", "C5", "C6"],
        "augment_human_options_randomized": ["Gold", "H1", "H2", "H3", "C1", "C2", "C3", "C4", "C5", "C6"],
        "augment_human_correct_answer_letter": "A",
        "augment_model": ["M1", "M2", "M3", "D1", "D2", "D3", "D4", "D5", "D6"],
        "augment_model_options_randomized": ["M1", "Gold", "M2", "M3", "D1", "D2", "D3", "D4", "D5", "D6"],
        "augment_model_correct_answer_letter": "B",
        "augment_ablation": ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"],
        "augment_ablation_options_randomized": ["Gold", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"],
        "augment_ablation_correct_answer_letter": "A",
    }
    DatasetDict(
        {
            "arc_challenge": Dataset.from_list([row]),
            "mmlu_pro": Dataset.from_list([]),
            "gpqa": Dataset.from_list([]),
        }
    ).save_to_disk(str(path))


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
