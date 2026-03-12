import json

from datasets import Dataset, DatasetDict
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, scorer
from inspect_ai.solver import solver

from analysis import benchmarker_analysis
from utils.constants import CHOICE_LABELS, FINAL5_SETTINGS, MODE_CHOICES, SETTING_SPECS
from utils.modeling import resolve_model_name, safe_name
from utils.sharding import sample_id_for_row


DATASET_ROWS = {
    "arc_challenge": {"id": "arc-1", "question": "ARC question 1", "answer": "arc gold"},
    "gpqa": {"id": "gpqa-1", "question": "GPQA question 1", "answer": "gpqa gold"},
    "mmlu_pro": {"question_id": 101, "question": "MMLU question 1", "answer": "mmlu gold"},
}

EXPECTED_FIGURES = {
    "fig2a_quality_mean.png",
    "fig2b_p_ge2_flaws.png",
    "fig2c_4choice_vs_10choice.png",
    "fig3a_rule_heatmap.png",
    "fig3b_model_sensitivity.png",
    "fig3c_augmentation_penalty.png",
    "fig4a_accuracy_avg.png",
    "fig4b_accuracy_by_model.png",
    "fig4c_delta_over_random.png",
    "fig5a_validity_vs_accuracy.png",
    "fig5b_validity_vs_accuracy_delta.png",
    "fig5c_p2flaws_vs_accuracy.png",
    "fig6a_within_group_correlation.png",
    "fig7a_mode_accuracy_comparison.png",
    "fig7b_mode_delta.png",
    "fig8a_per_rule_accuracy_corr.png",
    "fig9a_per_question_rule_accuracy_gap.png",
    "fig10a_nonfunctional_distractor_rate.png",
    "fig11a_normalised_accuracy.png",
    "fig11b_normalised_accuracy_delta.png",
}


def _choices(prefix: str, count: int) -> list[str]:
    return [f"{prefix}_{index}" for index in range(count)]


def _wrong_letter(gold_letter: str, choice_count: int, step: int = 1) -> str:
    gold_index = CHOICE_LABELS.index(gold_letter)
    return CHOICE_LABELS[(gold_index + step) % choice_count]


def _build_augmented_cache(path):
    splits: dict[str, Dataset] = {}
    for dataset_index, (dataset_type, base_row) in enumerate(DATASET_ROWS.items()):
        row = dict(base_row)
        for setting_index, setting in enumerate(FINAL5_SETTINGS):
            choice_count = SETTING_SPECS[setting]["num_choices"]
            row[f"{setting}_options_randomized"] = _choices(f"{dataset_type}_{setting}", choice_count)
            row[f"{setting}_correct_answer_letter"] = CHOICE_LABELS[(dataset_index + setting_index) % choice_count]
        splits[dataset_type] = Dataset.from_list([row])

    dataset = DatasetDict(splits)
    dataset.save_to_disk(str(path))
    return path


def _write_writing_flaws(path):
    rows = []
    for dataset_index, (dataset_type, base_row) in enumerate(DATASET_ROWS.items()):
        for setting_index, setting in enumerate(FINAL5_SETTINGS):
            fail_count = (dataset_index + setting_index) % 4
            outcomes = ["fail"] * fail_count + ["pass"] * (19 - fail_count)
            rows.append(
                {
                    "dataset": dataset_type,
                    "config": setting,
                    "question": base_row["question"],
                    "writing_flaw": {
                        "value": round(0.98 - dataset_index * 0.05 - setting_index * 0.03, 3),
                        "answer": str(outcomes),
                    },
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@solver
def _prediction_solver():
    async def solve(state, generate):  # noqa: ANN001
        state.output.completion = str(state.metadata.get("prediction", ""))
        return state

    return solve


@scorer(metrics=[])
def _metadata_scorer():
    async def score(state, target):  # noqa: ANN001
        prediction = str(state.output.completion or "").strip().upper()
        gold_letter = str(target.text or "").strip().upper()
        metadata = {
            "sample_id": state.metadata.get("sample_id"),
            "dataset_type": state.metadata.get("dataset_type"),
            "question_idx": int(state.metadata.get("row_index", -1)),
            "category": state.metadata.get("category", ""),
            "setting": state.metadata.get("setting"),
            "mode": state.metadata.get("mode"),
            "prediction": prediction,
            "prediction_type": state.metadata.get("prediction_type", "?"),
            "gold_answer_letter": gold_letter,
        }
        return Score(value=1.0 if prediction == gold_letter else 0.0, metadata=metadata)

    return score


def _write_eval_logs(root, *, generation_run_name: str, generation_model: str, evaluation_model: str):
    resolved_generation_model = resolve_model_name(generation_model)
    resolved_evaluation_model = resolve_model_name(evaluation_model)

    for mode in MODE_CHOICES:
        for setting_index, setting in enumerate(FINAL5_SETTINGS):
            samples = []
            choice_count = SETTING_SPECS[setting]["num_choices"]
            for dataset_index, (dataset_type, row) in enumerate(DATASET_ROWS.items()):
                sample_id = sample_id_for_row(dataset_type, row, 0)
                gold_letter = CHOICE_LABELS[(dataset_index + setting_index) % choice_count]
                if (dataset_index + setting_index + (0 if mode == "full_question" else 1)) % 2 == 0:
                    prediction = gold_letter
                else:
                    prediction = _wrong_letter(gold_letter, choice_count, step=1 + (mode == "choices_only"))
                samples.append(
                    Sample(
                        input=row["question"],
                        choices=_choices(f"{dataset_type}_{setting}", choice_count),
                        target=gold_letter,
                        id=sample_id,
                        metadata={
                            "sample_id": sample_id,
                            "dataset_type": dataset_type,
                            "row_index": 0,
                            "category": f"{dataset_type}_category",
                            "setting": setting,
                            "mode": mode,
                            "prediction": prediction,
                            "prediction_type": "G" if prediction == gold_letter else "M",
                        },
                    )
                )

            eval(
                Task(
                    name=f"benchmarker_{setting}_{mode}",
                    dataset=MemoryDataset(samples),
                    solver=_prediction_solver(),
                    scorer=_metadata_scorer(),
                    metadata={
                        "kind": "evaluation",
                        "generation_model": resolved_generation_model,
                        "generation_run_name": generation_run_name,
                        "evaluation_model": resolved_evaluation_model,
                        "setting": setting,
                        "mode": mode,
                    },
                ),
                log_dir=str(root),
                display="none",
            )


def test_benchmarker_analysis_recreates_legacy_figure_set_from_inspect_artifacts(tmp_path):
    generation_run_name = "bench-run"
    generation_model = "gpt-5.2-2025-12-11"
    evaluation_model = "Qwen/Qwen3-4B-Instruct-2507"

    cache_root = tmp_path / "augmented"
    cache_dir = cache_root / safe_name(generation_run_name) / safe_name(resolve_model_name(generation_model))
    results_root = tmp_path / "inspect-eval"
    writing_flaw_jsonl = tmp_path / "writing_flaws.jsonl"
    output_dir = tmp_path / "figures"

    _build_augmented_cache(cache_dir)
    _write_writing_flaws(writing_flaw_jsonl)
    _write_eval_logs(
        results_root,
        generation_run_name=generation_run_name,
        generation_model=generation_model,
        evaluation_model=evaluation_model,
    )

    rc = benchmarker_analysis.main(
        [
            "--writing-flaw-jsonl",
            str(writing_flaw_jsonl),
            "--results-root",
            str(results_root),
            "--cache-root",
            str(cache_root),
            "--generator-run-name",
            generation_run_name,
            "--generator-model",
            generation_model,
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == 0
    assert {path.name for path in output_dir.glob("*.png")} == EXPECTED_FIGURES
    assert all(path.stat().st_size > 0 for path in output_dir.glob("*.png"))
