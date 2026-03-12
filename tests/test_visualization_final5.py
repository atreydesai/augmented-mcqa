from pathlib import Path

from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, scorer
from inspect_ai.solver import solver

from analysis.visualize import collect_final5_results


@solver
def _solver():
    async def solve(state, generate):  # noqa: ANN001
        state.output.completion = "A"
        return state

    return solve


@scorer(metrics=[])
def _scorer(setting: str, mode: str, dataset_type: str):
    async def score(state, target):  # noqa: ANN001
        return Score(
            value=1.0,
            metadata={
                "dataset_type": dataset_type,
                "question_idx": 0,
                "prediction_type": "G",
                "category": "cat",
            },
        )

    return score


def _write_log(root: Path, *, task_name: str, setting: str, mode: str, dataset_type: str):
    eval(
        Task(
            name=task_name,
            dataset=MemoryDataset([Sample(input="Q", choices=["x", "y"], target="A", id=f"{dataset_type}:0")]),
            solver=_solver(),
            scorer=_scorer(setting, mode, dataset_type),
            metadata={
                "kind": "evaluation",
                "generation_model": "openai/gpt-5.2-2025-12-11",
                "evaluation_model": "vllm/Qwen/Qwen3-4B-Instruct-2507",
                "setting": setting,
                "mode": mode,
            },
        ),
        log_dir=str(root),
        display="none",
    )


def test_collect_final5_results_reads_inspect_eval_logs(tmp_path):
    root = tmp_path / "inspect"
    _write_log(root, task_name="eval_hfs_full", setting="human_from_scratch", mode="full_question", dataset_type="arc_challenge")
    _write_log(root, task_name="eval_mfs_choices", setting="model_from_scratch", mode="choices_only", dataset_type="gpqa")

    df = collect_final5_results(root)
    assert set(df["setting"]) == {"human_from_scratch", "model_from_scratch"}
    assert set(df["dataset"]) == {"arc_challenge", "gpqa"}
    assert all(df["accuracy"] == 1.0)
