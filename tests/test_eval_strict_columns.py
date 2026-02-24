from pathlib import Path

from experiments.config import ExperimentConfig
from experiments.runner import ExperimentRunner


class _Client:
    def generate_batch(self, prompts, **kwargs):  # noqa: ANN001
        class _Resp:
            text = "The answer is A"

            @staticmethod
            def extract_answer() -> str:
                return "A"

        return [_Resp() for _ in prompts]


def test_eval_requires_new_final5_columns_without_legacy_fallback(tmp_path, monkeypatch):
    cfg = ExperimentConfig(
        name="strict_cols",
        dataset_path=Path("datasets/augmented/final5"),
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        generator_dataset_label="gpt-5.2-2025-12-11",
        setting_id="human_from_scratch",
        num_human=3,
        num_model=0,
        output_dir=tmp_path / "out",
    )

    runner = ExperimentRunner(cfg, client=_Client())

    # Legacy-only row: has choices_human but missing human_from_scratch.
    rows = [
        {
            "question": "Q0",
            "answer": "A",
            "choices_answer": ["A"],
            "choices_human": ["B", "C", "D"],
        }
    ]
    monkeypatch.setattr(runner, "_load_data", lambda: rows)

    result = runner.run()

    # No fallback to choices_human for human_from_scratch.
    assert result.successful_entries == 0
    assert result.failed_entries == 1
    assert result.entry_failures
