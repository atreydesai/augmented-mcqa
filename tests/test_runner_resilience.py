from pathlib import Path

from config import DistractorType
from experiments.config import ExperimentConfig
from experiments.runner import ExperimentRunner


class _DummyResponse:
    def __init__(self, text: str, answer: str):
        self.text = text
        self._answer = answer

    def extract_answer(self) -> str:
        return self._answer


class _CountingClient:
    def __init__(self, answers: list[str]):
        self.answers = answers
        self.generate_call_count = 0

    @property
    def name(self) -> str:
        return "dummy-client"

    def generate(self, prompt: str, **kwargs):
        answer = self.answers[min(self.generate_call_count, len(self.answers) - 1)]
        self.generate_call_count += 1
        return _DummyResponse(text=f"The answer is {answer}", answer=answer)


def _entry(i: int) -> dict:
    return {
        "question": f"Question {i}",
        "choices_answer": ["gold"],
        "choices_human": ["h1", "h2", "h3"],
        "cond_model_q_a_scratch": ["m1", "m2", "m3"],
        "cond_model_q_a_dhuman": ["m1", "m2", "m3"],
        "cond_model_q_a_dmodel": ["m1", "m2", "m3"],
        "category": "test",
    }


def _config(tmp_path: Path, *, name: str, limit: int | None = None, save_interval: int = 1):
    return ExperimentConfig(
        name=name,
        dataset_path=Path("datasets/augmented/unified_processed_example"),
        model_name="gpt-5-mini-2025-08-07",
        generator_dataset_label="unit",
        num_human=1,
        num_model=1,
        model_distractor_type=DistractorType.COND_MODEL_Q_A_SCRATCH,
        output_dir=tmp_path / "unit" / "gpt-5-mini-2025-08-07_mmlu_pro_scratch" / "1H1M",
        dataset_type_filter="mmlu_pro",
        distractor_source="scratch",
        save_interval=save_interval,
        limit=limit,
    )


def test_runner_continues_after_single_entry_failure(tmp_path, monkeypatch):
    entries = [_entry(0), {"question": "bad row", "choices_answer": ["gold"]}, _entry(2)]
    cfg = _config(tmp_path, name="resilience", limit=3, save_interval=1)
    client = _CountingClient(["A", "A", "A"])
    runner = ExperimentRunner(cfg, client=client)
    monkeypatch.setattr(runner, "_load_data", lambda: entries)

    results = runner.run()

    assert results.attempted_entries == 3
    assert results.successful_entries == 2
    assert results.failed_entries == 1
    assert len(results.entry_failures) == 1
    assert client.generate_call_count == 2
    assert (cfg.output_dir / "results.json").exists()


def test_runner_auto_resumes_from_checkpoint_when_final_missing(tmp_path, monkeypatch):
    entries = [_entry(0), _entry(1), _entry(2)]

    cfg_first = _config(tmp_path, name="resume", limit=1, save_interval=1)
    client_first = _CountingClient(["A"])
    runner_first = ExperimentRunner(cfg_first, client=client_first)
    monkeypatch.setattr(runner_first, "_load_data", lambda: entries)
    first_results = runner_first.run()
    assert first_results.successful_entries == 1

    # Simulate interruption: checkpoint exists, final results removed.
    results_path = cfg_first.output_dir / "results.json"
    results_path.unlink()

    cfg_resume = _config(tmp_path, name="resume", limit=3, save_interval=1)
    client_resume = _CountingClient(["A", "A", "A"])
    runner_resume = ExperimentRunner(cfg_resume, client=client_resume)
    monkeypatch.setattr(runner_resume, "_load_data", lambda: entries)
    resumed = runner_resume.run()

    assert resumed.resumed_from_checkpoint is True
    assert resumed.resumed_next_index == 1
    assert resumed.attempted_entries == 3
    assert resumed.successful_entries == 3
    assert len(resumed.results) == 3
    assert client_resume.generate_call_count == 2
