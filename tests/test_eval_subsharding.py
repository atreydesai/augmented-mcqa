from pathlib import Path

from datasets import load_from_disk

from experiments.config import ExperimentConfig
from experiments.runner import ExperimentRunner


class _Resp:
    def __init__(self, text: str = "The answer is A", answer: str = "A"):
        self.text = text
        self._answer = answer

    def extract_answer(self) -> str:
        return self._answer


class _BatchClient:
    @property
    def name(self) -> str:
        return "dummy"

    def generate_batch(self, prompts, **kwargs):  # noqa: ANN001
        return [_Resp() for _ in prompts]


def _entry(i: int) -> dict:
    return {
        "question": f"Q{i}",
        "answer": "gold",
        "choices_answer": ["gold"],
        "category": "cat",
        "human_from_scratch": ["h1", "h2", "h3"],
        "model_from_scratch": ["m1", "m2", "m3"],
        "augment_human": ["c1", "c2", "c3", "c4", "c5", "c6"],
        "augment_model": ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9"],
        "augment_ablation": ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9"],
    }


def _cfg(
    tmp_path: Path,
    entry_shards: int,
    entry_shard_index: int,
    setting: str = "model_from_scratch",
    entry_shard_strategy: str = "contiguous",
):
    return ExperimentConfig(
        name=f"sub_{entry_shards}_{entry_shard_index}",
        dataset_path=Path("datasets/augmented/final5"),
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        generator_dataset_label="gpt-5.2-2025-12-11",
        setting_id=setting,
        num_human=0 if setting != "human_from_scratch" else 3,
        num_model=3 if setting == "model_from_scratch" else (9 if setting in {"augment_model", "augment_ablation"} else 0),
        output_dir=tmp_path / "out",
        entry_shards=entry_shards,
        entry_shard_index=entry_shard_index,
        entry_shard_strategy=entry_shard_strategy,
        inference_batch_size=4,
    )


def test_entry_subshards_contiguous_are_disjoint_and_complete(tmp_path):
    entries = [{"i": i} for i in range(15)]

    seen = set()
    for shard_index in range(4):
        runner = ExperimentRunner(_cfg(tmp_path, entry_shards=4, entry_shard_index=shard_index), client=_BatchClient())
        subset = runner._select_entry_shard(entries)
        indices = {x["i"] for x in subset}

        assert seen.isdisjoint(indices)
        seen.update(indices)

    assert seen == set(range(15))


def test_entry_subshards_modulo_are_disjoint_and_complete(tmp_path):
    entries = [{"i": i} for i in range(15)]

    seen = set()
    for shard_index in range(4):
        runner = ExperimentRunner(
            _cfg(
                tmp_path,
                entry_shards=4,
                entry_shard_index=shard_index,
                entry_shard_strategy="modulo",
            ),
            client=_BatchClient(),
        )
        subset = runner._select_entry_shard(entries)
        indices = {x["i"] for x in subset}

        assert seen.isdisjoint(indices)
        seen.update(indices)

    assert seen == set(range(15))


def test_runner_writes_entry_shard_partial_summary_and_rows_with_global_indices(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, entry_shards=3, entry_shard_index=1)
    runner = ExperimentRunner(cfg, client=_BatchClient())
    monkeypatch.setattr(runner, "_load_data", lambda: [_entry(i) for i in range(10)])

    results = runner.run()

    # Contiguous shard 1 of 3 over 10 rows -> rows [4,5,6].
    assert len(results.results) == 3
    assert [r.question_idx for r in results.results] == [4, 5, 6]

    expected_summary = (
        cfg.output_dir
        / "_partials"
        / "entry_shard_1_of_3"
        / "summary.json"
    )
    expected_rows = (
        cfg.output_dir
        / "_partials"
        / "entry_shard_1_of_3"
        / "rows"
    )
    assert expected_summary.exists()
    assert expected_rows.exists()

    rows_ds = load_from_disk(str(expected_rows))
    assert [int(r["question_idx"]) for r in rows_ds] == [4, 5, 6]
