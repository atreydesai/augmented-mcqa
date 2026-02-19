from argparse import Namespace
from pathlib import Path

from config import DistractorType
from experiments.config import ExperimentConfig
from scripts import eval_matrix


class _DummyResults:
    def __init__(self, accuracy: float = 0.5, total: int = 3):
        self.accuracy = accuracy
        self.results = [object()] * total


def _config(tmp_path: Path, name: str) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        dataset_path=Path("datasets/augmented/unified_processed_example"),
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        num_human=1,
        num_model=1,
        model_distractor_type=DistractorType.COND_MODEL_Q_A_SCRATCH,
        output_dir=tmp_path / name,
        dataset_type_filter="mmlu_pro",
        distractor_source="scratch",
    )


def test_cmd_run_reuses_shared_client_and_dataset_cache(tmp_path, monkeypatch):
    configs = [_config(tmp_path, "cfg_a"), _config(tmp_path, "cfg_b")]
    shared_client = object()
    get_client_calls = {"count": 0}
    run_calls = []

    def fake_get_client(model_name, **kwargs):
        get_client_calls["count"] += 1
        assert model_name == "Qwen/Qwen3-4B-Instruct-2507"
        return shared_client

    def fake_run_experiment(config, client=None, dataset_cache=None):
        run_calls.append((config.name, client, id(dataset_cache)))
        assert client is shared_client
        assert isinstance(dataset_cache, dict)
        return _DummyResults()

    monkeypatch.setattr(eval_matrix, "_resolve_configs", lambda args: configs)
    monkeypatch.setattr(eval_matrix, "maybe_select_shard", lambda cfgs, *_: cfgs)
    monkeypatch.setattr(eval_matrix, "get_client", fake_get_client)
    monkeypatch.setattr(eval_matrix, "run_experiment", fake_run_experiment)

    args = Namespace(
        preset="core16",
        model="Qwen/Qwen3-4B-Instruct-2507",
        dataset_path="datasets/augmented/unified_processed_example",
        dataset_types=["mmlu_pro"],
        distractor_sources=["scratch"],
        limit=None,
        eval_mode="behavioral",
        choices_only=False,
        seed=42,
        reasoning_effort=None,
        thinking_level=None,
        temperature=0.0,
        max_tokens=100,
        output_dir=str(tmp_path / "summaries"),
        num_shards=None,
        shard_index=None,
        manifest=None,
        skip_existing=False,
    )

    status = eval_matrix.cmd_run(args)

    assert status == 0
    assert get_client_calls["count"] == 1
    assert len(run_calls) == 2
    assert run_calls[0][1] is shared_client
    assert run_calls[1][1] is shared_client
    assert run_calls[0][2] == run_calls[1][2]
