import random
from pathlib import Path

import pytest

from config import DistractorType
from experiments.config import ExperimentConfig
from experiments.runner import ExperimentRunner


def _entry():
    return {
        "question": "Which option is correct?",
        "choices_answer": ["gold"],
        "choices_human": ["h1", "h2", "h3", "h4"],
        "cond_model_q_a_scratch": ["m1", "m2", "m3", "m4", "m5", "m6"],
        "cond_model_q_a_dhuman_h1": ["h1_m1", "h1_m2", "h1_m3", "h1_m4", "h1_m5"],
        "cond_model_q_a_dhuman_h2": ["h2_m1", "h2_m2", "h2_m3", "h2_m4"],
        "cond_model_q_a_dhuman_h3": ["h3_m1", "h3_m2", "h3_m3"],
        "category": "test",
    }


def _config(
    tmp_path,
    *,
    name: str,
    num_h: int,
    num_m: int,
    strategy: str,
    branching_mode: str = "shuffled_prefix",
) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        dataset_path=Path("datasets/augmented/unified_processed_example"),
        model_name="gpt-4.1",
        generator_dataset_label="test-gen",
        num_human=num_h,
        num_model=num_m,
        model_distractor_type=DistractorType.COND_MODEL_Q_A_SCRATCH,
        sampling_strategy=strategy,
        branching_mode=branching_mode,
        output_dir=tmp_path / name,
        dataset_type_filter="mmlu_pro",
        distractor_source="scratch",
    )


def test_branching_cumulative_shuffled_prefix_mode_uses_shuffled_prefix_selection(tmp_path):
    entry = _entry()
    idx = 7

    cfg_1h0m = _config(tmp_path, name="b_1h0m", num_h=1, num_m=0, strategy="branching_cumulative", branching_mode="shuffled_prefix")
    cfg_1h1m = _config(tmp_path, name="b_1h1m", num_h=1, num_m=1, strategy="branching_cumulative", branching_mode="shuffled_prefix")
    cfg_1h2m = _config(tmp_path, name="b_1h2m", num_h=1, num_m=2, strategy="branching_cumulative", branching_mode="shuffled_prefix")

    prepared_1h0m = ExperimentRunner(cfg_1h0m)._prepare_entry(entry, idx)
    prepared_1h1m = ExperimentRunner(cfg_1h1m)._prepare_entry(entry, idx)
    prepared_1h2m = ExperimentRunner(cfg_1h2m)._prepare_entry(entry, idx)

    assert prepared_1h0m is not None
    assert prepared_1h1m is not None
    assert prepared_1h2m is not None

    human_pool = list(entry["choices_human"])
    model_pool = list(entry["cond_model_q_a_scratch"])
    human_rng = random.Random(cfg_1h0m.seed + idx + 10_000_019)
    model_rng = random.Random(cfg_1h0m.seed + idx + 20_000_033)
    human_rng.shuffle(human_pool)
    model_rng.shuffle(model_pool)

    assert prepared_1h0m["selected_human"] == human_pool[:1]
    assert prepared_1h1m["selected_human"] == human_pool[:1]
    assert prepared_1h2m["selected_human"] == human_pool[:1]

    assert prepared_1h1m["selected_model"] == model_pool[:1]
    assert prepared_1h2m["selected_model"] == model_pool[:2]


def test_branching_cumulative_human_prefix_uses_d_columns(tmp_path):
    entry = _entry()
    idx = 3

    cfg_h1 = _config(
        tmp_path,
        name="hp_1h2m",
        num_h=1,
        num_m=2,
        strategy="branching_cumulative",
        branching_mode="human_prefix",
    )
    cfg_h2 = _config(
        tmp_path,
        name="hp_2h2m",
        num_h=2,
        num_m=2,
        strategy="branching_cumulative",
        branching_mode="human_prefix",
    )
    cfg_h3 = _config(
        tmp_path,
        name="hp_3h2m",
        num_h=3,
        num_m=2,
        strategy="branching_cumulative",
        branching_mode="human_prefix",
    )

    p_h1 = ExperimentRunner(cfg_h1)._prepare_entry(entry, idx)
    p_h2 = ExperimentRunner(cfg_h2)._prepare_entry(entry, idx)
    p_h3 = ExperimentRunner(cfg_h3)._prepare_entry(entry, idx)

    assert p_h1 is not None
    assert p_h2 is not None
    assert p_h3 is not None

    # Human branch is deterministic prefix: D1, D1+D2, D1+D2+D3.
    assert p_h1["selected_human"] == ["h1"]
    assert p_h2["selected_human"] == ["h1", "h2"]
    assert p_h3["selected_human"] == ["h1", "h2", "h3"]

    # Model branch pulls from matching dhuman_h{n} columns.
    assert p_h1["selected_model"] == ["h1_m1", "h1_m2"]
    assert p_h2["selected_model"] == ["h2_m1", "h2_m2"]
    assert p_h3["selected_model"] == ["h3_m1", "h3_m2"]


def test_independent_strategy_matches_random_sample_behavior(tmp_path):
    entry = _entry()
    idx = 11
    cfg = _config(tmp_path, name="ind_2h3m", num_h=2, num_m=3, strategy="independent")

    prepared = ExperimentRunner(cfg)._prepare_entry(entry, idx)
    assert prepared is not None

    rng = random.Random(cfg.seed + idx)
    expected_h = rng.sample(entry["choices_human"], 2)
    expected_m = rng.sample(entry["cond_model_q_a_scratch"], 3)

    assert prepared["selected_human"] == expected_h
    assert prepared["selected_model"] == expected_m


def test_skip_reason_detects_missing_model_distractors(tmp_path):
    cfg = _config(tmp_path, name="skip_reason", num_h=1, num_m=2, strategy="independent")
    runner = ExperimentRunner(cfg)
    entry = {
        "question": "q",
        "choices_answer": ["gold"],
        "choices_human": ["h1", "h2"],
        "cond_model_q_a_scratch": ["m1"],  # fewer than required
    }
    assert runner._get_skip_reason(entry) == "insufficient_model_distractors"


def test_human_prefix_requires_branch_specific_column(tmp_path):
    entry = _entry()
    del entry["cond_model_q_a_dhuman_h2"]

    cfg = _config(
        tmp_path,
        name="hp_missing_col",
        num_h=2,
        num_m=1,
        strategy="branching_cumulative",
        branching_mode="human_prefix",
    )

    with pytest.raises(KeyError, match="cond_model_q_a_dhuman_h2"):
        ExperimentRunner(cfg)._prepare_entry(entry, idx=0)
