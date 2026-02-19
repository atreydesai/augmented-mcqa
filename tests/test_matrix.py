from pathlib import Path

import pytest

from experiments.matrix import (
    MATRIX_PRESETS,
    build_manifest,
    build_matrix_configs,
    load_configs_from_manifest,
    maybe_select_shard,
    save_manifest,
    select_shard,
    sort_configs_for_sharding,
)


def test_core16_matrix_count(tmp_path):
    configs = build_matrix_configs(
        model="gpt-4.1",
        dataset_path=Path("datasets/augmented/unified_processed_example"),
        dataset_types=["mmlu_pro", "gpqa"],
        distractor_sources=["scratch", "dhuman"],
        preset="core16",
        output_base=tmp_path,
    )
    # Historical core16 naming; unique config set currently has 15 settings.
    assert len(configs) == len(MATRIX_PRESETS["core16"]) * 2 * 2


def test_branching21_matrix_count(tmp_path):
    configs = build_matrix_configs(
        model="gpt-4.1",
        dataset_path=Path("datasets/augmented/unified_processed_example"),
        dataset_types=["mmlu_pro"],
        distractor_sources=["scratch"],
        preset="branching21",
        output_base=tmp_path,
    )
    assert len(configs) == 21
    assert all(cfg.sampling_strategy == "branching_cumulative" for cfg in configs)
    assert all(cfg.branching_mode == "human_prefix" for cfg in configs)

    pairs = {(cfg.num_human, cfg.num_model) for cfg in configs}
    expected_pairs = (
        {(0, m) for m in range(1, 7)}
        | {(1, m) for m in range(0, 6)}
        | {(2, m) for m in range(0, 5)}
        | {(3, m) for m in range(0, 4)}
    )
    assert pairs == expected_pairs


def test_core16_uses_independent_sampling(tmp_path):
    configs = build_matrix_configs(
        model="gpt-4.1",
        dataset_path=Path("datasets/augmented/unified_processed_example"),
        dataset_types=["mmlu_pro"],
        distractor_sources=["scratch"],
        preset="core16",
        output_base=tmp_path,
    )
    assert all(cfg.sampling_strategy == "independent" for cfg in configs)


def test_sharding_is_deterministic_and_disjoint(tmp_path):
    configs = build_matrix_configs(
        model="gpt-4.1",
        dataset_path=Path("datasets/augmented/unified_processed_example"),
        dataset_types=["mmlu_pro", "gpqa"],
        distractor_sources=["scratch"],
        preset="core16",
        output_base=tmp_path,
    )

    shard_count = 5
    first_pass = [select_shard(configs, shard_count, idx) for idx in range(shard_count)]
    second_pass = [select_shard(configs, shard_count, idx) for idx in range(shard_count)]

    assert [[c.config_id for c in shard] for shard in first_pass] == [
        [c.config_id for c in shard] for shard in second_pass
    ]

    shard_sets = [set(c.config_id for c in shard) for shard in first_pass]
    for i, left in enumerate(shard_sets):
        for j, right in enumerate(shard_sets):
            if i == j:
                continue
            assert left.isdisjoint(right)

    union_ids = set().union(*shard_sets)
    sorted_ids = [c.config_id for c in sort_configs_for_sharding(configs)]
    assert union_ids == set(sorted_ids)


def test_maybe_select_shard_requires_both_args(tmp_path):
    configs = build_matrix_configs(
        model="gpt-4.1",
        dataset_path=Path("datasets/augmented/unified_processed_example"),
        dataset_types=["mmlu_pro"],
        distractor_sources=["scratch"],
        preset="core16",
        output_base=tmp_path,
    )

    with pytest.raises(ValueError):
        maybe_select_shard(configs, num_shards=4, shard_index=None)

    with pytest.raises(ValueError):
        maybe_select_shard(configs, num_shards=None, shard_index=0)


def test_manifest_round_trip(tmp_path):
    configs = build_matrix_configs(
        model="gpt-4.1",
        dataset_path=Path("datasets/augmented/unified_processed_example"),
        dataset_types=["mmlu_pro"],
        distractor_sources=["scratch"],
        preset="core16",
        output_base=tmp_path,
        limit=25,
    )

    manifest = build_manifest(
        configs,
        preset="core16",
        model="gpt-4.1",
        dataset_path=Path("datasets/augmented/unified_processed_example"),
        dataset_types=["mmlu_pro"],
        distractor_sources=["scratch"],
    )

    manifest_path = tmp_path / "matrix_manifest.json"
    save_manifest(manifest, manifest_path)
    loaded = load_configs_from_manifest(manifest_path)

    assert [c.config_id for c in loaded] == [c.config_id for c in sort_configs_for_sharding(configs)]


def test_supergpqa_dataset_type_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset types"):
        build_matrix_configs(
            model="gpt-4.1",
            dataset_path=Path("datasets/augmented/unified_processed_example"),
            dataset_types=["supergpqa"],
            distractor_sources=["scratch"],
            preset="core16",
            output_base=tmp_path,
        )
