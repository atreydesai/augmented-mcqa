import json
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset

from analysis.irt import (
    DEFAULT_ITEM_PRIOR_SD,
    _gradient_only,
    _objective_only,
    _safe_name,
    build_design,
    evaluator_severity_frame,
    fit_design,
    load_irt_frame,
    run_irt_analysis,
    setting_difficulty_frame,
)
from cli.app import main
from utils.constants import EVALUATED_STORE_MANIFEST


def _write_group(root: Path, *, generator: str, evaluator: str, rows_by_split: dict[tuple[str, str, str], list[dict[str, object]]]) -> None:
    group_root = root / _safe_name("irt_run") / _safe_name(generator) / _safe_name(evaluator)
    group_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "evaluated_mcqa_setting_mode_records_v1",
        "storage_kind": "evaluated_setting_mode_records",
        "dataset_types": sorted({dataset for dataset, _setting, _mode in rows_by_split}),
        "settings": sorted({setting for _dataset, setting, _mode in rows_by_split}),
        "modes": sorted({mode for _dataset, _setting, mode in rows_by_split}),
        "generation_run_name": "irt_run",
        "generation_model": generator,
        "evaluation_model": evaluator,
        "source_results_root": "/tmp/inspect",
    }
    (group_root / EVALUATED_STORE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    for (dataset, setting, mode), rows in rows_by_split.items():
        path = group_root / dataset / setting / mode
        path.parent.mkdir(parents=True, exist_ok=True)
        Dataset.from_list(rows).save_to_disk(str(path))


def _row(*, sample_id: str, dataset: str, setting: str, mode: str, num_choices: int, correct: bool) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "question": f"Question {sample_id}",
        "dataset_type": dataset,
        "evaluation_prediction": "A" if correct else "B",
        "evaluation_is_correct": correct,
        "evaluation_status": "success",
        "num_choices": num_choices,
        "setting": setting,
    }


def _build_synthetic_collected(root: Path) -> Path:
    rng = np.random.default_rng(7)
    generators = {
        "openai/gpt-5.2-2025-12-11": 0.55,
        "google/gemini-3.1-pro-preview": -0.1,
    }
    evaluators = {
        "vllm/nvidia/NVIDIA-Nemotron-Nano-9B-v2": 0.0,
        "vllm/Qwen/Qwen3-4B-Instruct-2507": 0.9,
    }
    settings = {
        "human_from_scratch": (0.0, 4),
        "augment_model": (1.1, 10),
    }
    modes = {"choices_only": 0.0, "full_question": -0.35}
    items = {f"item-{idx}": (-1.4 + 0.14 * idx) for idx in range(18)}

    by_group: dict[tuple[str, str], dict[tuple[str, str, str], list[dict[str, object]]]] = {}
    for generator, theta in generators.items():
        for evaluator, severity in evaluators.items():
            rows_by_split: dict[tuple[str, str, str], list[dict[str, object]]] = {}
            for setting, (delta, num_choices) in settings.items():
                for mode, gamma in modes.items():
                    split_rows: list[dict[str, object]] = []
                    for sample_id, difficulty in items.items():
                        c = 1.0 / num_choices
                        eta = theta - difficulty - delta - gamma - severity
                        logistic = 1.0 / (1.0 + np.exp(-eta))
                        p = c + (1.0 - c) * logistic
                        correct = bool(rng.random() < p)
                        split_rows.append(
                            _row(
                                sample_id=sample_id,
                                dataset="arc_challenge",
                                setting=setting,
                                mode=mode,
                                num_choices=num_choices,
                                correct=correct,
                            )
                        )
                    rows_by_split[("arc_challenge", setting, mode)] = split_rows
            by_group[(generator, evaluator)] = rows_by_split

    for (generator, evaluator), rows_by_split in by_group.items():
        _write_group(root, generator=generator, evaluator=evaluator, rows_by_split=rows_by_split)
    return root


def test_load_irt_frame_collects_observed_rows(tmp_path):
    root = _build_synthetic_collected(tmp_path / "collected")
    df = load_irt_frame(root)
    assert not df.empty
    assert set(df["setting"]) == {"human_from_scratch", "augment_model"}
    assert set(df["mode"]) == {"full_question"}
    assert set(df["choice_group"]) == {"4-choice", "10-choice"}
    assert df["obs_id"].is_unique


def test_design_gradient_matches_finite_difference(tmp_path):
    root = _build_synthetic_collected(tmp_path / "collected")
    frame = load_irt_frame(root)
    design = build_design(frame, interaction=False)
    beta = np.linspace(-0.1, 0.1, design.param_count)
    analytic = _gradient_only(beta, design, DEFAULT_ITEM_PRIOR_SD)
    epsilon = 1e-6
    for idx in (0, min(3, design.param_count - 1), design.param_count - 1):
        basis = np.zeros(design.param_count)
        basis[idx] = epsilon
        numeric = (
            _objective_only(beta + basis, design, DEFAULT_ITEM_PRIOR_SD)
            - _objective_only(beta - basis, design, DEFAULT_ITEM_PRIOR_SD)
        ) / (2.0 * epsilon)
        assert abs(analytic[idx] - numeric) < 1e-4


def test_fit_recovers_setting_and_evaluator_order(tmp_path):
    root = _build_synthetic_collected(tmp_path / "collected")
    frame = load_irt_frame(root)
    fit = fit_design(build_design(frame, interaction=False), maxiter=50)
    assert fit.success

    setting_rows = setting_difficulty_frame(fit)
    human = float(setting_rows.loc[setting_rows["level"] == "human_from_scratch", "estimate"].iloc[0])
    augment = float(setting_rows.loc[setting_rows["level"] == "augment_model", "estimate"].iloc[0])
    assert augment > human

    evaluator_rows = evaluator_severity_frame(fit)
    qwen = float(evaluator_rows.loc[evaluator_rows["evaluator"] == "vllm/Qwen/Qwen3-4B-Instruct-2507", "estimate"].iloc[0])
    nemo = float(evaluator_rows.loc[evaluator_rows["evaluator"] == "vllm/nvidia/NVIDIA-Nemotron-Nano-9B-v2", "estimate"].iloc[0])
    assert qwen > nemo


def test_run_irt_analysis_writes_outputs(tmp_path):
    root = _build_synthetic_collected(tmp_path / "collected")
    outputs = run_irt_analysis(collected_root=root, output_dir=tmp_path / "irt", maxiter=40)
    output_names = {path.name for path in outputs}
    assert "setting_difficulty.csv" in output_names
    assert "evaluator_severity.csv" in output_names
    assert "generator_ability.csv" in output_names
    assert "fit_summary.json" in output_names
    assert "setting_difficulty_forest.png" in output_names
    assert "item_anomalies.png" in output_names
    assert "evaluator_choice_count_interaction.csv" not in output_names
    assert "mode_effect.csv" not in output_names
    assert "mode_effect_forest.png" not in output_names


def test_cli_analyze_irt_smoke(tmp_path):
    root = _build_synthetic_collected(tmp_path / "collected")
    output_dir = tmp_path / "irt-cli"
    rc = main(
        [
            "analyze-irt",
            "--collected-root",
            str(root),
            "--output-dir",
            str(output_dir),
            "--maxiter",
            "30",
        ]
    )
    assert rc == 0
    assert (output_dir / "fit_summary.json").exists()


def test_run_irt_analysis_fails_on_unidentified_filtered_subset(tmp_path):
    root = _build_synthetic_collected(tmp_path / "collected")
    with pytest.raises(ValueError, match="co-occurs with only one|must each have at least two levels"):
        run_irt_analysis(
            collected_root=root,
            output_dir=tmp_path / "irt-bad",
            evaluators=["vllm/nvidia/NVIDIA-Nemotron-Nano-9B-v2"],
            maxiter=20,
        )
