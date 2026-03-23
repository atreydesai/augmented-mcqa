"""Final5 analysis and plotting utilities built from materialized evaluated datasets."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_from_disk

from utils.constants import DEFAULT_EVALUATION_MODELS, EVALUATED_STORE_MANIFEST, SETTING_SPECS


SETTING_OPTION_COUNTS: dict[str, int] = {
    setting: int(spec["num_choices"]) for setting, spec in SETTING_SPECS.items()
}
SETTING_RANDOM_BASELINES: dict[str, float] = {
    setting: 1.0 / options for setting, options in SETTING_OPTION_COUNTS.items()
}
PLOT_COMPARISONS: list[tuple[list[str], str]] = [
    (["human_from_scratch", "model_from_scratch"], "human_vs_model"),
    (["augment_human", "augment_model", "augment_ablation"], "augment_triplet"),
]
COMPARISON_DISPLAY_TITLES: dict[str, str] = {
    "human_vs_model": "Are human or model distractors better?",
    "augment_triplet": "If you are augmenting, which distractor source is better?",
}
MODE_DISPLAY_LABELS: dict[str, str] = {
    "full_question": "Full Question",
    "choices_only": "Choices-only",
}
GENERATOR_DISPLAY_ALIASES: list[tuple[str, str]] = [
    ("gpt-5.2", "gpt-5.2"),
    ("gemini-3.1-pro", "gemini-3.1-pro"),
    ("claude-opus-4-6", "opus-4.6"),
    ("opus-4-6", "opus-4.6"),
    ("Qwen3.5-397B-A17B", "Qwen3.5-397B"),
    ("Qwen3.5-9B", "Qwen3.5-9B"),
    ("Qwen3-4B-Instruct-2507", "Qwen3-4B"),
    ("Olmo-3-7B-Instruct", "Olmo3-7B"),
]
EVAL_MODEL_DISPLAY_LABELS: dict[str, str] = {
    "Qwen/Qwen3-4B-Instruct-2507": "Qwen3-4B",
    "vllm/Qwen/Qwen3-4B-Instruct-2507": "Qwen3-4B",
    "allenai/Olmo-3-7B-Instruct": "Olmo3-7B",
    "vllm/allenai/Olmo-3-7B-Instruct": "Olmo3-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama3.1-8B",
    "vllm/meta-llama/Llama-3.1-8B-Instruct": "Llama3.1-8B",
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2": "Nemotron-Nano-9B",
    "vllm/nvidia/NVIDIA-Nemotron-Nano-9B-v2": "Nemotron-Nano-9B",
    "mistralai/Ministral-3-14B-Instruct-2512": "Ministral-14B",
    "vllm/mistralai/Ministral-3-14B-Instruct-2512": "Ministral-14B",
}
SETTING_DISPLAY_LABELS: dict[str, str] = {
    "human_from_scratch": "human_from_scratch (Normal Benchmark)",
    "model_from_scratch": "model_from_scratch (LLM Distractors from Q+A)",
    "augment_human": "augment_human (Augment Human MCQ with LLM Distractors)",
    "augment_model": "augment_model (Augment Model MCQ with LLM Distractors)",
    "augment_ablation": "augment_ablation (Generate Full MCQ from Q+A in One Step)",
}
SETTING_SHORT_LABELS: dict[str, str] = {
    "human_from_scratch": "HFS",
    "model_from_scratch": "MFS",
    "augment_human": "AH",
    "augment_model": "AM",
    "augment_ablation": "AA",
}
SETTING_ORDER = list(SETTING_SPECS)
DEFAULT_EVAL_MODEL_VARIANTS = tuple(
    model if str(model).startswith("vllm/") else f"vllm/{model}" for model in DEFAULT_EVALUATION_MODELS
)
DATASET_PLOT_ORDER = ["arc_challenge", "mmlu_pro", "gpqa"]
MCNEMAR_SIGNIFICANCE_LEGEND = (
    "McNemar significance: ns (p>=0.05), * (p<0.05), ** (p<0.01), "
    "*** (p<0.001), **** (p<0.0001)"
)
COMPLETENESS_LEGEND = "Hatched bars = partial coverage, gray bars = missing result"
PREDICTION_TYPE_ORDER = ["G", "H", "M", "?"]
PREDICTION_LETTER_ORDER = list("ABCDEFGHIJ")
HIDDEN_EVAL_MODEL_IDENTITIES = {
    "mistralai/Ministral-3-14B-Instruct-2512",
}


def _display_generator(generator: str) -> str:
    raw = str(generator)
    for needle, display in GENERATOR_DISPLAY_ALIASES:
        if needle in raw:
            return display
    return raw


def _display_mode(mode: str) -> str:
    return MODE_DISPLAY_LABELS.get(str(mode), str(mode))


def _display_eval_model(model: str) -> str:
    raw = str(model)
    normalized = raw.replace("_", "/")
    return EVAL_MODEL_DISPLAY_LABELS.get(raw, EVAL_MODEL_DISPLAY_LABELS.get(normalized, raw))


def _display_setting(setting: str) -> str:
    return SETTING_DISPLAY_LABELS.get(str(setting), str(setting))


def _file_safe(value: str) -> str:
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _setting_sort_key(setting: str) -> tuple[int, str]:
    try:
        return (SETTING_ORDER.index(str(setting)), str(setting))
    except ValueError:
        return (len(SETTING_ORDER), str(setting))


def _dataset_sort_key(dataset: str) -> tuple[int, str]:
    try:
        return (DATASET_PLOT_ORDER.index(str(dataset)), str(dataset))
    except ValueError:
        return (len(DATASET_PLOT_ORDER), str(dataset))


def _mode_sort_key(mode: str) -> tuple[int, str]:
    modes = list(MODE_DISPLAY_LABELS)
    try:
        return (modes.index(str(mode)), str(mode))
    except ValueError:
        return (len(modes), str(mode))


def _model_identity(model: str) -> str:
    return str(model).replace("\\", "/").removeprefix("vllm/")


def _resolve_eval_model_variants(observed_models: Iterable[str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for model in observed_models:
        resolved[_model_identity(model)] = str(model)
    for model in DEFAULT_EVAL_MODEL_VARIANTS:
        model_id = _model_identity(model)
        if model_id in HIDDEN_EVAL_MODEL_IDENTITIES:
            continue
        resolved.setdefault(model_id, model)
    return resolved


def _is_hidden_eval_model(model: str) -> bool:
    return _model_identity(model) in HIDDEN_EVAL_MODEL_IDENTITIES


def _binomial_stderr(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    p = correct / total
    return math.sqrt(max(0.0, p * (1.0 - p)) / total)


def _binom_two_sided_pvalue(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    two_pow = 2.0**n
    obs_count = math.comb(n, k)
    mass = 0.0
    for i in range(0, n + 1):
        count = math.comb(n, i)
        if count <= obs_count:
            mass += count / two_pow
    return min(1.0, max(0.0, mass))


def _mcnemar_pvalue(correct_a: dict[int, bool], correct_b: dict[int, bool]) -> tuple[float, int, int, int]:
    keys = sorted(set(correct_a.keys()).intersection(correct_b.keys()))
    b = 0
    c = 0
    for idx in keys:
        a_val = bool(correct_a[idx])
        b_val = bool(correct_b[idx])
        if a_val and not b_val:
            b += 1
        elif (not a_val) and b_val:
            c += 1
    n = b + c
    if n == 0:
        return 1.0, b, c, n
    p = _binom_two_sided_pvalue(min(b, c), n)
    return p, b, c, n


def _significance_stars(p_value: float) -> str:
    if p_value < 1e-4:
        return "****"
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 5e-2:
        return "*"
    return "ns"


def _generator_label(generator: str, generation_run_name: str) -> str:
    run_name = str(generation_run_name or "")
    model_name = str(generator or "")
    if run_name and run_name not in model_name:
        return f"{run_name}/{model_name}"
    return model_name or run_name


def _iter_evaluated_groups(root: Path | str) -> Iterable[tuple[Path, dict[str, object]]]:
    for manifest_path in sorted(Path(root).rglob(EVALUATED_STORE_MANIFEST)):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        yield manifest_path.parent, payload


def _evaluated_row_frame(root: Path | str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_root, manifest in _iter_evaluated_groups(root):
        generator = str(manifest.get("generation_model", "") or "")
        generation_run_name = str(manifest.get("generation_run_name", "") or "")
        generator_key = _generator_label(generator, generation_run_name)
        eval_model = str(manifest.get("evaluation_model", "") or "")
        if _is_hidden_eval_model(eval_model):
            continue
        modes = list(manifest.get("modes") or [])
        settings = list(manifest.get("settings") or [])
        dataset_types = list(manifest.get("dataset_types") or [])
        for dataset_type in dataset_types:
            for setting in settings:
                for mode in modes:
                    dataset_path = group_root / dataset_type / setting / mode
                    if not dataset_path.exists():
                        continue
                    dataset = load_from_disk(str(dataset_path))
                    for row in dataset:
                        payload = dict(row)
                        payload.update(
                            {
                                "generator": generator,
                                "generator_key": generator_key,
                                "generation_run_name": generation_run_name,
                                "eval_model": eval_model,
                                "mode": mode,
                                "dataset": dataset_type,
                                "setting": setting,
                            }
                        )
                        rows.append(payload)
    return pd.DataFrame(rows)


def _empty_results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "generator",
            "generator_key",
            "generation_run_name",
            "eval_model",
            "mode",
            "dataset",
            "setting",
            "total",
            "observed_total",
            "expected_total",
            "missing_samples",
            "coverage_fraction",
            "correct",
            "accuracy",
            "stderr",
            "random_baseline",
            "delta_over_random",
            "status",
        ]
    )


def _add_missing_eval_model_rows(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    rows: list[dict[str, object]] = []
    grouped = summary_df.groupby(
        ["generator_key", "generator", "generation_run_name", "mode", "dataset", "setting"],
        sort=False,
    )
    for (generator_key, generator, generation_run_name, mode, dataset, setting), group in grouped:
        resolved = _resolve_eval_model_variants(group["eval_model"].tolist())
        expected_total = int(group["expected_total"].max())
        observed_models = set(group["eval_model"])
        for _model_id, eval_model in resolved.items():
            if eval_model in observed_models:
                continue
            random_baseline = SETTING_RANDOM_BASELINES.get(str(setting))
            if random_baseline is None:
                continue
            rows.append(
                {
                    "generator_key": generator_key,
                    "generator": generator,
                    "generation_run_name": generation_run_name,
                    "eval_model": eval_model,
                    "mode": mode,
                    "dataset": dataset,
                    "setting": setting,
                    "total": expected_total,
                    "observed_total": 0,
                    "expected_total": expected_total,
                    "missing_samples": expected_total,
                    "coverage_fraction": 0.0,
                    "correct": 0,
                    "accuracy": 0.0,
                    "stderr": 0.0,
                    "random_baseline": random_baseline,
                    "delta_over_random": -random_baseline,
                    "status": "missing",
                }
            )
    if not rows:
        return summary_df
    return pd.concat([summary_df, pd.DataFrame(rows)], ignore_index=True)


def collect_final5_results(results_root: Path | str) -> pd.DataFrame:
    row_df = _evaluated_row_frame(results_root)
    return _collect_final5_results_from_rows(row_df)


def _collect_final5_results_from_rows(row_df: pd.DataFrame) -> pd.DataFrame:
    if row_df.empty:
        return _empty_results_frame()

    row_df["evaluation_status"] = row_df["evaluation_status"].fillna("missing")
    row_df["evaluation_is_correct"] = row_df["evaluation_is_correct"].fillna(False)

    summary_rows: list[dict[str, object]] = []
    grouped = row_df.groupby(
        ["generator_key", "generator", "generation_run_name", "eval_model", "mode", "dataset", "setting"],
        sort=False,
    )
    for (generator_key, generator, generation_run_name, eval_model, mode, dataset, setting), group in grouped:
        expected_total = int(len(group))
        observed = group[group["evaluation_status"] != "missing"]
        observed_total = int(len(observed))
        correct = int(observed["evaluation_is_correct"].astype(bool).sum())
        accuracy = (correct / observed_total) if observed_total else 0.0
        if observed_total == 0:
            status = "missing"
        elif observed_total < expected_total:
            status = "partial"
        else:
            status = "complete"
        random_baseline = SETTING_RANDOM_BASELINES.get(str(setting))
        if random_baseline is None:
            continue
        summary_rows.append(
            {
                "generator_key": generator_key,
                "generator": generator,
                "generation_run_name": generation_run_name,
                "eval_model": eval_model,
                "mode": mode,
                "dataset": dataset,
                "setting": setting,
                "total": expected_total,
                "observed_total": observed_total,
                "expected_total": expected_total,
                "missing_samples": max(0, expected_total - observed_total),
                "coverage_fraction": (observed_total / expected_total) if expected_total else 0.0,
                "correct": correct,
                "accuracy": accuracy,
                "stderr": _binomial_stderr(correct, observed_total),
                "random_baseline": random_baseline,
                "delta_over_random": accuracy - random_baseline,
                "status": status,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = _add_missing_eval_model_rows(summary_df)
    summary_df.sort_values(
        ["generator_key", "dataset", "mode", "setting", "eval_model"],
        inplace=True,
        ignore_index=True,
    )
    return summary_df


def load_final5_analysis_frames(results_root: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_df = _evaluated_row_frame(results_root)
    summary_df = _collect_final5_results_from_rows(row_df)
    return row_df, summary_df


def write_final5_summary_table(
    results_root: Path | str,
    output_csv: Path | str,
    *,
    summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = collect_final5_results(results_root) if summary_df is None else summary_df.copy()
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def _comparison_subset(df: pd.DataFrame, settings: Iterable[str]) -> pd.DataFrame:
    return df[df["setting"].isin(set(settings))].copy()


def _coverage_label(row: pd.Series | None) -> str:
    if row is None:
        return ""
    status = str(row.get("status", ""))
    observed = int(row.get("observed_total", row.get("total", 0)) or 0)
    expected = int(row.get("expected_total", 0) or 0)
    if status == "missing":
        return "missing"
    if status == "partial":
        return f"{observed}/{expected}" if expected > 0 else "partial"
    return ""


def _correctness_map(
    row_df: pd.DataFrame,
    *,
    generator_key: str,
    eval_model: str,
    mode: str,
    dataset: str,
    setting: str,
) -> dict[int, bool]:
    subset = row_df[
        (row_df["generator_key"] == generator_key)
        & (row_df["eval_model"] == eval_model)
        & (row_df["mode"] == mode)
        & (row_df["dataset"] == dataset)
        & (row_df["setting"] == setting)
        & (row_df["evaluation_status"] != "missing")
    ]
    out: dict[int, bool] = {}
    for _, row in subset.iterrows():
        raw_idx = row.get("evaluation_question_idx", row.get("row_index", -1))
        idx = -1 if raw_idx is None else int(raw_idx)
        if idx >= 0:
            out[idx] = bool(row.get("evaluation_is_correct", False))
    return out


def _mcnemar_annotation_for_model(
    *,
    generator_key: str,
    dataset: str,
    mode: str,
    model: str,
    settings: list[str],
    by_setting: dict[str, pd.DataFrame],
    row_df: pd.DataFrame,
) -> str:
    pair_labels: list[str] = []
    for left_idx, left in enumerate(settings):
        for right in settings[left_idx + 1 :]:
            left_row = by_setting[left].set_index("eval_model").loc[model] if model in set(by_setting[left]["eval_model"]) else None
            right_row = by_setting[right].set_index("eval_model").loc[model] if model in set(by_setting[right]["eval_model"]) else None
            if left_row is None or right_row is None:
                continue
            left_status = str(left_row.get("status", ""))
            right_status = str(right_row.get("status", ""))
            if left_status != "complete" or right_status != "complete":
                level = "n/a"
            else:
                left_map = _correctness_map(
                    row_df,
                    generator_key=generator_key,
                    eval_model=model,
                    mode=mode,
                    dataset=dataset,
                    setting=left,
                )
                right_map = _correctness_map(
                    row_df,
                    generator_key=generator_key,
                    eval_model=model,
                    mode=mode,
                    dataset=dataset,
                    setting=right,
                )
                p_value, _b, _c, n_discordant = _mcnemar_pvalue(left_map, right_map)
                level = "n/a" if n_discordant == 0 else _significance_stars(p_value)
            if len(settings) == 2:
                pair_labels.append(level)
            else:
                pair_labels.append(
                    f"{SETTING_SHORT_LABELS.get(left, left)}/{SETTING_SHORT_LABELS.get(right, right)} {level}"
                )
    if not pair_labels:
        return "n/a"
    return pair_labels[0] if len(pair_labels) == 1 else "\n".join(pair_labels)


def _plot_comparison(
    ax,
    comp_df: pd.DataFrame,
    row_df: pd.DataFrame,
    *,
    generator_key: str,
    mode: str,
    dataset: str,
    settings: list[str],
    title: str,
) -> None:
    active_settings = [setting for setting in settings if not comp_df[comp_df["setting"] == setting].empty]
    if not active_settings:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_xticks([])
        return
    by_setting = {setting: comp_df[comp_df["setting"] == setting].copy() for setting in active_settings}
    models = sorted(
        set().union(*[set(frame["eval_model"].tolist()) for frame in by_setting.values()]),
        key=lambda value: (_display_eval_model(value), value),
    )
    x = np.arange(len(models), dtype=float)
    bar_width = 0.22 if len(active_settings) >= 3 else 0.28
    offsets = (
        np.array([0.0], dtype=float)
        if len(active_settings) == 1
        else np.linspace(
            -bar_width * (len(active_settings) - 1) / 2.0,
            bar_width * (len(active_settings) - 1) / 2.0,
            len(active_settings),
        )
    )
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    model_max = np.zeros(len(models), dtype=float)
    for idx, setting in enumerate(active_settings):
        setting_df = by_setting[setting].set_index("eval_model")
        for model_idx, model in enumerate(models):
            row = setting_df.loc[model] if model in setting_df.index else None
            if row is None:
                continue
            accuracy = float(row.get("accuracy", 0.0) or 0.0)
            stderr = float(row.get("stderr", 0.0) or 0.0)
            status = str(row.get("status", ""))
            if status == "missing":
                ax.bar(
                    x[model_idx] + float(offsets[idx]),
                    0.0,
                    width=bar_width,
                    color="#d9d9d9",
                    alpha=0.9,
                    edgecolor="#666666",
                    linewidth=0.5,
                    hatch="//",
                    label=_display_setting(setting) if model_idx == 0 else None,
                )
                model_max[model_idx] = max(model_max[model_idx], 0.02)
            else:
                ax.bar(
                    x[model_idx] + float(offsets[idx]),
                    accuracy,
                    width=bar_width,
                    yerr=stderr if accuracy > 0.0 else None,
                    capsize=3 if accuracy > 0.0 else 0,
                    color=colors[idx % len(colors)],
                    alpha=0.88 if status == "complete" else 0.45,
                    edgecolor="black",
                    linewidth=0.4,
                    hatch=None if status == "complete" else "//",
                    label=_display_setting(setting) if model_idx == 0 else None,
                )
                model_max[model_idx] = max(model_max[model_idx], accuracy + stderr)
            label = _coverage_label(row)
            if label:
                y_pos = 0.03 if status == "missing" else min(1.04, accuracy + stderr + 0.02)
                ax.text(
                    x[model_idx] + float(offsets[idx]),
                    y_pos,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )
    baseline_values = sorted({SETTING_RANDOM_BASELINES[setting] for setting in active_settings})
    if len(baseline_values) == 1:
        ax.axhline(baseline_values[0], linestyle="--", color="#555555", alpha=0.35, linewidth=1.0)
    else:
        for idx, setting in enumerate(active_settings):
            ax.axhline(
                SETTING_RANDOM_BASELINES[setting],
                linestyle="--",
                color=colors[idx % len(colors)],
                alpha=0.2,
                linewidth=1.0,
            )
    for idx, model in enumerate(models):
        annotation = _mcnemar_annotation_for_model(
            generator_key=generator_key,
            dataset=dataset,
            mode=mode,
            model=model,
            settings=active_settings,
            by_setting=by_setting,
            row_df=row_df,
        )
        ax.text(x[idx], min(1.12, model_max[idx] + 0.08), annotation, ha="center", va="bottom", fontsize=7)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([_display_eval_model(model) for model in models], rotation=15, ha="right")
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", alpha=0.2)


def _write_issue_tables(summary_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    issue_df = summary_df[summary_df["status"] != "complete"].copy()
    issue_df.sort_values(
        ["generator_key", "dataset", "mode", "setting", "eval_model"],
        inplace=True,
        ignore_index=True,
    )
    issue_path = tables_dir / "final5_missing_or_partial.csv"
    issue_df.to_csv(issue_path, index=False)
    outputs.append(issue_path)
    return outputs


def _write_failure_tables(row_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    failed = row_df[
        (row_df["evaluation_status"] != "missing")
        & (~row_df["evaluation_is_correct"].fillna(False).astype(bool))
    ].copy()
    failed.sort_values(
        ["generator_key", "mode", "dataset", "setting", "eval_model", "row_index"],
        inplace=True,
        ignore_index=True,
    )
    failed_path = tables_dir / "final5_failed_questions.csv"
    failed.to_csv(failed_path, index=False)
    outputs.append(failed_path)

    missing = row_df[row_df["evaluation_status"] == "missing"].copy()
    missing.sort_values(
        ["generator_key", "mode", "dataset", "setting", "eval_model", "row_index"],
        inplace=True,
        ignore_index=True,
    )
    missing_path = tables_dir / "final5_missing_questions.csv"
    missing.to_csv(missing_path, index=False)
    outputs.append(missing_path)
    return outputs


def _plot_distribution(
    row_df: pd.DataFrame,
    output_dir: Path,
    *,
    column: str,
    categories: list[str],
    filename_prefix: str,
    legend_title: str,
    only_observed: bool = True,
    reference_column: str | None = None,
    reference_label_prefix: str = "Gold",
) -> list[Path]:
    outputs: list[Path] = []
    all_rows = row_df.copy()
    filtered = row_df.copy()
    if only_observed:
        filtered = filtered[filtered["evaluation_status"] != "missing"].copy()
    if filtered.empty:
        return outputs
    filtered[column] = filtered[column].fillna("").astype(str)
    grouped = filtered.groupby(["generator_key", "generator", "mode"], sort=True)
    for (generator_key, generator, mode), group_df in grouped:
        datasets = [dataset for dataset in DATASET_PLOT_ORDER if dataset in set(group_df["dataset"])] or sorted(set(group_df["dataset"]))
        all_group_df = all_rows[
            (all_rows["generator_key"] == generator_key)
            & (all_rows["mode"] == mode)
        ].copy()
        bar_counts: list[int] = []
        for dataset in datasets:
            dataset_df = group_df[group_df["dataset"] == dataset].copy()
            observed_bars = len(
                dataset_df[["eval_model", "setting"]].drop_duplicates()
            )
            gold_bars = 0
            if reference_column is not None:
                gold_bars = len(dataset_df["setting"].drop_duplicates())
            bar_counts.append(observed_bars + gold_bars)
        fig_width = sum(max(6.8, 0.72 * count + 2.4) for count in bar_counts)
        fig, axes = plt.subplots(1, len(datasets), figsize=(fig_width, 6.6))
        if len(datasets) == 1:
            axes = [axes]
        legend_handles: list[object] = []
        legend_labels: list[str] = []
        for ax, dataset in zip(axes, datasets):
            dataset_df = group_df[group_df["dataset"] == dataset].copy()
            ordered_settings = sorted(set(dataset_df["setting"]), key=_setting_sort_key)
            ordered_models = sorted(
                set(dataset_df["eval_model"]),
                key=lambda value: (_display_eval_model(value), value),
            )
            label_order: list[str] = []
            for setting in ordered_settings:
                for eval_model in ordered_models:
                    match = dataset_df[
                        (dataset_df["setting"] == setting)
                        & (dataset_df["eval_model"] == eval_model)
                    ]
                    if match.empty:
                        continue
                    label_order.append(
                        f"{_display_eval_model(eval_model)} / {SETTING_SHORT_LABELS.get(str(setting), str(setting))}"
                    )

            dataset_df["bar_label"] = dataset_df["eval_model"].map(_display_eval_model) + " / " + dataset_df["setting"].map(
                lambda value: SETTING_SHORT_LABELS.get(str(value), str(value))
            )
            pivot = dataset_df.groupby(["bar_label", column]).size().unstack(fill_value=0)

            if reference_column is not None:
                reference_df = all_group_df[all_group_df["dataset"] == dataset].copy()
                reference_df[reference_column] = reference_df[reference_column].fillna("").astype(str)
                for setting in ordered_settings:
                    setting_reference = reference_df[reference_df["setting"] == setting]
                    if setting_reference.empty:
                        continue
                    label = f"{reference_label_prefix} / {SETTING_SHORT_LABELS.get(str(setting), str(setting))}"
                    label_order.append(label)
                    counts = setting_reference.groupby(reference_column).size()
                    for category, value in counts.items():
                        pivot.loc[label, str(category)] = float(value)

            ordered_categories = [category for category in categories if category in pivot.columns]
            extras = sorted([category for category in pivot.columns if category not in ordered_categories])
            ordered_categories.extend(extras)
            pivot = pivot.reindex(columns=ordered_categories, fill_value=0.0)
            pivot = pivot.reindex([label for label in label_order if label in pivot.index], fill_value=0.0)
            pivot = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0)

            bottom = np.zeros(len(pivot.index), dtype=float)
            color_map = plt.get_cmap("tab10")
            for idx, category in enumerate(pivot.columns):
                values = pivot[category].to_numpy(dtype=float)
                bars = ax.bar(
                    np.arange(len(pivot.index)),
                    values,
                    bottom=bottom,
                    color=color_map(idx % 10),
                    label=str(category),
                )
                bottom += values
                if str(category) not in legend_labels:
                    legend_handles.append(bars[0])
                    legend_labels.append(str(category))

            ax.set_title(str(dataset))
            ax.set_xticks(np.arange(len(pivot.index)))
            ax.set_xticklabels(list(pivot.index), rotation=28, ha="right", fontsize=8)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Rate")
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle(
            f"{legend_title} | generator={_display_generator(generator)} | mode={_display_mode(mode)}"
        )
        if legend_handles and legend_labels:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.02),
                ncols=min(5, len(legend_labels)),
                frameon=False,
                title=legend_title,
            )
            fig.subplots_adjust(top=0.84, bottom=0.3, wspace=0.22)
        else:
            fig.subplots_adjust(top=0.84, bottom=0.22, wspace=0.22)
        out_png = output_dir / f"{filename_prefix}_{_file_safe(generator_key)}_{_file_safe(mode)}.png"
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        outputs.append(out_png)
    return outputs


def plot_final5_pairwise(
    results_root: Path | str,
    output_dir: Path | str,
    include_tables: bool = True,
    *,
    row_df: pd.DataFrame | None = None,
    summary_df: pd.DataFrame | None = None,
) -> list[Path]:
    if row_df is None or summary_df is None:
        row_df, summary_df = load_final5_analysis_frames(results_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        return []

    outputs: list[Path] = []
    grouped = summary_df.groupby(["generator_key", "generator", "mode"], sort=True)
    for (generator_key, generator, mode), group_df in grouped:
        datasets_present = set(group_df["dataset"].tolist())
        datasets = [dataset for dataset in DATASET_PLOT_ORDER if dataset in datasets_present] or sorted(datasets_present)
        for settings, title_key in PLOT_COMPARISONS:
            fig, axes = plt.subplots(1, len(datasets), figsize=(6.4 * len(datasets), 6.8))
            if len(datasets) == 1:
                axes = [axes]
            pretty_title = COMPARISON_DISPLAY_TITLES.get(title_key, title_key.replace("_", " "))
            for ax, dataset in zip(axes, datasets):
                dataset_df = group_df[group_df["dataset"] == dataset]
                comp_df = _comparison_subset(dataset_df, settings)
                _plot_comparison(
                    ax,
                    comp_df,
                    row_df,
                    generator_key=generator_key,
                    mode=mode,
                    dataset=dataset,
                    settings=settings,
                    title=str(dataset),
                )
            fig.suptitle(
                f"{pretty_title} | generator={_display_generator(generator)} | mode={_display_mode(mode)}"
            )
            handles: list[object] = []
            labels: list[str] = []
            for ax in axes:
                h, l = ax.get_legend_handles_labels()
                if h:
                    handles = h
                    labels = l
                    break
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.13),
                    ncols=1,
                    frameon=False,
                )
            fig.text(0.5, 0.06, MCNEMAR_SIGNIFICANCE_LEGEND, ha="center", va="bottom", fontsize=9)
            fig.text(0.5, 0.03, COMPLETENESS_LEGEND, ha="center", va="bottom", fontsize=9)
            fig.subplots_adjust(top=0.86, bottom=0.34, wspace=0.22)
            out_png = out_dir / f"pairwise_{_file_safe(generator_key)}_{_file_safe(mode)}_{title_key}.png"
            fig.savefig(out_png, dpi=200)
            plt.close(fig)
            outputs.append(out_png)
            if include_tables:
                pairwise_csv = out_dir / "tables" / f"pairwise_{_file_safe(generator_key)}_{_file_safe(mode)}_{title_key}.csv"
                pairwise_csv.parent.mkdir(parents=True, exist_ok=True)
                _comparison_subset(group_df, settings).sort_values(
                    ["dataset", "setting", "eval_model"]
                ).to_csv(pairwise_csv, index=False)
                outputs.append(pairwise_csv)
    if include_tables:
        full_csv = out_dir / "tables" / "final5_results_summary.csv"
        full_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(full_csv, index=False)
        outputs.append(full_csv)
        outputs.extend(_write_issue_tables(summary_df, out_dir))
    outputs.extend(_write_failure_tables(row_df, out_dir))
    outputs.extend(
        _plot_distribution(
            row_df,
            out_dir,
            column="evaluation_prediction_type",
            categories=PREDICTION_TYPE_ORDER,
            filename_prefix="prediction_type_distribution",
            legend_title="Prediction Type",
        )
    )
    outputs.extend(
        _plot_distribution(
            row_df,
            out_dir,
            column="evaluation_prediction",
            categories=PREDICTION_LETTER_ORDER,
            filename_prefix="prediction_distribution",
            legend_title="Predicted Letter",
            reference_column="correct_answer_letter",
        )
    )
    return outputs
