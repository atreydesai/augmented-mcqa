"""Final5 analysis and plotting utilities."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SETTING_OPTION_COUNTS: dict[str, int] = {
    "human_from_scratch": 4,
    "model_from_scratch": 4,
    "augment_human": 10,
    "augment_model": 10,
    "augment_ablation": 10,
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
    "augment_triplet": (
        "If you are augmenting, are distractors based on humans or models "
        "(one-step or two-step) better?"
    ),
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
]

EVAL_MODEL_DISPLAY_LABELS: dict[str, str] = {
    "Qwen_Qwen3-4B-Instruct-2507": "Qwen3-4B",
    "Qwen/Qwen3-4B-Instruct-2507": "Qwen3-4B",
    "allenai_Olmo-3-7B-Instruct": "Olmo3-7B",
    "allenai/Olmo-3-7B-Instruct": "Olmo3-7B",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama3.1-8B",
}

SETTING_DISPLAY_LABELS: dict[str, str] = {
    "human_from_scratch": "human_from_scratch (Normal Benchmark)",
    "model_from_scratch": "model_from_scratch (LLM Distractors from Q+A)",
    "augment_human": "augment_human (Augment Human MCQ with LLM Distractors)",
    "augment_model": "augment_model (Augment Model MCQ with LLM Distractors)",
    "augment_ablation": "augment_ablation (Generate Full MCQ from Q+A in One Step (Ablation))",
}

DATASET_PLOT_ORDER = ["arc_challenge", "mmlu_pro", "gpqa"]


def _display_generator(generator: str) -> str:
    raw = str(generator)
    for needle, display in GENERATOR_DISPLAY_ALIASES:
        if needle in raw:
            return display
    return raw


def _display_mode(mode: str) -> str:
    return MODE_DISPLAY_LABELS.get(str(mode), str(mode))


def _display_eval_model(model: str) -> str:
    return EVAL_MODEL_DISPLAY_LABELS.get(str(model), str(model))


def _display_setting(setting: str) -> str:
    return SETTING_DISPLAY_LABELS.get(str(setting), str(setting))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = correct / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2.0 * total)) / denom
    spread = z * math.sqrt((p * (1.0 - p) + (z**2) / (4.0 * total)) / total) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def collect_final5_results(results_root: Path | str) -> pd.DataFrame:
    """Collect canonical Final5 results into a flat dataframe."""
    root = Path(results_root)
    rows: list[dict[str, object]] = []

    summary_paths = sorted(root.glob("*/*/*/*/*/summary.json"))
    summary_parents = {p.parent for p in summary_paths}
    legacy_paths = [
        p for p in sorted(root.glob("*/*/*/*/*/results.json"))
        if p.parent not in summary_parents
    ]

    for path in [*summary_paths, *legacy_paths]:
        rel = path.relative_to(root)
        if len(rel.parts) != 6:
            continue

        generator, eval_model, mode, dataset, setting, _ = rel.parts

        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        results = payload.get("results", [])

        total = _safe_int(summary.get("total"), len(results))
        if total <= 0:
            total = len(results)
        correct = _safe_int(summary.get("correct"), sum(1 for r in results if r.get("is_correct")))
        accuracy = _safe_float(summary.get("accuracy"), (correct / total) if total else 0.0)

        random_baseline = SETTING_RANDOM_BASELINES.get(setting)
        if random_baseline is None:
            continue

        rows.append(
            {
                "generator": generator,
                "eval_model": eval_model,
                "mode": mode,
                "dataset": dataset,
                "setting": setting,
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "random_baseline": random_baseline,
                "delta_over_random": accuracy - random_baseline,
                "results_path": str(path),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "generator",
                "eval_model",
                "mode",
                "dataset",
                "setting",
                "total",
                "correct",
                "accuracy",
                "random_baseline",
                "delta_over_random",
                "ci_low",
                "ci_high",
                "results_path",
            ]
        )

    df = pd.DataFrame(rows)
    return add_binomial_ci(df)


def add_binomial_ci(df: pd.DataFrame) -> pd.DataFrame:
    """Append Wilson 95% confidence interval columns."""
    if df.empty:
        out = df.copy()
        out["ci_low"] = []
        out["ci_high"] = []
        return out

    lows: list[float] = []
    highs: list[float] = []
    for correct, total in zip(df["correct"], df["total"]):
        lo, hi = _wilson_ci(int(correct), int(total))
        lows.append(lo)
        highs.append(hi)

    out = df.copy()
    out["ci_low"] = lows
    out["ci_high"] = highs
    return out


def write_final5_summary_table(results_root: Path | str, output_csv: Path | str) -> pd.DataFrame:
    """Write a CSV table containing baseline and delta metrics."""
    df = collect_final5_results(results_root)
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(
        ["generator", "dataset", "mode", "setting", "eval_model"],
        inplace=True,
        ignore_index=True,
    )
    df.to_csv(out, index=False)
    return df


def _comparison_subset(df: pd.DataFrame, settings: Iterable[str]) -> pd.DataFrame:
    setting_set = set(settings)
    return df[df["setting"].isin(setting_set)].copy()


def _plot_comparison(ax, comp_df: pd.DataFrame, settings: list[str], title: str) -> None:
    by_setting = {
        setting: comp_df[comp_df["setting"] == setting].set_index("eval_model")
        for setting in settings
    }
    model_sets = [set(df.index) for df in by_setting.values()]
    models = sorted(set.intersection(*model_sets)) if model_sets else []

    if not models:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No overlapping eval models", ha="center", va="center")
        ax.set_xticks([])
        return

    x = np.arange(len(models), dtype=float)
    if len(settings) == 1:
        offsets = [0.0]
    else:
        offsets = np.linspace(-0.22, 0.22, len(settings))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, setting in enumerate(settings):
        setting_df = by_setting[setting]
        acc = np.array([float(setting_df.loc[m, "accuracy"]) for m in models])
        lo = np.array([float(setting_df.loc[m, "ci_low"]) for m in models])
        hi = np.array([float(setting_df.loc[m, "ci_high"]) for m in models])

        color = colors[idx % len(colors)]
        ax.errorbar(
            x + float(offsets[idx]),
            acc,
            yerr=[acc - lo, hi - acc],
            fmt="o",
            capsize=3,
            label=_display_setting(setting),
            color=color,
        )

        baseline = SETTING_RANDOM_BASELINES[setting]
        ax.scatter(x + float(offsets[idx]), [baseline] * len(x), marker="x", color=color, alpha=0.8)
        ax.axhline(baseline, linestyle="--", color=color, alpha=0.2)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([_display_eval_model(m) for m in models], rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", alpha=0.2)


def plot_final5_pairwise(
    results_root: Path | str,
    output_dir: Path | str,
    include_tables: bool = True,
) -> list[Path]:
    """Create required Final5 pairwise plots per generator and mode.

    Each plot contains dataset subplots side-by-side (arc/mmlu_pro/gpqa when
    available), so each generator produces:
    - 2 comparisons x 2 modes = 4 PNGs
    """
    df = collect_final5_results(results_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    if include_tables:
        tables_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        return []

    outputs: list[Path] = []

    grouped = df.groupby(["generator", "mode"], sort=True)
    for (generator, mode), group_df in grouped:
        datasets_present = set(group_df["dataset"].tolist())
        datasets = [d for d in DATASET_PLOT_ORDER if d in datasets_present]
        if not datasets:
            datasets = sorted(datasets_present)

        for settings, title_key in PLOT_COMPARISONS:
            fig, axes = plt.subplots(1, len(datasets), figsize=(6.4 * len(datasets), 5.8))
            if len(datasets) == 1:
                axes = [axes]

            pretty_title = COMPARISON_DISPLAY_TITLES.get(title_key, title_key.replace("_", " "))
            for ax, dataset in zip(axes, datasets):
                dataset_df = group_df[group_df["dataset"] == dataset]
                comp_df = _comparison_subset(dataset_df, settings)
                _plot_comparison(
                    ax,
                    comp_df,
                    settings,
                    f"{dataset}",
                )

            fig.suptitle(
                f"{pretty_title} | generator={_display_generator(generator)} | "
                f"mode={_display_mode(mode)}"
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
                    bbox_to_anchor=(0.5, 0.0),
                    ncols=max(2, len(settings)),
                    frameon=False,
                )
            fig.subplots_adjust(top=0.82, bottom=0.26, wspace=0.22)

            out_png = out_dir / f"pairwise_{generator}_{mode}_{title_key}.png"
            fig.savefig(out_png, dpi=200)
            plt.close(fig)
            outputs.append(out_png)

            if include_tables:
                per_plot_df = _comparison_subset(group_df, settings)
                per_pair_csv = tables_dir / f"pairwise_{generator}_{mode}_{title_key}.csv"
                per_plot_df.sort_values(["dataset", "setting", "eval_model"], inplace=False).to_csv(
                    per_pair_csv,
                    index=False,
                )
                outputs.append(per_pair_csv)

    if include_tables:
        full_csv = tables_dir / "final5_results_summary.csv"
        df.sort_values(["generator", "dataset", "mode", "setting", "eval_model"]).to_csv(full_csv, index=False)
        outputs.append(full_csv)

    return outputs
