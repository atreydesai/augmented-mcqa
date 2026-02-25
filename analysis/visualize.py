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

PAIRWISE_COMPARISONS: list[tuple[str, str, str]] = [
    ("human_from_scratch", "model_from_scratch", "human_vs_model"),
    ("augment_human", "augment_model", "augment_human_vs_augment_model"),
    ("augment_model", "augment_ablation", "two_step_vs_one_step"),
]


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


def _pairwise_subset(df: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    pair = df[df["setting"].isin([left, right])].copy()
    return pair


def _plot_pair(ax, pair_df: pd.DataFrame, left: str, right: str, title: str) -> None:
    left_df = pair_df[pair_df["setting"] == left].set_index("eval_model")
    right_df = pair_df[pair_df["setting"] == right].set_index("eval_model")
    models = sorted(set(left_df.index).intersection(right_df.index))

    if not models:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No overlapping eval models", ha="center", va="center")
        ax.set_xticks([])
        return

    x = np.arange(len(models), dtype=float)
    left_acc = np.array([float(left_df.loc[m, "accuracy"]) for m in models])
    right_acc = np.array([float(right_df.loc[m, "accuracy"]) for m in models])

    left_lo = np.array([float(left_df.loc[m, "ci_low"]) for m in models])
    left_hi = np.array([float(left_df.loc[m, "ci_high"]) for m in models])
    right_lo = np.array([float(right_df.loc[m, "ci_low"]) for m in models])
    right_hi = np.array([float(right_df.loc[m, "ci_high"]) for m in models])

    ax.errorbar(
        x - 0.14,
        left_acc,
        yerr=[left_acc - left_lo, left_hi - left_acc],
        fmt="o",
        capsize=3,
        label=left,
        color="#1f77b4",
    )
    ax.errorbar(
        x + 0.14,
        right_acc,
        yerr=[right_acc - right_lo, right_hi - right_acc],
        fmt="o",
        capsize=3,
        label=right,
        color="#ff7f0e",
    )

    left_base = SETTING_RANDOM_BASELINES[left]
    right_base = SETTING_RANDOM_BASELINES[right]

    ax.scatter(x - 0.14, [left_base] * len(x), marker="x", color="#1f77b4", alpha=0.8)
    ax.scatter(x + 0.14, [right_base] * len(x), marker="x", color="#ff7f0e", alpha=0.8)

    ax.axhline(left_base, linestyle="--", color="#1f77b4", alpha=0.25)
    if abs(right_base - left_base) > 1e-12:
        ax.axhline(right_base, linestyle="--", color="#ff7f0e", alpha=0.25)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", alpha=0.2)


def plot_final5_pairwise(
    results_root: Path | str,
    output_dir: Path | str,
    include_tables: bool = True,
) -> list[Path]:
    """Create required Final5 pairwise plots per generator/dataset/mode."""
    df = collect_final5_results(results_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        return []

    outputs: list[Path] = []

    grouped = df.groupby(["generator", "dataset", "mode"], sort=True)
    for (generator, dataset, mode), group_df in grouped:
        for left, right, title_key in PAIRWISE_COMPARISONS:
            fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))
            pair_df = _pairwise_subset(group_df, left, right)
            pretty_title = title_key.replace("_", " ")
            _plot_pair(
                ax,
                pair_df,
                left,
                right,
                f"{pretty_title}\n{generator} | {dataset} | {mode}",
            )

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.02),
                    ncols=2,
                    frameon=False,
                )
            fig.subplots_adjust(top=0.84, bottom=0.22)

            out_png = out_dir / f"pairwise_{generator}_{dataset}_{mode}_{title_key}.png"
            fig.savefig(out_png, dpi=200)
            plt.close(fig)
            outputs.append(out_png)

            if include_tables:
                per_pair_csv = out_dir / f"pairwise_{generator}_{dataset}_{mode}_{title_key}.csv"
                pair_df.sort_values(["setting", "eval_model"], inplace=False).to_csv(per_pair_csv, index=False)
                outputs.append(per_pair_csv)

    if include_tables:
        full_csv = out_dir / "final5_results_summary.csv"
        df.sort_values(["generator", "dataset", "mode", "setting", "eval_model"]).to_csv(full_csv, index=False)
        outputs.append(full_csv)

    return outputs
