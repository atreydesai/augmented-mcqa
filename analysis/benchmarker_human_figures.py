"""Focused BenchMarker figures for human questions and human-vs-model comparison."""

from __future__ import annotations

import argparse
import ast
import json
import math
import zipfile
from pathlib import Path
from typing import Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


DATASET_ORDER = ["arc_challenge", "gpqa", "mmlu_pro"]
DATASET_LABELS = {"arc_challenge": "ARC-Challenge", "gpqa": "GPQA", "mmlu_pro": "MMLU-Pro"}
RULES_ORDER = [
    "avoid_k_type",
    "avoid_negatives",
    "avoid_repetition",
    "clear_language",
    "equal_length_options",
    "focused_stem",
    "grammatical_consistency",
    "no_absolute_terms",
    "no_all_of_the_above",
    "no_convergence_cues",
    "no_extraneous_info",
    "no_fill_in_blank",
    "no_logical_cues",
    "no_none_of_the_above",
    "no_vague_terms",
    "ordered_options",
    "plausible_distractors",
    "problem_in_stem",
    "single_best_answer",
]


def _iter_jsonl(path: Path) -> Iterator[dict]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            member = next((name for name in zf.namelist() if name.endswith(".jsonl")), None)
            if member is None:
                raise FileNotFoundError(f"No .jsonl file found in {path}")
            with zf.open(member) as handle:
                for raw_line in handle:
                    line = raw_line.decode("utf-8").strip()
                    if line:
                        yield json.loads(line)
        return

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _parse_rule_answers(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def load_rows(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for obj in _iter_jsonl(path):
        writing_flaw = dict(obj.get("writing_flaw", {}) or {})
        answers = _parse_rule_answers(writing_flaw.get("answer"))
        outcomes = [(item.strip().lower() == "pass") for item in answers[: len(RULES_ORDER)]]
        rows.append(
            {
                "dataset": str(obj.get("dataset", "")),
                "config": str(obj.get("config", "")),
                "generator_model": str(obj.get("model", "")),
                "flaw_value": float(writing_flaw.get("value", float("nan"))),
                "n_flaws": sum(not passed for passed in outcomes),
                **{
                    f"rule_{rule}": (outcomes[i] if i < len(outcomes) else np.nan)
                    for i, rule in enumerate(RULES_ORDER)
                },
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No writing flaw rows found in {path}")
    df["has_ge2_flaws"] = df["n_flaws"] >= 2
    rule_cols = [f"rule_{rule}" for rule in RULES_ORDER]
    long = df.melt(
        id_vars=["dataset", "config", "generator_model", "flaw_value"],
        value_vars=rule_cols,
        var_name="rule_col",
        value_name="passed",
    ).dropna(subset=["passed"])
    long["rule"] = long["rule_col"].str.replace("rule_", "", regex=False)
    long["failed"] = ~long["passed"].astype(bool)
    return df, long


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _summary_by_dataset(df: pd.DataFrame, config: str) -> pd.DataFrame:
    subset = df[df["config"] == config]
    return (
        subset.groupby("dataset", observed=True)
        .agg(
            mean_flaws=("n_flaws", "mean"),
            p_ge2=("has_ge2_flaws", "mean"),
            n=("n_flaws", "size"),
        )
        .reindex(DATASET_ORDER)
        .reset_index()
    )


def plot_human_figures(df: pd.DataFrame, long: pd.DataFrame, output_dir: Path) -> None:
    human = df[df["config"] == "human_from_scratch"].copy()
    human_long = long[long["config"] == "human_from_scratch"].copy()
    summary = _summary_by_dataset(df, "human_from_scratch")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(DATASET_ORDER))
    axes[0].bar(x, summary["mean_flaws"], color="#2196F3", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
    axes[0].set_ylabel("Average writing flaws")
    axes[0].set_title("Human MCQs: Mean Flaws")
    axes[1].bar(x, summary["p_ge2"], color="#1976D2", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axes[1].set_ylabel("P(>=2 writing flaws)")
    axes[1].set_title("Human MCQs: Multi-Flaw Rate")
    _save(fig, output_dir, "fig_human_summary.png")

    heat = (
        human_long.groupby(["rule", "dataset"], observed=True)["failed"]
        .mean()
        .reset_index()
        .pivot(index="rule", columns="dataset", values="failed")
        .reindex(index=RULES_ORDER, columns=DATASET_ORDER)
    )
    fig, ax = plt.subplots(figsize=(6.6, 8))
    sns.heatmap(
        heat,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.4,
        cbar_kws={"label": "Failure rate", "shrink": 0.7},
        xticklabels=[DATASET_LABELS[d] for d in DATASET_ORDER],
    )
    ax.set_xlabel("")
    ax.set_ylabel("BenchMarker rule")
    ax.set_title("Human MCQs: Rule Failure Rates")
    _save(fig, output_dir, "fig_human_rule_heatmap.png")

    top = human_long.groupby("rule", observed=True)["failed"].mean().reindex(RULES_ORDER).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.values, color="#2196F3", alpha=0.85)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Failure rate")
    ax.set_title("Human MCQs: Average Rule Failure Rate")
    ax.tick_params(axis="y", labelsize=8)
    _save(fig, output_dir, "fig_human_rule_failures_ranked.png")

    print("Human summary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Human rows used: {len(human):,}")


def plot_human_vs_model(df: pd.DataFrame, long: pd.DataFrame, output_dir: Path) -> None:
    compare = long[long["config"].isin(["human_from_scratch", "model_from_scratch"])].copy()
    compare["source"] = compare["config"].map(
        {"human_from_scratch": "Human", "model_from_scratch": "Avg Model"}
    )

    heat = (
        compare.groupby(["source", "rule"], observed=True)["failed"]
        .mean()
        .reset_index()
        .pivot(index="rule", columns="source", values="failed")
        .reindex(index=RULES_ORDER, columns=["Human", "Avg Model"])
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 8), sharey=True)
    for ax, source in zip(axes, ["Human", "Avg Model"]):
        sns.heatmap(
            heat[[source]],
            ax=ax,
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7},
            linewidths=0.4,
            cbar=source == "Avg Model",
            cbar_kws={"label": "Failure rate", "shrink": 0.7} if source == "Avg Model" else None,
        )
        ax.set_xlabel("")
        ax.set_ylabel("BenchMarker rule" if source == "Human" else "")
        ax.set_title(source)
    fig.suptitle("Rule Failure Rates: Human vs Average Model", y=1.02)
    _save(fig, output_dir, "fig_human_left_avg_model_right_rule_heatmap.png")

    summary = (
        df[df["config"].isin(["human_from_scratch", "model_from_scratch"])]
        .assign(source=lambda frame: frame["config"].map({"human_from_scratch": "Human", "model_from_scratch": "Avg Model"}))
        .groupby(["dataset", "source"], observed=True)
        .agg(mean_flaws=("n_flaws", "mean"), p_ge2=("has_ge2_flaws", "mean"))
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(DATASET_ORDER))
    width = 0.36
    colors = {"Human": "#2196F3", "Avg Model": "#FF9800"}
    for idx, source in enumerate(["Human", "Avg Model"]):
        sub = summary[summary["source"] == source].set_index("dataset").reindex(DATASET_ORDER)
        axes[0].bar(x + (idx - 0.5) * width, sub["mean_flaws"], width=width, color=colors[source], alpha=0.85, label=source)
        axes[1].bar(x + (idx - 0.5) * width, sub["p_ge2"], width=width, color=colors[source], alpha=0.85, label=source)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
        ax.legend(frameon=False)
    axes[0].set_ylabel("Average writing flaws")
    axes[0].set_title("Mean Flaws")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axes[1].set_ylabel("P(>=2 writing flaws)")
    axes[1].set_title("Multi-Flaw Rate")
    _save(fig, output_dir, "fig_human_vs_avg_model_summary.png")

    delta_rows: list[dict[str, float | str]] = []
    for rule in RULES_ORDER:
        sub_h = compare[(compare["source"] == "Human") & (compare["rule"] == rule)]["failed"].astype(float)
        sub_m = compare[(compare["source"] == "Avg Model") & (compare["rule"] == rule)]["failed"].astype(float)
        p_h = float(sub_h.mean())
        p_m = float(sub_m.mean())
        se = math.sqrt(p_h * (1.0 - p_h) / len(sub_h) + p_m * (1.0 - p_m) / len(sub_m))
        delta_rows.append({"rule": rule, "delta": p_m - p_h, "moe": 1.96 * se})
    delta_df = pd.DataFrame(delta_rows).sort_values("delta")
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3" if value < 0 else "#FF9800" for value in delta_df["delta"]]
    ax.barh(
        delta_df["rule"],
        delta_df["delta"],
        xerr=delta_df["moe"],
        color=colors,
        alpha=0.85,
        error_kw={"elinewidth": 1.0, "ecolor": "#333333", "capsize": 2},
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Failure-rate delta (Avg Model - Human)")
    ax.set_title("Rule Failure Delta")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(
        output_dir / "fig_model_from_scratch_minus_human_from_scratch_rule_delta.png",
        dpi=180,
        bbox_inches="tight",
    )
    print(f"Saved {output_dir / 'fig_model_from_scratch_minus_human_from_scratch_rule_delta.png'}")
    _save(fig, output_dir, "fig_human_vs_avg_model_rule_delta.png")

    print("Human vs average model summary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def plot_aug_human_vs_aug_model(df: pd.DataFrame, long: pd.DataFrame, output_dir: Path) -> None:
    compare = long[long["config"].isin(["augment_human", "augment_model"])].copy()
    compare["source"] = compare["config"].map(
        {"augment_human": "Aug-Human", "augment_model": "Aug-Model"}
    )

    heat = (
        compare.groupby(["source", "rule"], observed=True)["failed"]
        .mean()
        .reset_index()
        .pivot(index="rule", columns="source", values="failed")
        .reindex(index=RULES_ORDER, columns=["Aug-Human", "Aug-Model"])
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 8), sharey=True)
    for ax, source in zip(axes, ["Aug-Human", "Aug-Model"]):
        sns.heatmap(
            heat[[source]],
            ax=ax,
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7},
            linewidths=0.4,
            cbar=source == "Aug-Model",
            cbar_kws={"label": "Failure rate", "shrink": 0.7} if source == "Aug-Model" else None,
        )
        ax.set_xlabel("")
        ax.set_ylabel("BenchMarker rule" if source == "Aug-Human" else "")
        ax.set_title(source)
    fig.suptitle("Rule Failure Rates: 10-Choice Aug-Human vs Aug-Model", y=1.02)
    _save(fig, output_dir, "fig_aug_human_left_aug_model_right_rule_heatmap.png")

    summary = (
        df[df["config"].isin(["augment_human", "augment_model"])]
        .assign(source=lambda frame: frame["config"].map({"augment_human": "Aug-Human", "augment_model": "Aug-Model"}))
        .groupby(["dataset", "source"], observed=True)
        .agg(mean_flaws=("n_flaws", "mean"), p_ge2=("has_ge2_flaws", "mean"))
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(DATASET_ORDER))
    width = 0.36
    colors = {"Aug-Human": "#4CAF50", "Aug-Model": "#F44336"}
    for idx, source in enumerate(["Aug-Human", "Aug-Model"]):
        sub = summary[summary["source"] == source].set_index("dataset").reindex(DATASET_ORDER)
        axes[0].bar(x + (idx - 0.5) * width, sub["mean_flaws"], width=width, color=colors[source], alpha=0.85, label=source)
        axes[1].bar(x + (idx - 0.5) * width, sub["p_ge2"], width=width, color=colors[source], alpha=0.85, label=source)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
        ax.legend(frameon=False)
    axes[0].set_ylabel("Average writing flaws")
    axes[0].set_title("Mean Flaws")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axes[1].set_ylabel("P(>=2 writing flaws)")
    axes[1].set_title("Multi-Flaw Rate")
    _save(fig, output_dir, "fig_aug_human_vs_aug_model_summary.png")

    delta_rows: list[dict[str, float | str]] = []
    for rule in RULES_ORDER:
        sub_h = compare[(compare["source"] == "Aug-Human") & (compare["rule"] == rule)]["failed"].astype(float)
        sub_m = compare[(compare["source"] == "Aug-Model") & (compare["rule"] == rule)]["failed"].astype(float)
        p_h = float(sub_h.mean())
        p_m = float(sub_m.mean())
        se = math.sqrt(p_h * (1.0 - p_h) / len(sub_h) + p_m * (1.0 - p_m) / len(sub_m))
        delta_rows.append({"rule": rule, "delta": p_m - p_h, "moe": 1.96 * se})
    delta_df = pd.DataFrame(delta_rows).sort_values("delta")
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4CAF50" if value < 0 else "#F44336" for value in delta_df["delta"]]
    ax.barh(
        delta_df["rule"],
        delta_df["delta"],
        xerr=delta_df["moe"],
        color=colors,
        alpha=0.85,
        error_kw={"elinewidth": 1.0, "ecolor": "#333333", "capsize": 2},
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Failure-rate delta (Aug-Model - Aug-Human)")
    ax.set_title("10-Choice Augmentation Penalty by Rule")
    ax.tick_params(axis="y", labelsize=8)
    _save(fig, output_dir, "fig_aug_human_vs_aug_model_rule_delta.png")

    print("Aug-human vs aug-model summary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create focused BenchMarker human and human-vs-model figures")
    parser.add_argument("--writing-flaw-jsonl", required=True)
    parser.add_argument("--human-output-dir", default="analysis/figures/benchmarker/human")
    parser.add_argument("--comparison-output-dir", default="analysis/figures/benchmarker/human_vs_avg_model")
    parser.add_argument("--aug-comparison-output-dir", default="analysis/figures/benchmarker/aug_human_vs_aug_model")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    df, long = load_rows(Path(args.writing_flaw_jsonl))
    plot_human_figures(df, long, Path(args.human_output_dir))
    plot_human_vs_model(df, long, Path(args.comparison_output_dir))
    plot_aug_human_vs_aug_model(df, long, Path(args.aug_comparison_output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
