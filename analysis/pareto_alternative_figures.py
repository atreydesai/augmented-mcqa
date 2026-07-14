"""Alternative quality tradeoff figures for the Augmented MCQA IRT analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.irt_figures import use_inter_font
from analysis.irt_model import DATASET_LABELS, DATASET_ORDER, MODEL_ORDER


DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "augmented_mcqa_irt"
GENERATOR_SHORT = {
    "openai/gpt-5.2-2025-12-11": "GPT",
    "google/gemini-3.1-pro-preview": "Gemini",
    "together/Qwen/Qwen3.5-397B-A17B": "Qwen",
}
DATASET_MARKERS = {
    "arc_challenge": "^",
    "mmlu_pro": "o",
    "gpqa": "s",
}


def dataset_sort_key(dataset: str) -> tuple[int, str]:
    value = str(dataset)
    return (DATASET_ORDER.index(value), value) if value in DATASET_ORDER else (len(DATASET_ORDER), value)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def load_quality_tables(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    table_dir = output_dir / "tables"
    grouped = pd.read_csv(table_dir / "final_grouped_question_quality.csv")
    ablation = pd.read_csv(table_dir / "final_ablation_question_quality.csv")
    return grouped, ablation


def delta_quality_frame(grouped: pd.DataFrame, ablation: pd.DataFrame, *, include_ablation_panel: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for dataset, group in grouped.groupby("dataset", sort=False):
        human = group[group["source"] == "Human"]
        if human.empty:
            continue
        baseline = human.iloc[0]
        for _, row in group[group["source"] == "Model"].iterrows():
            rows.append(
                {
                    "panel": "Distractor Set Generation",
                    "dataset": dataset,
                    "generator": row["generator"],
                    "label": GENERATOR_SHORT.get(str(row["generator"]), str(row["label"])),
                    "delta_difficulty": float(row["difficulty"] - baseline["difficulty"]),
                    "delta_discrimination": float(row["discrimination"] - baseline["discrimination"]),
                    "delta_flaws": float(row["mean_flaws"] - baseline["mean_flaws"]),
                    "delta_writing_quality": float(baseline["mean_flaws"] - row["mean_flaws"]),
                    "absolute_flaws": float(row["mean_flaws"]),
                }
            )

    for (dataset, generator), group in ablation.groupby(["dataset", "generator"], sort=False):
        human = group[group["source"] == "Augment Human"]
        model = group[group["source"] == "Augment Model"]
        if human.empty or model.empty:
            continue
        baseline = human.iloc[0]
        row = model.iloc[0]
        rows.append(
            {
                "panel": "Distractor Set Extension",
                "dataset": dataset,
                "generator": generator,
                "label": GENERATOR_SHORT.get(str(generator), str(row["label"]).split()[0]),
                "delta_difficulty": float(row["difficulty"] - baseline["difficulty"]),
                "delta_discrimination": float(row["discrimination"] - baseline["discrimination"]),
                "delta_flaws": float(row["mean_flaws"] - baseline["mean_flaws"]),
                "delta_writing_quality": float(baseline["mean_flaws"] - row["mean_flaws"]),
                "absolute_flaws": float(row["mean_flaws"]),
            }
        )

        if include_ablation_panel:
            ablation_row = group[group["source"] == "Ablation"]
            if ablation_row.empty:
                continue
            row = ablation_row.iloc[0]
            rows.append(
                {
                    "panel": "Distractor Set Extension (Ablation)",
                    "dataset": dataset,
                    "generator": generator,
                    "label": GENERATOR_SHORT.get(str(generator), str(row["label"]).split()[0]),
                    "delta_difficulty": float(row["difficulty"] - baseline["difficulty"]),
                    "delta_discrimination": float(row["discrimination"] - baseline["discrimination"]),
                    "delta_flaws": float(row["mean_flaws"] - baseline["mean_flaws"]),
                    "delta_writing_quality": float(baseline["mean_flaws"] - row["mean_flaws"]),
                    "absolute_flaws": float(row["mean_flaws"]),
                }
            )

    return pd.DataFrame(rows)


def _delta_label_style(row: pd.Series) -> tuple[tuple[float, float], str, str]:
    key = (str(row["panel"]), str(row["dataset"]), str(row["label"]))
    if key == ("Distractor Set Extension", "gpqa", "GPT"):
        return (-4.0, -6.0), "right", "top"
    if key == ("Distractor Set Extension", "gpqa", "Gemini"):
        return (7.0, 4.0), "left", "bottom"
    if key == ("Distractor Set Extension", "arc_challenge", "Qwen"):
        return (4.0, -5.0), "left", "top"
    if str(row["label"]) in {"GPT", "Gemini"}:
        return (5.5, 3.0), "left", "bottom"
    return (4.0, 3.0), "left", "bottom"


def _add_better_arrows(ax: plt.Axes) -> None:
    arrow_color = "#777777"
    arrow_kw = {
        "arrowstyle": "-|>",
        "color": arrow_color,
        "linewidth": 0.75,
        "mutation_scale": 7.0,
        "alpha": 0.75,
    }
    ax.annotate(
        "",
        xy=(0.035, 0.94),
        xytext=(0.035, 0.81),
        xycoords="axes fraction",
        arrowprops=arrow_kw,
        annotation_clip=False,
        zorder=6,
    )
    ax.text(
        0.048,
        0.905,
        "Better",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7.0,
        color=arrow_color,
        alpha=0.85,
        zorder=6,
    )
    ax.annotate(
        "",
        xy=(0.965, 0.085),
        xytext=(0.835, 0.085),
        xycoords="axes fraction",
        arrowprops=arrow_kw,
        annotation_clip=False,
        zorder=6,
    )
    ax.text(
        0.90,
        0.115,
        "Better",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=arrow_color,
        alpha=0.85,
        zorder=6,
    )


def plot_delta_scatter(
    grouped: pd.DataFrame,
    ablation: pd.DataFrame,
    path: Path,
    *,
    include_ablation_panel: bool = False,
    fig_height: float = 2.2,
) -> Path:
    use_inter_font()
    delta = delta_quality_frame(grouped, ablation, include_ablation_panel=include_ablation_panel)
    if delta.empty:
        raise ValueError("No rows available for delta scatter.")

    max_abs_quality = max(float(delta["delta_writing_quality"].abs().max()), 0.1)
    norm = TwoSlopeNorm(vmin=-max_abs_quality, vcenter=0.0, vmax=max_abs_quality)
    cmap = plt.get_cmap("RdBu")

    panels = ["Distractor Set Generation", "Distractor Set Extension"]
    if include_ablation_panel:
        panels.append("Distractor Set Extension (Ablation)")

    fig_width = 13.4 if include_ablation_panel else 10.8
    fig, axes = plt.subplots(1, len(panels), figsize=(fig_width, fig_height), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, panel in zip(axes, panels, strict=True):
        frame = delta[delta["panel"] == panel]
        for dataset in sorted(frame["dataset"].unique(), key=dataset_sort_key):
            subset = frame[frame["dataset"] == dataset]
            ax.scatter(
                subset["delta_difficulty"],
                subset["delta_discrimination"],
                c=subset["delta_writing_quality"],
                cmap=cmap,
                norm=norm,
                marker=DATASET_MARKERS.get(str(dataset), "o"),
                s=72,
                edgecolors="#222222",
                linewidths=0.8,
                zorder=3,
            )
        for _, row in frame.iterrows():
            xytext, ha, va = _delta_label_style(row)
            ax.annotate(
                str(row["label"]),
                (float(row["delta_difficulty"]), float(row["delta_discrimination"])),
                xytext=xytext,
                textcoords="offset points",
                fontsize=7.2,
                ha=ha,
                va=va,
                color="#222222",
            )
        ax.axhline(0.0, color="#777777", linewidth=0.7, linestyle="--", alpha=0.65, zorder=1)
        ax.axvline(0.0, color="#777777", linewidth=0.7, linestyle="--", alpha=0.65, zorder=1)
        ax.set_title(panel, fontsize=9.5, pad=2.0)
        ax.set_xlabel("Δ IRT Difficulty", fontsize=9.0, labelpad=0.8)
        ax.grid(True, color="#dddddd", linewidth=0.5, alpha=0.55)
        ax.tick_params(axis="both", labelsize=8.0)
        ax.tick_params(axis="x", pad=1.0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.55)
            spine.set_color("#666666")
    axes[0].set_ylabel("Δ IRT Discriminability", fontsize=9.0)

    x_ranges = {
        "Distractor Set Generation": (-0.105, 0.405),
        "Distractor Set Extension": (-0.125, 0.105),
        "Distractor Set Extension (Ablation)": (-0.345, 0.035),
    }
    for ax, panel in zip(axes, panels, strict=True):
        ax.set_xlim(*x_ranges[panel])
    axes[0].set_ylim(-0.072, 0.072)
    for ax in axes:
        xmax = ax.get_xlim()[1]
        ymax = ax.get_ylim()[1]
        ax.axvspan(0.0, xmax, ymin=0.5, ymax=1.0, facecolor="#2ca25f", alpha=0.075, zorder=0)
    if not include_ablation_panel:
        for ax in axes:
            _add_better_arrows(ax)

    handles = [
        Line2D(
            [0],
            [0],
            marker=DATASET_MARKERS.get(dataset, "o"),
            color="none",
            markerfacecolor="#bdbdbd",
            markeredgecolor="none",
            markersize=6.5,
            label=DATASET_LABELS.get(dataset, dataset),
        )
        for dataset in sorted(delta["dataset"].unique(), key=dataset_sort_key)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.47, 0.165),
        ncols=len(handles),
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        fontsize=8.0,
        handletextpad=0.35,
        columnspacing=1.0,
    )

    fig.subplots_adjust(left=0.065, right=0.865, bottom=0.25, top=0.89, wspace=0.045)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.882, 0.25, 0.016, 0.64])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Δ Writing Quality\n(higher = fewer flaws)", fontsize=8.4)
    cbar.ax.tick_params(labelsize=7.6)
    return _save(fig, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create alternative quality figures from cached tables.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    grouped, ablation = load_quality_tables(args.output_dir)
    final_dir = args.output_dir / "figures" / "final_figures"
    outputs = [
        plot_delta_scatter(grouped, ablation, final_dir / "quality_delta_scatter.png"),
        plot_delta_scatter(
            grouped,
            ablation,
            final_dir / "quality_delta_scatter_with_ablation.png",
            include_ablation_panel=True,
        ),
    ]
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
