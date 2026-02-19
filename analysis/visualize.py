import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import re

import matplotlib.pyplot as plt


# Dataset type display names and colors
DATASET_TYPE_STYLES = {
    "mmlu_pro": {"label": "MMLU-Pro", "color": "#3498db", "marker": "o"},
    "gpqa": {"label": "GPQA", "color": "#e74c3c", "marker": "s"},
    "arc_easy": {"label": "ARC-Easy", "color": "#2ecc71", "marker": "^"},
    "arc_challenge": {"label": "ARC-Challenge", "color": "#9b59b6", "marker": "D"},
}

# Distractor source display names
DISTRACTOR_SOURCE_LABELS = {
    "scratch": "From Scratch",
    "dhuman": "Cond. Human",
    "dmodel": "Cond. Model",
}


def _extract_generator_model_from_dataset_path(dataset_path: str) -> str:
    if not dataset_path:
        return "unknown"

    name = Path(dataset_path).name
    if name.startswith("unified_processed_"):
        remainder = name[len("unified_processed_"):]
        match = re.match(r"(.+)_\d{8}_\d{6}$", remainder)
        if match:
            return match.group(1)
        return remainder or "unknown"
    return name or "unknown"


def _infer_generator_model(base_dir: Path, evaluation_model: str) -> str:
    model_safe = evaluation_model.replace("/", "_")
    pattern = f"{model_safe}_*_*/*/results.json"
    for results_path in sorted(base_dir.glob(pattern)):
        try:
            with open(results_path, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        config = data.get("config", {})
        if isinstance(config, dict):
            dataset_path = config.get("dataset_path", "")
            gen_model = _extract_generator_model_from_dataset_path(dataset_path)
            if gen_model and gen_model != "unknown":
                return gen_model
    return "unknown"


def _resolve_model_labels(
    base_dir: Path,
    model: str,
    generator_model: Optional[str] = None,
    evaluation_model: Optional[str] = None,
) -> tuple[str, str]:
    eval_label = evaluation_model or model
    gen_label = generator_model or _infer_generator_model(base_dir, eval_label)
    return gen_label, eval_label


def _require_generator_model_label(gen_label: str, generator_model: Optional[str]) -> None:
    if not generator_model and gen_label == "unknown":
        raise ValueError(
            "Could not infer generator_model from results metadata; pass generator_model explicitly."
        )


def load_results_file(results_path: Path) -> Optional[Dict]:
    if not results_path.exists():
        return None

    with open(results_path, "r") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    return {
        "accuracy": summary.get("accuracy", 0),
        "correct": summary.get("correct", 0),
        "total": summary.get("total", 0),
    }


def load_summary_file(base_dir: Path, setting_name: str, summary_filename: str) -> Optional[Dict]:
    summary_path = base_dir / setting_name / summary_filename
    if not summary_path.exists():
        return None

    with open(summary_path, "r") as f:
        summary = json.load(f)

    # Support both old and new formats
    return {
        "accuracy": summary.get("acc", summary.get("accuracy", 0)),
        "correct": summary.get("corr", summary.get("correct", 0)),
        "wrong": summary.get("wrong", summary.get("total", 0) - summary.get("correct", 0)),
    }


def _build_results_dir_name(model: str, dataset_type: str, distractor_source: str) -> str:
    return f"{model.replace('/', '_')}_{dataset_type}_{distractor_source}"


def load_3H_plus_M_results(
    base_dir: Path,
    model: Optional[str] = None,
    dataset_type: Optional[str] = None,
    distractor_source: Optional[str] = None,
) -> List[Dict]:
    results = []

    for m in range(0, 7):
        config_str = f"3H{m}M"

        if model and dataset_type and distractor_source:
            dir_name = _build_results_dir_name(model, dataset_type, distractor_source)
            results_path = base_dir / dir_name / config_str / "results.json"
        else:
            results_path = base_dir / config_str / "results.json"

        data = load_results_file(results_path)
        if data:
            results.append({
                "setting": config_str,
                "num_human": 3,
                "num_model": m,
                "total_distractors": 3 + m,
                **data,
            })

    return results


def load_human_only_results(
    base_dir: Path,
    model: Optional[str] = None,
    dataset_type: Optional[str] = None,
    distractor_source: Optional[str] = None,
) -> List[Dict]:
    results = []

    for h in range(1, 4):
        config_str = f"{h}H0M"

        if model and dataset_type and distractor_source:
            dir_name = _build_results_dir_name(model, dataset_type, distractor_source)
            results_path = base_dir / dir_name / config_str / "results.json"
        else:
            results_path = base_dir / config_str / "results.json"

        data = load_results_file(results_path)
        if data:
            results.append({
                "setting": config_str,
                "num_human": h,
                "num_model": 0,
                "total_distractors": h,
                **data,
            })

    return results


def load_model_only_results(
    base_dir: Path,
    model: Optional[str] = None,
    dataset_type: Optional[str] = None,
    distractor_source: Optional[str] = None,
) -> List[Dict]:
    results = []

    for m in range(1, 7):
        config_str = f"0H{m}M"

        if model and dataset_type and distractor_source:
            dir_name = _build_results_dir_name(model, dataset_type, distractor_source)
            results_path = base_dir / dir_name / config_str / "results.json"
        else:
            results_path = base_dir / config_str / "results.json"

        data = load_results_file(results_path)
        if data:
            results.append({
                "setting": config_str,
                "num_human": 0,
                "num_model": m,
                "total_distractors": m,
                **data,
            })

    return results


def _detect_available_configs(
    base_dir: Path,
    model: str,
) -> Dict[str, List[str]]:
    available = {}
    model_safe = model.replace("/", "_")
    prefix = f"{model_safe}_"

    if not base_dir.exists():
        return available

    for d in sorted(base_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        remainder = d.name[len(prefix):]
        # Parse: dataset_type_distractor_source
        for source in ["scratch", "dhuman", "dmodel"]:
            if remainder.endswith(f"_{source}"):
                dt = remainder[: -(len(source) + 1)]
                available.setdefault(dt, []).append(source)
                break

    return available


def _choose_dataset_type(
    available: Dict[str, List[str]],
    dataset_type: Optional[str] = None,
) -> str:
    if dataset_type:
        if dataset_type not in available:
            raise ValueError(
                f"Dataset type '{dataset_type}' not found. Available: {sorted(available.keys())}"
            )
        return dataset_type

    if not available:
        raise ValueError("No dataset types available in results directory")

    if len(available) > 1:
        raise ValueError(
            "Multiple dataset types found; specify dataset_type explicitly. "
            f"Available: {sorted(available.keys())}"
        )

    return sorted(available.keys())[0]


def _require_sources(
    available: Dict[str, List[str]],
    dataset_type: str,
    required_sources: List[str],
    context: str,
) -> None:
    present = set(available.get(dataset_type, []))
    missing = [s for s in required_sources if s not in present]
    if missing:
        raise ValueError(
            f"{context}: missing required sources for dataset={dataset_type}: {missing}. "
            f"Available: {sorted(present)}"
        )


def _load_accuracy(
    base_dir: Path,
    model: str,
    dataset_type: str,
    distractor_source: str,
    num_human: int,
    num_model: int,
) -> Optional[float]:
    config_str = f"{num_human}H{num_model}M"
    dir_name = _build_results_dir_name(model, dataset_type, distractor_source)
    results_path = base_dir / dir_name / config_str / "results.json"
    data = load_results_file(results_path)
    return None if data is None else data["accuracy"]


def _collect_xy(
    base_dir: Path,
    model: str,
    dataset_type: str,
    source: str,
    configs: List[tuple[int, int]],
    x_values: Optional[List[float]] = None,
    strict: bool = True,
) -> tuple[List[float], List[float]]:
    if x_values is None:
        x_values = [h + m for h, m in configs]

    xs: List[float] = []
    ys: List[float] = []
    for x, (h, m) in zip(x_values, configs):
        acc = _load_accuracy(base_dir, model, dataset_type, source, h, m)
        if acc is None:
            if strict:
                config_str = f"{h}H{m}M"
                dir_name = _build_results_dir_name(model, dataset_type, source)
                results_path = base_dir / dir_name / config_str / "results.json"
                raise FileNotFoundError(f"Missing expected results file: {results_path}")
            continue
        xs.append(x)
        ys.append(acc)
    return xs, ys


def plot_rq1_combined(
    base_dir: Path,
    model: str,
    distractor_source: str = "scratch",
    output_dir: Optional[Path] = None,
    show: bool = False,
    dataset_type: Optional[str] = None,
    generator_model: Optional[str] = None,
    evaluation_model: Optional[str] = None,
):
    """
    RQ1 on a single comparison chart:
    - Human | q,a (3 points)
    - Model | q,a (6 points)
    - Model | q,a,dhuman (6 points)
    - Model | q,a,dmodel (6 points)
    """
    base_dir = Path(base_dir)
    gen_label, eval_label = _resolve_model_labels(
        base_dir,
        model,
        generator_model=generator_model,
        evaluation_model=evaluation_model,
    )
    _require_generator_model_label(gen_label, generator_model)
    available = _detect_available_configs(base_dir, model)
    chosen_dt = _choose_dataset_type(available, dataset_type)
    _require_sources(available, chosen_dt, ["scratch", "dhuman", "dmodel"], "RQ1")

    line_specs = [
        (
            "Human | q,a",
            "scratch",
            [(1, 0), (2, 0), (3, 0)],
            "#1f77b4",
            "o",
        ),
        (
            "Model | q,a",
            "scratch",
            [(0, m) for m in range(1, 7)],
            "#ff7f0e",
            "s",
        ),
        (
            "Model | q,a,dhuman",
            "dhuman",
            [(0, m) for m in range(1, 7)],
            "#2ca02c",
            "^",
        ),
        (
            "Model | q,a,dmodel",
            "dmodel",
            [(0, m) for m in range(1, 7)],
            "#d62728",
            "D",
        ),
    ]

    fig, ax = plt.subplots(figsize=(11, 7))
    for label, source, configs, color, marker in line_specs:
        x, y = _collect_xy(base_dir, model, chosen_dt, source, configs)
        ax.plot(
            x,
            y,
            label=label,
            color=color,
            marker=marker,
            linewidth=2.5,
            markersize=8,
        )

    ax.set_xlabel("Number of Distractors", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title(
        f"RQ1 Comparison | dataset={chosen_dt}\nGen: {gen_label} | Eval: {eval_label}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=11, frameon=True)
    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "rq1_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()


def plot_rq2_human_distractors(
    base_dir: Path,
    model: str,
    output_dir: Optional[Path] = None,
    show: bool = False,
    dataset_type: Optional[str] = None,
    generator_model: Optional[str] = None,
    evaluation_model: Optional[str] = None,
):
    """
    RQ2 on a single chart with the requested composite comparisons:
    - 3 (Human | q,a) + 6 (Model | q,a,dhuman)
    - 3 (Model | q,a) + 6 (Model | q,a,dmodel)
    """
    base_dir = Path(base_dir)
    gen_label, eval_label = _resolve_model_labels(
        base_dir,
        model,
        generator_model=generator_model,
        evaluation_model=evaluation_model,
    )
    _require_generator_model_label(gen_label, generator_model)
    available = _detect_available_configs(base_dir, model)
    chosen_dt = _choose_dataset_type(available, dataset_type)
    _require_sources(available, chosen_dt, ["scratch", "dhuman", "dmodel"], "RQ2")

    stage_x = list(range(1, 10))

    line_a_sources = ["scratch"] * 3 + ["dhuman"] * 6
    line_a_configs = (
        [(1, 0), (2, 0), (3, 0)]
        + [(0, m) for m in range(1, 7)]
    )

    line_b_sources = ["scratch"] * 3 + ["dmodel"] * 6
    line_b_configs = (
        [(0, 1), (0, 2), (0, 3)]
        + [(0, m) for m in range(1, 7)]
    )

    def collect_composite(sources: List[str], configs: List[tuple[int, int]]) -> tuple[List[int], List[float]]:
        xs: List[int] = []
        ys: List[float] = []
        for idx, (source, (h, m)) in enumerate(zip(sources, configs), start=1):
            acc = _load_accuracy(base_dir, model, chosen_dt, source, h, m)
            if acc is None:
                config_str = f"{h}H{m}M"
                dir_name = _build_results_dir_name(model, chosen_dt, source)
                results_path = base_dir / dir_name / config_str / "results.json"
                raise FileNotFoundError(f"Missing expected results file: {results_path}")
            xs.append(idx)
            ys.append(acc)
        return xs, ys

    x_a, y_a = collect_composite(line_a_sources, line_a_configs)
    x_b, y_b = collect_composite(line_b_sources, line_b_configs)

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        x_a,
        y_a,
        label="3 (Human | q,a) + 6 (Model | q,a,dhuman)",
        color="#1f77b4",
        marker="o",
        linewidth=2.5,
        markersize=7,
    )
    ax.plot(
        x_b,
        y_b,
        label="3 (Model | q,a) + 6 (Model | q,a,dmodel)",
        color="#d62728",
        marker="s",
        linewidth=2.5,
        markersize=7,
    )

    ax.axvline(3.5, color="#555555", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("Comparison Stage (1-3 baseline, 4-9 conditioned)", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title(
        f"RQ2 Composite Comparison | dataset={chosen_dt}\nGen: {gen_label} | Eval: {eval_label}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(stage_x)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10, frameon=True)
    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "rq2_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()


def plot_rq3_model_distractors(
    base_dir: Path,
    model: str,
    output_dir: Optional[Path] = None,
    show: bool = False,
    dataset_type: Optional[str] = None,
    generator_model: Optional[str] = None,
    evaluation_model: Optional[str] = None,
):
    """
    RQ3 on a single chart: 3H+M progressive curves by source.
    """
    base_dir = Path(base_dir)
    gen_label, eval_label = _resolve_model_labels(
        base_dir,
        model,
        generator_model=generator_model,
        evaluation_model=evaluation_model,
    )
    _require_generator_model_label(gen_label, generator_model)
    available = _detect_available_configs(base_dir, model)
    chosen_dt = _choose_dataset_type(available, dataset_type)
    _require_sources(available, chosen_dt, ["scratch", "dhuman", "dmodel"], "RQ3")

    line_specs = [
        ("3H + M | q,a", "scratch", "#ff7f0e", "o"),
        ("3H + M | q,a,dhuman", "dhuman", "#2ca02c", "^"),
        ("3H + M | q,a,dmodel", "dmodel", "#d62728", "D"),
    ]

    fig, ax = plt.subplots(figsize=(11, 7))
    configs = [(3, m) for m in range(0, 7)]
    x_values = [3 + m for m in range(0, 7)]
    for label, source, color, marker in line_specs:
        x, y = _collect_xy(base_dir, model, chosen_dt, source, configs, x_values=x_values)
        ax.plot(
            x,
            y,
            label=label,
            color=color,
            marker=marker,
            linewidth=2.5,
            markersize=7,
        )

    ax.set_xlabel("Total Distractors (3H + M)", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title(
        f"RQ3 Progressive Source Comparison | dataset={chosen_dt}\nGen: {gen_label} | Eval: {eval_label}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks([3, 4, 5, 6, 7, 8, 9])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10, frameon=True)
    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "rq3_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()


def plot_branching_comparison(
    base_dir: Path,
    model: str,
    output_dir: Optional[Path] = None,
    show: bool = False,
    dataset_type: Optional[str] = None,
    generator_model: Optional[str] = None,
    evaluation_model: Optional[str] = None,
):
    """
    Branching view on a single chart:
    - Red line: D1 -> D1 + D2 -> D1 + D2 + D3
    - Blue solid line (no-human branch): M0_1 -> M0_1+M0_2 -> ... -> M0_1..M0_6
    - Blue lines (anchored on corresponding red node):
      - D1 + M1_1 -> ... -> D1 + M1_1..M1_5
      - D1 + D2 + M2_1 -> ... -> D1 + D2 + M2_1..M2_4
      - D1 + D2 + D3 + M3_1 -> ... -> D1 + D2 + D3 + M3_1..M3_3
    """
    base_dir = Path(base_dir)
    gen_label, eval_label = _resolve_model_labels(
        base_dir,
        model,
        generator_model=generator_model,
        evaluation_model=evaluation_model,
    )
    _require_generator_model_label(gen_label, generator_model)
    available = _detect_available_configs(base_dir, model)
    chosen_dt = _choose_dataset_type(available, dataset_type)
    _require_sources(available, chosen_dt, ["dhuman", "scratch"], "Branching")
    branch_source = "dhuman"
    model_only_source = "scratch"

    fig, ax = plt.subplots(figsize=(12, 7))

    # Red human-only branch
    human_configs = [(1, 0), (2, 0), (3, 0)]
    x_human = [1, 2, 3]
    hx, hy = _collect_xy(base_dir, model, chosen_dt, branch_source, human_configs, x_values=x_human)
    human_accuracy_by_prefix = {int(x): y for x, y in zip(hx, hy)}
    ax.plot(
        hx,
        hy,
        label="Human branch: D1 -> D1+D2 -> D1+D2+D3",
        color="#d62728",
        marker="o",
        linewidth=2.8,
        markersize=7,
    )

    # Blue model-only branch: M0_1 -> M0_1+M0_2 -> ... -> M0_1..M0_6
    model_only_configs = [(0, m) for m in range(1, 7)]
    model_only_x = [m for _, m in model_only_configs]
    mx, my = _collect_xy(
        base_dir,
        model,
        chosen_dt,
        model_only_source,
        model_only_configs,
        x_values=model_only_x,
    )
    ax.plot(
        mx,
        my,
        label="M0_1..M0_k (no human prefix)",
        color="#1f77b4",
        marker="o",
        linestyle="-",
        linewidth=2.6,
        markersize=6,
    )

    # Blue model branches
    blue_specs = {
        1: {"marker": "s", "linestyle": ":", "label": "D1 + M1_1..M1_k"},
        2: {"marker": "^", "linestyle": "--", "label": "D1+D2 + M2_1..M2_k"},
        3: {"marker": "D", "linestyle": "-.", "label": "D1+D2+D3 + M3_1..M3_k"},
    }
    for h, style in blue_specs.items():
        max_m = 6 - h
        configs = [(h, m) for m in range(1, max_m + 1)]
        x_vals = [h + m for _, m in configs]
        x, y = _collect_xy(base_dir, model, chosen_dt, branch_source, configs, x_values=x_vals)

        # Anchor each blue branch at its corresponding red human-only node.
        anchor_y = human_accuracy_by_prefix.get(h)
        if anchor_y is None:
            raise ValueError(
                f"Missing anchor point for human prefix h={h} in dataset={chosen_dt}."
            )
        branch_x = list(x)
        branch_y = list(y)
        branch_x = [h] + branch_x
        branch_y = [anchor_y] + branch_y
        ax.plot(
            branch_x,
            branch_y,
            label=style["label"],
            color="#1f77b4",
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.4,
            markersize=6,
        )

    ax.set_xlabel("Total Distractors in Evaluated Option Set", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    branch_source_label = DISTRACTOR_SOURCE_LABELS.get(branch_source, branch_source)
    model_source_label = DISTRACTOR_SOURCE_LABELS.get(model_only_source, model_only_source)
    source_title = f"{branch_source_label} branches + {model_source_label} M0"
    ax.set_title(
        (
            f"Branching Comparison | dataset={chosen_dt} | source={source_title}\n"
            f"Gen: {gen_label} | Eval: {eval_label}"
        ),
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, frameon=True)
    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "branching_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()


def plot_difficulty_comparison(
    base_dir: Path,
    model: str,
    output_dir: Optional[Path] = None,
    show: bool = False,
    distractor_source: str = "scratch",
    generator_model: Optional[str] = None,
    evaluation_model: Optional[str] = None,
):
    """Difficulty comparison on one chart (e.g., ARC-Easy vs ARC-Challenge)."""
    base_dir = Path(base_dir)
    gen_label, eval_label = _resolve_model_labels(
        base_dir,
        model,
        generator_model=generator_model,
        evaluation_model=evaluation_model,
    )
    _require_generator_model_label(gen_label, generator_model)
    available = _detect_available_configs(base_dir, model)
    if not available:
        raise ValueError(f"No results found for model={model} in {base_dir}")

    preferred = [dt for dt in ["arc_easy", "arc_challenge"] if dt in available]
    if len(preferred) < 2:
        raise ValueError(
            "Difficulty comparison requires both arc_easy and arc_challenge datasets."
        )
    dataset_types = preferred

    fig, ax = plt.subplots(figsize=(11, 7))
    configs = [(3, m) for m in range(0, 7)]
    x_values = [3 + m for m in range(0, 7)]
    for dt in dataset_types:
        if distractor_source not in available.get(dt, []):
            raise ValueError(
                f"Missing source '{distractor_source}' for dataset={dt}. "
                f"Available: {sorted(available.get(dt, []))}"
            )
        style = DATASET_TYPE_STYLES.get(dt, {"label": dt, "color": "#555555", "marker": "o"})
        x, y = _collect_xy(base_dir, model, dt, distractor_source, configs, x_values=x_values)
        ax.plot(
            x,
            y,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.5,
            markersize=7,
        )

    ax.set_xlabel("Total Distractors (3H + M)", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    source_label = DISTRACTOR_SOURCE_LABELS.get(distractor_source, distractor_source)
    ax.set_title(
        (
            f"Difficulty Comparison | source={source_label}\n"
            f"Gen: {gen_label} | Eval: {eval_label}"
        ),
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks([3, 4, 5, 6, 7, 8, 9])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10, frameon=True)
    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "difficulty_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close()


def plot_all_rq(
    base_dir: Path,
    model: str = "",
    output_dir: Optional[Path] = None,
    show: bool = False,
    dataset_type: Optional[str] = None,
    generator_model: Optional[str] = None,
    evaluation_model: Optional[str] = None,
):
    base_dir = Path(base_dir)

    if output_dir is None:
        output_dir = base_dir / "plots"

    if not model:
        raise ValueError("model must be provided explicitly (auto-detection disabled)")

    available = _detect_available_configs(base_dir, model)
    if not available:
        raise ValueError(f"No results found for model={model} in {base_dir}")

    print(f"Detected model: {model}")
    print(f"Detected dataset types: {list(available.keys())}")
    chosen_dt = _choose_dataset_type(available, dataset_type)
    print(f"Using dataset for RQ overlays: {chosen_dt}")
    gen_label, eval_label = _resolve_model_labels(
        base_dir,
        model,
        generator_model=generator_model,
        evaluation_model=evaluation_model,
    )
    _require_generator_model_label(gen_label, generator_model)
    print(f"Inferred generator model: {gen_label}")
    print(f"Evaluation model: {eval_label}")

    print("\nGenerating RQ1 single-chart comparison...")
    plot_rq1_combined(
        base_dir,
        model,
        output_dir=output_dir,
        show=show,
        dataset_type=chosen_dt,
        generator_model=gen_label,
        evaluation_model=eval_label,
    )

    print("Generating RQ2 single-chart comparison...")
    plot_rq2_human_distractors(
        base_dir,
        model,
        output_dir=output_dir,
        show=show,
        dataset_type=chosen_dt,
        generator_model=gen_label,
        evaluation_model=eval_label,
    )

    print("Generating RQ3 single-chart comparison...")
    plot_rq3_model_distractors(
        base_dir,
        model,
        output_dir=output_dir,
        show=show,
        dataset_type=chosen_dt,
        generator_model=gen_label,
        evaluation_model=eval_label,
    )

    print("Generating Branching single-chart comparison...")
    plot_branching_comparison(
        base_dir,
        model,
        output_dir=output_dir,
        show=show,
        dataset_type=chosen_dt,
        generator_model=gen_label,
        evaluation_model=eval_label,
    )

    print(f"\nAll plots saved to: {output_dir}")
