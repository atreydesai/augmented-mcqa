"""Many-facet IRT analysis for collected Augmented MCQA evaluations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_from_disk
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from utils.constants import EVALUATED_STORE_MANIFEST, MODE_CHOICES, SETTING_NAMES, SETTING_SPECS


DEFAULT_REFERENCE_SETTING = "human_from_scratch"
DEFAULT_REFERENCE_EVALUATOR = "vllm/nvidia/NVIDIA-Nemotron-Nano-9B-v2"
DEFAULT_OUTPUT_DIR = Path("results/augmented_mcqa_irt")
DEFAULT_ITEM_FIT_LIMIT = 50
DEFAULT_ITEM_PRIOR_SD = 3.0
SETTING_RANDOM_BASELINES = {
    setting: 1.0 / int(spec["num_choices"]) for setting, spec in SETTING_SPECS.items()
}
SETTING_DISPLAY = {
    "human_from_scratch": "Human From Scratch",
    "model_from_scratch": "Model From Scratch",
    "augment_human": "Augment Human",
    "augment_model": "Augment Model",
    "augment_ablation": "Augment Ablation",
}
EVALUATOR_DISPLAY = {
    "vllm/Qwen/Qwen3-4B-Instruct-2507": "Qwen3-4B",
    "vllm/allenai/Olmo-3-7B-Instruct": "Olmo-3-7B",
    "vllm/meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "vllm/nvidia/NVIDIA-Nemotron-Nano-9B-v2": "Nemotron-9B",
}


def _csv_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return values or None


def _safe_name(value: str) -> str:
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _display_setting(value: str) -> str:
    return SETTING_DISPLAY.get(str(value), str(value))


def _display_evaluator(value: str) -> str:
    return EVALUATOR_DISPLAY.get(str(value), str(value))


def _iter_group_roots(root: Path | str) -> Iterable[tuple[Path, dict[str, object]]]:
    for manifest_path in sorted(Path(root).rglob(EVALUATED_STORE_MANIFEST)):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        yield manifest_path.parent, payload


def load_irt_frame(
    root: Path | str,
    *,
    generators: list[str] | None = None,
    evaluators: list[str] | None = None,
    datasets: list[str] | None = None,
    settings: list[str] | None = None,
    modes: list[str] | None = None,
) -> pd.DataFrame:
    generator_filter = set(generators or [])
    evaluator_filter = set(evaluators or [])
    dataset_filter = set(datasets or [])
    setting_filter = set(settings or [])
    mode_filter = set(modes or ["full_question"])
    frames: list[pd.DataFrame] = []

    for group_root, manifest in _iter_group_roots(root):
        generator = str(manifest.get("generation_model", "") or "")
        evaluator = str(manifest.get("evaluation_model", "") or "")
        if generator_filter and generator not in generator_filter:
            continue
        if evaluator_filter and evaluator not in evaluator_filter:
            continue
        manifest_datasets = list(manifest.get("dataset_types") or [])
        manifest_settings = list(manifest.get("settings") or [])
        manifest_modes = list(manifest.get("modes") or [])
        for dataset in manifest_datasets:
            if dataset_filter and dataset not in dataset_filter:
                continue
            for setting in manifest_settings:
                if setting_filter and setting not in setting_filter:
                    continue
                for mode in manifest_modes:
                    if mode_filter and mode not in mode_filter:
                        continue
                    dataset_path = group_root / dataset / setting / mode
                    if not dataset_path.exists():
                        continue
                    frame = load_from_disk(str(dataset_path)).to_pandas()
                    if frame.empty:
                        continue
                    frame = frame[
                        [
                            "sample_id",
                            "question",
                            "dataset_type",
                            "evaluation_prediction",
                            "evaluation_is_correct",
                            "evaluation_status",
                            "num_choices",
                            "setting",
                        ]
                    ].copy()
                    frame["generator"] = generator
                    frame["evaluator"] = evaluator
                    frame["dataset"] = dataset
                    frame["mode"] = mode
                    frames.append(frame)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["evaluation_prediction"] = df["evaluation_prediction"].fillna("").astype(str)
    df = df[df["evaluation_prediction"].str.strip() != ""].copy()
    if df.empty:
        return df
    df["correct"] = df["evaluation_is_correct"].fillna(False).astype(bool).astype(int)
    df["choice_count"] = df["num_choices"].astype(int)
    df["guessing"] = df["setting"].map(SETTING_RANDOM_BASELINES).astype(float)
    df["choice_group"] = np.where(df["choice_count"] <= 4, "4-choice", "10-choice")
    df["item_id"] = df["sample_id"].astype(str)
    df["obs_id"] = np.arange(len(df), dtype=int)
    duplicate_cols = ["generator", "evaluator", "item_id", "setting"]
    if df.duplicated(duplicate_cols).any():
        dupes = int(df.duplicated(duplicate_cols).sum())
        raise ValueError(f"Found {dupes} duplicated evaluator judgments across {duplicate_cols}.")
    return df.reset_index(drop=True)


@dataclass(frozen=True)
class ParamBlock:
    name: str
    levels: tuple[str, ...]
    reference: str
    start: int

    @property
    def free_levels(self) -> tuple[str, ...]:
        return tuple(level for level in self.levels if level != self.reference)


@dataclass(frozen=True)
class IRTDesign:
    frame: pd.DataFrame
    X: csr_matrix
    y: np.ndarray
    c: np.ndarray
    blocks: dict[str, ParamBlock]
    interaction: bool
    param_count: int
    item_reference: str
    item_column_indices: np.ndarray
    facet_column_indices: np.ndarray


@dataclass
class FitResult:
    design: IRTDesign
    beta: np.ndarray
    objective: float
    success: bool
    message: str
    iterations: int
    hessian: np.ndarray
    covariance: np.ndarray | None
    standard_errors: np.ndarray | None
    facet_covariance: np.ndarray | None
    item_information_diag: np.ndarray | None
    probabilities: np.ndarray
    eta: np.ndarray
    fitted_frame: pd.DataFrame
    log_likelihood: float
    aic: float
    bic: float


def _resolve_reference(levels: list[str], preferred: str | None) -> str:
    if preferred and preferred in levels:
        return preferred
    return levels[0]


def _ordered_levels(values: Iterable[str], preferred_order: Iterable[str] | None = None) -> list[str]:
    raw = sorted({str(value) for value in values})
    if preferred_order is None:
        return raw
    order = {value: idx for idx, value in enumerate(preferred_order)}
    return sorted(raw, key=lambda value: (order.get(value, len(order)), value))


def validate_identification(frame: pd.DataFrame) -> None:
    pairs = [
        ("generator", "evaluator"),
        ("generator", "setting"),
        ("evaluator", "setting"),
    ]
    for left, right in pairs:
        if left not in frame.columns or right not in frame.columns:
            continue
        if frame[left].nunique() < 2 or frame[right].nunique() < 2:
            raise ValueError(
                f"{left} and {right} must each have at least two levels after filtering; "
                f"found {frame[left].nunique()} {left} level(s) and {frame[right].nunique()} {right} level(s)."
            )
        counts = frame.groupby(left, sort=False)[right].nunique()
        bad = counts[counts < 2]
        if not bad.empty:
            level = str(bad.index[0])
            raise ValueError(
                f"{left.capitalize()} {level} co-occurs with only one {right} level; "
                "effects are not separately identified."
            )
        reverse_counts = frame.groupby(right, sort=False)[left].nunique()
        reverse_bad = reverse_counts[reverse_counts < 2]
        if not reverse_bad.empty:
            level = str(reverse_bad.index[0])
            raise ValueError(
                f"{right.capitalize()} {level} co-occurs with only one {left} level; "
                "effects are not separately identified."
            )


def build_design(
    frame: pd.DataFrame,
    *,
    reference_setting: str = DEFAULT_REFERENCE_SETTING,
    reference_evaluator: str = DEFAULT_REFERENCE_EVALUATOR,
    interaction: bool = False,
) -> IRTDesign:
    if frame.empty:
        raise ValueError("Cannot build an IRT design from an empty frame.")
    work = frame.copy()
    generator_levels = _ordered_levels(work["generator"])
    item_levels = _ordered_levels(work["item_id"])
    setting_levels = _ordered_levels(work["setting"], SETTING_NAMES)
    evaluator_levels = _ordered_levels(work["evaluator"])

    generator_ref = generator_levels[0]
    item_ref = item_levels[0]
    setting_ref = _resolve_reference(setting_levels, reference_setting)
    evaluator_ref = _resolve_reference(evaluator_levels, reference_evaluator)

    blocks: dict[str, ParamBlock] = {}
    start = 0
    for name, levels, reference in (
        ("generator", tuple(generator_levels), generator_ref),
        ("item", tuple(item_levels), item_ref),
        ("setting", tuple(setting_levels), setting_ref),
        ("evaluator", tuple(evaluator_levels), evaluator_ref),
    ):
        blocks[name] = ParamBlock(name=name, levels=levels, reference=reference, start=start)
        start += max(0, len(levels) - 1)
    if interaction:
        blocks["evaluator_10choice"] = ParamBlock(
            name="evaluator_10choice",
            levels=tuple(evaluator_levels),
            reference=evaluator_ref,
            start=start,
        )
        start += max(0, len(evaluator_levels) - 1)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def add_block(level_series: pd.Series, block_name: str, sign: float, mask: np.ndarray | None = None) -> None:
        block = blocks[block_name]
        index = {level: block.start + idx for idx, level in enumerate(block.free_levels)}
        effective_mask = np.ones(len(work), dtype=bool) if mask is None else mask
        for row_idx, level in enumerate(level_series.astype(str)):
            if not effective_mask[row_idx] or level == block.reference:
                continue
            rows.append(row_idx)
            cols.append(index[level])
            vals.append(sign)

    add_block(work["generator"], "generator", +1.0)
    add_block(work["item_id"], "item", -1.0)
    add_block(work["setting"], "setting", -1.0)
    add_block(work["evaluator"], "evaluator", -1.0)
    if interaction:
        add_block(
            work["evaluator"],
            "evaluator_10choice",
            -1.0,
            mask=(work["choice_group"] == "10-choice").to_numpy(),
        )

    X = csr_matrix((vals, (rows, cols)), shape=(len(work), start), dtype=float)
    item_block = blocks["item"]
    item_cols = np.arange(item_block.start, item_block.start + len(item_block.free_levels), dtype=int)
    facet_cols = np.array([idx for idx in range(start) if idx not in set(item_cols.tolist())], dtype=int)
    return IRTDesign(
        frame=work,
        X=X,
        y=work["correct"].to_numpy(dtype=float),
        c=work["guessing"].to_numpy(dtype=float),
        blocks=blocks,
        interaction=interaction,
        param_count=start,
        item_reference=item_ref,
        item_column_indices=item_cols,
        facet_column_indices=facet_cols,
    )


def _fit_objective(beta: np.ndarray, design: IRTDesign) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eta = np.asarray(design.X @ beta).reshape(-1)
    u = 1.0 / (1.0 + np.exp(-eta))
    p = design.c + (1.0 - design.c) * u
    p = np.clip(p, 1e-8, 1.0 - 1e-8)

    du = u * (1.0 - u)
    d2u = du * (1.0 - 2.0 * u)
    dp = (1.0 - design.c) * du
    d2p = (1.0 - design.c) * d2u

    y = design.y
    nll = -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    dlog_dp = (y - p) / (p * (1.0 - p))
    d2log_dp2 = -(y / (p**2)) - ((1.0 - y) / ((1.0 - p) ** 2))
    grad_eta = -(dlog_dp * dp)
    hess_eta = -(d2log_dp2 * (dp**2) + dlog_dp * d2p)
    return nll, grad_eta, hess_eta, eta, p


def _objective_only(beta: np.ndarray, design: IRTDesign, item_prior_sd: float | None) -> float:
    nll, _, _, _, _ = _fit_objective(beta, design)
    if item_prior_sd is None or item_prior_sd <= 0.0 or len(design.item_column_indices) == 0:
        return nll
    item_beta = beta[design.item_column_indices]
    variance = float(item_prior_sd) ** 2
    return float(nll + 0.5 * np.sum((item_beta**2) / variance))


def _gradient_only(beta: np.ndarray, design: IRTDesign, item_prior_sd: float | None) -> np.ndarray:
    _, grad_eta, _, _, _ = _fit_objective(beta, design)
    grad = np.asarray(design.X.T @ grad_eta).reshape(-1)
    if item_prior_sd is not None and item_prior_sd > 0.0 and len(design.item_column_indices) > 0:
        grad = grad.copy()
        grad[design.item_column_indices] += beta[design.item_column_indices] / (float(item_prior_sd) ** 2)
    return grad


def _information_blocks(
    beta: np.ndarray,
    design: IRTDesign,
    *,
    item_prior_sd: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, hess_eta, _, _ = _fit_objective(beta, design)
    facet_cols = design.facet_column_indices
    item_cols = design.item_column_indices

    X_f = design.X[:, facet_cols]
    weighted_f = X_f.multiply(hess_eta[:, None])
    h_ff = (X_f.T @ weighted_f).toarray()

    if len(item_cols) == 0:
        return 0.5 * (h_ff + h_ff.T), np.zeros((len(facet_cols), 0), dtype=float), np.zeros(0, dtype=float)

    X_i = design.X[:, item_cols]
    weighted_i = X_i.multiply(hess_eta[:, None])
    h_fi = (X_f.T @ weighted_i).toarray()
    h_ii_diag = np.asarray(weighted_i.multiply(X_i).sum(axis=0)).reshape(-1)
    if item_prior_sd is not None and item_prior_sd > 0.0 and len(h_ii_diag) > 0:
        h_ii_diag = h_ii_diag + (1.0 / (float(item_prior_sd) ** 2))
    return 0.5 * (h_ff + h_ff.T), h_fi, h_ii_diag


def _schur_facet_covariance(
    h_ff: np.ndarray,
    h_fi: np.ndarray,
    h_ii_diag: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if h_ff.size == 0:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float)
    safe_diag = np.where(h_ii_diag > 1e-10, h_ii_diag, np.nan)
    if h_fi.size == 0:
        schur = h_ff
    else:
        reduced = h_fi * np.nan_to_num(1.0 / safe_diag, nan=0.0)[None, :]
        schur = h_ff - reduced @ h_fi.T
    try:
        cov = np.linalg.pinv(schur)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        return cov, se
    except np.linalg.LinAlgError:
        return None, None


def fit_design(
    design: IRTDesign,
    *,
    maxiter: int = 2000,
    maxfun: int = 50000,
    gtol: float = 1e-5,
    init_beta: np.ndarray | None = None,
    item_prior_sd: float | None = DEFAULT_ITEM_PRIOR_SD,
) -> FitResult:
    beta0 = np.zeros(design.param_count, dtype=float) if init_beta is None else np.asarray(init_beta, dtype=float).copy()
    result = minimize(
        _objective_only,
        beta0,
        args=(design, item_prior_sd),
        method="L-BFGS-B",
        jac=_gradient_only,
        options={"maxiter": int(maxiter), "maxfun": int(maxfun), "gtol": float(gtol)},
    )
    nll, _, _, eta, probs = _fit_objective(result.x, design)
    h_ff, h_fi, h_ii_diag = _information_blocks(result.x, design, item_prior_sd=item_prior_sd)
    facet_covariance, facet_standard_errors = _schur_facet_covariance(h_ff, h_fi, h_ii_diag)
    standard_errors = np.full(design.param_count, np.nan, dtype=float)
    if len(design.item_column_indices) > 0:
        item_se = np.sqrt(np.clip(np.where(h_ii_diag > 1e-10, 1.0 / h_ii_diag, np.nan), 0.0, None))
        standard_errors[design.item_column_indices] = item_se
    if facet_standard_errors is not None and len(design.facet_column_indices) > 0:
        standard_errors[design.facet_column_indices] = facet_standard_errors
    fitted = design.frame.copy()
    fitted["eta"] = eta
    fitted["probability"] = probs
    fitted["residual"] = fitted["correct"].astype(float) - probs
    fitted["variance"] = probs * (1.0 - probs)
    log_likelihood = -float(nll)
    aic = 2.0 * design.param_count - 2.0 * log_likelihood
    bic = math.log(len(fitted)) * design.param_count - 2.0 * log_likelihood
    return FitResult(
        design=design,
        beta=result.x,
        objective=float(nll),
        success=bool(result.success),
        message=str(result.message),
        iterations=int(getattr(result, "nit", 0) or 0),
        hessian=h_ff,
        covariance=None,
        standard_errors=standard_errors,
        facet_covariance=facet_covariance,
        item_information_diag=h_ii_diag,
        probabilities=probs,
        eta=eta,
        fitted_frame=fitted,
        log_likelihood=log_likelihood,
        aic=float(aic),
        bic=float(bic),
    )


def _block_frame(fit: FitResult, block_name: str) -> pd.DataFrame:
    block = fit.design.blocks[block_name]
    rows: list[dict[str, object]] = []
    free_levels = list(block.free_levels)
    for idx, level in enumerate(free_levels):
        coef_index = block.start + idx
        estimate = float(fit.beta[coef_index])
        stderr = None if fit.standard_errors is None else float(fit.standard_errors[coef_index])
        ci_low = None if stderr is None else estimate - 1.96 * stderr
        ci_high = None if stderr is None else estimate + 1.96 * stderr
        rows.append(
            {
                "block": block_name,
                "level": level,
                "reference": False,
                "estimate": estimate,
                "stderr": stderr,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    rows.append(
        {
            "block": block_name,
            "level": block.reference,
            "reference": True,
            "estimate": 0.0,
            "stderr": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
        }
    )
    return pd.DataFrame(rows)


def setting_difficulty_frame(fit: FitResult) -> pd.DataFrame:
    df = _block_frame(fit, "setting").copy()
    df["setting"] = df["level"]
    df["display"] = df["setting"].map(_display_setting)
    df["is_reference"] = df["reference"]
    return df.sort_values(["reference", "estimate"], ascending=[True, False]).reset_index(drop=True)


def evaluator_severity_frame(fit: FitResult) -> pd.DataFrame:
    base = _block_frame(fit, "evaluator").copy()
    base["evaluator"] = base["level"]
    base["display"] = base["evaluator"].map(_display_evaluator)
    base["is_reference"] = base["reference"]
    return base.sort_values("estimate", ascending=False).reset_index(drop=True)


def generator_ability_frame(fit: FitResult) -> pd.DataFrame:
    df = _block_frame(fit, "generator").copy()
    df["generator"] = df["level"]
    mean_estimate = float(df["estimate"].mean())
    df["estimate_centered"] = df["estimate"] - mean_estimate
    if fit.standard_errors is not None:
        centered_stderr = df["stderr"].fillna(0.0)
        df["ci_low_centered"] = df["estimate_centered"] - 1.96 * centered_stderr
        df["ci_high_centered"] = df["estimate_centered"] + 1.96 * centered_stderr
    else:
        df["ci_low_centered"] = np.nan
        df["ci_high_centered"] = np.nan
    return df.sort_values("estimate_centered", ascending=False).reset_index(drop=True)


def item_difficulty_frame(fit: FitResult) -> pd.DataFrame:
    df = _block_frame(fit, "item").copy()
    df["item_id"] = df["level"]
    item_meta = (
        fit.design.frame[["item_id", "question", "dataset"]]
        .drop_duplicates("item_id")
        .set_index("item_id")
    )
    df["question"] = df["item_id"].map(item_meta["question"])
    df["dataset"] = df["item_id"].map(item_meta["dataset"])
    return df.sort_values("estimate", ascending=False).reset_index(drop=True)


def item_fit_frame(fit: FitResult) -> pd.DataFrame:
    frame = fit.fitted_frame.copy()
    grouped = frame.groupby("item_id", sort=False)
    rows: list[dict[str, object]] = []
    meta = frame[["item_id", "question", "dataset"]].drop_duplicates("item_id").set_index("item_id")
    for item_id, group in grouped:
        var = np.clip(group["variance"].to_numpy(dtype=float), 1e-8, None)
        sq = (group["residual"].to_numpy(dtype=float) ** 2) / var
        outfit = float(np.mean(sq))
        infit = float(np.sum(group["residual"].to_numpy(dtype=float) ** 2) / np.sum(var))
        rows.append(
            {
                "item_id": item_id,
                "dataset": meta.loc[item_id, "dataset"],
                "question": meta.loc[item_id, "question"],
                "n_obs": int(len(group)),
                "outfit": outfit,
                "infit": infit,
                "underfit_flag": outfit > 1.5,
                "overfit_flag": outfit < 0.7,
            }
        )
    return pd.DataFrame(rows).sort_values("outfit", ascending=False).reset_index(drop=True)


def residual_summary_frame(fit: FitResult) -> pd.DataFrame:
    grouped = (
        fit.fitted_frame
        .groupby(["evaluator", "setting", "choice_group"], sort=False)
        .agg(
            n_obs=("obs_id", "count"),
            mean_residual=("residual", "mean"),
            mean_correct=("correct", "mean"),
            mean_predicted=("probability", "mean"),
        )
        .reset_index()
    )
    grouped["display_evaluator"] = grouped["evaluator"].map(_display_evaluator)
    grouped["display_setting"] = grouped["setting"].map(_display_setting)
    return grouped


def generator_setting_deltas(frame: pd.DataFrame, *, maxiter: int = 100) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for generator, subset in frame.groupby("generator", sort=False):
        design = build_design(
            subset,
            reference_setting=DEFAULT_REFERENCE_SETTING,
            reference_evaluator=DEFAULT_REFERENCE_EVALUATOR,
            interaction=False,
        )
        fit = fit_design(design, maxiter=maxiter)
        setting_df = setting_difficulty_frame(fit)
        setting_df["generator"] = generator
        rows.append(setting_df[["generator", "setting", "estimate", "stderr", "ci_low", "ci_high", "is_reference"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _plot_forest(df: pd.DataFrame, *, estimate_col: str, low_col: str, high_col: str, label_col: str, title: str, output_path: Path) -> Path:
    plot_df = df.copy()
    plot_df = plot_df.sort_values(estimate_col, ascending=True).reset_index(drop=True)
    fig_height = max(3.5, 0.45 * len(plot_df))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    y = np.arange(len(plot_df))
    estimates = plot_df[estimate_col].to_numpy(dtype=float)
    lows = plot_df[low_col].fillna(plot_df[estimate_col]).to_numpy(dtype=float)
    highs = plot_df[high_col].fillna(plot_df[estimate_col]).to_numpy(dtype=float)
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.errorbar(
        estimates,
        y,
        xerr=np.vstack((estimates - lows, highs - estimates)),
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df[label_col].astype(str))
    ax.set_title(title)
    ax.set_xlabel("Logit Estimate")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_item_anomalies(item_df: pd.DataFrame, fit_df: pd.DataFrame, output_path: Path) -> Path:
    merged = item_df.merge(fit_df, on=["item_id", "dataset", "question"], how="inner")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(merged["estimate"], merged["outfit"], alpha=0.4, s=18, color="#4c78a8")
    ax.axhline(1.5, color="#e45756", linestyle="--", linewidth=1.0)
    ax.axhline(0.7, color="#72b7b2", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Item Difficulty")
    ax.set_ylabel("Outfit")
    ax.set_title("Item Difficulty vs Outfit")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _selected_values(raw: str | None, allowed: Iterable[str] | None = None) -> list[str] | None:
    values = _csv_list(raw)
    if values is None or allowed is None:
        return values
    allowed_set = set(allowed)
    invalid = [value for value in values if value not in allowed_set]
    if invalid:
        raise ValueError(f"Unsupported values: {', '.join(invalid)}")
    return values


def run_irt_analysis(
    *,
    collected_root: Path | str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    generators: list[str] | None = None,
    evaluators: list[str] | None = None,
    datasets: list[str] | None = None,
    settings: list[str] | None = None,
    modes: list[str] | None = None,
    maxiter: int = 2000,
    maxfun: int = 50000,
    gtol: float = 1e-5,
    item_prior_sd: float | None = DEFAULT_ITEM_PRIOR_SD,
) -> list[Path]:
    frame = load_irt_frame(
        collected_root,
        generators=generators,
        evaluators=evaluators,
        datasets=datasets,
        settings=settings,
        modes=modes,
    )
    if frame.empty:
        raise ValueError("No observed evaluator judgments found under the selected collected root.")
    validate_identification(frame)

    additive_design = build_design(frame, interaction=False)
    additive_fit = fit_design(
        additive_design,
        maxiter=maxiter,
        maxfun=maxfun,
        gtol=gtol,
        item_prior_sd=item_prior_sd,
    )
    if not additive_fit.success:
        additive_fit = fit_design(
            additive_design,
            maxiter=maxiter,
            maxfun=maxfun,
            gtol=gtol,
            init_beta=additive_fit.beta,
            item_prior_sd=item_prior_sd,
        )

    output_root = Path(output_dir)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"

    setting_df = setting_difficulty_frame(additive_fit)
    evaluator_df = evaluator_severity_frame(additive_fit)
    generator_df = generator_ability_frame(additive_fit)
    item_df = item_difficulty_frame(additive_fit)
    item_fit_df = item_fit_frame(additive_fit)
    residual_df = residual_summary_frame(additive_fit)
    generator_setting_df = generator_setting_deltas(frame, maxiter=maxiter)

    outputs = [
        _write_csv(setting_df, tables_dir / "setting_difficulty.csv"),
        _write_csv(evaluator_df, tables_dir / "evaluator_severity.csv"),
        _write_csv(generator_df, tables_dir / "generator_ability.csv"),
        _write_csv(item_df, tables_dir / "item_difficulty.csv"),
        _write_csv(item_fit_df, tables_dir / "item_fit.csv"),
        _write_csv(residual_df, tables_dir / "residual_summary.csv"),
        _write_csv(generator_setting_df, tables_dir / "generator_setting_deltas.csv"),
    ]

    outputs.extend(
        [
            _plot_forest(
                setting_df.assign(label=setting_df["display"]),
                estimate_col="estimate",
                low_col="ci_low",
                high_col="ci_high",
                label_col="label",
                title="Setting Difficulty",
                output_path=figures_dir / "setting_difficulty_forest.png",
            ),
            _plot_forest(
                generator_df.assign(label=generator_df["generator"]),
                estimate_col="estimate_centered",
                low_col="ci_low_centered",
                high_col="ci_high_centered",
                label_col="label",
                title="Generator Model Ability",
                output_path=figures_dir / "generator_ability_forest.png",
            ),
            _plot_forest(
                evaluator_df.assign(label=evaluator_df["display"]),
                estimate_col="estimate",
                low_col="ci_low",
                high_col="ci_high",
                label_col="label",
                title="Evaluator Severity",
                output_path=figures_dir / "evaluator_severity_forest.png",
            ),
            _plot_item_anomalies(item_df, item_fit_df, figures_dir / "item_anomalies.png"),
        ]
    )

    top_hard = item_df.head(DEFAULT_ITEM_FIT_LIMIT)
    top_easy = item_df.tail(DEFAULT_ITEM_FIT_LIMIT)
    top_underfit = item_fit_df.head(DEFAULT_ITEM_FIT_LIMIT)
    top_overfit = item_fit_df.sort_values("outfit", ascending=True).head(DEFAULT_ITEM_FIT_LIMIT)
    outputs.extend(
        [
            _write_csv(top_hard, tables_dir / "hardest_items.csv"),
            _write_csv(top_easy, tables_dir / "easiest_items.csv"),
            _write_csv(top_underfit, tables_dir / "highest_outfit_items.csv"),
            _write_csv(top_overfit, tables_dir / "lowest_outfit_items.csv"),
        ]
    )

    summary = {
        "n_obs": int(len(frame)),
        "n_items": int(frame["item_id"].nunique()),
        "n_generators": int(frame["generator"].nunique()),
        "n_evaluators": int(frame["evaluator"].nunique()),
        "n_settings": int(frame["setting"].nunique()),
        "n_modes": int(frame["mode"].nunique()),
        "additive_model": {
            "success": additive_fit.success,
            "message": additive_fit.message,
            "iterations": additive_fit.iterations,
            "log_likelihood": additive_fit.log_likelihood,
            "aic": additive_fit.aic,
            "bic": additive_fit.bic,
        },
        "optimizer": {
            "method": "L-BFGS-B",
            "maxiter": int(maxiter),
            "maxfun": int(maxfun),
            "gtol": float(gtol),
            "item_prior_sd": item_prior_sd,
        },
        "reference_levels": {
            "setting": additive_fit.design.blocks["setting"].reference,
            "evaluator": additive_fit.design.blocks["evaluator"].reference,
            "item": additive_fit.design.item_reference,
            "generator": additive_fit.design.blocks["generator"].reference,
        },
        "filters": {
            "generators": generators or [],
            "evaluators": evaluators or [],
            "datasets": datasets or [],
            "settings": settings or [],
            "modes": modes or ["full_question"],
        },
    }
    summary_path = output_root / "fit_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    outputs.append(summary_path)
    return outputs


def run_cli(args) -> int:
    outputs = run_irt_analysis(
        collected_root=Path(args.collected_root),
        output_dir=Path(args.output_dir),
        generators=_selected_values(args.generators),
        evaluators=_selected_values(args.evaluators),
        datasets=_selected_values(args.datasets),
        settings=_selected_values(args.settings, SETTING_NAMES),
        modes=_selected_values(args.modes, MODE_CHOICES),
        maxiter=int(args.maxiter),
        maxfun=int(args.maxfun),
        gtol=float(args.gtol),
        item_prior_sd=None if float(args.item_prior_sd) <= 0.0 else float(args.item_prior_sd),
    )
    for output in outputs:
        print(output)
    return 0
