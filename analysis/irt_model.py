"""Fit a simple decomposed 3PL IRT model over cached MCQA evaluations."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from utils.constants import EVALUATED_STORE_MANIFEST, SETTING_NAMES, SETTING_SPECS


IRT_SCALING = 1.702
EPS = 1e-8
DEFAULT_OUTPUT_DIR = Path("results/augmented_mcqa_irt")
DEFAULT_BENCHMARKER_JSONL = Path("results/atrey_writing_flaw_rows_strict.jsonl")
DEFAULT_BENCHMARKER_TABLE = Path("results/augmented_mcqa_irt/tables/writing_flaw_rows.csv")
DEFAULT_REFERENCE_TEST_TAKER = "vllm/nvidia/NVIDIA-Nemotron-Nano-9B-v2"
DEFAULT_REFERENCE_SETTING = "human_from_scratch"
TOP_N_ITEMS = 20
DATASET_ORDER = ["arc_challenge", "mmlu_pro", "gpqa"]
DATASET_LABELS = {
    "arc_challenge": "ARC Challenge",
    "mmlu_pro": "MMLU",
    "gpqa": "GPQA",
}
MODEL_ORDER = [
    ("openai/gpt-5.2-2025-12-11", "GPT"),
    ("google/gemini-3.1-pro-preview", "Gemini"),
    ("together/Qwen/Qwen3.5-397B-A17B", "Qwen"),
]
STEM_PRIOR_SD = 3.0
ITEM_NOISE_PRIOR_SD = 1.0
LOG_DISCRIMINATION_PRIOR_SD = 0.75
GUESSING_PRIOR_SD = 1.0

SETTING_GUESSING = {name: 1.0 / int(spec["num_choices"]) for name, spec in SETTING_SPECS.items()}
SETTING_LABELS = {
    "human_from_scratch": "Human From Scratch",
    "model_from_scratch": "Model From Scratch",
    "augment_human": "Augment Human",
    "augment_model": "Augment Model",
    "augment_ablation": "Augment Ablation",
    "augment_human_m3": "Augment Human (m=3)",
    "augment_model_m3": "Augment Model (m=3)",
    "augment_human_m4": "Augment Human (m=4)",
    "augment_model_m4": "Augment Model (m=4)",
    "augment_human_m5": "Augment Human (m=5)",
    "augment_model_m5": "Augment Model (m=5)",
}
SETTING_SHORT_LABELS = {
    "human_from_scratch": "HFS",
    "model_from_scratch": "MFS",
    "augment_human": "AH",
    "augment_model": "AM",
    "augment_ablation": "AA",
    "augment_human_m3": "AH3",
    "augment_model_m3": "AM3",
    "augment_human_m4": "AH4",
    "augment_model_m4": "AM4",
    "augment_human_m5": "AH5",
    "augment_model_m5": "AM5",
}
TEST_TAKER_LABELS = {
    "vllm/Qwen/Qwen3-4B-Instruct-2507": "Qwen3-4B",
    "vllm/Qwen/Qwen3.5-35B-A3B-FP8": "Qwen3.5-35B-FP8",
    "vllm/Qwen/Qwen3.5-9B": "Qwen3.5-9B",
    "vllm/Qwen/Qwen3-14B": "Qwen3-14B",
    "vllm/allenai/Olmo-3-7B-Instruct": "Olmo-3-7B",
    "vllm/meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "vllm/meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "vllm/google/gemma-4-E4B-it": "Gemma4-E4B-it",
    "vllm/nvidia/NVIDIA-Nemotron-Nano-9B-v2": "Nemotron-9B",
    "openai/gpt-5.4-mini": "GPT-5.4-mini",
    "together/openai/gpt-oss-120b": "GPT-OSS-120B",
}
GENERATOR_LABEL_PARTS = [
    ("gpt-5.2", "gpt-5.2"),
    ("gemini-3.1-pro", "gemini-3.1-pro"),
    ("Qwen3.5-397B-A17B", "Qwen3.5-397B"),
]


@dataclass(frozen=True)
class Block:
    name: str
    levels: tuple[str, ...]
    reference: str | None
    start: int
    size: int

    @property
    def stop(self) -> int:
        return self.start + self.size

    @property
    def free_levels(self) -> tuple[str, ...]:
        return self.levels if self.reference is None else tuple(x for x in self.levels if x != self.reference)


@dataclass(frozen=True)
class Design:
    frame: pd.DataFrame
    X_theta: csr_matrix
    X_difficulty: csr_matrix
    y: np.ndarray
    blocks: dict[str, Block]
    item_index: np.ndarray
    guessing_center: np.ndarray
    n_params: int


@dataclass(frozen=True)
class Fit:
    design: Design
    beta: np.ndarray
    success: bool
    message: str
    iterations: int
    objective: float
    log_likelihood: float
    aic: float
    bic: float
    fitted_frame: pd.DataFrame


def csv_values(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return values or None


def safe_name(value: str) -> str:
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")


def generator_label(value: str) -> str:
    raw = str(value)
    for needle, label in GENERATOR_LABEL_PARTS:
        if needle in raw:
            return label
    return raw


def levels(series: pd.Series, order: list[str] | None = None) -> list[str]:
    values = sorted(series.astype(str).unique())
    if order is None:
        return values
    rank = {value: idx for idx, value in enumerate(order)}
    return sorted(values, key=lambda value: (rank.get(value, len(rank)), value))


def reference(values: list[str], preferred: str | None) -> str:
    return preferred if preferred in values else values[0]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


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
    test_taker_filter = set(evaluators or [])
    dataset_filter = set(datasets or [])
    setting_filter = set(settings or [])
    mode_filter = set(modes or ["full_question"])
    frames: list[pd.DataFrame] = []
    questions: dict[str, str] = {}

    for manifest_path in sorted(Path(root).rglob(EVALUATED_STORE_MANIFEST)):
        group_root = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        generator = str(manifest.get("generation_model", ""))
        test_taker = str(manifest.get("evaluation_model", ""))
        if generator_filter and generator not in generator_filter:
            continue
        if test_taker_filter and test_taker not in test_taker_filter:
            continue

        for dataset in manifest.get("dataset_types", []):
            if dataset_filter and dataset not in dataset_filter:
                continue
            for setting in manifest.get("settings", []):
                if setting_filter and setting not in setting_filter:
                    continue
                for mode in manifest.get("modes", []):
                    if mode_filter and mode not in mode_filter:
                        continue
                    path = group_root / dataset / setting / mode
                    if not path.exists():
                        continue
                    # Hugging Face can persist an empty Dataset directory without
                    # any Arrow shards. load_from_disk() raises IndexError for
                    # those directories, so treat them like any other absent
                    # dataset/setting/mode slice.
                    if not any(path.glob("*.arrow")):
                        continue
                    loaded = load_from_disk(str(path))
                    selected_columns = [
                        "sample_id",
                        "question",
                        "evaluation_status",
                        "evaluation_used_random_fallback",
                        "evaluation_prediction",
                        "evaluation_is_correct",
                        "num_choices",
                        "setting",
                    ]
                    missing_columns = sorted(set(selected_columns) - set(loaded.column_names))
                    if missing_columns:
                        missing = ", ".join(missing_columns)
                        raise ValueError(f"{path} is missing required evaluation columns: {missing}")
                    data = loaded.select_columns(selected_columns).to_pandas()
                    if data.empty:
                        continue
                    stem_ids = dataset + "::" + data["sample_id"].astype(str)
                    question_frame = pd.DataFrame({"stem_id": stem_ids, "question": data["question"].astype(str)})
                    questions.update(question_frame.drop_duplicates("stem_id").set_index("stem_id")["question"].to_dict())
                    data = data[
                        [
                            "sample_id",
                            "evaluation_status",
                            "evaluation_used_random_fallback",
                            "evaluation_prediction",
                            "evaluation_is_correct",
                            "num_choices",
                            "setting",
                        ]
                    ].copy()
                    data["dataset"] = dataset
                    data["generator"] = generator
                    data["test_taker"] = test_taker
                    data["mode"] = mode
                    frames.append(data)

    if not frames:
        return pd.DataFrame()

    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["evaluation_status"].fillna("").astype(str) == "success"].copy()
    frame = frame[frame["evaluation_prediction"].fillna("").astype(str).str.strip() != ""].copy()
    frame = frame[frame["evaluation_is_correct"].notna()].copy()
    frame["correct"] = frame["evaluation_is_correct"].astype(bool).astype(float)
    frame["choice_count"] = frame["num_choices"].astype(int)
    frame["choice_group"] = np.where(frame["choice_count"] <= 4, "4-choice", "10-choice")
    frame["stem_id"] = frame["dataset"].astype(str) + "::" + frame["sample_id"].astype(str)
    frame["question"] = frame["stem_id"].map(questions).fillna("")
    frame["item_id"] = (
        frame["dataset"].astype(str)
        + "::"
        + frame["sample_id"].astype(str)
        + "::"
        + frame["generator"].astype(str)
        + "::"
        + frame["setting"].astype(str)
    )
    frame["guessing"] = frame["setting"].map(SETTING_GUESSING).astype(float)
    frame["obs_id"] = np.arange(len(frame))
    return frame.reset_index(drop=True)


def add_block(rows: list[int], cols: list[int], vals: list[float], series: pd.Series, block: Block, sign: float) -> None:
    offset = {level: block.start + idx for idx, level in enumerate(block.free_levels)}
    for row, level in enumerate(series.astype(str)):
        if level == block.reference:
            continue
        rows.append(row)
        cols.append(offset[level])
        vals.append(sign)


def make_design(frame: pd.DataFrame) -> Design:
    if frame.empty:
        raise ValueError("No rows available for IRT.")
    if frame["test_taker"].nunique() < 2:
        raise ValueError("IRT needs at least two test-taker models.")

    level_map = {
        "theta": levels(frame["test_taker"]),
        "dataset": levels(frame["dataset"]),
        "stem": levels(frame["stem_id"]),
        "generator": levels(frame["generator"]),
        "setting": levels(frame["setting"], SETTING_NAMES),
        "item_noise": levels(frame["item_id"]),
        "log_discrimination": levels(frame["item_id"]),
        "guessing": levels(frame["item_id"]),
    }
    refs = {
        "theta": reference(level_map["theta"], DEFAULT_REFERENCE_TEST_TAKER),
        "dataset": level_map["dataset"][0],
        "stem": level_map["stem"][0],
        "generator": level_map["generator"][0],
        "setting": reference(level_map["setting"], DEFAULT_REFERENCE_SETTING),
        "item_noise": None,
        "log_discrimination": None,
        "guessing": None,
    }

    blocks: dict[str, Block] = {}
    start = 0
    for name, vals in level_map.items():
        size = len(vals) if refs[name] is None else len(vals) - 1
        blocks[name] = Block(name, tuple(vals), refs[name], start, size)
        start += size

    theta_rows: list[int] = []
    theta_cols: list[int] = []
    theta_vals: list[float] = []
    difficulty_rows: list[int] = []
    difficulty_cols: list[int] = []
    difficulty_vals: list[float] = []

    add_block(theta_rows, theta_cols, theta_vals, frame["test_taker"], blocks["theta"], 1.0)
    add_block(difficulty_rows, difficulty_cols, difficulty_vals, frame["dataset"], blocks["dataset"], 1.0)
    add_block(difficulty_rows, difficulty_cols, difficulty_vals, frame["stem_id"], blocks["stem"], 1.0)
    add_block(difficulty_rows, difficulty_cols, difficulty_vals, frame["generator"], blocks["generator"], 1.0)
    add_block(difficulty_rows, difficulty_cols, difficulty_vals, frame["setting"], blocks["setting"], 1.0)
    add_block(difficulty_rows, difficulty_cols, difficulty_vals, frame["item_id"], blocks["item_noise"], 1.0)

    shape = (len(frame), start)
    item_lookup = {item: idx for idx, item in enumerate(level_map["item_noise"])}
    item_index = frame["item_id"].map(item_lookup).to_numpy(dtype=int)
    guessing_center = (
        frame.drop_duplicates("item_id").set_index("item_id").loc[level_map["item_noise"], "guessing"].to_numpy(dtype=float)
    )
    return Design(
        frame=frame,
        X_theta=csr_matrix((theta_vals, (theta_rows, theta_cols)), shape=shape),
        X_difficulty=csr_matrix((difficulty_vals, (difficulty_rows, difficulty_cols)), shape=shape),
        y=frame["correct"].to_numpy(dtype=float),
        blocks=blocks,
        item_index=item_index,
        guessing_center=logit(guessing_center),
        n_params=start,
    )


def block_slice(design: Design, name: str) -> slice:
    block = design.blocks[name]
    return slice(block.start, block.stop)


def irt_probability(beta: np.ndarray, design: Design) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.asarray(design.X_theta @ beta).reshape(-1)
    difficulty = np.asarray(design.X_difficulty @ beta).reshape(-1)
    item_idx = design.item_index

    alpha = beta[block_slice(design, "log_discrimination")][item_idx]
    rho = beta[block_slice(design, "guessing")][item_idx]
    a = np.exp(np.clip(alpha, -8.0, 8.0))
    c = sigmoid(np.clip(rho, -12.0, 12.0))

    eta = IRT_SCALING * a * (theta - difficulty)
    u = sigmoid(eta)
    p = np.clip(c + (1.0 - c) * u, EPS, 1.0 - EPS)
    return theta, difficulty, a, c, eta, p


def regularization(beta: np.ndarray, design: Design) -> tuple[float, np.ndarray]:
    loss = 0.0
    grad = np.zeros_like(beta)

    for name, sd in (
        ("stem", STEM_PRIOR_SD),
        ("item_noise", ITEM_NOISE_PRIOR_SD),
        ("log_discrimination", LOG_DISCRIMINATION_PRIOR_SD),
    ):
        slc = block_slice(design, name)
        x = beta[slc]
        loss += 0.5 * float(np.sum((x / sd) ** 2))
        grad[slc] += x / (sd**2)

    slc = block_slice(design, "guessing")
    x = beta[slc] - design.guessing_center
    loss += 0.5 * float(np.sum((x / GUESSING_PRIOR_SD) ** 2))
    grad[slc] += x / (GUESSING_PRIOR_SD**2)
    return loss, grad


def objective_and_gradient(beta: np.ndarray, design: Design) -> tuple[float, np.ndarray]:
    _, _, _, c, eta, p = irt_probability(beta, design)
    y = design.y
    nll = -float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    u = np.clip((p - c) / np.clip(1.0 - c, EPS, None), EPS, 1.0 - EPS)
    dloss_dp = (p - y) / (p * (1.0 - p))
    dloss_deta = dloss_dp * (1.0 - c) * u * (1.0 - u)
    alpha = beta[block_slice(design, "log_discrimination")][design.item_index]
    a = np.exp(np.clip(alpha, -8.0, 8.0))

    grad = np.asarray(design.X_theta.T @ (dloss_deta * IRT_SCALING * a)).reshape(-1)
    grad -= np.asarray(design.X_difficulty.T @ (dloss_deta * IRT_SCALING * a)).reshape(-1)

    grad[block_slice(design, "log_discrimination")] += np.bincount(
        design.item_index,
        weights=dloss_deta * eta,
        minlength=design.blocks["log_discrimination"].size,
    )
    grad[block_slice(design, "guessing")] += np.bincount(
        design.item_index,
        weights=dloss_dp * (1.0 - u) * c * (1.0 - c),
        minlength=design.blocks["guessing"].size,
    )

    penalty, penalty_grad = regularization(beta, design)
    return nll + penalty, grad + penalty_grad


def objective(beta: np.ndarray, design: Design) -> float:
    return objective_and_gradient(beta, design)[0]


def gradient(beta: np.ndarray, design: Design) -> np.ndarray:
    return objective_and_gradient(beta, design)[1]


def initial_beta(design: Design) -> np.ndarray:
    beta = np.zeros(design.n_params)
    beta[block_slice(design, "guessing")] = design.guessing_center
    return beta


def fit_design(
    design: Design,
    *,
    maxiter: int = 2000,
    maxfun: int = 50000,
    gtol: float = 1e-5,
    init_beta: np.ndarray | None = None,
) -> Fit:
    result = minimize(
        objective,
        initial_beta(design) if init_beta is None else init_beta,
        args=(design,),
        method="L-BFGS-B",
        jac=gradient,
        options={"maxiter": maxiter, "maxfun": maxfun, "gtol": gtol},
    )
    _, _, _, _, eta, p = irt_probability(result.x, design)
    fitted = design.frame.copy()
    fitted["eta"] = eta
    fitted["probability"] = p
    fitted["residual"] = fitted["correct"] - p
    fitted["variance"] = p * (1.0 - p)
    log_likelihood = float(np.sum(design.y * np.log(p) + (1.0 - design.y) * np.log(1.0 - p)))
    return Fit(
        design=design,
        beta=result.x,
        success=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit),
        objective=float(result.fun),
        log_likelihood=log_likelihood,
        aic=2.0 * design.n_params - 2.0 * log_likelihood,
        bic=math.log(len(design.frame)) * design.n_params - 2.0 * log_likelihood,
        fitted_frame=fitted,
    )


def coefficient_table(fit: Fit, name: str) -> pd.DataFrame:
    block = fit.design.blocks[name]
    rows = [
        {"level": level, "estimate": float(fit.beta[block.start + idx]), "reference": False}
        for idx, level in enumerate(block.free_levels)
    ]
    if block.reference is not None:
        rows.append({"level": block.reference, "estimate": 0.0, "reference": True})
    return pd.DataFrame(rows)


def setting_difficulty_frame(fit: Fit) -> pd.DataFrame:
    table = coefficient_table(fit, "setting")
    table["setting"] = table["level"]
    table["display"] = table["setting"].map(lambda x: SETTING_LABELS.get(x, x))
    return table.sort_values("estimate", ascending=False).reset_index(drop=True)


def dataset_difficulty_frame(fit: Fit) -> pd.DataFrame:
    dataset = coefficient_table(fit, "dataset")[["level", "estimate"]].rename(
        columns={"level": "dataset", "estimate": "dataset_estimate"}
    )
    stem = stem_difficulty_frame(fit).rename(columns={"estimate": "stem_estimate"})
    summary = (
        stem.merge(dataset, on="dataset", how="left")
        .assign(estimate=lambda frame: frame["dataset_estimate"] + frame["stem_estimate"])
        .groupby("dataset", as_index=False)
        .agg(
            estimate=("estimate", "mean"),
            n_stems=("stem_id", "nunique"),
            dataset_estimate=("dataset_estimate", "first"),
            mean_stem_estimate=("stem_estimate", "mean"),
        )
    )
    summary["level"] = summary["dataset"]
    summary["reference"] = False
    return summary.sort_values("estimate", ascending=False).reset_index(drop=True)


def generator_difficulty_frame(fit: Fit) -> pd.DataFrame:
    table = coefficient_table(fit, "generator")
    table["generator"] = table["level"]
    table["display"] = table["generator"].map(generator_label)
    return table.sort_values("estimate", ascending=False).reset_index(drop=True)


def taker_ability_frame(fit: Fit) -> pd.DataFrame:
    table = coefficient_table(fit, "theta")
    table["test_taker"] = table["level"]
    table["display"] = table["test_taker"].map(lambda x: TEST_TAKER_LABELS.get(x, x))
    table["estimate_centered"] = table["estimate"] - table["estimate"].mean()
    return table.sort_values("estimate_centered", ascending=False).reset_index(drop=True)


def stem_difficulty_frame(fit: Fit) -> pd.DataFrame:
    table = coefficient_table(fit, "stem")
    table["stem_id"] = table["level"]
    meta = fit.design.frame.drop_duplicates("stem_id").set_index("stem_id")
    table["dataset"] = table["stem_id"].map(meta["dataset"])
    table["sample_id"] = table["stem_id"].map(meta["sample_id"])
    table["question"] = table["stem_id"].map(meta["question"])
    return table.sort_values("estimate", ascending=False).reset_index(drop=True)


def item_parameters_frame(fit: Fit) -> pd.DataFrame:
    frame = fit.design.frame.drop_duplicates("item_id").set_index("item_id")
    item_ids = list(fit.design.blocks["item_noise"].levels)
    alpha = fit.beta[block_slice(fit.design, "log_discrimination")]
    rho = fit.beta[block_slice(fit.design, "guessing")]
    pieces = {name: coefficient_table(fit, name).set_index("level")["estimate"] for name in ("dataset", "stem", "generator", "setting", "item_noise")}
    rows = []

    for idx, item_id in enumerate(item_ids):
        row = frame.loc[item_id]
        difficulty = (
            float(pieces["dataset"].get(row["dataset"], 0.0))
            + float(pieces["stem"].get(row["stem_id"], 0.0))
            + float(pieces["generator"].get(row["generator"], 0.0))
            + float(pieces["setting"].get(row["setting"], 0.0))
            + float(pieces["item_noise"].get(item_id, 0.0))
        )
        rows.append(
            {
                "item_id": item_id,
                "stem_id": row["stem_id"],
                "dataset": row["dataset"],
                "sample_id": row["sample_id"],
                "generator": row["generator"],
                "setting": row["setting"],
                "choice_count": int(row["choice_count"]),
                "question": row["question"],
                "difficulty": difficulty,
                "discrimination": float(np.exp(np.clip(alpha[idx], -8.0, 8.0))),
                "guessing": float(sigmoid(np.array([np.clip(rho[idx], -12.0, 12.0)]))[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("difficulty", ascending=False).reset_index(drop=True)


def item_fit_frame(fit: Fit) -> pd.DataFrame:
    rows = []
    meta = fit.design.frame.drop_duplicates("item_id").set_index("item_id")
    for item_id, group in fit.fitted_frame.groupby("item_id", sort=False):
        residual = group["residual"].to_numpy(dtype=float)
        variance = np.clip(group["variance"].to_numpy(dtype=float), EPS, None)
        outfit = float(np.mean((residual**2) / variance))
        rows.append(
            {
                "item_id": item_id,
                "dataset": meta.loc[item_id, "dataset"],
                "sample_id": meta.loc[item_id, "sample_id"],
                "generator": meta.loc[item_id, "generator"],
                "setting": meta.loc[item_id, "setting"],
                "question": meta.loc[item_id, "question"],
                "n_obs": int(len(group)),
                "outfit": outfit,
                "infit": float(np.sum(residual**2) / np.sum(variance)),
            }
        )
    return pd.DataFrame(rows).sort_values("outfit", ascending=False).reset_index(drop=True)


def residual_summary_frame(fit: Fit) -> pd.DataFrame:
    return (
        fit.fitted_frame.groupby(["test_taker", "dataset", "generator", "setting", "choice_group"], sort=False)
        .agg(
            n_obs=("obs_id", "count"),
            mean_correct=("correct", "mean"),
            mean_predicted=("probability", "mean"),
            mean_residual=("residual", "mean"),
        )
        .reset_index()
    )


def raw_generator_setting_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["generator", "setting"], sort=False)
        .agg(n_obs=("obs_id", "count"), mean_correct=("correct", "mean"))
        .reset_index()
    )


def irt_quality_summary_frame(items: pd.DataFrame) -> pd.DataFrame:
    summary = (
        items.groupby(["dataset", "setting"], sort=False)
        .agg(
            n_items=("item_id", "count"),
            difficulty=("difficulty", "mean"),
            difficulty_sd=("difficulty", "std"),
            discrimination=("discrimination", "mean"),
            discrimination_sd=("discrimination", "std"),
        )
        .reset_index()
    )
    summary["difficulty_se"] = summary["difficulty_sd"].fillna(0.0) / np.sqrt(summary["n_items"].clip(lower=1))
    summary["discrimination_se"] = summary["discrimination_sd"].fillna(0.0) / np.sqrt(summary["n_items"].clip(lower=1))
    return summary


def benchmarker_validity_frame(
    path: Path = DEFAULT_BENCHMARKER_JSONL,
    table_path: Path = DEFAULT_BENCHMARKER_TABLE,
) -> pd.DataFrame:
    if table_path.exists():
        flaw_df = pd.read_csv(table_path)
    elif path.exists():
        from analysis.benchmarker_analysis import load_writing_flaw_data

        flaw_df, _ = load_writing_flaw_data(path)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        flaw_df.to_csv(table_path, index=False)
    else:
        return pd.DataFrame(columns=["dataset", "setting", "n_questions", "mean_flaws", "mean_flaws_se"])

    flaw_df = flaw_df.rename(columns={"config": "setting"}).copy()
    flaw_df["generator"] = flaw_df["generator_model"].apply(lambda x: next((g for g, _ in MODEL_ORDER if x in g), x))
    grouped = (
        flaw_df.groupby(["dataset", "setting", "generator"], observed=True)
        .agg(
            n_questions=("n_flaws", "count"),
            mean_flaws=("n_flaws", "mean"),
            flaws_sd=("n_flaws", "std"),
        )
        .reset_index()
    )
    grouped["mean_flaws_se"] = grouped["flaws_sd"].fillna(0.0) / np.sqrt(grouped["n_questions"].clip(lower=1))
    return grouped.drop(columns=["flaws_sd"])


def validity_setting_level_frame(validity: pd.DataFrame) -> pd.DataFrame:
    if validity.empty:
        return pd.DataFrame(columns=["dataset", "setting", "n_questions", "mean_flaws", "mean_flaws_se"])
    if "generator" not in validity.columns:
        return validity.copy()

    rows: list[dict[str, float | int | str]] = []
    for (dataset, setting), group in validity.groupby(["dataset", "setting"], observed=True, sort=False):
        counts = group["n_questions"].fillna(0).astype(int).to_numpy(dtype=int)
        means = group["mean_flaws"].astype(float).to_numpy(dtype=float)
        ses = group["mean_flaws_se"].fillna(0.0).astype(float).to_numpy(dtype=float)
        sds = ses * np.sqrt(np.clip(counts, 1, None))
        total_n = int(np.sum(counts))
        if total_n <= 0:
            continue

        pooled_mean = float(np.average(means, weights=np.clip(counts, 1, None)))
        if total_n == 1:
            pooled_se = 0.0
        else:
            ss_within = float(np.sum(np.maximum(counts - 1, 0) * (sds ** 2)))
            ss_between = float(np.sum(counts * ((means - pooled_mean) ** 2)))
            pooled_var = max((ss_within + ss_between) / max(total_n - 1, 1), 0.0)
            pooled_se = math.sqrt(pooled_var / total_n)

        rows.append(
            {
                "dataset": str(dataset),
                "setting": str(setting),
                "n_questions": total_n,
                "mean_flaws": pooled_mean,
                "mean_flaws_se": float(pooled_se),
            }
        )
    return pd.DataFrame(rows)


def combined_quality_frame(irt_summary: pd.DataFrame, validity: pd.DataFrame) -> pd.DataFrame:
    validity_setting_level = validity_setting_level_frame(validity)
    merged = irt_summary.merge(validity_setting_level, on=["dataset", "setting"], how="left")
    merged["mean_flaws"] = merged["mean_flaws"].astype(float)
    merged["mean_flaws_se"] = merged["mean_flaws_se"].fillna(0.0).astype(float)
    return merged


def mean_se(values: pd.Series) -> tuple[float, float]:
    clean = pd.Series(values).dropna().astype(float)
    if clean.empty:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return float(clean.iloc[0]), 0.0
    return float(clean.mean()), float(clean.std(ddof=1) / np.sqrt(len(clean)))


def final_grouped_quality_frame(items: pd.DataFrame, validity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    validity_lookup = validity.set_index(["dataset", "setting", "generator"]) if not validity.empty and "generator" in validity.columns else pd.DataFrame()
    validity_setting_lookup = validity_setting_level_frame(validity).set_index(["dataset", "setting"])

    def add_row(dataset: str, section: str, label: str, setting: str, source: str, generator: str | None, validity_setting: str | None) -> None:
        subset = items[(items["dataset"] == dataset) & (items["setting"] == setting)]
        if generator is not None:
            subset = subset[subset["generator"] == generator]
        difficulty, difficulty_se = mean_se(subset["difficulty"])
        discrimination, discrimination_se = mean_se(subset["discrimination"])
        mean_flaws = float("nan")
        mean_flaws_se = float("nan")
        validity_available = False
        if validity_setting:
            row = None
            if generator is None:
                lookup_key = (dataset, validity_setting)
                if lookup_key in validity_setting_lookup.index:
                    row = validity_setting_lookup.loc[lookup_key]
            elif not validity_lookup.empty:
                lookup_key = (dataset, validity_setting, generator)
                if lookup_key in validity_lookup.index:
                    row = validity_lookup.loc[lookup_key]

            if row is not None:
                mean_flaws = float(row["mean_flaws"])
                mean_flaws_se = float(row["mean_flaws_se"])
                validity_available = True
        rows.append(
            {
                "dataset": dataset,
                "section": section,
                "label": label,
                "setting": setting,
                "source": source,
                "generator": generator or "",
                "n_items": int(len(subset)),
                "difficulty": difficulty,
                "difficulty_se": difficulty_se,
                "discrimination": discrimination,
                "discrimination_se": discrimination_se,
                "mean_flaws": mean_flaws,
                "mean_flaws_se": mean_flaws_se,
                "validity_available": validity_available,
            }
        )

    for dataset in sorted(items["dataset"].unique(), key=dataset_sort_key):
        add_row(dataset, "From Scratch", "Human", "human_from_scratch", "Human", None, "human_from_scratch")
        for generator, label in MODEL_ORDER:
            add_row(dataset, "From Scratch", label, "model_from_scratch", "Model", generator, "model_from_scratch")
    return pd.DataFrame(rows)


def final_ablation_quality_frame(items: pd.DataFrame, validity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    validity_lookup = validity.set_index(["dataset", "setting", "generator"]) if not validity.empty and "generator" in validity.columns else pd.DataFrame()
    validity_setting_lookup = validity_setting_level_frame(validity).set_index(["dataset", "setting"])

    def add_row(dataset: str, label: str, setting: str, source: str, generator: str | None, validity_setting: str | None) -> None:
        subset = items[(items["dataset"] == dataset) & (items["setting"] == setting)]
        if generator is not None:
            subset = subset[subset["generator"] == generator]
        difficulty, difficulty_se = mean_se(subset["difficulty"])
        discrimination, discrimination_se = mean_se(subset["discrimination"])
        mean_flaws = float("nan")
        mean_flaws_se = float("nan")
        validity_available = False
        if validity_setting:
            row = None
            if generator is None:
                lookup_key = (dataset, validity_setting)
                if lookup_key in validity_setting_lookup.index:
                    row = validity_setting_lookup.loc[lookup_key]
            elif not validity_lookup.empty:
                lookup_key = (dataset, validity_setting, generator)
                if lookup_key in validity_lookup.index:
                    row = validity_lookup.loc[lookup_key]

            if row is not None:
                mean_flaws = float(row["mean_flaws"])
                mean_flaws_se = float(row["mean_flaws_se"])
                validity_available = True
        rows.append(
            {
                "dataset": dataset,
                "label": label,
                "setting": setting,
                "source": source,
                "generator": generator or "",
                "n_items": int(len(subset)),
                "difficulty": difficulty,
                "difficulty_se": difficulty_se,
                "discrimination": discrimination,
                "discrimination_se": discrimination_se,
                "mean_flaws": mean_flaws,
                "mean_flaws_se": mean_flaws_se,
                "validity_available": validity_available,
            }
        )

    for dataset in sorted(items["dataset"].unique(), key=dataset_sort_key):
        for generator, label in MODEL_ORDER:
            add_row(dataset, f"{label} Augment Human", "augment_human", "Augment Human", generator, "augment_human")
        for generator, label in MODEL_ORDER:
            add_row(dataset, f"{label} Augment Model", "augment_model", "Augment Model", generator, "augment_model")
        for generator, label in MODEL_ORDER:
            add_row(dataset, f"{label} Augment Ablation", "augment_ablation", "Ablation", generator, "augment_ablation")
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path

def dataset_sort_key(dataset: str) -> tuple[int, str]:
    value = str(dataset)
    return (DATASET_ORDER.index(value), value) if value in DATASET_ORDER else (len(DATASET_ORDER), value)


def setting_sort_key(setting: str) -> tuple[int, str]:
    value = str(setting)
    return (SETTING_NAMES.index(value), value) if value in SETTING_NAMES else (len(SETTING_NAMES), value)


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
    item_prior_sd: float | None = None,
) -> list[Path]:
    del item_prior_sd
    frame = load_irt_frame(
        collected_root,
        generators=generators,
        evaluators=evaluators,
        datasets=datasets,
        settings=settings,
        modes=modes or ["full_question"],
    )
    design = make_design(frame)
    fit = fit_design(design, maxiter=maxiter, maxfun=maxfun, gtol=gtol)
    if not fit.success:
        fit = fit_design(design, maxiter=maxiter, maxfun=maxfun, gtol=gtol, init_beta=fit.beta)

    root = Path(output_dir)
    tables = root / "tables"

    setting = setting_difficulty_frame(fit)
    dataset = dataset_difficulty_frame(fit)
    generator = generator_difficulty_frame(fit)
    test_taker = taker_ability_frame(fit)
    stems = stem_difficulty_frame(fit)
    items = item_parameters_frame(fit)
    item_fit = item_fit_frame(fit)
    quality = irt_quality_summary_frame(items)
    validity = benchmarker_validity_frame()
    combined_quality = combined_quality_frame(quality, validity)
    final_quality = final_grouped_quality_frame(items, validity)
    ablation_quality = final_ablation_quality_frame(items, validity)

    outputs = [
        write_csv(setting, tables / "setting_difficulty.csv"),
        write_csv(dataset, tables / "dataset_difficulty.csv"),
        write_csv(generator, tables / "generator_difficulty.csv"),
        write_csv(test_taker, tables / "test_taker_ability.csv"),
        write_csv(stems, tables / "stem_difficulty.csv"),
        write_csv(items, tables / "instantiated_item_parameters.csv"),
        write_csv(item_fit, tables / "item_fit.csv"),
        write_csv(quality, tables / "irt_quality_by_dataset_setting.csv"),
        write_csv(validity, tables / "benchmarker_validity_by_dataset_setting.csv"),
        write_csv(combined_quality, tables / "question_quality_by_dataset_setting.csv"),
        write_csv(final_quality, tables / "final_grouped_question_quality.csv"),
        write_csv(ablation_quality, tables / "final_ablation_question_quality.csv"),
        write_csv(residual_summary_frame(fit), tables / "residual_summary.csv"),
        write_csv(raw_generator_setting_frame(frame), tables / "generator_setting_raw_accuracy.csv"),
        write_csv(items.head(TOP_N_ITEMS), tables / "hardest_instantiated_items.csv"),
        write_csv(items.tail(TOP_N_ITEMS), tables / "easiest_instantiated_items.csv"),
        write_csv(stems.head(TOP_N_ITEMS), tables / "hardest_stems.csv"),
        write_csv(stems.tail(TOP_N_ITEMS), tables / "easiest_stems.csv"),
        write_csv(item_fit.head(TOP_N_ITEMS), tables / "highest_outfit_items.csv"),
        write_csv(item_fit.tail(TOP_N_ITEMS), tables / "lowest_outfit_items.csv"),
    ]

    summary = {
        "n_obs": int(len(frame)),
        "n_test_takers": int(frame["test_taker"].nunique()),
        "n_stems": int(frame["stem_id"].nunique()),
        "n_instantiated_items": int(frame["item_id"].nunique()),
        "n_params": int(design.n_params),
        "equation": "P(X_ij=1)=c_i+(1-c_i)*sigmoid(1.702*a_i*(theta_j-b_i)); b_i=dataset+stem+generator+setting+item_noise",
        "fit": {
            "success": fit.success,
            "message": fit.message,
            "iterations": fit.iterations,
            "log_likelihood": fit.log_likelihood,
            "aic": fit.aic,
            "bic": fit.bic,
        },
        "regularization": {
            "stem_sd": STEM_PRIOR_SD,
            "item_noise_sd": ITEM_NOISE_PRIOR_SD,
            "log_discrimination_sd": LOG_DISCRIMINATION_PRIOR_SD,
            "guessing_logit_sd": GUESSING_PRIOR_SD,
            "guessing_center": "logit(1 / num_choices)",
        },
        "quality_figure": {"benchmarker_jsonl": str(DEFAULT_BENCHMARKER_JSONL)},
        "references": {name: block.reference for name, block in design.blocks.items() if block.reference is not None},
        "filters": {
            "generators": generators or [],
            "test_takers": evaluators or [],
            "datasets": datasets or [],
            "settings": settings or [],
            "modes": modes or ["full_question"],
        },
    }
    summary_path = root / "fit_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    outputs.append(summary_path)
    return outputs


def selected_values(raw: str | None, allowed: list[str] | None = None) -> list[str] | None:
    values = csv_values(raw)
    if values is None or allowed is None:
        return values
    bad = [value for value in values if value not in set(allowed)]
    if bad:
        raise ValueError(f"Unsupported values: {', '.join(bad)}")
    return values


def run_cli(args) -> int:
    outputs = run_irt_analysis(
        collected_root=Path(args.collected_root),
        output_dir=Path(args.output_dir),
        generators=selected_values(args.generators),
        evaluators=selected_values(args.evaluators),
        datasets=selected_values(args.datasets),
        settings=selected_values(args.settings, SETTING_NAMES),
        maxiter=int(args.maxiter),
        maxfun=int(args.maxfun),
        gtol=float(args.gtol),
    )
    for output in outputs:
        print(output)
    return 0

import argparse

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit IRT model")
    parser.add_argument("--collected-root", default="results/inspect/evaluation")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generators", default=None)
    parser.add_argument("--evaluators", default=None)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--settings", default=None)
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--maxfun", type=int, default=50000)
    parser.add_argument("--gtol", type=float, default=1e-5)
    return parser

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_cli(args)

if __name__ == "__main__":
    raise SystemExit(main())
