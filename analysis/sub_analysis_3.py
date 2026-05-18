"""Sub-analysis 3: distractor-choice sources and distractor diversity."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.visualize import _display_eval_model  # noqa: E402
from utils.constants import CHOICE_LABELS, DEFAULT_COLLECTED_DATASET_ROOT, EVALUATED_STORE_MANIFEST  # noqa: E402


FULL_QUESTION_MODE = "full_question"
SOURCE_SETTINGS = ("augment_human", "augment_model")
PREREQUISITE_SETTINGS = {"augment_human": "human_from_scratch", "augment_model": "model_from_scratch"}
SOURCE_LOAD_SETTINGS = (*SOURCE_SETTINGS, *PREREQUISITE_SETTINGS.values())
DIVERSITY_GROUPS = (
    ("human_from_scratch", "human", "human_distractors"),
    ("model_from_scratch", "model_from_scratch", "model_distractors"),
    ("augment_human", "augment_human_round2", "model_distractors"),
    ("augment_model", "augment_model_round1", "round1_distractors"),
    ("augment_model", "augment_model_round2", "round2_distractors"),
    ("augment_model", "augment_model_all", "model_distractors"),
    ("augment_ablation", "augment_ablation", "model_distractors"),
)
EXPECTED_ROUND_PROPORTIONS = {"round1": 1.0 / 3.0, "round2": 2.0 / 3.0}
SOURCE_COLORS = {"round1": "#4C78A8", "round2": "#F58518", "ambiguous": "#9E9E9E"}
IRT_TABLE_DIR = ROOT / "results" / "augmented_mcqa_irt" / "tables"


def _ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    try:
        _ = torch.ones(1, device="cuda") + 1
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"CUDA unavailable for this PyTorch build, falling back to CPU: {exc}", file=sys.stderr)
        return "cpu"
    return "cuda"


def _generator_key(generation_run_name: str, generation_model: str) -> str:
    if generation_run_name and generation_run_name not in generation_model:
        return f"{generation_run_name}/{generation_model}"
    return generation_model or generation_run_name


def _iter_evaluated_manifests(root: Path) -> Iterable[tuple[Path, dict[str, object]]]:
    for manifest_path in sorted(root.rglob(EVALUATED_STORE_MANIFEST)):
        yield manifest_path.parent, json.loads(manifest_path.read_text(encoding="utf-8"))


def _read_evaluated_dataset(
    group_root: Path,
    manifest: dict[str, object],
    *,
    dataset: str,
    setting: str,
) -> pd.DataFrame:
    path = group_root / dataset / setting / FULL_QUESTION_MODE
    if not path.exists():
        return pd.DataFrame()
    frame = load_from_disk(str(path)).to_pandas()
    if frame.empty:
        return frame
    generation_model = str(manifest.get("generation_model", "") or "")
    generation_run_name = str(manifest.get("generation_run_name", "") or "")
    frame["generator"] = generation_model
    frame["generation_run_name"] = generation_run_name
    frame["generator_key"] = _generator_key(generation_run_name, generation_model)
    frame["eval_model"] = str(manifest.get("evaluation_model", "") or "")
    frame["mode"] = FULL_QUESTION_MODE
    frame["dataset"] = dataset
    frame["setting"] = setting
    return frame


def load_source_rows(collected_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for group_root, manifest in _iter_evaluated_manifests(collected_root):
        for dataset in list(manifest.get("dataset_types") or []):
            for setting in SOURCE_LOAD_SETTINGS:
                frames.append(_read_evaluated_dataset(group_root, manifest, dataset=str(dataset), setting=setting))
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_diversity_rows(collected_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    seen_generation_groups: set[tuple[str, str]] = set()
    for group_root, manifest in _iter_evaluated_manifests(collected_root):
        generation_group = (
            str(manifest.get("generation_run_name", "") or ""),
            str(manifest.get("generation_model", "") or ""),
        )
        if generation_group in seen_generation_groups:
            continue
        seen_generation_groups.add(generation_group)
        for dataset in list(manifest.get("dataset_types") or []):
            for setting in {setting for setting, _, _ in DIVERSITY_GROUPS}:
                frames.append(_read_evaluated_dataset(group_root, manifest, dataset=str(dataset), setting=setting))
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _norm_text(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def _list_difference(values: list[str], remove: list[str]) -> list[str]:
    remove_norms = {_norm_text(item) for item in remove}
    return [item for item in values if _norm_text(item) not in remove_norms]


def _round_lookup_keys(df: pd.DataFrame) -> list[str]:
    preferred = ["generation_run_name", "generator", "dataset", "sample_id", "eval_model"]
    return [column for column in preferred if column in df.columns]


def _first_list(series: pd.Series) -> list[str]:
    for value in series:
        items = _as_list(value)
        if items:
            return items
    return []


def _lookup_prerequisite_distractors(row_df: pd.DataFrame, prerequisite_setting: str) -> dict[tuple[object, ...], list[str]]:
    keys = _round_lookup_keys(row_df)
    if not keys:
        return {}
    source = row_df[row_df["setting"] == prerequisite_setting].copy()
    if source.empty:
        return {}
    distractor_col = "human_distractors" if prerequisite_setting == "human_from_scratch" else "model_distractors"
    lookup: dict[tuple[object, ...], list[str]] = {}
    for key, group in source.groupby(keys, dropna=False, sort=False):
        lookup[key if isinstance(key, tuple) else (key,)] = _first_list(group[distractor_col])
    return lookup


def _row_key(row: pd.Series, keys: list[str]) -> tuple[object, ...]:
    return tuple(row.get(key) for key in keys)


def _round_distractors_from_row(row: pd.Series) -> tuple[list[str], list[str], str]:
    setting = str(row.get("setting", ""))
    human = _as_list(row.get("human_distractors"))
    model = _as_list(row.get("model_distractors"))

    if setting == "augment_human":
        round1 = _as_list(row.get("round1_distractors")) or human
        return round1, model, "matched" if _as_list(row.get("round1_distractors")) else "fallback_row_fields"
    if setting == "augment_model":
        round1 = _as_list(row.get("round1_distractors"))
        if round1:
            return round1, _list_difference(model, round1), "matched"
        return [], [], "missing_prerequisite"
    return [], [], "unsupported_setting"


def attach_round_distractors(row_df: pd.DataFrame) -> pd.DataFrame:
    if row_df.empty or "setting" not in row_df.columns:
        return row_df.copy()

    out = row_df.copy()
    keys = _round_lookup_keys(out)
    human_lookup = _lookup_prerequisite_distractors(out, "human_from_scratch")
    model_lookup = _lookup_prerequisite_distractors(out, "model_from_scratch")
    round1_values: list[list[str]] = []
    round2_values: list[list[str]] = []
    split_statuses: list[str] = []

    for _, row in out.iterrows():
        setting = str(row.get("setting", ""))
        key = _row_key(row, keys)
        human = _as_list(row.get("human_distractors"))
        model = _as_list(row.get("model_distractors"))
        if setting == "augment_human":
            round1 = human_lookup.get(key) or human
            round2 = model
            status = "matched" if key in human_lookup else "fallback_row_fields"
        elif setting == "augment_model":
            round1 = model_lookup.get(key)
            if round1:
                round2 = _list_difference(model, round1)
                status = "matched"
            else:
                round1 = []
                round2 = []
                status = "missing_prerequisite"
        else:
            round1 = []
            round2 = []
            status = "not_augmented"
        round1_values.append(round1)
        round2_values.append(round2)
        split_statuses.append(status)

    out["round1_distractors"] = round1_values
    out["round2_distractors"] = round2_values
    out["round_split_status"] = split_statuses
    return out


def _prediction_index(letter: object, option_count: int) -> int | None:
    prediction = str(letter or "").strip().upper()
    if prediction not in CHOICE_LABELS:
        return None
    index = CHOICE_LABELS.index(prediction)
    return index if index < option_count else None


def _source_for_prediction(row: pd.Series) -> tuple[str | None, str]:
    options = _as_list(row.get("options_randomized"))
    pred_idx = _prediction_index(row.get("evaluation_prediction"), len(options))
    if pred_idx is None:
        return None, "invalid_prediction"

    answer_idx = _prediction_index(row.get("correct_answer_letter"), len(options))
    if answer_idx is not None and pred_idx == answer_idx:
        return None, "correct"
    if bool(row.get("evaluation_used_random_fallback", False)):
        return None, "random_fallback"

    selected = _norm_text(options[pred_idx])
    setting = str(row.get("setting", ""))
    round1, round2, _ = _round_distractors_from_row(row)
    round1_norm = {_norm_text(item) for item in round1}
    round2_norm = {_norm_text(item) for item in round2}

    if setting == "augment_human":
        matches = []
        if selected in round1_norm:
            matches.append("round1")
        if selected in round2_norm:
            matches.append("round2")
    elif setting == "augment_model":
        if not round1_norm and not round2_norm:
            return None, "missing_prerequisite"
        matches = []
        if selected in round1_norm:
            matches.append("round1")
        if selected in round2_norm:
            matches.append("round2")
    else:
        return None, "unsupported_setting"

    if len(matches) == 1:
        return matches[0], "kept"
    if len(matches) > 1:
        return "ambiguous", "ambiguous"
    return None, "unmatched_distractor"


def build_source_choice_rows(row_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_df = attach_round_distractors(row_df)
    subset = row_df[
        (row_df["mode"] == FULL_QUESTION_MODE)
        & (row_df["setting"].isin(SOURCE_SETTINGS))
        & (row_df["evaluation_prediction"].fillna("").astype(str).str.strip() != "")
    ].copy()
    if subset.empty:
        return pd.DataFrame(), pd.DataFrame()

    labels: list[str | None] = []
    statuses: list[str] = []
    for _, row in subset.iterrows():
        source, status = _source_for_prediction(row)
        labels.append(source)
        statuses.append(status)
    subset["distractor_source"] = labels
    subset["source_status"] = statuses
    subset["eval_model_display"] = subset["eval_model"].map(_display_eval_model)
    kept = subset[subset["distractor_source"].isin(["round1", "round2", "ambiguous"])].copy()
    return kept, subset


def summarize_source_choices(source_df: pd.DataFrame) -> pd.DataFrame:
    if source_df.empty:
        return pd.DataFrame()
    group_cols = ["eval_model", "eval_model_display", "setting", "distractor_source"]
    counts = source_df.groupby(group_cols, dropna=False).size().reset_index(name="count")
    totals = counts.groupby(["eval_model", "setting"])["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals
    return counts.sort_values(["setting", "eval_model_display", "distractor_source"]).reset_index(drop=True)


def summarize_source_choices_by_dataset(source_df: pd.DataFrame) -> pd.DataFrame:
    if source_df.empty:
        return pd.DataFrame()
    group_cols = ["dataset", "eval_model", "eval_model_display", "setting", "distractor_source"]
    counts = source_df.groupby(group_cols, dropna=False).size().reset_index(name="count")
    totals = counts.groupby(["dataset", "eval_model", "setting"])["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals
    return counts.sort_values(["dataset", "setting", "eval_model_display", "distractor_source"]).reset_index(drop=True)


def summarize_source_choices_by_generator(source_df: pd.DataFrame) -> pd.DataFrame:
    if source_df.empty:
        return pd.DataFrame()
    group_cols = ["generator", "dataset", "eval_model", "eval_model_display", "setting", "distractor_source"]
    counts = source_df.groupby(group_cols, dropna=False).size().reset_index(name="count")
    totals = counts.groupby(["generator", "dataset", "eval_model", "setting"])["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals
    return counts.sort_values(["generator", "dataset", "setting", "eval_model_display", "distractor_source"]).reset_index(drop=True)


def summarize_diagnostics(diagnostic_df: pd.DataFrame) -> pd.DataFrame:
    if diagnostic_df.empty:
        return pd.DataFrame()
    cols = ["setting", "eval_model", "source_status"]
    out = diagnostic_df.groupby(cols, dropna=False).size().reset_index(name="count")
    out["eval_model_display"] = out["eval_model"].map(_display_eval_model)
    return out.sort_values(["setting", "eval_model_display", "source_status"]).reset_index(drop=True)


def _compact_source_rows(source_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "generation_run_name",
        "generator",
        "dataset",
        "sample_id",
        "eval_model",
        "eval_model_display",
        "setting",
        "evaluation_prediction",
        "correct_answer_letter",
        "distractor_source",
        "source_status",
        "round_split_status",
    ]
    return source_df[[column for column in columns if column in source_df.columns]].copy()


def _compact_diagnostic_rows(diagnostic_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "generation_run_name",
        "generator",
        "dataset",
        "sample_id",
        "eval_model",
        "setting",
        "evaluation_prediction",
        "correct_answer_letter",
        "distractor_source",
        "source_status",
        "round_split_status",
        "evaluation_used_random_fallback",
    ]
    compact = diagnostic_df[[column for column in columns if column in diagnostic_df.columns]].copy()
    if "eval_model" in compact.columns:
        compact["eval_model_display"] = compact["eval_model"].map(_display_eval_model)
    return compact


def _plot_stacked_source(summary: pd.DataFrame, out_path: Path) -> None:
    settings = [setting for setting in SOURCE_SETTINGS if setting in set(summary["setting"])]
    fig, axes = plt.subplots(len(settings), 1, figsize=(13, 4.8 * max(1, len(settings))), squeeze=False)
    for ax, setting in zip(axes[:, 0], settings):
        setting_df = summary[summary["setting"] == setting]
        pivot = (
            setting_df.pivot_table(
                index="eval_model_display",
                columns="distractor_source",
                values="proportion",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reindex(columns=["round1", "round2", "ambiguous"], fill_value=0.0)
            .sort_index()
        )
        x = np.arange(len(pivot))
        left = np.zeros(len(pivot))
        for source in pivot.columns:
            values = pivot[source].to_numpy()
            ax.bar(x, values, bottom=left, label=source, color=SOURCE_COLORS.get(source))
            left += values
        ax.axhline(EXPECTED_ROUND_PROPORTIONS["round1"], color="#333333", linestyle="--", linewidth=1)
        ax.text(
            len(pivot) - 0.5,
            EXPECTED_ROUND_PROPORTIONS["round1"] + 0.015,
            "33.3% round 1",
            ha="right",
            va="bottom",
            fontsize=9,
            color="#333333",
        )
        ax.set_title(setting)
        ax.set_ylabel("Share of wrong distractor picks")
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=35, ha="right")
        ax.legend(frameon=False, ncols=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_stacked_source_by_dataset(summary: pd.DataFrame, out_path: Path) -> None:
    datasets = sorted(summary["dataset"].unique())
    settings = [setting for setting in SOURCE_SETTINGS if setting in set(summary["setting"])]
    fig, axes = plt.subplots(
        len(settings),
        len(datasets),
        figsize=(6.0 * max(1, len(datasets)), 4.8 * max(1, len(settings))),
        squeeze=False,
        sharey=True,
    )
    for row_idx, setting in enumerate(settings):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            setting_df = summary[(summary["setting"] == setting) & (summary["dataset"] == dataset)]
            pivot = (
                setting_df.pivot_table(
                    index="eval_model_display",
                    columns="distractor_source",
                    values="proportion",
                    aggfunc="sum",
                    fill_value=0.0,
                )
                .reindex(columns=["round1", "round2", "ambiguous"], fill_value=0.0)
                .sort_index()
            )
            x = np.arange(len(pivot))
            left = np.zeros(len(pivot))
            for source in pivot.columns:
                values = pivot[source].to_numpy()
                ax.bar(x, values, bottom=left, color=SOURCE_COLORS.get(source), label=source)
                left += values
            ax.axhline(EXPECTED_ROUND_PROPORTIONS["round1"], color="#333333", linestyle="--", linewidth=1)
            ax.set_title(f"{setting} | {dataset}")
            ax.set_ylim(0, 1)
            ax.set_xticks(x)
            ax.set_xticklabels(pivot.index, rotation=55, ha="right", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel("Share")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, ncols=3, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_round1_by_generator(summary: pd.DataFrame, out_path: Path) -> None:
    round1 = summary[summary["distractor_source"] == "round1"].copy()
    if round1.empty:
        return
    generators = sorted(round1["generator"].unique())
    datasets = sorted(round1["dataset"].unique())
    fig, axes = plt.subplots(
        len(generators),
        len(SOURCE_SETTINGS),
        figsize=(8.0 * len(SOURCE_SETTINGS), 3.8 * len(generators)),
        squeeze=False,
        sharey=True,
    )
    for row_idx, generator in enumerate(generators):
        for col_idx, setting in enumerate(SOURCE_SETTINGS):
            ax = axes[row_idx, col_idx]
            subset = round1[(round1["generator"] == generator) & (round1["setting"] == setting)]
            pivot = subset.pivot_table(
                index="eval_model_display",
                columns="dataset",
                values="proportion",
                aggfunc="mean",
            ).reindex(columns=datasets).sort_index()
            x = np.arange(len(pivot))
            width = 0.8 / max(1, len(datasets))
            for d_idx, dataset in enumerate(datasets):
                ax.bar(
                    x + (d_idx - (len(datasets) - 1) / 2) * width,
                    pivot[dataset],
                    width=width,
                    label=dataset,
                )
            ax.axhline(EXPECTED_ROUND_PROPORTIONS["round1"], color="#222222", linestyle="--", linewidth=1)
            ax.set_ylim(0, 0.75)
            ax.set_title(f"{setting}\n{generator}")
            ax.set_xticks(x)
            ax.set_xticklabels(pivot.index, rotation=55, ha="right", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel("Round-1 share")
            if row_idx == 0 and col_idx == len(SOURCE_SETTINGS) - 1:
                ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _unique_distractor_sets(row_df: pd.DataFrame) -> pd.DataFrame:
    row_df = attach_round_distractors(row_df)
    metadata_cols = ["generation_run_name", "generator", "generator_key", "dataset", "sample_id", "setting"]
    cols = metadata_cols + ["answer", "human_distractors", "model_distractors", "round1_distractors", "round2_distractors"]
    source = row_df[row_df["mode"] == FULL_QUESTION_MODE].drop_duplicates(metadata_cols)
    rows: list[dict[str, object]] = []
    for _, row in source[cols].iterrows():
        for setting, group, selector in DIVERSITY_GROUPS:
            if row["setting"] != setting:
                continue
            if selector == "human_distractors":
                choices = _as_list(row.get("human_distractors"))
            else:
                model = _as_list(row.get("model_distractors"))
                if selector == "round1_distractors":
                    choices = _as_list(row.get("round1_distractors"))
                elif selector == "round2_distractors":
                    choices = _as_list(row.get("round2_distractors"))
                else:
                    choices = model
            rows.append(
                {
                    **{col: row[col] for col in metadata_cols},
                    "diversity_group": group,
                    "answer": str(row.get("answer", "") or "").strip(),
                    "choices": choices,
                    "choice_count": len(choices),
                }
            )
    return pd.DataFrame(rows)


class MiniLMEmbedder:
    def __init__(self, model_name: str, cache_dir: Path, batch_size: int, device: str, max_length: int) -> None:
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
        self.model = AutoModel.from_pretrained(model_name, cache_dir=str(cache_dir)).to(self.device)
        self.model.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        total_batches = math.ceil(len(texts) / self.batch_size) if texts else 0
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.model(**encoded)
                token_embeddings = output.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.cpu().numpy())
                batch_idx = (start // self.batch_size) + 1
                if batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0:
                    print(f"embedded batch {batch_idx}/{total_batches}", file=sys.stderr, flush=True)
        return np.vstack(vectors) if vectors else np.empty((0, 0), dtype=np.float32)


def _encode_text_chunk(
    payload: tuple[int, list[str], str, str, int, int, int],
) -> tuple[int, np.ndarray]:
    chunk_index, texts, model_name, cache_dir, batch_size, max_length, threads = payload
    torch.set_num_threads(max(1, threads))
    embedder = MiniLMEmbedder(model_name, Path(cache_dir), batch_size, "cpu", max_length)
    vectors = embedder.encode(texts)
    print(f"finished chunk {chunk_index}", file=sys.stderr, flush=True)
    return chunk_index, vectors


def _load_or_compute_embeddings(
    texts: list[str],
    *,
    model_name: str,
    cache_dir: Path,
    batch_size: int,
    device: str,
    max_length: int,
    workers: int,
    threads: int,
) -> dict[str, np.ndarray]:
    text_path = cache_dir / "texts.json"
    vector_path = cache_dir / "embeddings.npy"
    if text_path.exists() and vector_path.exists():
        cached_texts = json.loads(text_path.read_text(encoding="utf-8"))
        if cached_texts == texts:
            vectors = np.load(vector_path)
            return {text: vectors[index] for index, text in enumerate(texts)}

    resolved_device = _resolve_device(device)
    print(f"embedding {len(texts)} unique distractor strings on {resolved_device}", file=sys.stderr, flush=True)
    if resolved_device == "cpu" and workers > 1:
        chunk_size = math.ceil(len(texts) / workers)
        chunks = [texts[start : start + chunk_size] for start in range(0, len(texts), chunk_size)]
        worker_threads = max(1, threads // max(1, len(chunks)))
        payloads = [
            (index, chunk, model_name, str(cache_dir), batch_size, max_length, worker_threads)
            for index, chunk in enumerate(chunks)
        ]
        vectors_by_chunk: dict[int, np.ndarray] = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(payloads)) as executor:
            for chunk_index, chunk_vectors in executor.map(_encode_text_chunk, payloads):
                vectors_by_chunk[chunk_index] = chunk_vectors
        vectors = np.vstack([vectors_by_chunk[index] for index in range(len(payloads))])
    else:
        if resolved_device == "cpu" and threads > 0:
            torch.set_num_threads(threads)
        embedder = MiniLMEmbedder(model_name, cache_dir, batch_size, resolved_device, max_length)
        vectors = embedder.encode(texts)
    text_path.write_text(json.dumps(texts, indent=2), encoding="utf-8")
    np.save(vector_path, vectors)
    return {text: vectors[index] for index, text in enumerate(texts)}


def _mean_pairwise_cosine_distance(choices: list[str], embeddings: dict[str, np.ndarray]) -> float:
    normalized = [_norm_text(choice) for choice in choices if _norm_text(choice)]
    if len(normalized) < 2:
        return float("nan")
    matrix = np.vstack([embeddings[text] for text in normalized])
    sims = matrix @ matrix.T
    upper = sims[np.triu_indices(len(normalized), k=1)]
    return float(np.mean(1.0 - upper)) if len(upper) else float("nan")


def _answer_similarity_stats(answer: str, choices: list[str], embeddings: dict[str, np.ndarray]) -> tuple[float, float]:
    answer_norm = _norm_text(answer)
    choice_norms = [_norm_text(choice) for choice in choices if _norm_text(choice)]
    if not answer_norm or not choice_norms:
        return float("nan"), float("nan")
    answer_vec = embeddings.get(answer_norm)
    if answer_vec is None:
        return float("nan"), float("nan")
    sims = np.vstack([embeddings[text] for text in choice_norms]) @ answer_vec
    return float(np.mean(sims)), float(np.max(sims))


def compute_diversity(
    row_df: pd.DataFrame,
    *,
    model_name: str,
    cache_dir: Path,
    batch_size: int,
    device: str,
    max_length: int,
    workers: int,
    threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sets_df = _unique_distractor_sets(row_df)
    if sets_df.empty:
        return sets_df, pd.DataFrame()
    all_texts = sorted(
        {
            text
            for _, row in sets_df.iterrows()
            for text in [_norm_text(row.get("answer", "")), *[_norm_text(choice) for choice in row["choices"]]]
            if text
        }
    )
    embeddings = _load_or_compute_embeddings(
        all_texts,
        model_name=model_name,
        cache_dir=cache_dir,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
        workers=workers,
        threads=threads,
    )
    sets_df["mean_pairwise_cosine_distance"] = sets_df["choices"].map(
        lambda choices: _mean_pairwise_cosine_distance(choices, embeddings)
    )
    answer_stats = [
        _answer_similarity_stats(str(row.get("answer", "") or ""), row["choices"], embeddings)
        for _, row in sets_df.iterrows()
    ]
    sets_df["mean_answer_similarity"] = [item[0] for item in answer_stats]
    sets_df["max_answer_similarity"] = [item[1] for item in answer_stats]
    summary = (
        sets_df.dropna(subset=["mean_pairwise_cosine_distance"])
        .groupby(["generator_key", "dataset", "diversity_group"], dropna=False)
        .agg(
            n=("mean_pairwise_cosine_distance", "size"),
            mean_diversity=("mean_pairwise_cosine_distance", "mean"),
            sd_diversity=("mean_pairwise_cosine_distance", "std"),
            mean_answer_similarity=("mean_answer_similarity", "mean"),
            max_answer_similarity=("max_answer_similarity", "mean"),
            mean_choice_count=("choice_count", "mean"),
        )
        .reset_index()
    )
    summary["se_diversity"] = summary["sd_diversity"] / np.sqrt(summary["n"].clip(lower=1))
    summary["ci95"] = 1.96 * summary["se_diversity"].fillna(0.0)
    return sets_df, summary


def _plot_diversity_by_group(diversity_df: pd.DataFrame, out_path: Path) -> None:
    clean = diversity_df.dropna(subset=["mean_pairwise_cosine_distance"]).copy()
    groups = [group for _, group, _ in DIVERSITY_GROUPS if group in set(clean["diversity_group"])]
    data = [clean.loc[clean["diversity_group"] == group, "mean_pairwise_cosine_distance"].to_numpy() for group in groups]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.boxplot(data, tick_labels=groups, showfliers=False)
    means = [np.nanmean(values) if len(values) else np.nan for values in data]
    ax.scatter(np.arange(1, len(groups) + 1), means, color="#D62728", zorder=3, label="mean")
    ax.set_ylabel("Mean pairwise cosine distance")
    ax.set_title("Distractor semantic diversity by source group")
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_diversity_summary(summary: pd.DataFrame, out_path: Path) -> None:
    clean = summary.copy()
    groups = [group for _, group, _ in DIVERSITY_GROUPS if group in set(clean["diversity_group"])]
    datasets = sorted(clean["dataset"].unique())
    fig, axes = plt.subplots(len(datasets), 1, figsize=(13, 4.0 * max(1, len(datasets))), squeeze=False, sharex=True)
    for ax, dataset in zip(axes[:, 0], datasets):
        dataset_df = clean[clean["dataset"] == dataset]
        pivot = dataset_df.groupby("diversity_group", dropna=False).agg(
            mean=("mean_diversity", "mean"),
            ci=("ci95", "mean"),
        )
        values = [float(pivot.loc[group, "mean"]) if group in pivot.index else math.nan for group in groups]
        errors = [float(pivot.loc[group, "ci"]) if group in pivot.index else 0.0 for group in groups]
        x = np.arange(len(groups))
        ax.bar(x, values, yerr=errors, color="#4C78A8", capsize=3)
        ax.set_title(dataset)
        ax.set_ylabel("Cosine distance")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_answer_similarity_summary(summary: pd.DataFrame, out_path: Path) -> None:
    clean = summary.copy()
    groups = [group for _, group, _ in DIVERSITY_GROUPS if group in set(clean["diversity_group"])]
    datasets = sorted(clean["dataset"].unique())
    fig, axes = plt.subplots(len(datasets), 1, figsize=(13, 4.0 * max(1, len(datasets))), squeeze=False, sharex=True)
    for ax, dataset in zip(axes[:, 0], datasets):
        dataset_df = clean[clean["dataset"] == dataset]
        pivot = dataset_df.groupby("diversity_group", dropna=False)["mean_answer_similarity"].mean()
        values = [float(pivot.loc[group]) if group in pivot.index else math.nan for group in groups]
        x = np.arange(len(groups))
        ax.bar(x, values, color="#72B7B2")
        ax.set_title(dataset)
        ax.set_ylabel("Mean cosine similarity to answer")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _generator_from_key(generator_key: str) -> str:
    if "openai/gpt-5.2" in generator_key:
        return "openai/gpt-5.2-2025-12-11"
    if "google/gemini-3.1-pro-preview" in generator_key:
        return "google/gemini-3.1-pro-preview"
    if "together/Qwen/Qwen3.5-397B-A17B" in generator_key:
        return "together/Qwen/Qwen3.5-397B-A17B"
    return str(generator_key)


def _irt_relation_frame(source_by_generator: pd.DataFrame, diversity_summary: pd.DataFrame) -> pd.DataFrame:
    path = IRT_TABLE_DIR / "final_ablation_question_quality.csv"
    if not path.exists() or diversity_summary.empty:
        return pd.DataFrame()
    irt = pd.read_csv(path)
    div = diversity_summary.copy()
    div["generator"] = div["generator_key"].map(_generator_from_key)
    div_group_by_setting = {
        "augment_human": "augment_human_round2",
        "augment_model": "augment_model_all",
        "augment_ablation": "augment_ablation",
    }
    source_round1 = source_by_generator[source_by_generator["distractor_source"] == "round1"].copy()
    rows: list[dict[str, object]] = []
    for _, row in irt.iterrows():
        setting = str(row["setting"])
        div_group = div_group_by_setting.get(setting)
        drow = div[
            (div["dataset"] == row["dataset"])
            & (div["generator"] == row["generator"])
            & (div["diversity_group"] == div_group)
        ]
        srow = source_round1[
            (source_round1["dataset"] == row["dataset"])
            & (source_round1["generator"] == row["generator"])
            & (source_round1["setting"] == setting)
        ]
        out = row.to_dict()
        out["diversity_group"] = div_group
        for col in ["mean_diversity", "mean_answer_similarity", "max_answer_similarity"]:
            out[col] = float(drow[col].iloc[0]) if len(drow) else float("nan")
        if len(srow):
            out["round1_share"] = float(np.average(srow["proportion"], weights=srow["count"]))
            out["round1_count"] = int(srow["count"].sum())
        else:
            out["round1_share"] = float("nan")
            out["round1_count"] = 0
        rows.append(out)
    return pd.DataFrame(rows)


def _plot_irt_relation(frame: pd.DataFrame, out_path: Path) -> None:
    if frame.empty:
        return
    x_metrics = [
        ("mean_diversity", "Distractor diversity"),
        ("mean_answer_similarity", "Mean similarity to answer"),
        ("round1_share", "Round-1 chosen share"),
    ]
    y_metrics = [
        ("difficulty", "IRT difficulty"),
        ("discrimination", "IRT discrimination"),
        ("mean_flaws", "Writing flaws"),
    ]
    dataset_labels = {
        "arc_challenge": "ARC",
        "mmlu_pro": "MMLU",
        "gpqa": "GPQA",
    }
    fig, axes = plt.subplots(len(y_metrics), len(x_metrics), figsize=(14, 10.5), squeeze=False)
    colors = {"augment_human": "#4C78A8", "augment_model": "#F58518", "augment_ablation": "#54A24B"}
    markers = {"augment_human": "o", "augment_model": "s", "augment_ablation": "^"}
    for row_idx, (y_col, y_label) in enumerate(y_metrics):
        for col_idx, (x_col, x_label) in enumerate(x_metrics):
            ax = axes[row_idx, col_idx]
            for setting, subset in frame.groupby("setting", sort=True):
                ax.scatter(
                    subset[x_col],
                    subset[y_col],
                    label=setting,
                    alpha=0.85,
                    color=colors.get(setting),
                    marker=markers.get(setting, "o"),
                    s=42,
                    edgecolor="white",
                    linewidth=0.35,
                )
                for _, point in subset.iterrows():
                    if not np.isfinite(point.get(x_col, np.nan)) or not np.isfinite(point.get(y_col, np.nan)):
                        continue
                    ax.annotate(
                        dataset_labels.get(str(point.get("dataset", "")), str(point.get("dataset", ""))),
                        (float(point[x_col]), float(point[y_col])),
                        xytext=(4, 3),
                        textcoords="offset points",
                        fontsize=6,
                        alpha=0.75,
                        color="#222222",
                    )
            clean = frame[[x_col, y_col]].dropna()
            if len(clean) >= 2:
                corr = clean.corr().iloc[0, 1]
                ax.text(0.02, 0.95, f"r={corr:.2f}", transform=ax.transAxes, va="top", fontsize=9)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncols=len(labels),
            frameon=False,
            fontsize=9,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> list[Path]:
    collected_root = Path(args.collected_root)
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    cache_dir = Path(args.cache_dir)
    _ensure_dirs(figures_dir, tables_dir, cache_dir)

    source_input_df = load_source_rows(collected_root)
    if source_input_df.empty:
        raise ValueError(f"No evaluated rows found under {collected_root}")

    outputs: list[Path] = []
    source_df, diagnostic_df = build_source_choice_rows(source_input_df)
    source_summary = summarize_source_choices(source_df)
    dataset_source_summary = summarize_source_choices_by_dataset(source_df)
    generator_source_summary = summarize_source_choices_by_generator(source_df)
    diagnostics = summarize_diagnostics(diagnostic_df)

    for name, df in [
        ("distractor_source_rows.csv", _compact_source_rows(source_df)),
        ("distractor_source_summary.csv", source_summary),
        ("distractor_source_by_dataset.csv", dataset_source_summary),
        ("distractor_source_by_generator_dataset_eval_model.csv", generator_source_summary),
        ("distractor_source_diagnostics.csv", diagnostics),
        ("distractor_source_diagnostic_rows.csv", _compact_diagnostic_rows(diagnostic_df)),
    ]:
        path = tables_dir / name
        df.to_csv(path, index=False)
        outputs.append(path)

    if not source_summary.empty:
        path = figures_dir / "distractor_source_by_eval_model.png"
        _plot_stacked_source(source_summary, path)
        outputs.append(path)
    if not dataset_source_summary.empty:
        path = figures_dir / "distractor_source_by_eval_model_dataset.png"
        _plot_stacked_source_by_dataset(dataset_source_summary, path)
        outputs.append(path)
    if not generator_source_summary.empty:
        path = figures_dir / "round1_share_by_generator_eval_model.png"
        _plot_round1_by_generator(generator_source_summary, path)
        outputs.append(path)

    if not args.skip_diversity:
        diversity_input_df = load_diversity_rows(collected_root)
        if diversity_input_df.empty:
            raise ValueError(f"No diversity rows found under {collected_root}")
        diversity_df, diversity_summary = compute_diversity(
            diversity_input_df,
            model_name=args.embedding_model,
            cache_dir=cache_dir,
            batch_size=args.batch_size,
            device=args.device,
            max_length=args.max_length,
            workers=args.workers,
            threads=args.threads,
        )
        for name, df in [
            ("distractor_diversity_rows.csv", diversity_df),
            ("distractor_diversity_summary.csv", diversity_summary),
        ]:
            path = tables_dir / name
            df.to_csv(path, index=False)
            outputs.append(path)

        if not diversity_df.empty:
            path = figures_dir / "distractor_diversity_by_setting.png"
            _plot_diversity_by_group(diversity_df, path)
            outputs.append(path)
        if not diversity_summary.empty:
            path = figures_dir / "distractor_diversity_summary.png"
            _plot_diversity_summary(diversity_summary, path)
            outputs.append(path)
            path = figures_dir / "answer_similarity_summary.png"
            _plot_answer_similarity_summary(diversity_summary, path)
            outputs.append(path)
            relation = _irt_relation_frame(generator_source_summary, diversity_summary)
            relation_path = tables_dir / "irt_writing_flaws_relation.csv"
            relation.to_csv(relation_path, index=False)
            outputs.append(relation_path)
            relation_fig = figures_dir / "irt_writing_flaws_relation.png"
            _plot_irt_relation(relation, relation_fig)
            outputs.append(relation_fig)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected-root", type=Path, default=DEFAULT_COLLECTED_DATASET_ROOT)
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "analysis" / "figures" / "sub_analysis_3")
    parser.add_argument("--tables-dir", type=Path, default=ROOT / "analysis" / "tables" / "sub_analysis_3")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "analysis" / "cache" / "sub_analysis_3_embeddings")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device such as cuda:0")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-diversity", action="store_true")
    return parser.parse_args()


def main() -> None:
    outputs = run(parse_args())
    print("Wrote:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
