"""Hybrid distractor-overlap analysis for the memorization table.

This script computes a contamination/memorization proxy for 4-choice MCQs:

* ordinary text distractors match when cosine similarity >= 0.90
* numeric, mathematical, or formula-like distractors match only by normalized
  exact string match

It writes:

* hybrid_memorization_summary_by_dataset_model.csv
* hybrid_match_distribution_by_dataset_model.csv
* hybrid_flagged_items_0p90_formula_exact.csv
* gpqa_hybrid_manual_audit_30.csv
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.constants import DEFAULT_COLLECTED_DATASET_ROOT, EVALUATED_STORE_MANIFEST  # noqa: E402


DATASET_ORDER = ["arc_challenge", "mmlu_pro", "gpqa"]
GENERATOR_ORDER = [
    "openai/gpt-5.2-2025-12-11",
    "google/gemini-3.1-pro-preview",
    "together/Qwen/Qwen3.5-397B-A17B",
]
DATASET_LABEL = {"arc_challenge": "ARC", "mmlu_pro": "MMLU", "gpqa": "GPQA"}
GENERATOR_LABEL = {
    "openai/gpt-5.2-2025-12-11": "GPT-5.2",
    "google/gemini-3.1-pro-preview": "Gemini-3.1 Pro",
    "together/Qwen/Qwen3.5-397B-A17B": "Qwen-3.5",
}

FORMULA_SYMBOLS = set("=^_\\{}[]<>±≈≃≅≤≥≠√∑∫∞πΠΔδθλμℏ°%")
NUMBER_RE = re.compile(r"\d")
LATEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]+")
CHEM_FORMULA_RE = re.compile(r"(?=.*[A-Z])(?=.*\d)[A-Za-z0-9_{}\\\[\]\(\)\-+=^/]+")


def norm_for_embedding(value: object) -> str:
    return " ".join(str(value).strip().casefold().split())


def norm_for_exact_match(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    translations = str.maketrans(
        {
            "−": "-",
            "–": "-",
            "—": "-",
            "×": "x",
            "∙": "*",
            "·": "*",
            "⁄": "/",
        }
    )
    text = text.translate(translations)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\;", "").replace("\\:", "").replace("\\!", "")
    return re.sub(r"\s+", "", text)


def is_formula_like(value: object) -> bool:
    """Return True for choices where embeddings often overstate similarity.

    This intentionally errs on the side of sending numeric/scientific choices
    to exact matching. Examples: numbers, equations, chemical formulae,
    LaTeX/math notation, inequalities, percentages, and degree-marked values.
    """

    text = str(value).strip()
    if not text:
        return False
    if NUMBER_RE.search(text):
        return True
    if LATEX_COMMAND_RE.search(text):
        return True
    if any(ch in FORMULA_SYMBOLS for ch in text):
        return True
    if CHEM_FORMULA_RE.fullmatch(text.replace(" ", "")):
        return True
    return False


def pair_matches(model_choice: str, human_choice: str, similarity: float, threshold: float) -> tuple[bool, str]:
    if is_formula_like(model_choice) or is_formula_like(human_choice):
        return norm_for_exact_match(model_choice) == norm_for_exact_match(human_choice), "normalized_exact_formula_like"
    return similarity >= threshold, "cosine_text"


def generator_key(generation_run_name: str, generation_model: str) -> str:
    if generation_run_name and generation_run_name not in generation_model:
        return f"{generation_run_name}/{generation_model}"
    return generation_model or generation_run_name


def load_gpqa_question_lookup(collected_root: Path) -> dict[tuple[str, str], dict[str, object]]:
    """Load raw GPQA stems for the manual audit file."""

    lookup: dict[tuple[str, str], dict[str, object]] = {}
    seen_generators: set[str] = set()
    for manifest_path in sorted(collected_root.rglob(EVALUATED_STORE_MANIFEST)):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "gpqa" not in (manifest.get("dataset_types") or []):
            continue
        generation_model = str(manifest.get("generation_model", "") or "")
        generation_run_name = str(manifest.get("generation_run_name", "") or "")
        gen_key = generator_key(generation_run_name, generation_model)
        if gen_key in seen_generators:
            continue
        data_path = manifest_path.parent / "gpqa" / "model_from_scratch" / "full_question"
        if not data_path.exists():
            continue
        seen_generators.add(gen_key)
        dataset = load_from_disk(str(data_path))
        for item in dataset:
            lookup[(gen_key, str(item.get("sample_id")))] = {
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
            }
    return lookup


def best_hybrid_matches(
    row: pd.Series,
    embeddings: dict[str, np.ndarray],
    permutations: list[tuple[int, int, int]],
    threshold: float,
) -> tuple[list[str], list[str], list[tuple[int, int, float, str]]]:
    human_choices = ast.literal_eval(row["human_choices"])
    model_choices = ast.literal_eval(row["model_choices"])
    similarities = np.array(
        [
            [float(embeddings[norm_for_embedding(model)] @ embeddings[norm_for_embedding(human)]) for human in human_choices]
            for model in model_choices
        ]
    )

    matchable: list[list[bool]] = []
    methods: list[list[str]] = []
    for model_idx, model_choice in enumerate(model_choices):
        match_row: list[bool] = []
        method_row: list[str] = []
        for human_idx, human_choice in enumerate(human_choices):
            ok, method = pair_matches(model_choice, human_choice, similarities[model_idx, human_idx], threshold)
            match_row.append(ok)
            method_row.append(method)
        matchable.append(match_row)
        methods.append(method_row)

    best: tuple[tuple[int, float], list[tuple[int, int, float, str]]] | None = None
    for permutation in permutations:
        matched = [
            (model_idx, permutation[model_idx], similarities[model_idx, permutation[model_idx]], methods[model_idx][permutation[model_idx]])
            for model_idx in range(3)
            if matchable[model_idx][permutation[model_idx]]
        ]
        score = (len(matched), sum(float(item[2]) for item in matched))
        if best is None or score > best[0]:
            best = (score, matched)
    assert best is not None
    return human_choices, model_choices, best[1]


def build_item_rows(args: argparse.Namespace) -> pd.DataFrame:
    texts = json.loads(Path(args.embedding_texts).read_text(encoding="utf-8"))
    vectors = np.load(args.embedding_vectors)
    embeddings = {text: vectors[index] for index, text in enumerate(texts)}
    permutations = list(itertools.permutations(range(3)))

    source = pd.read_csv(args.overlap_csv)
    rows: list[dict[str, object]] = []
    for _, row in source.iterrows():
        human_choices, model_choices, matched = best_hybrid_matches(row, embeddings, permutations, args.threshold)
        all_choices = [*human_choices, *model_choices]
        rows.append(
            {
                "generator_key": row["generator_key"],
                "generator": row["generator"],
                "generator_display": GENERATOR_LABEL.get(row["generator"], row["generator"]),
                "dataset": row["dataset"],
                "dataset_display": DATASET_LABEL.get(row["dataset"], row["dataset"]),
                "sample_id": row["sample_id"],
                "cosine_sem_0.90_count": int(row["sem_0.90"]),
                "hybrid_match_count": len(matched),
                "hybrid_match_proportion": len(matched) / 3.0,
                "any_formula_like_choice": any(is_formula_like(choice) for choice in all_choices),
                "formula_like_choice_count": sum(is_formula_like(choice) for choice in all_choices),
                "exact_rule_match_count": sum(1 for *_, method in matched if method == "normalized_exact_formula_like"),
                "cosine_rule_match_count": sum(1 for *_, method in matched if method == "cosine_text"),
                "human_choices": human_choices,
                "model_choices": model_choices,
                "matched_human_distractors": [human_choices[human_idx] for _, human_idx, _, _ in matched],
                "matched_model_distractors": [model_choices[model_idx] for model_idx, _, _, _ in matched],
                "matched_pair_similarities": [round(float(score), 6) for _, _, score, _ in matched],
                "matched_pair_methods": [method for _, _, _, method in matched],
            }
        )
    return pd.DataFrame(rows)


def write_summary(items: pd.DataFrame, output_dir: Path) -> Path:
    summary = (
        items.groupby(["dataset", "dataset_display", "generator", "generator_display"], dropna=False)
        .agg(
            n_items=("sample_id", "size"),
            hybrid_avg_match_proportion=("hybrid_match_proportion", "mean"),
            prior_cosine_avg_match_proportion=("cosine_sem_0.90_count", lambda s: (s / 3.0).mean()),
            any_formula_like_item_rate=("any_formula_like_choice", "mean"),
            mean_formula_like_choice_count=("formula_like_choice_count", "mean"),
        )
        .reset_index()
    )
    summary["hybrid_avg_match_percent"] = 100 * summary["hybrid_avg_match_proportion"]
    summary["prior_cosine_avg_match_percent"] = 100 * summary["prior_cosine_avg_match_proportion"]
    summary["delta_vs_prior_percent_points"] = (
        summary["hybrid_avg_match_percent"] - summary["prior_cosine_avg_match_percent"]
    )
    path = output_dir / "hybrid_memorization_summary_by_dataset_model.csv"
    summary.to_csv(path, index=False)
    return path


def write_distribution(items: pd.DataFrame, output_dir: Path) -> Path:
    dist = (
        items.groupby(["dataset", "dataset_display", "generator", "generator_display", "hybrid_match_count"], dropna=False)
        .size()
        .reset_index(name="n_items")
    )
    full_index = pd.MultiIndex.from_product(
        [DATASET_ORDER, GENERATOR_ORDER, [0, 1, 2, 3]],
        names=["dataset", "generator", "hybrid_match_count"],
    )
    dist = dist.set_index(["dataset", "generator", "hybrid_match_count"]).reindex(full_index, fill_value=0).reset_index()
    dist["dataset_display"] = dist["dataset"].map(DATASET_LABEL)
    dist["generator_display"] = dist["generator"].map(GENERATOR_LABEL)
    dist["percent_items"] = dist.groupby(["dataset", "generator"])["n_items"].transform(lambda s: 100 * s / s.sum())
    path = output_dir / "hybrid_match_distribution_by_dataset_model.csv"
    dist.to_csv(path, index=False)
    return path


def _join_lines(values: object) -> str:
    if isinstance(values, list):
        return "\n".join(map(str, values))
    return str(values)


def write_flagged_items(items: pd.DataFrame, output_dir: Path) -> Path:
    flagged = items[items["hybrid_match_count"] > 0].copy()
    for column in [
        "human_choices",
        "model_choices",
        "matched_human_distractors",
        "matched_model_distractors",
        "matched_pair_similarities",
        "matched_pair_methods",
    ]:
        flagged[column] = flagged[column].map(_join_lines)
    path = output_dir / "hybrid_flagged_items_0p90_formula_exact.csv"
    flagged.sort_values(
        ["dataset", "hybrid_match_count", "exact_rule_match_count", "cosine_rule_match_count", "sample_id"],
        ascending=[True, False, False, False, True],
    ).to_csv(path, index=False)
    return path


def write_gpqa_audit(items: pd.DataFrame, output_dir: Path, collected_root: Path) -> Path:
    questions = load_gpqa_question_lookup(collected_root)
    gpqa = items[items["dataset"] == "gpqa"].copy()
    gpqa["question"] = gpqa.apply(lambda row: questions.get((row["generator_key"], row["sample_id"]), {}).get("question", ""), axis=1)
    gpqa["answer"] = gpqa.apply(lambda row: questions.get((row["generator_key"], row["sample_id"]), {}).get("answer", ""), axis=1)

    audit_parts: list[pd.DataFrame] = []
    for generator in GENERATOR_ORDER:
        group = gpqa[gpqa["generator"] == generator].copy()

        hybrid_flagged = group[group["hybrid_match_count"] > 0].copy()
        hybrid_flagged["audit_bucket"] = "hybrid_flagged"
        hybrid_flagged = hybrid_flagged.sort_values(
            ["hybrid_match_count", "exact_rule_match_count", "cosine_rule_match_count", "cosine_sem_0.90_count", "sample_id"],
            ascending=[False, False, False, False, True],
        ).head(5)

        reduced = group[group["cosine_sem_0.90_count"] > group["hybrid_match_count"]].copy()
        reduced["audit_bucket"] = "cosine_flag_removed_or_reduced_by_exact_rule"
        reduced["reduction"] = reduced["cosine_sem_0.90_count"] - reduced["hybrid_match_count"]
        reduced = reduced.sort_values(
            ["reduction", "cosine_sem_0.90_count", "hybrid_match_count", "sample_id"],
            ascending=[False, False, True, True],
        ).head(5)

        audit_parts.extend([hybrid_flagged, reduced])

    audit = pd.concat(audit_parts, ignore_index=True)
    audit["human_distractors"] = audit["human_choices"].map(_join_lines)
    audit["model_distractors"] = audit["model_choices"].map(_join_lines)
    audit["matched_human_distr"] = audit["matched_human_distractors"].map(_join_lines)
    audit["matched_model_distr"] = audit["matched_model_distractors"].map(_join_lines)
    audit["matched_pair_similarities"] = audit["matched_pair_similarities"].map(_join_lines)
    audit["matched_pair_methods"] = audit["matched_pair_methods"].map(_join_lines)
    audit["annotation_label"] = ""
    audit["annotation_notes"] = ""

    columns = [
        "audit_bucket",
        "generator_display",
        "sample_id",
        "cosine_sem_0.90_count",
        "hybrid_match_count",
        "exact_rule_match_count",
        "cosine_rule_match_count",
        "question",
        "answer",
        "human_distractors",
        "model_distractors",
        "matched_human_distr",
        "matched_model_distr",
        "matched_pair_similarities",
        "matched_pair_methods",
        "annotation_label",
        "annotation_notes",
    ]
    path = output_dir / "gpqa_hybrid_manual_audit_30.csv"
    audit[columns].to_csv(path, index=False)
    return path


def print_latex_rows(summary_path: Path) -> None:
    summary = pd.read_csv(summary_path)
    lookup = summary.set_index(["dataset", "generator"])["hybrid_avg_match_proportion"]
    print("\nMain table rows:")
    for dataset in DATASET_ORDER:
        values = [lookup.loc[(dataset, generator)] for generator in GENERATOR_ORDER]
        print(DATASET_LABEL[dataset] + " & " + " & ".join(f"{value:.3f}" for value in values) + r" \\")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap-csv", type=Path, default=ROOT / "analysis" / "tables" / "sub_analysis_3" / "model_human_distractor_overlap_cached.csv")
    parser.add_argument("--embedding-texts", type=Path, default=ROOT / "analysis" / "cache" / "sub_analysis_3_embeddings" / "texts.json")
    parser.add_argument("--embedding-vectors", type=Path, default=ROOT / "analysis" / "cache" / "sub_analysis_3_embeddings" / "embeddings.npy")
    parser.add_argument("--collected-root", type=Path, default=DEFAULT_COLLECTED_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis" / "tables" / "sub_analysis_3")
    parser.add_argument("--threshold", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items = build_item_rows(args)
    summary_path = write_summary(items, args.output_dir)
    distribution_path = write_distribution(items, args.output_dir)
    flagged_path = write_flagged_items(items, args.output_dir)
    audit_path = write_gpqa_audit(items, args.output_dir, args.collected_root)

    print("Wrote:")
    for path in [summary_path, distribution_path, flagged_path, audit_path]:
        print(f"  {path}")
    print_latex_rows(summary_path)


if __name__ == "__main__":
    main()
