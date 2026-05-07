#!/usr/bin/env python3
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_from_disk


ROOT = Path(__file__).resolve().parents[2]
STRICT_ROOT = ROOT / "datasets" / "augmented_filtered" / "strict"
ALL_PATH = ROOT / "results" / "atrey_writing_flaw_rows_all.jsonl"
TRUNC_PATH = ROOT / "results" / "atrey_writing_flaw_rows_truncation.jsonl"
OUT_PATH = ROOT / "results" / "atrey_writing_flaw_rows_strict.jsonl"
MISSING_PATH = ROOT / "results" / "atrey_writing_flaw_rows_strict_missing.json"
REPORT_PATH = ROOT / "results" / "atrey_writing_flaw_rows_strict_report.json"

RUN_TO_BENCH_MODEL = {
    "gemini": "gemini-3.1-pro-preview",
    "gpt": "gpt-5.2-2025-12-11",
    "qwen": "Qwen3.5-397B-A17B",
}

TRUNC_MODEL_TO_RUN = {
    "google_gemini-3.1-pro-preview": "gemini",
    "together_Qwen_Qwen3.5-397B-A17B": "qwen",
}

DATASETS = ["arc_challenge", "gpqa", "mmlu_pro"]
SETTINGS = [
    "augment_ablation",
    "augment_human",
    "augment_model",
    "human_from_scratch",
    "model_from_scratch",
]


def norm_text(value):
    return re.sub(r"\s+", " ", value or "").strip()


def choice_key(choices):
    return tuple(norm_text(choice) for choice in choices or [])


def read_jsonl(path):
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                row["_line_no"] = line_no
                yield row


def load_strict_rows():
    strict = {}
    manifests = {}
    for run in RUN_TO_BENCH_MODEL:
        manifest = json.loads((STRICT_ROOT / run / "augmented_manifest.json").read_text())
        manifests[run] = manifest
        for dataset in DATASETS:
            for setting in SETTINGS:
                ds = load_from_disk(str(STRICT_ROOT / run / dataset / setting))
                strict[(run, dataset, setting)] = {row["sample_id"]: row for row in ds}
    return strict, manifests


def load_source_rows(manifests):
    source = {}
    for run, manifest in manifests.items():
        source_root = ROOT / manifest["source_augmented_store"]
        for dataset in DATASETS:
            for setting in SETTINGS:
                ds = load_from_disk(str(source_root / dataset / setting))
                source[(run, dataset, setting)] = list(ds)
    return source


def attach_sample_ids(all_rows_by_combo, source):
    attached = {}
    unmatched = []
    ambiguous = []

    for run, model in RUN_TO_BENCH_MODEL.items():
        for dataset in DATASETS:
            for setting in SETTINGS:
                combo = (run, dataset, setting)
                bench_rows = all_rows_by_combo[(model, dataset, setting)]
                source_rows = source[combo]
                paired = []

                positional = len(bench_rows) == len(source_rows) and all(
                    choice_key(bench_rows[i]["choices"]) == choice_key(source_rows[i]["options_randomized"])
                    for i in range(len(bench_rows))
                )

                if positional:
                    for bench_row, source_row in zip(bench_rows, source_rows):
                        paired.append((source_row["sample_id"], bench_row))
                else:
                    by_choices = defaultdict(list)
                    by_question_choices = defaultdict(list)
                    for source_row in source_rows:
                        ck = choice_key(source_row["options_randomized"])
                        qck = (norm_text(source_row["question"]), ck)
                        by_choices[ck].append(source_row)
                        by_question_choices[qck].append(source_row)

                    used = Counter()
                    for bench_row in bench_rows:
                        ck = choice_key(bench_row["choices"])
                        qck = (norm_text(bench_row["question"]), ck)
                        candidates = by_question_choices.get(qck) or by_choices.get(ck) or []
                        unused = [row for row in candidates if used[row["sample_id"]] == 0]
                        candidates = unused or candidates
                        if not candidates:
                            unmatched.append(
                                {
                                    "run": run,
                                    "model": model,
                                    "dataset": dataset,
                                    "setting": setting,
                                    "line_no": bench_row["_line_no"],
                                    "question": bench_row.get("question"),
                                    "choices": bench_row.get("choices"),
                                }
                            )
                            continue
                        if len(candidates) > 1:
                            ambiguous.append(
                                {
                                    "run": run,
                                    "model": model,
                                    "dataset": dataset,
                                    "setting": setting,
                                    "line_no": bench_row["_line_no"],
                                    "sample_ids": [row["sample_id"] for row in candidates],
                                }
                            )
                        source_row = candidates[0]
                        used[source_row["sample_id"]] += 1
                        paired.append((source_row["sample_id"], bench_row))

                attached[combo] = paired

    return attached, unmatched, ambiguous


def load_truncation_replacements(strict):
    replacements = {}
    unmatched = []
    ambiguous = []

    for row in read_jsonl(TRUNC_PATH):
        run = TRUNC_MODEL_TO_RUN[row["model"]]
        combo = (run, row["dataset"], row["config"])
        ck = choice_key(row["choices"])
        qck = (norm_text(row["question"]), ck)
        candidates = []
        for strict_row in strict[combo].values():
            strict_ck = choice_key(strict_row["options_randomized"])
            if (norm_text(strict_row["question"]), strict_ck) == qck:
                candidates.append(strict_row)
        if not candidates:
            for strict_row in strict[combo].values():
                if choice_key(strict_row["options_randomized"]) == ck:
                    candidates.append(strict_row)

        if not candidates:
            unmatched.append(
                {
                    "run": run,
                    "dataset": row["dataset"],
                    "setting": row["config"],
                    "line_no": row["_line_no"],
                    "question": row.get("question"),
                    "choices": row.get("choices"),
                }
            )
            continue
        if len(candidates) > 1:
            ambiguous.append(
                {
                    "run": run,
                    "dataset": row["dataset"],
                    "setting": row["config"],
                    "line_no": row["_line_no"],
                    "sample_ids": [candidate["sample_id"] for candidate in candidates],
                }
            )
        sample_id = candidates[0]["sample_id"]
        replacements[(run, row["dataset"], row["config"], sample_id)] = row["writing_flaw"]

    return replacements, unmatched, ambiguous


def has_bad_ellipsis(text):
    text = (text or "").rstrip()
    return text.endswith("...") or text.endswith("…")


def has_key_removed_artifact(choices):
    joined = "".join(str(choice) for choice in choices or [])
    return "Key removed from summary" in joined


def strict_row_has_trailing_ellipsis(strict_row):
    return has_bad_ellipsis(strict_row["question"])


def main():
    strict, manifests = load_strict_rows()
    source = load_source_rows(manifests)

    all_rows_by_combo = defaultdict(list)
    for row in read_jsonl(ALL_PATH):
        all_rows_by_combo[(row["model"], row["dataset"], row["config"])].append(row)

    attached, unmatched_all, ambiguous_all = attach_sample_ids(all_rows_by_combo, source)
    replacements, unmatched_trunc, ambiguous_trunc = load_truncation_replacements(strict)

    missing = []
    output_rows = []
    used_replacements = 0

    for run, model in RUN_TO_BENCH_MODEL.items():
        for dataset in DATASETS:
            for setting in SETTINGS:
                combo = (run, dataset, setting)
                strict_rows = strict[combo]
                rows_by_sample_id = {}
                for sample_id, bench_row in attached[combo]:
                    if sample_id in strict_rows:
                        rows_by_sample_id[sample_id] = bench_row

                for sample_id, strict_row in strict_rows.items():
                    bench_row = rows_by_sample_id.get(sample_id)
                    if bench_row is None:
                        missing.append(
                            {
                                "run": run,
                                "model": model,
                                "dataset": dataset,
                                "setting": setting,
                                "sample_id": sample_id,
                                "id": strict_row.get("id"),
                            }
                        )
                        continue

                    writing_flaw = replacements.get((run, dataset, setting, sample_id), bench_row["writing_flaw"])
                    if (run, dataset, setting, sample_id) in replacements:
                        used_replacements += 1

                    output_rows.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "config": setting,
                            "sample_id": sample_id,
                            "id": strict_row.get("id"),
                            "question": strict_row["question"],
                            "choices": list(strict_row["options_randomized"]),
                            "answer": strict_row["correct_answer_letter"],
                            "writing_flaw": writing_flaw,
                        }
                    )

    bad_ellipsis = [
        {
            "model": row["model"],
            "dataset": row["dataset"],
            "config": row["config"],
            "sample_id": row["sample_id"],
            "question": row["question"],
        }
        for row in output_rows
        if has_bad_ellipsis(row["question"])
    ]
    bad_key_artifacts = [
        {
            "model": row["model"],
            "dataset": row["dataset"],
            "config": row["config"],
            "sample_id": row["sample_id"],
            "choices": row["choices"],
        }
        for row in output_rows
        if has_key_removed_artifact(row["choices"])
    ]

    with OUT_PATH.open("w") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    MISSING_PATH.write_text(json.dumps(missing, indent=2, ensure_ascii=False) + "\n")

    counts = Counter((row["model"], row["dataset"], row["config"]) for row in output_rows)
    report = {
        "output_path": str(OUT_PATH.relative_to(ROOT)),
        "missing_path": str(MISSING_PATH.relative_to(ROOT)),
        "rows_written": len(output_rows),
        "strict_rows_expected": sum(len(rows) for rows in strict.values()),
        "strict_rows_excluded_for_trailing_ellipsis": 0,
        "truncation_replacements_used": used_replacements,
        "missing_strict_questions": len(missing),
        "unmatched_all_rows": len(unmatched_all),
        "ambiguous_all_rows": len(ambiguous_all),
        "unmatched_truncation_rows": len(unmatched_trunc),
        "ambiguous_truncation_rows": len(ambiguous_trunc),
        "trailing_ellipsis_questions": len(bad_ellipsis),
        "key_removed_choice_artifacts": len(bad_key_artifacts),
        "counts_by_model_dataset_config": {
            "|".join(key): value for key, value in sorted(counts.items())
        },
        "unmatched_all_examples": unmatched_all[:20],
        "ambiguous_all_examples": ambiguous_all[:20],
        "unmatched_truncation_examples": unmatched_trunc[:20],
        "ambiguous_truncation_examples": ambiguous_trunc[:20],
        "excluded_trailing_ellipsis_examples": [],
        "trailing_ellipsis_examples": bad_ellipsis[:20],
        "key_removed_choice_artifact_examples": bad_key_artifacts[:20],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
