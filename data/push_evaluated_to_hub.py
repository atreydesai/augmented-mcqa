from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Sequence, Value, concatenate_datasets, load_from_disk
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASETS_DIR, HF_TOKEN
from utils.constants import MODE_CHOICES, SETTING_NAMES

PROVIDER_ROOTS = {
    "gemini": DATASETS_DIR / "evaluated" / "gemini_from_scratch_testing" / "google_gemini-3.1-pro-preview",
    "gpt": DATASETS_DIR / "evaluated" / "gen_gpt52_v2" / "openai_gpt-5.2-2025-12-11",
    "together": DATASETS_DIR / "evaluated" / "together_from_scratch_testing" / "together_Qwen_Qwen3.5-397B-A17B",
}


def iter_eval_model_roots(provider_root: Path) -> list[Path]:
    return sorted(path for path in provider_root.iterdir() if (path / "evaluated_manifest.json").is_file())


def load_setting_split(eval_model_root: Path, setting: str, mode: str) -> Dataset | None:
    parts = [
        load_from_disk(str(path))
        for path in sorted(eval_model_root.glob(f"*/{setting}/{mode}"))
        if path.is_dir()
    ]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def normalized_features(dataset: Dataset) -> Features:
    features = dataset.features.copy()
    if "question_id" in features:
        features["question_id"] = Value("int64")
    if "traces" in features:
        features["traces"] = Value("string")
    for column in (
        "options",
        "choices_human",
        "human_distractors",
        "model_distractors",
        "distractors",
        "options_randomized",
    ):
        if column in features:
            features[column] = Sequence(Value("string"))
    return Features(features)


def normalize_split(dataset: Dataset, features: Features | None = None) -> Dataset:
    if "traces" in dataset.column_names:
        dataset = dataset.remove_columns("traces").add_column("traces", [""] * len(dataset))
    features = features or normalized_features(dataset)
    return dataset.cast(features)


def build_subset_dataset(eval_model_root: Path, mode: str) -> DatasetDict:
    splits = {
        setting: split
        for setting in SETTING_NAMES
        if (split := load_setting_split(eval_model_root, setting, mode)) is not None
    }
    if not splits:
        raise ValueError(f"No evaluated splits found under {eval_model_root} for mode={mode!r}")
    features = normalized_features(normalize_split(next(iter(splits.values()))))
    return DatasetDict({name: normalize_split(split, features) for name, split in splits.items()})


def parse_repo_assignments(values: list[str]) -> dict[tuple[str, str], str]:
    assignments: dict[tuple[str, str], str] = {}
    for value in values:
        key, sep, repo_id = value.partition("=")
        provider, slash, mode = key.partition("/")
        if not sep or not slash or not repo_id:
            raise ValueError(f"Invalid --repo value {value!r}; expected provider/mode=org_or_user/repo")
        if provider not in PROVIDER_ROOTS:
            raise ValueError(f"Unknown provider {provider!r}; expected one of {sorted(PROVIDER_ROOTS)}")
        if mode not in MODE_CHOICES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {MODE_CHOICES}")
        assignments[(provider, mode)] = repo_id
    return assignments


def push_provider_mode(provider: str, mode: str, repo_id: str, *, dry_run: bool) -> None:
    print(f"\n[{provider}/{mode}] -> {repo_id}")
    for eval_model_root in iter_eval_model_roots(PROVIDER_ROOTS[provider]):
        subset = eval_model_root.name
        dataset = build_subset_dataset(eval_model_root, mode)
        split_summary = ", ".join(f"{split}={dataset[split].num_rows}" for split in dataset)
        print(f"  {subset}: {split_summary}")
        if dry_run:
            continue
        push_kwargs = {"config_name": subset}
        if HF_TOKEN:
            push_kwargs["token"] = HF_TOKEN
        dataset.push_to_hub(repo_id, **push_kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Push evaluated datasets to the Hugging Face Hub as six repos: "
            "provider x mode, with eval models as configs and settings as splits."
        )
    )
    parser.add_argument(
        "--repo",
        action="append",
        required=True,
        metavar="PROVIDER/MODE=REPO_ID",
        help=(
            "Repeat for each target repo, for example "
            "gemini/full_question=user/repo and gemini/choices_only=user/repo."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned uploads without pushing.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    assignments = parse_repo_assignments(args.repo)

    try:
        for provider, mode in sorted(assignments):
            push_provider_mode(provider, mode, assignments[(provider, mode)], dry_run=args.dry_run)
    except (HfHubHTTPError, RepositoryNotFoundError) as exc:
        message = str(exc)
        if "401" in message or "Invalid username or password" in message:
            raise RuntimeError(
                "Hugging Face authentication failed. Set HF_TOKEN to a valid write-scoped token and rerun."
            ) from exc
        raise


if __name__ == "__main__":
    main()
