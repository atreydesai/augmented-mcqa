"""Final5 experiment runner."""

from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from config import DistractorType
from evaluation.evaluator import build_mcqa_prompt
from models import ModelClient, get_client
from models.local_client import LocalClient
from .config import ExperimentConfig
from .defaults import DEFAULT_EVAL_STOP


CHOICE_LABELS = "ABCDEFGHIJ"
SETTING_IDS = {
    "human_from_scratch",
    "model_from_scratch",
    "augment_human",
    "augment_model",
    "augment_ablation",
}


@dataclass
class EvalResult:
    question_idx: int
    question: str
    gold_answer: str
    gold_answer_letter: str
    gold_index: int
    model_answer: str
    model_prediction: str
    is_correct: bool
    category: str
    prediction_type: Optional[str] = None
    response_text: str = ""
    latency_ms: float = 0.0
    eval_options_randomized: List[str] = field(default_factory=list)
    eval_correct_answer_letter: str = ""
    eval_full_question: str = ""
    eval_model_input: str = ""
    eval_model_output: str = ""
    selected_human_distractors: List[str] = field(default_factory=list)
    selected_model_distractors: List[str] = field(default_factory=list)
    human_option_indices: List[int] = field(default_factory=list)
    model_option_indices: List[int] = field(default_factory=list)


@dataclass
class ExperimentResults:
    config: ExperimentConfig
    results: List[EvalResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    attempted_entries: int = 0
    successful_entries: int = 0
    failed_entries: int = 0
    entry_failures: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        correct = sum(1 for r in self.results if r.is_correct)
        return correct / len(self.results)

    @property
    def behavioral_counts(self) -> Dict[str, int]:
        counts = {"G": 0, "H": 0, "M": 0, "?": 0}
        for r in self.results:
            key = r.prediction_type or "?"
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def accuracy_by_category(self) -> Dict[str, float]:
        by_cat: Dict[str, List[bool]] = {}
        for r in self.results:
            by_cat.setdefault(r.category or "", []).append(r.is_correct)
        return {
            k: (sum(v) / len(v) if v else 0.0)
            for k, v in by_cat.items()
        }

    def _row_payloads(self) -> List[dict[str, Any]]:
        return [
            {
                **asdict(r),
                "options_randomized": r.eval_options_randomized,
                "correct_answer_letter": r.eval_correct_answer_letter,
                "full_question": r.eval_full_question,
                "model_input": r.eval_model_input,
                "model_output": r.eval_model_output,
            }
            for r in self.results
        ]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "summary": {
                "total": len(self.results),
                "correct": sum(1 for r in self.results if r.is_correct),
                "accuracy": self.accuracy,
                "attempted_entries": self.attempted_entries,
                "successful_entries": self.successful_entries,
                "failed_entries": self.failed_entries,
                "entry_failure_count": len(self.entry_failures),
                "behavioral_counts": self.behavioral_counts,
                "accuracy_by_category": self.accuracy_by_category,
            },
            "timing": {
                "start": self.start_time.isoformat() if self.start_time else None,
                "end": self.end_time.isoformat() if self.end_time else None,
            },
            "entry_failures": self.entry_failures,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_summary_dict()
        payload["results"] = self._row_payloads()
        return payload

    def save(self, *, summary_path: Path, rows_path: Path) -> tuple[Path, Path]:
        from datasets import Dataset

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        rows_path.parent.mkdir(parents=True, exist_ok=True)

        if rows_path.exists():
            if rows_path.is_dir():
                shutil.rmtree(rows_path)
            else:
                rows_path.unlink()

        rows = self._row_payloads()
        Dataset.from_list(rows).save_to_disk(str(rows_path))
        summary_path.write_text(json.dumps(self.to_summary_dict(), indent=2), encoding="utf-8")
        return summary_path, rows_path


def determine_prediction_type(
    predicted_idx: int,
    gold_idx: int,
    human_indices: List[int],
    model_indices: List[int],
) -> str:
    if predicted_idx == gold_idx:
        return "G"
    if predicted_idx in human_indices:
        return "H"
    if predicted_idx in model_indices:
        return "M"
    return "?"


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        client: Optional[ModelClient] = None,
        dataset_cache: Optional[Dict[tuple[Any, ...], Any]] = None,
    ):
        self.config = config
        self._client = client
        self._dataset_cache = dataset_cache

    def _get_client(self) -> ModelClient:
        if self._client is None:
            kwargs: Dict[str, Any] = {}
            if self.config.reasoning_effort:
                kwargs["reasoning_effort"] = self.config.reasoning_effort
            if self.config.thinking_level:
                kwargs["thinking_level"] = self.config.thinking_level
            if self.config.model_name == "local" or "/" in self.config.model_name:
                if self.config.vllm_max_num_batched_tokens is not None:
                    kwargs["max_num_batched_tokens"] = self.config.vllm_max_num_batched_tokens
                if self.config.vllm_max_num_seqs is not None:
                    kwargs["max_num_seqs"] = self.config.vllm_max_num_seqs
                if self.config.vllm_enable_chunked_prefill is not None:
                    kwargs["enable_chunked_prefill"] = self.config.vllm_enable_chunked_prefill
            self._client = get_client(self.config.model_name, **kwargs)
        return self._client

    def _load_data(self):
        from datasets import load_from_disk

        dataset_key = str(self.config.dataset_path)
        if self._dataset_cache is not None and (dataset_key,) in self._dataset_cache:
            dataset = self._dataset_cache[(dataset_key,)]
        else:
            dataset = load_from_disk(dataset_key)
            if self._dataset_cache is not None:
                self._dataset_cache[(dataset_key,)] = dataset

        if self.config.dataset_type_filter:
            dt = self.config.dataset_type_filter
            if hasattr(dataset, "keys") and dt in dataset:
                return dataset[dt]
            if hasattr(dataset, "column_names") and "dataset_type" in dataset.column_names:
                filtered = dataset.filter(lambda x: x["dataset_type"] == dt)
                if len(filtered) == 0:
                    raise ValueError(f"dataset_type_filter='{dt}' matched zero rows")
                return filtered
            raise ValueError(f"dataset_type_filter='{dt}' not found in dataset")

        if hasattr(dataset, "keys"):
            if len(dataset.keys()) == 1:
                return next(iter(dataset.values()))
            raise ValueError("dataset has multiple splits; set dataset_type_filter")

        return dataset

    def _select_entry_shard(self, entries: List[Any]) -> List[Any]:
        if self.config.entry_shards <= 1:
            return entries
        if self.config.entry_shard_strategy == "modulo":
            return [
                e for idx, e in enumerate(entries)
                if idx % self.config.entry_shards == self.config.entry_shard_index
            ]

        total = len(entries)
        shards = self.config.entry_shards
        shard_idx = self.config.entry_shard_index
        base = total // shards
        remainder = total % shards
        start = shard_idx * base + min(shard_idx, remainder)
        size = base + (1 if shard_idx < remainder else 0)
        end = start + size
        return entries[start:end]

    def _prepare_entry(self, entry: Dict[str, Any], idx: int) -> Dict[str, Any]:
        setting = self.config.setting_id
        if setting not in SETTING_IDS:
            raise ValueError(f"Unsupported setting_id={setting}")

        gold_answer = ""
        if isinstance(entry.get("choices_answer"), list) and entry["choices_answer"]:
            gold_answer = str(entry["choices_answer"][0]).strip()
        if not gold_answer:
            gold_answer = str(entry.get("answer", "")).strip()
        if not gold_answer:
            raise ValueError("Missing gold answer")

        selected_human: List[str] = []
        selected_model: List[str] = []

        if setting == "human_from_scratch":
            selected_human = list(entry.get("human_from_scratch") or [])[:3]
        elif setting == "model_from_scratch":
            selected_model = list(entry.get("model_from_scratch") or [])[:3]
        elif setting == "augment_human":
            selected_human = list(entry.get("human_from_scratch") or [])[:3]
            selected_model = list(entry.get("augment_human") or [])[:6]
        elif setting == "augment_model":
            selected_model = list(entry.get("augment_model") or [])[:9]
        elif setting == "augment_ablation":
            selected_model = list(entry.get("augment_ablation") or [])[:9]

        if len(selected_human) < self.config.num_human:
            raise ValueError(
                f"Insufficient human distractors for {setting}: expected {self.config.num_human}, got {len(selected_human)}"
            )
        if len(selected_model) < self.config.num_model:
            raise ValueError(
                f"Insufficient model distractors for {setting}: expected {self.config.num_model}, got {len(selected_model)}"
            )

        selected_human = selected_human[: self.config.num_human]
        selected_model = selected_model[: self.config.num_model]

        options = [gold_answer] + selected_human + selected_model
        order = list(range(len(options)))
        rng = random.Random(
            self.config.seed
            + idx * 1009
            + self.config.num_human * 97
            + self.config.num_model * 89
        )
        rng.shuffle(order)

        shuffled = [options[i] for i in order]
        gold_idx = order.index(0)
        human_indices = [order.index(i) for i in range(1, 1 + len(selected_human))]
        model_indices = [
            order.index(i)
            for i in range(1 + len(selected_human), 1 + len(selected_human) + len(selected_model))
        ]

        return {
            "question": str(entry.get("question", "")),
            "options": shuffled,
            "gold_idx": gold_idx,
            "gold_answer": gold_answer,
            "human_indices": human_indices,
            "model_indices": model_indices,
            "selected_human": selected_human,
            "selected_model": selected_model,
            "category": str(entry.get("category", "")),
        }

    @staticmethod
    def _build_eval_full_question(question: str, options: List[str]) -> str:
        lines = [f"{CHOICE_LABELS[i]}: {opt}" for i, opt in enumerate(options)]
        return f"Question: {question}\n" + "\n".join(lines)

    def _partial_root(self) -> Path:
        if self.config.entry_shards <= 1:
            return self.config.output_dir
        return (
            self.config.output_dir
            / "_partials"
            / f"entry_shard_{self.config.entry_shard_index}_of_{self.config.entry_shards}"
        )

    def _summary_path(self) -> Path:
        return self._partial_root() / "summary.json"

    def _rows_path(self) -> Path:
        return self._partial_root() / "rows"

    def run(self) -> ExperimentResults:
        results = ExperimentResults(config=self.config)
        results.start_time = datetime.now()

        dataset = self._load_data()
        entries = list(enumerate(dataset))
        if self.config.limit is not None:
            entries = entries[: self.config.limit]
        entries = self._select_entry_shard(entries)

        client = self._get_client()

        generate_kwargs: Dict[str, Any] = {"max_tokens": self.config.max_tokens}
        if self.config.temperature is not None:
            generate_kwargs["temperature"] = self.config.temperature
        if isinstance(client, LocalClient):
            stops = self.config.stop if self.config.stop is not None else DEFAULT_EVAL_STOP
            if stops:
                generate_kwargs["stop"] = stops

        batch_size = max(1, int(self.config.inference_batch_size))
        pending: List[Dict[str, Any]] = []

        def flush_pending() -> None:
            nonlocal pending
            if not pending:
                return

            prompts = [p["prompt"] for p in pending]
            start = time.perf_counter()
            try:
                responses = client.generate_batch(prompts, **generate_kwargs)
            except Exception as exc:  # noqa: BLE001
                for item in pending:
                    results.failed_entries += 1
                    results.attempted_entries += 1
                    results.entry_failures.append(
                        {
                            "question_idx": item["idx"],
                            "stage": "generate",
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        }
                    )
                pending = []
                return

            latency_ms = ((time.perf_counter() - start) * 1000.0) / max(1, len(pending))
            for item, response in zip(pending, responses):
                idx = item["idx"]
                prepared = item["prepared"]
                prompt = item["prompt"]
                eval_full_question = item["eval_full_question"]

                try:
                    pred = response.extract_answer()
                    gold_letter = CHOICE_LABELS[prepared["gold_idx"]]
                    is_correct = pred == gold_letter
                    ptype = None
                    if pred in CHOICE_LABELS:
                        predicted_idx = CHOICE_LABELS.index(pred)
                        ptype = determine_prediction_type(
                            predicted_idx,
                            prepared["gold_idx"],
                            prepared["human_indices"],
                            prepared["model_indices"],
                        )

                    res = EvalResult(
                        question_idx=idx,
                        question=prepared["question"],
                        gold_answer=prepared["gold_answer"],
                        gold_answer_letter=gold_letter,
                        gold_index=prepared["gold_idx"],
                        model_answer=pred,
                        model_prediction=pred,
                        is_correct=is_correct,
                        category=prepared["category"],
                        prediction_type=ptype,
                        response_text=response.text,
                        latency_ms=latency_ms,
                        eval_options_randomized=prepared["options"],
                        eval_correct_answer_letter=gold_letter,
                        eval_full_question=eval_full_question,
                        eval_model_input=prompt,
                        eval_model_output=response.text,
                        selected_human_distractors=prepared["selected_human"],
                        selected_model_distractors=prepared["selected_model"],
                        human_option_indices=prepared["human_indices"],
                        model_option_indices=prepared["model_indices"],
                    )
                    results.results.append(res)
                    results.successful_entries += 1
                except Exception as exc:  # noqa: BLE001
                    results.failed_entries += 1
                    results.entry_failures.append(
                        {
                            "question_idx": idx,
                            "stage": "score",
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        }
                    )
                finally:
                    results.attempted_entries += 1

            pending = []

        iterator = tqdm(entries, desc="Evaluating", total=len(entries))
        for idx, entry in iterator:
            try:
                prepared = self._prepare_entry(entry, idx)
                prompt = build_mcqa_prompt(
                    prepared["question"],
                    prepared["options"],
                    choices_only=self.config.choices_only,
                )
                pending.append(
                    {
                        "idx": idx,
                        "prepared": prepared,
                        "prompt": prompt,
                        "eval_full_question": self._build_eval_full_question(
                            prepared["question"], prepared["options"]
                        ),
                    }
                )
                if len(pending) >= batch_size:
                    flush_pending()
            except Exception as exc:  # noqa: BLE001
                results.failed_entries += 1
                results.attempted_entries += 1
                results.entry_failures.append(
                    {
                        "question_idx": idx,
                        "stage": "prepare",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                )

        flush_pending()

        results.end_time = datetime.now()
        summary_path = self._summary_path()
        rows_path = self._rows_path()
        results.save(summary_path=summary_path, rows_path=rows_path)
        print(f"Saved summary: {summary_path}")
        print(f"Saved rows: {rows_path}")
        return results


def run_experiment(
    config: ExperimentConfig,
    client: Optional[ModelClient] = None,
    dataset_cache: Optional[Dict[tuple[Any, ...], Any]] = None,
) -> ExperimentResults:
    runner = ExperimentRunner(config, client=client, dataset_cache=dataset_cache)
    return runner.run()


def run_batch(configs: List[ExperimentConfig]) -> List[ExperimentResults]:
    results = []
    for i, config in enumerate(configs, start=1):
        print(f"[{i}/{len(configs)}] Running: {config.name}")
        results.append(run_experiment(config))
    return results
