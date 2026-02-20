"""
Experiment runner for Augmented MCQA.

Orchestrates evaluation runs using ExperimentConfig.
"""

import json
import os
import random
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from tqdm import tqdm

from .config import ExperimentConfig
from models import get_client, ModelClient
from models.local_client import LocalClient
from config import DistractorType
from evaluation.evaluator import build_mcqa_prompt
from .defaults import DEFAULT_EVAL_STOP


CHOICE_LABELS = "ABCDEFGHIJ"


@dataclass
class EvalResult:
    """Result for a single evaluation."""
    question_idx: int
    question: str
    gold_answer: str
    gold_answer_letter: str
    gold_index: int
    model_answer: str
    model_prediction: str
    is_correct: bool
    category: str
    
    # For behavioral analysis
    prediction_type: Optional[str] = None  # "G" (gold), "H" (human), "M" (model)
    
    # Metadata
    response_text: str = ""
    latency_ms: float = 0.0

    # Eval trace fields
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
    """Results from an experiment run."""
    config: ExperimentConfig
    results: List[EvalResult] = field(default_factory=list)
    
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Robustness / resume bookkeeping
    attempted_entries: int = 0
    successful_entries: int = 0
    failed_entries: int = 0
    entry_failures: List[Dict[str, Any]] = field(default_factory=list)
    resumed_from_checkpoint: bool = False
    resumed_checkpoint_path: Optional[str] = None
    resumed_next_index: int = 0
    stage_timing_seconds: Dict[str, float] = field(default_factory=dict)
    
    # Aggregates (computed on demand)
    _accuracy: Optional[float] = None
    _behavioral_counts: Optional[Dict[str, int]] = None
    
    @property
    def accuracy(self) -> float:
        """Compute overall accuracy."""
        if self._accuracy is None:
            if not self.results:
                return 0.0
            correct = sum(1 for r in self.results if r.is_correct)
            self._accuracy = correct / len(self.results)
        return self._accuracy
    
    @property
    def behavioral_counts(self) -> Dict[str, int]:
        """Count G/H/M prediction types."""
        if self._behavioral_counts is None:
            counts = {"G": 0, "H": 0, "M": 0, "?": 0}
            for r in self.results:
                if r.prediction_type:
                    counts[r.prediction_type] = counts.get(r.prediction_type, 0) + 1
            self._behavioral_counts = counts
        return self._behavioral_counts
    
    @property
    def accuracy_by_category(self) -> Dict[str, float]:
        """Compute accuracy broken down by category."""
        by_cat: Dict[str, List[bool]] = {}
        for r in self.results:
            if r.category not in by_cat:
                by_cat[r.category] = []
            by_cat[r.category].append(r.is_correct)
        
        return {
            cat: sum(results) / len(results)
            for cat, results in by_cat.items()
        }
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "config": self.config.to_dict(),
            "summary": {
                "total": len(self.results),
                "correct": sum(1 for r in self.results if r.is_correct),
                "accuracy": self.accuracy,
                "accuracy_success_only": self.accuracy,
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
                "duration_seconds": (self.end_time - self.start_time).total_seconds()
                    if self.start_time and self.end_time else None,
                "stage_seconds": self.stage_timing_seconds,
            },
            "resume": {
                "resumed_from_checkpoint": self.resumed_from_checkpoint,
                "resumed_checkpoint_path": self.resumed_checkpoint_path,
                "resumed_next_index": self.resumed_next_index,
            },
            "entry_failures": self.entry_failures,
            "results": [
                {
                    "question_idx": r.question_idx,
                    "question": r.question[:100] + "..." if len(r.question) > 100 else r.question,
                    "gold_answer": r.gold_answer,
                    "gold_answer_letter": r.gold_answer_letter,
                    "gold_index": r.gold_index,
                    "model_answer": r.model_answer,
                    "model_prediction": r.model_prediction,
                    "is_correct": r.is_correct,
                    "category": r.category,
                    "prediction_type": r.prediction_type,
                    "latency_ms": r.latency_ms,
                    "eval_options_randomized": r.eval_options_randomized,
                    "eval_correct_answer_letter": r.eval_correct_answer_letter,
                    "eval_full_question": r.eval_full_question,
                    "eval_model_input": r.eval_model_input,
                    "eval_model_output": r.eval_model_output,
                    # Aliases matching generation-style naming patterns.
                    "options_randomized": r.eval_options_randomized,
                    "correct_answer_letter": r.eval_correct_answer_letter,
                    "full_question": r.eval_full_question,
                    "model_input": r.eval_model_input,
                    "model_output": r.eval_model_output,
                    "selected_human_distractors": r.selected_human_distractors,
                    "selected_model_distractors": r.selected_model_distractors,
                    "human_option_indices": r.human_option_indices,
                    "model_option_indices": r.model_option_indices,
                }
                for r in self.results
            ],
        }
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save results to JSON file."""
        if path is None:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.config.output_dir / "results.json"
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return path





def determine_prediction_type(
    predicted_idx: int,
    gold_idx: int,
    human_indices: List[int],
    model_indices: List[int],
) -> str:
    """
    Determine behavioral prediction type (G/H/M).
    
    Args:
        predicted_idx: Index of predicted answer
        gold_idx: Index of gold answer
        human_indices: Indices of human distractors
        model_indices: Indices of model distractors
        
    Returns:
        "G" for gold, "H" for human distractor, "M" for model distractor, "?" for unknown
    """
    if predicted_idx == gold_idx:
        return "G"
    elif predicted_idx in human_indices:
        return "H"
    elif predicted_idx in model_indices:
        return "M"
    else:
        return "?"


class ExperimentRunner:
    """
    Runs experiments based on ExperimentConfig.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        client: Optional[ModelClient] = None,
        dataset_cache: Optional[Dict[tuple[Any, ...], Any]] = None,
    ):
        """
        Initialize runner with configuration.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self._client: Optional[ModelClient] = client
        self._dataset_cache = dataset_cache
        self._adapter: Optional[Any] = None
        self._timing_events: List[Dict[str, Any]] = []
        self._timing_log_paths = self._resolve_timing_log_paths()

    def _resolve_timing_log_paths(self) -> List[Path]:
        paths = [self.config.output_dir / "timing_events.jsonl"]
        mirror_dir = os.getenv("EVAL_TIMING_LOG_DIR", "").strip()
        if mirror_dir:
            safe_name = self.config.name.replace("/", "_")
            paths.append(Path(mirror_dir) / f"{safe_name}_timing_events.jsonl")
        return paths

    def _record_timing(
        self,
        results: ExperimentResults,
        stage: str,
        duration_s: float,
        *,
        count: int = 1,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        duration_s = max(0.0, float(duration_s))
        results.stage_timing_seconds[stage] = results.stage_timing_seconds.get(stage, 0.0) + duration_s
        payload: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "config_name": self.config.name,
            "config_id": self.config.config_id,
            "stage": stage,
            "duration_s": duration_s,
            "count": count,
        }
        if meta:
            payload.update(meta)
        self._timing_events.append(payload)

    def _flush_timing_events(self) -> None:
        if not self._timing_events:
            return
        for path in self._timing_log_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                for event in self._timing_events:
                    f.write(json.dumps(event) + "\n")
        self._timing_events = []
    
    def _get_client(self) -> ModelClient:
        """Get or create model client."""
        if self._client is None:
            kwargs = {}
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
    
    def _load_data(self) -> Any:
        """Load and optionally filter dataset by dataset_type."""
        if self._adapter is not None:
            return self._adapter

        workpack_path = (
            Path(self.config.workpack_path)
            if self.config.workpack_path is not None
            else None
        )
        if workpack_path is not None:
            workpack_key = ("__workpack__", str(workpack_path), self.config.workpack_format)
            if self._dataset_cache is not None and workpack_key in self._dataset_cache:
                self._adapter = self._dataset_cache[workpack_key]
                return self._adapter

            if not workpack_path.exists():
                raise FileNotFoundError(f"Configured workpack_path does not exist: {workpack_path}")

            if self.config.workpack_format == "parquet":
                from datasets import load_dataset
                data = load_dataset("parquet", data_files=str(workpack_path), split="train")
            elif self.config.workpack_format == "arrow":
                from datasets import load_from_disk
                data = load_from_disk(str(workpack_path))
            elif self.config.workpack_format == "none":
                data = None
            else:
                raise ValueError(
                    f"Unsupported workpack_format={self.config.workpack_format} "
                    "expected one of none|parquet|arrow"
                )

            if data is not None:
                self._adapter = data
                if self._dataset_cache is not None:
                    self._dataset_cache[workpack_key] = data
                return self._adapter

        dataset_path_key = str(self.config.dataset_path)
        dataset_type_key = self.config.dataset_type_filter or "__all__"
        adapter_cache_key = (dataset_path_key, dataset_type_key)
        raw_cache_key = (dataset_path_key, "__raw__")

        if self._dataset_cache is not None and adapter_cache_key in self._dataset_cache:
            self._adapter = self._dataset_cache[adapter_cache_key]
            return self._adapter

        from datasets import load_from_disk
        if self._dataset_cache is not None and raw_cache_key in self._dataset_cache:
            dataset = self._dataset_cache[raw_cache_key]
        else:
            dataset = load_from_disk(str(self.config.dataset_path))
            if self._dataset_cache is not None:
                self._dataset_cache[raw_cache_key] = dataset

        # If dataset_type_filter is specified, try to use it as a split name first
        # (unified datasets are DatasetDicts with splits like "mmlu_pro", "arc_easy", etc.)
        if self.config.dataset_type_filter:
            if hasattr(dataset, "keys") and self.config.dataset_type_filter in dataset:
                data = dataset[self.config.dataset_type_filter]
            elif hasattr(dataset, "column_names") and "dataset_type" in dataset.column_names:
                # Flat dataset with a dataset_type column
                dt_filter = self.config.dataset_type_filter
                data = dataset.filter(lambda x: x["dataset_type"] == dt_filter)
                if len(data) == 0:
                    raise ValueError(
                        f"dataset_type_filter='{dt_filter}' matched zero rows in {self.config.dataset_path}"
                    )
            else:
                available_splits = list(dataset.keys()) if hasattr(dataset, "keys") else []
                raise ValueError(
                    f"dataset_type_filter='{self.config.dataset_type_filter}' not found in {self.config.dataset_path}. "
                    f"Available splits: {available_splits}"
                )
        else:
            # No filter: require an unambiguous split selection.
            if hasattr(dataset, "keys"):
                if "test" in dataset:
                    data = dataset["test"]
                elif len(dataset.keys()) == 1:
                    data = next(iter(dataset.values()))
                else:
                    raise ValueError(
                        f"Dataset has multiple splits {list(dataset.keys())}; "
                        "set dataset_type_filter explicitly."
                    )
            else:
                data = dataset

        self._adapter = data
        if self._dataset_cache is not None:
            self._dataset_cache[adapter_cache_key] = data
        return self._adapter

    def _branching_model_column(self) -> Optional[str]:
        """Return branching-specific model column for human-prefix mode."""
        if self.config.branching_mode != "human_prefix":
            return None

        if self.config.num_human <= 0:
            return DistractorType.COND_MODEL_Q_A_SCRATCH.value

        column_map = {
            1: "cond_model_q_a_dhuman_h1",
            2: "cond_model_q_a_dhuman_h2",
            3: "cond_model_q_a_dhuman_h3",
        }
        return column_map.get(self.config.num_human)

    @staticmethod
    def _require_distractor_column(entry: Dict[str, Any], column_name: str) -> List[str]:
        values = entry.get(column_name)
        if values is None:
            raise KeyError(f"Missing required distractor column '{column_name}'")
        return list(values)

    def _get_model_distractors(self, entry: Dict[str, Any]) -> List[str]:
        """Resolve model distractor list for the current config/entry."""
        branch_column = self._branching_model_column()
        if branch_column:
            if branch_column not in entry or entry[branch_column] is None:
                raise KeyError(
                    f"Missing required branching column '{branch_column}' for "
                    f"branching_mode={self.config.branching_mode}, num_human={self.config.num_human}"
                )
            return list(entry[branch_column])
        return self._require_distractor_column(entry, self.config.model_distractor_type.value)

    def _select_distractors(
        self,
        human_distractors: List[str],
        model_distractors: List[str],
        idx: int,
    ) -> tuple[List[str], List[str]]:
        """
        Select distractors according to the configured sampling strategy.

        - independent: sample per configuration
        - branching_cumulative: use deterministic per-question order and take prefixes
        """
        if self.config.sampling_strategy == "branching_cumulative":
            if self.config.branching_mode == "human_prefix":
                return (
                    list(human_distractors[: self.config.num_human]),
                    list(model_distractors[: self.config.num_model]),
                )

            human_order = list(human_distractors)
            model_order = list(model_distractors)

            # Keep selection order stable across branching configs for the same question.
            human_rng = random.Random(self.config.seed + idx + 10_000_019)
            model_rng = random.Random(self.config.seed + idx + 20_000_033)
            human_rng.shuffle(human_order)
            model_rng.shuffle(model_order)

            return human_order[: self.config.num_human], model_order[: self.config.num_model]

        rng = random.Random(self.config.seed + idx)
        return (
            rng.sample(human_distractors, self.config.num_human),
            rng.sample(model_distractors, self.config.num_model),
        )

    def _shuffle_options(
        self,
        all_options: List[str],
        idx: int,
        num_selected_human: int,
    ) -> tuple[List[str], int, List[int], List[int]]:
        """
        Shuffle options deterministically and return new index mappings.

        Shuffle seed is config-specific so branching configurations can carry
        distractor membership while still re-randomizing option order.
        """
        gold_original_idx = 0
        human_original_indices = list(range(1, 1 + num_selected_human))
        model_original_indices = list(range(1 + num_selected_human, len(all_options)))

        shuffle_seed = (
            self.config.seed
            + (idx * 104_729)
            + (self.config.num_human * 1_009)
            + (self.config.num_model * 9_173)
        )
        shuffle_rng = random.Random(shuffle_seed)
        indices = list(range(len(all_options)))
        shuffle_rng.shuffle(indices)

        shuffled_options = [all_options[i] for i in indices]
        new_gold_idx = indices.index(gold_original_idx)
        new_human_indices = [indices.index(i) for i in human_original_indices]
        new_model_indices = [indices.index(i) for i in model_original_indices]
        return shuffled_options, new_gold_idx, new_human_indices, new_model_indices

    @staticmethod
    def _build_eval_full_question(question: str, options: List[str]) -> str:
        lines = [f"{CHOICE_LABELS[i]}: {opt}" for i, opt in enumerate(options)]
        return f"Question: {question}\n" + "\n".join(lines)

    @staticmethod
    def _serialize_eval_result(result: EvalResult) -> Dict[str, Any]:
        return asdict(result)

    @staticmethod
    def _deserialize_eval_result(data: Dict[str, Any]) -> EvalResult:
        payload = dict(data)
        required_defaults = {
            "prediction_type": None,
            "response_text": "",
            "latency_ms": 0.0,
            "eval_options_randomized": [],
            "eval_correct_answer_letter": "",
            "eval_full_question": "",
            "eval_model_input": "",
            "eval_model_output": "",
            "selected_human_distractors": [],
            "selected_model_distractors": [],
            "human_option_indices": [],
            "model_option_indices": [],
        }
        for key, default in required_defaults.items():
            payload.setdefault(key, default)
        return EvalResult(**payload)

    def _checkpoint_dir(self) -> Path:
        checkpoint_dir = self.config.checkpoint_dir or (self.config.output_dir / "checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    def _list_checkpoints(self) -> List[Path]:
        checkpoint_dir = self._checkpoint_dir()
        return sorted(
            checkpoint_dir.glob("eval_checkpoint_*.json"),
            key=lambda p: p.stat().st_mtime,
        )

    def _save_checkpoint(
        self,
        *,
        results: ExperimentResults,
        next_index: int,
        total_entries: int,
    ) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"eval_checkpoint_{next_index}_{ts}.json"
        path = self._checkpoint_dir() / filename
        payload = {
            "checkpoint_version": 1,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "config_id": self.config.config_id,
            "config_name": self.config.name,
            "next_index": next_index,
            "total_entries": total_entries,
            "attempted_entries": results.attempted_entries,
            "successful_entries": results.successful_entries,
            "failed_entries": results.failed_entries,
            "entry_failures": results.entry_failures,
            "stage_timing_seconds": results.stage_timing_seconds,
            "results": [self._serialize_eval_result(r) for r in results.results],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    def _load_latest_checkpoint(self) -> tuple[Optional[ExperimentResults], int, Optional[Path]]:
        results_path = self.config.output_dir / "results.json"
        if results_path.exists():
            return None, 0, None

        checkpoint_files = self._list_checkpoints()
        if not checkpoint_files:
            return None, 0, None

        latest = checkpoint_files[-1]
        try:
            with open(latest, "r") as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"Warning: failed to read checkpoint {latest}: {exc}")
            return None, 0, None

        if payload.get("config_id") != self.config.config_id:
            print(
                "Warning: checkpoint config mismatch "
                f"({payload.get('config_id')} != {self.config.config_id}); ignoring {latest}"
            )
            return None, 0, None

        restored = ExperimentResults(config=self.config)
        try:
            restored.results = [
                self._deserialize_eval_result(r) for r in payload.get("results", [])
            ]
        except Exception as exc:
            print(f"Warning: invalid checkpoint results payload in {latest}: {exc}")
            return None, 0, None

        restored.entry_failures = list(payload.get("entry_failures", []))
        attempted_raw = payload.get("attempted_entries")
        successful_raw = payload.get("successful_entries")
        failed_raw = payload.get("failed_entries")
        restored.attempted_entries = (
            int(attempted_raw) if attempted_raw is not None else len(restored.results)
        )
        restored.successful_entries = (
            int(successful_raw) if successful_raw is not None else len(restored.results)
        )
        restored.failed_entries = (
            int(failed_raw) if failed_raw is not None else len(restored.entry_failures)
        )
        restored.stage_timing_seconds = dict(payload.get("stage_timing_seconds", {}))
        restored.resumed_from_checkpoint = True
        restored.resumed_checkpoint_path = str(latest)

        next_index_raw = payload.get("next_index")
        next_index = int(next_index_raw) if next_index_raw is not None else restored.attempted_entries
        return restored, max(0, next_index), latest

    def _get_skip_reason(self, entry: Dict[str, Any]) -> str:
        gold_answers = entry.get("choices_answer", [])
        gold_answer = gold_answers[0] if gold_answers else ""
        if not gold_answer:
            return "missing_gold_answer"

        human_distractors = self._require_distractor_column(
            entry, DistractorType.COND_HUMAN_Q_A.value
        )
        if len(human_distractors) < self.config.num_human:
            return "insufficient_human_distractors"

        model_distractors = self._get_model_distractors(entry)
        if len(model_distractors) < self.config.num_model:
            return "insufficient_model_distractors"

        return "unknown"
    
    def _prepare_entry(self, entry, idx: int) -> Dict[str, Any]:
        """
        Prepare an entry for evaluation.
        
        Assembles options with specified distractor configuration.
        
        Raises ValueError if required fields/distractors are missing.
        """
        # Gold answer must be provided by processed datasets.
        gold_answers = entry.get("choices_answer", [])
        gold_answer = gold_answers[0] if gold_answers else ""
        if not gold_answer:
            raise ValueError("Missing choices_answer[0]; processed dataset schema is required")
        
        # Get distractors
        human_distractors = self._require_distractor_column(
            entry, DistractorType.COND_HUMAN_Q_A.value
        )
        model_distractors = self._get_model_distractors(entry)
        
        # Check if we have enough
        if len(human_distractors) < self.config.num_human:
            raise ValueError(
                f"Insufficient human distractors: required={self.config.num_human}, "
                f"found={len(human_distractors)}"
            )
        if len(model_distractors) < self.config.num_model:
            raise ValueError(
                f"Insufficient model distractors: required={self.config.num_model}, "
                f"found={len(model_distractors)}"
            )

        selected_human, selected_model = self._select_distractors(
            human_distractors,
            model_distractors,
            idx,
        )

        all_options = [gold_answer] + selected_human + selected_model
        shuffled_options, new_gold_idx, new_human_indices, new_model_indices = self._shuffle_options(
            all_options,
            idx,
            num_selected_human=len(selected_human),
        )
        
        return {
            "question": entry.get("question", ""),
            "options": shuffled_options,
            "gold_idx": new_gold_idx,
            "gold_answer": gold_answer,
            "human_indices": new_human_indices,
            "model_indices": new_model_indices,
            "selected_human": selected_human,
            "selected_model": selected_model,
            "category": entry.get("category", ""),
        }
    
    def run(self) -> ExperimentResults:
        """
        Run the experiment.
        
        Returns:
            ExperimentResults with all evaluation results
        """
        results = ExperimentResults(config=self.config)
        results.start_time = datetime.now()

        # Load data
        data_start = time.perf_counter()
        dataset = self._load_data()
        self._record_timing(results, "data_load", time.perf_counter() - data_start)

        # Get client
        client_init_start = time.perf_counter()
        client = self._get_client()
        self._record_timing(results, "client_init", time.perf_counter() - client_init_start)

        # Filter by categories if specified
        dataset_path_key = str(self.config.dataset_path)
        dataset_type_key = self.config.dataset_type_filter or "__all__"
        workpack_key = (
            str(self.config.workpack_path)
            if self.config.workpack_path is not None
            else ""
        )
        category_filter = tuple(sorted(c.lower() for c in (self.config.categories or [])))
        entries_cache_key = (
            "__entries__",
            dataset_path_key,
            dataset_type_key,
            workpack_key,
            category_filter,
        )

        entries_materialize_start = time.perf_counter()
        if self._dataset_cache is not None and entries_cache_key in self._dataset_cache:
            entries = self._dataset_cache[entries_cache_key]
        else:
            entries = list(dataset)
            if self.config.categories:
                allowed_categories = {c.lower() for c in self.config.categories}
                entries = [
                    e for e in entries
                    if str(e.get("category", "")).lower() in allowed_categories
                ]
            if self._dataset_cache is not None:
                self._dataset_cache[entries_cache_key] = entries
        self._record_timing(
            results,
            "data_materialize",
            time.perf_counter() - entries_materialize_start,
            count=len(entries),
        )

        # Apply limit
        if self.config.limit:
            entries = entries[:self.config.limit]

        start_index = 0
        restored_results, restored_next_index, restored_path = self._load_latest_checkpoint()
        if restored_results is not None:
            results = restored_results
            start_index = min(restored_next_index, len(entries))
            results.start_time = datetime.now()
            results.resumed_next_index = start_index
            print(
                f"Resuming from checkpoint: {restored_path} "
                f"(next_index={start_index}/{len(entries)})"
            )

        print(f"\n=== Running Experiment: {self.config.name} ===")
        print(f"Model: {client.name}")
        print(f"Config: {self.config.distractor_config_str}")
        print(f"Entries: {len(entries)}")
        print(f"Start index: {start_index}")
        print(f"Checkpoint save interval: {self.config.save_interval}")
        print(f"Inference batch size: {self.config.inference_batch_size}")

        # Save config
        save_config_start = time.perf_counter()
        self.config.save()
        self._record_timing(results, "write", time.perf_counter() - save_config_start, meta={"artifact": "config"})

        # Run evaluations
        quiet = getattr(self.config, "_quiet", False)
        run_entries = entries[start_index:]
        iterator = (
            ((start_index + i, entry) for i, entry in enumerate(run_entries))
            if quiet
                else enumerate(
                    tqdm(run_entries, desc="Evaluating", initial=start_index, total=len(entries)),
                    start=start_index,
                )
        )
        batch_size = max(1, int(self.config.inference_batch_size))
        pending_items: List[Dict[str, Any]] = []
        generate_kwargs = {"max_tokens": self.config.max_tokens}
        if self.config.temperature is not None:
            generate_kwargs["temperature"] = self.config.temperature
        # For local models, apply stop sequences to prevent runaway generation.
        if isinstance(client, LocalClient):
            stop_seqs = self.config.stop if self.config.stop is not None else DEFAULT_EVAL_STOP
            if stop_seqs:
                generate_kwargs["stop"] = stop_seqs
        model_load_recorded = False

        def _record_failure(idx: int, stage: str, exc: Exception, entry: Any) -> None:
            preview = str(entry.get("question", "")) if isinstance(entry, dict) else str(entry)
            failure = {
                "config_name": self.config.name,
                "config_id": self.config.config_id,
                "question_idx": idx,
                "stage": stage,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "question_preview": preview[:200],
            }
            results.entry_failures.append(failure)
            results.failed_entries += 1
            print(
                f"⚠️ Entry failed but continuing | idx={idx} stage={stage} "
                f"type={type(exc).__name__}: {exc}"
            )

        def _finalize_attempt(idx: int) -> None:
            results.attempted_entries += 1
            if results.attempted_entries % self.config.save_interval == 0:
                checkpoint_start = time.perf_counter()
                checkpoint_path = self._save_checkpoint(
                    results=results,
                    next_index=idx + 1,
                    total_entries=len(entries),
                )
                self._record_timing(
                    results,
                    "write",
                    time.perf_counter() - checkpoint_start,
                    meta={"artifact": "checkpoint", "path": str(checkpoint_path)},
                )
                print(f"💾 Checkpoint saved: {checkpoint_path}")

        def _flush_pending() -> None:
            nonlocal pending_items, model_load_recorded
            if not pending_items:
                return

            prompts = [item["prompt"] for item in pending_items]
            generate_start = time.perf_counter()
            try:
                responses = client.generate_batch(prompts, **generate_kwargs)
            except Exception as exc:
                self._record_timing(
                    results,
                    "generate",
                    time.perf_counter() - generate_start,
                    count=len(prompts),
                    meta={"status": "batch_error"},
                )
                for item in pending_items:
                    _record_failure(item["idx"], "model_generate", exc, item["entry"])
                    _finalize_attempt(item["idx"])
                pending_items = []
                return

            generate_duration = time.perf_counter() - generate_start
            self._record_timing(results, "generate", generate_duration, count=len(prompts))

            if not model_load_recorded:
                model_load_seconds = getattr(client, "model_load_seconds", None)
                if isinstance(model_load_seconds, (int, float)) and model_load_seconds > 0:
                    self._record_timing(results, "model_load", float(model_load_seconds))
                    model_load_recorded = True

            if len(responses) != len(pending_items):
                mismatch_exc = RuntimeError(
                    f"Batch response size mismatch: prompts={len(pending_items)} responses={len(responses)}"
                )
                for item in pending_items:
                    _record_failure(item["idx"], "model_generate", mismatch_exc, item["entry"])
                    _finalize_attempt(item["idx"])
                pending_items = []
                return

            per_item_latency_ms = (generate_duration * 1000.0) / max(1, len(pending_items))
            score_start = time.perf_counter()
            for item, response in zip(pending_items, responses):
                idx = item["idx"]
                entry = item["entry"]
                prepared = item["prepared"]
                prompt = item["prompt"]
                eval_full_question = item["eval_full_question"]
                try:
                    model_answer = response.extract_answer()
                    gold_letter = CHOICE_LABELS[prepared["gold_idx"]]
                    is_correct = model_answer == gold_letter

                    prediction_type = None
                    if self.config.eval_mode == "behavioral" and model_answer in CHOICE_LABELS:
                        predicted_idx = CHOICE_LABELS.index(model_answer)
                        prediction_type = determine_prediction_type(
                            predicted_idx,
                            prepared["gold_idx"],
                            prepared["human_indices"],
                            prepared["model_indices"],
                        )

                    result = EvalResult(
                        question_idx=idx,
                        question=prepared["question"],
                        gold_answer=prepared["gold_answer"],
                        gold_answer_letter=gold_letter,
                        gold_index=prepared["gold_idx"],
                        model_answer=model_answer,
                        model_prediction=model_answer,
                        is_correct=is_correct,
                        category=prepared["category"],
                        prediction_type=prediction_type,
                        response_text=response.text,
                        latency_ms=per_item_latency_ms,
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
                    results.results.append(result)
                    results.successful_entries += 1

                    # Sanity-check print
                    raw_preview = response.text[:500].replace("\n", " ")
                    correct_mark = "✓" if is_correct else "✗"
                    print(
                        f"  [idx={idx}] gold={gold_letter} | pred={model_answer or '?'} "
                        f"{correct_mark} | type={prediction_type} | "
                        f"raw({len(response.text)}t): {raw_preview!r}"
                    )
                except Exception as exc:
                    _record_failure(idx, "score", exc, entry)
                finally:
                    _finalize_attempt(idx)

            self._record_timing(results, "score", time.perf_counter() - score_start, count=len(pending_items))
            pending_items = []

        for idx, entry in iterator:
            stage = "prepare_entry"
            try:
                prepare_start = time.perf_counter()
                prepared = self._prepare_entry(entry, idx)
                stage = "build_prompt"
                prompt = build_mcqa_prompt(
                    prepared["question"],
                    prepared["options"],
                    choices_only=self.config.choices_only,
                )
                eval_full_question = self._build_eval_full_question(
                    prepared["question"],
                    prepared["options"],
                )
                self._record_timing(
                    results,
                    "prompt_build",
                    time.perf_counter() - prepare_start,
                    count=1,
                )
                pending_items.append(
                    {
                        "idx": idx,
                        "entry": entry,
                        "prepared": prepared,
                        "prompt": prompt,
                        "eval_full_question": eval_full_question,
                    }
                )
                if len(pending_items) >= batch_size:
                    _flush_pending()
            except Exception as exc:
                _record_failure(idx, stage, exc, entry)
                _finalize_attempt(idx)

        _flush_pending()

        results.end_time = datetime.now()

        # Save final checkpoint and final results
        final_checkpoint_start = time.perf_counter()
        final_checkpoint = self._save_checkpoint(
            results=results,
            next_index=len(entries),
            total_entries=len(entries),
        )
        self._record_timing(
            results,
            "write",
            time.perf_counter() - final_checkpoint_start,
            meta={"artifact": "final_checkpoint", "path": str(final_checkpoint)},
        )
        print(f"💾 Final checkpoint saved: {final_checkpoint}")

        print(f"\n=== Results ===")
        print(
            "Accuracy (successful entries): "
            f"{results.accuracy:.2%} | attempted={results.attempted_entries} "
            f"successful={results.successful_entries} failed={results.failed_entries}"
        )
        if self.config.eval_mode == "behavioral":
            print(f"Behavioral: {results.behavioral_counts}")
        if results.entry_failures:
            print(f"Entry failures logged: {len(results.entry_failures)}")

        # Save results
        results_write_start = time.perf_counter()
        results.save()
        self._record_timing(
            results,
            "write",
            time.perf_counter() - results_write_start,
            meta={"artifact": "results_json"},
        )
        self._flush_timing_events()
        print(f"Saved to: {self.config.output_dir}")

        return results


def run_experiment(
    config: ExperimentConfig,
    client: Optional[ModelClient] = None,
    dataset_cache: Optional[Dict[tuple[Any, ...], Any]] = None,
) -> ExperimentResults:
    """
    Convenience function to run an experiment.
    
    Args:
        config: Experiment configuration
        
    Returns:
        ExperimentResults
    """
    runner = ExperimentRunner(config, client=client, dataset_cache=dataset_cache)
    return runner.run()


def run_batch(configs: List[ExperimentConfig]) -> List[ExperimentResults]:
    """
    Run a batch of experiments.
    
    Args:
        configs: List of experiment configurations
        
    Returns:
        List of ExperimentResults
    """
    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running: {config.name}")
        result = run_experiment(config)
        results.append(result)
    
    return results
