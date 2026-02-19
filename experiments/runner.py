"""
Experiment runner for Augmented MCQA.

Orchestrates evaluation runs using ExperimentConfig.
"""

import json
import random
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from tqdm import tqdm

from .config import ExperimentConfig
from models import get_client, ModelClient
from config import DistractorType
from evaluation.evaluator import build_mcqa_prompt


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
                "behavioral_counts": self.behavioral_counts,
                "accuracy_by_category": self.accuracy_by_category,
            },
            "timing": {
                "start": self.start_time.isoformat() if self.start_time else None,
                "end": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": (self.end_time - self.start_time).total_seconds()
                    if self.start_time and self.end_time else None,
            },
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
        dataset_cache: Optional[Dict[tuple[str, str], Any]] = None,
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
    
    def _get_client(self) -> ModelClient:
        """Get or create model client."""
        if self._client is None:
            kwargs = {}
            if self.config.reasoning_effort:
                kwargs["reasoning_effort"] = self.config.reasoning_effort
            if self.config.thinking_level:
                kwargs["thinking_level"] = self.config.thinking_level
            
            self._client = get_client(self.config.model_name, **kwargs)
        return self._client
    
    def _load_data(self) -> Any:
        """Load and optionally filter dataset by dataset_type."""
        if self._adapter is not None:
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
        dataset = self._load_data()
        
        # Get client
        client = self._get_client()
        
        # Filter by categories if specified
        entries = list(dataset)
        if self.config.categories:
            entries = [
                e for e in entries
                if str(e.get("category", "")).lower()
                in [c.lower() for c in self.config.categories]
            ]
        
        # Apply limit
        if self.config.limit:
            entries = entries[:self.config.limit]
        
        print(f"\n=== Running Experiment: {self.config.name} ===")
        print(f"Model: {client.name}")
        print(f"Config: {self.config.distractor_config_str}")
        print(f"Entries: {len(entries)}")
        
        # Save config
        self.config.save()
        
        # Run evaluations
        quiet = getattr(self.config, '_quiet', False)
        iterator = enumerate(entries) if quiet else enumerate(tqdm(entries, desc="Evaluating"))
        for idx, entry in iterator:
            # Prepare entry
            prepared = self._prepare_entry(entry, idx)
            
            # Build prompt
            prompt = build_mcqa_prompt(
                prepared["question"],
                prepared["options"],
                choices_only=self.config.choices_only,
            )
            eval_full_question = self._build_eval_full_question(
                prepared["question"],
                prepared["options"],
            )
            
            # Generate
            start_ms = time.time() * 1000
            response = client.generate(
                prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            latency_ms = time.time() * 1000 - start_ms
            
            # Extract answer
            model_answer = response.extract_answer()
            
            # Check correctness
            gold_letter = CHOICE_LABELS[prepared["gold_idx"]]
            is_correct = model_answer == gold_letter
            
            # Determine prediction type for behavioral analysis
            prediction_type = None
            if self.config.eval_mode == "behavioral" and model_answer in CHOICE_LABELS:
                predicted_idx = CHOICE_LABELS.index(model_answer)
                prediction_type = determine_prediction_type(
                    predicted_idx,
                    prepared["gold_idx"],
                    prepared["human_indices"],
                    prepared["model_indices"],
                )
            
            # Create result
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
            results.results.append(result)
        
        results.end_time = datetime.now()
        
        print(f"\n=== Results ===")
        print(f"Accuracy: {results.accuracy:.2%}")
        if self.config.eval_mode == "behavioral":
            print(f"Behavioral: {results.behavioral_counts}")
        
        # Save results
        results.save()
        print(f"Saved to: {self.config.output_dir}")
        
        return results


def run_experiment(
    config: ExperimentConfig,
    client: Optional[ModelClient] = None,
    dataset_cache: Optional[Dict[tuple[str, str], Any]] = None,
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
