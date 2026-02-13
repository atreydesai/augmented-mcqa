"""
Experiment runner for Augmented MCQA.

Orchestrates evaluation runs using ExperimentConfig.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from tqdm import tqdm

from .config import ExperimentConfig
from models import get_client, ModelClient, GenerationResult
from config import (
    DistractorType,
    get_distractor_column,
    get_options_from_entry,
    get_answer_index,
)
from evaluation.evaluator import build_mcqa_prompt


CHOICE_LABELS = "ABCDEFGHIJ"


@dataclass
class EvalResult:
    """Result for a single evaluation."""
    question_idx: int
    question: str
    gold_answer: str
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
                    "gold_index": r.gold_index,
                    "model_answer": r.model_answer,
                    "model_prediction": r.model_prediction,
                    "is_correct": r.is_correct,
                    "category": r.category,
                    "prediction_type": r.prediction_type,
                    "latency_ms": r.latency_ms,
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
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize runner with configuration.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self._client: Optional[ModelClient] = None
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
        if self._adapter is None:
            from datasets import load_from_disk
            dataset = load_from_disk(str(self.config.dataset_path))

            # If dataset_type_filter is specified, try to use it as a split name first
            # (unified datasets are DatasetDicts with splits like "mmlu_pro", "arc_easy", etc.)
            if self.config.dataset_type_filter:
                if hasattr(dataset, "keys") and self.config.dataset_type_filter in dataset:
                    data = dataset[self.config.dataset_type_filter]
                elif hasattr(dataset, "column_names") and "dataset_type" in dataset.column_names:
                    # Flat dataset with a dataset_type column
                    dt_filter = self.config.dataset_type_filter
                    data = dataset.filter(lambda x: x["dataset_type"] == dt_filter)
                else:
                    # Fallback: use first available split
                    if "test" in dataset:
                        data = dataset["test"]
                    elif hasattr(dataset, "values"):
                        data = list(dataset.values())[0]
                    else:
                        data = dataset
            else:
                # No filter: use test split or first split
                if "test" in dataset:
                    data = dataset["test"]
                elif hasattr(dataset, "values"):
                    data = list(dataset.values())[0]
                else:
                    data = dataset

            self._adapter = data
        return self._adapter
    
    def _prepare_entry(self, entry, idx: int) -> Optional[Dict[str, Any]]:
        """
        Prepare an entry for evaluation.
        
        Assembles options with specified distractor configuration.
        
        Returns None if entry doesn't have enough distractors.
        """
        import random
        
        # Get gold answer
        gold_answers = entry.get("choices_answer", [])
        if gold_answers:
            gold_answer = gold_answers[0]
        else:
            # Fallback
            options = entry.get("options", [])
            answer_idx = entry.get("answer_index", 0)
            gold_answer = options[answer_idx] if answer_idx < len(options) else ""
            
        if not gold_answer:
            return None
        
        # Get distractors
        human_distractors = get_distractor_column(entry, DistractorType.COND_HUMAN_Q_A)
        model_distractors = get_distractor_column(entry, self.config.model_distractor_type)
        
        # Check if we have enough
        if len(human_distractors) < self.config.num_human:
            return None
        if len(model_distractors) < self.config.num_model:
            return None
        
        # Sample distractors deterministically
        rng = random.Random(self.config.seed + idx)
        
        selected_human = rng.sample(human_distractors, self.config.num_human)
        selected_model = rng.sample(model_distractors, self.config.num_model)
        
        # Combine all options
        all_options = [gold_answer] + selected_human + selected_model
        
        # Track which indices are which
        gold_original_idx = 0
        human_original_indices = list(range(1, 1 + len(selected_human)))
        model_original_indices = list(range(1 + len(selected_human), len(all_options)))
        
        # Shuffle deterministically
        indices = list(range(len(all_options)))
        rng.shuffle(indices)
        
        shuffled_options = [all_options[i] for i in indices]
        
        # Find new positions
        new_gold_idx = indices.index(gold_original_idx)
        new_human_indices = [indices.index(i) for i in human_original_indices]
        new_model_indices = [indices.index(i) for i in model_original_indices]
        
        return {
            "question": entry.get("question", ""),
            "options": shuffled_options,
            "gold_idx": new_gold_idx,
            "gold_answer": gold_answer,
            "human_indices": new_human_indices,
            "model_indices": new_model_indices,
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
                if e.category.lower() in [c.lower() for c in self.config.categories]
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
        skipped = 0
        for idx, entry in enumerate(tqdm(entries, desc="Evaluating", disable=getattr(self.config, '_quiet', False))):
            # Prepare entry
            prepared = self._prepare_entry(entry, idx)
            if prepared is None:
                skipped += 1
                continue
            
            # Build prompt
            prompt = build_mcqa_prompt(
                prepared["question"],
                prepared["options"],
                choices_only=self.config.choices_only,
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
                gold_index=prepared["gold_idx"],
                model_answer=gold_letter,
                model_prediction=model_answer,
                is_correct=is_correct,
                category=prepared["category"],
                prediction_type=prediction_type,
                response_text=response.text,
                latency_ms=latency_ms,
            )
            results.results.append(result)
        
        results.end_time = datetime.now()
        
        if skipped > 0:
            print(f"  Skipped {skipped} entries (insufficient distractors)")
        
        print(f"\n=== Results ===")
        print(f"Accuracy: {results.accuracy:.2%}")
        if self.config.eval_mode == "behavioral":
            print(f"Behavioral: {results.behavioral_counts}")
        
        # Save results
        results.save()
        print(f"Saved to: {self.config.output_dir}")
        
        return results


def run_experiment(config: ExperimentConfig) -> ExperimentResults:
    """
    Convenience function to run an experiment.
    
    Args:
        config: Experiment configuration
        
    Returns:
        ExperimentResults
    """
    runner = ExperimentRunner(config)
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
