"""
Experiment configuration for Augmented MCQA.

Defines ExperimentConfig for specifying evaluation parameters:
- Number of human vs model distractors
- Dataset selection
- Model selection
- Evaluation modes
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal
from pathlib import Path
import json

from config import (
    DATASETS_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
    DistractorType,
)


# Evaluation modes
EvalMode = Literal["accuracy", "behavioral"]
SamplingStrategy = Literal["independent", "branching_cumulative"]


@dataclass
class ExperimentConfig:
    """
    Configuration for an MCQA evaluation experiment.
    
    Attributes:
        name: Experiment name (used for results directory)
        dataset_path: Path to processed dataset
        model_name: Model identifier (e.g., "gpt-5.2-2025-12-11")
        
        num_human: Number of human distractors to include
        num_model: Number of model distractors to include
        model_distractor_type: Which type of model distractors to use
        
        eval_mode: "accuracy" (just check correctness) or "behavioral" (track G/H/M patterns)
        
        limit: Maximum entries to evaluate (None = all)
        seed: Random seed for reproducibility
        
        reasoning_effort: For OpenAI GPT-5 models
        thinking_level: For Anthropic/Gemini models
        
        temperature: Sampling temperature
        max_tokens: Max tokens for response
        
        output_dir: Where to save results
    """
    # Required
    name: str
    dataset_path: Path
    model_name: str
    
    # Distractor configuration
    num_human: int = 3
    num_model: int = 0
    model_distractor_type: DistractorType = DistractorType.COND_MODEL_Q_A
    
    # Evaluation settings
    eval_mode: EvalMode = "accuracy"
    sampling_strategy: SamplingStrategy = "independent"
    choices_only: bool = False
    limit: Optional[int] = None
    seed: int = RANDOM_SEED
    
    # Model settings
    reasoning_effort: Optional[str] = None  # OpenAI GPT-5
    thinking_level: Optional[str] = None     # Anthropic/Gemini
    temperature: float = 0.0
    max_tokens: int = 100
    
    # Output
    output_dir: Optional[Path] = None
    
    # Categories to include (None = all)
    categories: Optional[List[str]] = None

    # Filter unified dataset by dataset_type field (e.g., "mmlu_pro", "supergpqa")
    dataset_type_filter: Optional[str] = None

    # Track which distractor source this config uses (scratch/dhuman/dmodel)
    distractor_source: Optional[str] = None
    
    def __post_init__(self):
        """Validate and set defaults after initialization."""
        # Convert paths
        if isinstance(self.dataset_path, str):
            self.dataset_path = Path(self.dataset_path)
        
        # Set default output directory
        if self.output_dir is None:
            self.output_dir = RESULTS_DIR / self.name
        elif isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        
        # Validate distractor counts
        total = self.num_human + self.num_model
        if total < 1:
            raise ValueError("Must have at least 1 distractor (num_human + num_model >= 1)")
        if total > 9:
            raise ValueError("Cannot have more than 9 distractors (num_human + num_model <= 9)")
        if self.sampling_strategy not in {"independent", "branching_cumulative"}:
            raise ValueError(
                "sampling_strategy must be 'independent' or 'branching_cumulative'"
            )
    
    @property
    def config_id(self) -> str:
        """Generate a unique ID for this configuration."""
        parts = [
            self.name,
            f"{self.num_human}H{self.num_model}M",
            self.model_name.replace("/", "_"),
        ]
        if self.reasoning_effort:
            parts.append(f"re_{self.reasoning_effort}")
        if self.thinking_level:
            parts.append(f"tl_{self.thinking_level}")
        return "_".join(parts)
    
    @property
    def distractor_config_str(self) -> str:
        """Human-readable distractor configuration."""
        return f"{self.num_human}H+{self.num_model}M"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "dataset_path": str(self.dataset_path),
            "model_name": self.model_name,
            "num_human": self.num_human,
            "num_model": self.num_model,
            "model_distractor_type": self.model_distractor_type.value,
            "eval_mode": self.eval_mode,
            "sampling_strategy": self.sampling_strategy,
            "choices_only": self.choices_only,
            "limit": self.limit,
            "seed": self.seed,
            "reasoning_effort": self.reasoning_effort,
            "thinking_level": self.thinking_level,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "output_dir": str(self.output_dir),
            "categories": self.categories,
            "dataset_type_filter": self.dataset_type_filter,
            "distractor_source": self.distractor_source,
            "config_id": self.config_id,
        }
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save configuration to JSON file."""
        if path is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / "config.json"
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return path
    
    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """Create from dictionary."""
        # Convert distractor type
        if "model_distractor_type" in data:
            data["model_distractor_type"] = DistractorType(data["model_distractor_type"])
        
        # Remove computed fields
        data.pop("config_id", None)
        
        return cls(**data)
    
    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


def create_batch_configs(
    base_name: str,
    dataset_path: Path,
    models: List[str],
    distractor_configs: List[tuple[int, int]],
    **kwargs,
) -> List[ExperimentConfig]:
    """
    Create a batch of experiment configurations.
    
    Args:
        base_name: Base experiment name
        dataset_path: Dataset path
        models: List of model names
        distractor_configs: List of (num_human, num_model) tuples
        **kwargs: Additional config parameters
        
    Returns:
        List of ExperimentConfig objects
    """
    configs = []
    
    for model in models:
        for num_human, num_model in distractor_configs:
            name = f"{base_name}_{num_human}H{num_model}M_{model.replace('/', '_')}"
            
            config = ExperimentConfig(
                name=name,
                dataset_path=dataset_path,
                model_name=model,
                num_human=num_human,
                num_model=num_model,
                **kwargs,
            )
            configs.append(config)
    
    return configs


def save_batch_configs(configs: List[ExperimentConfig], output_dir: Path) -> Path:
    """
    Save a batch of configurations to a single JSON file.
    
    Args:
        configs: List of configurations
        output_dir: Output directory
        
    Returns:
        Path to saved batch config file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    batch_path = output_dir / "batch_configs.json"
    
    batch_data = {
        "count": len(configs),
        "configs": [c.to_dict() for c in configs],
    }
    
    with open(batch_path, 'w') as f:
        json.dump(batch_data, f, indent=2)
    
    return batch_path


def load_batch_configs(path: Path) -> List[ExperimentConfig]:
    """
    Load a batch of configurations from JSON file.
    
    Args:
        path: Path to batch config file
        
    Returns:
        List of ExperimentConfig objects
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    return [ExperimentConfig.from_dict(c) for c in data["configs"]]
