"""
Experiment configuration for Augmented MCQA.

Defines ExperimentConfig for specifying evaluation parameters:
- Number of human vs model distractors
- Dataset selection
- Model selection
- Evaluation modes
"""

from dataclasses import dataclass
from typing import List, Literal, Optional
from pathlib import Path
import json

from config import (
    RESULTS_DIR,
    DEFAULT_EVAL_MAX_TOKENS,
    DEFAULT_EVAL_SAVE_INTERVAL,
    DEFAULT_EVAL_SEED,
    DEFAULT_EVAL_TEMPERATURE,
    DEFAULT_NUM_HUMAN_DISTRACTORS,
    DEFAULT_NUM_MODEL_DISTRACTORS,
)


EntryShardStrategy = Literal["contiguous", "modulo"]


@dataclass
class ExperimentConfig:
    """
    Configuration for an MCQA evaluation experiment.

    Required:
        name: Experiment name (used for results directory)
        dataset_path: Path to processed dataset
        model_name: Model identifier (e.g., "gpt-5.2-2025-12-11")
        generator_dataset_label: Label identifying which generation run produced the dataset
        setting_id: Which Final5 setting to evaluate (human_from_scratch, model_from_scratch, etc.)

    Distractor counts:
        num_human / num_model: How many human/model distractors to include

    Model options:
        reasoning_effort: For OpenAI reasoning models
        thinking_level: For Anthropic extended thinking

    Output:
        output_dir: Where to save results (defaults to results/<name>)
    """
    # Required
    name: str
    dataset_path: Path
    model_name: str
    generator_dataset_label: str
    setting_id: str = "human_from_scratch"

    # Distractor configuration
    num_human: int = DEFAULT_NUM_HUMAN_DISTRACTORS
    num_model: int = DEFAULT_NUM_MODEL_DISTRACTORS

    # Evaluation settings
    choices_only: bool = False
    limit: Optional[int] = None
    seed: int = DEFAULT_EVAL_SEED

    # Model settings
    reasoning_effort: Optional[str] = None  # OpenAI reasoning models
    thinking_level: Optional[str] = None     # Anthropic extended thinking
    temperature: Optional[float] = DEFAULT_EVAL_TEMPERATURE
    max_tokens: int = DEFAULT_EVAL_MAX_TOKENS

    # Output
    output_dir: Optional[Path] = None
    save_interval: int = DEFAULT_EVAL_SAVE_INTERVAL

    # Categories to include (None = all)
    categories: Optional[List[str]] = None

    # Filter unified dataset by dataset_type field (e.g., "mmlu_pro", "gpqa")
    dataset_type_filter: Optional[str] = None

    # Sub-sharding within config rows (for SLURM parallelism)
    entry_shards: int = 1
    entry_shard_index: int = 0
    entry_shard_strategy: EntryShardStrategy = "contiguous"

    # Local inference acceleration knobs
    inference_batch_size: int = 1
    vllm_max_num_batched_tokens: Optional[int] = None
    vllm_max_num_seqs: Optional[int] = None
    vllm_enable_chunked_prefill: Optional[bool] = None
    stop: Optional[List[str]] = None  # None → use DEFAULT_EVAL_STOP for local models
    
    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if isinstance(self.dataset_path, str):
            self.dataset_path = Path(self.dataset_path)

        if self.output_dir is None:
            self.output_dir = RESULTS_DIR / self.name
        elif isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        if not str(self.generator_dataset_label).strip():
            raise ValueError("generator_dataset_label is required and cannot be blank")
        if not str(self.setting_id).strip():
            raise ValueError("setting_id is required and cannot be blank")

        if self.save_interval <= 0:
            raise ValueError(f"save_interval must be > 0, got {self.save_interval}")
        if self.inference_batch_size <= 0:
            raise ValueError(f"inference_batch_size must be > 0, got {self.inference_batch_size}")
        if self.entry_shards <= 0:
            raise ValueError("entry_shards must be > 0")
        if self.entry_shard_index < 0 or self.entry_shard_index >= self.entry_shards:
            raise ValueError(
                f"entry_shard_index must be in [0,{self.entry_shards - 1}], got {self.entry_shard_index}"
            )
        if self.entry_shard_strategy not in {"contiguous", "modulo"}:
            raise ValueError("entry_shard_strategy must be 'contiguous' or 'modulo'")

        total = self.num_human + self.num_model
        if total < 1:
            raise ValueError("Must have at least 1 distractor (num_human + num_model >= 1)")
        if total > 9:
            raise ValueError("Cannot have more than 9 distractors (num_human + num_model <= 9)")
    
    @property
    def config_id(self) -> str:
        """Generate a unique ID for this configuration."""
        parts = [
            self.name,
            self.setting_id,
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
            "generator_dataset_label": self.generator_dataset_label,
            "setting_id": self.setting_id,
            "num_human": self.num_human,
            "num_model": self.num_model,
            "choices_only": self.choices_only,
            "limit": self.limit,
            "seed": self.seed,
            "reasoning_effort": self.reasoning_effort,
            "thinking_level": self.thinking_level,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "output_dir": str(self.output_dir),
            "save_interval": self.save_interval,
            "categories": self.categories,
            "dataset_type_filter": self.dataset_type_filter,
            "entry_shards": self.entry_shards,
            "entry_shard_index": self.entry_shard_index,
            "entry_shard_strategy": self.entry_shard_strategy,
            "inference_batch_size": self.inference_batch_size,
            "vllm_max_num_batched_tokens": self.vllm_max_num_batched_tokens,
            "vllm_max_num_seqs": self.vllm_max_num_seqs,
            "vllm_enable_chunked_prefill": self.vllm_enable_chunked_prefill,
            "stop": self.stop,
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
        data = dict(data)
        # Remove computed fields and removed fields (for backward compat with old saved configs)
        for key in ("config_id", "model_distractor_type", "eval_mode", "sampling_strategy",
                    "branching_mode", "workpack_format", "workpack_path",
                    "distractor_source", "checkpoint_dir"):
            data.pop(key, None)
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
    generator_label = kwargs.pop("generator_dataset_label", base_name)
    
    for model in models:
        for num_human, num_model in distractor_configs:
            name = f"{base_name}_{num_human}H{num_model}M_{model.replace('/', '_')}"
            
            config = ExperimentConfig(
                name=name,
                dataset_path=dataset_path,
                model_name=model,
                generator_dataset_label=generator_label,
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
