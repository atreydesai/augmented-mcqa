"""
Data Adapter for unified column name mapping.

Provides a consistent interface for accessing dataset columns regardless
of whether they use legacy or unified naming conventions.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from datasets import Dataset, DatasetDict, load_from_disk

from config import (
    DistractorType,
    LEGACY_COLUMN_MAPPING,
    UNIFIED_TO_LEGACY,
    get_distractor_column,
)


@dataclass
class AdaptedEntry:
    """
    A dataset entry with unified column access.
    
    Provides attribute access for all distractor types using the unified
    naming convention, regardless of how the underlying data is stored.
    """
    _raw: Dict[str, Any]
    
    @property
    def question(self) -> str:
        return self._raw.get("question", "")
    
    @property
    def gold_answer(self) -> str:
        """Get the gold/correct answer."""
        # Try choices_answer first
        answers = self._raw.get("choices_answer", [])
        if answers:
            return answers[0]
        
        # Fallback to options + answer_index
        options = self._raw.get("options", [])
        answer_idx = self._raw.get("answer_index", 0)
        if answer_idx < len(options):
            return options[answer_idx]
        
        return ""
    
    @property
    def cond_human_q_a(self) -> List[str]:
        """Human distractors from original MMLU."""
        return get_distractor_column(self._raw, DistractorType.COND_HUMAN_Q_A)
    
    @property
    def cond_model_q_a(self) -> List[str]:
        """Model distractors conditioned on q+a only."""
        return get_distractor_column(self._raw, DistractorType.COND_MODEL_Q_A)
    
    @property
    def cond_model_q_a_dhuman(self) -> List[str]:
        """Model distractors conditioned on q+a+human."""
        return get_distractor_column(self._raw, DistractorType.COND_MODEL_Q_A_DHUMAN)
    
    @property
    def cond_model_q_a_dmodel(self) -> List[str]:
        """Model distractors conditioned on q+a+model."""
        return get_distractor_column(self._raw, DistractorType.COND_MODEL_Q_A_DMODEL)
    
    @property
    def assembled_options(self) -> List[str]:
        """Pre-assembled and shuffled options (from filter module)."""
        return self._raw.get("assembled_options", [])
    
    @property
    def answer(self) -> str:
        """Answer letter (A, B, C, ...) for assembled options."""
        return self._raw.get("answer", "")
    
    @property
    def answer_index(self) -> int:
        """Answer index for assembled options."""
        return self._raw.get("answer_index", 0)
    
    @property
    def category(self) -> str:
        """Question category/subject."""
        return self._raw.get("category", "")
    
    @property
    def source_mmlu_subset(self) -> str:
        """Original MMLU subset this question came from."""
        return self._raw.get("source_mmlu_subset", "")
    
    def get_distractors(self, distractor_type: DistractorType) -> List[str]:
        """Get distractors by type."""
        return get_distractor_column(self._raw, distractor_type)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get any raw field."""
        return self._raw.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get the raw dictionary."""
        return self._raw


class DataAdapter:
    """
    Adapter for loading and accessing datasets with unified column names.
    
    Usage:
        adapter = DataAdapter.from_path("/path/to/dataset")
        for entry in adapter:
            print(entry.cond_human_q_a)  # Always works, regardless of underlying column names
    """
    
    def __init__(self, dataset: Dataset):
        """Initialize with a Dataset."""
        self._dataset = dataset
        self._detect_columns()
    
    def _detect_columns(self) -> None:
        """Detect which column naming convention is used."""
        if len(self._dataset) == 0:
            self._columns = set()
            return
        
        sample = self._dataset[0]
        self._columns = set(sample.keys()) if isinstance(sample, dict) else set()
        
        # Detect naming convention
        self._uses_unified = any(
            dt.value in self._columns for dt in DistractorType
        )
        self._uses_legacy = any(
            col in self._columns for col in LEGACY_COLUMN_MAPPING
        )
    
    @classmethod
    def from_path(cls, path: str, split: Optional[str] = None) -> "DataAdapter":
        """
        Load a dataset from disk and create an adapter.
        
        Args:
            path: Path to dataset
            split: Optional split name (default: auto-detect)
            
        Returns:
            DataAdapter instance
        """
        dataset = load_from_disk(path)
        
        if isinstance(dataset, DatasetDict):
            if split:
                dataset = dataset[split]
            elif "test" in dataset:
                dataset = dataset["test"]
            else:
                dataset = list(dataset.values())[0]
        
        return cls(dataset)
    
    def __len__(self) -> int:
        return len(self._dataset)
    
    def __getitem__(self, idx: int) -> AdaptedEntry:
        return AdaptedEntry(dict(self._dataset[idx]))
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    @property
    def columns(self) -> List[str]:
        """Get column names."""
        return list(self._columns)
    
    @property
    def uses_unified_naming(self) -> bool:
        """Check if dataset uses unified naming convention."""
        return self._uses_unified
    
    @property
    def uses_legacy_naming(self) -> bool:
        """Check if dataset uses legacy naming convention."""
        return self._uses_legacy
    
    def get_raw(self, idx: int) -> Dict[str, Any]:
        """Get raw entry dictionary."""
        return dict(self._dataset[idx])
    
    def select(self, indices: List[int]) -> "DataAdapter":
        """Create a new adapter with selected indices."""
        selected = self._dataset.select(indices)
        return DataAdapter(selected)
    
    def filter_by_category(self, category: str) -> "DataAdapter":
        """Create a new adapter with only entries from a specific category."""
        indices = [
            i for i, entry in enumerate(self._dataset)
            if entry.get("category", "").lower() == category.lower()
        ]
        return self.select(indices)
    
    def to_dataset(self) -> Dataset:
        """Get the underlying Dataset."""
        return self._dataset
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the dataset."""
        if len(self) == 0:
            return {"count": 0}
        
        sample = self[0]
        
        return {
            "count": len(self),
            "columns": self.columns,
            "uses_unified_naming": self.uses_unified_naming,
            "uses_legacy_naming": self.uses_legacy_naming,
            "sample_human_count": len(sample.cond_human_q_a),
            "sample_model_count": len(sample.cond_model_q_a),
            "categories": self._get_categories(),
        }
    
    def _get_categories(self) -> List[str]:
        """Get unique categories in the dataset."""
        categories = set()
        for entry in self._dataset:
            cat = entry.get("category", "")
            if cat:
                categories.add(cat)
        return sorted(categories)


def load_adapted(path: str, split: Optional[str] = None) -> DataAdapter:
    """
    Convenience function to load an adapted dataset.
    
    Args:
        path: Path to dataset
        split: Optional split name
        
    Returns:
        DataAdapter instance
    """
    return DataAdapter.from_path(path, split)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect a dataset with DataAdapter")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to dataset",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="Number of sample entries to show",
    )
    args = parser.parse_args()
    
    adapter = load_adapted(args.path)
    
    print("\n=== Dataset Summary ===")
    summary = adapter.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print(f"\n=== Sample Entries ({args.sample}) ===")
    for i in range(min(args.sample, len(adapter))):
        entry = adapter[i]
        print(f"\nEntry {i}:")
        print(f"  Question: {entry.question[:80]}...")
        print(f"  Gold: {entry.gold_answer}")
        print(f"  Human distractors: {len(entry.cond_human_q_a)}")
        print(f"  Model distractors: {len(entry.cond_model_q_a)}")
        print(f"  Category: {entry.category}")
