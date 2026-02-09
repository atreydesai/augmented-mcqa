"""
Utility functions for Hugging Face Hub integration.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from datasets import Dataset, DatasetDict
from config import HF_TOKEN, HF_SKIP_PUSH

# Re-exporting logging to avoid direct logging if preferred, but using standard logging here
logger = logging.getLogger(__name__)

def get_default_repo_id(name: str, suffix: str = "") -> str:
    """Get default repo ID for a dataset name."""
    clean_name = name.lower().replace("_", "-").replace(" ", "-")
    if suffix:
        clean_name = f"{clean_name}-{suffix.lower().replace('_', '-')}"
    return f"atreydesai/qgqa-{clean_name}"

def push_dataset_to_hub(
    dataset_or_entries: Union[Dataset, DatasetDict, List[Dict[str, Any]]],
    repo_id: Optional[str] = None,
    dataset_name: Optional[str] = None,
    suffix: str = "",
    private: bool = False,
    **kwargs
) -> Optional[str]:
    """
    Push a dataset to the Hugging Face Hub.
    
    Args:
        dataset_or_entries: Dataset object or list of entries
        repo_id: Hugging Face repository ID (e.g., 'user/repo')
        dataset_name: Name of the dataset (used if repo_id is None)
        suffix: Suffix for the repo name (used if repo_id is None)
        private: Whether the repository should be private
        **kwargs: Additional arguments for push_to_hub
        
    Returns:
        The repository URL if successful, None otherwise
    """
    if HF_SKIP_PUSH:
        print(f"Skipping HF push (HF_SKIP_PUSH=True)")
        return None
        
    if not HF_TOKEN:
        print(f"⚠️ Skipping HF push (HF_TOKEN not set in .env)")
        return None

    # Derive repo_id if not provided
    if repo_id is None:
        if dataset_name is None:
            print("❌ Cannot push to hub: repo_id or dataset_name must be provided")
            return None
        repo_id = get_default_repo_id(dataset_name, suffix)

    try:
        # Convert list of entries to Dataset if needed
        if isinstance(dataset_or_entries, (list, tuple)):
            dataset = Dataset.from_list(list(dataset_or_entries))
        else:
            dataset = dataset_or_entries

        # Push to hub
        print(f"Pushing dataset to Hugging Face: {repo_id}...")
        dataset.push_to_hub(
            repo_id=repo_id,
            token=HF_TOKEN,
            private=private,
            **kwargs
        )
        url = f"https://huggingface.co/datasets/{repo_id}"
        print(f"✅ Successfully pushed dataset to {url}")
        return url
    except Exception as e:
        print(f"❌ Failed to push dataset to {repo_id}: {e}")
        return None
