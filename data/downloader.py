"""Dataset downloading utilities for Augmented MCQA."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import DatasetDict, get_dataset_config_names, load_dataset, load_from_disk
from tqdm import tqdm

from config import DATASET_CONFIGS, HF_TOKEN, RAW_DATASETS_DIR


def _safe_cache_name(value: str) -> str:
    return value.replace("/", "_").replace("-", "_")


def _default_dataset_save_path(dataset_name: str, config_name: Optional[str]) -> Path:
    if dataset_name in DATASET_CONFIGS:
        base_path = Path(DATASET_CONFIGS[dataset_name].local_path)
    else:
        base_path = RAW_DATASETS_DIR / _safe_cache_name(dataset_name)
    return base_path / _safe_cache_name(config_name) if config_name else base_path


def _filter_dataset_splits(dataset: DatasetDict, splits: Optional[List[str]]) -> DatasetDict:
    if not splits:
        return dataset

    available_splits = list(dataset.keys())
    filtered = {}
    for split in splits:
        if split in available_splits:
            filtered[split] = dataset[split]
        else:
            print(f"  Warning: Split '{split}' not found. Available: {available_splits}")
    return DatasetDict(filtered) if filtered else DatasetDict()


def download_dataset(
    dataset_name: str,
    config_name: Optional[str] = None,
    splits: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    force_download: bool = False,
) -> DatasetDict:
    """
    Download a dataset from HuggingFace Hub and save to disk.
    
    Args:
        dataset_name: HuggingFace dataset path or config key from DATASET_CONFIGS
        config_name: Optional config/subset name for datasets with multiple configs
        splits: List of splits to download (default: all available)
        save_path: Where to save the dataset locally
        force_download: If True, download even if already exists locally
        
    Returns:
        The downloaded DatasetDict
    """
    # Check if it's a predefined config
    if dataset_name in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset_name]
        hf_path = config.hf_path
        if save_path is None:
            save_path = _default_dataset_save_path(dataset_name, config_name)
        if splits is None:
            splits = config.splits
    else:
        hf_path = dataset_name
        if save_path is None:
            save_path = _default_dataset_save_path(dataset_name, config_name)
    
    save_path = Path(save_path)
    
    # Check if already downloaded
    if save_path.exists() and not force_download:
        print(f"Dataset already exists at {save_path}. Use force_download=True to re-download.")
        return _filter_dataset_splits(load_from_disk(str(save_path)), splits)
    
    print(f"Downloading dataset: {hf_path}")
    if config_name:
        print(f"  Config: {config_name}")
    
    if config_name:
        dataset = load_dataset(hf_path, config_name)
    else:
        dataset = load_dataset(hf_path)
    
    dataset = _filter_dataset_splits(dataset, splits)
    
    # Save to disk
    save_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(save_path))
    print(f"  Saved to: {save_path}")
    
    return dataset


def download_mmlu_pro(save_path: Optional[Path] = None) -> DatasetDict:
    """Download MMLU-Pro dataset."""
    return download_dataset("mmlu_pro", save_path=save_path)


def download_mmlu_all_configs(save_path: Optional[Path] = None) -> Dict[str, DatasetDict]:
    """
    Download all MMLU configs (subjects) and save them organized by subject.
    
    Returns:
        Dict mapping config name to DatasetDict
    """
    hf_path = "cais/mmlu"
    if save_path is None:
        save_path = RAW_DATASETS_DIR / "mmlu_all"
    save_path = Path(save_path)
    
    # Get all available configs (subjects)
    print(f"Fetching available configs for {hf_path}...")
    configs = get_dataset_config_names(hf_path)
    print(f"Found {len(configs)} configs (subjects)")
    
    results = {}
    for config in tqdm(configs, desc="Downloading MMLU subjects"):
        config_path = save_path / config
        
        if config_path.exists():
            print(f"  {config} already exists, skipping...")
            results[config] = load_from_disk(str(config_path))
            continue
            
        try:
            dataset = load_dataset(hf_path, config)
            config_path.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(config_path))
            results[config] = dataset
        except Exception as e:
            print(f"  Error downloading {config}: {e}")
            continue
    
    print(f"Downloaded {len(results)} MMLU subjects to {save_path}")
    return results


def download_arc(save_path: Optional[Path] = None) -> Dict[str, DatasetDict]:
    """
    Download ARC-Challenge only.
    
    Returns:
        Dict with key 'arc_challenge'
    """
    hf_path = "allenai/ai2_arc"
    if save_path is None:
        save_path = RAW_DATASETS_DIR / "arc"
    save_path = Path(save_path)
    
    results = {}
    config = "ARC-Challenge"
    config_key = "arc_challenge"
    config_path = save_path / config_key

    if config_path.exists():
        print(f"  {config} already exists, loading from disk...")
        results[config_key] = load_from_disk(str(config_path))
        return results

    print(f"Downloading {config}...")
    dataset = load_dataset(hf_path, config)
    config_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(config_path))
    results[config_key] = dataset
    
    return results


def download_gpqa(
    save_path: Optional[Path] = None,
    subset: str = "gpqa_main",
    split: str = "train",
    force_redownload: bool = False,
) -> DatasetDict:
    """Download a GPQA subset/split into an argument-specific local cache path."""
    hf_path = "Idavidrein/gpqa"
    if save_path is None:
        save_path = RAW_DATASETS_DIR / "gpqa"
        if subset != "gpqa_main" or split != "train":
            save_path = save_path / _safe_cache_name(subset) / _safe_cache_name(split)
    save_path = Path(save_path)
    
    if save_path.exists() and not force_redownload:
        print(f"GPQA already exists at {save_path}")
        return load_from_disk(str(save_path))
    
    print(f"Downloading GPQA ({subset}/{split})...")
    load_kwargs: Dict[str, Any] = {}
    if HF_TOKEN:
        load_kwargs["token"] = HF_TOKEN
    if force_redownload:
        load_kwargs["download_mode"] = "force_redownload"

    dataset_split = load_dataset(hf_path, subset, split=split, **load_kwargs)
    dataset = DatasetDict({split: dataset_split})
    
    save_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(save_path))
    print(f"Saved to {save_path}")
    
    return dataset
