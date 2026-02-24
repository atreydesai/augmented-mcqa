"""
Dataset downloading utilities for Augmented MCQA.

Supports downloading and saving datasets from HuggingFace Hub:
- MMLU-Pro (TIGER-Lab/MMLU-Pro)
- MMLU (cais/mmlu)
- ARC-Challenge (allenai/ai2_arc, ARC-Challenge config)
- GPQA (Idavidrein/gpqa, subset=gpqa_main)
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import json

from datasets import load_dataset, get_dataset_config_names, DatasetDict, Dataset
from tqdm import tqdm

from config import DATASETS_DIR, RAW_DATASETS_DIR, DATASET_CONFIGS, DatasetConfig, HF_TOKEN


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
            save_path = config.local_path
        if splits is None:
            splits = config.splits
    else:
        hf_path = dataset_name
        if save_path is None:
            # Create path from dataset name
            safe_name = dataset_name.replace("/", "_").replace("-", "_")
            save_path = RAW_DATASETS_DIR / safe_name
    
    save_path = Path(save_path)
    
    # Check if already downloaded
    if save_path.exists() and not force_download:
        print(f"Dataset already exists at {save_path}. Use force_download=True to re-download.")
        from datasets import load_from_disk
        return load_from_disk(str(save_path))
    
    print(f"Downloading dataset: {hf_path}")
    if config_name:
        print(f"  Config: {config_name}")
    
    try:
        if config_name:
            dataset = load_dataset(hf_path, config_name)
        else:
            dataset = load_dataset(hf_path)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise
    
    # Filter to requested splits if specified
    if splits:
        available_splits = list(dataset.keys())
        filtered = {}
        for split in splits:
            if split in available_splits:
                filtered[split] = dataset[split]
            else:
                print(f"  Warning: Split '{split}' not found. Available: {available_splits}")
        if filtered:
            dataset = DatasetDict(filtered)
    
    # Save to disk
    save_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(save_path))
    print(f"  Saved to: {save_path}")
    
    # Push to Hugging Face
    from data.hub_utils import push_dataset_to_hub
    push_dataset_to_hub(dataset, dataset_name=dataset_name, suffix="raw")
    
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
            from datasets import load_from_disk
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
        from datasets import load_from_disk
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
    """Download GPQA dataset (subset=gpqa_main, split=train)."""
    hf_path = "Idavidrein/gpqa"
    if save_path is None:
        save_path = RAW_DATASETS_DIR / "gpqa"
    save_path = Path(save_path)
    
    if save_path.exists() and not force_redownload:
        print(f"GPQA already exists at {save_path}")
        from datasets import load_from_disk
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


def get_dataset_info(dataset: DatasetDict) -> Dict[str, Any]:
    """Get summary information about a dataset."""
    info = {
        "splits": {},
        "total_examples": 0,
    }
    
    for split_name, split_data in dataset.items():
        split_info = {
            "num_examples": len(split_data),
            "columns": list(split_data.column_names),
        }
        info["splits"][split_name] = split_info
        info["total_examples"] += len(split_data)
        
        # Sample first entry
        if len(split_data) > 0:
            split_info["sample_entry"] = dict(split_data[0])
    
    return info


def print_dataset_info(dataset: DatasetDict, name: str = "Dataset") -> None:
    """Print summary information about a dataset."""
    info = get_dataset_info(dataset)
    print(f"\n=== {name} ===")
    print(f"Total examples: {info['total_examples']}")
    
    for split_name, split_info in info["splits"].items():
        print(f"\n  Split: {split_name}")
        print(f"    Examples: {split_info['num_examples']}")
        print(f"    Columns: {split_info['columns']}")
        
        if "sample_entry" in split_info:
            print(f"    Sample keys: {list(split_info['sample_entry'].keys())}")


def download_all_datasets() -> Dict[str, Any]:
    """
    Download all required datasets.
    
    Returns:
        Dict mapping dataset name to downloaded dataset
    """
    results = {}
    
    print("=" * 50)
    print("Downloading MMLU-Pro...")
    print("=" * 50)
    results["mmlu_pro"] = download_mmlu_pro()
    
    print("\n" + "=" * 50)
    print("Downloading MMLU (all configs)...")
    print("=" * 50)
    results["mmlu"] = download_mmlu_all_configs()
    
    print("\n" + "=" * 50)
    print("Downloading ARC...")
    print("=" * 50)
    results["arc"] = download_arc()
    
    print("\n" + "=" * 50)
    print("Downloading GPQA...")
    print("=" * 50)
    results["gpqa"] = download_gpqa()
    
    print("\n" + "=" * 50)
    print("All datasets downloaded!")
    print("=" * 50)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download datasets for Augmented MCQA")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mmlu_pro", "mmlu", "arc", "gpqa", "all"],
        default="all",
        help="Which dataset to download",
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only print info about existing datasets, don't download",
    )
    args = parser.parse_args()
    
    if args.dataset in ["mmlu_pro", "all"]:
        if args.info_only:
            from datasets import load_from_disk
            path = DATASETS_DIR / "mmlu_pro"
            if path.exists():
                ds = load_from_disk(str(path))
                print_dataset_info(ds, "MMLU-Pro")
        else:
            ds = download_mmlu_pro()
            print_dataset_info(ds, "MMLU-Pro")
    
    if args.dataset in ["mmlu", "all"]:
        if not args.info_only:
            download_mmlu_all_configs()
    
    if args.dataset in ["arc", "all"]:
        if not args.info_only:
            results = download_arc()
            for name, ds in results.items():
                print_dataset_info(ds, f"ARC-{name}")
    
    if args.dataset in ["gpqa", "all"]:
        if not args.info_only:
            ds = download_gpqa()
            print_dataset_info(ds, "GPQA")
