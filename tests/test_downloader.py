from datasets import Dataset, DatasetDict
import pytest

from data import downloader


def test_download_dataset_filters_cached_splits(tmp_path, monkeypatch):
    cache_path = tmp_path / "cached-dataset"
    DatasetDict(
        {
            "train": Dataset.from_list([{"id": "train-row"}]),
            "test": Dataset.from_list([{"id": "test-row"}]),
        }
    ).save_to_disk(str(cache_path))

    monkeypatch.setattr(downloader, "load_dataset", lambda *args, **kwargs: pytest.fail("should load from disk"))

    dataset = downloader.download_dataset("custom/dataset", splits=["test"], save_path=cache_path)

    assert list(dataset.keys()) == ["test"]
    assert dataset["test"][0]["id"] == "test-row"


def test_download_gpqa_uses_argument_specific_default_cache_paths(tmp_path, monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_load_dataset(hf_path: str, subset: str, *, split: str, **kwargs):
        calls.append((subset, split))
        return Dataset.from_list([{"subset": subset, "split": split}])

    monkeypatch.setattr(downloader, "RAW_DATASETS_DIR", tmp_path)
    monkeypatch.setattr(downloader, "HF_TOKEN", "")
    monkeypatch.setattr(downloader, "load_dataset", fake_load_dataset)

    default_dataset = downloader.download_gpqa()
    alternate_dataset = downloader.download_gpqa(subset="gpqa_diamond", split="validation")

    assert calls == [("gpqa_main", "train"), ("gpqa_diamond", "validation")]
    assert default_dataset["train"][0]["subset"] == "gpqa_main"
    assert alternate_dataset["validation"][0]["subset"] == "gpqa_diamond"
    assert (tmp_path / "gpqa").exists()
    assert (tmp_path / "gpqa" / "gpqa_diamond" / "validation").exists()
