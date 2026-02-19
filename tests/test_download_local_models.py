from argparse import Namespace

from scripts import download_local_models


def test_main_without_scratch_dir_is_noop(monkeypatch):
    monkeypatch.setattr(
        download_local_models,
        "parse_args",
        lambda: Namespace(scratch_dir="", execute=False, hf_token=""),
    )
    assert download_local_models.main() == 0


def test_main_dry_run_does_not_download(monkeypatch, tmp_path):
    monkeypatch.setattr(
        download_local_models,
        "parse_args",
        lambda: Namespace(scratch_dir=str(tmp_path), execute=False, hf_token=""),
    )

    assert download_local_models.main() == 0
