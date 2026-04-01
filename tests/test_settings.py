from config import settings


def test_ensure_writable_dir_falls_back_when_existing_dir_is_not_writable(tmp_path, monkeypatch):
    primary = tmp_path / "primary"
    fallback = tmp_path / "fallback"
    primary.mkdir()

    monkeypatch.setattr(
        settings.os,
        "access",
        lambda path, mode: False if path == primary else True,
    )

    resolved = settings._ensure_writable_dir(primary, fallback, "TEST_DIR")

    assert resolved == fallback
    assert fallback.exists()
