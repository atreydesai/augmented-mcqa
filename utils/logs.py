from __future__ import annotations

from pathlib import Path
from typing import Iterator

from inspect_ai.log import read_eval_log


def find_eval_logs(path: Path | str) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root] if root.suffix == ".eval" else []
    return sorted(root.glob("**/*.eval"))


def read_log(path: Path | str):
    return read_eval_log(str(path))


def iter_eval_logs(path: Path | str, *, kind: str | None = None) -> Iterator[tuple[Path, object]]:
    for log_path in find_eval_logs(path):
        log = read_log(log_path)
        metadata = getattr(log.eval, "metadata", {}) or {}
        if kind is not None and metadata.get("kind") != kind:
            continue
        yield log_path, log

