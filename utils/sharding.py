from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def sample_id_for_row(dataset_type: str, row: dict, index: int) -> str:
    for key in ("id", "question_id"):
        value = row.get(key)
        if value not in (None, ""):
            return f"{dataset_type}:{value}"
    return f"{dataset_type}:row-{index}"


def select_shard(
    items: Sequence[T],
    shard_count: int = 1,
    shard_index: int = 0,
    strategy: str = "contiguous",
) -> list[T]:
    if shard_count <= 0:
        raise ValueError("shard_count must be > 0")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count - 1}]")
    if shard_count == 1:
        return list(items)
    if strategy == "modulo":
        return [item for idx, item in enumerate(items) if idx % shard_count == shard_index]
    if strategy != "contiguous":
        raise ValueError("strategy must be 'contiguous' or 'modulo'")

    total = len(items)
    base = total // shard_count
    remainder = total % shard_count
    start = shard_index * base + min(shard_index, remainder)
    size = base + (1 if shard_index < remainder else 0)
    return list(items[start : start + size])

