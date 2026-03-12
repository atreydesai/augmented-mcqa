from utils.sharding import select_shard


def test_contiguous_sharding_balances_remainder_to_early_shards():
    items = list(range(8))
    assert select_shard(items, shard_count=3, shard_index=0, strategy="contiguous") == [0, 1, 2]
    assert select_shard(items, shard_count=3, shard_index=1, strategy="contiguous") == [3, 4, 5]
    assert select_shard(items, shard_count=3, shard_index=2, strategy="contiguous") == [6, 7]


def test_modulo_sharding_is_stable_and_disjoint():
    items = list(range(7))
    assert select_shard(items, shard_count=3, shard_index=0, strategy="modulo") == [0, 3, 6]
    assert select_shard(items, shard_count=3, shard_index=1, strategy="modulo") == [1, 4]
    assert select_shard(items, shard_count=3, shard_index=2, strategy="modulo") == [2, 5]
