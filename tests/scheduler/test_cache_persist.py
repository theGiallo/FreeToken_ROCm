"""CachePersister end-to-end: plain-radix and hybrid-GDN snapshots, save -> fresh
CacheManager -> on-demand materialize. CPU-only: real RadixPrefixCache/HybridRadixCache +
LinearStatePool, simple kv pool stand-in with the MHAKVCache buffer shape."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.distributed import DistributedInfo
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.cache_persist import CachePersister, default_kv_cache_dir
from freetoken.scheduler.config import SchedulerConfig

MODEL_PATH = "/fake/model-dir/others/Danube-XYZ-4B"


def _kv_pool(num_pages=64, dtype=torch.bfloat16):
    # (2, storage_layers, num_pages, page_size, kv_heads, head_dim) -- MHAKVCache layout.
    return SimpleNamespace(
        _kv_buffer=torch.zeros((2, 2, num_pages, 1, 2, 4), dtype=dtype), dtype=dtype
    )


def _linear_pool(num_slots=16):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)


def _cfg(tmp_path, **overrides):
    kwargs = dict(
        model_path=MODEL_PATH, tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16, page_size=1, cache_type="radix",
        kv_persist=True, kv_persist_dir=str(tmp_path),
    )
    kwargs.update(overrides)
    return SchedulerConfig(**kwargs)


def _stamp(kv_pool):
    buf = kv_pool._kv_buffer
    for i in range(buf.shape[2]):
        buf[:, :, i] = i + 1  # page i holds the constant (i+1)


def test_plain_radix_round_trip(tmp_path):
    cm = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "radix")
    pool = _kv_pool()
    ids = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.int32)
    idx = torch.arange(len(ids), dtype=torch.int32)
    cm.prefix_cache.insert_prefix(ids, idx)
    _stamp(pool)
    assert CachePersister(_cfg(tmp_path), cm, pool, None).save()

    cm2 = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "radix")
    pool2 = _kv_pool()
    p = CachePersister(_cfg(tmp_path), cm2, pool2, None)
    assert p.materialize(torch.cat([ids, torch.tensor([99], dtype=torch.int32)])) == len(ids)
    assert torch.equal(pool2._kv_buffer[:, :, : len(ids)], pool._kv_buffer[:, :, : len(ids)])
    mr = cm2.prefix_cache.match_prefix(ids)
    assert mr.cuda_handle.cached_len == len(ids)
    assert p.tokens_restored == len(ids)


def test_hybrid_round_trip_bit_exact_gdn(tmp_path):
    pool = _linear_pool(16)
    cm = CacheManager(16, 1, torch.zeros(4, 64, dtype=torch.int32), "hybrid_radix",
                      linear_state_pool=pool)
    kv_pool = _kv_pool()
    ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32)
    idx = torch.arange(len(ids), dtype=torch.int32)
    slot = pool.alloc(1)[0]
    pool.conv_states[:, slot] = 0.25
    pool.recurrent_states[:, slot] = 0.5
    cm.prefix_cache.insert(ids, idx, slot)
    _stamp(kv_pool)
    assert CachePersister(_cfg(tmp_path, cache_type="hybrid_radix"), cm, kv_pool,
                          pool).save()

    pool2 = _linear_pool(16)
    cm2 = CacheManager(16, 1, torch.zeros(4, 64, dtype=torch.int32), "hybrid_radix",
                       linear_state_pool=pool2)
    kv_pool2 = _kv_pool()
    p = CachePersister(_cfg(tmp_path, cache_type="hybrid_radix"), cm2, kv_pool2, pool2)
    got = p.materialize(torch.cat([ids, torch.tensor([9], dtype=torch.int32)]))
    assert got == len(ids)
    assert torch.equal(kv_pool2._kv_buffer[:, :, : len(ids)], kv_pool._kv_buffer[:, :, : len(ids)])
    m = cm2.prefix_cache.match_prefix(ids)
    assert m.cached_len == len(ids)
    assert m.mamba_value is not None
    assert torch.equal(pool2.conv_states[:, m.mamba_value], pool.conv_states[:, slot])
    assert torch.equal(pool2.recurrent_states[:, m.mamba_value], pool.recurrent_states[:, slot])


def test_non_match_serves_cold(tmp_path):
    cm = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "radix")
    pool = _kv_pool()
    ids = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    cm.prefix_cache.insert_prefix(ids, torch.arange(4, dtype=torch.int32))
    _stamp(pool)
    assert CachePersister(_cfg(tmp_path), cm, pool, None).save()

    cm2 = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "radix")
    p = CachePersister(_cfg(tmp_path), cm2, _kv_pool(), None)
    other = torch.tensor([77, 88, 99, 100, 111], dtype=torch.int32)
    assert p.materialize(other) == 0
    assert cm2.prefix_cache.match_prefix(ids).cuda_handle.cached_len == 0


def test_disabled_and_geometry_mismatch_are_cold(tmp_path):
    cm = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "radix")
    pool = _kv_pool()
    p = CachePersister(_cfg(tmp_path, kv_persist=False), cm, pool, None)
    assert not p.enabled and not p.save()

    # a different page_size writes to a different model-key dir -> fresh load finds nothing
    cm2 = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "radix")
    ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    cm2.prefix_cache.insert_prefix(ids, torch.arange(3, dtype=torch.int32))
    pool2 = _kv_pool()
    _stamp(pool2)
    assert CachePersister(_cfg(tmp_path), cm2, pool2, None).save()
    cfg_page4 = _cfg(tmp_path, page_size=4)
    cm3 = CacheManager(16, 4, torch.zeros(4, 64, dtype=torch.int32), "radix")
    p3 = CachePersister(cfg_page4, cm3, _kv_pool(), None)
    assert p3.materialize(ids) == 0


def test_max_gb_skips_save(tmp_path):
    cm = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "radix")
    pool = _kv_pool()
    cm.prefix_cache.insert_prefix(
        torch.tensor([1, 2, 3], dtype=torch.int32), torch.arange(3, dtype=torch.int32)
    )
    p = CachePersister(_cfg(tmp_path, kv_persist_max_gb=0), cm, pool, None)
    assert not p.save()


def test_stale_snapshot_loads_cold(tmp_path):
    cm = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "radix")
    pool = _kv_pool()
    ids = torch.tensor([5, 6, 7, 8], dtype=torch.int32)
    cm.prefix_cache.insert_prefix(ids, torch.arange(4, dtype=torch.int32))
    _stamp(pool)
    assert CachePersister(_cfg(tmp_path), cm, pool, None).save()

    cm2 = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "radix")
    p = CachePersister(_cfg(tmp_path, kv_persist_max_age_h=0.0), cm2, _kv_pool(), None)
    assert p.materialize(ids) == 0


def test_default_kv_cache_dir_is_xdg_aware(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_kv_cache_dir() == str(tmp_path / "ft" / "KV_cache")
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert default_kv_cache_dir().endswith(("ft" + "/" + "KV_cache"))


def test_page_size4_multi_node_chain(tmp_path):
    cm = CacheManager(64, 4, torch.zeros(4, 64, dtype=torch.int32), "radix")
    pool = _kv_pool()
    ids = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.int32)
    # page-aligned page bases: tokens 0..3 in page 0, tokens 4..7 in page 4
    idx = torch.arange(8, dtype=torch.int32) // 4 * 4
    cm.prefix_cache.insert_prefix(ids[:4], idx[:4])     # node A = page 0
    cm.prefix_cache.insert_prefix(ids, idx)             # node B (child of A) = page 4
    _stamp(pool)
    assert CachePersister(_cfg(tmp_path, page_size=4), cm, pool, None).save()

    cm2 = CacheManager(64, 4, torch.zeros(4, 64, dtype=torch.int32), "radix")
    pool2 = _kv_pool()
    p = CachePersister(_cfg(tmp_path, page_size=4), cm2, pool2, None)
    got = p.materialize(torch.cat([ids, torch.tensor([99], dtype=torch.int32)]))
    assert got == len(ids)
    assert torch.equal(pool2._kv_buffer[:, :, :2], pool._kv_buffer[:, :, :2])
    assert cm2.prefix_cache.match_prefix(ids).cuda_handle.cached_len == len(ids)


def test_two_sessions_restore_each_on_demand(tmp_path):
    cm = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "radix")
    pool = _kv_pool()
    a = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    b = torch.tensor([5, 6, 7, 8], dtype=torch.int32)
    cm.prefix_cache.insert_prefix(a, torch.arange(4, dtype=torch.int32))
    cm.prefix_cache.insert_prefix(b, torch.arange(4, 8, dtype=torch.int32))
    _stamp(pool)
    assert CachePersister(_cfg(tmp_path), cm, pool, None).save()

    cm2 = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "radix")
    pool2 = _kv_pool()
    p = CachePersister(_cfg(tmp_path), cm2, pool2, None)
    assert p.materialize(torch.cat([a, torch.tensor([9], dtype=torch.int32)])) == len(a)
    assert cm2.prefix_cache.match_prefix(a).cuda_handle.cached_len == len(a)
    assert cm2.prefix_cache.match_prefix(b).cuda_handle.cached_len == 0  # B stayed parked
    assert p.materialize(torch.cat([b, torch.tensor([9], dtype=torch.int32)])) == len(b)
    assert cm2.prefix_cache.match_prefix(b).cuda_handle.cached_len == len(b)
    assert torch.equal(pool2._kv_buffer[:, :, :8], pool._kv_buffer[:, :, :8])


def test_partial_prefix_serves_cold(tmp_path):
    cm = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "radix")
    pool = _kv_pool()
    ids = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.int32)
    cm.prefix_cache.insert_prefix(ids, torch.arange(6, dtype=torch.int32))
    _stamp(pool)
    assert CachePersister(_cfg(tmp_path), cm, pool, None).save()

    cm2 = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "radix")
    p = CachePersister(_cfg(tmp_path), cm2, _kv_pool(), None)
    cut = ids[:3]  # the parked node is 6 tokens: a 3-token prefix cannot resume it
    assert p.materialize(cut) == 0
    assert cm2.prefix_cache.match_prefix(ids).cuda_handle.cached_len == 0