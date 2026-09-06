"""KV / GDN prefix-cache persistence (--kv-persist).

The KV radix tree lives only in RAM, so an ft restart loses every warm prefix and a long
client conversation cold-prefills again (minutes on a hybrid 35B). This module dumps the
tree plus the pages and GDN snapshots it references at shutdown and re-materializes them
per session on demand at runtime.

Save (shutdown): walk the live prefix cache; pack every referenced _kv_buffer page into
kv.bin and every referenced LinearStatePool slot into gdn.bin; serialize the node tree to
tree.json; meta.json is the atomic commit marker written last (tmp + os.replace). A
partially written snapshot (crash mid-save) is refused on load because meta is missing.

Restore (load): read meta + tree.json into a parked RadixTreeNode graph (metadata only,
zero GPU copies, so CUDA-graph addresses stay valid). The first request whose token prefix
matches a parked path triggers materialization: fresh pool pages + GDN slots are filled
from the blobs and the re-pointed path is inserted into the LIVE cache through the same
insert calls the runtime commit path uses. After that the scheduler's normal match_req
sees a genuine prefix hit.

Session association is implicit: a session IS its token prefix (the tree key); a parked
path that matches the request is exactly that session's cached context, and non-matching
parked sessions stay cold until a request names them. pi sends no user/metadata id
(verified in the reqlog), which is why there is no explicit key.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import TYPE_CHECKING, Iterator, List, Tuple

import numpy as np
import torch
from freetoken.utils import align_down, init_logger

if TYPE_CHECKING:
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.kvcache.radix_cache import RadixTreeNode
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.config import SchedulerConfig

logger = init_logger(__name__)

_SNAPSHOT_VERSION = 1


def default_kv_cache_dir() -> str:
    """XDG-aware snapshot root; ~/.cache/ft/KV_cache when XDG_CACHE_HOME is unset."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(base, "ft", "KV_cache")


def _cpu_bytes(t: torch.Tensor) -> bytes:
    """Raw bytes of a (device) tensor: cpu copy first so the uint8 reinterpret does
    not alias a GPU view, then a numpy-only-safe uint8 pass (bf16 has no numpy dtype)."""
    return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


class CachePersister:
    """Save the KV/GDN cache to disk at shutdown and restore it per session on demand.

    v1 scope: MHA-pool radix and GDN-hybrid radix only. Naive, SWA/DSV4 and owned-KV
    pools refuse loudly (their secondary currencies / owned tiers are not snapshot-able).
    """

    def __init__(
        self,
        config: "SchedulerConfig",
        cache_manager: "CacheManager",
        kv_pool: object | None,
        linear_state_pool: "LinearStatePool | None",
    ) -> None:
        self._config = config
        self._cm = cache_manager
        self._kv_pool = kv_pool
        self._lsp = linear_state_pool
        self._tokens_restored = 0

        # Parked store (loaded at startup, metadata only; zero GPU work).
        self._root: "RadixTreeNode | None" = None
        self._key_fn = None
        self._meta: dict | None = None
        self._kv_mmap: np.memmap | None = None
        self._gdn_mmap: np.memmap | None = None
        self._loaded = False
        self._save_ok_path = ""

        self._supported = self._check_supported()
        if self._supported:
            if config.kv_persist_dir:
                # Validate an explicit dir early: a typo should fail at startup, not at stop.
                os.makedirs(config.kv_persist_dir, exist_ok=True)
            self.load()

    # ------------------------------------------------------------------ setup
    def _check_supported(self) -> bool:
        if not self._config.kv_persist:
            return False
        cm = self._cm
        if cm.cache_type not in ("radix", "hybrid_radix"):
            logger.warning_rank0(
                "kv-persist: cache_type %s is not snapshot-able; disabled", cm.cache_type
            )
            return False
        if cm.is_swa:
            logger.warning_rank0("kv-persist: SWA caches are not snapshot-able; disabled")
            return False
        if cm.swa_paged:
            logger.warning_rank0(
                "kv-persist: owned/sliding-window pools are not snapshot-able; disabled"
            )
            return False
        if self._kv_pool is None or not hasattr(self._kv_pool, "_kv_buffer"):
            logger.warning_rank0("kv-persist: no paged KV buffer available; disabled")
            return False
        if cm.is_hybrid and self._lsp is None:
            logger.warning_rank0("kv-persist: hybrid model has no LinearStatePool; disabled")
            return False
        return True

    @property
    def enabled(self) -> bool:
        return self._supported

    @property
    def tokens_restored(self) -> int:
        return self._tokens_restored

    # ------------------------------------------------------------------ layout
    def _model_key(self) -> str:
        shape = self._kv_pool._kv_buffer.shape  # (2, L, P, ps, H, D)
        parts = (
            os.path.normpath(self._config.model_path),
            self._config.cache_type,
            self._config.page_size,
            str(self._config.dtype),
            shape[1],
            shape[4],
            shape[5],
            # TP ranks hold disjoint KV shards with per-rank pages; never let one rank's
            # blobs clobber another's in the shared snapshot dir.
            getattr(getattr(self._config, "tp_info", None), "rank", 0),
        )
        h = hashlib.sha256("||".join(str(p) for p in parts).encode()).hexdigest()
        return f"{os.path.basename(parts[0]) or 'model'}-{h[:12]}"

    def _snapshot_dir(self) -> str:
        base = self._config.kv_persist_dir or default_kv_cache_dir()
        return os.path.join(base, self._model_key())

    def _page_bytes(self) -> int:
        shape = self._kv_pool._kv_buffer.shape
        # One pool page as stored in kv.bin: K+V over all storage layers, excluding the
        # num_pages dim -- (2, L, P, ps, H, D) -> prod(shape[0], shape[1], shape[3:]) bytes.
        per = shape[0] * int(np.prod((shape[1], *shape[3:]), dtype=np.int64))
        return int(per) * self._kv_pool.dtype.itemsize

    def _bytes_per_token(self) -> int:
        return self._page_bytes() // self._config.page_size

    # ------------------------------------------------------------------ save
    def save(self) -> bool:
        """Write the current live tree + referenced pages/slots. Returns False when
        disabled, unsupported or over the --kv-persist-max-gb budget (skipped with a log)."""
        if not self._supported:
            return False
        cm = self._cm
        page_bytes = self._page_bytes()
        nodes = _preorder_nodes(cm.prefix_cache)
        if not nodes:
            logger.info_rank0("kv-persist: empty tree, nothing to save")
            return False
        unique_pages = _unique_pages(nodes, cm.page_size)
        page_pos = {p: i for i, p in enumerate(unique_pages)}

        gdn_layout = None
        unique_slots: List[int] = []
        if cm.is_hybrid:
            gdn_layout = self._gdn_layout()
            unique_slots = _unique_gdn_slots(nodes)
            gdn_layout["num_slots"] = len(unique_slots)
        slot_bytes = gdn_layout["slot_bytes"] if gdn_layout else 0

        kv_total = len(unique_pages) * page_bytes
        gdn_total = len(unique_slots) * slot_bytes
        est_total = kv_total + gdn_total + 64 * len(nodes) + 4096
        if self._config.kv_persist_max_gb is not None and (
            est_total > self._config.kv_persist_max_gb * (1 << 30)
        ):
            logger.warning_rank0(
                "kv-persist: snapshot of %d KV pages (%.3f GiB) exceeds "
                "--kv-persist-max-gb %d; skipping the save",
                len(unique_pages),
                kv_total / (1 << 30),
                self._config.kv_persist_max_gb,
            )
            return False

        d = self._snapshot_dir()
        os.makedirs(d, exist_ok=True)
        tmp_kv, tmp_gdn, tmp_tree, tmp_meta = (
            os.path.join(d, n)
            for n in ("kv.bin.tmp", "gdn.bin.tmp", "tree.json.tmp", "meta.json.tmp")
        )
        try:
            buf = self._kv_pool._kv_buffer  # (2, L, P, ps, H, D)
            with open(tmp_kv, "wb") as f:
                for r0, r1 in _page_runs(unique_pages):
                    # Reorder each run to per-page slabs (nrun, 2, L, ps, H, D) so a node's
                    # bytes sit at page_pos[first_page] * page_bytes, matching kv_off.
                    run = buf[:, :, r0 : r1 + 1].permute(2, 0, 1, 3, 4, 5)
                    f.write(_cpu_bytes(run))
            if gdn_layout is not None:
                with open(tmp_gdn, "wb") as f:
                    for slot in unique_slots:
                        f.write(_gdn_slot_bytes(self._lsp, slot))
            records = []
            for node in nodes:
                p0 = int(node.value[0]) // cm.page_size
                mamba_off = None
                if node.mamba_value is not None:
                    mamba_off = unique_slots.index(node.mamba_value) * slot_bytes
                records.append(
                    {
                        "key": node._key.cpu().tolist(),
                        "parent": node._persist_parent,
                        "kv_off": page_pos[p0] * page_bytes,
                        "kv_len": (node.length // cm.page_size) * page_bytes,
                        "mamba_off": mamba_off,
                    }
                )
            with open(tmp_tree, "w") as f:
                json.dump(records, f)
            meta = {
                "version": _SNAPSHOT_VERSION,
                "cache_type": cm.cache_type,
                "page_size": cm.page_size,
                "dtype": str(self._kv_pool.dtype),
                "kv_shape": list(self._kv_pool._kv_buffer.shape),
                "num_pages": len(unique_pages),
                "num_tokens": sum(n.length for n in nodes),
                "node_count": len(nodes),
                "saved_at": time.time(),
                "page_bytes": page_bytes,
                "kv_blob_bytes": kv_total,
                "gdn": gdn_layout,
            }
            with open(tmp_meta, "w") as f:
                json.dump(meta, f, indent=2)
            # Commit: kv/gdn/tree first, meta LAST -- its presence is the load guard.
            os.replace(tmp_kv, os.path.join(d, "kv.bin"))
            if gdn_layout is not None:
                os.replace(tmp_gdn, os.path.join(d, "gdn.bin"))
            os.replace(tmp_tree, os.path.join(d, "tree.json"))
            os.replace(tmp_meta, os.path.join(d, "meta.json"))
        except Exception:  # noqa: BLE001 -- a failed save must never kill a graceful stop
            logger.warning_rank0("kv-persist: failed to write snapshot to %s", d, exc_info=True)
            for p in (tmp_kv, tmp_gdn, tmp_tree, tmp_meta):
                try:
                    os.remove(p)
                except OSError:
                    pass
            return False
        self._save_ok_path = d
        logger.info_rank0(
            "kv-persist: saved %d tokens / %d pages / %d GDN slots to %s",
            meta["num_tokens"],
            len(unique_pages),
            len(unique_slots),
            d,
        )
        return True

    # ------------------------------------------------------------------ load
    def load(self) -> bool:
        if not self._supported:
            return False
        d = self._snapshot_dir()
        meta_path = os.path.join(d, "meta.json")
        if not os.path.exists(meta_path):
            return False
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if not self._meta_ok(meta):
                return False
            kv_path = os.path.join(d, "kv.bin")
            self._kv_mmap = np.memmap(kv_path, dtype=np.uint8, mode="r")
            if len(self._kv_mmap) != meta.get("kv_blob_bytes", len(self._kv_mmap)):
                logger.warning_rank0("kv-persist: kv.bin size mismatch; serving cold")
                return False
            if meta["gdn"] is not None:
                if not self._slots_fit(meta):
                    return False
                self._gdn_mmap = np.memmap(os.path.join(d, "gdn.bin"), dtype=np.uint8, mode="r")
            self._build_parked_tree(meta)
            self._meta = meta
            self._loaded = True
            logger.info_rank0(
                "kv-persist: loaded %d parked nodes (%d tokens) from %s",
                meta["node_count"],
                meta["num_tokens"],
                d,
            )
            return True
        except Exception:  # noqa: BLE001 -- a corrupt snapshot must never block startup
            logger.warning_rank0("kv-persist: snapshot at %s unreadable; serving cold", d, exc_info=True)
            return False

    def _meta_ok(self, meta: dict) -> bool:
        if meta.get("version") != _SNAPSHOT_VERSION:
            return False
        cm = self._cm
        if meta.get("cache_type") != cm.cache_type or meta.get("page_size") != cm.page_size:
            logger.warning_rank0("kv-persist: cache geometry mismatch; serving cold")
            return False
        if str(self._kv_pool.dtype) != meta.get("dtype"):
            logger.warning_rank0("kv-persist: KV dtype mismatch; serving cold")
            return False
        saved_shape = meta.get("kv_shape")
        shape = self._kv_pool._kv_buffer.shape
        if saved_shape is not None and (
            saved_shape[0] != shape[0]
            or tuple(saved_shape[1:]) != tuple(shape[1:])
        ):
            logger.warning_rank0("kv-persist: pool geometry mismatch; serving cold")
            return False
        if meta.get("page_bytes") != self._page_bytes():
            logger.warning_rank0("kv-persist: page layout mismatch; serving cold")
            return False
        max_age = self._config.kv_persist_max_age_h
        if max_age is not None and time.time() - meta.get("saved_at", 0) > max_age * 3600:
            logger.info_rank0(
                "kv-persist: snapshot is older than %sh; serving cold", max_age
            )
            return False
        return True

    def _slots_fit(self, meta: dict) -> bool:
        need = (meta["gdn"] or {}).get("num_slots", 0)
        cap = self._lsp.num_slots - 1  # slot 0 is the padding sink
        if need > cap:
            logger.warning_rank0(
                "kv-persist: snapshot needs %d GDN slots, pool has %d; serving cold",
                need,
                cap,
            )
            return False
        return True

    def _build_parked_tree(self, meta: dict) -> None:
        from freetoken.kvcache.radix_cache import RadixTreeNode, _get_key_fn

        key_fn = _get_key_fn(self._config.page_size)
        self._key_fn = key_fn
        root = RadixTreeNode(key_fn)
        root.set_key_value(
            torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.int32)
        )
        root.ref_count = 1
        with open(os.path.join(self._snapshot_dir(), "tree.json")) as f:
            records: List[dict] = json.load(f)
        built: List[RadixTreeNode] = []
        for rec in records:
            n = RadixTreeNode(key_fn)
            n.set_key_value(
                torch.tensor(rec["key"], dtype=torch.int32),
                torch.empty(len(rec["key"]), dtype=torch.int32),
            )
            n.persist_kv_off = rec["kv_off"]
            n.persist_kv_len = rec["kv_len"]
            n.persist_mamba_off = rec["mamba_off"]
            n.persist_start = 0
            n.persist_end = 0
            parent = root if rec["parent"] is None else built[rec["parent"]]
            n.set_parent(parent)
            built.append(n)
        self._root = root

    # ------------------------------------------------------------------ restore
    def materialize(self, input_ids: torch.Tensor, mm_embeds: torch.Tensor | None = None) -> int:
        """Restore the parked session path matched by ``input_ids`` into the live cache.

        Mirrors CacheManager.match_req's prefix convention (the last token is never
        matched). Returns the newly resident prefix length (0 = cold / no-op)."""
        if not self._loaded or input_ids.numel() == 0:
            return 0
        if mm_embeds is not None:
            return 0  # multimodal deliberately matches against nothing (see match_req)
        ids = input_ids[: input_ids.numel() - 1]
        if ids.numel() == 0:
            return 0
        cm = self._cm
        restore_len, parked = self._match_parked(ids)
        if restore_len == 0:
            return 0
        live = cm.prefix_cache.match_prefix(ids[:restore_len])
        if cm.is_hybrid:
            live_len, live_pages = live.cached_len, live.kv_indices
        else:
            live_len = live.cuda_handle.cached_len
            if live_len == 0:
                # get_matched_indices() needs at least one matched node below root.
                live_pages = torch.empty(0, dtype=torch.int32)
            else:
                live_pages = live.cuda_handle.get_matched_indices()
        if live_len >= restore_len:
            return 0  # fully resident: a prior materialization owns this path
        return self._materialize_range(ids, parked, live_len, live_pages, restore_len)

    def _materialize_range(self, ids, parked, live_len, live_pages, restore_len) -> int:
        cm = self._cm
        num_pages = (restore_len - live_len) // cm.page_size
        key = ids[:restore_len]
        matched = cm.prefix_cache.match_prefix(key)
        if cm.is_hybrid:
            from freetoken.kvcache.hybrid_radix_cache import HybridCacheHandle

            handle = HybridCacheHandle(matched.cached_len, matched.node, matched.kv_indices)
        else:
            handle = matched.cuda_handle
        if live_len > 0:
            cm.lock(handle)
        try:
            fresh = cm._allocate(num_pages)  # page bases (token offsets), one per page
            fresh_page0 = int(fresh[0]) // cm.page_size
            # Canonical per-token page bases: repeat each page base page_size times.
            fresh_tokens = fresh.repeat_interleave(cm.page_size)
            pages_all = torch.cat([live_pages, fresh_tokens])  # length == restore_len
            for node in parked:
                b = node.persist_end
                if b <= live_len:
                    continue
                if b > restore_len:
                    break
                s = node.persist_start
                self._restore_kv(node, s, b, fresh_page0 + (s - live_len) // cm.page_size)
                if cm.is_hybrid:
                    slot = self._claim_gdn(node.persist_mamba_off)
                    _, exist = cm.prefix_cache.insert(key[:b], pages_all[:b], slot)
                    if exist:
                        self._lsp.free([slot])  # an identical snapshot already won
                else:
                    cm.prefix_cache.insert_prefix(key[:b], pages_all[:b])
        finally:
            if live_len > 0:
                cm.unlock(handle)
        self._tokens_restored += restore_len
        logger.info_rank0(
            "kv-persist: materialized %d tokens (%d pages) for the matched session",
            restore_len,
            num_pages,
        )
        return restore_len

    def _claim_gdn(self, off: int | None) -> int:
        """Allocate a fresh GDN slot and fill it from the blob; returns the slot id."""
        slot = self._lsp.alloc(1)[0]
        self._restore_gdn_slot(slot, off)
        return slot

    def _restore_kv(self, node, s: int, b: int, p0: int) -> None:
        page_bytes = self._page_bytes()
        per_token = self._bytes_per_token()
        off = node.persist_kv_off + (s - node.persist_start) * per_token
        length = (b - s) * per_token
        pages = length // page_bytes
        raw = self._kv_mmap[off : off + length].copy()
        per_page = page_bytes // self._kv_pool.dtype.itemsize
        flat = torch.frombuffer(raw, dtype=torch.uint8).view(self._kv_pool.dtype)
        shape = self._kv_pool._kv_buffer.shape
        for j in range(pages):
            # kv.bin stores per-page slabs (page-major), one (2, L, ps, H, D) block each.
            self._kv_pool._kv_buffer[:, :, p0 + j].copy_(
                flat[j * per_page : (j + 1) * per_page].reshape(
                    shape[0], shape[1], shape[3], shape[4], shape[5]
                )
            )

    def _restore_gdn_slot(self, dst: int, off: int) -> None:
        pool = self._lsp
        gdn = self._meta["gdn"]
        total = gdn["slot_bytes"]
        raw = self._gdn_mmap[off : off + total].copy()
        cursor = 0

        def _take(num_bytes: int, shape, dtype: torch.dtype) -> torch.Tensor:
            chunk = raw[cursor : cursor + num_bytes]
            t = torch.frombuffer(chunk, dtype=torch.uint8).view(dtype)
            return t.reshape(shape)

        # Order must match _gdn_slot_bytes exactly: conv, recurrent, slot_states.
        cb = gdn["conv_bytes"]
        pool.conv_states[:, dst].copy_(
            _take(cb, pool.conv_states[:, dst].shape, pool.conv_states.dtype)
        )
        cursor += cb
        rb = gdn["rec_bytes"]
        pool.recurrent_states[:, dst].copy_(
            _take(rb, pool.recurrent_states[:, dst].shape, pool.recurrent_states.dtype)
        )
        cursor += rb
        for name in gdn["slot_state_names"]:
            t = pool.slot_states[name]
            b = gdn["slot_state_bytes"][name]
            t[:, dst].copy_(_take(b, t[:, dst].shape, t.dtype))
            cursor += b

    def _match_parked(self, ids):
        """Walk the parked tree; return (deepest fully-matched prefix end, path nodes).

        For hybrid caches the walk stops at a tombstoned boundary (a parked node whose GDN
        snapshot was evicted in the prior run): the live caches only resume from a live
        snapshot boundary, so restoring past it would attach KV to a node with no restore
        point -- the same truncation HybridRadixCache.match_prefix applies."""
        node = self._root
        prefix = 0
        path: List["RadixTreeNode"] = []
        while node is not None and prefix < len(ids):
            child = node.children.get(self._key_fn(ids[prefix:]))
            if child is None:
                break
            m = align_down(child.get_match_len(ids[prefix:]), self._config.page_size)
            if m == 0 or m < child.length:
                break
            if self._cm.is_hybrid and child.persist_mamba_off is None:
                break  # tombstoned snapshot: no resume point at this boundary
            child.persist_start = prefix
            prefix += m
            child.persist_end = prefix
            path.append(child)
            node = child
        return prefix, path

    # ------------------------------------------------------------------ gdn layout
    def _gdn_layout(self) -> dict:
        pool = self._lsp
        conv = int(np.prod(pool.conv_states[:, 0].shape, dtype=np.int64)) * pool.conv_states.element_size()
        rec = int(np.prod(pool.recurrent_states[:, 0].shape, dtype=np.int64)) * pool.recurrent_states.element_size()
        state_bytes = {
            name: int(np.prod(t[:, 0].shape, dtype=np.int64)) * t.element_size()
            for name, t in pool.slot_states.items()
        }
        return {
            "slot_bytes": conv + rec + sum(state_bytes.values()),
            "conv_bytes": conv,
            "rec_bytes": rec,
            "slot_state_names": list(pool.slot_states.keys()),
            "slot_state_bytes": state_bytes,
            "num_slots": 0,
            "conv_dtype": str(pool.conv_states.dtype),
            "rec_dtype": str(pool.recurrent_states.dtype),
            "slot_state_dtypes": {n: str(t.dtype) for n, t in pool.slot_states.items()},
        }


# ------------------------------------------------------------------ tree walkers
def _preorder_nodes(cache) -> List:
    """Iterative pre-order node list (a deep session can exceed the recursion limit).
    Attaches ``node._persist_parent`` = list index of the node's parent (None for root
    children); consistent with the load-side parent lookup."""
    root = cache.root if hasattr(cache, "root") else cache.root_node
    out: List = []
    stack: List[Tuple[object, "None | int"]] = [
        (child, None) for child in reversed(list(root.children.values()))
    ]
    while stack:
        node, parent_idx = stack.pop()
        node._persist_parent = parent_idx
        out.append(node)
        idx = len(out) - 1
        for child in reversed(list(node.children.values())):
            stack.append((child, idx))
    return out


def _unique_pages(nodes, page_size: int) -> List[int]:
    pages: set[int] = set()
    for n in nodes:
        v = n.value
        if v.numel():
            pages.update(int(t) // page_size for t in v)
    return sorted(pages)


def _page_runs(pages: List[int]) -> Iterator[Tuple[int, int]]:
    """(start, end) inclusive runs of consecutive page numbers, for one big cpu copy."""
    i = 0
    while i < len(pages):
        start = pages[i]
        end = start
        while i + 1 < len(pages) and pages[i + 1] == end + 1:
            i += 1
            end += 1
        yield start, end
        i += 1


def _unique_gdn_slots(nodes) -> List[int]:
    slots = {int(n.mamba_value) for n in nodes if n.mamba_value is not None}
    return sorted(slots)


def _gdn_slot_bytes(pool: "LinearStatePool", slot: int) -> bytes:
    parts = [
        pool.conv_states[:, slot],
        pool.recurrent_states[:, slot],
    ]
    parts.extend(t[:, slot] for t in pool.slot_states.values())
    return b"".join(_cpu_bytes(p) for p in parts)


__all__ = ["CachePersister", "default_kv_cache_dir"]