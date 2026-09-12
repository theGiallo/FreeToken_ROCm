"""P2b integration: CacheManager hybrid path (match_req -> cache_req donate -> prefix hit).
CPU, real LinearStatePool + page_table, hand-built Reqs. Exercises the two-currency wiring
without the full scheduler/engine."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager


def _pool(num_slots=16):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)


def _pend(ids):
    # int32 to match production Req.input_ids dtype (fast_compare_key needs consistent dtype)
    t = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(input_ids=t, input_len=len(ids), mm_embeds=None)


def test_hybrid_cache_manager_donate_then_hit():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)
    assert cm.is_hybrid

    # cold match on an empty tree
    mr = cm.match_req(_pend([1, 2, 3, 4, 5]))
    assert mr.cuda_handle.cached_len == 0 and mr.mamba_value is None

    # admit req A: allocate live + ping-pong, stage KV pages, mark a ×N snapshot at boundary 4
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    page_table[0, :4] = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
    reqA = Req(input_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32), table_idx=0,
               cached_len=4, output_len=1, uid=0, sampling_params=SamplingParams(),
               cache_handle=mr.cuda_handle)
    reqA.linear_slot_idx, reqA.mamba_ping_pong = live, pp
    reqA.mamba_next_track_idx = 1            # flipped from 0 in build_fla_metadata; frozen = pp[0]
    reqA.mamba_last_track_seqlen = 4
    cm.lock(mr.cuda_handle)

    free_before = pool.num_free_slots
    cm.cache_req(reqA, finished=False)       # donate pp[0] at boundary 4; replace it in the pair
    # pp[0] donated to the tree; a fresh replacement was alloc'd -> net free-slot count unchanged
    assert pool.num_free_slots == free_before - 1  # one replacement alloc'd (donated slot now tree-owned)
    assert reqA.mamba_ping_pong[0] != pp[0]        # slot 0 replaced; pp[0] now lives in the tree

    # req B shares the [1,2,3,4] prefix -> HIT: returns the donated snapshot + reused KV
    mrB = cm.match_req(_pend([1, 2, 3, 4, 9]))
    assert mrB.cuda_handle.cached_len == 4
    assert mrB.mamba_value == pp[0]
    assert mrB.cuda_handle.get_matched_indices().tolist() == [100, 101, 102, 103]


def test_hybrid_finish_donates_live_slot():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)

    mr = cm.match_req(_pend([7, 8, 9, 10]))
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    page_table[1, :3] = torch.tensor([200, 201, 202], dtype=torch.int32)
    req = Req(input_ids=torch.tensor([7, 8, 9, 10], dtype=torch.int32), table_idx=1,
              cached_len=3, output_len=1, uid=1, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    cm.lock(mr.cuda_handle)

    cm.cache_req(req, finished=True)         # donate the live slot directly (final state)
    # ping-pong pair freed; live slot kept (now owned by the tree)
    mr2 = cm.match_req(_pend([7, 8, 9, 10]))
    assert mr2.cuda_handle.cached_len == 3 and mr2.mamba_value == live


def test_free_req_slots_idempotent():
    """C2: a finish/abort double-free of the same request must NOT push its GDN slots twice."""
    pool = _pool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    req = Req(input_ids=torch.tensor([1, 2, 3], dtype=torch.int32), table_idx=0, cached_len=2,
              output_len=1, uid=0, sampling_params=SamplingParams(), cache_handle=None)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    base = pool.num_free_slots
    cm._free_req_slots(req)
    assert pool.num_free_slots == base + 3        # live + 2 ping-pong returned once
    cm._free_req_slots(req)                        # second free (abort/finish race)
    assert pool.num_free_slots == base + 3         # idempotent: nothing pushed twice


def test_rebuild_reclaims_donated_gdn_slots():
    """C5: a runtime cache rebuild must return the discarded tree's GDN snapshot slots (idle)."""
    pool = _pool(num_slots=16)
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    mr = cm.match_req(_pend([7, 8, 9, 10]))
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    pt[1, :3] = torch.tensor([200, 201, 202], dtype=torch.int32)
    req = Req(input_ids=torch.tensor([7, 8, 9, 10], dtype=torch.int32), table_idx=1, cached_len=3,
              output_len=1, uid=1, sampling_params=SamplingParams(), cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    cm.lock(mr.cuda_handle)
    cm.cache_req(req, finished=True)              # donates `live` to the tree, frees ping-pong
    assert pool.num_free_slots < pool.num_slots - 1   # a slot is now tree-owned
    cm.rebuild(64, pt)                            # idle rebuild discards the tree
    assert pool.num_free_slots == pool.num_slots - 1  # all GDN slots reclaimed (no leak)


def test_prefill_chunk_ends_on_a_page_boundary():
    """A hybrid chunk must end page-aligned: the snapshot commit skips any other boundary."""
    from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool()
    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "hybrid_radix", linear_state_pool=pool)
    assert cm.prefill_chunk_align == 64
    tm = TableManager(max_running_reqs=4, page_table=pt)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))

    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    req = adder.try_add_one(pending)
    assert isinstance(req, ChunkedReq) and req.extend_len == 64

    # a budget below one page keeps the unaligned chunk rather than stalling the request
    adder = PrefillAdder(token_budget=40, reserved_size=0, cache_manager=cm, table_manager=tm)
    assert adder.try_add_one(pending).extend_len == 40


def test_naive_cache_does_not_align_prefill_chunks():
    """The alignment hook is hybrid-only; every other cache keeps the raw budget chunk."""
    from freetoken.scheduler.prefill import PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "radix")
    assert cm.prefill_chunk_align == 1
    tm = TableManager(max_running_reqs=4, page_table=pt)
    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))
    assert adder.try_add_one(pending).extend_len == 100


def test_last_msg_boundary_len_skips_generation_prompt():
    """The cap is the header of the LAST STORED message (the one a continuation re-renders),
    not the template's bare <|im_start|>assistant generation prompt that tails every request.
    Without a resolved user-role id the fallback is the deepest stored message header."""
    from freetoken.scheduler.scheduler import _last_msg_boundary_len

    im_start, im_end = 900, 901
    # ... stored assistant answer ... <|im_end|> <|im_start|>assistant  (generation prompt tail)
    ids = torch.tensor([0, im_start, 1, 2, im_end, im_start], dtype=torch.int32)
    assert _last_msg_boundary_len(ids, im_start, im_end) == 1
    # generation prompt only (no stored message before the tail): no cap
    ids2 = torch.tensor([im_start, 3], dtype=torch.int32)
    assert _last_msg_boundary_len(ids2, im_start, im_end) is None
    # no tokens at all after the last stored header: the stored answer precedes the gen prompt
    # and the header is the cap even when the answer is the entire body before the im_end
    ids3 = torch.tensor([77, im_start, 5, im_end, im_start, 8], dtype=torch.int32)
    assert _last_msg_boundary_len(ids3, im_start, im_end) == 1


def test_last_msg_boundary_len_ends_with_user_stays_deepest():
    """A prompt that ends with a top-level user turn re-renders as an exact extension, so the
    cap is the deepest stored message header -- the last user turn (the deeper assistant
    answer never sits past it on a user-terminated prompt)."""
    from freetoken.scheduler.scheduler import _last_msg_boundary_len

    im_start, im_end, user, assistant = 900, 901, 10, 20
    # user .. assistant .. user (stored) .. assistant (generation prompt, no <|im_end|>)
    ids = torch.tensor(
        [im_start, user, 5, im_end, im_start, assistant, 6, im_end,
         im_start, user, 9, im_end, im_start, assistant],
        dtype=torch.int32,
    )
    assert _last_msg_boundary_len(ids, im_start, im_end, user) == 8
    assert _last_msg_boundary_len(ids, im_start, im_end) == 8


def test_last_msg_boundary_len_ends_with_answer_caps_at_user():
    """A prompt whose last stored message is the model's answer must cap at the LAST TOP-LEVEL
    USER turn: a continuation re-renders the answer (the template adds a thinking prefix), so
    the streams diverge at the answer start, not its end."""
    from freetoken.scheduler.scheduler import _last_msg_boundary_len

    im_start, im_end, user, assistant = 900, 901, 10, 20
    # user .. assistant answer .. (generation prompt)
    ids = torch.tensor(
        [im_start, user, 5, im_end, im_start, assistant, 6, im_end, im_start, assistant],
        dtype=torch.int32,
    )
    assert _last_msg_boundary_len(ids, im_start, im_end, user) == 0


def test_last_msg_boundary_len_skips_tool_result_folds():
    """<|im_start|>user\n<tool_response> inside the answer's tool round is NOT a top-level
    user turn and must be skipped when looking for the cap (a fold sits AFTER the last real
    user query, i.e. past the divergence point)."""
    from freetoken.scheduler.scheduler import _last_msg_boundary_len

    im_start, im_end, user, assistant = 900, 901, 10, 20
    fold = [27, 13766, 9367, 29]
    # user .. assistant .. tool-fold(user) .. assistant answer .. (generation prompt)
    ids = torch.tensor(
        [im_start, user, 5, im_end, im_start, assistant, 6, im_end,
         im_start, user, 198, *fold, 7, im_end,
         im_start, assistant, 8, im_end, im_start, assistant],
        dtype=torch.int32,
    )
    assert _last_msg_boundary_len(ids, im_start, im_end, user, fold) == 0
    # without the fold ids the fold IS treated as a top-level user -> wrong (too deep)
    assert _last_msg_boundary_len(ids, im_start, im_end, user) == 8


def test_msg_boundary_donation_reused_by_diverging_continuation():
    """A prefill that donates at the message-boundary-clamped x64 boundary is reusable by a
    continuation that re-renders the trailing assistant answer (diverging a few tokens past the
    message start); the same request donating at the deepest x64 boundary is NOT (that snapshot
    sits past the divergence, unreachable by the walk-up)."""
    from freetoken.attention.linear import _track_chunk

    CHUNK = 64
    # pre-prefill mirror of the store req: cached_len=4, extend 260, last <|im_start|> at 150.
    # The re-render diverges just past 150, i.e. strictly between the clamped (132) and the
    # deepest (260) boundaries.
    base = list(range(260))
    diverged = base[:150] + [9999, 9998, 9997]

    def donate_and_hit(cm_cls_boundary, expect_cached):
        pool = _pool()
        pt = torch.zeros(4, 512, dtype=torch.int32)
        cm = CacheManager(CHUNK, 1, pt, "hybrid_radix", linear_state_pool=pool)
        assert cm.is_hybrid

        # cold admit A; complete_one already advanced cached_len to the post-prefill 260
        mr = cm.match_req(_pend(base))
        live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
        pt[0, :260] = torch.arange(3000, 3260, dtype=torch.int32)
        reqA = Req(input_ids=torch.tensor(base + [0], dtype=torch.int32), table_idx=0,
                   cached_len=260, output_len=1, uid=0, sampling_params=SamplingParams(),
                   cache_handle=mr.cuda_handle)
        reqA.linear_slot_idx, reqA.mamba_ping_pong = live, pp
        reqA.mamba_next_track_idx = 1              # frozen = pp[0] (flipped from 0 pre-forward)
        reqA.mamba_last_track_seqlen = cm_cls_boundary
        cm.lock(mr.cuda_handle)
        cm.cache_req(reqA, finished=False)         # donate pp[0] at the chosen boundary

        # continuation B diverges a few tokens past the message start (150)
        mrB = cm.match_req(_pend(diverged))
        assert mrB.cuda_handle.cached_len == expect_cached, (
            mrB.cuda_handle.cached_len, expect_cached)
        if expect_cached > 0:
            assert mrB.mamba_value == pp[0]        # the donated snapshot is restored

    # clamped clamp -> deepest x64 <= 150
    c = _track_chunk(SimpleNamespace(extend_len=260, cached_len=4, mamba_msg_boundary=150), CHUNK)
    donate_and_hit(4 + c * CHUNK, expect_cached=4 + c * CHUNK)   # donates at 132, B HITs at 132
    # un-clamped (flag off) -> deepest boundary 260, past the divergence at ~150 -> MISS
    donate_and_hit(4 + (260 - 1) // CHUNK * CHUNK, expect_cached=0)


def test_track_chunk_clamps_donation_to_message_boundary():
    """gdn_message_boundary_snapshots: the snapshot donation boundary is the deepest xCHUNK at
    or below the last message start (a continuation re-renders the trailing assistant answer
    and diverges just past it), NOT the deepest boundary of the whole extend."""
    from freetoken.attention.linear import _track_chunk

    # no boundary (disabled) / boundary in an earlier chunk: keep the deepest mid-extend pick
    plain = SimpleNamespace(extend_len=260, cached_len=4, mamba_msg_boundary=None)
    assert _track_chunk(plain, 64) == 4                 # 4 + 4*64 = 260
    earlier = SimpleNamespace(extend_len=260, cached_len=4, mamba_msg_boundary=4)
    assert _track_chunk(earlier, 64) == 4
    # a boundary inside the first chunk (<64 past cached_len) has no usable snap point: deepest
    first_chunk = SimpleNamespace(extend_len=260, cached_len=4, mamba_msg_boundary=60)
    assert _track_chunk(first_chunk, 64) == 4
    # clamped: deepest x64 <= msg boundary 150 -> 4 + 2*64 = 132
    clamped = SimpleNamespace(extend_len=260, cached_len=4, mamba_msg_boundary=150)
    assert _track_chunk(clamped, 64) == 2
    # message start that itself lands on a boundary: deepest <= it
    on_bound = SimpleNamespace(extend_len=260, cached_len=4, mamba_msg_boundary=132)
    assert _track_chunk(on_bound, 64) == 2
    # extend shorter than a full chunk: no donatable boundary anywhere
    short = SimpleNamespace(extend_len=60, cached_len=0, mamba_msg_boundary=None)
    assert _track_chunk(short, 64) == 0


def test_msg_boundary_donation_reused_by_diverging_continuation():
    """A prefill that donates at the message-boundary-clamped x64 boundary is reusable by a
    continuation that re-renders the trailing assistant answer (diverging a few tokens past the
    message start); the same request donating at the deepest x64 boundary is NOT (that snapshot
    sits past the divergence, unreachable by the walk-up)."""
    from freetoken.attention.linear import _track_chunk

    CHUNK = 64
    # pre-prefill mirror of the store req: cached_len=4, extend 260, last <|im_start|> at 150.
    # The re-render diverges just past 150, i.e. strictly between the clamped (132) and the
    # deepest (260) boundaries.
    base = list(range(260))
    diverged = base[:150] + [9999, 9998, 9997]

    def donate_and_hit(cm_cls_boundary, expect_cached):
        pool = _pool()
        pt = torch.zeros(4, 512, dtype=torch.int32)
        cm = CacheManager(CHUNK, 1, pt, "hybrid_radix", linear_state_pool=pool)
        assert cm.is_hybrid

        # cold admit A; complete_one already advanced cached_len to the post-prefill 260
        mr = cm.match_req(_pend(base))
        live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
        pt[0, :260] = torch.arange(3000, 3260, dtype=torch.int32)
        reqA = Req(input_ids=torch.tensor(base + [0], dtype=torch.int32), table_idx=0,
                   cached_len=260, output_len=1, uid=0, sampling_params=SamplingParams(),
                   cache_handle=mr.cuda_handle)
        reqA.linear_slot_idx, reqA.mamba_ping_pong = live, pp
        reqA.mamba_next_track_idx = 1              # frozen = pp[0] (flipped from 0 pre-forward)
        reqA.mamba_last_track_seqlen = cm_cls_boundary
        cm.lock(mr.cuda_handle)
        cm.cache_req(reqA, finished=False)         # donate pp[0] at the chosen boundary

        # continuation B diverges a few tokens past the message start (150)
        mrB = cm.match_req(_pend(diverged))
        assert mrB.cuda_handle.cached_len == expect_cached, (
            mrB.cuda_handle.cached_len, expect_cached)
        if expect_cached > 0:
            assert mrB.mamba_value == pp[0]        # the donated snapshot is restored

    # clamped clamp -> deepest x64 <= 150
    c = _track_chunk(SimpleNamespace(extend_len=260, cached_len=4, mamba_msg_boundary=150), CHUNK)
    donate_and_hit(4 + c * CHUNK, expect_cached=4 + c * CHUNK)   # donates at 132, B HITs at 132
    # un-clamped (flag off) -> deepest boundary 260, past the divergence at ~150 -> MISS
    donate_and_hit(4 + (260 - 1) // CHUNK * CHUNK, expect_cached=0)


def test_long_prompt_donation_survives_finish_donate():
    """Reproduce the chunked 85K-token scenario: a request prefill-donates at message-boundary
    84800, then the finish path live-donates at its post-generation cached_len (85505), then a
    byte-identical re-send must still match the 84800 snapshot via the walk-up."""
    pool = _pool(num_slots=24)
    pt = torch.zeros(4, 90000, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    assert cm.is_hybrid

    base_len = 85272
    dec_end = 85505
    boundary = 84800
    prompt = torch.arange(base_len, dtype=torch.int32)       # the byte-identical re-send
    full = torch.arange(dec_end, dtype=torch.int32)          # prompt + generated tail

    pt[0, :dec_end] = torch.arange(3000, 3000 + dec_end, dtype=torch.int32)

    mr = cm.match_req(_pend(prompt))
    assert mr.cuda_handle.cached_len == 0
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    # final chunk is constructed at chunk start, then complete_one() advances cached_len
    req = Req(input_ids=full, table_idx=0, cached_len=81920, output_len=300, uid=0,
              sampling_params=SamplingParams(), cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    req.mamba_next_track_idx = 1
    req.cached_len = base_len
    req.mamba_last_track_seqlen = boundary
    cm.lock(mr.cuda_handle)
    cm.cache_req(req, finished=False)              # donate the clamped-boundary frozen snapshot
    assert req.mamba_last_track_seqlen is None

    # decode to 85505, then finish: live-insert extends the tree past the donated node
    req.cached_len = dec_end
    cm.cache_req(req, finished=True)

    # byte-identical re-send: must reuse the 84800 donation via the walk-up
    mr2 = cm.match_req(_pend(prompt))
    assert mr2.cuda_handle.cached_len == boundary, (
        "re-send lost the message-boundary donation", mr2.cuda_handle.cached_len)
    assert mr2.mamba_value is not None


def test_pool_sizing_covers_4mr_floor():
    """C6: pool must reserve the 4-slot-per-request non-evictable floor even at a tiny ratio."""
    from types import SimpleNamespace
    from freetoken.kvcache.linear_state_pool import _linear_pool_num_slots
    for mr in (1, 8, 64):
        c = SimpleNamespace(max_running_req=mr, cache_type="hybrid_radix",
                            linear_state_cache_ratio=0.1)
        assert _linear_pool_num_slots(c) >= 4 * mr + 1, (mr, _linear_pool_num_slots(c))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
