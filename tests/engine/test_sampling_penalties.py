"""Unit tests for the frequency/repeat/min_p sampling paths (CPU-only, no triton).

Companion to tests/engine/test_presence_penalty.py — presence, frequency and repeat all
act on the same per-row occurrence-count matrix, and min_p is a torch-side filter applied
inside sample_impl after softmax.
"""
from __future__ import annotations

import torch
import pytest

import freetoken.core as core
from freetoken.engine import sample as sample_mod
from freetoken.engine.sample import BatchSamplingArgs, Sampler


def _req(input_ids, sampling_params):
    return core.Req(
        input_ids=input_ids,
        table_idx=0,
        cached_len=0,
        output_len=10,
        uid=0,
        sampling_params=sampling_params,
        cache_handle=None,
    )


def _batch(reqs):
    return core.Batch(reqs=reqs, phase="decode")


# ---------------------------------------------------------------------------
# counts matrix
# ---------------------------------------------------------------------------
def test_counts_matrix():
    s = Sampler(device=torch.device("cpu"), vocab_size=8)
    r = _req(torch.tensor([1, 2, 2, 3, 1]), core.SamplingParams(presence_penalty=0.5))
    counts = s._counts(_batch([r]))
    assert counts.dtype == torch.int32
    assert counts.shape == (1, 8)
    assert counts[0].tolist() == [0, 2, 2, 1, 0, 0, 0, 0]


def test_counts_multiple_rows():
    s = Sampler(device=torch.device("cpu"), vocab_size=5)
    reqs = [
        _req(torch.tensor([0, 0, 1]), core.SamplingParams(frequency_penalty=0.5)),
        _req(torch.tensor([4]), core.SamplingParams(frequency_penalty=0.5)),
    ]
    counts = s._counts(_batch(reqs))
    assert counts[0].tolist() == [2, 1, 0, 0, 0]
    assert counts[1].tolist() == [0, 0, 0, 0, 1]


# ---------------------------------------------------------------------------
# prepare() gating
# ---------------------------------------------------------------------------
def test_no_penalty_when_zeros_and_defaults():
    s = Sampler(device=torch.device("cpu"), vocab_size=8)
    r = _req(torch.tensor([1, 2]), core.SamplingParams(temperature=0.7))
    args = s.prepare(_batch([r]))
    assert args.counts is None
    assert args.presence_penalties is None
    assert args.frequency_penalties is None
    assert args.repeat_penalties is None


def test_prepare_builds_all_penalty_tensors():
    s = Sampler(device=torch.device("cpu"), vocab_size=8)
    r = _req(
        torch.tensor([1, 1, 2]),
        core.SamplingParams(
            temperature=0.7,
            presence_penalty=0.5,
            frequency_penalty=0.3,
            repeat_penalty=1.2,
            min_p=0.05,
        ),
    )
    args = s.prepare(_batch([r]))
    assert args.counts is not None
    assert args.presence_penalties.tolist() == pytest.approx([0.5])
    assert args.frequency_penalties.tolist() == pytest.approx([0.3])
    assert args.repeat_penalties.tolist() == pytest.approx([1.2])
    assert args.min_p.tolist() == pytest.approx([0.05])


def test_repeat_penalty_disabled_at_one_does_not_trigger():
    s = Sampler(device=torch.device("cpu"), vocab_size=8)
    r = _req(torch.tensor([1, 2]), core.SamplingParams(repeat_penalty=1.0, temperature=0.7))
    args = s.prepare(_batch([r]))
    assert args.counts is None  # no penalty active


def test_min_p_triggers_without_penalty():
    s = Sampler(device=torch.device("cpu"), vocab_size=8)
    r = _req(torch.tensor([1, 2]), core.SamplingParams(temperature=0.7, min_p=0.05))
    args = s.prepare(_batch([r]))
    assert args.counts is None
    assert args.min_p.tolist() == pytest.approx([0.05])


def test_greedy_fastpath_only_when_no_penalty_and_no_min_p():
    s = Sampler(device=torch.device("cpu"), vocab_size=8)
    # plain greedy -> fast path (no tensors)
    args = s.prepare(_batch([_req(torch.tensor([1]), core.SamplingParams())]))
    assert args.temperatures is None
    assert args.counts is None
    # greedy + min_p -> must NOT take fast path (min_p neutral on greedy but keep consistency)
    args2 = s.prepare(_batch([_req(torch.tensor([1]), core.SamplingParams(min_p=0.1))]))
    assert args2.counts is None
    assert args2.min_p is not None
    # greedy + repeat -> penalties present
    args3 = s.prepare(_batch([_req(torch.tensor([1]), core.SamplingParams(repeat_penalty=1.3))]))
    assert args3.counts is not None


# ---------------------------------------------------------------------------
# frequency penalty application
# ---------------------------------------------------------------------------
def test_frequency_penalty_applied(monkeypatch):
    s = Sampler(device=torch.device("cpu"), vocab_size=4)
    r = _req(torch.tensor([0, 0, 1]), core.SamplingParams(frequency_penalty=0.5, temperature=0.7))
    args = s.prepare(_batch([r]))

    captured = {}

    def fake_sample_impl(logits, temps, top_k, top_p, min_p=None):
        captured["logits"] = logits.clone()
        return torch.argmax(logits, dim=-1)

    monkeypatch.setattr(sample_mod, "sample_impl", fake_sample_impl)

    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])  # counts for 0,1,2,3 = [2,1,0,0]
    out = s.sample(logits, args)
    post = captured["logits"][0]
    # token0: -0.5*2 = -1.0 ; token1: -0.5*1 = -0.5 ; token2,3 unchanged
    assert post.tolist() == [-1.0, 0.5, 2.0, 3.0]
    assert out.tolist() == [3]


# ---------------------------------------------------------------------------
# repeat penalty (llama.cpp multiplicative semantics)
# ---------------------------------------------------------------------------
def test_repeat_penalty_math(monkeypatch):
    s = Sampler(device=torch.device("cpu"), vocab_size=4)
    r = _req(torch.tensor([0, 1]), core.SamplingParams(repeat_penalty=2.0, temperature=0.7))
    args = s.prepare(_batch([r]))

    captured = {}

    def fake_sample_impl(logits, temps, top_k, top_p, min_p=None):
        captured["logits"] = logits.clone()
        return torch.argmax(logits, dim=-1)

    monkeypatch.setattr(sample_mod, "sample_impl", fake_sample_impl)

    # tokens 0,1 seen; 2,3 unseen. token0 logit>0 -> /2 ; token1 logit<0 -> *2
    logits = torch.tensor([[4.0, -4.0, 8.0, -8.0]])
    out = s.sample(logits, args)
    post = captured["logits"][0]
    assert post.tolist() == [2.0, -8.0, 8.0, -8.0]
    assert out.tolist() == [2]


def test_repeat_penalty_zero_logit_untouched(monkeypatch):
    s = Sampler(device=torch.device("cpu"), vocab_size=3)
    r = _req(torch.tensor([0]), core.SamplingParams(repeat_penalty=2.0, temperature=0.7))
    args = s.prepare(_batch([r]))

    captured = {}

    def fake_sample_impl(logits, temps, top_k, top_p, min_p=None):
        captured["logits"] = logits.clone()
        return torch.argmax(logits, dim=-1)

    monkeypatch.setattr(sample_mod, "sample_impl", fake_sample_impl)
    logits = torch.tensor([[0.0, 1.0, 2.0]])  # token0 logit==0 -> unchanged (0*2/2 stays 0)
    s.sample(logits, args)
    assert captured["logits"][0].tolist() == [0.0, 1.0, 2.0]


def test_repeat_penalty_greedy(monkeypatch):
    s = Sampler(device=torch.device("cpu"), vocab_size=3)
    r = _req(torch.tensor([0]), core.SamplingParams(repeat_penalty=3.0))  # greedy
    args = s.prepare(_batch([r]))
    assert args.counts is not None

    # greedy argmax after /3 on token0: [3, 4, 2] -> [1, 4, 2] -> argmax picks token1
    logits = torch.tensor([[3.0, 4.0, 2.0]])
    out = s.sample(logits, args)
    assert out.tolist() == [1]


# ---------------------------------------------------------------------------
# min_p (probability-domain filter) — tests the real _apply_min_p helper
# ---------------------------------------------------------------------------
def test_min_p_keeps_above_threshold():
    # logits [0,5,4.9]: exp ratio to max (token1) is e^-5≈0.0067 (token0) and e^-0.1≈0.905
    # (token2). At min_p 0.5: token0 dropped, token1 & token2 kept.
    probs = torch.softmax(torch.tensor([[0.0, 5.0, 4.9]]), dim=-1)
    un = sample_mod._apply_min_p(probs, 0.5)
    assert (un[0] > 0).tolist() == [False, True, True]


def test_min_p_high_threshold_only_top():
    probs = torch.softmax(torch.tensor([[0.0, 5.0, 4.9]]), dim=-1)
    un = sample_mod._apply_min_p(probs, 0.95)
    # only the argmax (token1, ratio 1.0) survives at min_p 0.95
    assert (un[0] > 0).tolist() == [False, True, False]
    assert (un[0] >= 0).all()


def test_min_p_none_and_zero_are_noop():
    probs = torch.softmax(torch.tensor([[0.0, 5.0, 4.9]]), dim=-1)
    assert torch.allclose(sample_mod._apply_min_p(probs, None), probs)
    assert torch.allclose(sample_mod._apply_min_p(probs, 0.0), probs)


def test_min_p_batched():
    probs = torch.softmax(torch.tensor([[0.0, 5.0, 4.9], [5.0, 4.0, 3.0]]), dim=-1)
    mp = torch.tensor([0.5, 0.0], dtype=torch.float32)
    un = sample_mod._apply_min_p(probs, mp)
    assert (un[0] > 0).tolist() == [False, True, True]   # min_p 0.5 active for row0
    assert (un[1] > 0).all()                              # min_p 0 for row1 -> no filter


def test_min_p_filter_passes_to_sample_impl(monkeypatch):
    s = Sampler(device=torch.device("cpu"), vocab_size=8)
    r = _req(torch.tensor([1, 2]), core.SamplingParams(temperature=0.7, min_p=0.25))
    args = s.prepare(_batch([r]))

    captured = {}

    def fake_sample_impl(logits, temps, top_k, top_p, min_p=None):
        captured["min_p"] = min_p
        return torch.argmax(logits, dim=-1)

    monkeypatch.setattr(sample_mod, "sample_impl", fake_sample_impl)
    s.sample(torch.zeros((1, 8)), args)
    assert captured["min_p"] is not None
    assert captured["min_p"].tolist() == pytest.approx([0.25])
