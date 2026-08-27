"""Unit tests for the presence-penalty path in the batch sampler (CPU-only, no triton)."""
from __future__ import annotations

import importlib
import sys

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


def test_presence_mask_distinct_tokens():
    s = Sampler(device=torch.device("cpu"), vocab_size=8)
    r = _req(
        torch.tensor([1, 2, 2, 3, 1]),
        core.SamplingParams(presence_penalty=0.5),
    )
    mask = s._presence_mask(_batch([r]))
    assert mask.dtype == torch.bool
    assert mask.shape == (1, 8)
    assert list(mask[0]) == [False, True, True, True, False, False, False, False]


def test_no_penalty_when_zero():
    s = Sampler(device=torch.device("cpu"), vocab_size=8)
    r = _req(
        torch.tensor([1, 2]),
        core.SamplingParams(presence_penalty=0.0, temperature=0.7, top_p=0.9),
    )
    args = s.prepare(_batch([r]))
    assert args.presence_penalties is None
    assert args.presence_mask is None


def test_prepare_builds_penalty_and_mask():
    s = Sampler(device=torch.device("cpu"), vocab_size=8)
    reqs = [
        _req(torch.tensor([1, 2]), core.SamplingParams(presence_penalty=0.5, temperature=0.7)),
        _req(torch.tensor([2, 2, 7]), core.SamplingParams(presence_penalty=1.0, temperature=0.7)),
    ]
    args = s.prepare(_batch(reqs))
    assert args.presence_penalties is not None
    assert args.presence_mask is not None
    assert args.presence_penalties.tolist() == [0.5, 1.0]
    assert list(args.presence_mask[0]) == [False, True, True, False, False, False, False, False]
    assert list(args.presence_mask[1]) == [False, False, True, False, False, False, False, True]


def test_sample_applies_penalty_before_sampling(monkeypatch):
    s = Sampler(device=torch.device("cpu"), vocab_size=4)
    r = _req(
        torch.tensor([0, 1, 1, 2]),
        core.SamplingParams(presence_penalty=0.5, temperature=0.7, top_p=0.9),
    )
    args = s.prepare(_batch([r]))

    captured = {}

    def fake_sample_impl(logits, temps, top_k, top_p):
        captured["logits"] = logits.clone()
        # canonical greedy-ish: return index of max post-penalty logit
        return torch.argmax(logits, dim=-1)

    monkeypatch.setattr(sample_mod, "sample_impl", fake_sample_impl)

    # logits [1, 4]. Present tokens: 0, 1, 2 -> each gets -0.5; token 3 untouched.
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    out = s.sample(logits, args)

    post = captured["logits"][0]
    assert post.tolist() == [-0.5, 0.5, 1.5, 3.0]
    # token 3 has the highest post-penalty logit -> chosen
    assert out.tolist() == [3]


def test_greedy_path_still_applies_penalty(monkeypatch):
    s = Sampler(device=torch.device("cpu"), vocab_size=4)
    r = _req(
        torch.tensor([0, 1, 2]),
        core.SamplingParams(presence_penalty=1.0),  # temperature 0 -> greedy
    )
    args = s.prepare(_batch([r]))
    assert args.presence_penalties is not None  # penalty still captured for greedy rows

    captured = {}

    def fake_sample_impl(logits, temps, top_k, top_p):
        captured["logits"] = logits.clone()
        return torch.argmax(logits, dim=-1)

    monkeypatch.setattr(sample_mod, "sample_impl", fake_sample_impl)

    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    out = s.sample(logits, args)
    # After -1.0 on tokens 0,1,2: [-1, 0, 1, 3] -> argmax picks token 3
    assert out.tolist() == [3]
    assert captured["logits"][0].tolist() == [-1.0, 0.0, 1.0, 3.0]
