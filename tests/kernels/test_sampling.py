"""Unit tests for the exact top-k/top-p triton samplers' launch fallback bookkeeping.

The sampling kernels themselves need a GPU, so these cover the cooperative-launch
degrade path with a fake launcher and run anywhere.
"""
from __future__ import annotations

import pytest
import torch

from freetoken.kernel.triton import sampling as sampling_mod


# The exact AMD/ROCm failure mode: the driver requests cooperative launch via an
# assert (not a RuntimeError), so the fallback must treat it like any other.
_AMD_COOP_ASSERT = AssertionError("Cooperative launch requested but not supported by device")


@pytest.fixture(autouse=True)
def _clear_disabled():
    sampling_mod._COOPERATIVE_DISABLED.clear()
    yield
    sampling_mod._COOPERATIVE_DISABLED.clear()


def _fake_plan(B, V, device, force_single=False):
    return 2, 1


def test_assertion_from_missing_cooperative_launch_falls_back(monkeypatch):
    def _fake_launch(probs, kernel, tk, tp, draw, seed, offset, force_single=False):
        if not force_single:
            raise _AMD_COOP_ASSERT
        return "single-cta-result"

    monkeypatch.setattr(sampling_mod, "_fused_plan", _fake_plan)
    monkeypatch.setattr(sampling_mod, "_fused_launch", _fake_launch)

    probs = torch.randn(4, 32000)
    out = sampling_mod._exact_launch(probs, sampling_mod._topp_fused, None, 0.9, True, None, None)

    assert out == "single-cta-result"
    assert (probs.device, "topp", True) in sampling_mod._COOPERATIVE_DISABLED


def test_disabled_device_skips_cooperative_on_later_calls(monkeypatch):
    calls: list[bool] = []

    def _fake_launch(probs, kernel, tk, tp, draw, seed, offset, force_single=False):
        calls.append(force_single)
        return "ok"

    monkeypatch.setattr(sampling_mod, "_fused_plan", _fake_plan)
    monkeypatch.setattr(sampling_mod, "_fused_launch", _fake_launch)
    probs = torch.randn(4, 32000)
    sampling_mod._COOPERATIVE_DISABLED.add((probs.device, "topp", True))
    out = sampling_mod._exact_launch(probs, sampling_mod._topp_fused, None, 0.9, True, None, None)

    assert out == "ok"
    assert calls == [True]


def test_unrelated_assertion_propagates(monkeypatch):
    def _fake_launch(probs, kernel, tk, tp, draw, seed, offset, force_single=False):
        raise AssertionError("something else entirely")

    monkeypatch.setattr(sampling_mod, "_fused_plan", _fake_plan)
    monkeypatch.setattr(sampling_mod, "_fused_launch", _fake_launch)

    probs = torch.randn(4, 32000)
    with pytest.raises(AssertionError, match="something else entirely"):
        sampling_mod._exact_launch(probs, sampling_mod._topp_fused, None, 0.9, True, None, None)
    assert sampling_mod._COOPERATIVE_DISABLED == set()