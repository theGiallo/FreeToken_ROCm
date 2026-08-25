"""Fused router top-k + renormalize-softmax for decode-sized batches.

Replaces the pure-torch chain (``torch.topk`` + k-wide ``softmax`` + dtype/
contiguity casts -- 4-5 kernel launches per MoE layer per token, ~60 launches
per generated token at 36 layers) with ONE launch: each program scores one
token row, iteratively extracts the top-k expert ids, and renormalizes the
selected weights with a k-wide softmax. Exact match against the torch
reference (softmax-then-renormalize cancels the global denominator, so top-k
on raw logits + k-wide softmax IS the HF routing semantics).
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# Measurement escape hatch: force the torch chain back on.
_FORCED = os.getenv("FREETOKEN_TORCH_TOPK", "").strip().lower() in {"1", "true", "yes", "on"}


@triton.jit(do_not_specialize=["stride_row"])
def _router_topk_kernel(
    logits_ptr,
    weights_ptr,
    ids_ptr,
    stride_row,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK_E)
    emask = offs < E
    x = tl.load(logits_ptr + t * stride_row + offs, mask=emask, other=float("-inf")).to(tl.float32)

    k_range = tl.arange(0, BLOCK_K)
    k_mask = k_range < K
    vals = tl.zeros((BLOCK_K,), dtype=tl.float32) - float("inf")
    ids  = tl.zeros((BLOCK_K,), dtype=tl.int32)
    for i in tl.static_range(BLOCK_K):
        v   = tl.max(x, axis=0)
        idx = tl.argmax(x, axis=0).to(tl.int32)
        x   = tl.where(offs == idx, float("-inf"), x)
        vals = tl.where(k_range == i, v, vals)
        ids  = tl.where(k_range == i, idx, ids)

    # Renormalize over the selected k only (the global softmax denominator cancels).
    vals_safe = tl.where(k_mask, vals, float("-inf"))
    m = tl.max(vals_safe, axis=0)
    e = tl.exp(vals_safe - m)
    w = e / tl.sum(e, axis=0)

    out_off = t * K + k_range
    tl.store(weights_ptr + out_off, tl.where(k_mask, w, 0.0))
    tl.store(ids_ptr     + out_off, tl.where(k_mask, ids, -1))


def _pow2(n: int) -> int:
    return triton.next_power_of_2(n)


def router_topk_supported(gating_output: torch.Tensor, topk: int) -> bool:
    """Whether :func:`router_topk_softmax` can serve this call exactly."""
    if _FORCED or not gating_output.is_cuda or gating_output.dim() != 2:
        return False
    if gating_output.dtype not in (torch.bfloat16, torch.float32):
        return False
    num_experts = gating_output.shape[1]
    return 1 <= topk <= 16 and num_experts <= 4096


def router_topk_softmax(gating_output: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k expert ids (int32) + renormalized softmax weights (fp32), one launch."""
    num_tokens, num_experts = gating_output.shape
    weights = torch.empty((num_tokens, topk), dtype=torch.float32, device=gating_output.device)
    ids = torch.empty((num_tokens, topk), dtype=torch.int32, device=gating_output.device)
    block_e = _pow2(num_experts)
    block_k = _pow2(topk)
    _router_topk_kernel[(num_tokens,)](
        gating_output,
        weights,
        ids,
        gating_output.stride(0),
        K=topk,
        BLOCK_K=block_k,
        E=num_experts,
        BLOCK_E=block_e,
        num_warps=4 if block_e >= 256 else 1,
    )
    return weights, ids
