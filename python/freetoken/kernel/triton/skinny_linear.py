"""Skinny (M==1) bf16/fp16 dense linear via a Triton GEMV.

rocBLAS (via ``aten::mm`` / ``torch.mv``) picks a fat tiled Cijk GEMM kernel for
skinny shapes on gfx1100 -- measured ~157us for a ``[1, K] @ [K, 1]`` product whose
weight is 4 KB (should be ~2us memory-bound). Decode hits these shapes every layer
through replicated linears (MoE router gate, shared-expert gate) and merged
col/row-parallel projections, so the bad pick costs milliseconds per token.

This module provides a bandwidth-bound fallback for the M == 1 case: one Triton
program per BLOCK_N outputs, streaming K in BLOCK_K chunks with an fp32
accumulator. Dispatch lives in :func:`skinny_linear_forward`; anything that is not
a CUDA bf16/fp16 single-row matvec falls through to ``F.linear`` unchanged.
Set ``FREETOKEN_SKINNY_LINEAR=0`` to force the stock path.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gemv_kernel(
    x_ptr, w_ptr, y_ptr, bias_ptr,
    N, K,
    stride_k,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        xv = tl.load(x_ptr + offs_k * stride_k, mask=mask_k, other=0.0).to(tl.float32)
        wv = tl.load(
            w_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :], other=0.0,
        ).to(tl.float32)
        acc += tl.sum(wv * xv[None, :], axis=1)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs_n, acc.to(y_ptr.dtype.element_ty), mask=mask_n)


def skinny_linear_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
    """``x @ weight.T (+ bias)`` for ``x.shape[0] == 1``; falls back to ``F.linear``."""
    if os.environ.get("FREETOKEN_SKINNY_LINEAR", "1") == "0":
        return F.linear(x, weight, bias)
    if (
        x.shape[0] != 1
        or not x.is_cuda
        or x.dtype not in (torch.bfloat16, torch.float16)
        or weight.dtype != x.dtype
    ):
        return F.linear(x, weight, bias)

    K = x.shape[-1]
    O = weight.shape[0]
    if weight.stride(1) != 1 or (weight.shape[1] if weight.dim() > 1 else K) != K or weight.dim() != 2:
        return F.linear(x, weight, bias)

    xc = x if x.is_contiguous() else x.contiguous()
    y = torch.empty(1, O, device=x.device, dtype=x.dtype)
    # Tiny-O shapes (shared-expert gate: O == 1) are latency-bound if launched as a
    # single narrow program -- widen the K chunk instead so the loop stays short
    # (~4 iterations for K=2048); wide-O shapes get a register-safe 64x128 tile.
    block_n = min(64, max(2, triton.next_power_of_2(O)))
    block_k = 512 if block_n <= 4 else 128
    _gemv_kernel[(triton.cdiv(O, block_n),)](
        xc, weight, y,
        bias if bias is not None else xc,  # dead pointer when unused
        O, K,
        xc.stride(-1),
        HAS_BIAS=bias is not None,
        BLOCK_N=block_n,
        BLOCK_K=min(block_k, triton.next_power_of_2(max(block_k, K))),
        num_warps=4,
    )
    return y
