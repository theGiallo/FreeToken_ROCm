"""qwen35 GGUF adapter: pure-torch pieces that don't need a checkpoint or GPU.

The V-head tile reorder is the one transform llama.cpp's converter applies to the
linear-attention weights (everything else is a rename); getting its inverse wrong
silently scrambles heads, so the round-trip against the converter's own permute is
pinned here. The Q8_0 requantizer feeds ssm_out (its permutation runs across packed
superblocks), so its byte layout must match ggml's quantize_row_q8_0 exactly.
"""
from __future__ import annotations

import torch


def test_v_head_tile_reorder_inverse_round_trips():
    """grouped -> (converter tile reorder) -> (adapter inverse) == grouped."""
    from freetoken.models.qwen3_5_moe.gguf import _tiled_to_grouped_head_index

    for nk in (1, 2, 4, 16):
        for r in (1, 2, 3, 4):
            nv = nk * r
            # The converter reshapes [nk, r] and swaps axes; flat pos j*nk+i holds
            # grouped head i*r+j.
            tiled = [i * r + j for j in range(r) for i in range(nk)]
            v2g = _tiled_to_grouped_head_index(nv, nk)
            restored = [tiled[int(v2g[t])] for t in range(nv)]
            assert restored == list(range(nv)), (nk, r, restored)


def test_row_and_channel_gather_indices_expand_head_permutation():
    from freetoken.models.qwen3_5_moe.gguf import (
        _channel_gather_index,
        _row_gather_index,
        _tiled_to_grouped_head_index,
    )

    hd, nv, nk = 4, 6, 3
    v2g = _tiled_to_grouped_head_index(nv, nk)
    rows = _row_gather_index(v2g, hd)
    cols = _channel_gather_index(v2g, hd)
    assert rows.shape == (nv * hd,) and cols.shape == (nv * hd,)
    # rows: per-head contiguous blocks, one block per head in v2g order.
    assert rows.tolist() == (v2g.repeat_interleave(hd) * hd + torch.arange(hd).repeat(nv)).tolist()
    # cols: element-level expansion of the same head permutation.
    assert cols.tolist()[:hd] == (int(v2g[0]) * hd + torch.arange(hd)).tolist()
    # conv1d-style passthrough: q/k channels first, then offset v channels.
    k_dim = 8
    conv_idx = torch.cat([torch.arange(2 * k_dim), cols + 2 * k_dim])
    assert conv_idx.shape == (2 * k_dim + nv * hd,)
    assert (conv_idx[: 2 * k_dim] == torch.arange(2 * k_dim)).all()


def test_q8_0_requant_round_trip_matches_reference_dequant():
    from freetoken.models.gguf.dequant import GGML_Q8_0, dequantize
    from freetoken.models.qwen3_5_moe.gguf import _requant_q8_0

    torch.manual_seed(7)
    w = torch.randn(5, 96) * 0.3
    packed = _requant_q8_0(w, torch.device("cpu"))
    assert packed.dtype == torch.uint8 and packed.shape == (5, 96 // 32 * 34)
    back = dequantize(packed.reshape(-1), GGML_Q8_0, torch.float32).reshape(5, 96)
    # Per-block d=amax/127 fp16 scale + rounded int8 codes: tight but not exact.
    assert (back - w).abs().max().item() < 0.02


def test_registry_exports_resolve():
    """The GGUF spec's parse/iter functions are exported by the package (the generic
    registry test resolves them via the module, not the package __all__ -- this pins
    both views)."""
    import importlib

    from freetoken.models.register import get_model_spec

    spec = get_model_spec("Qwen35GGUFForCausalLM")
    module = importlib.import_module(spec.module)
    assert callable(getattr(module, spec.model_cls))
    assert callable(getattr(module, spec.parse_config))
    assert callable(getattr(module, spec.iter_weights))

    from freetoken.models.gguf.config import GGUF_ARCH_TO_REGISTRY

    assert GGUF_ARCH_TO_REGISTRY["qwen35"] == "Qwen35GGUFForCausalLM"


def test_tokenizer_arch_maps_to_transformers_converter_key():
    from freetoken.models.gguf.tokenizer import _TOKENIZER_ARCH

    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    assert _TOKENIZER_ARCH["qwen35"] in GGUF_TO_FAST_CONVERTERS
