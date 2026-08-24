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


# --------------------------------------------------------------------------------------
# MoE ("qwen35moe") adapter
# --------------------------------------------------------------------------------------


def _scalar_ref_dequant_q4_k(raw: torch.Tensor) -> list[float]:
    """Independent port of ggml's ``dequantize_row_q4_K`` (one loop per sub-block pair)."""
    import struct

    out: list[float] = []
    for b in range(raw.shape[0]):
        blk = raw[b].tolist()
        (d,) = struct.unpack("<e", bytes(blk[0:2]))
        (dmin,) = struct.unpack("<e", bytes(blk[2:4]))
        scales, qs = blk[4:16], blk[16:]

        def get_sc_min(j):
            if j < 4:
                return scales[j] & 63, scales[j + 4] & 63
            return (
                (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
                (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
            )

        y = [0.0] * 256
        is_i, q_i = 0, 0
        for j in range(0, 256, 64):
            sc1, mn1 = get_sc_min(is_i)
            sc2, mn2 = get_sc_min(is_i + 1)
            d1, mm1 = d * sc1, dmin * mn1
            d2, mm2 = d * sc2, dmin * mn2
            for l in range(32):
                y[j + l] = d1 * (qs[q_i + l] & 0xF) - mm1
            for l in range(32):
                y[j + 32 + l] = d2 * (qs[q_i + l] >> 4) - mm2
            q_i += 32
            is_i += 2
        out.extend(y)
    return out


def test_dequant_q4_k_matches_scalar_reference():
    """The vectorized Q4_K reference must agree with a literal ggml loop port: ssm_beta/
    ssm_alpha arrive Q4_K on MoE checkpoints and flow through it."""
    import struct

    from freetoken.models.gguf.dequant import GGML_Q4_K, dequantize

    torch.manual_seed(11)
    nb = 7
    raw = torch.randint(0, 256, (nb, 144), dtype=torch.uint8)
    # f16-representable positive scales / mins so both paths see identical inputs.
    for b in range(nb):
        raw[b, 0:2] = torch.tensor(
            list(struct.pack("<e", 0.25 + 0.125 * (b % 5))), dtype=torch.uint8
        )
        raw[b, 2:4] = torch.tensor(list(struct.pack("<e", 0.0625 * (b % 3))), dtype=torch.uint8)

    got = dequantize(raw.reshape(-1), GGML_Q4_K, torch.float32)
    want = torch.tensor(_scalar_ref_dequant_q4_k(raw))
    assert got.dtype == torch.float32 and got.shape == (nb * 256,)
    assert torch.allclose(got, want, atol=1e-4, rtol=1e-4)


def test_parse_qwen35moe_config_expert_fields(monkeypatch):
    """All-MoE geometry + native-packed expert facts travel into ModelConfig."""
    from freetoken.models.gguf.config import GgufConfigShim
    from freetoken.models.qwen3_5_moe import gguf as qg

    m = {
        "qwen35moe.block_count": 5,
        "qwen35moe.embedding_length": 64,
        "qwen35moe.attention.head_count": 8,
        "qwen35moe.attention.head_count_kv": 2,
        "qwen35moe.attention.key_length": 32,
        "qwen35moe.attention.value_length": 32,
        "qwen35moe.attention.layer_norm_rms_epsilon": 1e-5,
        "qwen35moe.rope.dimension_count": 16,
        "qwen35moe.rope.freq_base": 10000000.0,
        "qwen35moe.context_length": 4096,
        "qwen35moe.ssm.group_count": 4,
        "qwen35moe.ssm.time_step_rank": 8,
        "qwen35moe.ssm.inner_size": 128,
        "qwen35moe.ssm.state_size": 32,
        "qwen35moe.ssm.conv_kernel": 4,
        "qwen35moe.expert_count": 16,
        "qwen35moe.expert_used_count": 4,
        "qwen35moe.expert_feed_forward_length": 48,
        "qwen35moe.expert_shared_feed_forward_length": 24,
        # feed_forward_length intentionally absent (all layers are MoE; value would be 0).
    }
    shim = GgufConfigShim(
        architectures=["Qwen35MoeGGUFForCausalLM"],
        model_path="dummy.gguf",
        model_type="qwen35moe",
        metadata=m,
        vocab_size=99,
        tie_word_embeddings=False,
    )
    monkeypatch.setattr(qg, "_moe_gguf_types", lambda path: (12, 14))

    cfg = qg.parse_qwen35moe_gguf_config(shim)

    assert cfg.num_layers == 5
    assert cfg.moe_enabled and cfg.num_experts == 16 and cfg.num_experts_per_tok == 4
    assert cfg.moe_intermediate_size == 48
    assert cfg.shared_expert_intermediate_size == 24
    assert cfg.intermediate_size == 48  # no dense MLP: falls back to the MoE width
    assert cfg.norm_topk_prob
    assert cfg.expert_quant == "q4_0" and cfg.moe_weight_format == "q4_0"
    assert cfg.expert_gguf_types == (12, 14)
    full = {i for grp in cfg.attention_groups if grp.name == "full" for i in grp.layer_ids}
    linear = {i for grp in cfg.attention_groups if grp.name == "linear" for i in grp.layer_ids}
    assert full == {3} and linear == {0, 1, 2, 4}  # interval 4 over 5 layers
    g = cfg.linear_attention_group()
    assert g.num_key_heads == 4 and g.num_value_heads == 8  # nk, nv (= time_step_rank)
    assert g.value_head_dim == 128 // 8


def test_q40_expert_bank_specs_use_checkpoint_types():
    """Bank row bytes follow each bank's own ggml type (Q4_K gate/up vs Q6_K down here)."""
    from types import SimpleNamespace

    from freetoken.models.qwen3_5_moe.gguf import _q4_0_expert_specs

    cfg = SimpleNamespace(
        num_experts=3,
        hidden_size=2048,
        moe_intermediate_size=512,
        expert_gguf_types=(12, 14),
    )
    specs = _q4_0_expert_specs(cfg)
    assert specs["gate_up"][0] == (3, 1024, 2048 // 256 * 144)  # Q4_K rows
    assert specs["down"][0] == (3, 2048, 512 // 256 * 210)  # Q6_K rows
    assert all(s[1] == torch.uint8 for s in specs.values())


def test_registry_and_tokenizer_wiring_for_moe():
    import importlib

    from freetoken.models.register import get_model_spec
    from freetoken.models.gguf.config import GGUF_ARCH_TO_REGISTRY
    from freetoken.models.gguf.tokenizer import _TOKENIZER_ARCH

    spec = get_model_spec("Qwen35MoeGGUFForCausalLM")
    module = importlib.import_module(spec.module)
    assert callable(getattr(module, spec.model_cls))
    assert callable(getattr(module, spec.parse_config))
    assert callable(getattr(module, spec.iter_weights))
    # The offload provider resolves these hooks through the same module namespace.
    assert callable(getattr(module, "load_q4_0_expert_sources"))
    assert callable(getattr(module, "dummy_q4_0_expert_sources"))

    assert GGUF_ARCH_TO_REGISTRY["qwen35moe"] == "Qwen35MoeGGUFForCausalLM"
    assert _TOKENIZER_ARCH["qwen35moe"] == "qwen3"


def test_moe_layer_receives_gguf_types_via_extra_attrs(monkeypatch):
    """Qwen3_5MoE forwards config.expert_gguf_types onto the experts layer -- the fused
    kernel dispatch reads them back with getattr."""
    from types import SimpleNamespace

    from freetoken.distributed import set_tp_info

    set_tp_info(rank=0, size=1)
    from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE

    cfg = SimpleNamespace(
        num_experts=4,
        num_experts_per_tok=2,
        hidden_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=8,
        norm_topk_prob=True,
        moe_backend="offload",
        expert_quant="q4_0",
        expert_gguf_types=(12, 14),
    )
    captured = {}

    def fake_make(config, **kw):
        captured["kw"] = kw
        return SimpleNamespace()

    import freetoken.models.qwen3_5_moe.moe as moe_mod

    monkeypatch.setattr(moe_mod, "make_moe_layer", fake_make)
    Qwen3_5MoE(cfg, layer_id=0)
    kw = captured["kw"]
    assert kw["layer_id"] == 0 and kw["renormalize"] is True
    assert kw["extra_attrs"] == {"gguf_gate_up_type": 12, "gguf_down_type": 14}
