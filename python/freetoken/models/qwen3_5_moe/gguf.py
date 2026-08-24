"""Qwen3.5/3.6 (dense) GGUF adapter: build the FreeToken ``ModelConfig`` from GGUF
metadata and stream the checkpoint's native-quantized weights onto the model.

The GGUF geometry matches the HF dense qwen3_5 model (hybrid GatedDeltaNet / gated
full attention with ``full_attention_interval``), so this produces the same
``ModelConfig`` as ``qwen3_5_moe.config.parse_config`` -- only the source is GGUF KV
metadata. Differences from the HF path that this adapter absorbs:

* llama.cpp's converter folds the MTP draft head into ``block_count``
  (``block_count == num_hidden_layers + nextn_predict_layers``) and stores its
  decoder block as plain ``blk.<N>.*`` tensors next to ``blk.<N>.nextn.*``. The base
  model is the first ``block_count - nextn_predict_layers`` blocks; everything at or
  above that index is skipped here.
* The full-attention projections are gated (the q rows carry a per-head sigmoid gate)
  and arrive as separate ``attn_q``/``attn_k``/``attn_v`` tensors, often with mixed
  ggml types -- fused FreeToken buffers are built from per-type parts instead of one
  concatenated packed tensor (see :class:`GGUFMergedLinear`).
* The converter reorders the linear-attention V heads from HF's grouped-by-K-head
  order to ggml's tiled order in the ``attn_qkv``/``attn_gate`` v/z rows,
  ``ssm_alpha``/``ssm_beta``/``ssm_a``/``ssm_dt.bias``, ``ssm_conv1d`` channels and
  ``ssm_out`` input columns. The fla kernels expect the grouped order, so every one of
  those is permuted back on load. All of those permute whole packed rows / fp32 lanes
  except ``ssm_out``, whose input columns live *inside* Q5_K superblocks: it is
  dequantized (ggml kernel), permuted and requantized to Q8_0 once at load.
* ``ssm_a`` stores ``-exp(A_log)`` rather than A_log itself; the HF parametrization is
  restored via ``log(-x)``.
* Gemma-style (1+weight) norms are stored pre-incremented by the converter, matching
  what the HF loader bakes in, so they pass through unchanged; the GDN ``norm`` is a
  plain RMS norm and also passes through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.layers.base import BaseOP
from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)
from freetoken.models.gguf.dequant import GGML_F32, GGML_Q8_0, dequantize, row_bytes

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


# --------------------------------------------------------------------------------------
# Config parsing
# --------------------------------------------------------------------------------------


def parse_qwen35_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    m = shim.metadata

    def g(key):
        val = m.get(f"qwen35.{key}")
        if val is None:
            raise KeyError(f"missing GGUF metadata key qwen35.{key}")
        return val

    def gopt(key, default):
        return m.get(f"qwen35.{key}", default)

    block_count = int(g("block_count"))
    # The MTP draft head is folded into block_count; the base stack excludes it.
    num_layers = block_count - int(gopt("nextn_predict_layers", 0))
    assert num_layers > 0, f"empty base stack (block_count={block_count})"
    interval = int(gopt("full_attention_interval", 4))
    # Recurrent layers = every non-interval layer of the base stack (the MTP tail, when
    # present, is always a dense full-attention block).
    full_layer_ids = tuple(i for i in range(num_layers) if (i + 1) % interval == 0)
    linear_layer_ids = tuple(i for i in range(num_layers) if (i + 1) % interval != 0)
    assert full_layer_ids, "no full-attention layers"

    hidden = int(g("embedding_length"))
    full_head_dim = int(g("attention.key_length"))
    assert int(g("attention.value_length")) == full_head_dim, "asymmetric full-attn head dims"

    rotary = RotaryConfig(
        head_dim=full_head_dim,
        rotary_dim=int(g("rope.dimension_count")),
        max_position=int(g("context_length")),
        base=float(g("rope.freq_base")),
        scaling=None,
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_layer_ids,
        num_kv_heads=int(g("attention.head_count_kv")),
        head_dim=full_head_dim,
        rotary_config=rotary,
    )
    num_key_heads = int(g("ssm.group_count"))
    num_value_heads = int(g("ssm.time_step_rank"))
    value_head_dim = int(g("ssm.inner_size")) // num_value_heads
    assert num_value_heads * value_head_dim == int(g("ssm.inner_size")), "bad ssm.inner_size"
    key_head_dim = int(g("ssm.state_size"))  # converter writes linear_key_head_dim here
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_layer_ids,
        num_key_heads=num_key_heads,
        num_value_heads=num_value_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        conv_kernel_dim=int(g("ssm.conv_kernel")),
        output_gate=True,
    )
    groups = tuple(sorted((full_group, linear_group), key=lambda grp: grp.layer_ids[0]))

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=int(g("attention.head_count")),
        num_kv_heads=int(g("attention.head_count_kv")),
        head_dim=full_head_dim,
        hidden_size=hidden,
        vocab_size=int(shim.vocab_size),
        intermediate_size=int(g("feed_forward_length")),
        hidden_act="silu",
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        rotary_config=rotary,
        num_experts=int(gopt("expert_count", 0)),
        num_experts_per_tok=int(gopt("expert_used_count", 0)),
        moe_intermediate_size=int(gopt("expert_feed_forward_length", 0)),
        norm_topk_prob=bool(gopt("norm_topk_prob", False)),
        moe_enabled=False,
        use_qk_norm=True,
        model_type=str(shim.model_type),
        architectures=list(shim.architectures),
        vision_config=None,
        attention_groups=groups,
        gguf_source_path=shim.model_path,
    )


# --------------------------------------------------------------------------------------
# Weight loading: GGUF tensor names -> FreeToken qwen3_5_moe module params.
# --------------------------------------------------------------------------------------


def _require_tp1(what: str) -> None:
    """GGUF quant layers are not sharded; reject TP>1 with a clear error (mirrors the
    HF qwen3_5_moe loader's TP=1 restriction)."""
    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        raise NotImplementedError(f"qwen35 GGUF {what} currently supports TP=1 only.")


def _tiled_to_grouped_head_index(num_v_heads: int, num_key_heads: int) -> torch.Tensor:
    """Inverse of llama.cpp's V-head tile reorder: grouped index -> tiled storage index.

    The converter reshapes the ``nv`` heads ``[nk, r]`` and swaps the axes, so tiled
    position ``t`` holds grouped head ``(t % nk) * r + t // nk``; inverting gives
    ``t(gv) = (gv % r) * nk + gv // r``.
    """
    r = num_v_heads // num_key_heads
    assert num_key_heads * r == num_v_heads, "num_value_heads must be a multiple of num_key_heads"
    return torch.tensor(
        [(gv % r) * num_key_heads + gv // r for gv in range(num_v_heads)], dtype=torch.long
    )


def _row_gather_index(head_index: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Expand a per-head permutation into a row-level index over packed weight rows."""
    return (head_index * head_dim).repeat_interleave(head_dim) + torch.arange(head_dim).repeat(
        len(head_index)
    )


def _channel_gather_index(head_index: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Expand a per-head permutation into an element-level channel/column index."""
    n = len(head_index) * head_dim
    return head_index.repeat_interleave(head_dim) * head_dim + torch.arange(n) % head_dim


def _to_bf16(t) -> torch.Tensor:
    """Dequantize a GgufTensor (F32 here) to a dense bf16 tensor of its torch shape."""
    flat = dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16)
    return flat.reshape(t.shape)


def _f32(t) -> torch.Tensor:
    """Materialize an F32 GgufTensor as a float tensor of its torch shape."""
    assert t.ggml_type == GGML_F32, f"{t.name}: expected F32, got type {t.ggml_type}"
    return t.packed().reshape(-1).view(torch.float32).reshape(t.shape)


def _requant_q8_0(w: torch.Tensor, device) -> torch.Tensor:
    """Quantize a float ``[out, in]`` matrix to native Q8_0 packed bytes.

    Mirrors ggml's ``quantize_row_q8_0``: per 32-element block, ``d = amax/127`` stored
    as fp16 followed by the rounded int8 codes.
    """
    out_features, in_features = w.shape
    assert in_features % 32 == 0
    blocks = w.to(device).reshape(out_features, in_features // 32, 32).float()
    d = (blocks.abs().amax(dim=-1, keepdim=True) / 127.0).to(torch.float16).float()
    d = torch.where(d > 0, d, torch.ones_like(d))  # all-zero blocks quantize to zero codes
    q = torch.round(blocks / d).clamp_(-127, 127).to(torch.int8)
    raw = torch.empty(out_features, in_features // 32, 34, dtype=torch.uint8, device=device)
    raw[:, :, :2] = d.to(torch.float16).view(torch.uint8)
    raw[:, :, 2:] = q.view(torch.uint8)
    return raw.reshape(out_features, in_features // 32 * 34)


def iter_qwen35_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (param_name, tensor) for every dense qwen3_5 param.

    Quantized projections stay in their native packed layout and are yielded as
    ``.qweight`` (uint8); F32 norms dequantize to bf16 (A_log/dt_bias stay fp32). Fused
    FreeToken projections are assembled from per-ggml-type parts (``part{i}.qweight`` /
    ``part{i}.weight``) since the checkpoint mixes types inside one logical projection.
    All linear-attention tensors are permuted out of ggml's tiled V-head order back
    into the HF grouped order the fla kernels expect; ``ssm_out`` additionally goes
    through a dequant -> permute -> requantize-Q8_0 pass because its permutation axis
    runs across packed superblocks.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.utils import cached_load_hf_config

    # Dense checkpoint: there are no routed experts, so ``include_moe_experts`` carries
    # no meaning here (the engine passes True for any non-offload model).
    assert include_non_moe
    _require_tp1("weight loading")

    config = parse_qwen35_gguf_config(cached_load_hf_config(model_path))
    g = config.linear_attention_group()
    assert g is not None
    k_dim = g.num_key_heads * g.key_head_dim
    v_dim = g.num_value_heads * g.value_head_dim
    hd = g.value_head_dim
    v2g = _tiled_to_grouped_head_index(g.num_value_heads, g.num_key_heads)
    v_rows = _row_gather_index(v2g, hd)  # z/v packed-row permutation
    v_cols = _channel_gather_index(v2g, hd)  # conv/out_proj channel permutation

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "token_embd.weight":
            yield "model.embed_tokens.qweight", t.packed()
            continue
        if name == "output_norm.weight":
            yield "model.norm.weight", _to_bf16(t)
            continue
        if name == "output.weight":
            yield "lm_head.qweight", t.packed()
            continue
        if not name.startswith("blk."):
            continue

        layer = int(name.split(".")[1])
        if layer >= config.num_layers:
            continue  # MTP draft block (plain + .nextn.* tensors): not part of the base stack
        suffix = name.split(".", 2)[2]
        base = f"model.layers.{layer}"

        if suffix == "attn_norm.weight":
            yield f"{base}.input_layernorm.weight", _to_bf16(t)
        elif suffix == "post_attention_norm.weight":
            yield f"{base}.post_attention_layernorm.weight", _to_bf16(t)

        elif suffix == "attn_q_norm.weight":
            yield f"{base}.self_attn.q_norm.weight", _to_bf16(t)
        elif suffix == "attn_k_norm.weight":
            yield f"{base}.self_attn.k_norm.weight", _to_bf16(t)
        elif suffix in ("attn_q.weight", "attn_k.weight", "attn_v.weight"):
            part = {"attn_q": 0, "attn_k": 1, "attn_v": 2}[suffix[:6]]
            yield f"{base}.self_attn.qkv_proj.part{part}.qweight", t.packed()
        elif suffix == "attn_output.weight":
            yield f"{base}.self_attn.o_proj.qweight", t.packed()

        elif suffix == "ffn_gate.weight":
            yield f"{base}.mlp.gate_up_proj.part0.qweight", t.packed()
        elif suffix == "ffn_up.weight":
            yield f"{base}.mlp.gate_up_proj.part1.qweight", t.packed()
        elif suffix == "ffn_down.weight":
            yield f"{base}.mlp.down_proj.qweight", t.packed()

        elif suffix == "attn_qkv.weight":
            rows = t.packed()  # [q | k | v_tiled] along the output dim
            qk, v_tiled = rows[: 2 * k_dim], rows[2 * k_dim :]
            yield f"{base}.linear_attn.in_proj.part0.qweight", torch.cat(
                [qk, v_tiled.index_select(0, v_rows)], dim=0
            ).contiguous()
        elif suffix == "attn_gate.weight":
            yield f"{base}.linear_attn.in_proj.part1.qweight", t.packed().index_select(0, v_rows)
        elif suffix == "ssm_beta.weight":
            yield f"{base}.linear_attn.in_proj.part2.weight", _f32(t).index_select(0, v2g).to(
                torch.bfloat16
            )
        elif suffix == "ssm_alpha.weight":
            yield f"{base}.linear_attn.in_proj.part3.weight", _f32(t).index_select(0, v2g).to(
                torch.bfloat16
            )

        elif suffix == "ssm_conv1d.weight":
            # Channels run [q | k | v]: q/k pass through, only the trailing v-head
            # channels carry the tile order.
            conv_idx = torch.cat(
                [torch.arange(2 * k_dim), v_cols + 2 * k_dim]
            )
            chans = _f32(t).index_select(0, conv_idx)  # [conv_dim, K] in torch order
            yield f"{base}.linear_attn.conv1d.weight", chans.unsqueeze(1).contiguous().to(
                torch.bfloat16
            )
        elif suffix == "ssm_dt.bias":
            yield f"{base}.linear_attn.dt_bias", _f32(t).index_select(0, v2g).contiguous()
        elif suffix == "ssm_a":
            # Stored as -exp(A_log); restore the HF parametrization the kernel expects.
            a_log = torch.log(_f32(t).neg()).index_select(0, v2g).contiguous()
            yield f"{base}.linear_attn.A_log", a_log
        elif suffix == "ssm_norm.weight":
            yield f"{base}.linear_attn.norm.weight", _to_bf16(t)
        elif suffix == "ssm_out.weight":
            from freetoken.kernel.gguf import ggml_dequantize

            # The ggml dequant kernel runs on-device: host->device once here, permute,
            # requantize -- the bf16/fp32 dense matrix never lands back on the host.
            w = ggml_dequantize(
                t.packed().to(device, non_blocking=True),
                t.ggml_type,
                config.hidden_size,
                v_dim,
                torch.float32,
            )
            w = w.index_select(1, v_cols.to(device))  # group the input (V-head) columns
            yield f"{base}.linear_attn.out_proj.qweight", _requant_q8_0(w, device)
        else:
            raise ValueError(f"unmapped qwen35 GGUF tensor: {name}")


# --------------------------------------------------------------------------------------
# Native-GGUF module swaps
# --------------------------------------------------------------------------------------


def is_qwen35_gguf_model(config: ModelConfig) -> bool:
    """True when the model was parsed from a qwen35 GGUF checkpoint (native-quant path)."""
    return "Qwen35GGUFForCausalLM" in getattr(config, "architectures", [])


class _DensePart(BaseOP):
    """bf16 output-part of :class:`GGUFMergedLinear` (unquantized rows, e.g. b|a)."""

    def __init__(self, out_features: int, in_features: int):
        self.out_features = out_features
        self.weight = torch.empty(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


class GGUFMergedLinear(BaseOP):
    """Output-concatenated projection whose parts carry different ggml types.

    The qwen35 GGUF splits each fused projection across tensors that rarely share a
    quant type (q/k Q4_K vs v Q5_K; GDN qkv Q5_K + gate Q4_K + bf16 b/a), so instead of
    one concatenated packed tensor there is one native-quant GEMM per part. Parts are
    named ``part0..partN`` so the generic state-dict walk routes each ``.qweight`` /
    ``.weight``; TP=1 only (like every GGUF layer).
    """

    def __init__(self, in_features: int, parts: list[tuple[int, int | None]]):
        from freetoken.layers.gguf import GGUFLinear

        self._mods: list[BaseOP] = []
        for i, (out_features, quant_type) in enumerate(parts):
            mod: BaseOP
            if quant_type is None:
                mod = _DensePart(out_features, in_features)
            else:
                mod = GGUFLinear(in_features, out_features, quant_type)
            setattr(self, f"part{i}", mod)
            self._mods.append(mod)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [mod.forward(x) for mod in self._mods]
        return torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]


class GGUFLMHead(BaseOP):
    """Untied LM head over a native block-quantized ``output.weight``.

    Mirrors ``ParallelLMHead.forward`` at TP=1: slice to the last token per sequence at
    prefill, then logits via the ggml matmul kernels. Owns its ``qweight`` (loaded via
    the generic walk), unlike gemma4's tied head which aliases the embedding table.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, quant_type: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._quant_type = quant_type
        self.qweight = torch.empty(
            num_embeddings, row_bytes(embedding_dim, quant_type), dtype=torch.uint8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx
        from freetoken.layers.gguf import fused_mul_mat_gguf

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        return fused_mul_mat_gguf(x, self.qweight, self._quant_type)


def convert_qwen35_to_gguf(model, config: ModelConfig) -> None:
    """In place: replace the dense projections + embedding + lm_head with native GGUF ops.

    Swapped (packed, dequantized only inside the ggml kernels): attention qkv/o, MLP
    gate/up/down, GDN in_proj parts and out_proj, the token embedding (Q4_K here) and
    the untied lm_head (Q6_K). Left dense: every norm plus the small fp32 GDN params.
    Part layouts must match :func:`iter_qwen35_gguf_weights`.
    """
    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear
    from freetoken.models.gguf.reader import iter_gguf_tensors

    path = config.gguf_source_path
    assert path is not None, "GGUF model config lacks gguf_source_path"
    types = {t.name: t.ggml_type for t in iter_gguf_tensors(path)}

    g = config.linear_attention_group()
    assert g is not None
    k_dim = g.num_key_heads * g.key_head_dim
    v_dim = g.num_value_heads * g.value_head_dim
    nv = g.num_value_heads
    hidden = config.hidden_size
    inter = config.intermediate_size
    q_gate = config.num_qo_heads * config.head_dim * 2
    kv = config.num_kv_heads * config.head_dim

    inner = model.model
    inner.embed_tokens = GGUFEmbedding(
        num_embeddings=config.vocab_size,
        embedding_dim=hidden,
        quant_type=types["token_embd.weight"],
    )
    if config.tie_word_embeddings:
        raise NotImplementedError("tied embeddings are not supported on the qwen35 GGUF path")
    model.lm_head = GGUFLMHead(config.vocab_size, hidden, types["output.weight"])

    for layer in inner.layers.op_list:
        lid = layer._layer_id
        if layer._is_linear:
            attn = layer.linear_attn
            attn.in_proj = GGUFMergedLinear(
                hidden,
                [
                    (2 * k_dim + v_dim, types[f"blk.{lid}.attn_qkv.weight"]),
                    (v_dim, types[f"blk.{lid}.attn_gate.weight"]),
                    (nv, None),
                    (nv, None),
                ],
            )
            # out_proj arrives as Q5_K with its permutation running across packed
            # superblocks -> requantized to Q8_0 by the iterator.
            attn.out_proj = GGUFLinear(v_dim, hidden, GGML_Q8_0)
        else:
            attn = layer.self_attn
            attn.qkv_proj = GGUFMergedLinear(
                hidden,
                [
                    (q_gate, types[f"blk.{lid}.attn_q.weight"]),
                    (kv, types[f"blk.{lid}.attn_k.weight"]),
                    (kv, types[f"blk.{lid}.attn_v.weight"]),
                ],
            )
            attn.o_proj = GGUFLinear(q_gate // 2, hidden, types[f"blk.{lid}.attn_output.weight"])
        mlp = layer.mlp
        mlp.gate_up_proj = GGUFMergedLinear(
            hidden,
            [
                (inter, types[f"blk.{lid}.ffn_gate.weight"]),
                (inter, types[f"blk.{lid}.ffn_up.weight"]),
            ],
        )
        mlp.down_proj = GGUFLinear(inter, hidden, types[f"blk.{lid}.ffn_down.weight"])


__all__ = [
    "parse_qwen35_gguf_config",
    "iter_qwen35_gguf_weights",
    "convert_qwen35_to_gguf",
    "is_qwen35_gguf_model",
    "GGUFMergedLinear",
    "GGUFLMHead",
]
