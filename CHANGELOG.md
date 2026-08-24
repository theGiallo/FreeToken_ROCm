# Changelog

All notable changes to FreeToken are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **HIP port of the GGUF quant kernels** (`mmvq`, `mmq`, MoE, dequantize): the
  vendored llama.cpp-derived CUDA sources now JIT-build with `hipcc` on ROCm,
  enabling quantized GGUF inference on AMD RDNA3 GPUs (gfx1100 tested). Includes a
  small header shim, 64-bit warp-shuffle masks for wave64, source staging onto a
  native filesystem for WSL2 `/mnt/*` builds, and `PYTORCH_ROCM_ARCH` pinning so JIT
  builds compile only the detected GPU target (~1 min instead of ~1 h).
- **Qwen3.8-27B GGUF support** (`Qwen35GGUFForCausalLM`, arch `qwen35`): loads
  `*.gguf` checkpoints of the dense GDN-hybrid model directly — no conversion step.
  Handles MTP draft-layer exclusion, the interval-4 full/GDN layer pattern, V-head
  tile-order permutation, `ssm_out` requantization (permutation crosses packed
  superblocks, so it is dequantized, permuted, and re-packed as Q8_0), and an untied
  GGUF LM head. The tokenizer maps `qwen35` onto the existing `qwen3` BPE converter.
- K-quant support in the engine dispatch tables: `Q4_K` / `Q5_K` constants, block
  shapes, MMVQ/MMQ/dequant entries, and a torch reference `dequant_q8_0`.
- `ModelConfig.gguf_source_path` records the originating `.gguf` file for post-init
  conversion hooks.
- `tests/models/test_qwen35_gguf_adapter.py`: permutation round-trips, requant
  layout, registry/tokenizer wiring.
- **Qwen3.6-35B-A3B MoE GGUF support** (`Qwen35MoeGGUFForCausalLM`, arch
  `qwen35moe`): loads Ollama/llama.cpp GGUF checkpoints of the routed-MoE GDN
  hybrid directly. Every decoder layer is a 256-expert top-8 MoE with a gated
  shared expert; the router and shared-expert gate dequantize to bf16, the shared
  expert keeps native packed projections, and the routed experts stay packed and
  stream through the offload banks: `expert_quant="q4_0"` reuses the native-GGUF
  bank format while new `ModelConfig.expert_gguf_types` carries each bank's actual
  ggml type (these checkpoints mix Q4_K gate/up with Q6_K down) end-to-end into
  `ggml_moe_a8_vec` dispatch. Also handles per-layer KV-head arrays (0 on GDN
  layers), bare `ssm_dt`/`ssm_a` tensor names, Q4_K `ssm_beta`/`ssm_alpha`
  (row-permuted while still packed), and MTP tensors under the separate `mtp.*`
  prefix.
- Torch reference `dequant_q4_k` in the GGUF dequant module (vectorized port of
  ggml's `dequantize_row_q4_K`, pinned against a scalar loop port in tests).
- The ROCm quantized-checkpoint guard now admits `expert_quant="q4_0"` (native
  GGUF K-quants run on the HIP-ported kernels; NVFP4/MXFP4/fp8 remain rejected).
- **CUDA graphs are now enabled by default on ROCm.** Profiling showed eager decode
  was host-dispatch-bound (~2000 launches + ~192 small copies per token, GPU mostly
  idle); stream capture works cleanly on RX 7900 XTX / WSL2 / ROCm 7.1 at bs=1 and
  lifted Qwen3.6-35B-A3B offload decode from ~12-14 tok/s to 52-53 tok/s telemetry
  steady-state (~37.5 tok/s sustained over an 8k-token generation). Set
  `FREETOKEN_ROCM_GRAPHS=0` to restore the previous eager behavior.
- `_torch_fused_topk` fast path for `renormalize=True`: top-k on raw logits plus a
  k-wide softmax (softmax-then-renormalize cancels the global denominator), removing
  the full-vocab fp32 copy + softmax — ~60 fewer eager launches per token on the
  fallback router (Windows/ROCm). Expert ids verified bitwise-equal to the old path.
- Device-side `_invalidate_prefill_buffer` (`invalidate_slot_range` Triton kernel +
  CPU-reference fallback): the old boolean-mask indexing formulation hid a nonzero
  D2H per call — 40 queue-draining host syncs per prefill on Qwen3.6-35B. Short-bench
  prefill telemetry improved ~15-20% (~20 -> ~23.7 tok/s at 39 tokens); decode
  unchanged, as expected.

### Verified

- RX 7900 XTX under WSL2 (Ubuntu 26.04, ROCm 7.x, torch 2.11): coherent generation
  on Qwen3.8-27B-Q4_K_S.gguf at ~13 tok/s decode; full 851-key shape/dtype
  reconciliation against a meta-device build; adapter tests green.
- Qwen3.6-35B-A3B GGUF (23.9 GB, from the `qwen3.6:35b-a3b_128k` Ollama blob):
  all 613 offload-layout keys reconcile shape+dtype against a meta-device build;
  end-to-end greedy generation byte-identical across repeat runs, with and without
  CUDA graphs. With graphs (now default): short-prompt decode 33-38 tok/s wall /
  52-53 tok/s steady-state; long-context (517-token prompt, 8k max output)
  ~33.5 tok/s wall / ~37.5 tok/s decode-only, flat across context. Same
  prompt/settings on ollama 0.32.14 (embedded llama.cpp, model fully VRAM-resident):
  ~44-51 tok/s decode — remaining gap ~1.16x at long context. Full numbers and the
  profiling trail: `BENCHMARK_RESULTS.md`, `PERF_INVESTIGATION_PLAN.md`.

### Known limitations

- Default `max_running_req=4` exceeds the KV/state budget next to ~16 GB of weights
  on 24 GB cards (GDN hybrid-radix slots ≈ 154 MB/slot); use
  `--max-running-requests 1` for this model.
- On RDNA3 the Triton attention/GDN kernels fall back off matrix-core instructions
  (`no matching matrix core intrinsic` warnings during warmup) — measured impact at
  bs=1 is small (the GDN update kernel is 0.4 ms/token), but prefill kernels and
  multi-request decode likely still pay for it. The split-K flash-decode kernel's
  MFMA shape selection fails to compile on gfx1100 and serves via a slower fallback.
- CUDA-graph capture on ROCm is validated for the qwen3.5/qwen35moe GGUF paths with
  the triton attention backend; other model/backend combinations may still trip
  capture corners — use `FREETOKEN_ROCM_GRAPHS=0` if so.
- The MoE GGUF path requires `--moe-backend offload` (engine assertion); resident
  MoE layers only support bf16/fp8_block formats.
