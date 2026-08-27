# Changelog

All notable changes to FreeToken are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **`presence_penalty` is now actually applied** (was parsed-but-ignored no-op) for
  OpenAI/chat completions. Plumbed from the wire request through
  `SamplingParams.presence_penalty` (`core.py`) and `resolve_sampling`
  (`generation.py` / `openai_api.py`) into the batch sampler
  (`engine/sample.py`): `Sampler.prepare` builds a per-row boolean presence mask
  (one hit per *distinct* token already generated, from each request's sequence)
  and `Sampler.sample` subtracts the penalty from those logits **before**
  temperature/softmax/top-k/top-p. Also honored on the greedy (argmax) path.
  `frequency_penalty` is carried into `SamplingParams` for parity but is still a
  no-op; `min_p` likewise remains accepted-but-unapplied. Unit-tested in
  `tests/engine/test_presence_penalty.py`.

### Qwen3.6-35B-A3B tool-call parsing (drift dialects)

- **Tolerant non-streaming parser**: `Qwen3CoderDetector` (`function_call_parser.py`)
  now normalizes the drift dialects the model emits into the canonical grammar before
  parsing — a bare function tag (`<bash>` instead of `<function=bash>`) with bare
  (`<command>…</command>`) or canonical (`<parameter=command>…</parameter>`) parameters,
  with or without the `<tool_call>` wrapper. `has_tool_call` detects the bare/drift
  forms; `detect_and_parse` applies `_normalize_qwen35_drift`; the `_parse_tool_response`
  gate in `generation.py` relaxes to a tolerant `has_tool_call` fallback and unwraps the
  model's `standard_tool_calling` meta-call into the real tool name/args
  (`_unwrap_standard_tool_calling`). `args.py` maps the `qwen35` parser alias onto
  `Qwen3CoderDetector` (and receives qwen35 in `--tool-call-parser`).
- **Streaming drift handling** (`generation.py` `_generate_events_impl`): bare
  `<bash>`/`<parameter=>` blocks that stream out as plain text are now held
  (`drift_buf`) until a closing tag or EOS, then re-parsed with the tolerant one-shot
  parser and emitted as a real `tool_calls` delta instead of raw markup pasted into
  content; trailing `</tool_call>`/whitespace noise after a completed call is swallowed
  (`swallow_trailing_close`). Canonical streaming already yielded
  `finish_reason=tool_calls` with valid args (19/19, 48-run sweep clean of drift leaks).
- Test coverage: `tests/test_qwen35_drift.py` (14) + `tests/test_streaming_drift.py` (4).

### Verified

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
- **Skinny-linear Triton GEMV** (`kernel/triton/skinny_linear.py`, dispatched from
  `_LinearTPImpl.forward` for `x.shape[0]==1` CUDA bf16/fp16, kill switch
  `FREETOKEN_SKINNY_LINEAR=0`): rocBLAS routes every `[1,K]` replicated linear
  (MoE router gate, shared-expert gate) to a tiled Cijk GEMM costing ~138-157 us
  per call — for a 4 KB weight read — on gfx1100; identical across
  `F.linear`/`mm`/`mv`. The bandwidth-bound Triton matvec removes 5.7 ms/token of
  device time on Qwen3.6-35B-A3B decode: device-busy 18.12 -> 12.39 ms/tok, warm
  e2e 40.0 -> **52.68 tok/s** (+32%) at unchanged VRAM. Reference re-measurement
  (same file, ctx that fits VRAM) puts llama.cpp at 107.9 tok/s and ollama at
  103.5 — the previously reported ~50 tok/s reference was an artifact of the
   `_128k` tag forcing partial offload via `num_ctx=131072`.
- **Fused Triton router top-k** (`kernel/triton/router_topk.py`): replaces the
  pure-torch fallback chain (`torch.topk` + k-wide `softmax` + dtype/contiguity
  casts — ~5 kernel launches per MoE layer per generated token, ~180 at 36 layers)
  with a single Triton launch per token row: iterative argmax extraction of the
  top-k experts + inline renormalize-softmax. Kill switch `FREETOKEN_TORCH_TOPK=1`.
  Warm e2e on Qwen3.6-35B-A3B: 57.2 -> **59.5 tok/s** (+4.0%), steady decode
  windows 77-79 -> **83-84 tok/s** (~0.4 ms/tok saved on gfx1100).
- **Portable prefill hit-D2D miss path** (`OffloadMoeCache._copy_miss_rows_portable`):
  the prefill hit/miss split no longer requires `cudaMemcpyBatchAsync` (CUDA >= 13) —
  where that API is unavailable (ROCm, older CUDA, failed JIT build) miss rows cross
  PCIe as per-run sliced async `copy_` calls on the prefill copy stream (small banks
  copy whole-layer), and the feature degrades with an info log instead of disabling.
  `moe_prefill_hit_d2d` now defaults **on** (opt-out `--disable-moe-prefill-hit-d2d`);
  measured warm e2e on Qwen3.6-35B-A3B: 52.7 -> **57.2 tok/s** (+8.6%). Gain is
  bounded by near-zero prefill hit rate on short prompts (a 39-token prefill routes
  ~312 (token, expert) pairs per layer over 256 experts).

### Verified

- RX 7900 XTX under WSL2 (Ubuntu 26.04, ROCm 7.x, torch 2.11): coherent generation
  on Qwen3.8-27B-Q4_K_S.gguf at ~13 tok/s decode; full 851-key shape/dtype
  reconciliation against a meta-device build; adapter tests green.
- Qwen3.6-35B-A3B GGUF (23.9 GB, from the `qwen3.6:35b-a3b_128k` Ollama blob):
  all 613 offload-layout keys reconcile shape+dtype against a meta-device build;
  end-to-end greedy generation byte-identical across repeat runs, with and without
  CUDA graphs. With graphs + the skinny-linear GEMV + fused router top-k +
  prefill hit-D2D: short-prompt decode
  **44.8 tok/s wall cold / 59.5 warm** (was 35/40 before optimizations); steady
  decode windows **83-84 tok/s** (~12.0 ms/token device-busy). Three-way
  reference on identical workloads (see `BENCHMARK_RESULTS.md` UPDATE 2):
   llama.cpp b10615+PR#25334 **107.9 tok/s @ 21.6 GiB**, ollama **103.5 @ 22.7 GiB**,
   FreeToken **59.5 @ ~22.1 GiB** — remaining gap vs llama.cpp's 107.9 is
   mostly irreducible: ~4.6 ms/tok of thin rocBLAS GEMMs (un-attributed ×60/tok),
   quantize_q8_1 activation overhead, and PCIe fetch of cache-miss experts.

### Known limitations

- Default `max_running_req=4` exceeds the KV/state budget next to ~16 GB of weights
  on 24 GB cards (GDN hybrid-radix slots ≈ 154 MB/slot); use
  `--max-running-requests 1` for this model.
- On RDNA3 the Triton attention/GDN kernels previously fell back off matrix-core
  instructions (`no matching matrix core intrinsic` during warmup): the split-K
  flash-decode kernel built an M=8 head tile for GQA group 8 (Qwen3.6: 16 q / 2 kv
  heads), below the WMMA minimum M=16. The decode tile is now padded to M=16 with
  dead lanes masked end-to-end — kernels compile onto WMMA, warnings are gone, and
  all 95 triton-attention/backend tests pass. Measured bs=1 decode is unchanged
  (attention was already cheap on the fallback), but prefill and multi-request
  serving should benefit.
- CUDA-graph capture on ROCm is validated for the qwen3.5/qwen35moe GGUF paths with
  the triton attention backend; other model/backend combinations may still trip
  capture corners — use `FREETOKEN_ROCM_GRAPHS=0` if so.
- The MoE GGUF path requires `--moe-backend offload` (engine assertion); resident
  MoE layers only support bf16/fp8_block formats.
