# Benchmark Results — Qwen3.6-35B-A3B GGUF on RX 7900 XTX

Native-GGUF serving through FreeToken (`qwen35moe` adapter, offload MoE backend)
vs ollama 0.32.14 (embedded llama.cpp) on identical prompts and greedy sampling.
All runs use thinking mode (default chat template), temperature 0.

## Environment

| | |
|---|---|
| GPU | AMD Radeon RX 7900 XTX (gfx1100, 24 GB) |
| Stack | WSL2 Ubuntu 26.04, ROCm 7.x, PyTorch 2.11, Triton (RDNA3 backend) |
| Model | `qwen3.6:35b-a3b_128k` Ollama blob — same 23.9 GB file for both engines |
| File layout | arch `qwen35moe`, 40 layers (all MoE: 256 experts top-8, I=512, shared expert I=512), Q4_K gate/up + Q6_K down experts, Q4_K/Q5_K/Q6_K attention |
| FreeToken flags | `LLM(gguf, max_running_req=1, moe_backend="offload", moe_cache_auto=True)` (short test) / `moe_cache_size=4096` (long test) |

**Free VRAM at start of tests: 22.83 / 23.94 GiB (95%)** — measured via a separate
probe process immediately before engine init (the long test); the short-prompt
session started from an equivalent state (~23 GiB free).

## Test 1 — short prompt (39 tokens), 256 max output

Two runs each; report the warm run. FreeToken wall time includes prefill.

| Engine | Prefill | Decode (steady-state) | Decode (wall avg) |
|---|---|---|---|
| FreeToken | 19.9 tok/s | **13–14 tok/s** | 11.6–12.1 tok/s |
| ollama | 622 tok/s | **49–51 tok/s** | — |

Notes:
- FreeToken prefill is measured over only 39 tokens → dominated by fixed per-batch
  overhead; not comparable to ollama's sub-100 ms prompt eval at this length.
- Output byte-identical across FreeToken repeat runs (greedy determinism holds).

## Test 2 — long prompt (517 tokens), 8192 max output

Same prompt text for both engines. FreeToken used `moe_cache_size=4096`
(expert slot cache shrunk so KV gets the freed VRAM: 539,765 pages allocated,
10.3 GiB). Load time 37 s.

| Engine | Run | Output toks | Decode rate | Notes |
|---|---|---|---|---|
| FreeToken | 1 | 8191 (hit cap) | **11.92 tok/s wall / ~12.4 decode-only** | steady windows 11–13 tok/s, flat over full 8k context |
| FreeToken | 2 | 7137 (natural end) | **12.59 tok/s wall / ~13.2 decode-only** | steady windows 12–14 tok/s |
| ollama | 1 | 7583 (stop) | **43.18 tok/s** | prefill 637 tok/s |
| ollama | 2 | 7583 (stop) | **44.07 tok/s** | prefill 6450 tok/s (cached) |

Notes:
- ollama's decode *dropped* from ~50 tok/s (short context) to ~43–44 at multi-k
  context; FreeToken stays flat (~12–13) because GDN hybrid-radix decode cost is
  context-independent and its bottleneck lies elsewhere.
- Radix cache works: FreeToken run 2 reused 512/517 prompt tokens
  (`#cached-token: 512`).
- Observed defect (quality, not perf): after the natural `<|endoftext|>` FreeToken
  continued decoding for a while, emitting `<|im_start|>` garbage until another
  stop id matched — stop-token handling for this arch needs a look.

## Summary

Decode gap ≈ **3.4×** in favor of llama.cpp (ollama), consistent across short and
long contexts. Prefill gap is much larger but unmeasured meaningfully here (the
FreeToken prefill numbers above are overhead-dominated at these lengths; see the
dense-model measurements in CHANGELOG for the compute-bound picture).

The working hypothesis remains the RDNA3 software stack: Triton kernels falling
back off matrix-core instructions (`no matching matrix core intrinsic` during
warmup) plus CUDA graphs disabled on ROCm. The MoE offload path itself works
correctly; profiling where decode time actually goes (bank streaming vs attention
kernel vs router/dispatch overhead) is the next step.
