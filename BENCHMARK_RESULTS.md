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

---

# UPDATE — CUDA graphs enabled on ROCm (same day)

Root cause found by profiling (`PERF_INVESTIGATION_PLAN.md`): decode was
**host-dispatch-bound** — ~2000 eager kernel launches + aten dispatch per token
with the GPU mostly idle. Bandwidth was never the limiter (all-miss worst case
needs 51 ms/token at the measured 12.8 GB/s PCIe; we measured 80 ms with a warm
cache). Re-enabling CUDA-graph capture on ROCm (`engine/graph.py`) collapses the
entire eager loop into one replayed graph:

## Short prompt (39 in / 256 out), graphs ON, default config

| Engine | Decode (steady-state) | Decode (wall avg) |
|---|---|---|
| FreeToken + graphs | **52–53 tok/s** | **33.4–38.2 tok/s** |
| FreeToken (before) | 13–14 tok/s | 11.6–12.1 tok/s |
| ollama | 49–51 tok/s | — |

## Long prompt (517 in / 8192 max out), `moe_cache_size=4096`, graphs ON

| Engine | Run | Output toks | Rate |
|---|---|---|---|
| FreeToken + graphs | 1 | 8191 (hit cap) | **33.6 tok/s wall / ~37.6 decode-only** |
| FreeToken + graphs | 2 | 7907 (stop) | **33.2 tok/s wall / ~37.3 decode-only** |
| ollama | best | 7583 (stop) | 44.07 tok/s |

- Capture cost: 1.24 s at bs=1, ~0.4 GiB VRAM. Outputs remain greedy-deterministic.
- Remaining gap vs ollama: **~1.16×** — while ollama holds all 23.9 GB resident
  in VRAM and FreeToken streams experts over PCIe from system RAM (its design
  target is models larger than VRAM).
- New default: graphs ON on ROCm; set `FREETOKEN_ROCM_GRAPHS=0` to restore the
  old eager behavior.

Also landed: `_torch_fused_topk` fast path (renormalize=True routes via top-k on
raw logits + k-wide softmax — mathematically identical, verified bitwise-equal
expert ids across shapes/scales). No measurable e2e delta once graphs are on,
but it removes ~60 eager launches/token for any eager-fallback user.

---

# UPDATE 2 — true reference, kernel-level attribution, skinny-linear fix

## The "~1.16× gap" above was measured against a crippled reference

The ollama runs above used tag `qwen3.6:35b-a3b_128k`, which sets
`num_ctx=131072` (`ollama show` confirms). That KV reservation forces llama.cpp
into **partial expert offload** even on a 24 GB card, capping it at ~50 tok/s —
i.e. we were comparing our full-GPU-resident design point against *their*
offload design point. With `num_ctx` sized to fit (the plain `qwen3.6:35b-a3b`
tag / explicit `-c`), the same hardware does ~104–108 tok/s.

## Procedure for all numbers below (2026-08-25)

Identical workload per engine, single stream, greedy, Windows host:

1. **llama.cpp** (master `f280b2698` = b10615 + cherry-picked upstream PR #25334,
   which fixes qwen35 Ollama-GGUF loading): `llama-server -m <blob> --jinja
   -fa on -ngl 999 -c 65536 --no-warmup -p 8090`; one warmup chat completion +
   one measured completion (21-token prompt, `n_predict=256`); perf recap read
   from server log (`eval time … / 256 tokens`). Peak VRAM sampled at 50 Hz via
   `\GPU Process Memory(pid_*)\Dedicated Usage`.
2. **ollama** (`qwen3.6:35b-a3b`, default ctx): same prompt shape through
   `/api/chat`; tok/s from `eval_count/eval_duration`. VRAM attributed to the
   embedded engine process (`...\AMD\AI_Bundle\Ollama\lib\ollama\llama-server.exe`),
   *not* the `ollama.exe` front-end.
3. **FreeToken**: e2e script, two consecutive generate() calls of 256 tokens on
   the same engine; report both (run 1 includes cold LRU fill).

Model file identical throughout (the Ollama blob, 23,938,321,664 B).

## Three-way results (before today's fix)

| Engine | Decode | Peak VRAM |
|---|---|---|
| llama.cpp b10615+#25334 | **107.9 tok/s** (9.27 ms/tok) | 22,135 MB (21.6 GiB) |
| ollama 0.32.14 | **103.5 tok/s** | 23,281 MB (22.7 GiB) |
| FreeToken @ graphs-on baseline | 34.8–40.0 tok/s wall (steady windows ~54) | ~22.1 GiB (1.8 GiB free) |

Real gap ≈ **2×**, not 1.16×.

## Where decode time actually goes (chrome trace, graphs on, bs=1)

torch.profiler over a decode-only window (383 tokens), prefill mega-events
filtered (>400 µs). Device-busy totals matched engine telemetry, i.e. graph-mode
decode is GPU-bound, not dispatch-bound.

Before the fix — **18.12 ms/tok**:

| ms/tok | ×/tok | kernel | verdict |
|---|---|---|---|
| 5.51 | 40 | rocBLAS `Cijk_Ailk…MT128x32x16` (one per MoE layer, `aten::linear`) | 🔴 pathological pick |
| 1.71 | 170 | ggml `mul_mat_vec_q` Q4_K (resident projections) | legit |
| 1.59 | 100 | rocBLAS thin `Cijk…MT32x32x32` | mediocre |
| 1.04 | 332 | `quantize_q8_1` (activation quant for ggml kernels) | semi-legit |
| 0.99 | 40 | `fast_index_copy_multi` (PCIe expert misses) | coverage-bound |
| 0.91 | 120 | router top-K chain (`warpMergeSortTopK`+`bitonicSortKVInPlace`) | torch fallback |
| 1.35 | 80 | `moe_vec_q` Q4_K/Q6_K (the actual routed-expert math) | legit |

Standalone microbench pinned the 🔴 item: **every** dispatch route
(`F.linear`, `mm`, `mv`) sends `[1,K]×[K,1]` shapes to the same ~157 µs fat
kernel — a 4 KB weight read taking 157 µs. Per MoE layer the shared-expert gate
(`LinearReplicated(2048→1)`) pays this once ⇒ 40 × ~138 µs ≈ the entire 5.5 ms.
(The earlier full-coverage experiment — `FT_MOE_CACHE=10496`, 97% slot coverage —
moved wall speed 39.93→40.15 tok/s, i.e. nothing: the PCIe-miss cost is real but
small, contradicting PERF_INVESTIGATION_PLAN §7 whose "~1.4 MB/token" estimate
forgot the ×40-layers factor; correct worst case is ~57 MB/token ≈ 4.5 ms, and
in practice far less because misses are rare and the copy kernel is cheap.)

## Fix: Triton GEMV for M==1 linears

New `kernel/triton/skinny_linear.py`: bandwidth-bound single-row matvec (fp32
accumulate), dispatched from `_LinearTPImpl.forward` (covers `LinearReplicated`,
col/row-parallel merged projections) whenever `x.shape[0]==1` on CUDA bf16/fp16;
everything else falls through to `F.linear` unchanged. Kill switch:
`FREETOKEN_SKINNY_LINEAR=0`.

After — **12.39 ms/tok** device-busy (−5.7): fat Cijk gone from decode entirely;
`_gemv_kernel` ×80/tok costs 0.79 ms total (~10 µs/call incl. the previously
fat shapes).

## End-to-end after the fix

| Run | Wall decode | VRAM free |
|---|---|---|
| 1 (cold LRU) | **45.23 tok/s** | 1.7 / 23.9 GiB |
| 2 (warm) | **52.68 tok/s** | 1.7 / 23.9 GiB |

Baseline was 35.1 / 40.0 tok/s ⇒ **+32 % warm**, VRAM unchanged. Tests:
`tests/moe` 14 pre-existing failures unchanged (identical counts with change
stashed); adapter tests pass; no new failures anywhere.

## Remaining known headroom

- Wall-vs-device gap: e2e wall is 19.0 ms/tok vs 12.4 ms/tok device-busy ⇒
  ~6.5 ms/token of host-side work between graph replays (scheduler/sampler/
  detokenize path) — under the profiler the same loop sustains ~70 tok/s wall,
  so this overhead is measurable and worth a dedicated pass.
- Router top-K chain ~0.9 ms/tok (pure-torch fallback; `triton_kernels` wheel
  unavailable on this stack).
- The post-fix `_gemv_kernel` ×80/tok accounts for the router + shared-expert
  gate (`LinearReplicated` pairs); the shared expert's gate_up/down were already
  ggml-vec-kernel territory (inside the `mul_mat_vec_q` counts), and the residual
  ×60/tok thin rocBLAS calls are un-attributed (graph replay drops cpu-side
  correlation) — worth one more identification pass.
- Expert-slot coverage: auto picks ~9.5k/10.75k slots (88–90 %); misses cost
  ≤1 ms/token at current rates. Full coverage does not fit alongside KV at
  default `memory_ratio`; not worth forcing given the measured impact.
