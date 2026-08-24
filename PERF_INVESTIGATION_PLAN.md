# Perf Investigation: Qwen3.6-35B-A3B decode gap on RX 7900 XTX (ROCm/WSL)

Companion to `BENCHMARK_RESULTS.md`. Goal: explain the ~3.4x decode gap vs ollama
(12.4 vs 43-51 tok/s), check it against the paper's own numbers, and lay out
ordered experiments + fixes.

## 1. Reference targets (paper arXiv 2608.16157, Qwen3.6-35B-A3B BF16)

| System | B_P PCIe GB/s | B_H host GB/s | FreeToken tok/s | vs strongest baseline |
|---|---|---|---|---|
| RTX 5090 server | 52.7 | 77.3 | **77-83** | 1.8-2.3x |
| RTX 5090 desktop | 49.0 | 53.8 | ~similar | 2.1x |
| RTX 4090 | 25.1 | 63.2 | ~42.7 (implied, W2) | 1.3x |
| RTX 3090 | 25.3 | 56.7 | n/a (ratios only) | 1.3x |
| RTX 4060 laptop 8GB (NVFP4) | 11.8 | 47.5 | **39.3** | 1.8x |

Key paper facts:
- Decode is bandwidth-bound *by design*: bytes/token = routed expert reads; the
  cache absorbs most of them (measured 16% miss rate at 37% pool capacity on
  LRU vs 41% KTransformers / 62% llama.cpp static split).
- CUDA graphs are central to their decode path ("CUDA-graph-compatible LRU");
  baselines lose 20% from host moves, FreeToken only 4%.
- No AMD/ROCm content anywhere in the paper. Fast path is CUDA-only.

## 2. Our measurements (this box) and why the gap is NOT fundamental

RX 7900 XTX 24GB, WSL2, experts Q4_K gate/up + Q6_K down:
- FreeToken `offload`: 12.4 tok/s steady decode (~80 ms/token), prefill 19.9 tok/s
- ollama (llama.cpp embedded): 43-51 tok/s decode, 622-6450 tok/s prefill

Bytes/token arithmetic for our GGUF:
- Per expert read: gate_up Q4_K ~1.18 MB + down Q6_K ~0.86 MB = ~2.04 MB.
- Routed reads/token: 40 layers x top-8 = 320 -> **653 MB/token if every read missed**.
- Cache capacity used in the long bench: 4096/10240 slots (~39%), same order as
  the paper's 37%-capacity config with a measured 16% miss rate -> expect
  ~100-130 MB/token over PCIe -> at even 15-20 GB/s pinned H2D that is
  **~6-9 ms/token**, i.e. bandwidth alone permits >110 tok/s.
- Compute: routed-expert MACs ~0.1 GFLOP/token - trivial vs the card's matrix
  throughput.

Measured 80 ms/token therefore means **>=85% of decode time is execution
overhead, not expert transfer**. This matches the observation that run 2
(warm radix cache, 512/517 prefix hit) decoded at the same speed.

## 3. Confirmed defects on the ROCm port (evidence)

- **D1 - CUDA graphs disabled**: `engine/graph.py:111-116` unconditionally skips
  capture on ROCm ("stream capture trips over driver/runtime corners on RDNA,
  buys little without PDL"). Every token pays full kernel-launch + Python
  dispatch for hundreds of kernels across 40 layers. Log: "CUDA graph is disabled."
- **D2 - flash-decode split-K Triton kernel fails to compile on gfx1100**:
  `kernel/triton/attention.py:240` (`tl.dot(q,k)`) and `:252` (`tl.dot(p,v)`)
  -> "no matching matrix core intrinsic for wmma version 1, instruction shape
  [0, 0, 32] / [0, 0, 256]". Malformed shapes resolved for this arch; some
  fallback served attention instead (coherent output proves it ran). Which
  fallback and its cost are unknown (E6).
- **D3 - router fallback per layer per token**: "fused_topk: triton_kernels is
  not installed -> pure-torch router fallback (slower)". The warning says it is
  installable on Linux - we are on Linux (WSL). Quick win candidate (E2).
- **D4 - batch-memcpy path unverified on HIP**: `moe/offload_cache.py`
  resolves `freetoken.kernel.batch_memcpy` lazily; no "unavailable" line in our
  log suggests it loaded, but effectiveness on HIP is unmeasured (E3/E0).

## 4. Hypotheses, ranked by expected payoff

- **H1 (high)**: D1 dominates - without graphs, launch/dispatch overhead is the
  80 ms/token floor. Consistent with dense-model finding (13 tok/s vs llama.cpp 36).
- **H2 (high)**: D2 - attention runs a slow fallback every token, every layer.
- **H3 (medium)**: D3 - pure-torch topk router adds fixed Python cost per layer
  per token; possibly removable in minutes via pip install.
- **H4 (medium)**: D4 - misses may be copied sub-optimally (no batched memcpy /
  fused copy disabled on HIP), inflating the 6-9 ms transfer estimate.
- **H5 (low)**: cache hit rate materially worse than paper's LRU simulation.
  Weak: warm-run speed unchanged, so hits don't dominate time either way.

## 5. Experiments (ordered, each cheap)

- **E0 - machine bandwidth truth**: `ft bench bw` (python/freetoken/moe/benchbw.py).
  Get pinned H2D B_P and host-stream B_H; compare against Table 1 rows. Also
  confirms whether batch memcpy / fused copy primitives even build here.
- **E1 - profile one decode step**: torch profiler over ~32 steady-state decode
  tokens; bucket time into attention / MoE GEMM / router-topk / copies / other;
  count launches/token. This decides how much H1+H2+H3 are worth and in what mix.
- **E2 - install triton_kernels** (`pip install triton-kernels`) in venv-ft,
  rerun e2e script, compare tok/s. Pure win if it loads on ROCm.
- **E3 - copy-path A/B**: rerun e2e with `FREETOKEN_FUSED_COPY=0`; add a debug
  log of `_resolve_batch_memcpy()` outcome; compare ms/token. Isolates H4.
- **E4 - probe graph capture on RDNA**: temporary env-gated override of the
  `is_rocm()` early-out; record the exact capture failure (which op/driver
  call). Outcome tells us whether graphs are fixable locally or need upstream
  (hipGraph/PDL) work.
- **E5 - microbench the ggml MoE kernel**: call `fused_experts_gguf_q4_0` alone
  at bs=1 with a warm slot cache in a loop; compare achieved GB/s vs roofline.
  Separates MoE kernel health from everything around it.
- **E6 - name the attention fallback**: trace which codepath actually serves
  decode when the split-K Triton kernel fails MFMA compile (read backend
  selection in the attn backend; instrument once).

## 6. Fix roadmap (mapped to findings)

1. Router + small kernels: install/repair triton_kernels path (H3).
2. Attention on gfx1100: fix the MFMA shape selection so the split-K kernel
   compiles (RDNA3 supports WMMA via different intrinsic shapes); or wire an
   explicitly fast fallback (H2).
3. Graphs: pursue E4 outcome - if capture trips on a narrow op, shim just that
   op; hipGraphs parity or PDL-dependent wins likely need upstream work (H1).
4. Transfer path: verify batch memcpy on HIP; consider raising moe_cache_size
   now that VRAM headroom is known (95% free pre-load; KV was the binding
   constraint at 4096 slots) (H4/H5).

Realistic near-term goal: close the overhead gap toward ~30+ tok/s on this box.
Fairness note for any published comparison: ollama keeps the full 24 GB model
resident in VRAM (it barely fits); FreeToken's design target is models larger
than VRAM - the honest baseline set is the paper's Table 1 hardware, not a
same-card llama.cpp fully-resident run.
