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

---

# FINDINGS (2026-08-24, experiments executed)

- **E0 - machine ceilings** (`ft bench bw`): CPU STREAM read 27.9 GB/s, PCIe
  linear H2D 12.8 / D2H 13.6 GB/s (WSL paravirtualization ~halves PCIe vs native
  3090-class 25 GB/s), random-row gather 12.1-12.6 GB/s. All-miss decode ceiling
  653 MB/tok / 12.8 GB/s = 51 ms -> ~19.5 tok/s even with ZERO cache hits;
  measured pre-fix was 12.4 tok/s with a warm cache. Transfer never was the
  bottleneck. The bench also flags CPU-MoE/PCIe ratio 2.18x in favor of hybrid
  fetch policies on this host (future lever).
- **E1/E1c - profile attribution** (chrome-trace stack analysis, decode window):
  - ~2000 kernel launches/token (1599 hipLaunchKernel + 372 module launches),
    ~192 aten::copy_/token, GPU idle between micro-kernels: total HOST time
    ~81 ms/token ~= whole step wall. Decode is **host-dispatch-bound**.
  - Real math is small: ggml vec kernels + rocBLAS + GDN core ~25 ms traced
    (itself inflated); GDN update kernel alone 0.4 ms/token. Attention/GDN NOT
    the problem (the split-K MFMA compile failure at attention.py:240/252 falls
    back to a path that is cheap enough at bs=1; still worth fixing someday).
  - `hipPointerGetAttribute` x1800/tok: every Triton launch queries every pointer
    arg (amd/driver.c extractPointer). Patched locally to probe UVA identity once
    then skip (venv site-packages, not committed); measured e2e effect: none -
    the queries were cheap, the tracer overstated them.
  - Mid-step sync barriers exist but are PREFILL-only (40x `_invalidate_prefill_buffer`
    boolean-mask indexing = hidden nonzero D2H per prefill) plus ~2 scalar H2D +
    1 event sync per decode step. Decode loop itself is sync-clean.
  - `_torch_fused_topk` ran ~120 tiny eager ops/token (full-vocab fp32 copy +
    softmax + topk + renorm glue). Fast path added (see below).
- **E2 - triton_kernels**: installable on Linux but deliberately gated off on
  ROCm (kernel/backend.py:56 - its fused router needs NVIDIA TMA/warp-spec).
  Dead end by design; the torch fallback is the intended ROCm router.
- **E4 - CUDA graphs on RDNA**: with the blanket `is_rocm()` early-out removed,
  stream capture **succeeds cleanly** on RX 7900 XTX / WSL2 / ROCm 7.1 at bs=1
  (1.24 s, 0.4 GiB). No corner tripped for this model/backend (triton attention,
  offload MoE). Decode went **12-14 -> 52-53 tok/s telemetry steady-state**
  (33-38 tok/s wall incl. overheads); long-context sustained ~37.5 tok/s
  decode-only over 8k tokens. Graphs are now default-ON on ROCm
  (`FREETOKEN_ROCM_GRAPHS=0` restores old behavior).
- Remaining gap after fix: ~1.16x vs ollama (44 tok/s) which runs fully
  VRAM-resident. Next levers, in order of expected value:
  1. Hybrid MoE backend (q* policy): B_H/B_P = 27.9/12.8 favors fetching only a
     bandwidth-matched fraction of misses and computing overflow on CPU
     (ensure_experts_hybrid exists; q4_0 CPU weight path availability TBD).
  2. Prefill sync removal: device-side rewrite of `_invalidate_prefill_buffer`;
     the 40x nonzero stalls also gate prefill overlap quality.
  3. Attention split-K MFMA shapes on gfx1100 (compile failure -> fallback).
  4. Larger captured graph sizes for multi-request serving (only bs=1 captured here).

## 7. Follow-up session (post-graphs): miss-rate measurement closes the hybrid question

With graphs landed, the ranked levers were re-tested against data:

- **Prefill sync removal — DONE.** `_invalidate_prefill_buffer` now calls a
  device-side `invalidate_slot_range` Triton kernel (single program, BLOCK =
  num_experts; frees owner ids, clears usage, no host readback) with a torch-op
  fallback for CPU-resident test caches. The old formulation's boolean-mask
  indexing hid one nonzero-D2H per prefilled layer (40/prefill here). Short-bench
  prefill telemetry: ~20 -> ~23.7 tok/s (+15-20% at 39 tokens); decode unchanged.
  `tests/moe/{test_prefill_hit_d2d,test_offload,test_fused_copy,test_hybrid_fetch}.py`
  all green (23 passed, 6 skipped).
- **Hybrid MoE backend — CLOSED, not worth it on this workload.** Two facts:
  1. Load-time blocker: the CPU executor only parses strict Q4_0 rows
     (`cpu_executor.py:_resolve_q4_0_banks`, 18 B/32 K); our Qwen3.6 GGUF banks are
     mixed K-quants (gate/up Q4_K = 144 B/256 K, down Q6_K = 210 B/256 K), so
     `LLM(..., moe_backend="hybrid")` asserts out. Enabling it would require new
     CPU W4A16 GEMV kernels for Q4_K/Q6_K in csrc/cpu_moe/cpu_moe_ext.cpp.
  2. Payoff bound that kills it: with `moe_collect_stats=True`, the long bench
     (4096 slots = 39% of experts) measured **decode miss_rate = 9%**
     (0.7 missing of 8 active experts/step) — miss traffic is ~1.4 MB/token,
     ~0.1 ms at PCIe bandwidth. Even a perfect hybrid would move decode <1%.
     Sanity anchor for the other extreme: forcing `moe_cache_size=512`
     (all-miss decode) drops steady-state from ~37 to ~16 tok/s, confirming
     misses only matter when the cache actually thrashes — ours does not.
- **Conclusion:** remaining decode time is hit-path GPU work + fixed overheads.
  Next candidates if more speed is ever needed: attention split-K MFMA shapes on
  gfx1100, larger graph sizes for bs>1 serving, and GPU GEMV efficiency of the
  q4_K/q6_K expert kernels.

## 8. Correction to §7 + the real post-graphs bottleneck (2026-08-25)

Two errors in the reasoning above, both caught by later measurement:

1. **§7's miss-traffic arithmetic was wrong by ×40.** "0.7 missing experts/step
   ≈ 1.4 MB/token" counted ONE layer; the correct figure is per-layer bytes
   (≈2.04 MB/expert instance: Q4_K gate/up row 1024×1152 B + Q6_K down row
   2048×420 B) × 40 layers ⇒ ~57 MB/token at 9% misses, ~4.5 ms at PCIe random-
   gather bandwidth. The hybrid-closure *verdict* survives anyway — see (2).
2. **The measured miss cost is far below that bound.** Raising slot coverage
   88% → 97.5% (`FT_MOE_CACHE=10496`) moved warm wall speed 39.93 → 40.15 tok/s:
   statistically nothing. The `fast_index_copy_multi` kernel shows up at
   ~25 µs/layer regardless of miss count in traces, i.e. launch/latency-bound,
   not bandwidth-bound at these sizes.

The actual dominant term was found by chrome-tracing a decode-only window
(graphs on): **5.5 ms/token went to one rocBLAS tiled-GEMM kernel picked for
every `[1,K]` replicated linear** (MoE router gate, shared-expert gate) —
~138 µs per call for a 4 KB weight read, identical across `F.linear`/`mm`/`mv`
dispatch. Fixed with a Triton GEMV specialization (`kernel/triton/skinny_linear.py`,
dispatched from `_LinearTPImpl.forward`): device-busy decode 18.12 → 12.39 ms/tok,
warm e2e 40.0 → **52.68 tok/s** (+32%), VRAM unchanged. Full data and the
three-way reference re-measurement (llama.cpp 107.9 / ollama 103.5 / FreeToken
52.7 on identical workloads) live in `BENCHMARK_RESULTS.md` UPDATE 2. Remaining
headroom: ~6.5 ms/token host-side between graph replays (wall vs device-busy),
router top-K torch fallback (~0.9 ms), residual unattributed thin rocBLAS calls
(~0.9 ms).

## 9. Next lever: parallelize expert-bank PLACEMENT (the packed->bank copies), not the read (2026-09-13)

Read is already overlapped by the O_DIRECT frontier (UPDATE 2: whole-file
17.4 -> 11.2 s build, read ~4.6-5.5 s tucked under placement). This section is
the *plan* for the remaining tail, so it inherits the harness rules: every
claim must be backed by a measurement this harness performs (or be explicitly
marked a hypothesis), and no code lands before a byte-identical + ordering-
preserving test passes.

### 9.1 What the serial fill does (real contract, from `git show HEAD`, not memory)

`_q4_fill_banks` (python/freetoken/models/qwen3_5_moe/gguf.py:851) iterates the
`(name, packed)` stream *in stream order*; three packed-byte writes complete a
layer:
- `ffn_gate_exps.weight` -> `banks["gate_up"][layer][:, :I]` (left band)
- `ffn_up_exps.weight`   -> `banks["gate_up"][layer][:, I:]` (right band)
- `ffn_down_exps.weight` -> `banks["down"][layer]`

i.e. `gate_up` merges gate at `[:, :I]` with up at `[:, I:]` (a 2I row bank of
`rb(H)` rows), `down` is `[E, H, rb(I)]`. Every write is `rows * rb` contiguous
bytes on both sides (a packed row reshaped into the bank's `[hi-lo, rb]` band).
`tracker.note(layer)` fires once per tensor **after** its copy; the
`LayerCompletionTracker(3, hb, sink)` fires the pin pipeline / converter sink
once per completed layer, and the MoE host kernels consume per-layer banks. The
builds are minutes-long, so progress is logged every 8 completed layers.

### 9.2 The parallel placement that is provably safe

Each individual `.copy_` is `rows * rb` contiguous on BOTH sides (packed row
buffer and the disjoint bank band). Splitting ONE tensor's copy into
`placement_workers` disjoint **row bands** (plain memcpy per band, non-overlapping
byte ranges on both sides) is byte-identical to the serial copy and farms out
linearly -- exactly the trick the O_DIRECT frontier reader already uses for reads,
so it is battle-proven in this codebase. Ordering is untouched: the consumer loop
still calls `tracker.note(layer)` once per tensor in stream order after ALL its
rows have landed, so layers keep completing gate-then-up-then-down in the exact
sequence the serial path produced; the pin pipeline and the MoE kernels see the
same per-layer byte sequence (their contract only requires every row byte below
`h_bytes`/`i_bytes` to be written once, which disjoint bands preserve).

Deliberately NOT done: parallelizing the *consumer loop itself* (dispatching
whole tensors to a pool and noting out of stream order). That changes when
`tracker.note` fires relative to the serial path and would risk reordering the
pin/sink sequence; it is off the table until 9.5's ordering test proves the
tracker tolerates it (the tracker docstring claims it does -- "write layers from
many threads in arbitrary order" -- but "claims" is not "measured").

### 9.3 Hypothesis (ranked): placement is now the dominant uncovered term

- **P1 (high)**: after UPDATE 2, the read is hidden and placement is the
  single-threaded `copy_` tail. Expected farm-out: row-band memcpy across
  `placement_workers` threads moves placement cost down toward the frontier
  reader's measured O_DIRECT band memcpy throughput on this box.
- **P2 (low)**: gains wash out because the copies are already near memcpy
  roofline and pinned-bank writes are bandwidth-bound. That would say the real
  lever is GPU-side (attention split-K / graph replay spacing), not placement.

### 9.4 Experiment to close it (cheap, this harness)

1. Write `tests/models/test_qwen35_placement_parallel.py` (or extend
   `tests/models/test_qwen35_gguf_adapter.py`): a synthetic `(name, packed)`
   stream over `L` layers builds the banks twice -- once via the SAAS_mik serial
   path, once via a row-band placement with `placement_workers=8` -- and asserts
   (a) **byte-identical** banks (`torch.equal` per bank, full `[E, 2I, rb(H)]` /
   `[E, H, rb(I)]`), (b) **identical `tracker.note` order** (capture the fired
   layer sequence from the sink and compare to the serial sequence), (c)
   `(seen_gu, seen_dn)` completeness. Runs on CPU; no GPU/ROCm needed, so it is
   CI-green in seconds.
2. Only if (a)+(b) pass: add `placement_workers=8` to the O_DIRECT parallel
   loader and re-measure cold build on the 7900 XTX. Land the code with the test;
   keep the serial path default so nothing else changes.
3. Record tok/s + build seconds and the memcpy farm-out factor in
   `BENCHMARK_RESULTS.md` UPDATE 3 and update this plan's verdict.

### 9.5 Gate for any wider reordering (do not skip)

Before whole-tensor pooling / out-of-stream-order `note` is ever considered,
extend 9.4's test (b) to fire `tracker.note` in a deliberately shuffled order for
real `LayerCompletionTracker` + `PinPipeline` and assert the sink still fires
once per completed layer in ascending layer order. If the tracker's "arbitrary
order" claim holds up under test, that path reopens; until then, placement stays
row-band-within-tensor and the note sequence is byte-identical to serial.

---

(Back to the rank-ordered list for the decode gap proper -- §1-8 unchanged; this §9
is closed as a *plan* pending 9.4's measurement, per the harness rule that
hypotheses ship as plans, not as code without a passing test.)

### 9.4 Measurement anchor (byte-verified 2026-09-13)

9.2's serial/target numbers and any byte-identity measurement MUST be taken with
this box's actual entrypoints, which live in WSL (FreeToken runs *inside* WSL,
not on the Windows side):

- `nexus/f/programming/llm/launch_qwen35.sh` (161 lines) -> emits
  `wsl -e bash -lc "exec $FT serve --model ... --moe-backend offload
  --kv-reserve-tokens $KVRES --moe-cache-rate $RATE --num-tokens $CTX ..."`
  with budget 21.15 GiB, reserve 1.5 GiB, KV 20 KiB/tok, TOTAL_EXPERTS=10240,
  PER_SLOT_BYTES=1376256 -- the exact build+placement workload this plan grates
  against. Measurement command:
  `wsl -e bash -lc "exec $FT serve --model ... --moe-backend offload ..."`.

- `launch_qwen_3.6_35b-a3b_max-ctx_kv-persist.sh` (2 lines) -> thin wrapper:
  `./launch_qwen35.sh --ctx 262144 --mode thinking --kv-persist 1
  --kv-persist-max-gb 10` -- the max-ctx KV-persist variant 9.2 also targets.

Do NOT measure with a hand-typed `ft serve` invocation that contradicts these;
the launch scripts are the single source of truth for the on-box flags.
