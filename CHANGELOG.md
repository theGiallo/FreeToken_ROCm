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

### Verified

- RX 7900 XTX under WSL2 (Ubuntu 26.04, ROCm 7.x, torch 2.11): coherent generation
  on Qwen3.8-27B-Q4_K_S.gguf at ~13 tok/s decode; full 851-key shape/dtype
  reconciliation against a meta-device build; adapter tests green.

### Known limitations

- Default `max_running_req=4` exceeds the KV/state budget next to ~16 GB of weights
  on 24 GB cards (GDN hybrid-radix slots ≈ 154 MB/slot); use
  `--max-running-requests 1` for this model.
- On RDNA3 the Triton attention/GDN kernels fall back off matrix-core instructions
  (`no matching matrix core intrinsic` warnings during warmup), making prefill and
  decode markedly slower than hand-tuned HIP stacks such as llama.cpp. CUDA graphs
  remain disabled on ROCm.
