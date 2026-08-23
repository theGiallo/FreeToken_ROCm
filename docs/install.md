# Install

## Requirements

- Linux x86_64, Python >= 3.10, with [uv](https://docs.astral.sh/uv/) recommended
  (plain `pip` + `venv` works too)
- One of:
  - **NVIDIA GPU**, driver r580+ (CUDA 13)
  - **AMD GPU** (ROCm), gfx110x (RX 7900 XTX/XT/GRE) verified; see
    [ROCm setup](#amd-rocm-setup) below

Baseline-functional on ROCm: bf16 (unquantized) models only, single GPU,
Triton attention/MoE backends. CUDA graphs, FP8/FP4 quants, flashinfer/sgl
backends and expert-offload memops are NVIDIA-only and are rejected or cleanly
disabled at startup.

## Method 1: Install from PyPI

```bash
uv venv && source .venv/bin/activate
uv pip install "freetoken[accel]"
```

CUDA kernels are JIT-compiled on first use, need a CUDA 13 toolkit with `nvcc` on PATH.

## Method 2: Install from source

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

## AMD ROCm setup

No system-wide ROCm installation is required: the AMD multi-arch pip wheels
bundle the ROCm runtime, `hipcc` and an AMD Triton build as regular wheels.

```bash
python -m venv ~/venv-ft && source ~/venv-ft/bin/activate
pip install "torch[device-gfx1100]" \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/
pip install -e /path/to/FreeToken --no-build-isolation   # or freetoken from PyPI
```

`freetoken` detects the pip-bundled SDK automatically (via
`freetoken.kernel.platform.ensure_hip_build_env()`), fixes the missing
unversioned `.so` linker symlinks in `_rocm_sdk_core/lib`, and points the JIT
toolchain at it — no `amdgpu-install`, no `HSA_OVERRIDE_GFX_VERSION`.

Verify the environment:

```bash
python scripts/check_rocm_env.py
```

## Verify

```bash
source .venv/bin/activate
ft --version
ft serve --model ~/path/to/Qwen3.6-35B-A3B
curl http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}]}'
```

Then head to [quickstart.md](quickstart.md).
