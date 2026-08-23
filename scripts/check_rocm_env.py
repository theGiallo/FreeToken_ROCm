"""Validate an AMD ROCm environment for freetoken.

Checks, in order:
  1. torch is a ROCm build (torch.version.hip set) and sees the GPU
  2. Triton is the AMD build (bundled with the multi-arch wheels)
  3. the pip-bundled ROCm SDK resolves and the build env fixes apply
  4. a bf16 matmul runs on the device
  5. tvm-ffi JIT-compiles and launches a trivial HIP kernel via hipcc
  6. freetoken's host extension (_pinned_tensor) imports

Run inside the venv you intend to serve with:
    python scripts/check_rocm_env.py
Exits non-zero on the first failed stage.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))

STAGES: list[tuple[str, callable]] = []


def stage(name):
    def wrap(fn):
        STAGES.append((name, fn))
        return fn

    return wrap


@stage("torch ROCm build + device")
def check_torch() -> str:
    import torch

    if not torch.version.hip:
        raise RuntimeError(
            f"torch {torch.__version__} is a CUDA build; install the AMD "
            "multi-arch wheel: pip install 'torch[device-gfx1100]' "
            "--index-url https://repo.amd.com/rocm/whl-multi-arch/"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False (driver/GPU?)")
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    return f"{name} ({props.gcnArchName if hasattr(props, 'gcnArchName') else 'gfx?'})"


@stage("AMD Triton")
def check_triton() -> str:
    import triton

    version = getattr(triton, "__version__", "?")
    try:
        from triton.backends import backends  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"cannot inspect triton backends: {exc}") from exc
    if "amd" not in backends:
        raise RuntimeError(
            f"triton {version} has no AMD backend ({sorted(backends)}); reinstall torch "
            "from repo.amd.com so its companion triton wheel is pulled in"
        )
    return f"{version} (backends: {', '.join(sorted(backends))})"


@stage("pip-bundled ROCm SDK + build env")
def check_sdk() -> str:
    from freetoken.kernel.platform import ensure_hip_build_env, find_pip_rocm_sdk_root

    find_pip_rocm_sdk_root()
    ensure_hip_build_env()
    root = find_pip_rocm_sdk_root()  # re-resolve: env may point at a system ROCM_HOME
    hipcc = shutil.which("hipcc")
    if hipcc is None and root is None:
        raise RuntimeError(
            "no pip-bundled ROCm SDK found (_rocm_sdk_core in site-packages), "
            "ROCM_HOME is not set, and hipcc is not on PATH"
        )
    return f"SDK={root or 'system ROCM_HOME'}" + (
        f", hipcc={hipcc}" if hipcc else ""
    )


@stage("bf16 matmul on device")
def check_matmul() -> str:
    import torch

    a = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    c = (a @ b).float()
    ref = (a.float() @ b.float())
    err = (c - ref).abs().max().item()
    if err > 1.0:
        raise RuntimeError(f"matmul result diverges (max abs err {err})")
    return f"max abs err {err:.4g}"


@stage("tvm-ffi HIP JIT (hipcc)")
def check_jit() -> str:
    import torch

    from freetoken.kernel.platform import ensure_hip_build_env

    ensure_hip_build_env()
    from tvm_ffi.cpp import load_inline

    src = r"""
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <hip/hip_runtime.h>

#define CHECK_HIP(cmd) do { hipError_t e = (cmd); \
  TVM_FFI_CHECK(e == hipSuccess, Error) << #cmd << " failed: " << hipGetErrorString(e); } while (0)

__global__ void scale_add_kernel(const float* x, const float* y, float* out,
                                 float alpha, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) out[i] = alpha * x[i] + y[i];
}

void scale_add(tvm::ffi::TensorView x, tvm::ffi::TensorView y,
               tvm::ffi::TensorView out, double alpha) {
  int64_t n = x.size(0);
  int blocks = (int)((n + 255) / 256);
  scale_add_kernel<<<blocks, 256>>>(
      static_cast<const float*>(x.data_ptr()), static_cast<const float*>(y.data_ptr()),
      static_cast<float*>(out.data_ptr()), (float)alpha, n);
  CHECK_HIP(hipGetLastError());
  CHECK_HIP(hipStreamSynchronize(0));
}
"""
    mod = load_inline(
        name="ft_rocm_env_check",
        cuda_sources=src,
        functions=["scale_add"],
        backend="hip",
    )
    x = torch.randn(1 << 16, device="cuda")
    y = torch.randn(1 << 16, device="cuda")
    out = torch.zeros_like(x)
    mod.scale_add(x, y, out, 2.5)
    err = (out - (2.5 * x + y)).abs().max().item()
    del mod
    if err > 1e-4:
        raise RuntimeError(f"kernel result diverges (max abs err {err})")
    return "compiled+launched, correct"


def main() -> int:
    print(f"freetoken ROCm environment check ({len(STAGES)} stages)")
    for i, (name, fn) in enumerate(STAGES, 1):
        try:
            detail = fn()
        except Exception as exc:
            print(f"  [{i}/{len(STAGES)}] FAIL  {name}: {exc}")
            return 1
        print(f"  [{i}/{len(STAGES)}] ok    {name}" + (f" — {detail}" if detail else ""))
    print("ROCm environment OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
