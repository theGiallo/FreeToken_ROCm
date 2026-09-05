from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


ROOT = Path(__file__).parent
KERNEL = ROOT / "python" / "freetoken" / "kernel"


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_toolchain() -> None:
    # No-ops when torch.version.cuda is unset (ROCm/CPU builds).
    _load_by_path("_freetoken_toolchain", KERNEL / "_toolchain.py").check_nvcc_matches_torch()


def _is_rocm() -> bool:
    return _load_by_path("_freetoken_platform", KERNEL / "platform.py").is_rocm()


def _rocm_runtime_paths() -> tuple[list[str], list[str]]:
    """Include/lib dirs of the ROCm installation backing this torch build.

    Uses kernel/platform.py to resolve the pip-bundled SDK (or an explicit
    ROCM_HOME) so `pip install` works in both editable and regular flows.
    """
    platform_mod = _load_by_path("_freetoken_platform", KERNEL / "platform.py")
    platform_mod.ensure_hip_build_env()
    home = Path(os.environ["ROCM_HOME"])
    return [str(home / "include")], [str(home / "lib")]


def _runtime_paths() -> tuple[list[str], list[str], list[str], list[str]]:
    """(include_dirs, library_dirs, libraries, defines) for the host extensions."""
    if _is_rocm():
        include_dirs, library_dirs = _rocm_runtime_paths()
        return include_dirs, library_dirs, ["amdhip64"], ["__HIP_PLATFORM_AMD__=1"]
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is required to build freetoken.kernel._pinned_tensor "
            "because it links against the CUDA runtime API."
        )
    cuda_home = Path(CUDA_HOME)
    library_dirs = [str(cuda_home / "lib64")]
    if (cuda_home / "lib").exists():
        library_dirs.append(str(cuda_home / "lib"))
    return [str(cuda_home / "include")], library_dirs, ["cudart"], []


def _tvm_ffi_include_dir() -> str:
    spec = importlib.util.find_spec("tvm_ffi")
    if spec is None or spec.origin is None:
        raise RuntimeError("apache-tvm-ffi must be installed to build freetoken")
    return str(Path(spec.origin).parent / "include")


_check_toolchain()
cuda_include_dirs, cuda_library_dirs, runtime_libs, runtime_defines = _runtime_paths()
# device_compat.h (CUDA/HIP shim) lives with the kernel headers; dlpack comes
# from apache-tvm-ffi.
freetoken_include_dirs = (
    cuda_include_dirs + [str(KERNEL / "csrc" / "include"), _tvm_ffi_include_dir()]
)


setup(
    ext_modules=[
        CppExtension(
            name="freetoken.kernel._pinned_tensor",
            sources=[
                "python/freetoken/kernel/csrc/pinned_tensor.cpp",
            ],
            include_dirs=freetoken_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=runtime_libs,
            define_macros=[(d.split("=", 1)[0], d.split("=", 1)[1]) for d in runtime_defines],
            extra_compile_args=["-O3", "-std=c++17"],
        ),
        # CPU-compute MoE executor for --moe-backend cpu. Links cudart for the
        # cudaLaunchHostFunc submit/sync graph nodes; the bf16 GEMV microkernels
        # use per-function target attributes (avx512bf16/avx512f) + a runtime
        # __builtin_cpu_supports dispatch, so the single binary stays portable
        # (scalar fallback) -- no global -march is set.
        CppExtension(
            name="freetoken.kernel._cpu_moe",
            sources=[
                "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp",
            ],
            include_dirs=freetoken_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=runtime_libs,
            define_macros=[(d.split("=", 1)[0], d.split("=", 1)[1]) for d in runtime_defines],
            extra_compile_args=["-O3", "-std=c++17", "-pthread"],
        ),
        # --ple-backend disk row store; Linux-only until the TableFile/BatchReader seams grow Windows bodies
        *([
            CppExtension(
                name="freetoken.kernel._ple_store",
                sources=[
                    "python/freetoken/kernel/csrc/ple_store/ple_store_ext.cpp",
                ],
                extra_compile_args=["-O3", "-std=c++17"],
            )
        ] if sys.platform == "linux" else []),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
