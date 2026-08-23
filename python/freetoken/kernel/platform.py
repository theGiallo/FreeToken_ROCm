"""GPU platform detection and HIP build-environment setup.

On ROCm, freetoken builds its kernels against the ROCm SDK shipped as pip
wheels (``rocm_sdk_core``/``rocm_sdk_libraries``, installed alongside an AMD
torch build such as ``torch[device-gfx1100]`` from repo.amd.com). Those wheels
do not provide a global ``/opt/rocm`` tree and ship shared libraries without
the unversioned ``lib<name>.so`` symlinks linkers expect, so
:func:`ensure_hip_build_env` prepares the environment before any JIT/AOT
kernel build:

- resolves the SDK root and exports ``ROCM_HOME``, ``PATH`` and
  ``LD_LIBRARY_PATH`` so ``tvm-ffi`` finds ``hipcc`` and the HIP libraries,
- creates the missing unversioned symlinks next to the versioned libraries
  (best-effort; skipped when the filesystem is read-only),
- exports ``TVM_FFI_ROCM_ARCH_LIST`` from the detected GPU architecture.
"""

from __future__ import annotations

import functools
import os
import pathlib

# Libraries whose unversioned `.so` linker names tvm-ffi's HIP backend links
# or dlopens by name; the pip SDK wheels only ship `<name>.so.<N>`.
_HIP_LIBS_NEEDING_LINKER_SYMLINK = ("libamdhip64", "librocdxg", "libhsa-runtime64")


def is_rocm() -> bool:
    """True when torch runs against ROCm/HIP (as opposed to CUDA or CPU-only)."""
    import torch

    return bool(getattr(torch.version, "hip", None))


def hip_major() -> int | None:
    """Major version of the ROCm stack torch was built against, e.g. 7."""
    import torch

    hip = getattr(torch.version, "hip", None)
    return int(hip.split(".")[0]) if hip else None


def gfx_arch() -> str | None:
    """The raw GCN architecture of device 0 (e.g. ``gfx1100``), or None."""
    import torch

    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(0)
        name = getattr(props, "gcnArchName", "") or ""
        return name.split(":")[0] or None
    except Exception:
        return None


def find_pip_rocm_sdk_root() -> pathlib.Path | None:
    """Locate the ``_rocm_sdk_core`` payload directory of the pip ROCm SDK."""
    try:
        import rocm_sdk_core  # noqa: PLC0415
    except ModuleNotFoundError:
        return None
    candidate = pathlib.Path(rocm_sdk_core.__file__).parent.parent / "_rocm_sdk_core"
    if (candidate / "bin" / "hipcc").exists():
        return candidate
    return None


@functools.cache
def ensure_hip_build_env() -> None:
    """Make the HIP toolchain discoverable for tvm-ffi kernel builds.

    Idempotent. A user-provided ``ROCM_HOME``/``ROCM_PATH`` always wins (system
    ROCm install). Without one, the pip-bundled SDK is used when available.
    """
    home = os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH")
    sdk_root: pathlib.Path | None = pathlib.Path(home) if home else find_pip_rocm_sdk_root()
    if sdk_root is None or not (sdk_root / "bin" / "hipcc").exists():
        raise RuntimeError(
            "HIP kernel build requested but no usable ROCm installation found. "
            "Install torch built for ROCm (e.g. pip install "
            "'torch[device-gfx1100]' --index-url "
            "https://repo.amd.com/rocm/whl-multi-arch/) or set ROCM_HOME."
        )

    lib_dir = sdk_root / "lib"
    _create_linker_symlinks(lib_dir)

    os.environ.setdefault("ROCM_HOME", str(sdk_root))
    os.environ["PATH"] = f"{sdk_root / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if str(lib_dir) not in ld.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}{os.pathsep}{ld}" if ld else str(lib_dir)

    if "TVM_FFI_ROCM_ARCH_LIST" not in os.environ:
        arch = gfx_arch()
        if arch:
            os.environ["TVM_FFI_ROCM_ARCH_LIST"] = arch


def _create_linker_symlinks(lib_dir: pathlib.Path) -> None:
    for base in _HIP_LIBS_NEEDING_LINKER_SYMLINK:
        symlink = lib_dir / f"{base}.so"
        if symlink.exists():
            continue
        candidates = [
            p
            for p in lib_dir.glob(f"{base}.so.*")
            if not p.name.endswith(".so")
        ]
        if not candidates:
            continue

        def _version_key(p: pathlib.Path) -> tuple[int, int]:
            last = p.suffixes[-1].lstrip(".")
            return (0, int(last)) if last.isdigit() else (1, 0)

        target = max(candidates, key=_version_key)
        try:
            symlink.symlink_to(target.name)
        except OSError:
            pass
