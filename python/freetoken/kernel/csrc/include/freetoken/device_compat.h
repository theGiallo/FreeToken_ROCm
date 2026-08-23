#pragma once

// Platform shim: freetoken kernels are written against the CUDA runtime API.
// Under hipcc (`-D__HIP_PLATFORM_AMD__=1`, added automatically by tvm-ffi's
// HIP backend) this header maps the exact subset freetoken uses onto HIP, so
// the same sources compile unchanged on ROCm. Include it before any cuda*
// reference (freetoken/utils.cuh does; host extensions include it directly).
//
// Not covered here on purpose:
// - cudaMemcpyBatchAsync (CUDA >= 13 only): batch_memcpy.cuh gates on its own,
// - PDL launch attributes (no HIP equivalent): utils.cuh rejects use_pdl=true,
// - NVIDIA PTX inline asm: fast_index_copy.cuh carries HIP-native branches.

#ifdef __HIP_PLATFORM_AMD__

#include <dlpack/dlpack.h>
#include <hip/hip_runtime.h>

// DLPack device codes reported by torch under ROCm; ported kernels match
// tensors against these instead of the hard-coded kDLCUDA/kDLCUDAHost.
inline constexpr DLDeviceType kFTGpuDevice = kDLROCM;
inline constexpr DLDeviceType kFTGpuHostDevice = kDLROCMHost;

// ---------------------------------------------------------------------------
// CUDA runtime API -> HIP mapping for the subset freetoken uses.
using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;

inline constexpr cudaError_t cudaSuccess = hipSuccess;

inline auto cudaGetErrorString(cudaError_t error) -> const char * {
  return hipGetErrorString(error);
}

inline auto cudaGetDevice(int *device) -> cudaError_t {
  return hipGetDevice(device);
}

inline auto cudaDriverGetVersion(int *version) -> cudaError_t {
  return hipDriverGetVersion(version);
}

inline auto
cudaDeviceGetAttribute(int *value, hipDeviceAttribute_t attr, int device)
    -> cudaError_t {
  return hipDeviceGetAttribute(value, attr, device);
}

inline constexpr auto cudaDevAttrUnifiedAddressing =
    hipDeviceAttributeUnifiedAddressing;
inline constexpr auto cudaDevAttrCanUseHostPointerForRegisteredMem =
    hipDeviceAttributeCanUseHostPointerForRegisteredMem;

// pinned / mapped host memory
inline constexpr auto cudaHostAllocMapped = hipHostMallocMapped;
inline constexpr auto cudaHostAllocPortable = hipHostMallocPortable;
inline constexpr auto cudaHostRegisterMapped = hipHostRegisterMapped;
inline constexpr auto cudaHostRegisterPortable = hipHostRegisterPortable;

inline auto cudaHostAlloc(void **ptr, std::size_t size, unsigned flags)
    -> cudaError_t {
  return hipHostMalloc(ptr, size, flags);
}
inline auto cudaMallocHost(void **ptr, std::size_t size) -> cudaError_t {
  return hipHostMalloc(ptr, size, hipHostMallocDefault);
}
inline auto cudaFreeHost(void *ptr) -> cudaError_t {
  return hipHostFree(ptr);
}
inline auto cudaHostRegister(void *ptr, std::size_t size, unsigned flags)
    -> cudaError_t {
  return hipHostRegister(ptr, size, flags);
}
inline auto cudaHostGetDevicePointer(void **device_ptr, void *host_ptr,
                                     unsigned flags) -> cudaError_t {
  return hipHostGetDevicePointer(device_ptr, host_ptr, flags);
}

// streams / host callbacks / memcpy
inline auto cudaStreamSynchronize(cudaStream_t stream) -> cudaError_t {
  return hipStreamSynchronize(stream);
}
inline auto
cudaLaunchHostFunc(cudaStream_t stream, void (*callback)(void *), void *userData)
    -> cudaError_t {
  return hipLaunchHostFunc(stream, callback, userData);
}
inline auto
cudaMemcpyAsync(void *dst, const void *src, std::size_t sizeBytes,
                hipMemcpyKind kind, cudaStream_t stream = nullptr)
    -> cudaError_t {
  return hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
}
inline constexpr auto cudaMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
inline constexpr auto cudaMemcpyDeviceToHost = hipMemcpyDeviceToHost;
inline constexpr auto cudaMemcpyHostToDevice = hipMemcpyHostToDevice;

// per-function attributes (dynamic shared memory opt-in)
inline constexpr auto cudaFuncAttributeMaxDynamicSharedMemorySize =
    hipFuncAttributeMaxDynamicSharedMemorySize;
template <typename K>
inline auto
cudaFuncSetAttribute(K kernel, int attr, int value) -> cudaError_t {
  return hipFuncSetAttribute(reinterpret_cast<const void *>(kernel),
                             static_cast<hipFuncAttribute>(attr), value);
}

inline auto cudaGetLastError() -> cudaError_t { return hipGetLastError(); }

// CUDA headers define CUDART_CB as an empty marker on host callbacks.
#ifndef CUDART_CB
#define CUDART_CB
#endif

// clang-hip does not implement CUDA's __grid_constant__ parameter qualifier;
// by-value struct params are semantically equivalent for these kernels.
#define FT_GRID_CONSTANT

// launch config: freetoken never sets PDL attrs on ROCm (utils.cuh rejects),
// so the attribute cache stays inert and the launch lowers to a plain
// triple-chevron dispatch. Only reachable under hipcc (__HIP__): the host
// extensions compile with plain c++, which cannot parse <<<>>>.
struct cudaLaunchAttribute {
  int id;
  union {
    int programmaticStreamSerializationAllowed;
  } val;
};
inline constexpr auto cudaLaunchAttributeProgrammaticStreamSerialization = 1;

struct cudaLaunchConfig_t {
  dim3 gridDim;
  dim3 blockDim;
  std::size_t dynamicSmemBytes;
  cudaStream_t stream;
  cudaLaunchAttribute *attrs;
  int numAttrs;
};

#ifdef __HIP__
template <typename K, typename... Args>
inline auto
cudaLaunchKernelEx(const cudaLaunchConfig_t *config, K kernel, Args &&...args)
    -> cudaError_t {
  kernel<<<config->gridDim, config->blockDim, config->dynamicSmemBytes,
           config->stream>>>(std::forward<Args>(args)...);
  return hipGetLastError();
}
#endif // __HIP__

#else // !__HIP_PLATFORM_AMD__

#include <cuda_runtime_api.h>

inline constexpr DLDeviceType kFTGpuDevice = kDLCUDA;
inline constexpr DLDeviceType kFTGpuHostDevice = kDLCUDAHost;

#define FT_GRID_CONSTANT __grid_constant__

#endif // __HIP_PLATFORM_AMD__
