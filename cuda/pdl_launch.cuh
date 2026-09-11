#pragma once

// Hopper/Blackwell programmatic dependent launch (PDL) for Quartz OPT-030.
// Adapted from llama.cpp ggml/src/ggml-cuda/common.cuh at
// cc83d7b4824f73cfdda4dfbb47ee39804f71b328 (MIT, The ggml authors):
// ggml_cuda_kernel_launch / ggml_cuda_pdl_config /
// ggml_cuda_kernel_can_use_pdl / ggml_cuda_pdl_sync / ggml_cuda_pdl_lc.
// Provenance is that launch protocol (cudaLaunchKernelEx + a single
// cudaLaunchAttributeProgrammaticStreamSerialization attribute, plus
// device cudaGridDependencySynchronize / cudaTriggerProgrammaticLaunchCompletion
// for __CUDA_ARCH__ >= 900). Not a vendor of common.cuh. Do not include
// ggml headers. Do not honor GGML_CUDA_PDL. Quartz uses a compile-time
// enumerator plus a paired A/B. PDL is forbidden while a stream is
// capturing (OPT-012 FFN graphs stay ordinary kernel nodes).

#include <cstddef>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>

#include <cuda_runtime.h>

namespace qw38::cuda {

// OPT-030 A/B winner. Legal values: off, pdl_host, pdl. Default until A/B: off.
constexpr char kSelectedPdlPath[] = "off";

constexpr bool kPdlDevicePrimitivesSelected =
    kSelectedPdlPath[0] == 'p' && kSelectedPdlPath[1] == 'd' &&
    kSelectedPdlPath[2] == 'l' && kSelectedPdlPath[3] == '\0';

// Per-TU without RDC. Launch TUs register a setter that copies this TU's flag.
#if defined(__CUDACC__)
static __device__ int quartz_pdl_device_ops = 0;
#endif

inline thread_local int g_pdl_scope_depth = 0;
inline thread_local const char* g_pdl_path_override = nullptr;
inline thread_local bool g_pdl_last_used_ex = false;

inline bool pdl_path_eq(const char* path, const char* want) noexcept {
  return path != nullptr && want != nullptr && std::strcmp(path, want) == 0;
}

inline const char* selected_pdl_path() noexcept { return kSelectedPdlPath; }

inline const char* effective_pdl_path() noexcept {
  return g_pdl_path_override != nullptr ? g_pdl_path_override : kSelectedPdlPath;
}

inline void quartz_set_pdl_path_override(const char* path) noexcept {
  g_pdl_path_override = path;
}

inline bool pdl_uses_launch_ex() noexcept {
  return pdl_path_eq(kSelectedPdlPath, "pdl_host") ||
         pdl_path_eq(kSelectedPdlPath, "pdl");
}

inline bool pdl_uses_device_primitives() noexcept {
  return pdl_path_eq(kSelectedPdlPath, "pdl");
}

inline bool pdl_path_uses_launch_ex(const char* path) noexcept {
  return pdl_path_eq(path, "pdl_host") || pdl_path_eq(path, "pdl");
}

inline bool pdl_path_uses_device_primitives(const char* path) noexcept {
  return pdl_path_eq(path, "pdl");
}

inline bool pdl_scope_is_active() noexcept { return g_pdl_scope_depth > 0; }

inline bool pdl_last_used_launch_ex() noexcept { return g_pdl_last_used_ex; }

struct PdlScope final {
  PdlScope() noexcept { ++g_pdl_scope_depth; }
  ~PdlScope() noexcept {
    if (g_pdl_scope_depth > 0) --g_pdl_scope_depth;
  }
  PdlScope(const PdlScope&) = delete;
  PdlScope& operator=(const PdlScope&) = delete;
};

inline bool pdl_stream_is_capturing(cudaStream_t stream) noexcept {
  if (stream == nullptr) return false;
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  unsigned long long id = 0;
  const cudaError_t error = cudaStreamGetCaptureInfo(stream, &status, &id);
  if (error != cudaSuccess) return true;
  return status == cudaStreamCaptureStatusActive;
}

inline bool pdl_kernel_can_use(const void* kernel) noexcept {
  if (kernel == nullptr) return false;
  static std::mutex cache_mutex;
  static std::unordered_map<const void*, bool> cache;
  std::lock_guard<std::mutex> lock(cache_mutex);
  const auto it = cache.find(kernel);
  if (it != cache.end()) return it->second;
  cudaFuncAttributes attr{};
  const cudaError_t error = cudaFuncGetAttributes(&attr, kernel);
  const bool can_use = error == cudaSuccess && attr.ptxVersion >= 90;
  cache.emplace(kernel, can_use);
  return can_use;
}

using PdlDeviceOpsSetter = cudaError_t (*)(bool) noexcept;

inline PdlDeviceOpsSetter* pdl_device_ops_slots(int** count) noexcept {
  static PdlDeviceOpsSetter setters[16]{};
  static int n = 0;
  if (count != nullptr) *count = &n;
  return setters;
}

inline void pdl_register_device_ops_setter(PdlDeviceOpsSetter setter) noexcept {
  if (setter == nullptr) return;
  int* n = nullptr;
  PdlDeviceOpsSetter* setters = pdl_device_ops_slots(&n);
  if (n == nullptr || *n >= 16) return;
  for (int index = 0; index < *n; ++index) {
    if (setters[index] == setter) return;
  }
  setters[*n] = setter;
  ++(*n);
}

inline cudaError_t quartz_set_pdl_device_ops(bool enabled) noexcept {
  int* n = nullptr;
  PdlDeviceOpsSetter* setters = pdl_device_ops_slots(&n);
  cudaError_t error = cudaSuccess;
  const int count = n == nullptr ? 0 : *n;
  for (int index = 0; index < count; ++index) {
    if (setters[index] == nullptr) continue;
    const cudaError_t next = setters[index](enabled);
    if (error == cudaSuccess) error = next;
  }
  return error;
}

static __device__ __forceinline__ void quartz_pdl_sync() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (kPdlDevicePrimitivesSelected) {
    cudaGridDependencySynchronize();
  } else if (quartz_pdl_device_ops != 0) {
    cudaGridDependencySynchronize();
  }
#endif
}

static __device__ __forceinline__ void quartz_pdl_lc() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (kPdlDevicePrimitivesSelected) {
    cudaTriggerProgrammaticLaunchCompletion();
  } else if (quartz_pdl_device_ops != 0) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}

template <typename Kernel, typename... Args>
cudaError_t quartz_launch_kernel(Kernel kernel, dim3 grid, dim3 block,
                                 std::size_t shared, cudaStream_t stream,
                                 Args&&... args) {
  g_pdl_last_used_ex = false;
  const char* path = effective_pdl_path();
  const bool want_ex = pdl_path_uses_launch_ex(path) && pdl_scope_is_active() &&
                       !pdl_stream_is_capturing(stream) &&
                       pdl_kernel_can_use(reinterpret_cast<const void*>(kernel));
  if (want_ex) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = shared;
    cfg.stream = stream;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    const cudaError_t error =
        cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...);
    g_pdl_last_used_ex = error == cudaSuccess;
    return error;
  }
#if defined(__CUDACC__)
  kernel<<<grid, block, shared, stream>>>(std::forward<Args>(args)...);
  return cudaGetLastError();
#else
  (void)kernel;
  (void)grid;
  (void)block;
  (void)shared;
  (void)stream;
  return cudaErrorInvalidDeviceFunction;
#endif
}

}  // namespace qw38::cuda

#define QW38_PDL_REGISTER_DEVICE_OPS()                                        \
  namespace {                                                                 \
  cudaError_t qw38_pdl_set_device_ops_tu(bool enabled) noexcept {             \
    const int value = enabled ? 1 : 0;                                        \
    return cudaMemcpyToSymbol(qw38::cuda::quartz_pdl_device_ops, &value,      \
                              sizeof(value));                                 \
  }                                                                           \
  struct Qw38PdlDeviceOpsReg final {                                          \
    Qw38PdlDeviceOpsReg() {                                                   \
      qw38::cuda::pdl_register_device_ops_setter(&qw38_pdl_set_device_ops_tu); \
    }                                                                         \
  };                                                                          \
  const Qw38PdlDeviceOpsReg qw38_pdl_device_ops_reg{};                        \
  }
