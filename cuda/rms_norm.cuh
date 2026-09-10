#pragma once

// OPT-045 cooperative RMSNorm helpers. Serial kernels in full_scheduler.cu and
// scheduler_primitives.cu remain the strict reference. Production selects a
// path pin; A/B may override it on the calling thread. Square accumulation
// uses explicit fmaf. The epilogue keeps (x * inverse) * scale order and one
// BF16 rounding. Epsilon stays 1e-6. Reference: pinned llama
// ggml/src/ggml-cuda/norm.cu::rms_norm_f32 (cc83d7b, MIT).

#include <cstring>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace qw38::cuda {

constexpr char kLegalRmsNormPathSerial[] = "serial";
constexpr char kLegalRmsNormPathParallelFma[] = "parallel_fma";
constexpr char kLegalRmsNormPathParallelRsqrt[] = "parallel_rsqrt";

// Production pin. Keep sitting may switch away from serial; reject restores it.
constexpr char kSelectedRmsNormPath[] = "parallel_fma";
constexpr int kSelectedRmsNormThreads = 256;
constexpr int kSelectedGdnNormThreads = 32;

constexpr float kRmsNormEpsilon = 1.0e-6F;
constexpr int kRmsNormWarpSize = 32;
constexpr int kRmsNormMaxWarps = 32;

inline bool legal_rms_norm_path(const char* path) noexcept {
  return path != nullptr &&
         (std::strcmp(path, kLegalRmsNormPathSerial) == 0 ||
          std::strcmp(path, kLegalRmsNormPathParallelFma) == 0 ||
          std::strcmp(path, kLegalRmsNormPathParallelRsqrt) == 0);
}

inline bool legal_rms_norm_threads(int threads) noexcept {
  return threads == 32 || threads == 128 || threads == 256;
}

inline thread_local const char* g_rms_norm_path_override = nullptr;
inline thread_local int g_rms_norm_threads_override = 0;
inline thread_local int g_gdn_norm_threads_override = 0;

inline const char* selected_rms_norm_path() noexcept {
  return kSelectedRmsNormPath;
}

inline int selected_rms_norm_threads() noexcept {
  return kSelectedRmsNormThreads;
}

inline int selected_gdn_norm_threads() noexcept {
  return kSelectedGdnNormThreads;
}

inline const char* effective_rms_norm_path() noexcept {
  return g_rms_norm_path_override != nullptr ? g_rms_norm_path_override
                                             : kSelectedRmsNormPath;
}

inline int effective_rms_norm_threads() noexcept {
  return g_rms_norm_threads_override > 0 ? g_rms_norm_threads_override
                                         : kSelectedRmsNormThreads;
}

inline int effective_gdn_norm_threads() noexcept {
  return g_gdn_norm_threads_override > 0 ? g_gdn_norm_threads_override
                                         : kSelectedGdnNormThreads;
}

inline void set_rms_norm_path_override(const char* path, int residual_threads,
                                       int gdn_threads) noexcept {
  g_rms_norm_path_override = path;
  g_rms_norm_threads_override = residual_threads;
  g_gdn_norm_threads_override = gdn_threads;
}

inline void clear_rms_norm_path_override() noexcept {
  g_rms_norm_path_override = nullptr;
  g_rms_norm_threads_override = 0;
  g_gdn_norm_threads_override = 0;
}

inline bool rms_norm_path_is_serial(const char* path) noexcept {
  return path != nullptr &&
         std::strcmp(path, kLegalRmsNormPathSerial) == 0;
}

inline bool rms_norm_path_is_rsqrt(const char* path) noexcept {
  return path != nullptr &&
         std::strcmp(path, kLegalRmsNormPathParallelRsqrt) == 0;
}

inline bool rms_norm_uses_serial() noexcept {
  return rms_norm_path_is_serial(effective_rms_norm_path());
}

inline bool rms_norm_uses_rsqrt() noexcept {
  return rms_norm_path_is_rsqrt(effective_rms_norm_path());
}

struct RmsNormPathScope final {
  RmsNormPathScope(const char* path, int residual_threads,
                   int gdn_threads) noexcept {
    set_rms_norm_path_override(path, residual_threads, gdn_threads);
  }
  ~RmsNormPathScope() { clear_rms_norm_path_override(); }
  RmsNormPathScope(const RmsNormPathScope&) = delete;
  RmsNormPathScope& operator=(const RmsNormPathScope&) = delete;
};

#if defined(__CUDACC__)

template <bool UseRsqrt>
__device__ __forceinline__ float rms_inverse_from_sum(float sum,
                                                      std::size_t width) {
  const float mean = sum / static_cast<float>(width) + kRmsNormEpsilon;
  if constexpr (UseRsqrt) {
    return rsqrtf(mean);
  }
  return 1.0F / sqrtf(mean);
}

template <bool UseRsqrt>
__device__ __forceinline__ float cooperative_rms_inverse(float partial,
                                                         std::size_t width) {
  const int lane = threadIdx.x & (kRmsNormWarpSize - 1);
  const int warp = threadIdx.x / kRmsNormWarpSize;
  const int nwarps =
      (blockDim.x + kRmsNormWarpSize - 1) / kRmsNormWarpSize;
  for (int offset = kRmsNormWarpSize / 2; offset > 0; offset /= 2) {
    partial += __shfl_down_sync(0xFFFFFFFFU, partial, offset,
                                kRmsNormWarpSize);
  }
  __shared__ float warp_sums[kRmsNormMaxWarps];
  __shared__ float inverse;
  if (lane == 0) warp_sums[warp] = partial;
  __syncthreads();
  float block = 0.0F;
  if (threadIdx.x < nwarps) block = warp_sums[threadIdx.x];
  if (threadIdx.x < kRmsNormWarpSize) {
    for (int offset = kRmsNormWarpSize / 2; offset > 0; offset /= 2) {
      block += __shfl_down_sync(0xFFFFFFFFU, block, offset,
                                kRmsNormWarpSize);
    }
    if (threadIdx.x == 0) {
      inverse = rms_inverse_from_sum<UseRsqrt>(block, width);
    }
  }
  __syncthreads();
  return inverse;
}

template <bool UseRsqrt>
__device__ __forceinline__ float cooperative_rms_inverse_fp32(
    const float* row, std::size_t width) {
  float sum = 0.0F;
  for (std::size_t index = threadIdx.x; index < width;
       index += blockDim.x) {
    const float value = row[index];
    sum = fmaf(value, value, sum);
  }
  return cooperative_rms_inverse<UseRsqrt>(sum, width);
}

template <bool UseRsqrt>
__device__ __forceinline__ float cooperative_rms_inverse_bf16(
    const __nv_bfloat16* row, std::size_t width) {
  float sum = 0.0F;
  for (std::size_t index = threadIdx.x; index < width;
       index += blockDim.x) {
    const float value = __bfloat162float(row[index]);
    sum = fmaf(value, value, sum);
  }
  return cooperative_rms_inverse<UseRsqrt>(sum, width);
}

__device__ __forceinline__ __nv_bfloat16
rms_norm_store_bf16(float value, float inverse, float scale) {
  return __float2bfloat16_rn(
      __fmul_rn(__fmul_rn(value, inverse), scale));
}

#endif  // __CUDACC__

}  // namespace qw38::cuda
