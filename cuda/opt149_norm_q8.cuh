#pragma once

// OPT-149 decode RMSNorm/residual fused into Q8_1 staging screen.
// Candidate `norm_to_q8_1_screen_v1` produces the exact quartz_q8_1_sum_q
// encoding used by mixer Q8/Q6 consumers, rounding through BF16 in
// registers before quantization. Standalone BF16 RMSNorm and
// launch_quantize_bf16_q8_1 remain the control. Mixer input_norm Q8_1
// groups only; FFN llama_q4k_mmvq / Q8Block / logits BF16 stay on the
// existing path. OPT-110 is not revived. Byte-size alone is not a win.

#include "rms_norm.cuh"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace qw38::cuda {
namespace opt149 {

constexpr char kControlId[] = "current_bf16_q8_staging";
constexpr char kCandidateId[] = "norm_to_q8_1_screen_v1";
constexpr char kLaunchRms[] = "rms_norm_fp32_to_q8_1";
constexpr char kLaunchResidual[] = "residual_add_norm_fp32_to_q8_1";
constexpr char kQ81Format[] = "quartz_q8_1_sum_q";
constexpr char kSelectedGroup[] = "mixer_input_norm_q8_1";
constexpr int kWidth = 5120;
constexpr int kThreads = 256;
constexpr int kWarp = 32;
constexpr int kQ8Block = 32;
constexpr int kMixerLayers = 64;
constexpr int kGdnLayers = 48;
constexpr int kAttentionLayers = 16;
constexpr int kProjCount = 4;
constexpr int kProjRows[kProjCount] = {8448, 5120, 48, 48};

// Exact quartz_q8_1_sum_q layout (half d, half sum(q), int8[32]). Distinct
// from FP32-scale Q8Block and from llama GPU sum(x) staging.
struct Q81Block {
  __half scale;
  __half q8_sum;
  std::int8_t values[32];
};
static_assert(sizeof(Q81Block) == 36, "Q8_1 block must stay 36 bytes");

// Production pin. Stay false until a keep sitting flips it.
constexpr bool kSelectedDecodeNormQ81Fusion = false;

inline thread_local int g_decode_norm_q81_override = -1;
inline thread_local int g_fused_launches = 0;
inline thread_local int g_control_launches = 0;
inline thread_local const char* g_last_launch = "";

inline bool legal_decode_norm_q81_ident(const char* path) noexcept {
  return path != nullptr &&
         (std::strcmp(path, kCandidateId) == 0 ||
          std::strcmp(path, kControlId) == 0 ||
          std::strcmp(path, "current_bf16_q8_staging") == 0 ||
          std::strcmp(path, "off") == 0 || std::strcmp(path, "on") == 0);
}

inline bool decode_norm_q81_fusion_enabled() noexcept {
  if (g_decode_norm_q81_override >= 0) return g_decode_norm_q81_override != 0;
  return kSelectedDecodeNormQ81Fusion;
}

inline void set_decode_norm_q81_fusion_override(bool enabled) noexcept {
  g_decode_norm_q81_override = enabled ? 1 : 0;
}

inline void clear_decode_norm_q81_fusion_override() noexcept {
  g_decode_norm_q81_override = -1;
}

inline bool apply_decode_norm_q81_ident(const char* path) noexcept {
  if (path != nullptr && std::strcmp(path, kCandidateId) == 0) {
    g_decode_norm_q81_override = 1;
    return true;
  }
  if (path != nullptr &&
      (std::strcmp(path, kControlId) == 0 || std::strcmp(path, "off") == 0)) {
    g_decode_norm_q81_override = 0;
    return true;
  }
  if (path != nullptr && std::strcmp(path, "on") == 0) {
    g_decode_norm_q81_override = 1;
    return true;
  }
  return false;
}

inline const char* effective_decode_norm_q81_path() noexcept {
  return decode_norm_q81_fusion_enabled() ? kCandidateId : kControlId;
}

// Eligible when every live consumer is Q8_0×Q8_1. BF16 or Q8Block mixed
// consumers retain the existing BF16 materialization + restage path.
inline bool mixer_group_q8_1_eligible(const bool* q8_0_consumer,
                                      const bool* q8block_consumer,
                                      const bool* bf16_consumer,
                                      int count) noexcept {
  if (count <= 0) return false;
  bool any = false;
  for (int index = 0; index < count; ++index) {
    if (q8block_consumer != nullptr && q8block_consumer[index]) return false;
    if (bf16_consumer != nullptr && bf16_consumer[index]) return false;
    if (q8_0_consumer != nullptr && q8_0_consumer[index]) any = true;
  }
  return any;
}

struct DecodeNormQ81Scope final {
  explicit DecodeNormQ81Scope(bool enabled) noexcept {
    set_decode_norm_q81_fusion_override(enabled);
  }
  ~DecodeNormQ81Scope() { clear_decode_norm_q81_fusion_override(); }
  DecodeNormQ81Scope(const DecodeNormQ81Scope&) = delete;
  DecodeNormQ81Scope& operator=(const DecodeNormQ81Scope&) = delete;
};

#if defined(__CUDACC__) && !defined(QW38_OPT149_HOST_ONLY)

template <bool UseRsqrt, bool WriteBf16>
__global__ void __launch_bounds__(kThreads, 1)
rms_norm_fp32_to_q8_1(const float* input, const float* scale, std::size_t count,
                      Q81Block* q8, __nv_bfloat16* normalized) {
  const float inverse = cooperative_rms_inverse_fp32<UseRsqrt>(input, count);
  const int lane = static_cast<int>(threadIdx.x) & (kWarp - 1);
  const int warp = static_cast<int>(threadIdx.x) / kWarp;
  const int nwarps = static_cast<int>(blockDim.x) / kWarp;
  const std::size_t nblocks = count / static_cast<std::size_t>(kQ8Block);
  for (std::size_t block = static_cast<std::size_t>(warp); block < nblocks;
       block += static_cast<std::size_t>(nwarps)) {
    const std::size_t index =
        block * static_cast<std::size_t>(kQ8Block) + static_cast<std::size_t>(lane);
    const __nv_bfloat16 stored =
        rms_norm_store_bf16(input[index], inverse, scale[index]);
    if constexpr (WriteBf16) {
      if (normalized != nullptr) normalized[index] = stored;
    }
    const float value = __bfloat162float(stored);
    float maximum = fabsf(value);
    for (int offset = 16; offset > 0; offset /= 2) {
      maximum = fmaxf(maximum,
                      __shfl_down_sync(0xFFFFFFFFU, maximum, offset, kWarp));
    }
    maximum = __shfl_sync(0xFFFFFFFFU, maximum, 0, kWarp);
    const float qscale = maximum == 0.0F ? 0.0F : maximum / 127.0F;
    const std::int8_t quant =
        qscale == 0.0F ? 0 : static_cast<std::int8_t>(roundf(value / qscale));
    int q_sum = static_cast<int>(quant);
    for (int offset = 16; offset > 0; offset /= 2) {
      q_sum += __shfl_down_sync(0xFFFFFFFFU, q_sum, offset, kWarp);
    }
    q8[block].values[lane] = quant;
    if (lane == 0) {
      q8[block].scale = __float2half_rn(qscale);
      q8[block].q8_sum = __float2half_rn(static_cast<float>(q_sum));
    }
  }
}

template <bool UseRsqrt, bool WriteBf16>
__global__ void __launch_bounds__(kThreads, 1)
residual_add_norm_fp32_to_q8_1(const float* residual, const float* correction,
                               const float* scale, std::size_t count,
                               float* output, Q81Block* q8,
                               __nv_bfloat16* normalized) {
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    output[index] = __fadd_rn(residual[index], correction[index]);
  }
  __syncthreads();
  const float inverse = cooperative_rms_inverse_fp32<UseRsqrt>(output, count);
  const int lane = static_cast<int>(threadIdx.x) & (kWarp - 1);
  const int warp = static_cast<int>(threadIdx.x) / kWarp;
  const int nwarps = static_cast<int>(blockDim.x) / kWarp;
  const std::size_t nblocks = count / static_cast<std::size_t>(kQ8Block);
  for (std::size_t block = static_cast<std::size_t>(warp); block < nblocks;
       block += static_cast<std::size_t>(nwarps)) {
    const std::size_t index =
        block * static_cast<std::size_t>(kQ8Block) + static_cast<std::size_t>(lane);
    const __nv_bfloat16 stored =
        rms_norm_store_bf16(output[index], inverse, scale[index]);
    if constexpr (WriteBf16) {
      if (normalized != nullptr) normalized[index] = stored;
    }
    const float value = __bfloat162float(stored);
    float maximum = fabsf(value);
    for (int offset = 16; offset > 0; offset /= 2) {
      maximum = fmaxf(maximum,
                      __shfl_down_sync(0xFFFFFFFFU, maximum, offset, kWarp));
    }
    maximum = __shfl_sync(0xFFFFFFFFU, maximum, 0, kWarp);
    const float qscale = maximum == 0.0F ? 0.0F : maximum / 127.0F;
    const std::int8_t quant =
        qscale == 0.0F ? 0 : static_cast<std::int8_t>(roundf(value / qscale));
    int q_sum = static_cast<int>(quant);
    for (int offset = 16; offset > 0; offset /= 2) {
      q_sum += __shfl_down_sync(0xFFFFFFFFU, q_sum, offset, kWarp);
    }
    q8[block].values[lane] = quant;
    if (lane == 0) {
      q8[block].scale = __float2half_rn(qscale);
      q8[block].q8_sum = __float2half_rn(static_cast<float>(q_sum));
    }
  }
}

inline dim3 fusion_block() noexcept {
  const int threads = effective_rms_norm_threads();
  return dim3(static_cast<unsigned int>(
      legal_rms_norm_threads(threads) ? threads : kThreads));
}

inline cudaError_t launch_rms_norm_fp32_to_q8_1(
    const float* input, const float* scale, std::size_t count, void* q8,
    __nv_bfloat16* normalized, bool write_bf16, cudaStream_t stream) noexcept {
  if (input == nullptr || scale == nullptr || q8 == nullptr || count == 0 ||
      count % static_cast<std::size_t>(kQ8Block) != 0) {
    return cudaErrorInvalidValue;
  }
  if (write_bf16 && normalized == nullptr) return cudaErrorInvalidValue;
  auto* blocks = static_cast<Q81Block*>(q8);
  const dim3 block = fusion_block();
  const bool rsqrt = rms_norm_uses_rsqrt();
  ++g_fused_launches;
  g_last_launch = kLaunchRms;
  if (rsqrt && write_bf16) {
    rms_norm_fp32_to_q8_1<true, true>
        <<<1, block, 0, stream>>>(input, scale, count, blocks, normalized);
  } else if (rsqrt) {
    rms_norm_fp32_to_q8_1<true, false>
        <<<1, block, 0, stream>>>(input, scale, count, blocks, normalized);
  } else if (write_bf16) {
    rms_norm_fp32_to_q8_1<false, true>
        <<<1, block, 0, stream>>>(input, scale, count, blocks, normalized);
  } else {
    rms_norm_fp32_to_q8_1<false, false>
        <<<1, block, 0, stream>>>(input, scale, count, blocks, normalized);
  }
  return cudaPeekAtLastError();
}

inline cudaError_t launch_residual_add_norm_fp32_to_q8_1(
    const float* residual, const float* correction, const float* scale,
    std::size_t count, float* output, void* q8, __nv_bfloat16* normalized,
    bool write_bf16, cudaStream_t stream) noexcept {
  if (residual == nullptr || correction == nullptr || scale == nullptr ||
      output == nullptr || q8 == nullptr || count == 0 ||
      count % static_cast<std::size_t>(kQ8Block) != 0) {
    return cudaErrorInvalidValue;
  }
  if (write_bf16 && normalized == nullptr) return cudaErrorInvalidValue;
  auto* blocks = static_cast<Q81Block*>(q8);
  const dim3 block = fusion_block();
  const bool rsqrt = rms_norm_uses_rsqrt();
  ++g_fused_launches;
  g_last_launch = kLaunchResidual;
  if (rsqrt && write_bf16) {
    residual_add_norm_fp32_to_q8_1<true, true>
        <<<1, block, 0, stream>>>(residual, correction, scale, count, output,
                                  blocks, normalized);
  } else if (rsqrt) {
    residual_add_norm_fp32_to_q8_1<true, false>
        <<<1, block, 0, stream>>>(residual, correction, scale, count, output,
                                  blocks, normalized);
  } else if (write_bf16) {
    residual_add_norm_fp32_to_q8_1<false, true>
        <<<1, block, 0, stream>>>(residual, correction, scale, count, output,
                                  blocks, normalized);
  } else {
    residual_add_norm_fp32_to_q8_1<false, false>
        <<<1, block, 0, stream>>>(residual, correction, scale, count, output,
                                  blocks, normalized);
  }
  return cudaPeekAtLastError();
}

inline cudaError_t kernel_attributes(int* regs, std::size_t* local,
                                     int* occupancy) noexcept {
  cudaFuncAttributes attrs{};
  const cudaError_t error =
      cudaFuncGetAttributes(&attrs, rms_norm_fp32_to_q8_1<false, false>);
  if (error != cudaSuccess) return error;
  if (regs != nullptr) *regs = attrs.numRegs;
  if (local != nullptr) *local = static_cast<std::size_t>(attrs.localSizeBytes);
  if (occupancy != nullptr) {
    const cudaError_t occ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        occupancy, rms_norm_fp32_to_q8_1<false, false>, kThreads, 0);
    if (occ != cudaSuccess) return occ;
  }
  return cudaSuccess;
}

#endif  // __CUDACC__

}  // namespace opt149
}  // namespace qw38::cuda
