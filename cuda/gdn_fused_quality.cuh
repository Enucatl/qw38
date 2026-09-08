#pragma once

// Warp-column fused GDN recurrence for Quartz OPT-019.
// Adapted from llama.cpp ggml/src/ggml-cuda/gated_delta_net.cu and
// gated_delta_net.cuh at cc83d7b4824f73cfdda4dfbb47ee39804f71b328
// (MIT, The ggml authors). Not a vendor of that file. ds4 has no GDN analog.
// Explicit expf/sqrtf are not compiler FMA contraction; Makefile NVCCFLAGS
// --fmad=false stays unchanged.

#include "gdn_step.h"

#include <cstdint>

namespace qw38::cuda {

constexpr int kGdnQualityWarps = 4;
constexpr int kGdnQualityThreads = 128;
constexpr int kGdnQualityWidth = 128;
constexpr float kGdnQualityL2Epsilon = 1.0e-6F;

// Warp sum with __fadd_rn. Butterfly association is admitted at OPT-013
// 5e-8 / 5e-9; ordered lane fold is the documented first repair if that misses.
inline __device__ float gdn_warp_sum(float value) {
  for (int mask = 16; mask > 0; mask >>= 1) {
    value = __fadd_rn(value, __shfl_xor_sync(0xffffffffU, value, mask));
  }
  return value;
}

// Grid (value_heads, 1, value_width/4); block (32, 4). Each warp owns one
// value column. s_shard[r] = S[r*32 + lane][col] as in llama.cpp S_v=128.
__global__ void __launch_bounds__(kGdnQualityThreads, 2)
prepare_recurrence_fused_warp_column(
    GdnConfig config, const float* convolution_output, const float* log_decay,
    const float* beta, const float* source, float* candidate, float* output,
    std::size_t token_count, bool value_is_tiled) {
  const std::uint32_t value_head = blockIdx.x;
  const int lane = static_cast<int>(threadIdx.x);
  const std::uint32_t col = blockIdx.z * 4U + threadIdx.y;
  if (value_head >= config.value_heads || col >= config.value_width) return;

  const std::uint32_t reuse = config.value_heads / config.key_heads;
  const std::uint32_t key_head = value_head / reuse;
  const std::size_t query_count =
      static_cast<std::size_t>(config.key_heads) * config.key_width;
  const std::size_t head_base = static_cast<std::size_t>(value_head) *
                                config.key_width * config.value_width;
  const std::size_t channels =
      2 * query_count +
      static_cast<std::size_t>(config.value_heads) * config.value_width;
  const std::uint32_t replica = value_head % reuse;
  const std::uint32_t tiled_head = replica * config.key_heads + key_head;
  const std::uint32_t source_head = value_is_tiled ? tiled_head : value_head;

  float s_shard[4];
#pragma unroll
  for (int row = 0; row < 4; ++row) {
    const std::uint32_t key_lane =
        static_cast<std::uint32_t>(row * 32 + lane);
    s_shard[row] = source[head_base + static_cast<std::size_t>(key_lane) *
                                          config.value_width +
                          col];
  }

  for (std::size_t token = 0; token < token_count; ++token) {
    const float* token_convolution = convolution_output + token * channels;
    const float* query = token_convolution + key_head * config.key_width;
    const float* key =
        token_convolution + query_count + key_head * config.key_width;
    const float* value = token_convolution + 2 * query_count +
                         source_head * config.value_width;

    float q_reg[4];
    float k_reg[4];
    float query_squares = 0.0F;
    float key_squares = 0.0F;
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      const int index = row * 32 + lane;
      q_reg[row] = query[index];
      k_reg[row] = key[index];
      query_squares =
          __fadd_rn(query_squares, __fmul_rn(q_reg[row], q_reg[row]));
      key_squares = __fadd_rn(key_squares, __fmul_rn(k_reg[row], k_reg[row]));
    }
    query_squares = gdn_warp_sum(query_squares);
    key_squares = gdn_warp_sum(key_squares);
    const float query_inverse =
        1.0F / sqrtf(query_squares + kGdnQualityL2Epsilon) /
        sqrtf(static_cast<float>(config.key_width));
    const float key_inverse = 1.0F / sqrtf(key_squares + kGdnQualityL2Epsilon);

    const float decay =
        expf(log_decay[token * config.value_heads + value_head]);
    const float beta_val = beta[token * config.value_heads + value_head];

    float kv_shard = 0.0F;
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      const float k_scale = __fmul_rn(k_reg[row], key_inverse);
      kv_shard = __fadd_rn(kv_shard, __fmul_rn(s_shard[row], k_scale));
    }
    const float kv_acc = gdn_warp_sum(kv_shard);
    const float delta =
        __fmul_rn(value[col] - __fmul_rn(decay, kv_acc), beta_val);

    float attn_shard = 0.0F;
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      const float k_scale = __fmul_rn(k_reg[row], key_inverse);
      const float q_scale = __fmul_rn(q_reg[row], query_inverse);
      s_shard[row] =
          __fadd_rn(__fmul_rn(decay, s_shard[row]), __fmul_rn(k_scale, delta));
      attn_shard = __fadd_rn(attn_shard, __fmul_rn(q_scale, s_shard[row]));
    }
    const float attn_acc = gdn_warp_sum(attn_shard);
    if (lane == 0) {
      output[(token * config.value_heads + value_head) * config.value_width +
             col] = attn_acc;
    }
  }

#pragma unroll
  for (int row = 0; row < 4; ++row) {
    const std::uint32_t key_lane =
        static_cast<std::uint32_t>(row * 32 + lane);
    candidate[head_base + static_cast<std::size_t>(key_lane) *
                              config.value_width +
              col] = s_shard[row];
  }
}

inline cudaError_t launch_gdn_quality_recurrence(
    const GdnConfig& config, const float* convolution_output,
    const float* log_decay, const float* beta, const float* source,
    float* candidate, float* output, std::size_t token_count,
    bool value_is_tiled, cudaStream_t stream) noexcept {
  const dim3 grid(config.value_heads, 1U, config.value_width / 4U);
  const dim3 block(32U, 4U, 1U);
  prepare_recurrence_fused_warp_column<<<grid, block, 0, stream>>>(
      config, convolution_output, log_decay, beta, source, candidate, output,
      token_count, value_is_tiled);
  return cudaPeekAtLastError();
}

}  // namespace qw38::cuda
