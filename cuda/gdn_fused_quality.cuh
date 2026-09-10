#pragma once

// Warp-column fused GDN recurrence for Quartz OPT-019 / OPT-029 / OPT-040
// plus the OPT-052 preprocessing sibling.
// Adapted from llama.cpp ggml/src/ggml-cuda/gated_delta_net.cu and
// gated_delta_net.cuh at cc83d7b4824f73cfdda4dfbb47ee39804f71b328
// (MIT, The ggml authors), plus qwen35.cpp pre-loop ggml_l2_norm on
// q_conv/k_conv. Not a vendor of those files. ds4 has no GDN analog.
// This increment hoists scaled Q/K and decay into overlay scratch; decay
// hoisting is a Quartz proposal (llama evaluates expf inside its loop).
// Causal conv / gated-output collapse into that token loop is Quartz-owned;
// llama.cpp keeps conv as a separate op (chunked-kernel TODO).
// Explicit expf/sqrtf are not compiler FMA contraction; Makefile NVCCFLAGS
// --fmad=false stays unchanged. Do not use rsqrtf.

#include "gdn_step.h"
#include "pdl_launch.cuh"

#include <cstdint>
#include <cstring>

#include <cuda_bf16.h>

namespace qw38::cuda {

constexpr int kGdnQualityWarps = 4;
constexpr int kGdnQualityThreads = 128;
constexpr int kGdnQualityWidth = 128;
constexpr int kGdnFuseHeadThreads = 1024;
constexpr int kGdnFuseConvWidth = 4;
constexpr float kGdnQualityL2Epsilon = 1.0e-6F;

// OPT-029 A/B winner. Legal values: off, fuse_conv, fuse_gate, fuse_both.
constexpr char kSelectedGdnFusePath[] = "off";

// OPT-040 A/B winner. Legal values: repeated, shared. Default until A/B.
constexpr char kSelectedGdnInversePath[] = "shared";

// OPT-052 A/B winner. Legal values: off, preproc, preproc_fma, approx_exp,
// transpose. Default until A/B: off (production stays OPT-040 shared inverse).
constexpr char kSelectedGdnPreprocPath[] = "transpose";

inline const char* gdn_fuse_path_or_selected(const char* path) noexcept {
  return path == nullptr || path[0] == '\0' ? kSelectedGdnFusePath : path;
}

inline const char* gdn_inverse_path_or_selected(const char* path) noexcept {
  return path == nullptr || path[0] == '\0' ? kSelectedGdnInversePath : path;
}

inline bool gdn_path_eq(const char* path, const char* want) noexcept {
  return path != nullptr && want != nullptr && std::strcmp(path, want) == 0;
}

const char* selected_gdn_fuse_path() noexcept { return kSelectedGdnFusePath; }

bool gdn_fuses_conv() noexcept {
  return gdn_path_eq(kSelectedGdnFusePath, "fuse_conv") ||
         gdn_path_eq(kSelectedGdnFusePath, "fuse_both");
}

bool gdn_fuses_gated_output() noexcept {
  return gdn_path_eq(kSelectedGdnFusePath, "fuse_gate") ||
         gdn_path_eq(kSelectedGdnFusePath, "fuse_both");
}

const char* selected_gdn_inverse_path() noexcept {
  return kSelectedGdnInversePath;
}

bool gdn_uses_shared_inverse() noexcept {
  return gdn_path_eq(kSelectedGdnInversePath, "shared");
}

std::size_t gdn_shared_inverse_floats(std::size_t token_count,
                                     std::uint32_t key_heads) noexcept {
  if (token_count == 0 || key_heads == 0) return 0;
  return 2 * token_count * static_cast<std::size_t>(key_heads);
}

inline bool gdn_production_inverse_shape(const GdnConfig& config) noexcept {
  return config.key_heads == 16 && config.value_heads == 48 &&
         config.key_width == 128 && config.value_width == 128 &&
         config.convolution_width == 4;
}

inline bool gdn_can_launch_shared_inverse(
    const GdnConfig& config, std::size_t token_count, const char* fuse_path,
    const char* inverse_path, const float* inverse_scratch,
    std::size_t inverse_scratch_floats) noexcept {
  fuse_path = gdn_fuse_path_or_selected(fuse_path);
  inverse_path = gdn_inverse_path_or_selected(inverse_path);
  if (!gdn_path_eq(inverse_path, "shared") || !gdn_path_eq(fuse_path, "off") ||
      token_count < 2 || !gdn_production_inverse_shape(config) ||
      inverse_scratch == nullptr) {
    return false;
  }
  return inverse_scratch_floats >=
         gdn_shared_inverse_floats(token_count, config.key_heads);
}

inline const char* gdn_preproc_path_or_selected(const char* path) noexcept {
  return path == nullptr || path[0] == '\0' ? kSelectedGdnPreprocPath : path;
}

inline bool gdn_preproc_is_off(const char* path) noexcept {
  return path == nullptr || path[0] == '\0' || gdn_path_eq(path, "off");
}

inline bool gdn_preproc_uses_fma(const char* path) noexcept {
  return gdn_path_eq(path, "preproc_fma");
}

inline bool gdn_preproc_uses_fast_exp(const char* path) noexcept {
  return gdn_path_eq(path, "approx_exp");
}

inline bool gdn_preproc_uses_transpose(const char* path) noexcept {
  return gdn_path_eq(path, "transpose");
}

inline bool gdn_preproc_is_legal(const char* path) noexcept {
  return gdn_preproc_is_off(path) || gdn_path_eq(path, "preproc") ||
         gdn_path_eq(path, "preproc_fma") || gdn_path_eq(path, "approx_exp") ||
         gdn_path_eq(path, "transpose");
}

const char* selected_gdn_preproc_path() noexcept {
  return kSelectedGdnPreprocPath;
}

bool gdn_uses_preproc() noexcept {
  return !gdn_preproc_is_off(kSelectedGdnPreprocPath);
}

inline std::size_t gdn_preproc_qk_floats(std::size_t token_count,
                                         std::uint32_t key_heads,
                                         std::uint32_t key_width) noexcept {
  if (token_count == 0 || key_heads == 0 || key_width == 0) return 0;
  return 2 * token_count * static_cast<std::size_t>(key_heads) * key_width;
}

inline std::size_t gdn_preproc_decay_floats(std::size_t token_count,
                                            std::uint32_t value_heads) noexcept {
  if (token_count == 0 || value_heads == 0) return 0;
  return token_count * static_cast<std::size_t>(value_heads);
}

std::size_t gdn_preproc_floats(const GdnConfig& config,
                               std::size_t token_count) noexcept {
  return gdn_preproc_qk_floats(token_count, config.key_heads, config.key_width) +
         gdn_preproc_decay_floats(token_count, config.value_heads);
}

std::size_t gdn_preproc_transpose_floats(const GdnConfig& config,
                                         std::size_t token_count) noexcept {
  return gdn_preproc_floats(config, token_count) +
         gdn_recurrent_values(config);
}

inline __host__ __device__ float* gdn_preproc_q_ptr(float* scratch,
                                                    std::size_t) noexcept {
  return scratch;
}

inline __host__ __device__ const float* gdn_preproc_q_ptr(const float* scratch,
                                                          std::size_t) noexcept {
  return scratch;
}

inline __host__ __device__ float* gdn_preproc_k_ptr(float* scratch,
                                                    std::size_t token_count,
                                                    std::uint32_t key_heads,
                                                    std::uint32_t key_width) noexcept {
  return scratch + token_count * static_cast<std::size_t>(key_heads) * key_width;
}

inline __host__ __device__ const float* gdn_preproc_k_ptr(
    const float* scratch, std::size_t token_count, std::uint32_t key_heads,
    std::uint32_t key_width) noexcept {
  return scratch + token_count * static_cast<std::size_t>(key_heads) * key_width;
}

inline float* gdn_preproc_decay_ptr(float* scratch, std::size_t token_count,
                                    std::uint32_t key_heads,
                                    std::uint32_t key_width) noexcept {
  return scratch +
         gdn_preproc_qk_floats(token_count, key_heads, key_width);
}

inline const float* gdn_preproc_decay_ptr(const float* scratch,
                                          std::size_t token_count,
                                          std::uint32_t key_heads,
                                          std::uint32_t key_width) noexcept {
  return scratch +
         gdn_preproc_qk_floats(token_count, key_heads, key_width);
}

inline float* gdn_preproc_state_t_ptr(float* scratch, const GdnConfig& config,
                                      std::size_t token_count) noexcept {
  return scratch + gdn_preproc_floats(config, token_count);
}

inline bool gdn_can_launch_preproc(const GdnConfig& config,
                                   std::size_t token_count,
                                   const char* fuse_path,
                                   const char* preproc_path,
                                   const float* scratch,
                                   std::size_t scratch_floats) noexcept {
  fuse_path = gdn_fuse_path_or_selected(fuse_path);
  preproc_path = gdn_preproc_path_or_selected(preproc_path);
  if (gdn_preproc_is_off(preproc_path) || !gdn_preproc_is_legal(preproc_path) ||
      !gdn_path_eq(fuse_path, "off") || token_count < 2 ||
      !gdn_production_inverse_shape(config) || scratch == nullptr) {
    return false;
  }
  const std::size_t need = gdn_preproc_uses_transpose(preproc_path)
                               ? gdn_preproc_transpose_floats(config, token_count)
                               : gdn_preproc_floats(config, token_count);
  return scratch_floats >= need;
}

// Warp sum with __fadd_rn. Butterfly association is admitted at OPT-013
// 5e-8 / 5e-9; ordered lane fold is the documented first repair if that misses.
inline __device__ float gdn_warp_sum(float value) {
  for (int mask = 16; mask > 0; mask >>= 1) {
    value = __fadd_rn(value, __shfl_xor_sync(0xffffffffU, value, mask));
  }
  return value;
}

// Quality Q/K inverse: 4-row unroll, interleaved squares, warp-sum, 1/sqrtf.
// Shared by the hoisted inverse kernel and the repeated warp-column loop.
inline __device__ void gdn_quality_load_qk_and_inverses(
    const float* query, const float* key, std::uint32_t key_width, int lane,
    float q_reg[4], float k_reg[4], float* query_inverse, float* key_inverse) {
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
  *query_inverse = 1.0F / sqrtf(query_squares + kGdnQualityL2Epsilon) /
                   sqrtf(static_cast<float>(key_width));
  *key_inverse = 1.0F / sqrtf(key_squares + kGdnQualityL2Epsilon);
}

inline __device__ void gdn_shift_hist(float hist[kGdnFuseConvWidth],
                                     float incoming) {
  hist[0] = hist[1];
  hist[1] = hist[2];
  hist[2] = hist[3];
  hist[3] = incoming;
}

inline __device__ float gdn_depthwise_silu(
    const float hist[kGdnFuseConvWidth], const float* weights) {
  float convolution = 0.0F;
#pragma unroll
  for (int tap = 0; tap < kGdnFuseConvWidth; ++tap) {
    convolution =
        __fadd_rn(convolution, __fmul_rn(hist[tap], weights[tap]));
  }
  return convolution / (1.0F + expf(-convolution));
}

inline __device__ void gdn_load_hist(float hist[kGdnFuseConvWidth],
                                    const float* source, std::size_t channel,
                                    std::uint32_t width) {
#pragma unroll
  for (int tap = 0; tap < kGdnFuseConvWidth; ++tap) {
    hist[tap] = source[channel * width + static_cast<std::size_t>(tap)];
  }
}

inline __device__ void gdn_store_hist(float* candidate,
                                     const float hist[kGdnFuseConvWidth],
                                     std::size_t channel, std::uint32_t width) {
#pragma unroll
  for (int tap = 0; tap < kGdnFuseConvWidth; ++tap) {
    candidate[channel * width + static_cast<std::size_t>(tap)] = hist[tap];
  }
}

// Grid (value_heads, 1, value_width/4); block (32, 4). Each warp owns one
// value column. s_shard[r] = S[r*32 + lane][col] as in llama.cpp S_v=128.
__global__ void __launch_bounds__(kGdnQualityThreads, 2)
prepare_recurrence_fused_warp_column(
    GdnConfig config, const float* convolution_output, const float* log_decay,
    const float* beta, const float* source, float* candidate, float* output,
    std::size_t token_count, bool value_is_tiled) {
  quartz_pdl_sync();
  const std::uint32_t value_head = blockIdx.x;
  const int lane = static_cast<int>(threadIdx.x);
  const std::uint32_t col = blockIdx.z * 4U + threadIdx.y;
  if (value_head >= config.value_heads || col >= config.value_width) {
    quartz_pdl_lc();
    return;
  }

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
    float query_inverse = 0.0F;
    float key_inverse = 0.0F;
    gdn_quality_load_qk_and_inverses(query, key, config.key_width, lane, q_reg,
                                    k_reg, &query_inverse, &key_inverse);

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
  quartz_pdl_lc();
}

// Grid (key_heads, token_count); block (32). One warp owns (token, key_head).
__global__ void prepare_gdn_shared_inverses(GdnConfig config,
                                           const float* convolution_output,
                                           float* inverses,
                                           std::size_t token_count) {
  quartz_pdl_sync();
  const std::uint32_t key_head = blockIdx.x;
  const std::size_t token = static_cast<std::size_t>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x);
  if (key_head >= config.key_heads || token >= token_count) {
    quartz_pdl_lc();
    return;
  }
  const std::size_t query_count =
      static_cast<std::size_t>(config.key_heads) * config.key_width;
  const std::size_t channels =
      2 * query_count +
      static_cast<std::size_t>(config.value_heads) * config.value_width;
  const float* token_convolution = convolution_output + token * channels;
  const float* query = token_convolution + key_head * config.key_width;
  const float* key =
      token_convolution + query_count + key_head * config.key_width;
  float q_reg[4];
  float k_reg[4];
  float query_inverse = 0.0F;
  float key_inverse = 0.0F;
  gdn_quality_load_qk_and_inverses(query, key, config.key_width, lane, q_reg,
                                  k_reg, &query_inverse, &key_inverse);
  if (lane == 0) {
    const std::size_t base =
        token * static_cast<std::size_t>(config.key_heads) * 2U +
        static_cast<std::size_t>(key_head) * 2U;
    inverses[base] = query_inverse;
    inverses[base + 1] = key_inverse;
  }
  quartz_pdl_lc();
}

// Same grid/block/S-update as prepare_recurrence_fused_warp_column; loads
// hoisted per-(token, key_head) inverses instead of recomputing squares.
__global__ void __launch_bounds__(kGdnQualityThreads, 2)
prepare_recurrence_fused_warp_column_shared(
    GdnConfig config, const float* convolution_output, const float* log_decay,
    const float* beta, const float* source, float* candidate, float* output,
    const float* inverses, std::size_t token_count, bool value_is_tiled) {
  quartz_pdl_sync();
  const std::uint32_t value_head = blockIdx.x;
  const int lane = static_cast<int>(threadIdx.x);
  const std::uint32_t col = blockIdx.z * 4U + threadIdx.y;
  if (value_head >= config.value_heads || col >= config.value_width) {
    quartz_pdl_lc();
    return;
  }

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
    const float* inv =
        inverses + token * static_cast<std::size_t>(config.key_heads) * 2U +
        static_cast<std::size_t>(key_head) * 2U;
    const float query_inverse = inv[0];
    const float key_inverse = inv[1];

    float q_reg[4];
    float k_reg[4];
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      const int index = row * 32 + lane;
      q_reg[row] = query[index];
      k_reg[row] = key[index];
    }

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
  quartz_pdl_lc();
}

// Grid (key_heads, token_count); block (32). One warp owns (token, key_head)
// and writes L2-scaled Q/K. Convolution V is left untouched.
__global__ void prepare_gdn_scaled_qk(GdnConfig config,
                                      const float* convolution_output,
                                      float* scratch, std::size_t token_count) {
  quartz_pdl_sync();
  const std::uint32_t key_head = blockIdx.x;
  const std::size_t token = static_cast<std::size_t>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x);
  if (key_head >= config.key_heads || token >= token_count) {
    quartz_pdl_lc();
    return;
  }
  const std::size_t query_count =
      static_cast<std::size_t>(config.key_heads) * config.key_width;
  const std::size_t channels =
      2 * query_count +
      static_cast<std::size_t>(config.value_heads) * config.value_width;
  const float* token_convolution = convolution_output + token * channels;
  const float* query = token_convolution + key_head * config.key_width;
  const float* key =
      token_convolution + query_count + key_head * config.key_width;
  float q_reg[4];
  float k_reg[4];
  float query_inverse = 0.0F;
  float key_inverse = 0.0F;
  gdn_quality_load_qk_and_inverses(query, key, config.key_width, lane, q_reg,
                                  k_reg, &query_inverse, &key_inverse);
  float* q_scaled = gdn_preproc_q_ptr(scratch, token_count) +
                    token * query_count +
                    static_cast<std::size_t>(key_head) * config.key_width;
  float* k_scaled =
      gdn_preproc_k_ptr(scratch, token_count, config.key_heads,
                        config.key_width) +
      token * query_count +
      static_cast<std::size_t>(key_head) * config.key_width;
#pragma unroll
  for (int row = 0; row < 4; ++row) {
    const int index = row * 32 + lane;
    q_scaled[index] = __fmul_rn(q_reg[row], query_inverse);
    k_scaled[index] = __fmul_rn(k_reg[row], key_inverse);
  }
  quartz_pdl_lc();
}

template <bool kFastExp>
__global__ void prepare_gdn_decay(GdnConfig config, const float* log_decay,
                                  float* decay_out, std::size_t token_count) {
  quartz_pdl_sync();
  const std::size_t token = static_cast<std::size_t>(blockIdx.x);
  const std::uint32_t head = threadIdx.x;
  if (token >= token_count || head >= config.value_heads) {
    quartz_pdl_lc();
    return;
  }
  const std::size_t index = token * config.value_heads + head;
  const float log_g = log_decay[index];
  if constexpr (kFastExp) {
    decay_out[index] = __expf(log_g);
  } else {
    decay_out[index] = expf(log_g);
  }
  quartz_pdl_lc();
}

__global__ void gdn_transpose_state_to_column_major(
    const float* row_major, float* col_major, std::uint32_t value_heads,
    std::uint32_t key_width, std::uint32_t value_width) {
  quartz_pdl_sync();
  const std::uint32_t head = blockIdx.x;
  const std::uint32_t row = blockIdx.y;
  const std::uint32_t col = threadIdx.x;
  if (head >= value_heads || row >= key_width || col >= value_width) {
    quartz_pdl_lc();
    return;
  }
  const std::size_t base =
      static_cast<std::size_t>(head) * key_width * value_width;
  col_major[base + static_cast<std::size_t>(col) * key_width + row] =
      row_major[base + static_cast<std::size_t>(row) * value_width + col];
  quartz_pdl_lc();
}

__global__ void gdn_transpose_state_to_row_major(
    const float* col_major, float* row_major, std::uint32_t value_heads,
    std::uint32_t key_width, std::uint32_t value_width) {
  quartz_pdl_sync();
  const std::uint32_t head = blockIdx.x;
  const std::uint32_t row = blockIdx.y;
  const std::uint32_t col = threadIdx.x;
  if (head >= value_heads || row >= key_width || col >= value_width) {
    quartz_pdl_lc();
    return;
  }
  const std::size_t base =
      static_cast<std::size_t>(head) * key_width * value_width;
  row_major[base + static_cast<std::size_t>(row) * value_width + col] =
      col_major[base + static_cast<std::size_t>(col) * key_width + row];
  quartz_pdl_lc();
}

template <bool kUseFma, bool kTransposed>
__global__ void __launch_bounds__(kGdnQualityThreads, 2)
prepare_recurrence_fused_warp_column_preproc(
    GdnConfig config, const float* convolution_output, const float* decay,
    const float* beta, const float* source, float* candidate, float* output,
    const float* scratch, std::size_t token_count, bool value_is_tiled) {
  quartz_pdl_sync();
  const std::uint32_t value_head = blockIdx.x;
  const int lane = static_cast<int>(threadIdx.x);
  const std::uint32_t col = blockIdx.z * 4U + threadIdx.y;
  if (value_head >= config.value_heads || col >= config.value_width) {
    quartz_pdl_lc();
    return;
  }

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
  const float* q_base = gdn_preproc_q_ptr(scratch, token_count);
  const float* k_base = gdn_preproc_k_ptr(scratch, token_count, config.key_heads,
                                          config.key_width);

  float s_shard[4];
#pragma unroll
  for (int row = 0; row < 4; ++row) {
    const std::uint32_t key_lane =
        static_cast<std::uint32_t>(row * 32 + lane);
    if constexpr (kTransposed) {
      s_shard[row] = source[head_base + static_cast<std::size_t>(col) *
                                            config.key_width +
                            key_lane];
    } else {
      s_shard[row] = source[head_base + static_cast<std::size_t>(key_lane) *
                                            config.value_width +
                            col];
    }
  }

  for (std::size_t token = 0; token < token_count; ++token) {
    const float* token_convolution = convolution_output + token * channels;
    const float* value = token_convolution + 2 * query_count +
                         source_head * config.value_width;
    const float* query =
        q_base + token * query_count +
        static_cast<std::size_t>(key_head) * config.key_width;
    const float* key = k_base + token * query_count +
                       static_cast<std::size_t>(key_head) * config.key_width;
    float q_reg[4];
    float k_reg[4];
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      const int index = row * 32 + lane;
      q_reg[row] = query[index];
      k_reg[row] = key[index];
    }
    const float decay_val = decay[token * config.value_heads + value_head];
    const float beta_val = beta[token * config.value_heads + value_head];

    float kv_shard = 0.0F;
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      if constexpr (kUseFma) {
        kv_shard = __fmaf_rn(s_shard[row], k_reg[row], kv_shard);
      } else {
        kv_shard = __fadd_rn(kv_shard, __fmul_rn(s_shard[row], k_reg[row]));
      }
    }
    const float kv_acc = gdn_warp_sum(kv_shard);
    float delta = 0.0F;
    if constexpr (kUseFma) {
      delta = __fmul_rn(__fmaf_rn(-decay_val, kv_acc, value[col]), beta_val);
    } else {
      delta = __fmul_rn(value[col] - __fmul_rn(decay_val, kv_acc), beta_val);
    }

    float attn_shard = 0.0F;
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      if constexpr (kUseFma) {
        s_shard[row] = __fmaf_rn(k_reg[row], delta,
                                 __fmul_rn(decay_val, s_shard[row]));
        attn_shard = __fmaf_rn(q_reg[row], s_shard[row], attn_shard);
      } else {
        s_shard[row] = __fadd_rn(__fmul_rn(decay_val, s_shard[row]),
                                 __fmul_rn(k_reg[row], delta));
        attn_shard =
            __fadd_rn(attn_shard, __fmul_rn(q_reg[row], s_shard[row]));
      }
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
    if constexpr (kTransposed) {
      candidate[head_base + static_cast<std::size_t>(col) * config.key_width +
                key_lane] = s_shard[row];
    } else {
      candidate[head_base + static_cast<std::size_t>(key_lane) *
                                config.value_width +
                col] = s_shard[row];
    }
  }
  quartz_pdl_lc();
}

// fuse_conv: inline width-4 causal conv + SiLU into the warp-column loop.
__global__ void __launch_bounds__(kGdnQualityThreads, 2)
prepare_recurrence_fused_conv(
    GdnConfig config, const float* convolution_input,
    const float* convolution_weights, const float* committed_convolution,
    float* candidate_convolution, float* convolution_output,
    const float* log_decay, const float* beta, const float* source,
    float* candidate, float* output, std::size_t token_count,
    bool value_is_tiled) {
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
  const std::uint32_t width = config.convolution_width;

  float s_shard[4];
  float q_hist[4][kGdnFuseConvWidth];
  float k_hist[4][kGdnFuseConvWidth];
  float v_hist[kGdnFuseConvWidth];
#pragma unroll
  for (int row = 0; row < 4; ++row) {
    const std::uint32_t key_lane =
        static_cast<std::uint32_t>(row * 32 + lane);
    s_shard[row] = source[head_base + static_cast<std::size_t>(key_lane) *
                                          config.value_width +
                          col];
    const std::size_t q_channel =
        static_cast<std::size_t>(key_head) * config.key_width + key_lane;
    const std::size_t k_channel = query_count + q_channel;
    gdn_load_hist(q_hist[row], committed_convolution, q_channel, width);
    gdn_load_hist(k_hist[row], committed_convolution, k_channel, width);
  }
  const std::size_t v_channel = 2 * query_count +
                                static_cast<std::size_t>(source_head) *
                                    config.value_width +
                                col;
  gdn_load_hist(v_hist, committed_convolution, v_channel, width);

  for (std::size_t token = 0; token < token_count; ++token) {
    const float* token_input = convolution_input + token * channels;
    float q_reg[4];
    float k_reg[4];
    float query_squares = 0.0F;
    float key_squares = 0.0F;
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      const int index = row * 32 + lane;
      const std::size_t q_channel =
          static_cast<std::size_t>(key_head) * config.key_width +
          static_cast<std::uint32_t>(index);
      gdn_shift_hist(q_hist[row], token_input[q_channel]);
      gdn_shift_hist(k_hist[row], token_input[query_count + q_channel]);
      q_reg[row] = gdn_depthwise_silu(
          q_hist[row], convolution_weights + q_channel * width);
      k_reg[row] = gdn_depthwise_silu(
          k_hist[row],
          convolution_weights + (query_count + q_channel) * width);
      query_squares =
          __fadd_rn(query_squares, __fmul_rn(q_reg[row], q_reg[row]));
      key_squares = __fadd_rn(key_squares, __fmul_rn(k_reg[row], k_reg[row]));
    }
    gdn_shift_hist(v_hist, token_input[v_channel]);
    const float v_reg =
        gdn_depthwise_silu(v_hist, convolution_weights + v_channel * width);
    if (convolution_output != nullptr) {
#pragma unroll
      for (int row = 0; row < 4; ++row) {
        const int index = row * 32 + lane;
        const std::size_t q_channel =
            static_cast<std::size_t>(key_head) * config.key_width +
            static_cast<std::uint32_t>(index);
        if (replica == 0 && blockIdx.z == 0) {
          convolution_output[token * channels + q_channel] = q_reg[row];
          convolution_output[token * channels + query_count + q_channel] =
              k_reg[row];
        }
      }
      if (lane == 0) {
        convolution_output[token * channels + v_channel] = v_reg;
      }
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
        __fmul_rn(v_reg - __fmul_rn(decay, kv_acc), beta_val);

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
    if (replica == 0 && blockIdx.z == 0) {
      const std::size_t q_channel =
          static_cast<std::size_t>(key_head) * config.key_width + key_lane;
      gdn_store_hist(candidate_convolution, q_hist[row], q_channel, width);
      gdn_store_hist(candidate_convolution, k_hist[row],
                     query_count + q_channel, width);
    }
  }
  gdn_store_hist(candidate_convolution, v_hist, v_channel, width);
}

// fuse_gate / fuse_both: one block per value head, 32 warps, 4 column groups.
// kFuseConv inlines width-4 causal conv; otherwise reads convolution_output.
template <bool kFuseConv>
__global__ void __launch_bounds__(kGdnFuseHeadThreads, 1)
prepare_recurrence_fused_head(
    GdnConfig config, const float* convolution_input,
    const float* convolution_weights, const float* committed_convolution,
    float* candidate_convolution, const float* convolution_output,
    const float* log_decay, const float* beta, const float* source,
    float* candidate, float* output, const float* gate_tiled, const float* norm,
    __nv_bfloat16* output_bf16, std::size_t token_count, bool value_is_tiled) {
  const std::uint32_t value_head = blockIdx.x;
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  if (value_head >= config.value_heads) return;

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
  const std::uint32_t width = config.convolution_width;
  const std::size_t heads = config.value_heads;

  __shared__ float attn[kGdnQualityWidth];
  __shared__ float inverse;

  float s_shard[4][4];
  float q_hist[4][kGdnFuseConvWidth];
  float k_hist[4][kGdnFuseConvWidth];
  float v_hist[4][kGdnFuseConvWidth];
#pragma unroll
  for (int group = 0; group < 4; ++group) {
    const std::uint32_t col =
        static_cast<std::uint32_t>(group * 32 + warp);
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      const std::uint32_t key_lane =
          static_cast<std::uint32_t>(row * 32 + lane);
      s_shard[group][row] =
          source[head_base + static_cast<std::size_t>(key_lane) *
                                 config.value_width +
                 col];
    }
  }
  if (kFuseConv) {
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      const std::uint32_t key_lane =
          static_cast<std::uint32_t>(row * 32 + lane);
      const std::size_t q_channel =
          static_cast<std::size_t>(key_head) * config.key_width + key_lane;
      gdn_load_hist(q_hist[row], committed_convolution, q_channel, width);
      gdn_load_hist(k_hist[row], committed_convolution, query_count + q_channel,
                    width);
    }
#pragma unroll
    for (int group = 0; group < 4; ++group) {
      const std::uint32_t col =
          static_cast<std::uint32_t>(group * 32 + warp);
      const std::size_t v_channel =
          2 * query_count +
          static_cast<std::size_t>(source_head) * config.value_width + col;
      gdn_load_hist(v_hist[group], committed_convolution, v_channel, width);
    }
  }

  for (std::size_t token = 0; token < token_count; ++token) {
    float q_reg[4];
    float k_reg[4];
    float v_reg[4];
    float query_squares = 0.0F;
    float key_squares = 0.0F;
    if (kFuseConv) {
      const float* token_input = convolution_input + token * channels;
#pragma unroll
      for (int row = 0; row < 4; ++row) {
        const int index = row * 32 + lane;
        const std::size_t q_channel =
            static_cast<std::size_t>(key_head) * config.key_width +
            static_cast<std::uint32_t>(index);
        gdn_shift_hist(q_hist[row], token_input[q_channel]);
        gdn_shift_hist(k_hist[row], token_input[query_count + q_channel]);
        q_reg[row] = gdn_depthwise_silu(
            q_hist[row], convolution_weights + q_channel * width);
        k_reg[row] = gdn_depthwise_silu(
            k_hist[row],
            convolution_weights + (query_count + q_channel) * width);
        query_squares =
            __fadd_rn(query_squares, __fmul_rn(q_reg[row], q_reg[row]));
        key_squares =
            __fadd_rn(key_squares, __fmul_rn(k_reg[row], k_reg[row]));
      }
#pragma unroll
      for (int group = 0; group < 4; ++group) {
        const std::uint32_t col =
            static_cast<std::uint32_t>(group * 32 + warp);
        const std::size_t v_channel =
            2 * query_count +
            static_cast<std::size_t>(source_head) * config.value_width + col;
        gdn_shift_hist(v_hist[group], token_input[v_channel]);
        v_reg[group] = gdn_depthwise_silu(
            v_hist[group], convolution_weights + v_channel * width);
      }
    } else {
      const float* token_convolution = convolution_output + token * channels;
      const float* query = token_convolution + key_head * config.key_width;
      const float* key =
          token_convolution + query_count + key_head * config.key_width;
      const float* value = token_convolution + 2 * query_count +
                           source_head * config.value_width;
#pragma unroll
      for (int row = 0; row < 4; ++row) {
        const int index = row * 32 + lane;
        q_reg[row] = query[index];
        k_reg[row] = key[index];
        query_squares =
            __fadd_rn(query_squares, __fmul_rn(q_reg[row], q_reg[row]));
        key_squares =
            __fadd_rn(key_squares, __fmul_rn(k_reg[row], k_reg[row]));
      }
#pragma unroll
      for (int group = 0; group < 4; ++group) {
        const std::uint32_t col =
            static_cast<std::uint32_t>(group * 32 + warp);
        v_reg[group] = value[col];
      }
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

#pragma unroll
    for (int group = 0; group < 4; ++group) {
      const std::uint32_t col =
          static_cast<std::uint32_t>(group * 32 + warp);
      float kv_shard = 0.0F;
#pragma unroll
      for (int row = 0; row < 4; ++row) {
        const float k_scale = __fmul_rn(k_reg[row], key_inverse);
        kv_shard =
            __fadd_rn(kv_shard, __fmul_rn(s_shard[group][row], k_scale));
      }
      const float kv_acc = gdn_warp_sum(kv_shard);
      const float delta =
          __fmul_rn(v_reg[group] - __fmul_rn(decay, kv_acc), beta_val);
      float attn_shard = 0.0F;
#pragma unroll
      for (int row = 0; row < 4; ++row) {
        const float k_scale = __fmul_rn(k_reg[row], key_inverse);
        const float q_scale = __fmul_rn(q_reg[row], query_inverse);
        s_shard[group][row] = __fadd_rn(__fmul_rn(decay, s_shard[group][row]),
                                        __fmul_rn(k_scale, delta));
        attn_shard = __fadd_rn(attn_shard,
                               __fmul_rn(q_scale, s_shard[group][row]));
      }
      const float attn_acc = gdn_warp_sum(attn_shard);
      if (lane == 0) attn[col] = attn_acc;
    }
    __syncthreads();
    if (warp == 0 && lane == 0) {
      float sum = 0.0F;
      for (int index = 0; index < kGdnQualityWidth; ++index) {
        sum = __fadd_rn(sum, __fmul_rn(attn[index], attn[index]));
      }
      inverse = 1.0F / sqrtf(sum / static_cast<float>(kGdnQualityWidth) +
                             1.0e-6F);
    }
    __syncthreads();
#pragma unroll
    for (int group = 0; group < 4; ++group) {
      const std::uint32_t col =
          static_cast<std::uint32_t>(group * 32 + warp);
      if (lane == 0) {
        output[(token * heads + value_head) * config.value_width + col] =
            attn[col];
        const std::size_t tiled_base =
            (token * heads + tiled_head) * config.value_width;
        const float gate = gate_tiled[tiled_base + col];
        const float silu = gate / (1.0F + expf(-gate));
        const float value = __fmul_rn(__fmul_rn(attn[col], inverse), norm[col]);
        output_bf16[tiled_base + col] =
            __float2bfloat16_rn(__fmul_rn(value, silu));
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int group = 0; group < 4; ++group) {
    const std::uint32_t col =
        static_cast<std::uint32_t>(group * 32 + warp);
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      const std::uint32_t key_lane =
          static_cast<std::uint32_t>(row * 32 + lane);
      candidate[head_base + static_cast<std::size_t>(key_lane) *
                                config.value_width +
                col] = s_shard[group][row];
    }
    if (kFuseConv) {
      const std::size_t v_channel =
          2 * query_count +
          static_cast<std::size_t>(source_head) * config.value_width + col;
      gdn_store_hist(candidate_convolution, v_hist[group], v_channel, width);
    }
  }
  if (kFuseConv && replica == 0 && warp == 0) {
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      const std::uint32_t key_lane =
          static_cast<std::uint32_t>(row * 32 + lane);
      const std::size_t q_channel =
          static_cast<std::size_t>(key_head) * config.key_width + key_lane;
      gdn_store_hist(candidate_convolution, q_hist[row], q_channel, width);
      gdn_store_hist(candidate_convolution, k_hist[row],
                     query_count + q_channel, width);
    }
  }
}

__global__ void gdn_gated_output_rows_split(
    const float* recurrent, const float* gate_tiled, const float* norm,
    std::size_t key_heads, std::size_t replicas, std::size_t head_width,
    __nv_bfloat16* output_tiled) {
  quartz_pdl_sync();
  const std::size_t row = blockIdx.y;
  const std::size_t grouped_head = blockIdx.x;
  const std::size_t lane = threadIdx.x;
  const std::size_t heads = key_heads * replicas;
  const std::size_t grouped_base =
      (row * heads + grouped_head) * head_width;
  __shared__ float inverse;
  if (lane == 0) {
    float sum = 0.0F;
    for (std::size_t index = 0; index < head_width; ++index) {
      const float value = recurrent[grouped_base + index];
      sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    inverse = 1.0F /
              sqrtf(sum / static_cast<float>(head_width) + 1.0e-6F);
  }
  __syncthreads();
  if (lane < head_width) {
    const std::size_t key = grouped_head / replicas;
    const std::size_t replica = grouped_head % replicas;
    const std::size_t tiled_head = replica * key_heads + key;
    const std::size_t tiled_base =
        (row * heads + tiled_head) * head_width;
    const float gate = gate_tiled[tiled_base + lane];
    const float silu = gate / (1.0F + expf(-gate));
    const float value = __fmul_rn(
        __fmul_rn(recurrent[grouped_base + lane], inverse), norm[lane]);
    output_tiled[tiled_base + lane] =
        __float2bfloat16_rn(__fmul_rn(value, silu));
  }
  quartz_pdl_lc();
}

inline cudaError_t launch_gdn_quality_recurrence(
    const GdnConfig& config, const float* convolution_output,
    const float* log_decay, const float* beta, const float* source,
    float* candidate, float* output, std::size_t token_count,
    bool value_is_tiled, cudaStream_t stream) noexcept {
  const dim3 grid(config.value_heads, 1U, config.value_width / 4U);
  const dim3 block(32U, 4U, 1U);
  return quartz_launch_kernel(
      prepare_recurrence_fused_warp_column, grid, block, 0, stream, config,
      convolution_output, log_decay, beta, source, candidate, output,
      token_count, value_is_tiled);
}

cudaError_t launch_gdn_shared_inverses(
    const GdnConfig& config, const float* convolution_output,
    std::size_t token_count, float* inverses, cudaStream_t stream) noexcept {
  if (convolution_output == nullptr || inverses == nullptr ||
      token_count == 0 || config.key_heads == 0) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid(config.key_heads, static_cast<unsigned int>(token_count));
  return quartz_launch_kernel(prepare_gdn_shared_inverses, grid, dim3(32U), 0,
                              stream, config, convolution_output, inverses,
                              token_count);
}

inline cudaError_t launch_gdn_quality_recurrence_shared(
    const GdnConfig& config, const float* convolution_output,
    const float* log_decay, const float* beta, const float* source,
    float* candidate, float* output, const float* inverses,
    std::size_t token_count, bool value_is_tiled, cudaStream_t stream) noexcept {
  if (inverses == nullptr) return cudaErrorInvalidValue;
  const dim3 grid(config.value_heads, 1U, config.value_width / 4U);
  const dim3 block(32U, 4U, 1U);
  return quartz_launch_kernel(
      prepare_recurrence_fused_warp_column_shared, grid, block, 0, stream,
      config, convolution_output, log_decay, beta, source, candidate, output,
      inverses, token_count, value_is_tiled);
}

cudaError_t launch_gdn_gated_output_rows(
    const float* recurrent, const float* gate_tiled, const float* norm,
    std::size_t key_heads, std::size_t replicas, std::size_t head_width,
    std::size_t token_count, __nv_bfloat16* output_tiled,
    cudaStream_t stream) noexcept {
  if (recurrent == nullptr || gate_tiled == nullptr || norm == nullptr ||
      output_tiled == nullptr || key_heads == 0 || replicas == 0 ||
      head_width == 0 || token_count == 0) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid(static_cast<unsigned int>(key_heads * replicas),
                  static_cast<unsigned int>(token_count));
  return quartz_launch_kernel(
      gdn_gated_output_rows_split, grid, dim3(256), 0, stream, recurrent,
      gate_tiled, norm, key_heads, replicas, head_width, output_tiled);
}

int gdn_fuse_occupancy(const char* path) noexcept {
  path = gdn_fuse_path_or_selected(path);
  int occupancy = 0;
  cudaError_t error = cudaSuccess;
  if (gdn_path_eq(path, "fuse_gate")) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, prepare_recurrence_fused_head<false>, kGdnFuseHeadThreads,
        0);
  } else if (gdn_path_eq(path, "fuse_both")) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, prepare_recurrence_fused_head<true>, kGdnFuseHeadThreads,
        0);
  } else if (gdn_path_eq(path, "fuse_conv")) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, prepare_recurrence_fused_conv, kGdnQualityThreads, 0);
  } else {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, prepare_recurrence_fused_warp_column, kGdnQualityThreads,
        0);
  }
  return error == cudaSuccess ? occupancy : 0;
}

int gdn_shared_inverse_occupancy() noexcept {
  int occupancy = 0;
  const cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, prepare_gdn_shared_inverses, 32, 0);
  return error == cudaSuccess ? occupancy : 0;
}

int gdn_shared_recurrence_occupancy() noexcept {
  int occupancy = 0;
  const cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, prepare_recurrence_fused_warp_column_shared,
      kGdnQualityThreads, 0);
  return error == cudaSuccess ? occupancy : 0;
}

inline cudaError_t launch_gdn_scaled_qk(const GdnConfig& config,
                                        const float* convolution_output,
                                        std::size_t token_count, float* scratch,
                                        cudaStream_t stream) noexcept {
  if (convolution_output == nullptr || scratch == nullptr || token_count == 0 ||
      config.key_heads == 0) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid(config.key_heads, static_cast<unsigned int>(token_count));
  return quartz_launch_kernel(prepare_gdn_scaled_qk, grid, dim3(32U), 0, stream,
                              config, convolution_output, scratch, token_count);
}

inline cudaError_t launch_gdn_decay(const GdnConfig& config,
                                    const float* log_decay,
                                    std::size_t token_count, float* decay_out,
                                    bool fast_exp, cudaStream_t stream) noexcept {
  if (log_decay == nullptr || decay_out == nullptr || token_count == 0 ||
      config.value_heads == 0) {
    return cudaErrorInvalidValue;
  }
  unsigned int threads = 32U;
  while (threads < config.value_heads) threads += 32U;
  const dim3 grid(static_cast<unsigned int>(token_count));
  if (fast_exp) {
    return quartz_launch_kernel(prepare_gdn_decay<true>, grid, dim3(threads), 0,
                                stream, config, log_decay, decay_out,
                                token_count);
  }
  return quartz_launch_kernel(prepare_gdn_decay<false>, grid, dim3(threads), 0,
                              stream, config, log_decay, decay_out, token_count);
}

inline cudaError_t launch_gdn_transpose_state_in(
    const GdnConfig& config, const float* row_major, float* col_major,
    cudaStream_t stream) noexcept {
  if (row_major == nullptr || col_major == nullptr) return cudaErrorInvalidValue;
  const dim3 grid(config.value_heads, config.key_width);
  return quartz_launch_kernel(gdn_transpose_state_to_column_major, grid,
                              dim3(config.value_width), 0, stream, row_major,
                              col_major, config.value_heads, config.key_width,
                              config.value_width);
}

inline cudaError_t launch_gdn_transpose_state_out(
    const GdnConfig& config, const float* col_major, float* row_major,
    cudaStream_t stream) noexcept {
  if (row_major == nullptr || col_major == nullptr) return cudaErrorInvalidValue;
  const dim3 grid(config.value_heads, config.key_width);
  return quartz_launch_kernel(gdn_transpose_state_to_row_major, grid,
                              dim3(config.value_width), 0, stream, col_major,
                              row_major, config.value_heads, config.key_width,
                              config.value_width);
}

inline cudaError_t launch_gdn_quality_recurrence_preproc(
    const GdnConfig& config, const float* convolution_output,
    const float* decay, const float* beta, const float* source,
    float* candidate, float* output, const float* scratch,
    std::size_t token_count, bool value_is_tiled, bool use_fma, bool transposed,
    cudaStream_t stream) noexcept {
  if (decay == nullptr || scratch == nullptr) return cudaErrorInvalidValue;
  const dim3 grid(config.value_heads, 1U, config.value_width / 4U);
  const dim3 block(32U, 4U, 1U);
  if (use_fma && transposed) {
    return quartz_launch_kernel(
        prepare_recurrence_fused_warp_column_preproc<true, true>, grid, block, 0,
        stream, config, convolution_output, decay, beta, source, candidate,
        output, scratch, token_count, value_is_tiled);
  }
  if (use_fma) {
    return quartz_launch_kernel(
        prepare_recurrence_fused_warp_column_preproc<true, false>, grid, block,
        0, stream, config, convolution_output, decay, beta, source, candidate,
        output, scratch, token_count, value_is_tiled);
  }
  if (transposed) {
    return quartz_launch_kernel(
        prepare_recurrence_fused_warp_column_preproc<false, true>, grid, block,
        0, stream, config, convolution_output, decay, beta, source, candidate,
        output, scratch, token_count, value_is_tiled);
  }
  return quartz_launch_kernel(
      prepare_recurrence_fused_warp_column_preproc<false, false>, grid, block, 0,
      stream, config, convolution_output, decay, beta, source, candidate,
      output, scratch, token_count, value_is_tiled);
}

int gdn_preproc_occupancy() noexcept {
  int occupancy = 0;
  const cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, prepare_gdn_scaled_qk, 32, 0);
  return error == cudaSuccess ? occupancy : 0;
}

int gdn_preproc_decay_occupancy() noexcept {
  int occupancy = 0;
  const cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, prepare_gdn_decay<false>, 64, 0);
  return error == cudaSuccess ? occupancy : 0;
}

int gdn_preproc_recurrence_occupancy() noexcept {
  int occupancy = 0;
  const cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, prepare_recurrence_fused_warp_column_preproc<false, false>,
      kGdnQualityThreads, 0);
  return error == cudaSuccess ? occupancy : 0;
}

int gdn_preproc_fma_occupancy() noexcept {
  int occupancy = 0;
  const cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, prepare_recurrence_fused_warp_column_preproc<true, false>,
      kGdnQualityThreads, 0);
  return error == cudaSuccess ? occupancy : 0;
}

int gdn_preproc_transpose_occupancy() noexcept {
  int occupancy = 0;
  const cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, prepare_recurrence_fused_warp_column_preproc<false, true>,
      kGdnQualityThreads, 0);
  return error == cudaSuccess ? occupancy : 0;
}

}  // namespace qw38::cuda
