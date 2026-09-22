#include "cuda/attention.hpp"

#include <cstddef>
#include <cmath>
#include <limits>

namespace qw38::cuda {
namespace {

__device__ __forceinline__ float bf16_to_fp32(std::uint16_t h) {
  return __uint_as_float(static_cast<std::uint32_t>(h) << 16);
}

__device__ __forceinline__ std::uint16_t fp32_to_bf16_rne(float x) {
  std::uint32_t const bits = __float_as_uint(x);
  std::uint32_t const exp = (bits >> 23) & 0xFFu;
  if (exp == 0xFFu) {
    std::uint16_t h = static_cast<std::uint16_t>(bits >> 16);
    if ((bits & 0x7FFFFFu) != 0) {
      h = static_cast<std::uint16_t>(h | 0x0040u);
    }
    return h;
  }
  std::uint32_t const lsb = (bits >> 16) & 1u;
  std::uint32_t const add = 0x7FFFu + lsb;
  return static_cast<std::uint16_t>((bits + add) >> 16);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, off);
  }
  return v;
}

__device__ float block_sum128(float v) {
  constexpr int kWarps = kAttnPrepThreads / 32;
  v = warp_sum(v);
  __shared__ float sm[kWarps];
  int const warp = threadIdx.x >> 5;
  int const lane = threadIdx.x & 31;
  if (lane == 0) {
    sm[warp] = v;
  }
  __syncthreads();
  float t = 0.0f;
  if (warp == 0) {
    t = (lane < kWarps) ? sm[lane] : 0.0f;
    t = warp_sum(t);
    if (lane == 0) {
      sm[0] = t;
    }
  }
  __syncthreads();
  return sm[0];
}

__device__ void rms_rope_store(std::uint16_t const* x, std::uint16_t const* gamma,
                               float const* inv_freq, float eps, float position,
                               std::uint16_t* out) {
  __shared__ float y[kAttnHeadDim];
  float sumsq = 0.0f;
  for (std::uint32_t i = static_cast<std::uint32_t>(threadIdx.x); i < kAttnHeadDim;
       i += static_cast<std::uint32_t>(kAttnPrepThreads)) {
    float const v = bf16_to_fp32(x[i]);
    y[i] = v;
    sumsq += v * v;
  }
  sumsq = block_sum128(sumsq);
  float const rms =
      sqrtf(sumsq / static_cast<float>(kAttnHeadDim) + eps);
  float const inv_rms = 1.0f / rms;
  for (std::uint32_t i = static_cast<std::uint32_t>(threadIdx.x); i < kAttnHeadDim;
       i += static_cast<std::uint32_t>(kAttnPrepThreads)) {
    float const g = bf16_to_fp32(gamma[i]);
    y[i] = (1.0f + g) * y[i] * inv_rms;
  }
  __syncthreads();
  constexpr std::uint32_t kHalf = kAttnRotaryDim / 2;
  if (static_cast<std::uint32_t>(threadIdx.x) < kHalf) {
    std::uint32_t const j = static_cast<std::uint32_t>(threadIdx.x);
    float const x0 = y[j];
    float const x1 = y[j + kHalf];
    float const phase = position * inv_freq[j];
    float const c = cosf(phase);
    float const s = sinf(phase);
    y[j] = x0 * c - x1 * s;
    y[j + kHalf] = x1 * c + x0 * s;
  }
  __syncthreads();
  for (std::uint32_t i = static_cast<std::uint32_t>(threadIdx.x); i < kAttnHeadDim;
       i += static_cast<std::uint32_t>(kAttnPrepThreads)) {
    out[i] = fp32_to_bf16_rne(y[i]);
  }
}

__device__ void copy_head(std::uint16_t const* src, std::uint16_t* dst) {
  for (std::uint32_t i = static_cast<std::uint32_t>(threadIdx.x); i < kAttnHeadDim;
       i += static_cast<std::uint32_t>(kAttnPrepThreads)) {
    dst[i] = src[i];
  }
}

__global__ void attention_prepare_kernel(
    std::uint16_t const* qg, std::uint16_t const* k_raw,
    std::uint16_t const* v_raw, std::uint16_t const* gamma_q,
    std::uint16_t const* gamma_k, float const* inv_freq, float eps,
    float position, std::uint16_t* q_out, std::uint16_t* g_out,
    std::uint16_t* kv, std::uint32_t attn_layer, std::uint64_t capacity,
    std::uint64_t token) {
  std::size_t const head_stride =
      static_cast<std::size_t>(capacity) * kAttnHeadDim;
  std::size_t const comp_stride =
      static_cast<std::size_t>(kAttnKvHeads) * head_stride;
  std::size_t const layer_stride = 2u * comp_stride;
  std::uint16_t* layer = kv + static_cast<std::size_t>(attn_layer) * layer_stride;
  std::uint16_t* k_base = layer;
  std::uint16_t* v_base = layer + comp_stride;

  if (blockIdx.x < kAttnQueryHeads) {
    std::uint32_t const h = blockIdx.x;
    std::uint16_t const* q_src =
        qg + static_cast<std::size_t>(h) * (2u * kAttnHeadDim);
    std::uint16_t const* g_src = q_src + kAttnHeadDim;
    rms_rope_store(q_src, gamma_q, inv_freq, eps, position,
                   q_out + static_cast<std::size_t>(h) * kAttnHeadDim);
    copy_head(g_src, g_out + static_cast<std::size_t>(h) * kAttnHeadDim);
    return;
  }

  std::uint32_t const kv_h = blockIdx.x - kAttnQueryHeads;
  std::uint16_t const* k_src =
      k_raw + static_cast<std::size_t>(kv_h) * kAttnHeadDim;
  std::uint16_t const* v_src =
      v_raw + static_cast<std::size_t>(kv_h) * kAttnHeadDim;
  std::uint16_t* k_dst =
      k_base + static_cast<std::size_t>(kv_h) * head_stride +
      static_cast<std::size_t>(token) * kAttnHeadDim;
  std::uint16_t* v_dst =
      v_base + static_cast<std::size_t>(kv_h) * head_stride +
      static_cast<std::size_t>(token) * kAttnHeadDim;
  rms_rope_store(k_src, gamma_k, inv_freq, eps, position, k_dst);
  copy_head(v_src, v_dst);
}

bool finite_pos(float x) noexcept {
  return x == x && x <= std::numeric_limits<float>::max() &&
         x >= std::numeric_limits<float>::lowest() && x > 0.0f;
}

std::expected<void, Error> require_stream(Stream const& stream,
                                          std::string_view op) {
  if (stream.empty()) {
    return std::unexpected(
        make_error(ErrorCode::InvalidArgument, op, "empty stream"));
  }
  return {};
}

}  // namespace

std::expected<void, Error> launch_attention_prepare(
    std::uint16_t const* qg, std::uint16_t const* k_raw,
    std::uint16_t const* v_raw, std::uint16_t const* gamma_q,
    std::uint16_t const* gamma_k, float const* inv_freq, float eps,
    std::int32_t position, std::uint16_t* q_out, std::uint16_t* g_out,
    std::uint16_t* kv, std::uint32_t attn_layer, std::uint64_t capacity,
    std::uint64_t token, Stream const& stream) {
  auto st = require_stream(stream, "attention_prepare");
  if (!st) {
    return st;
  }
  if (qg == nullptr || k_raw == nullptr || v_raw == nullptr ||
      gamma_q == nullptr || gamma_k == nullptr || inv_freq == nullptr ||
      q_out == nullptr || g_out == nullptr || kv == nullptr) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "attention_prepare",
                                      "null preparation operand"));
  }
  if (qg == q_out || qg == g_out || q_out == g_out || k_raw == kv ||
      v_raw == kv) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "attention_prepare",
                                      "projected, prepared, and cache buffers must be distinct"));
  }
  if (attn_layer >= kAttnLayers) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "attention_prepare",
                                      "attn_layer must be < 16"));
  }
  if (capacity == 0 || token >= capacity) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "attention_prepare",
                                      "token must be < capacity"));
  }
  if (position < 0) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "attention_prepare",
                                      "position must be >= 0"));
  }
  if (!finite_pos(eps)) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "attention_prepare",
                                      "epsilon must be finite and > 0"));
  }
  float const pos = static_cast<float>(position);
  attention_prepare_kernel<<<kAttnPrepBlocks, kAttnPrepThreads, 0,
                             stream.native()>>>(
      qg, k_raw, v_raw, gamma_q, gamma_k, inv_freq, eps, pos, q_out, g_out, kv,
      attn_layer, capacity, token);
  return check(cudaGetLastError(), "attention_prepare_kernel");
}

}  // namespace qw38::cuda
