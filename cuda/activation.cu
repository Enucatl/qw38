#include "cuda/activation.hpp"

#include "cuda/copy.hpp"

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

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
  }
  return v;
}

template <int Threads>
__device__ float block_sum(float v) {
  static_assert(Threads == 128 || Threads == 256);
  constexpr int kWarps = Threads / 32;
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

template <int Threads>
__device__ float block_max(float v) {
  static_assert(Threads == 128 || Threads == 256);
  constexpr int kWarps = Threads / 32;
  v = warp_max(v);
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
    t = warp_max(t);
    if (lane == 0) {
      sm[0] = t;
    }
  }
  __syncthreads();
  return sm[0];
}

__device__ __forceinline__ float sigmoid_fp32(float u) {
  if (u >= 0.0f) {
    float const e = expf(-u);
    return 1.0f / (1.0f + e);
  }
  float const e = expf(u);
  return e / (1.0f + e);
}

__device__ __forceinline__ float silu_fp32(float z) {
  return z * sigmoid_fp32(z);
}

__global__ void embed_gather_kernel(std::uint16_t const* table,
                                    std::uint32_t token_id, float* residual) {
  std::uint32_t const row = token_id * kHidden;
  for (std::uint32_t i = threadIdx.x; i < kHidden; i += blockDim.x) {
    residual[i] = bf16_to_fp32(table[row + i]);
  }
}

__global__ void hidden_rms_kernel(float const* residual,
                                  std::uint16_t const* gamma, float eps,
                                  std::uint16_t* out_bf16) {
  float const* x = residual + static_cast<std::size_t>(blockIdx.x) * kHidden;
  std::uint16_t* y =
      out_bf16 + static_cast<std::size_t>(blockIdx.x) * kHidden;
  float scale = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < kHidden; i += blockDim.x) {
    scale = fmaxf(scale, fabsf(x[i]));
  }
  scale = block_max<kHiddenRmsThreads>(scale);
  float const safe_scale = scale == 0.0f ? 1.0f : scale;
  float sumsq = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < kHidden; i += blockDim.x) {
    float const scaled = x[i] / safe_scale;
    sumsq += scaled * scaled;
  }
  sumsq = block_sum<kHiddenRmsThreads>(sumsq);
  float const scaled_eps = sqrtf(eps) / safe_scale;
  float const inv_scaled_rms =
      1.0f / sqrtf(sumsq / static_cast<float>(kHidden) +
                   scaled_eps * scaled_eps);
  for (std::uint32_t i = threadIdx.x; i < kHidden; i += blockDim.x) {
    float const g = bf16_to_fp32(gamma[i]);
    float const normalized = (x[i] / safe_scale) * inv_scaled_rms;
    float const v = (1.0f + g) * normalized;
    y[i] = fp32_to_bf16_rne(v);
  }
}

__global__ void qk_rms_rope_kernel(std::uint16_t const* projected_heads,
                                   std::uint16_t const* gamma, float eps,
                                   float const* inv_freq, float position,
                                   std::uint16_t* out_bf16) {
  std::uint16_t const* x =
      projected_heads + static_cast<std::size_t>(blockIdx.x) * kHeadDim;
  std::uint16_t* out =
      out_bf16 + static_cast<std::size_t>(blockIdx.x) * kHeadDim;
  __shared__ float normalized[kHeadDim];
  float scale = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    float const v = bf16_to_fp32(x[i]);
    normalized[i] = v;
    scale = fmaxf(scale, fabsf(v));
  }
  scale = block_max<kHeadNormThreads>(scale);
  float const safe_scale = scale == 0.0f ? 1.0f : scale;
  float sumsq = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    float const scaled = normalized[i] / safe_scale;
    sumsq += scaled * scaled;
  }
  sumsq = block_sum<kHeadNormThreads>(sumsq);
  float const scaled_eps = sqrtf(eps) / safe_scale;
  float const inv_scaled_rms =
      1.0f / sqrtf(sumsq / static_cast<float>(kHeadDim) +
                   scaled_eps * scaled_eps);
  for (std::uint32_t i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    float const g = bf16_to_fp32(gamma[i]);
    float const unit = (normalized[i] / safe_scale) * inv_scaled_rms;
    normalized[i] = (1.0f + g) * unit;
  }
  __syncthreads();

  constexpr std::uint32_t kHalf = kRotaryDim / 2;
  if (threadIdx.x < kHalf) {
    std::uint32_t const j = threadIdx.x;
    float const x0 = normalized[j];
    float const x1 = normalized[j + kHalf];
    float const phase = position * inv_freq[j];
    float const c = cosf(phase);
    float const s = sinf(phase);
    out[j] = fp32_to_bf16_rne(x0 * c - x1 * s);
    out[j + kHalf] = fp32_to_bf16_rne(x1 * c + x0 * s);
  }
  for (std::uint32_t i = kRotaryDim + threadIdx.x; i < kHeadDim;
       i += blockDim.x) {
    out[i] = fp32_to_bf16_rne(normalized[i]);
  }
}

__global__ void qk_rms_kernel(float const* heads, std::uint16_t const* gamma,
                              float eps, std::uint16_t* out_bf16) {
  float const* x = heads + static_cast<std::size_t>(blockIdx.x) * kHeadDim;
  std::uint16_t* y =
      out_bf16 + static_cast<std::size_t>(blockIdx.x) * kHeadDim;
  float scale = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    scale = fmaxf(scale, fabsf(x[i]));
  }
  scale = block_max<kHeadNormThreads>(scale);
  float const safe_scale = scale == 0.0f ? 1.0f : scale;
  float sumsq = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    float const scaled = x[i] / safe_scale;
    sumsq += scaled * scaled;
  }
  sumsq = block_sum<kHeadNormThreads>(sumsq);
  float const scaled_eps = sqrtf(eps) / safe_scale;
  float const inv_scaled_rms =
      1.0f / sqrtf(sumsq / static_cast<float>(kHeadDim) +
                   scaled_eps * scaled_eps);
  for (std::uint32_t i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    float const g = bf16_to_fp32(gamma[i]);
    float const normalized = (x[i] / safe_scale) * inv_scaled_rms;
    float const v = (1.0f + g) * normalized;
    y[i] = fp32_to_bf16_rne(v);
  }
}

__global__ void gdn_gated_rms_kernel(float const* o,
                                     std::uint16_t const* z_bf16,
                                     std::uint16_t const* gamma, float eps,
                                     std::uint16_t* out_bf16) {
  float const* oh = o + static_cast<std::size_t>(blockIdx.x) * kGdnHeadDim;
  std::uint16_t const* zh =
      z_bf16 + static_cast<std::size_t>(blockIdx.x) * kGdnHeadDim;
  std::uint16_t* y =
      out_bf16 + static_cast<std::size_t>(blockIdx.x) * kGdnHeadDim;
  float scale = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < kGdnHeadDim; i += blockDim.x) {
    scale = fmaxf(scale, fabsf(oh[i]));
  }
  scale = block_max<kHeadNormThreads>(scale);
  float const safe_scale = scale == 0.0f ? 1.0f : scale;
  float sumsq = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < kGdnHeadDim; i += blockDim.x) {
    float const scaled = oh[i] / safe_scale;
    sumsq += scaled * scaled;
  }
  sumsq = block_sum<kHeadNormThreads>(sumsq);
  float const scaled_eps = sqrtf(eps) / safe_scale;
  float const inv_scaled_rms =
      1.0f / sqrtf(sumsq / static_cast<float>(kGdnHeadDim) +
                   scaled_eps * scaled_eps);
  for (std::uint32_t i = threadIdx.x; i < kGdnHeadDim; i += blockDim.x) {
    float const g = bf16_to_fp32(gamma[i]);
    float const z = bf16_to_fp32(zh[i]);
    float const normalized = (oh[i] / safe_scale) * inv_scaled_rms;
    float const v = g * normalized * silu_fp32(z);
    y[i] = fp32_to_bf16_rne(v);
  }
}

__global__ void sigmoid_fp32_kernel(float const* in, float* out,
                                    std::uint32_t n) {
  std::uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = sigmoid_fp32(in[i]);
  }
}

__global__ void silu_fp32_kernel(float const* in, float* out, std::uint32_t n) {
  std::uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = silu_fp32(in[i]);
  }
}

__global__ void partial_rope_kernel(std::uint16_t const* heads,
                                    float const* inv_freq, float position,
                                    std::uint16_t* out) {
  std::uint16_t const* x =
      heads + static_cast<std::size_t>(blockIdx.x) * kHeadDim;
  std::uint16_t* y = out + static_cast<std::size_t>(blockIdx.x) * kHeadDim;
  constexpr std::uint32_t kHalf = kRotaryDim / 2;
  if (threadIdx.x < kHalf) {
    std::uint32_t const j = threadIdx.x;
    float const x0 = bf16_to_fp32(x[j]);
    float const x1 = bf16_to_fp32(x[j + kHalf]);
    float const phase = position * inv_freq[j];
    float const c = cosf(phase);
    float const s = sinf(phase);
    y[j] = fp32_to_bf16_rne(x0 * c - x1 * s);
    y[j + kHalf] = fp32_to_bf16_rne(x1 * c + x0 * s);
  }
  for (std::uint32_t i = kRotaryDim + threadIdx.x; i < kHeadDim;
       i += blockDim.x) {
    y[i] = x[i];
  }
}

__device__ __forceinline__ void warp_maxloc(float& v, std::uint32_t& idx) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    float const ov = __shfl_down_sync(0xffffffffu, v, off);
    std::uint32_t const oi = __shfl_down_sync(0xffffffffu, idx, off);
    if (ov > v || (ov == v && oi < idx)) {
      v = ov;
      idx = oi;
    }
  }
}

__global__ void argmax_fp32_kernel(float const* logits, std::uint32_t n,
                                   std::uint32_t* out_index) {
  __shared__ std::uint32_t non_finite;
  if (threadIdx.x == 0) {
    non_finite = 0;
  }
  __syncthreads();

  float best = __int_as_float(0xff800000);
  std::uint32_t best_i = 0xffffffffu;
  for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    float const v = logits[i];
    if (!isfinite(v)) {
      atomicExch(&non_finite, 1u);
      continue;
    }
    if (v > best || (v == best && i < best_i)) {
      best = v;
      best_i = i;
    }
  }
  warp_maxloc(best, best_i);
  __shared__ float sm_v[8];
  __shared__ std::uint32_t sm_i[8];
  int const warp = threadIdx.x >> 5;
  int const lane = threadIdx.x & 31;
  if (lane == 0) {
    sm_v[warp] = best;
    sm_i[warp] = best_i;
  }
  __syncthreads();
  if (warp == 0) {
    best = (lane < 8) ? sm_v[lane] : __int_as_float(0xff800000);
    best_i = (lane < 8) ? sm_i[lane] : 0xffffffffu;
    warp_maxloc(best, best_i);
    if (lane == 0) {
      *out_index = non_finite == 0 ? best_i : 0xffffffffu;
    }
  }
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

unsigned elementwise_blocks(std::uint32_t n) {
  unsigned const threads = 256;
  unsigned blocks = static_cast<unsigned>((n + threads - 1) / threads);
  return blocks == 0 ? 1u : blocks;
}

}  // namespace

std::expected<void, Error> launch_embed_gather(std::uint16_t const* table,
                                               std::uint32_t vocab,
                                               std::uint32_t token_id,
                                               float* residual,
                                               Stream const& stream) {
  auto st = require_stream(stream, "embed_gather");
  if (!st) {
    return st;
  }
  if (table == nullptr || residual == nullptr) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "embed_gather",
                                      "null table or residual"));
  }
  if (vocab == 0 || token_id >= vocab) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "embed_gather",
                                      "token_id >= vocab"));
  }
  embed_gather_kernel<<<1, kHiddenRmsThreads, 0, stream.native()>>>(
      table, token_id, residual);
  return check(cudaGetLastError(), "embed_gather_kernel");
}

std::expected<void, Error> launch_hidden_rms(float const* residual,
                                             std::uint16_t const* gamma,
                                             float eps, std::uint32_t n_tokens,
                                             std::uint16_t* out_bf16,
                                             Stream const& stream) {
  auto st = require_stream(stream, "hidden_rms");
  if (!st) {
    return st;
  }
  if (residual == nullptr || gamma == nullptr || out_bf16 == nullptr) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "hidden_rms",
                                      "null residual, gamma, or out"));
  }
  if (n_tokens == 0) {
    return std::unexpected(
        make_error(ErrorCode::InvalidArgument, "hidden_rms", "n_tokens == 0"));
  }
  if (!finite_pos(eps)) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "hidden_rms",
                                      "epsilon must be finite and > 0"));
  }
  hidden_rms_kernel<<<n_tokens, kHiddenRmsThreads, 0, stream.native()>>>(
      residual, gamma, eps, out_bf16);
  return check(cudaGetLastError(), "hidden_rms_kernel");
}

std::expected<void, Error> launch_qk_rms_rope(
    std::uint16_t const* projected_heads_bf16, std::uint16_t const* gamma,
    float eps, float const* inv_freq, std::int32_t position,
    std::uint32_t n_heads, std::uint16_t* out_bf16, Stream const& stream) {
  auto st = require_stream(stream, "qk_rms_rope");
  if (!st) {
    return st;
  }
  if (projected_heads_bf16 == nullptr || gamma == nullptr ||
      inv_freq == nullptr || out_bf16 == nullptr) {
    return std::unexpected(make_error(
        ErrorCode::InvalidArgument, "qk_rms_rope",
        "null projected heads, gamma, inv_freq, or out"));
  }
  if (n_heads == 0) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "qk_rms_rope", "n_heads == 0"));
  }
  if (!finite_pos(eps)) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "qk_rms_rope",
                                      "epsilon must be finite and > 0"));
  }
  if (position < 0) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "qk_rms_rope",
                                      "position must be >= 0"));
  }
  float const pos = static_cast<float>(position);
  qk_rms_rope_kernel<<<n_heads, kHeadNormThreads, 0, stream.native()>>>(
      projected_heads_bf16, gamma, eps, inv_freq, pos, out_bf16);
  return check(cudaGetLastError(), "qk_rms_rope_kernel");
}

std::expected<void, Error> launch_qk_rms(float const* heads,
                                         std::uint16_t const* gamma, float eps,
                                         std::uint32_t n_heads,
                                         std::uint16_t* out_bf16,
                                         Stream const& stream) {
  auto st = require_stream(stream, "qk_rms");
  if (!st) {
    return st;
  }
  if (heads == nullptr || gamma == nullptr || out_bf16 == nullptr) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "qk_rms",
                                      "null heads, gamma, or out"));
  }
  if (n_heads == 0) {
    return std::unexpected(
        make_error(ErrorCode::InvalidArgument, "qk_rms", "n_heads == 0"));
  }
  if (!finite_pos(eps)) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "qk_rms",
                                      "epsilon must be finite and > 0"));
  }
  qk_rms_kernel<<<n_heads, kHeadNormThreads, 0, stream.native()>>>(
      heads, gamma, eps, out_bf16);
  return check(cudaGetLastError(), "qk_rms_kernel");
}

std::expected<void, Error> launch_gdn_gated_rms(
    float const* o, std::uint16_t const* z_bf16, std::uint16_t const* gamma,
    float eps, std::uint32_t n_heads, std::uint16_t* out_bf16,
    Stream const& stream) {
  auto st = require_stream(stream, "gdn_gated_rms");
  if (!st) {
    return st;
  }
  if (o == nullptr || z_bf16 == nullptr || gamma == nullptr ||
      out_bf16 == nullptr) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "gdn_gated_rms",
                                      "null o, z, gamma, or out"));
  }
  if (n_heads == 0) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "gdn_gated_rms", "n_heads == 0"));
  }
  if (!finite_pos(eps)) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "gdn_gated_rms",
                                      "epsilon must be finite and > 0"));
  }
  gdn_gated_rms_kernel<<<n_heads, kHeadNormThreads, 0, stream.native()>>>(
      o, z_bf16, gamma, eps, out_bf16);
  return check(cudaGetLastError(), "gdn_gated_rms_kernel");
}

std::expected<void, Error> launch_sigmoid_fp32(float const* in, float* out,
                                               std::uint32_t n,
                                               Stream const& stream) {
  auto st = require_stream(stream, "sigmoid_fp32");
  if (!st) {
    return st;
  }
  if (in == nullptr || out == nullptr || n == 0) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "sigmoid_fp32",
                                      "null in/out or n == 0"));
  }
  sigmoid_fp32_kernel<<<elementwise_blocks(n), 256, 0, stream.native()>>>(in,
                                                                         out, n);
  return check(cudaGetLastError(), "sigmoid_fp32_kernel");
}

std::expected<void, Error> launch_silu_fp32(float const* in, float* out,
                                            std::uint32_t n,
                                            Stream const& stream) {
  auto st = require_stream(stream, "silu_fp32");
  if (!st) {
    return st;
  }
  if (in == nullptr || out == nullptr || n == 0) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "silu_fp32",
                                      "null in/out or n == 0"));
  }
  silu_fp32_kernel<<<elementwise_blocks(n), 256, 0, stream.native()>>>(in, out,
                                                                      n);
  return check(cudaGetLastError(), "silu_fp32_kernel");
}

std::expected<void, Error> launch_partial_rope(std::uint16_t const* heads_bf16,
                                               float const* inv_freq,
                                               std::int32_t position,
                                               std::uint32_t n_heads,
                                               std::uint16_t* out_bf16,
                                               Stream const& stream) {
  auto st = require_stream(stream, "partial_rope");
  if (!st) {
    return st;
  }
  if (heads_bf16 == nullptr || inv_freq == nullptr || out_bf16 == nullptr) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "partial_rope",
                                      "null heads, inv_freq, or out"));
  }
  if (n_heads == 0) {
    return std::unexpected(
        make_error(ErrorCode::InvalidArgument, "partial_rope", "n_heads == 0"));
  }
  if (position < 0) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "partial_rope",
                                      "position must be >= 0"));
  }
  float const pos = static_cast<float>(position);
  partial_rope_kernel<<<n_heads, kHeadNormThreads, 0, stream.native()>>>(
      heads_bf16, inv_freq, pos, out_bf16);
  return check(cudaGetLastError(), "partial_rope_kernel");
}

std::expected<void, Error> launch_argmax_fp32(float const* logits,
                                              std::uint32_t n,
                                              std::uint32_t* out_index,
                                              Stream const& stream) {
  auto st = require_stream(stream, "argmax_fp32");
  if (!st) {
    return st;
  }
  if (logits == nullptr || out_index == nullptr || n == 0) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "argmax_fp32",
                                      "null logits/out or n == 0"));
  }
  argmax_fp32_kernel<<<1, kHiddenRmsThreads, 0, stream.native()>>>(logits, n,
                                                                  out_index);
  if (auto launch = check(cudaGetLastError(), "argmax_fp32_kernel"); !launch) {
    return launch;
  }
  std::uint32_t host_index = 0xffffffffu;
  if (auto copy = copy_d2h(&host_index, out_index, sizeof(host_index), stream);
      !copy) {
    return copy;
  }
  if (auto sync = stream.sync(); !sync) {
    return sync;
  }
  if (host_index == 0xffffffffu) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "argmax_fp32",
                                      "logits must contain only finite values"));
  }
  return {};
}

}  // namespace qw38::cuda
