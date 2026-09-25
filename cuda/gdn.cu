#include "cuda/gdn.hpp"

#include "cuda/activation.hpp"
#include "cuda/prefill.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <initializer_list>
#include <string_view>

namespace qw38::cuda {
namespace {

inline constexpr std::uint32_t kGdnKernelHeadDim = 128;

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

__device__ float block_sum_128(float v, float* sm) {
  v = warp_sum(v);
  int const warp = threadIdx.x >> 5;
  int const lane = threadIdx.x & 31;
  if (lane == 0) {
    sm[warp] = v;
  }
  __syncthreads();
  float t = 0.0f;
  if (warp == 0) {
    t = (lane < 4) ? sm[lane] : 0.0f;
    t = warp_sum(t);
    if (lane == 0) {
      sm[0] = t;
    }
  }
  __syncthreads();
  return sm[0];
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
  }
  return v;
}

__device__ float block_max_128(float v, float* sm) {
  v = warp_max(v);
  int const warp = threadIdx.x >> 5;
  int const lane = threadIdx.x & 31;
  if (lane == 0) {
    sm[warp] = v;
  }
  __syncthreads();
  float t = 0.0f;
  if (warp == 0) {
    t = (lane < 4) ? sm[lane] : 0.0f;
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

__device__ __forceinline__ float softplus_fp32(float x) {
  return fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
}

__global__ void gdn_conv_silu_kernel(std::uint16_t const* qkv,
                                     std::uint16_t const* taps,
                                     std::uint16_t* history,
                                     std::uint32_t cursor,
                                     std::uint16_t* convolved) {
  std::uint32_t const c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= kGdnQkvWidth) {
    return;
  }
  float acc = 0.0f;
#pragma unroll
  for (std::uint32_t j = 0; j < kGdnConvKernel; ++j) {
    std::uint16_t xbits = qkv[c];
    if (j < kGdnConvHistoryTaps) {
      std::uint32_t const slot = (cursor + j) % kGdnConvHistoryTaps;
      xbits = history[slot * kGdnQkvWidth + c];
    }
    float const w = bf16_to_fp32(taps[j * kGdnQkvWidth + c]);
    acc += w * bf16_to_fp32(xbits);
  }
  convolved[c] = fp32_to_bf16_rne(silu_fp32(acc));
  history[cursor * kGdnQkvWidth + c] = qkv[c];
}

__global__ void gdn_prefill_conv_kernel(std::uint16_t const* qkv,
                                        std::uint16_t const* taps,
                                        std::uint16_t const* history,
                                        std::uint32_t cursor,
                                        std::uint16_t* convolved,
                                        std::uint32_t valid_tokens) {
  std::uint32_t const c = blockIdx.x * blockDim.x + threadIdx.x;
  std::uint32_t const token = blockIdx.y;
  if (c >= kGdnQkvWidth || token >= valid_tokens) return;
  float acc = 0.0f;
#pragma unroll
  for (std::uint32_t tap = 0; tap < kGdnConvKernel; ++tap) {
    int const source = static_cast<int>(token) + static_cast<int>(tap) - 3;
    auto const x = source < 0
        ? history[((cursor + token + tap) % 3u) * kGdnQkvWidth + c]
        : qkv[static_cast<std::size_t>(source) * kGdnQkvWidth + c];
    acc += bf16_to_fp32(taps[tap * kGdnQkvWidth + c]) * bf16_to_fp32(x);
  }
  convolved[static_cast<std::size_t>(token) * kGdnQkvWidth + c] =
      fp32_to_bf16_rne(silu_fp32(acc));
}

__global__ void gdn_prefill_history_commit_kernel(
    std::uint16_t const* qkv, std::uint16_t* history,
    std::uint32_t cursor, std::uint32_t valid_tokens) {
  std::uint32_t const c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= kGdnQkvWidth) return;
  // One thread owns all three slots for a channel. Load old values before any
  // stores, including the one- and two-token cases that still need old slots.
  std::uint16_t saved[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    int const source = static_cast<int>(valid_tokens) - 3 + j;
    saved[j] = source < 0
        ? history[((cursor + static_cast<std::uint32_t>(source + 3)) % 3u)
                  * kGdnQkvWidth + c]
        : qkv[static_cast<std::size_t>(source) * kGdnQkvWidth + c];
  }
  std::uint32_t const next = (cursor + valid_tokens) % 3u;
#pragma unroll
  for (std::uint32_t j = 0; j < 3; ++j)
    history[((next + j) % 3u) * kGdnQkvWidth + c] = saved[j];
}

__device__ void l2_normalize_head(std::uint16_t const* src, float* dst, float eps,
                                  float* sm) {
  float max_abs = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < kGdnKernelHeadDim; i += blockDim.x) {
    max_abs = fmaxf(max_abs, fabsf(bf16_to_fp32(src[i])));
  }
  max_abs = block_max_128(max_abs, sm);
  if (max_abs == 0.0f) {
    for (std::uint32_t i = threadIdx.x; i < kGdnKernelHeadDim; i += blockDim.x) {
      dst[i] = 0.0f;
    }
    __syncthreads();
    return;
  }

  float sumsq = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < kGdnKernelHeadDim; i += blockDim.x) {
    float const scaled = bf16_to_fp32(src[i]) / max_abs;
    sumsq += scaled * scaled;
  }
  sumsq = block_sum_128(sumsq, sm);
  float const denom = hypotf(sqrtf(sumsq), sqrtf(eps) / max_abs);
  for (std::uint32_t i = threadIdx.x; i < kGdnKernelHeadDim; i += blockDim.x) {
    dst[i] = (bf16_to_fp32(src[i]) / max_abs) / denom;
  }
  __syncthreads();
}

__global__ void gdn_prepare_kernel(std::uint16_t const* convolved, float const* a,
                                   float const* b, std::uint16_t const* a_log,
                                   std::uint16_t const* dt_bias, float eps,
                                   float* q_hat, float* k_hat, float* alpha,
                                   float* beta) {
  __shared__ float sm[4];
  std::uint32_t const token = blockIdx.y;
  std::uint32_t const head = blockIdx.x;
  std::uint32_t const q_off = head * kGdnKernelHeadDim;
  std::uint32_t const k_off = (kGdnKeyHeads + head) * kGdnKernelHeadDim;
  convolved += static_cast<std::size_t>(token) * kGdnQkvWidth;
  q_hat += static_cast<std::size_t>(token) * kGdnKeyHeads * kGdnKernelHeadDim;
  k_hat += static_cast<std::size_t>(token) * kGdnKeyHeads * kGdnKernelHeadDim;
  a += static_cast<std::size_t>(token) * kGdnValueHeads;
  b += static_cast<std::size_t>(token) * kGdnValueHeads;
  alpha += static_cast<std::size_t>(token) * kGdnValueHeads;
  beta += static_cast<std::size_t>(token) * kGdnValueHeads;
  l2_normalize_head(convolved + q_off, q_hat + q_off, eps, sm);
  l2_normalize_head(convolved + k_off, k_hat + q_off, eps, sm);

  if (threadIdx.x < kGdnRepeat) {
    std::uint32_t const vh = head * kGdnRepeat + threadIdx.x;
    float const alog = bf16_to_fp32(a_log[vh]);
    float const dt = bf16_to_fp32(dt_bias[vh]);
    float const sp = softplus_fp32(a[vh] + dt);
    alpha[vh] = expf(-expf(alog) * sp);
    beta[vh] = sigmoid_fp32(b[vh]);
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

struct DeviceSpan {
  void const* pointer;
  std::uint64_t bytes;
  std::uint32_t alignment;
};

std::expected<void, Error> validate_prefill_spans(
    Stream const& stream, std::initializer_list<DeviceSpan> spans) {
  auto guard = stream.activate();
  if (!guard) return std::unexpected(guard.error());
  for (auto const& span : spans) {
    if (auto st = validate_prefill_device_span(
            span.pointer, span.bytes, span.alignment, stream.device()); !st)
      return st;
  }
  for (auto a = spans.begin(); a != spans.end(); ++a) {
    auto const ab = reinterpret_cast<std::uintptr_t>(a->pointer);
    for (auto b = a + 1; b != spans.end(); ++b) {
      auto const bb = reinterpret_cast<std::uintptr_t>(b->pointer);
      if (ab < bb + b->bytes && bb < ab + a->bytes)
        return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                          "gdn_prefill.alias", "live device spans overlap"));
    }
  }
  return {};
}

__global__ void gdn_recurrence_kernel(float const* __restrict__ q_hat,
                                      float const* __restrict__ k_hat,
                                      float const* __restrict__ alpha,
                                      float const* __restrict__ beta,
                                      std::uint16_t const* __restrict__ v,
                                      float* __restrict__ s,
                                      float* __restrict__ o) {
  __shared__ float q_sm[kGdnKernelHeadDim];
  __shared__ float k_sm[kGdnKernelHeadDim];

  std::uint32_t const vh =
      blockIdx.x / static_cast<std::uint32_t>(kGdnRecurBlocksPerHead);
  std::uint32_t const vgroup =
      blockIdx.x % static_cast<std::uint32_t>(kGdnRecurBlocksPerHead);
  std::uint32_t const warp = static_cast<std::uint32_t>(threadIdx.x) >> 5;
  std::uint32_t const lane = static_cast<std::uint32_t>(threadIdx.x) & 31u;
  std::uint32_t const value_j =
      vgroup * static_cast<std::uint32_t>(kGdnRecurValuesPerBlock) + warp;
  std::uint32_t const key_head = vh / kGdnRepeat;

  q_sm[threadIdx.x] =
      q_hat[key_head * kGdnKernelHeadDim + static_cast<std::uint32_t>(threadIdx.x)];
  k_sm[threadIdx.x] =
      k_hat[key_head * kGdnKernelHeadDim + static_cast<std::uint32_t>(threadIdx.x)];
  __syncthreads();

  float const a = alpha[vh];
  float const b = beta[vh];
  float const vj = bf16_to_fp32(v[vh * kGdnKernelHeadDim + value_j]);
  float* row = s + (static_cast<std::size_t>(vh) * kGdnKernelHeadDim + value_j) * kGdnKernelHeadDim;

  float s0 = row[lane];
  float s1 = row[lane + 32u];
  float s2 = row[lane + 64u];
  float s3 = row[lane + 96u];
  float const d0 = a * s0;
  float const d1 = a * s1;
  float const d2 = a * s2;
  float const d3 = a * s3;
  float const k0 = k_sm[lane];
  float const k1 = k_sm[lane + 32u];
  float const k2 = k_sm[lane + 64u];
  float const k3 = k_sm[lane + 96u];

  float p = d0 * k0 + d1 * k1 + d2 * k2 + d3 * k3;
  p = warp_sum(p);
  p = __shfl_sync(0xffffffffu, p, 0);
  float const e = b * (vj - p);

  s0 = d0 + k0 * e;
  s1 = d1 + k1 * e;
  s2 = d2 + k2 * e;
  s3 = d3 + k3 * e;
  row[lane] = s0;
  row[lane + 32u] = s1;
  row[lane + 64u] = s2;
  row[lane + 96u] = s3;

  float const scale = 1.0f / sqrtf(static_cast<float>(kGdnKernelHeadDim));
  float acc = s0 * q_sm[lane] * scale + s1 * q_sm[lane + 32u] * scale +
              s2 * q_sm[lane + 64u] * scale + s3 * q_sm[lane + 96u] * scale;
  acc = warp_sum(acc);
  if (lane == 0u) {
    o[vh * kGdnKernelHeadDim + value_j] = acc;
  }
}

__global__ void gdn_prefill_recurrence_kernel(
    float const* __restrict__ q_hat, float const* __restrict__ k_hat,
    float const* __restrict__ alpha, float const* __restrict__ beta,
    std::uint16_t const* __restrict__ convolved, float* __restrict__ s,
    float* __restrict__ o, std::uint32_t first, std::uint32_t count) {
  __shared__ float q_sm[kGdnKernelHeadDim];
  __shared__ float k_sm[kGdnKernelHeadDim];
  std::uint32_t const vh = blockIdx.x / kGdnRecurBlocksPerHead;
  std::uint32_t const vgroup = blockIdx.x % kGdnRecurBlocksPerHead;
  std::uint32_t const warp = static_cast<std::uint32_t>(threadIdx.x) >> 5;
  std::uint32_t const lane = static_cast<std::uint32_t>(threadIdx.x) & 31u;
  std::uint32_t const value_j = vgroup * kGdnRecurValuesPerBlock + warp;
  std::uint32_t const key_head = vh / kGdnRepeat;
  float* row = s + (static_cast<std::size_t>(vh) * kGdnKernelHeadDim + value_j) * kGdnKernelHeadDim;
  float s0 = row[lane], s1 = row[lane + 32u];
  float s2 = row[lane + 64u], s3 = row[lane + 96u];
  float const scale = 1.0f / sqrtf(static_cast<float>(kGdnKernelHeadDim));
  for (std::uint32_t t = first; t < first + count; ++t) {
    std::size_t const qk_base =
        (static_cast<std::size_t>(t) * kGdnKeyHeads + key_head) * kGdnKernelHeadDim;
    q_sm[threadIdx.x] = q_hat[qk_base + threadIdx.x];
    k_sm[threadIdx.x] = k_hat[qk_base + threadIdx.x];
    __syncthreads();
    float const a = alpha[static_cast<std::size_t>(t) * kGdnValueHeads + vh];
    float const b = beta[static_cast<std::size_t>(t) * kGdnValueHeads + vh];
    float const vj = bf16_to_fp32(convolved[
        static_cast<std::size_t>(t) * kGdnQkvWidth + 4096u +
        vh * kGdnKernelHeadDim + value_j]);
    float const d0 = a * s0, d1 = a * s1, d2 = a * s2, d3 = a * s3;
    float const k0 = k_sm[lane], k1 = k_sm[lane + 32u];
    float const k2 = k_sm[lane + 64u], k3 = k_sm[lane + 96u];
    float p = d0 * k0 + d1 * k1 + d2 * k2 + d3 * k3;
    p = __shfl_sync(0xffffffffu, warp_sum(p), 0);
    float const e = b * (vj - p);
    s0 = d0 + k0 * e; s1 = d1 + k1 * e;
    s2 = d2 + k2 * e; s3 = d3 + k3 * e;
    float acc = s0 * q_sm[lane] * scale +
                s1 * q_sm[lane + 32u] * scale +
                s2 * q_sm[lane + 64u] * scale +
                s3 * q_sm[lane + 96u] * scale;
    acc = warp_sum(acc);
    if (lane == 0u)
      o[(static_cast<std::size_t>(t) * kGdnValueHeads + vh) * kGdnKernelHeadDim + value_j] = acc;
    __syncthreads();
  }
  row[lane] = s0; row[lane + 32u] = s1;
  row[lane + 64u] = s2; row[lane + 96u] = s3;
}

}  // namespace

std::expected<void, Error> launch_gdn_conv_silu(
    std::uint16_t const* qkv, std::uint16_t const* taps, std::uint16_t* history,
    std::uint32_t cursor, std::uint16_t* convolved, Stream const& stream) {
  auto st = require_stream(stream, "gdn_conv_silu");
  if (!st) {
    return st;
  }
  if (qkv == nullptr || taps == nullptr || history == nullptr ||
      convolved == nullptr) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "gdn_conv_silu",
                                      "null qkv, taps, history, or convolved"));
  }
  if (cursor >= kGdnConvHistoryTaps) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "gdn_conv_silu",
                                      "cursor must be < 3"));
  }
  if (qkv == convolved) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "gdn_conv_silu",
                                      "qkv and convolved must be distinct"));
  }
  unsigned const blocks =
      (kGdnQkvWidth + static_cast<unsigned>(kGdnConvThreads) - 1u) /
      static_cast<unsigned>(kGdnConvThreads);
  auto guard = stream.activate();
  if (!guard) return std::unexpected(guard.error());
  gdn_conv_silu_kernel<<<blocks, kGdnConvThreads, 0, stream.native()>>>(
      qkv, taps, history, cursor, convolved);
  return check(cudaGetLastError(), "gdn_conv_silu_kernel");
}

std::expected<void, Error> launch_gdn_prefill_conv(
    std::uint16_t const* qkv, std::uint16_t const* taps,
    std::uint16_t const* history, std::uint32_t cursor,
    std::uint16_t* convolved, std::uint32_t valid_tokens,
    Stream const& stream) {
  if (auto st = require_stream(stream, "gdn_prefill_conv"); !st) return st;
  if (!qkv || !taps || !history || !convolved || cursor >= 3u ||
      valid_tokens == 0 || valid_tokens > 1024u || qkv == convolved)
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "gdn_prefill_conv", "invalid operand or count"));
  if (auto st = validate_prefill_spans(stream, {
          {qkv, static_cast<std::uint64_t>(valid_tokens) * kGdnQkvWidth * 2u, 2u},
          {taps, kGdnConvKernel * kGdnQkvWidth * 2u, 2u},
          {history, kGdnConvHistoryTaps * kGdnQkvWidth * 2u, 2u},
          {convolved, static_cast<std::uint64_t>(valid_tokens) * kGdnQkvWidth * 2u, 2u}});
      !st) return st;
  auto guard = stream.activate();
  if (!guard) return std::unexpected(guard.error());
  dim3 const grid{(kGdnQkvWidth + kGdnConvThreads - 1u) / kGdnConvThreads,
                  valid_tokens};
  gdn_prefill_conv_kernel<<<grid, kGdnConvThreads, 0, stream.native()>>>(
      qkv, taps, history, cursor, convolved, valid_tokens);
  return check(cudaGetLastError(), "gdn_prefill_conv_kernel");
}

std::expected<void, Error> launch_gdn_prefill_history_commit(
    std::uint16_t const* qkv, std::uint16_t* history, std::uint32_t cursor,
    std::uint32_t valid_tokens, Stream const& stream) {
  if (auto st = require_stream(stream, "gdn_prefill_history_commit"); !st) return st;
  if (!qkv || !history || cursor >= 3u || valid_tokens == 0 ||
      valid_tokens > 1024u)
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "gdn_prefill_history_commit",
                                      "invalid operand or count"));
  if (auto st = validate_prefill_spans(stream, {
          {qkv, static_cast<std::uint64_t>(valid_tokens) * kGdnQkvWidth * 2u, 2u},
          {history, kGdnConvHistoryTaps * kGdnQkvWidth * 2u, 2u}});
      !st) return st;
  auto guard = stream.activate();
  if (!guard) return std::unexpected(guard.error());
  gdn_prefill_history_commit_kernel<<<
      (kGdnQkvWidth + kGdnConvThreads - 1u) / kGdnConvThreads,
      kGdnConvThreads, 0, stream.native()>>>(qkv, history, cursor, valid_tokens);
  return check(cudaGetLastError(), "gdn_prefill_history_commit_kernel");
}

std::expected<void, Error> launch_gdn_prepare(
    std::uint16_t const* convolved, float const* a, float const* b,
    std::uint16_t const* a_log, std::uint16_t const* dt_bias, float eps,
    float* q_hat, float* k_hat, float* alpha, float* beta, Stream const& stream) {
  auto st = require_stream(stream, "gdn_prepare");
  if (!st) {
    return st;
  }
  if (convolved == nullptr || a == nullptr || b == nullptr ||
      a_log == nullptr || dt_bias == nullptr || q_hat == nullptr ||
      k_hat == nullptr || alpha == nullptr || beta == nullptr) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "gdn_prepare",
                                      "null prepare operand"));
  }
  if (!finite_pos(eps)) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "gdn_prepare",
                                      "epsilon must be finite and > 0"));
  }
  if (q_hat == k_hat) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "gdn_prepare",
                                      "q_hat and k_hat must be distinct"));
  }
  auto guard = stream.activate();
  if (!guard) return std::unexpected(guard.error());
  gdn_prepare_kernel<<<kGdnKeyHeads, kGdnPrepThreads, 0, stream.native()>>>(
      convolved, a, b, a_log, dt_bias, eps, q_hat, k_hat, alpha, beta);
  return check(cudaGetLastError(), "gdn_prepare_kernel");
}

std::expected<void, Error> launch_gdn_prefill_prepare(
    std::uint16_t const* convolved, float const* a, float const* b,
    std::uint16_t const* a_log, std::uint16_t const* dt_bias, float eps,
    float* q_hat, float* k_hat, float* alpha, float* beta,
    std::uint32_t valid_tokens, Stream const& stream) {
  if (auto st = require_stream(stream, "gdn_prefill_prepare"); !st) return st;
  if (!convolved || !a || !b || !a_log || !dt_bias || !q_hat || !k_hat ||
      !alpha || !beta || q_hat == k_hat || !finite_pos(eps) ||
      valid_tokens == 0 || valid_tokens > 1024u)
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "gdn_prefill_prepare", "invalid operand or count"));
  if (auto st = validate_prefill_spans(stream, {
          {convolved, static_cast<std::uint64_t>(valid_tokens) * kGdnQkvWidth * 2u, 2u},
          {a, static_cast<std::uint64_t>(valid_tokens) * kGdnValueHeads * 4u, 4u},
          {b, static_cast<std::uint64_t>(valid_tokens) * kGdnValueHeads * 4u, 4u},
          {a_log, kGdnValueHeads * 2u, 2u},
          {dt_bias, kGdnValueHeads * 2u, 2u},
          {q_hat, static_cast<std::uint64_t>(valid_tokens) * kGdnKeyHeads * kGdnKernelHeadDim * 4u, 4u},
          {k_hat, static_cast<std::uint64_t>(valid_tokens) * kGdnKeyHeads * kGdnKernelHeadDim * 4u, 4u},
          {alpha, static_cast<std::uint64_t>(valid_tokens) * kGdnValueHeads * 4u, 4u},
          {beta, static_cast<std::uint64_t>(valid_tokens) * kGdnValueHeads * 4u, 4u}});
      !st) return st;
  auto guard = stream.activate();
  if (!guard) return std::unexpected(guard.error());
  dim3 const grid{kGdnKeyHeads, valid_tokens};
  gdn_prepare_kernel<<<grid, kGdnPrepThreads, 0, stream.native()>>>(
      convolved, a, b, a_log, dt_bias, eps, q_hat, k_hat, alpha, beta);
  return check(cudaGetLastError(), "gdn_prefill_prepare_kernel");
}

std::expected<void, Error> launch_gdn_recurrence(
    float const* q_hat, float const* k_hat, float const* alpha, float const* beta,
    std::uint16_t const* v, float* s, std::uint32_t s_layer, float* o,
    Stream const& stream) {
  auto st = require_stream(stream, "gdn_recurrence");
  if (!st) {
    return st;
  }
  if (q_hat == nullptr || k_hat == nullptr || alpha == nullptr || beta == nullptr ||
      v == nullptr || s == nullptr || o == nullptr) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "gdn_recurrence",
                                      "null recurrence operand"));
  }
  if (s_layer >= kGdnLayers) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "gdn_recurrence",
                                      "s_layer must be < 48"));
  }
  if (q_hat == k_hat) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "gdn_recurrence",
                                      "q_hat and k_hat must be distinct"));
  }
  if (s == o) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "gdn_recurrence",
                                      "s and o must be distinct"));
  }
  float* layer_s = s + static_cast<std::size_t>(s_layer) * kGdnSElemsPerLayer;
  auto guard = stream.activate();
  if (!guard) return std::unexpected(guard.error());
  gdn_recurrence_kernel<<<kGdnRecurBlocksPerLayer, kGdnRecurThreads, 0,
                          stream.native()>>>(q_hat, k_hat, alpha, beta, v, layer_s,
                                           o);
  return check(cudaGetLastError(), "gdn_recurrence_kernel");
}

std::expected<void, Error> launch_gdn_prefill_recurrence(
    float const* q_hat, float const* k_hat, float const* alpha,
    float const* beta, std::uint16_t const* convolved, float* s,
    std::uint32_t s_layer, float* o, std::uint32_t valid_tokens,
    std::uint32_t interval, Stream const& stream) {
  if (auto st = require_stream(stream, "gdn_prefill_recurrence"); !st) return st;
  if (!q_hat || !k_hat || !alpha || !beta || !convolved || !s || !o ||
      q_hat == k_hat || s_layer >= kGdnLayers || valid_tokens == 0 ||
      valid_tokens > 1024u || interval == 0 || interval > 1024u)
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "gdn_prefill_recurrence", "invalid operand or count"));
  if (auto st = validate_prefill_spans(stream, {
          {q_hat, static_cast<std::uint64_t>(valid_tokens) * kGdnKeyHeads * kGdnKernelHeadDim * 4u, 4u},
          {k_hat, static_cast<std::uint64_t>(valid_tokens) * kGdnKeyHeads * kGdnKernelHeadDim * 4u, 4u},
          {alpha, static_cast<std::uint64_t>(valid_tokens) * kGdnValueHeads * 4u, 4u},
          {beta, static_cast<std::uint64_t>(valid_tokens) * kGdnValueHeads * 4u, 4u},
          {convolved, static_cast<std::uint64_t>(valid_tokens) * kGdnQkvWidth * 2u, 2u},
          {s, (static_cast<std::uint64_t>(s_layer) + 1u) * kGdnSElemsPerLayer * 4u, 4u},
          {o, static_cast<std::uint64_t>(valid_tokens) * kGdnValueHeads * kGdnKernelHeadDim * 4u, 4u}});
      !st) return st;
  auto guard = stream.activate();
  if (!guard) return std::unexpected(guard.error());
  float* layer_s = s + static_cast<std::size_t>(s_layer) * kGdnSElemsPerLayer;
  for (std::uint32_t first = 0; first < valid_tokens; first += interval) {
    std::uint32_t const count =
        first + interval < valid_tokens ? interval : valid_tokens - first;
    gdn_prefill_recurrence_kernel<<<kGdnRecurBlocksPerLayer, kGdnRecurThreads,
                                    0, stream.native()>>>(
        q_hat, k_hat, alpha, beta, convolved, layer_s, o, first, count);
    if (auto st = check(cudaGetLastError(), "gdn_prefill_recurrence_kernel"); !st)
      return st;
  }
  return {};
}

std::expected<void, Error> launch_gdn_output_transform(
    float const* o, std::uint16_t const* z, std::uint16_t const* gamma, float eps,
    std::uint16_t* u, Stream const& stream) {
  auto st = require_stream(stream, "gdn_output_transform");
  if (!st) {
    return st;
  }
  if (o == nullptr || z == nullptr || gamma == nullptr || u == nullptr) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "gdn_output_transform",
                                      "null o, z, gamma, or u"));
  }
  if (o == reinterpret_cast<float const*>(u)) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "gdn_output_transform",
                                      "o and u must be distinct"));
  }
  return launch_gdn_gated_rms(o, z, gamma, eps, kGdnValueHeads, u, stream);
}

std::expected<GdnRecurrenceResources, Error> gdn_recurrence_resources() {
  cudaFuncAttributes attr{};
  auto st = check(cudaFuncGetAttributes(&attr, gdn_recurrence_kernel),
                  "cudaFuncGetAttributes(gdn_recurrence_kernel)");
  if (!st) {
    return std::unexpected(st.error());
  }
  int occupancy = 0;
  auto occ = check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &occupancy, gdn_recurrence_kernel, kGdnRecurThreads, 0),
                   "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
  if (!occ) {
    return std::unexpected(occ.error());
  }
  GdnRecurrenceResources r;
  r.registers = attr.numRegs;
  r.shared_bytes = static_cast<std::size_t>(attr.sharedSizeBytes);
  r.local_bytes = static_cast<std::size_t>(attr.localSizeBytes);
  r.const_bytes = static_cast<std::size_t>(attr.constSizeBytes);
  r.max_threads_per_block = attr.maxThreadsPerBlock;
  r.occupancy_blocks_per_sm = occupancy;
  return r;
}

std::expected<GdnRecurrenceResources, Error>
gdn_prefill_recurrence_resources() {
  cudaFuncAttributes attr{};
  auto st = check(cudaFuncGetAttributes(&attr, gdn_prefill_recurrence_kernel),
                  "cudaFuncGetAttributes(gdn_prefill_recurrence_kernel)");
  if (!st) return std::unexpected(st.error());
  int occupancy = 0;
  st = check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, gdn_prefill_recurrence_kernel, kGdnRecurThreads, 0),
      "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
  if (!st) return std::unexpected(st.error());
  return GdnRecurrenceResources{
      .registers = attr.numRegs,
      .shared_bytes = static_cast<std::size_t>(attr.sharedSizeBytes),
      .local_bytes = static_cast<std::size_t>(attr.localSizeBytes),
      .const_bytes = static_cast<std::size_t>(attr.constSizeBytes),
      .max_threads_per_block = attr.maxThreadsPerBlock,
      .occupancy_blocks_per_sm = occupancy};
}

}  // namespace qw38::cuda
