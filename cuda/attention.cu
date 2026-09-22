#include "cuda/attention.hpp"

#include <array>
#include <cstddef>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>

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

__device__ __forceinline__ float sigmoid_fp32(float u) {
  if (u >= 0.0f) {
    float const e = expf(-u);
    return 1.0f / (1.0f + e);
  }
  float const e = expf(u);
  return e / (1.0f + e);
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

// The segment scan deliberately keeps scores and K/V staging in shared memory.
// A block owns one query head and one 256-key segment.  Only one 32-key tile is
// staged at a time; partials are the sole sequence-length-dependent global
// output.
__global__ void attention_scan_kernel(
    std::uint16_t const* q, std::uint16_t const* kv,
    std::uint32_t attn_layer, std::uint64_t capacity,
    std::uint64_t populated, float* partials, std::uint32_t n_segments) {
  __shared__ float q_shared[kAttnHeadDim];
  __shared__ std::uint16_t tile[kAttnSubtileKeys * kAttnHeadDim];
  __shared__ float scores[kAttnSubtileKeys];
  __shared__ float m_shared;
  __shared__ float l_shared;
  __shared__ float num_shared[kAttnHeadDim];
  __shared__ float old_scale;
  __shared__ float tile_scale;

  std::uint32_t const h = blockIdx.x;
  std::uint32_t const segment = blockIdx.y;
  std::uint32_t const tid = threadIdx.x;
  std::size_t const head_stride =
      static_cast<std::size_t>(capacity) * kAttnHeadDim;
  std::size_t const comp_stride =
      static_cast<std::size_t>(kAttnKvHeads) * head_stride;
  std::size_t const layer_stride = 2u * comp_stride;
  std::uint16_t const* layer =
      kv + static_cast<std::size_t>(attn_layer) * layer_stride;
  std::uint16_t const* k_base = layer;
  std::uint16_t const* v_base = layer + comp_stride;
  std::uint32_t const kv_h = h / kAttnGqaGroup;
  std::uint16_t const* q_head = q + static_cast<std::size_t>(h) * kAttnHeadDim;
  std::size_t const kv_head_base = static_cast<std::size_t>(kv_h) * head_stride;

  for (std::uint32_t d = tid; d < kAttnHeadDim; d += kAttnScanThreads) {
    q_shared[d] = bf16_to_fp32(q_head[d]);
    num_shared[d] = 0.0f;
  }
  if (tid == 0) {
    m_shared = -INFINITY;
    l_shared = 0.0f;
  }
  __syncthreads();

  std::uint64_t const segment_begin =
      static_cast<std::uint64_t>(segment) * kAttnSegmentKeys;
  for (std::uint32_t sub = 0; sub < kAttnSegmentKeys;
       sub += kAttnSubtileKeys) {
    std::uint32_t const key_count = kAttnSubtileKeys;
    for (std::uint32_t i = tid; i < key_count * kAttnHeadDim;
         i += kAttnScanThreads) {
      std::uint32_t const local_key = i / kAttnHeadDim;
      std::uint32_t const d = i % kAttnHeadDim;
      std::uint64_t const token = segment_begin + sub + local_key;
      if (token < populated && token < capacity) {
        tile[i] = k_base[kv_head_base + token * kAttnHeadDim + d];
      } else {
        tile[i] = 0;
      }
    }
    __syncthreads();

    // One thread computes each score.  This preserves a simple, explicit
    // FP32 dot-product definition while the 128-thread block performs staging.
    if (tid < kAttnSubtileKeys) {
      std::uint64_t const token = segment_begin + sub + tid;
      if (token < populated && token < capacity) {
        float score = 0.0f;
        for (std::uint32_t d = 0; d < kAttnHeadDim; ++d) {
          score += q_shared[d] * bf16_to_fp32(tile[tid * kAttnHeadDim + d]);
        }
        scores[tid] = score * kAttnScale;
      } else {
        scores[tid] = -INFINITY;
      }
    }
    __syncthreads();

    if (tid == 0) {
      float tile_max = -INFINITY;
      for (std::uint32_t i = 0; i < key_count; ++i) {
        tile_max = fmaxf(tile_max, scores[i]);
      }
      float const m_new = fmaxf(m_shared, tile_max);
      old_scale = (m_shared == -INFINITY)
                      ? 0.0f
                      : expf(m_shared - m_new);
      tile_scale = (tile_max == -INFINITY)
                       ? 0.0f
                       : expf(tile_max - m_new);
      l_shared *= old_scale;
      float tile_sum = 0.0f;
      for (std::uint32_t i = 0; i < key_count; ++i) {
        if (scores[i] != -INFINITY) {
          tile_sum += expf(scores[i] - tile_max);
        }
      }
      l_shared += tile_scale * tile_sum;
      for (std::uint32_t d = 0; d < kAttnHeadDim; ++d) {
        num_shared[d] *= old_scale;
      }
      m_shared = m_new;
    }
    __syncthreads();

    for (std::uint32_t i = tid; i < key_count * kAttnHeadDim;
         i += kAttnScanThreads) {
      std::uint32_t const local_key = i / kAttnHeadDim;
      std::uint32_t const d = i % kAttnHeadDim;
      std::uint64_t const token = segment_begin + sub + local_key;
      if (token < populated && token < capacity) {
        tile[i] = v_base[kv_head_base + token * kAttnHeadDim + d];
      } else {
        tile[i] = 0;
      }
    }
    __syncthreads();
    // There are 256 value coordinates but only 128 threads, so each thread
    // owns two coordinates.  Every numerator lane must be updated; leaving
    // the upper half untouched would silently zero half of attention output.
    for (std::uint32_t d = tid; d < kAttnHeadDim; d += kAttnScanThreads) {
      float add = 0.0f;
      if (m_shared != -INFINITY) {
        for (std::uint32_t i = 0; i < key_count; ++i) {
          if (scores[i] != -INFINITY) {
            add += expf(scores[i] - m_shared) *
                   bf16_to_fp32(tile[i * kAttnHeadDim + d]);
          }
        }
      }
      num_shared[d] += add;
    }
    __syncthreads();
  }

  std::size_t const out =
      (static_cast<std::size_t>(h) * n_segments + segment) *
      static_cast<std::size_t>(kAttnPartialStride);
  for (std::uint32_t d = tid; d < kAttnHeadDim; d += kAttnScanThreads) {
    partials[out + 2u + d] = num_shared[d];
  }
  if (tid == 0) {
    partials[out] = m_shared;
    partials[out + 1u] = l_shared;
  }
}

__global__ void attention_merge_kernel(
    float const* partials, std::uint16_t const* g, std::uint32_t n_segments,
    std::uint16_t* y_out) {
  __shared__ float num[kAttnHeadDim];
  __shared__ float m;
  __shared__ float l;
  __shared__ float old_scale;
  __shared__ float tile_scale;
  std::uint32_t const h = blockIdx.x;
  std::uint32_t const tid = threadIdx.x;
  if (tid == 0) {
    m = -INFINITY;
    l = 0.0f;
  }
  for (std::uint32_t d = tid; d < kAttnHeadDim; d += kAttnMergeThreads) {
    num[d] = 0.0f;
  }
  __syncthreads();
  for (std::uint32_t s = 0; s < n_segments; ++s) {
    std::size_t const off =
        (static_cast<std::size_t>(h) * n_segments + s) *
        static_cast<std::size_t>(kAttnPartialStride);
    if (tid == 0) {
      float const bm = partials[off];
      float const bl = partials[off + 1u];
      float const mn = fmaxf(m, bm);
      old_scale = (m == -INFINITY) ? 0.0f : expf(m - mn);
      tile_scale = (bm == -INFINITY) ? 0.0f : expf(bm - mn);
      l = old_scale * l + tile_scale * bl;
      m = mn;
    }
    __syncthreads();
    for (std::uint32_t d = tid; d < kAttnHeadDim; d += kAttnMergeThreads) {
      num[d] = old_scale * num[d] + tile_scale * partials[off + 2u + d];
    }
    __syncthreads();
  }
  if (tid == 0) {
    for (std::uint32_t d = 0; d < kAttnHeadDim; ++d) {
      float const a = (l > 0.0f) ? num[d] / l : 0.0f;
      float const gate = bf16_to_fp32(g[static_cast<std::size_t>(h) * kAttnHeadDim + d]);
      float const sig = sigmoid_fp32(gate);
      float gated = a;
      gated *= sig;
      y_out[static_cast<std::size_t>(h) * kAttnHeadDim + d] =
          fp32_to_bf16_rne(gated);
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

struct ByteInterval {
  std::string_view name;
  std::uintptr_t begin;
  std::uintptr_t end;
};

template <typename Pointer>
std::expected<ByteInterval, Error> checked_interval(
    Pointer pointer, std::uint64_t bytes, std::string_view name) {
  auto const begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (pointer == nullptr || bytes == 0 ||
      bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
    return std::unexpected(make_error(ErrorCode::Overflow,
                                      "attention_prepare",
                                      "preparation byte interval is invalid"));
  }
  return ByteInterval{
      name, begin, begin + static_cast<std::uintptr_t>(bytes)};
}

bool overlaps(ByteInterval const& a, ByteInterval const& b) noexcept {
  return a.begin < b.end && b.begin < a.end;
}

std::expected<void, Error> require_disjoint_prepare_operands(
    std::uint16_t const* qg, std::uint16_t const* k_raw,
    std::uint16_t const* v_raw, std::uint16_t const* gamma_q,
    std::uint16_t const* gamma_k, float const* inv_freq,
    std::uint16_t* q_out, std::uint16_t* g_out, std::uint16_t* kv,
    std::uint64_t capacity) {
  constexpr std::uint64_t kBf16Bytes = sizeof(std::uint16_t);
  constexpr std::uint64_t kQGBytes =
      static_cast<std::uint64_t>(kAttnQueryHeads) * 2u * kAttnHeadDim *
      kBf16Bytes;
  constexpr std::uint64_t kKvProjectionBytes =
      static_cast<std::uint64_t>(kAttnKvHeads) * kAttnHeadDim * kBf16Bytes;
  constexpr std::uint64_t kPreparedBytes =
      static_cast<std::uint64_t>(kAttnQueryHeads) * kAttnHeadDim *
      kBf16Bytes;
  constexpr std::uint64_t kGammaBytes =
      static_cast<std::uint64_t>(kAttnHeadDim) * kBf16Bytes;
  constexpr std::uint64_t kInvFreqBytes =
      static_cast<std::uint64_t>(kAttnRopeFreqs) * sizeof(float);
  constexpr std::uint64_t kKvBytesPerToken =
      static_cast<std::uint64_t>(kAttnLayers) * 2u * kAttnKvHeads *
      kAttnHeadDim * kBf16Bytes;
  if (capacity > std::numeric_limits<std::uint64_t>::max() /
                     kKvBytesPerToken) {
    return std::unexpected(make_error(ErrorCode::Overflow,
                                      "attention_prepare",
                                      "cache byte count overflows"));
  }
  auto const kv_bytes = capacity * kKvBytesPerToken;

  std::array<std::expected<ByteInterval, Error>, 9> checked{
      checked_interval(qg, kQGBytes, "qg"),
      checked_interval(k_raw, kKvProjectionBytes, "k_raw"),
      checked_interval(v_raw, kKvProjectionBytes, "v_raw"),
      checked_interval(gamma_q, kGammaBytes, "gamma_q"),
      checked_interval(gamma_k, kGammaBytes, "gamma_k"),
      checked_interval(inv_freq, kInvFreqBytes, "inv_freq"),
      checked_interval(q_out, kPreparedBytes, "q_out"),
      checked_interval(g_out, kPreparedBytes, "g_out"),
      checked_interval(kv, kv_bytes, "kv")};
  std::array<ByteInterval, 9> intervals{};
  for (std::size_t i = 0; i < checked.size(); ++i) {
    if (!checked[i]) return std::unexpected(checked[i].error());
    intervals[i] = *checked[i];
  }
  // The preparation kernel has no authorized aliases: input projection
  // values and parameters remain live while Q, g, and the cache are written.
  for (std::size_t i = 0; i < intervals.size(); ++i) {
    for (std::size_t j = i + 1; j < intervals.size(); ++j) {
      if (overlaps(intervals[i], intervals[j])) {
        std::string detail(intervals[i].name);
        detail += " overlaps ";
        detail += intervals[j].name;
        return std::unexpected(make_error(
            ErrorCode::InvalidArgument, "attention_prepare",
            detail));
      }
    }
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
  if (auto alias = require_disjoint_prepare_operands(
          qg, k_raw, v_raw, gamma_q, gamma_k, inv_freq, q_out, g_out, kv,
          capacity);
      !alias) {
    return alias;
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

std::expected<void, Error> launch_attention_scan(
    std::uint16_t const* q, std::uint16_t const* kv, std::uint32_t attn_layer,
    std::uint64_t capacity, std::uint64_t populated, float* partials,
    std::uint32_t n_segments, Stream const& stream) {
  auto st = require_stream(stream, "attention_scan");
  if (!st) return st;
  if (q == nullptr || kv == nullptr || partials == nullptr) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "attention_scan",
                                       "null scan operand"));
  }
  if (attn_layer >= kAttnLayers) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "attention_scan",
                                       "attn_layer must be < 16"));
  }
  if (capacity == 0 || populated > capacity) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "attention_scan",
                                       "populated length exceeds capacity"));
  }
  if (n_segments == 0) return {};
  auto const expected = static_cast<std::uint64_t>(n_segments) * kAttnSegmentKeys;
  if (populated > expected) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "attention_scan",
                                       "n_segments is smaller than populated length"));
  }
  attention_scan_kernel<<<dim3(kAttnQueryHeads, n_segments, 1), kAttnScanThreads, 0,
                          stream.native()>>>(q, kv, attn_layer, capacity, populated,
                                              partials, n_segments);
  return check(cudaGetLastError(), "attention_scan_kernel");
}

std::expected<void, Error> launch_attention_merge(
    float const* partials, std::uint16_t const* g, std::uint32_t n_segments,
    std::uint16_t* y_out, Stream const& stream) {
  auto st = require_stream(stream, "attention_merge");
  if (!st) return st;
  if (partials == nullptr || g == nullptr || y_out == nullptr) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "attention_merge",
                                       "null merge operand"));
  }
  if (n_segments == 0) {
    // There is no causal key; launch the same deterministic merge path with a
    // zero segment count, which emits zero gated output.
  }
  attention_merge_kernel<<<kAttnQueryHeads, kAttnMergeThreads, 0, stream.native()>>>(
      partials, g, n_segments, y_out);
  return check(cudaGetLastError(), "attention_merge_kernel");
}

}  // namespace qw38::cuda
