#include "cuda/attention.hpp"
#include "cuda/fp8_device.cuh"
#include "cuda/prefill.hpp"

#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/copy_sm80.hpp>

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
    std::uint64_t token, DecodeControl const* control = nullptr) {
  if (control) { position = static_cast<float>(control->position); token = control->position; }
  std::size_t const row = blockIdx.y;
  qg += row * kAttnQueryHeads * 2u * kAttnHeadDim;
  k_raw += row * kAttnKvHeads * kAttnHeadDim;
  v_raw += row * kAttnKvHeads * kAttnHeadDim;
  q_out += row * kAttnQueryHeads * kAttnHeadDim;
  g_out += row * kAttnQueryHeads * kAttnHeadDim;
  position += static_cast<float>(row);
  token += row;
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

// Cooperatively stage split statistics, then combine each coordinate in a
// deterministic partition order. No barrier is needed inside that order.
__global__ void attention_merge_kernel(
    float const* partials, std::uint16_t const* g, std::uint32_t n_segments,
    std::uint16_t* y_out, DecodeControl const* control, std::uint32_t partition_limit) {
  if (control) n_segments = min(static_cast<std::uint32_t>((control->populated + 255) / 256), partition_limit);
  __shared__ float maxima[kAttnMaxPartitions], sums[kAttnMaxPartitions];
  __shared__ float weights[kAttnMaxPartitions], denominator;
  unsigned const tid = threadIdx.x, h = blockIdx.x;
  auto const* head = partials + std::size_t(h) * n_segments * kAttnPartialStride;
  if (tid < n_segments) {
    maxima[tid] = head[tid * kAttnPartialStride];
    sums[tid] = head[tid * kAttnPartialStride + 1];
  }
  __syncthreads();
  float best = -INFINITY;
  for (unsigned s = 0; s < n_segments; ++s) best = fmaxf(best, maxima[s]);
  if (tid < n_segments)
    weights[tid] = maxima[tid] == -INFINITY ? 0.f : expf(maxima[tid] - best);
  __syncthreads();
  if (tid == 0) {
    float total = 0;
    for (unsigned s = 0; s < n_segments; ++s) total += weights[s] * sums[s];
    denominator = total;
  }
  __syncthreads();
  for (unsigned d = tid; d < kAttnHeadDim; d += kAttnMergeThreads) {
    float num = 0;
    for (unsigned s = 0; s < n_segments; ++s)
      num += weights[s] * head[s * kAttnPartialStride + 2 + d];
    auto const off = std::size_t(h) * kAttnHeadDim + d;
    y_out[off] = fp32_to_bf16_rne((denominator > 0 ? num / denominator : 0.f) *
        sigmoid_fp32(bf16_to_fp32(g[off])));
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
    std::uint64_t capacity, std::uint32_t rows = 1) {
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
      checked_interval(qg, kQGBytes * rows, "qg"),
      checked_interval(k_raw, kKvProjectionBytes * rows, "k_raw"),
      checked_interval(v_raw, kKvProjectionBytes * rows, "v_raw"),
      checked_interval(gamma_q, kGammaBytes, "gamma_q"),
      checked_interval(gamma_k, kGammaBytes, "gamma_k"),
      checked_interval(inv_freq, kInvFreqBytes, "inv_freq"),
      checked_interval(q_out, kPreparedBytes * rows, "q_out"),
      checked_interval(g_out, kPreparedBytes * rows, "g_out"),
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

template <std::uint32_t Q>
__global__ void attention_prefill_scan_kernel(
    std::uint16_t const* q, std::uint16_t const* g,
    std::uint16_t const* kv, std::uint32_t attn_layer,
    std::uint64_t capacity, std::uint64_t first_position,
    std::uint32_t valid_tokens, std::uint16_t* y) {
  constexpr std::uint32_t K = kAttnPrefillKeyTileControl;
  __shared__ float qs[Q * kAttnHeadDim];
  __shared__ float num[Q * kAttnHeadDim];
  __shared__ std::uint16_t tile[K * kAttnHeadDim];
  __shared__ float scores[Q * K];
  __shared__ float m[Q], l[Q], old_scale[Q];
  std::uint32_t const h = blockIdx.x;
  std::uint32_t const row0 = blockIdx.y * Q;
  std::uint32_t const tid = threadIdx.x;
  std::size_t const head_stride = static_cast<std::size_t>(capacity) * kAttnHeadDim;
  std::size_t const comp_stride = kAttnKvHeads * head_stride;
  std::uint16_t const* layer = kv + static_cast<std::size_t>(attn_layer) *
      2u * comp_stride;
  std::uint16_t const* k_base = layer + (h / kAttnGqaGroup) * head_stride;
  std::uint16_t const* v_base = k_base + comp_stride;
  for (std::uint32_t i = tid; i < Q * kAttnHeadDim; i += blockDim.x) {
    std::uint32_t const r = i / kAttnHeadDim;
    std::uint32_t const d = i % kAttnHeadDim;
    qs[i] = row0 + r < valid_tokens
        ? bf16_to_fp32(q[(static_cast<std::size_t>(row0 + r) * kAttnQueryHeads + h) *
                         kAttnHeadDim + d]) : 0.0f;
    num[i] = 0.0f;
  }
  if (tid < Q) { m[tid] = -INFINITY; l[tid] = 0.0f; }
  __syncthreads();
  std::uint64_t const last = first_position +
      min(valid_tokens, row0 + Q);
  for (std::uint64_t base = 0; base < last; base += K) {
    for (std::uint32_t i = tid; i < K * kAttnHeadDim; i += blockDim.x) {
      std::uint64_t const token = base + i / kAttnHeadDim;
      tile[i] = token < last
          ? k_base[token * kAttnHeadDim + i % kAttnHeadDim] : 0;
    }
    __syncthreads();
    if (tid < Q * K) {
      std::uint32_t const r = tid / K, key = tid % K;
      std::uint64_t const token = base + key;
      if (row0 + r < valid_tokens && token <= first_position + row0 + r) {
        float dot = 0.0f;
        for (std::uint32_t d = 0; d < kAttnHeadDim; ++d)
          dot += qs[r * kAttnHeadDim + d] *
                 bf16_to_fp32(tile[key * kAttnHeadDim + d]);
        scores[tid] = dot * kAttnScale;
      } else {
        scores[tid] = -INFINITY;
      }
    }
    __syncthreads();
    if (tid < Q) {
      float best = m[tid];
      for (std::uint32_t key = 0; key < K; ++key)
        best = fmaxf(best, scores[tid * K + key]);
      old_scale[tid] = m[tid] == -INFINITY ? 0.0f : expf(m[tid] - best);
      if (best != -INFINITY) {
        l[tid] *= old_scale[tid];
        for (std::uint32_t key = 0; key < K; ++key) {
          float const score = scores[tid * K + key];
          float const p = score == -INFINITY ? 0.0f : expf(score - best);
          scores[tid * K + key] = p;
          l[tid] += p;
        }
        m[tid] = best;
      }
    }
    __syncthreads();
    for (std::uint32_t i = tid; i < K * kAttnHeadDim; i += blockDim.x) {
      std::uint64_t const token = base + i / kAttnHeadDim;
      tile[i] = token < last
          ? v_base[token * kAttnHeadDim + i % kAttnHeadDim] : 0;
    }
    __syncthreads();
    for (std::uint32_t i = tid; i < Q * kAttnHeadDim; i += blockDim.x) {
      std::uint32_t const r = i / kAttnHeadDim, d = i % kAttnHeadDim;
      float acc = num[i] * old_scale[r];
      for (std::uint32_t key = 0; key < K; ++key)
        acc += scores[r * K + key] *
               bf16_to_fp32(tile[key * kAttnHeadDim + d]);
      num[i] = acc;
    }
    __syncthreads();
  }
  for (std::uint32_t i = tid; i < Q * kAttnHeadDim; i += blockDim.x) {
    std::uint32_t const r = i / kAttnHeadDim, d = i % kAttnHeadDim;
    if (row0 + r < valid_tokens) {
      std::size_t const off = (static_cast<std::size_t>(row0 + r) *
          kAttnQueryHeads + h) * kAttnHeadDim + d;
      float const value = num[i] / l[r] * sigmoid_fp32(bf16_to_fp32(g[off]));
      y[off] = fp32_to_bf16_rne(value);
    }
  }
}

constexpr int kPrefillThreads = 128;
constexpr int kKvTileStride = 264; // Eight BF16 padding elements avoid bank aliasing.
constexpr std::size_t kPrefillSharedBytes = 2u * 64 * kKvTileStride * 2u;

__device__ __forceinline__ unsigned bf16_pair(unsigned short a, unsigned short b) {
  return unsigned(a) | (unsigned(b) << 16);
}

__device__ __forceinline__ void attention_mma(float (&c)[4],
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  cute::SM80_16x8x16_F32BF16BF16F32_TN::fma(c[0], c[1], c[2], c[3],
      a0, a1, a2, a3, b0, b1, c[0], c[1], c[2], c[3]);
}

__device__ void stage_attention_kv(std::uint16_t const* source,
    std::uint16_t* tile, std::uint64_t base, std::uint64_t end) {
  for (int i = threadIdx.x; i < 64 * 32; i += kPrefillThreads) {
    auto const token = base + i / 32;
    bool const valid = token < end;
    // Even a zero-fill copy uses an in-bounds source address.
    auto const* src = reinterpret_cast<uint4 const*>(source +
        (valid ? token * 256 + (i % 32) * 8 : 0));
    auto* dst = reinterpret_cast<uint4*>(tile + (i / 32) * kKvTileStride + (i % 32) * 8);
    cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint4>::copy(*src, *dst, valid);
  }
  cute::cp_async_fence();
}

// KQ is [key,query], PV is [value,query]. Each m16n8 accumulator lane owns
// rows lane/4 and lane/4+8, columns 2*(lane%4) and 2*(lane%4)+1.
// The eight-column register layout, separate K/V pipeline and grouped heads
// follow pinned llama.cpp fattn-mma-f16.cuh (see third_party attribution).
// Both probability components and all statistics/accumulators remain FP32/BF16.
template <bool Decode>
__global__ void attention_mma_kernel(
    std::uint16_t const* q, std::uint16_t const* g,
    std::uint16_t const* kv, std::uint32_t attn_layer,
    std::uint64_t capacity, std::uint64_t first_position,
    std::uint32_t valid_tokens, std::uint16_t* y,
    std::uint8_t* fp8_codes, float* fp8_scales,
    float* partials = nullptr, std::uint32_t partitions = 0,
    DecodeControl const* control = nullptr, std::uint32_t partition_limit = 0) {
  if constexpr (Decode) {
    if (control) {
      first_position = control->populated;
      partitions = min(static_cast<std::uint32_t>((first_position + 255) / 256), partition_limit);
      if (blockIdx.y >= partitions) return;
    }
  }
  constexpr int D = 256, K = 64, Q = 32;
  constexpr int ValueTiles = Decode ? 4 : 16;
  extern __shared__ __align__(16) unsigned char storage[];
  auto* ks = reinterpret_cast<std::uint16_t*>(storage);
  auto* vs = ks + K * kKvTileStride;
  int const warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  unsigned const h = blockIdx.x, row0 = Decode ? 0 : blockIdx.y * Q;
  unsigned const pm = (valid_tokens + 127) / 128 * 128;
  if constexpr (!Decode) {
    if (fp8_codes && row0 >= valid_tokens) {
      for (unsigned r = 0; r < Q; ++r)
        fp8_pack_row(nullptr, row0 + r, pm, D, fp8_codes, fp8_scales, h * 2, 6144);
      return;
    }
  }
  std::size_t const head_stride = capacity * D;
  std::size_t const comp_stride = kAttnKvHeads * head_stride;
  auto const* kb = kv + attn_layer * 2u * comp_stride +
      (Decode ? h : h / kAttnGqaGroup) * head_stride;
  auto const* vb = kb + comp_stride;
  // Decode's four warps share six Q heads and divide the output coordinates.
  // Prefill's four warps each own eight query rows and all 256 coordinates.
  unsigned const qcol = Decode ? lane / 4 : row0 + warp * 8 + lane / 4;
  bool const qvalid = Decode ? qcol < 6 : qcol < valid_tokens;
  std::size_t const qoff = (Decode ? h * 6 + qcol : qcol * kAttnQueryHeads + h) * D;
  unsigned qr[16][2];
#pragma unroll
  for (int d = 0; d < 16; ++d) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      auto const off = qoff + d * 16 + (lane % 4) * 2 + j * 8;
      qr[d][j] = qvalid ? bf16_pair(q[off], q[off + 1]) : 0;
    }
  }
  float pv[ValueTiles][4]{};
  float maximum[2] = {-INFINITY, -INFINITY}, denominator[2]{};
  std::uint64_t begin = 0;
  std::uint64_t end = first_position + min(valid_tokens, row0 + Q);
  if constexpr (Decode) {
    // Partition the active tiles, independently of the allocated KV capacity.
    // Fixed launch buckets may contain empty partitions; those store neutral data.
    auto const tiles = (first_position + K - 1) / K;
    begin = (tiles * blockIdx.y / partitions) * K;
    end = min(first_position, (tiles * (blockIdx.y + 1) / partitions) * K);
  }
  stage_attention_kv(kb, ks, begin, end);
  cute::cp_async_wait<0>();
  __syncthreads();
  for (std::uint64_t base = begin; base < end; base += K) {
    stage_attention_kv(vb, vs, base, end); // Overlap V with QK.
    float score[4][4]{};
#pragma unroll
    for (int d = 0; d < 16; ++d) {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        int const r = t * 16 + lane / 4, c = d * 16 + (lane % 4) * 2;
        attention_mma(score[t],
            bf16_pair(ks[r * kKvTileStride + c], ks[r * kKvTileStride + c + 1]),
            bf16_pair(ks[(r + 8) * kKvTileStride + c], ks[(r + 8) * kKvTileStride + c + 1]),
            bf16_pair(ks[r * kKvTileStride + c + 8], ks[r * kKvTileStride + c + 9]),
            bf16_pair(ks[(r + 8) * kKvTileStride + c + 8], ks[(r + 8) * kKvTileStride + c + 9]),
            qr[d][0], qr[d][1]);
      }
    }
    __syncthreads(); // All K readers finish before the next K tile arrives.
    stage_attention_kv(kb, ks, base + K, end); // Overlap next K with softmax/PV.
    float best[2] = {maximum[0], maximum[1]};
#pragma unroll
    for (int t = 0; t < 4; ++t) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        unsigned const col = (Decode ? 0 : row0 + warp * 8) + (lane % 4) * 2 + i % 2;
        auto const key = base + t * 16 + lane / 4 + (i / 2) * 8;
        bool const visible = Decode ? col < 6 && key < end
            : col < valid_tokens && key < end && key <= first_position + col;
        score[t][i] = visible ? score[t][i] * kAttnScale : -INFINITY;
        best[i % 2] = fmaxf(best[i % 2], score[t][i]);
      }
    }
#pragma unroll
    for (int c = 0; c < 2; ++c) {
#pragma unroll
      for (int off = 16; off >= 4; off /= 2)
        best[c] = fmaxf(best[c], __shfl_xor_sync(0xffffffff, best[c], off));
      float const scale = maximum[c] == -INFINITY ? 0.f : expf(maximum[c] - best[c]);
      float sum = 0;
#pragma unroll
      for (int t = 0; t < 4; ++t) {
#pragma unroll
        for (int i = c; i < 4; i += 2) {
          score[t][i] = score[t][i] == -INFINITY ? 0.f : expf(score[t][i] - best[c]);
          sum += score[t][i];
        }
      }
#pragma unroll
      for (int off = 16; off >= 4; off /= 2) sum += __shfl_xor_sync(0xffffffff, sum, off);
      denominator[c] = denominator[c] * scale + sum;
      maximum[c] = best[c];
#pragma unroll
      for (int d = 0; d < ValueTiles; ++d) {
        pv[d][c] *= scale;
        pv[d][c + 2] *= scale;
      }
    }
    cute::cp_async_wait<1>(); // V is ready; the next K group may remain in flight.
    __syncthreads();
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      unsigned high[2], low[2];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        int const src = (lane % 4) * 8 + (lane / 4) / 2;
        float const a = __shfl_sync(0xffffffff, score[t][j * 2], src);
        float const b = __shfl_sync(0xffffffff, score[t][j * 2 + 1], src);
        float const c = __shfl_sync(0xffffffff, score[t][j * 2], src + 4);
        float const d = __shfl_sync(0xffffffff, score[t][j * 2 + 1], src + 4);
        float const p0 = (lane / 4) % 2 ? b : a;
        float const p1 = (lane / 4) % 2 ? d : c;
        auto const hi0 = fp32_to_bf16_rne(p0), hi1 = fp32_to_bf16_rne(p1);
        high[j] = bf16_pair(hi0, hi1);
        low[j] = bf16_pair(fp32_to_bf16_rne(p0 - bf16_to_fp32(hi0)),
                           fp32_to_bf16_rne(p1 - bf16_to_fp32(hi1)));
      }
#pragma unroll
      for (int d = 0; d < ValueTiles; ++d) {
        int const r = (Decode ? warp * 64 : 0) + d * 16 + lane / 4;
        int const c = t * 16 + (lane % 4) * 2;
        unsigned const a0 = bf16_pair(vs[c * kKvTileStride + r], vs[(c + 1) * kKvTileStride + r]);
        unsigned const a1 = bf16_pair(vs[c * kKvTileStride + r + 8], vs[(c + 1) * kKvTileStride + r + 8]);
        unsigned const a2 = bf16_pair(vs[(c + 8) * kKvTileStride + r], vs[(c + 9) * kKvTileStride + r]);
        unsigned const a3 = bf16_pair(vs[(c + 8) * kKvTileStride + r + 8], vs[(c + 9) * kKvTileStride + r + 8]);
        attention_mma(pv[d], a0, a1, a2, a3, high[0], high[1]);
        attention_mma(pv[d], a0, a1, a2, a3, low[0], low[1]);
      }
    }
    cute::cp_async_wait<0>();
    __syncthreads(); // Next K ready; all V readers done before V is reused.
  }
#pragma unroll
  for (int d = 0; d < ValueTiles; ++d) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      unsigned const col = (Decode ? 0 : row0 + warp * 8) + (lane % 4) * 2 + i % 2;
      unsigned const coord = (Decode ? warp * 64 : 0) + d * 16 + lane / 4 + (i / 2) * 8;
      if constexpr (Decode) {
        if (col < 6) {
          auto const off = ((h * 6 + col) * std::size_t(partitions) + blockIdx.y) * kAttnPartialStride;
          partials[off + 2 + coord] = pv[d][i];
          if (coord == 0) {
            partials[off] = maximum[i % 2];
            partials[off + 1] = denominator[i % 2];
          }
        }
      } else {
        std::uint16_t rounded = 0;
        if (col < valid_tokens) {
          auto const off = (std::size_t(col) * kAttnQueryHeads + h) * D + coord;
          rounded = fp32_to_bf16_rne(pv[d][i] / denominator[i % 2] * sigmoid_fp32(bf16_to_fp32(g[off])));
          if (!fp8_codes) y[off] = rounded;
        }
        if (fp8_codes) ks[(col - row0) * D + coord] = rounded;
      }
    }
  }
  if constexpr (!Decode) {
    if (fp8_codes) {
      __syncthreads();
      for (unsigned r = 0; r < Q; ++r)
        fp8_pack_row(ks + r * D, row0 + r, pm, D, fp8_codes, fp8_scales, h * 2, 6144);
    }
  }
}

template <bool Decode = false>
std::expected<void, Error> configure_prefill_mma() {
  return check(cudaFuncSetAttribute(attention_mma_kernel<Decode>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kPrefillSharedBytes),
      "cudaFuncSetAttribute(attention_mma_kernel)");
}

}  // namespace

std::expected<void, Error> launch_attention_prepare(
    std::uint16_t const* qg, std::uint16_t const* k_raw,
    std::uint16_t const* v_raw, std::uint16_t const* gamma_q,
    std::uint16_t const* gamma_k, float const* inv_freq, float eps,
    std::int32_t position, std::uint16_t* q_out, std::uint16_t* g_out,
    std::uint16_t* kv, std::uint32_t attn_layer, std::uint64_t capacity,
    std::uint64_t token, Stream const& stream, DecodeControl const* control) {
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
  auto guard = stream.activate();
  if (!guard) return std::unexpected(guard.error());
  attention_prepare_kernel<<<kAttnPrepBlocks, kAttnPrepThreads, 0,
                             stream.native()>>>(
      qg, k_raw, v_raw, gamma_q, gamma_k, inv_freq, eps, pos, q_out, g_out, kv,
      attn_layer, capacity, token, control);
  return check(cudaGetLastError(), "attention_prepare_kernel");
}

std::expected<void, Error> launch_attention_prepare_chunk(
    std::uint16_t const* qg, std::uint16_t const* k_raw,
    std::uint16_t const* v_raw, std::uint16_t const* gamma_q,
    std::uint16_t const* gamma_k, float const* inv_freq, float eps,
    std::uint64_t first_position, std::uint32_t valid_tokens,
    std::uint16_t* q_out, std::uint16_t* g_out, std::uint16_t* kv,
    std::uint32_t attn_layer, std::uint64_t capacity, Stream const& stream) {
  if (auto st = require_stream(stream, "attention_prepare_chunk"); !st) return st;
  if (!qg || !k_raw || !v_raw || !gamma_q || !gamma_k || !inv_freq ||
      !q_out || !g_out || !kv || attn_layer >= kAttnLayers ||
      !valid_tokens || valid_tokens > 1024u ||
      first_position >= capacity || valid_tokens > capacity - first_position ||
      first_position + valid_tokens - 1u >
          static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
      !finite_pos(eps))
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "attention_prepare_chunk", "invalid chunk geometry or operand"));
  if (auto st = require_disjoint_prepare_operands(
          qg, k_raw, v_raw, gamma_q, gamma_k, inv_freq,
          q_out, g_out, kv, capacity, valid_tokens); !st) return st;
  auto const kv_bytes = static_cast<std::uint64_t>(kAttnLayers) * 2u *
      kAttnKvHeads * capacity * kAttnHeadDim * sizeof(std::uint16_t);
  struct Span { void const* ptr; std::uint64_t bytes; std::uint32_t alignment; };
  std::array<Span, 9> const spans{{
      {qg, static_cast<std::uint64_t>(valid_tokens) * kAttnQueryHeads * 2u *
               kAttnHeadDim * 2u, 2u},
      {k_raw, static_cast<std::uint64_t>(valid_tokens) * kAttnKvHeads *
                  kAttnHeadDim * 2u, 2u},
      {v_raw, static_cast<std::uint64_t>(valid_tokens) * kAttnKvHeads *
                  kAttnHeadDim * 2u, 2u},
      {gamma_q, kAttnHeadDim * 2u, 2u}, {gamma_k, kAttnHeadDim * 2u, 2u},
      {inv_freq, kAttnRopeFreqs * 4u, 4u},
      {q_out, static_cast<std::uint64_t>(valid_tokens) * kAttnQueryHeads *
                  kAttnHeadDim * 2u, 2u},
      {g_out, static_cast<std::uint64_t>(valid_tokens) * kAttnQueryHeads *
                  kAttnHeadDim * 2u, 2u},
      {kv, kv_bytes, 2u}}};
  for (auto const& span : spans)
    if (auto st = validate_prefill_device_span(
            span.ptr, span.bytes, span.alignment, stream.device()); !st) return st;
  auto guard = stream.activate();
  if (!guard) return std::unexpected(guard.error());
  dim3 const grid(kAttnPrepBlocks, valid_tokens);
  attention_prepare_kernel<<<grid, kAttnPrepThreads, 0, stream.native()>>>(
      qg, k_raw, v_raw, gamma_q, gamma_k, inv_freq, eps,
      static_cast<float>(first_position), q_out, g_out, kv,
      attn_layer, capacity, first_position);
  return check(cudaGetLastError(), "attention_prepare_chunk_kernel");
}

std::expected<void, Error> launch_attention_prefill_scan(
    std::uint16_t const* q, std::uint16_t const* g,
    std::uint16_t const* kv, std::uint32_t attn_layer,
    std::uint64_t capacity, std::uint64_t first_position,
    std::uint32_t valid_tokens, std::uint16_t* y,
    Stream const& stream, std::uint32_t query_tile, std::uint8_t* fp8_codes, float* fp8_scales) {
  if (auto st = require_stream(stream, "attention_prefill_scan"); !st) return st;
  if (!q || !g || !kv || !y || reinterpret_cast<std::uintptr_t>(kv) % 16 != 0 ||
      attn_layer >= kAttnLayers ||
      !valid_tokens || valid_tokens > 1024u ||
      first_position >= capacity || valid_tokens > capacity - first_position ||
      (query_tile != kAttnPrefillQueryTile &&
       query_tile != kAttnPrefillQueryTileScalar &&
       query_tile != kAttnPrefillQueryTileControl))
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "attention_prefill_scan", "invalid chunk geometry or operand"));
  auto const prepared_bytes = static_cast<std::uint64_t>(valid_tokens) *
      kAttnQueryHeads * kAttnHeadDim * 2u;
  constexpr std::uint64_t kv_bytes_per_token =
      static_cast<std::uint64_t>(kAttnLayers) * 2u * kAttnKvHeads *
      kAttnHeadDim * 2u;
  if (capacity > std::numeric_limits<std::uint64_t>::max() / kv_bytes_per_token)
    return std::unexpected(make_error(ErrorCode::Overflow,
                                      "attention_prefill_scan", "cache byte count overflows"));
  auto const kv_bytes = static_cast<std::uint64_t>(kAttnLayers) * 2u *
      kAttnKvHeads * capacity * kAttnHeadDim * 2u;
  std::array<std::expected<ByteInterval, Error>, 4> checked{
      checked_interval(q, prepared_bytes, "q"),
      checked_interval(g, prepared_bytes, "g"),
      checked_interval(kv, kv_bytes, "kv"),
      checked_interval(y, prepared_bytes, "y")};
  std::array<ByteInterval, 4> ranges{};
  for (std::size_t i = 0; i < checked.size(); ++i) {
    if (!checked[i]) return std::unexpected(checked[i].error());
    ranges[i] = *checked[i];
    if (auto st = validate_prefill_device_span(
            reinterpret_cast<void const*>(ranges[i].begin),
            ranges[i].end - ranges[i].begin, 2u, stream.device()); !st) return st;
  }
  for (std::size_t i = 0; i < ranges.size(); ++i)
    for (std::size_t j = i + 1; j < ranges.size(); ++j)
      if (overlaps(ranges[i], ranges[j]))
        return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                          "attention_prefill_scan", "live operands overlap"));
  auto guard = stream.activate();
  if (!guard) return std::unexpected(guard.error());
  unsigned rows=valid_tokens;
  if(fp8_codes || fp8_scales) {
    if(!fp8_codes || !fp8_scales || query_tile!=kAttnPrefillQueryTile)
      return std::unexpected(make_error(ErrorCode::InvalidArgument,"fp8.attention","invalid packed output"));
    rows=(valid_tokens+127)/128*128;
    if(auto st=validate_prefill_device_span(fp8_codes,std::uint64_t(rows)*6144,16,stream.device());!st)return st;
    if(auto st=validate_prefill_device_span(fp8_scales,std::uint64_t(rows)*48*4,16,stream.device());!st)return st;
    auto c=checked_interval(fp8_codes,std::uint64_t(rows)*6144,"fp8.codes");
    auto s=checked_interval(fp8_scales,std::uint64_t(rows)*48*4,"fp8.scales");
    if(!c || !s || overlaps(*c,*s))return std::unexpected(make_error(ErrorCode::InvalidArgument,"fp8.attention","pack overlap"));
    for(auto r:ranges)if(overlaps(r,*c) || overlaps(r,*s))
      return std::unexpected(make_error(ErrorCode::InvalidArgument,"fp8.attention","pack aliases live operand"));
  }
  dim3 const grid(kAttnQueryHeads,(rows + query_tile - 1u) / query_tile);
  if (query_tile == kAttnPrefillQueryTile) {
    if (auto st = configure_prefill_mma(); !st) return st;
    attention_mma_kernel<false>
        <<<grid, kPrefillThreads, kPrefillSharedBytes, stream.native()>>>(
            q, g, kv, attn_layer, capacity, first_position, valid_tokens, y, fp8_codes, fp8_scales);
  } else if (query_tile == kAttnPrefillQueryTileScalar)
    attention_prefill_scan_kernel<kAttnPrefillQueryTileScalar>
        <<<grid, kAttnScanThreads, 0, stream.native()>>>(
            q, g, kv, attn_layer, capacity, first_position, valid_tokens, y);
  else
    attention_prefill_scan_kernel<kAttnPrefillQueryTileControl>
        <<<grid, kAttnScanThreads, 0, stream.native()>>>(
            q, g, kv, attn_layer, capacity, first_position, valid_tokens, y);
  return check(cudaGetLastError(), "attention_prefill_scan_kernel");
}

std::expected<AttentionPrefillResources, Error> attention_prefill_resources(
    std::uint32_t query_tile) {
  if (query_tile != kAttnPrefillQueryTile &&
      query_tile != kAttnPrefillQueryTileScalar &&
      query_tile != kAttnPrefillQueryTileControl)
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "attention_prefill_resources", "invalid query tile"));
  void const* kernel = query_tile == kAttnPrefillQueryTile
      ? reinterpret_cast<void const*>(attention_mma_kernel<false>)
      : query_tile == kAttnPrefillQueryTileScalar
          ? reinterpret_cast<void const*>(attention_prefill_scan_kernel<kAttnPrefillQueryTileScalar>)
          : reinterpret_cast<void const*>(attention_prefill_scan_kernel<kAttnPrefillQueryTileControl>);
  bool const mma = query_tile == kAttnPrefillQueryTile;
  if (mma) {
    if (auto st = configure_prefill_mma(); !st) return std::unexpected(st.error());
  }
  std::size_t const dynamic_bytes = mma ? kPrefillSharedBytes : 0;
  cudaFuncAttributes attr{};
  auto st = check(cudaFuncGetAttributes(&attr, kernel),
                  "cudaFuncGetAttributes(attention_prefill_scan_kernel)");
  if (!st) return std::unexpected(st.error());
  int occupancy = 0;
  st = check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, kernel, mma ? kPrefillThreads : kAttnScanThreads, dynamic_bytes),
      "cudaOccupancyMaxActiveBlocksPerMultiprocessor(attention_prefill_scan_kernel)");
  if (!st) return std::unexpected(st.error());
  return AttentionPrefillResources{attr.numRegs,
      static_cast<std::size_t>(attr.sharedSizeBytes) + dynamic_bytes,
      static_cast<std::size_t>(attr.localSizeBytes), occupancy};
}

std::expected<AttentionPrefillResources, Error> attention_decode_resources() {
  if (auto st = configure_prefill_mma<true>(); !st) return std::unexpected(st.error());
  cudaFuncAttributes attr{};
  int occupancy = 0;
  if (auto st = check(cudaFuncGetAttributes(&attr, attention_mma_kernel<true>),
          "attention decode attributes"); !st) return std::unexpected(st.error());
  if (auto st = check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,
          attention_mma_kernel<true>, kPrefillThreads, kPrefillSharedBytes),
          "attention decode occupancy"); !st) return std::unexpected(st.error());
  return AttentionPrefillResources{attr.numRegs, attr.sharedSizeBytes + kPrefillSharedBytes,
      attr.localSizeBytes, occupancy};
}

std::expected<void, Error> launch_attention_scan(
    std::uint16_t const* q, std::uint16_t const* kv, std::uint32_t attn_layer,
    std::uint64_t capacity, std::uint64_t populated, float* partials,
    std::uint32_t n_segments, Stream const& stream, DecodeControl const* control,
    std::uint32_t partition_limit) {
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
  if (n_segments > kAttnMaxPartitions || (n_segments == 0 && populated != 0)) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "attention_scan",
                                       "invalid bounded partition count"));
  }
  if (n_segments == 0) return {};
  constexpr auto kv_bytes_per_token = 16ull * 2 * kAttnKvHeads * kAttnHeadDim * 2;
  if (capacity > std::numeric_limits<std::size_t>::max() / kv_bytes_per_token)
    return std::unexpected(make_error(ErrorCode::Overflow, "attention_scan", "KV bytes overflow"));
  if (!control) {
  if (auto st = validate_prefill_device_span(kv, capacity * kv_bytes_per_token, 16, stream.device()); !st) return st;
  if (auto st = validate_prefill_device_span(q, kAttnQueryHeads * kAttnHeadDim * 2u, 2, stream.device()); !st) return st;
  if (auto st = validate_prefill_device_span(partials,
      std::size_t(kAttnQueryHeads) * n_segments * kAttnPartialStride * 4u, 4, stream.device()); !st) return st;
  }
  auto guard = stream.activate();
  if (!guard) return std::unexpected(guard.error());
  if (!control) {
    if (auto st = configure_prefill_mma<true>(); !st) return st;
  }
  attention_mma_kernel<true><<<dim3(kAttnKvHeads, n_segments), kPrefillThreads,
      kPrefillSharedBytes, stream.native()>>>(q, nullptr, kv, attn_layer, capacity,
          populated, 1, nullptr, nullptr, nullptr, partials, n_segments, control, partition_limit);
  return check(cudaGetLastError(), "attention_mma_kernel<true>");
}

std::expected<void, Error> launch_attention_merge(
    float const* partials, std::uint16_t const* g, std::uint32_t n_segments,
    std::uint16_t* y_out, Stream const& stream, DecodeControl const* control,
    std::uint32_t partition_limit) {
  auto st = require_stream(stream, "attention_merge");
  if (!st) return st;
  if (partials == nullptr || g == nullptr || y_out == nullptr || n_segments > kAttnMaxPartitions) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "attention_merge",
                                       "null merge operand"));
  }
  if (n_segments == 0) {
    // There is no causal key; launch the same deterministic merge path with a
    // zero segment count, which emits zero gated output.
  }
  auto guard = stream.activate();
  if (!guard) return std::unexpected(guard.error());
  attention_merge_kernel<<<kAttnQueryHeads, kAttnMergeThreads, 0, stream.native()>>>(
      partials, g, n_segments, y_out, control, partition_limit);
  return check(cudaGetLastError(), "attention_merge_kernel");
}

}  // namespace qw38::cuda
