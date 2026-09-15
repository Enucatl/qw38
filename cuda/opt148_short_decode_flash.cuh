#pragma once

// OPT-148 Quartz-owned flash-style vector decode attention screen over
// shipping D2048 `vec128_online_decode_attention`. llama.cpp
// flash_attn_ext_vec<256,1> (MIT, The ggml authors, revision
// cc83d7b4824f73cfdda4dfbb47ee39804f71b328) is the implementation
// boundary only: interleaved K/V tiles, 16-byte BF16 copies, float2 V
// accumulators, online reduction. Not a vendor of ggml tensor/runtime.
// BF16 KV, partial RoPE, GQA 24/4, online-softmax merge, and OPT-137
// MMA at positions >=8192 stay in place. n_parts stays 16; no OPT-130
// occupancy path. Mechanism remains unknown unless separately evidenced.

#include "attention_decode.h"

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace qw38::cuda {
namespace opt148 {

constexpr char kControlId[] = "vec128_online";
constexpr char kCandidateId[] = "decode_attention_flash_vec_v1";
constexpr char kLaunchCandidate[] = "flash_vec_decode_attention";
constexpr char kLlamaRevision[] = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr int kD = 256;
constexpr int kKvHeads = 4;
constexpr int kThreads = 128;
constexpr int kWarpSize = 32;
constexpr int kWarps = 4;
constexpr int kCpyBytes = 16;
constexpr int kCpyNe = 4;
constexpr int kNthreadsKq = 8;
constexpr int kNthreadsV = 8;
constexpr int kVRowsPerThread = 8;
constexpr int kVColsPerIter = 4;
constexpr int kQRegs = (kD / 2) / kNthreadsKq;
constexpr int kNeCombine = kWarps * kVColsPerIter * kD;
constexpr float kScale = 0.0625F;
constexpr float kRmsEpsilon = 1.0e-6F;
constexpr float kRopeTheta = 10000000.0F;
constexpr int kCorrectnessPositions[] = {0, 1023, 1024, 2047, 2048, 4096, 8192};
constexpr int kCorrectnessPositionCount = 7;

inline bool is_candidate_path(const char* path) noexcept {
  return path != nullptr && std::strcmp(path, kCandidateId) == 0;
}

__device__ __forceinline__ void cpy16(void* dst, const void* src) {
  *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(src);
}

template <int Width>
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = Width / 2; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, offset, Width);
  }
  return x;
}

template <int Width>
__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
  for (int offset = Width / 2; offset > 0; offset >>= 1) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, Width));
  }
  return x;
}

__device__ __forceinline__ float2 bf162_to_float2(nv_bfloat162 value) {
#if __CUDA_ARCH__ >= 800
  return __bfloat1622float2(value);
#else
  return make_float2(__bfloat162float(value.x), __bfloat162float(value.y));
#endif
}

__device__ __forceinline__ float mad_float2(float acc, float2 left,
                                            float2 right) {
  acc += left.x * right.x;
  acc += left.y * right.y;
  return acc;
}

__device__ __forceinline__ const __nv_bfloat16* kv_row(
    std::size_t token, std::size_t position, std::uint32_t kv_head,
    std::uint32_t capacity, const __nv_bfloat16* committed,
    const __nv_bfloat16* candidate) {
  if (token < position) {
    return committed + attention_kv_physical_index(token, kv_head, 0, capacity,
                                                   static_cast<std::uint32_t>(kD));
  }
  const std::size_t row =
      static_cast<std::size_t>(kKvHeads) * static_cast<std::size_t>(kD);
  return candidate + (token - position) * row +
         static_cast<std::size_t>(kv_head) * static_cast<std::size_t>(kD);
}

// Flash-style interleaved K/V tile decode. Writes vec128-compatible
// partial_vkq / meta so merge_decode_kv_parts and output-gate stay shared.
// Internal linkage: this header is included from attention_decode.cu and
// the native test.
static __global__ void __launch_bounds__(kThreads, 1)
flash_vec_decode_attention(
    AttentionConfig config, std::size_t position, int n_parts,
    const float* query, const float* query_scale,
    const __nv_bfloat16* committed_key, const __nv_bfloat16* committed_value,
    const __nv_bfloat16* candidate_key, const __nv_bfloat16* candidate_value,
    float* normalized_query, float* partial_vkq, float* meta,
    const DecodeLaunchState* launch) {
  position = decode_launch_position(launch, position);
  constexpr int nthreads_kq = kNthreadsKq;
  constexpr int nthreads_v = kNthreadsV;
  const int n_kv = static_cast<int>(position + 1);
  const std::uint32_t query_head = blockIdx.x;
  const int part = static_cast<int>(blockIdx.y);
  const int tid = kWarpSize * static_cast<int>(threadIdx.y) +
                  static_cast<int>(threadIdx.x);
  const std::uint32_t group = config.query_heads / config.kv_heads;
  const std::uint32_t kv_head = query_head / group;
  const std::size_t qbase = static_cast<std::size_t>(query_head) * kD;
  const std::size_t width = config.head_width;

  __shared__ float q_scratch[kD];
  __shared__ float inverse;
  if (threadIdx.y == 0) {
    float inv = 0.0F;
    if (threadIdx.x == 0) {
      float sum = 0.0F;
      for (std::uint32_t i = 0; i < config.head_width; ++i) {
        const float item = query[qbase + i];
        sum = __fadd_rn(sum, item * item);
      }
      inverse = 1.0F / sqrtf(sum / static_cast<float>(width) + kRmsEpsilon);
      inv = inverse;
    }
    inv = __shfl_sync(0xffffffff, inv, 0);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const std::uint32_t dim =
          static_cast<std::uint32_t>(threadIdx.x) * 8U +
          static_cast<std::uint32_t>(i);
      q_scratch[dim] = query[qbase + dim] * inv * query_scale[dim];
    }
    __syncwarp();
    const std::uint32_t half = config.rotary_width / 2;
    for (std::uint32_t pair = threadIdx.x; pair < half; pair += kWarpSize) {
      const float first = q_scratch[pair];
      const float second = q_scratch[half + pair];
      const float exponent =
          static_cast<float>(pair * 2) / static_cast<float>(config.rotary_width);
      const float angle =
          static_cast<float>(position) / powf(kRopeTheta, exponent);
      const float c = cosf(angle);
      const float s = sinf(angle);
      q_scratch[pair] = first * c - second * s;
      q_scratch[half + pair] = second * c + first * s;
    }
    __syncwarp();
    if (part == 0) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const std::uint32_t dim =
            static_cast<std::uint32_t>(threadIdx.x) * 8U +
            static_cast<std::uint32_t>(i);
        normalized_query[qbase + dim] = q_scratch[dim];
      }
    }
  }
  __syncthreads();

  float2 vkq[kQRegs];
  float2 q_reg[kQRegs];
#pragma unroll
  for (int i = 0; i < kQRegs; ++i) {
    vkq[i] = make_float2(0.0F, 0.0F);
    q_reg[i] = make_float2(0.0F, 0.0F);
  }
  const float2* q_head = reinterpret_cast<const float2*>(q_scratch);
#pragma unroll
  for (int i0 = 0; i0 < kD / 2; i0 += nthreads_kq * kCpyNe) {
    const int i =
        i0 + (static_cast<int>(threadIdx.x) % nthreads_kq) * kCpyNe;
    cpy16(&q_reg[i0 / nthreads_kq], &q_head[i]);
    cpy16(&q_reg[i0 / nthreads_kq + kCpyNe / 2], &q_head[i + kCpyNe / 2]);
  }
#pragma unroll
  for (int k = 0; k < kQRegs; ++k) {
    q_reg[k].x *= kScale;
    q_reg[k].y *= kScale;
  }

  float kq_max = -FLT_MAX / 2.0F;
  float kq_sum = 0.0F;
  __shared__ float kq_shared[kNeCombine];
  for (int k_vkq0 = part * kThreads; k_vkq0 < n_kv;
       k_vkq0 += n_parts * kThreads) {
    float kq_reg = 0.0F;
    float kq_max_new = kq_max;
#pragma unroll
    for (int i_kq0 = 0; i_kq0 < nthreads_kq; ++i_kq0) {
      const int i_kq =
          static_cast<int>(threadIdx.y) * kWarpSize +
          (static_cast<int>(threadIdx.x) & ~(nthreads_kq - 1)) + i_kq0;
      const int token = k_vkq0 + i_kq;
      float sum = 0.0F;
      if (token < n_kv) {
        const nv_bfloat162* k_row = reinterpret_cast<const nv_bfloat162*>(
            kv_row(static_cast<std::size_t>(token), position, kv_head,
                   config.capacity, committed_key, candidate_key));
#pragma unroll
        for (int k0 = 0; k0 < kD / 2; k0 += nthreads_kq * kCpyNe) {
          __align__(16) nv_bfloat162 tmp[kCpyNe];
          cpy16(tmp, k_row + k0 +
                         (static_cast<int>(threadIdx.x) % nthreads_kq) * kCpyNe);
#pragma unroll
          for (int k1 = 0; k1 < kCpyNe; ++k1) {
            sum = mad_float2(sum, bf162_to_float2(tmp[k1]),
                             q_reg[k0 / nthreads_kq + k1]);
          }
        }
      }
      sum = warp_sum<nthreads_kq>(sum);
      if (token < n_kv) {
        kq_max_new = fmaxf(kq_max_new, sum);
      }
      if ((static_cast<int>(threadIdx.x) % nthreads_kq) == i_kq0) {
        kq_reg = token < n_kv ? sum : 0.0F;
      }
    }
#pragma unroll
    for (int offset = nthreads_kq; offset < kWarpSize; offset <<= 1) {
      kq_max_new =
          fmaxf(kq_max_new, __shfl_xor_sync(0xffffffff, kq_max_new, offset,
                                            kWarpSize));
    }
    const float kq_max_scale = expf(kq_max - kq_max_new);
    kq_max = kq_max_new;
    const int my_token = k_vkq0 + tid;
    if (my_token < n_kv) {
      kq_reg = expf(kq_reg - kq_max);
    } else {
      kq_reg = 0.0F;
    }
    kq_sum = kq_sum * kq_max_scale + kq_reg;
    kq_shared[tid] = kq_reg;
#pragma unroll
    for (int i = 0; i < kQRegs; ++i) {
      vkq[i].x *= kq_max_scale;
      vkq[i].y *= kq_max_scale;
    }
    __syncwarp();
#pragma unroll
    for (int k0 = 0; k0 < kWarpSize; k0 += kVColsPerIter) {
      const int k =
          static_cast<int>(threadIdx.y) * kWarpSize + k0 +
          static_cast<int>(threadIdx.x) / nthreads_v;
      const int token = k_vkq0 + k;
      const float kq_k = kq_shared[k];
#pragma unroll
      for (int i0 = 0; i0 < kD / 2;
           i0 += nthreads_v * kVRowsPerThread / 2) {
        float2 tmp[kVRowsPerThread / 2];
#pragma unroll
        for (int t = 0; t < kVRowsPerThread / 2; ++t) {
          tmp[t] = make_float2(0.0F, 0.0F);
        }
        if (token < n_kv) {
          const __nv_bfloat16* v_row =
              kv_row(static_cast<std::size_t>(token), position, kv_head,
                     config.capacity, committed_value, candidate_value);
          const int i_v =
              2 * i0 +
              (static_cast<int>(threadIdx.x) % nthreads_v) * kVRowsPerThread;
          __align__(16) nv_bfloat162 packed[kVRowsPerThread / 2];
          cpy16(packed, v_row + i_v);
#pragma unroll
          for (int t = 0; t < kVRowsPerThread / 2; ++t) {
            tmp[t] = bf162_to_float2(packed[t]);
          }
        }
#pragma unroll
        for (int t = 0; t < kVRowsPerThread / 2; ++t) {
          vkq[i0 / nthreads_v + t].x += tmp[t].x * kq_k;
          vkq[i0 / nthreads_v + t].y += tmp[t].y * kq_k;
        }
      }
    }
  }

  __shared__ float kq_max_shared[kWarpSize];
  __shared__ float kq_sum_shared[kWarpSize];
  if (threadIdx.y == 0) {
    kq_max_shared[threadIdx.x] = -FLT_MAX / 2.0F;
    kq_sum_shared[threadIdx.x] = 0.0F;
  }
  __syncthreads();
  if (threadIdx.x == 0) kq_max_shared[threadIdx.y] = kq_max;
  __syncthreads();

  float kqmax_new = kq_max_shared[threadIdx.x];
  kqmax_new = warp_max<kWarpSize>(kqmax_new);
  const float kqmax_scale = expf(kq_max - kqmax_new);
  kq_max = kqmax_new;
#pragma unroll
  for (int i = 0; i < kQRegs; ++i) {
    vkq[i].x *= kqmax_scale;
    vkq[i].y *= kqmax_scale;
  }
  float2* vkq_tmp =
      reinterpret_cast<float2*>(kq_shared) +
      static_cast<int>(threadIdx.y) * (kVColsPerIter * kD / 2) +
      (static_cast<int>(threadIdx.x) / nthreads_v) * (kD / 2);
#pragma unroll
  for (int i0 = 0; i0 < kD / 2; i0 += nthreads_v * kVRowsPerThread / 2) {
    const int i_vkq =
        i0 + (static_cast<int>(threadIdx.x) % nthreads_v) *
                 (kVRowsPerThread / 2);
    cpy16(vkq_tmp + i_vkq, &vkq[i0 / nthreads_v]);
    cpy16(vkq_tmp + i_vkq + kVRowsPerThread / 4,
          &vkq[i0 / nthreads_v + kVRowsPerThread / 4]);
  }
  kq_sum *= kqmax_scale;
  kq_sum = warp_sum<kWarpSize>(kq_sum);
  if (threadIdx.x == 0) kq_sum_shared[threadIdx.y] = kq_sum;
  __syncthreads();

  kq_sum = kq_sum_shared[threadIdx.x];
  kq_sum = warp_sum<kWarpSize>(kq_sum);
  const std::size_t vkq_base =
      (static_cast<std::size_t>(query_head) * static_cast<std::size_t>(n_parts) +
       static_cast<std::size_t>(part)) *
      static_cast<std::size_t>(kD);
#pragma unroll
  for (int i0 = 0; i0 < kD; i0 += kThreads) {
    float dst_val = 0.0F;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
#pragma unroll
      for (int v = 0; v < kVColsPerIter; ++v) {
        dst_val += kq_shared[w * kVColsPerIter * kD + v * kD + i0 + tid];
      }
    }
    partial_vkq[vkq_base + static_cast<std::size_t>(i0 + tid)] = dst_val;
  }
  if (tid == 0) {
    const std::size_t meta_base =
        (static_cast<std::size_t>(query_head) *
             static_cast<std::size_t>(n_parts) +
         static_cast<std::size_t>(part)) *
        2U;
    meta[meta_base] = kq_max;
    meta[meta_base + 1] = kq_sum;
  }
}

inline int occupancy() noexcept {
  int blocks = 0;
  const cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, flash_vec_decode_attention, kThreads, 0);
  if (error != cudaSuccess) return 0;
  return blocks;
}

inline void kernel_attributes(int* registers, std::size_t* local_bytes,
                              int* occ) noexcept {
  cudaFuncAttributes attrs{};
  cudaFuncGetAttributes(&attrs, flash_vec_decode_attention);
  if (registers != nullptr) *registers = attrs.numRegs;
  if (local_bytes != nullptr) {
    *local_bytes = static_cast<std::size_t>(attrs.localSizeBytes);
  }
  if (occ != nullptr) *occ = occupancy();
}

inline cudaError_t launch_core(
    const AttentionConfig& config, std::size_t position, int n_parts,
    const float* query, const float* query_norm_scale,
    const AttentionCache& committed, const AttentionCache& candidate_row,
    float* normalized_query, float* partial_vkq, float* meta,
    cudaStream_t stream, const DecodeLaunchState* launch) noexcept {
  if (!legal_vec128_n_parts(n_parts) || query == nullptr ||
      query_norm_scale == nullptr || committed.key == nullptr ||
      committed.value == nullptr || candidate_row.key == nullptr ||
      candidate_row.value == nullptr || normalized_query == nullptr ||
      partial_vkq == nullptr || meta == nullptr) {
    return cudaErrorInvalidValue;
  }
  dim3 attention(config.query_heads, static_cast<unsigned>(n_parts), 1);
  dim3 block(kWarpSize, kWarps, 1);
  flash_vec_decode_attention<<<attention, block, 0, stream>>>(
      config, position, n_parts, query, query_norm_scale, committed.key,
      committed.value, candidate_row.key, candidate_row.value,
      normalized_query, partial_vkq, meta, launch);
  return cudaPeekAtLastError();
}

}  // namespace opt148
}  // namespace qw38::cuda
