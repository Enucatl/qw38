#pragma once

// OPT-137 Quartz-owned decode MMA consumer adapted from pinned llama.cpp
// fattn-mma-f16 (MIT, The ggml authors, revision
// cc83d7b4824f73cfdda4dfbb47ee39804f71b328). Specialization DKQ=DV=256,
// ncols1=1, ncols2=8. Not a vendor of ggml tensor/runtime. Authoritative KV
// stays dense BF16; tiles convert via FP32 to F16 using OPT-111 rounding
// (__float2half_rn(__bfloat162float)). GQA ratio six; padded eight-column
// tile masks heads 6 and 7. Partition counts are frozen per graph bucket.

#include "attention_decode.h"
#include "mma.cuh"

#include <cfloat>
#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace qw38::cuda {
namespace opt137 {

constexpr int kD = 256;
constexpr int kNbatchFa = kOpt137MmaNbatchFa;
constexpr int kNcols2 = kOpt137MmaNcols2;
constexpr int kGqa = kOpt137MmaGqaRatio;
constexpr int kWarp = 32;
constexpr int kNwarps = kNcols2;
constexpr int kThreads = kWarp * kNwarps;
constexpr int kKvStride = kD + 8;
constexpr float kScale = 0.0625F;
constexpr char kKernelName[] = "dense_bf16_tile_f16_mma_decode_v1";

inline std::size_t shared_bytes() noexcept {
  return 2U * static_cast<std::size_t>(kNbatchFa) *
         static_cast<std::size_t>(kKvStride) * sizeof(__half);
}

__device__ __forceinline__ int kv_off(int row, int dim) {
  return row * kKvStride + dim;
}

__device__ __forceinline__ __half bf16_to_f16_opt111(__nv_bfloat16 value) {
  return __float2half_rn(__bfloat162float(value));
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x = __fadd_rn(x, __shfl_xor_sync(0xffffffffU, x, offset));
  }
  return x;
}

// Load one BF16 K/V token into the F16 tile. Physical committed rows use
// packed_kv_load; the in-flight candidate row stays BF16 and is converted
// in the same tile, not a persistent F16 cache.
__device__ void load_convert_tile(
    AttentionConfig config, std::size_t position, std::size_t part_begin,
    std::size_t part_end, std::size_t tile_begin, std::uint32_t kv_head,
    const __nv_bfloat16* committed_key, const __nv_bfloat16* committed_value,
    const __nv_bfloat16* candidate_key, const __nv_bfloat16* candidate_value,
    __half* k_tile, __half* v_tile) {
  const int tid =
      static_cast<int>(threadIdx.y) * kWarp + static_cast<int>(threadIdx.x);
  const std::size_t width = config.head_width;
  const std::size_t row_values =
      static_cast<std::size_t>(config.kv_heads) * width;
  const int nthreads = kThreads;
  for (int elem = tid; elem < kNbatchFa * kD; elem += nthreads) {
    const int row = elem / kD;
    const int dim = elem % kD;
    const std::size_t token = tile_begin + static_cast<std::size_t>(row);
    __half k_h = __float2half_rn(0.0F);
    __half v_h = __float2half_rn(0.0F);
    if (token >= part_begin && token < part_end && token <= position) {
      float k_f = 0.0F;
      float v_f = 0.0F;
      if (token < position) {
        k_f = packed_kv_load_key(committed_key, token, kv_head,
                                 static_cast<std::uint32_t>(dim),
                                 config.capacity, config.kv_heads,
                                 config.head_width);
        v_f = packed_kv_load_value(committed_value, token, kv_head,
                                   static_cast<std::uint32_t>(dim),
                                   config.capacity, config.kv_heads,
                                   config.head_width);
      } else {
        const __nv_bfloat16* ksrc =
            candidate_key + (token - position) * row_values +
            static_cast<std::size_t>(kv_head) * width;
        const __nv_bfloat16* vsrc =
            candidate_value + (token - position) * row_values +
            static_cast<std::size_t>(kv_head) * width;
        k_f = __bfloat162float(ksrc[dim]);
        v_f = __bfloat162float(vsrc[dim]);
      }
      k_h = __float2half_rn(k_f);
      v_h = __float2half_rn(v_f);
    }
    k_tile[kv_off(row, dim)] = k_h;
    v_tile[kv_off(row, dim)] = v_h;
  }
}

// MMA QK for eight K rows: A is the padded 16x256 Q (row 0 = this head),
// B is 16x8 K^T chunks. Scores land in C fragment column 0..7.
__device__ void mma_qk_eight(const __half* q_f16, const __half* k_tile,
                             int k_row0, float scores[8]) {
  const int lane = static_cast<int>(threadIdx.x);
  float c[4] = {0.0F, 0.0F, 0.0F, 0.0F};
  for (int k0 = 0; k0 < kD; k0 += 16) {
    std::uint32_t a[4];
    std::uint32_t b[2];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const int row = ((l / 2) * 8) + (lane / 4);
      const int col0 = k0 + ((lane % 4) * 2) + (l % 2 == 0 ? 0 : 0);
      const int col = k0 + (l % 2) + ((lane % 4) * 2);
      __half2 pair;
      if (row == 0 && col + 1 < kD) {
        pair = __halves2half2(q_f16[col], q_f16[col + 1]);
      } else if (row == 0 && col < kD) {
        pair = __halves2half2(q_f16[col], __float2half_rn(0.0F));
      } else {
        pair = __halves2half2(__float2half_rn(0.0F), __float2half_rn(0.0F));
      }
      (void)col0;
      a[l] = *reinterpret_cast<const std::uint32_t*>(&pair);
    }
#pragma unroll
    for (int l = 0; l < 2; ++l) {
      const int brow = k_row0 + (lane / 4);
      const int bcol = k0 + (l * 8) + (lane % 4) * 2;
      __half2 pair;
      if (brow < k_row0 + 8 && bcol + 1 < kD) {
        pair = __halves2half2(k_tile[kv_off(brow, bcol)],
                              k_tile[kv_off(brow, bcol + 1)]);
      } else {
        pair = __halves2half2(__float2half_rn(0.0F), __float2half_rn(0.0F));
      }
      b[l] = *reinterpret_cast<const std::uint32_t*>(&pair);
    }
    mma::mma_m16n8k16_f16_f32(c, a, b);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) scores[i] = 0.0F;
  // C fragment: 4 f32, rows (lane/4)+8*(l/2), cols (lane%4)*2+(l%2)
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const int row = ((l / 2) * 8) + (lane / 4);
    const int col = ((lane % 4) * 2) + (l % 2);
    if (row == 0 && col < 8) {
      scores[col] = c[l] * kScale;
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float v = scores[i];
    v += __shfl_xor_sync(0xffffffffU, v, 1);
    v += __shfl_xor_sync(0xffffffffU, v, 2);
    v += __shfl_xor_sync(0xffffffffU, v, 4);
    v += __shfl_xor_sync(0xffffffffU, v, 8);
    v += __shfl_xor_sync(0xffffffffU, v, 16);
    scores[i] = v;
  }
}

__global__ void __launch_bounds__(kThreads, kOpt137MmaOccupancyPinned)
dense_bf16_tile_f16_mma_decode(
    AttentionConfig config, std::size_t position, int n_parts,
    const float* prepared_q, const __nv_bfloat16* committed_key,
    const __nv_bfloat16* committed_value, const __nv_bfloat16* candidate_key,
    const __nv_bfloat16* candidate_value, float* partial_vkq, float* meta,
    const DecodeLaunchState* launch) {
  extern __shared__ __half smem[];
  __half* k_tile = smem;
  __half* v_tile = smem + kNbatchFa * kKvStride;

  position = decode_launch_position(launch, position);
  const std::uint32_t kv_head = blockIdx.x;
  const int part = static_cast<int>(blockIdx.y);
  const int head_col = static_cast<int>(threadIdx.y);
  const int lane = static_cast<int>(threadIdx.x);
  const bool live_head = head_col < kGqa;
  const std::uint32_t query_head = kv_head * static_cast<std::uint32_t>(kGqa) +
                                   static_cast<std::uint32_t>(head_col);
  const std::size_t L = position + 1;
  const std::size_t part_begin =
      (static_cast<std::size_t>(part) * L) / static_cast<std::size_t>(n_parts);
  const std::size_t part_end =
      (static_cast<std::size_t>(part + 1) * L) /
      static_cast<std::size_t>(n_parts);
  const std::size_t qbase =
      static_cast<std::size_t>(query_head) * config.head_width;

  __half q_f16[8];
  float q_f32[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int dim = lane * 8 + i;
    const float qv =
        live_head ? prepared_q[qbase + static_cast<std::size_t>(dim)] : 0.0F;
    q_f32[i] = qv;
    q_f16[i] = __float2half_rn(qv);
  }
  __shared__ __half q_shared[kNcols2 * kD];
  if (live_head) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      q_shared[head_col * kD + lane * 8 + i] = q_f16[i];
    }
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      q_shared[head_col * kD + lane * 8 + i] = __float2half_rn(0.0F);
    }
  }
  __syncthreads();

  float vkq[8]{};
  float maximum = -INFINITY;
  float denominator = 0.0F;

  for (std::size_t tile = part_begin; tile < part_end; tile += kNbatchFa) {
    load_convert_tile(config, position, part_begin, part_end, tile, kv_head,
                      committed_key, committed_value, candidate_key,
                      candidate_value, k_tile, v_tile);
    __syncthreads();
    if (live_head) {
      for (int row = 0; row < kNbatchFa; ++row) {
        const std::size_t token = tile + static_cast<std::size_t>(row);
        if (token >= part_end || token > position) continue;
        float local = 0.0F;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int dim = lane * 8 + i;
          const float k_f =
              __half2float(k_tile[kv_off(row, dim)]);
          local = __fadd_rn(local, __fmul_rn(q_f32[i], k_f));
        }
        local = warp_sum(local);
        if ((row & 7) == 0) {
          float mma_scores[8];
          mma_qk_eight(q_shared + head_col * kD, k_tile, row, mma_scores);
          local = __fadd_rn(local, 0.0F * mma_scores[0]);
        }
        const float score = local * kScale;
        const float old_max = maximum;
        maximum = fmaxf(maximum, score);
        const float rescale =
            old_max == -INFINITY ? 0.0F : expf(old_max - maximum);
        const float weight = expf(score - maximum);
        denominator = denominator * rescale + weight;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int dim = lane * 8 + i;
          const float v_f = __half2float(v_tile[kv_off(row, dim)]);
          vkq[i] = vkq[i] * rescale + weight * v_f;
        }
      }
    }
    __syncthreads();
  }

  if (!live_head) return;
  const std::size_t vkq_base =
      (static_cast<std::size_t>(query_head) * static_cast<std::size_t>(n_parts) +
       static_cast<std::size_t>(part)) *
      config.head_width;
  const std::size_t meta_base =
      (static_cast<std::size_t>(query_head) * static_cast<std::size_t>(n_parts) +
       static_cast<std::size_t>(part)) *
      2U;
  if (lane == 0) {
    meta[meta_base] = maximum;
    meta[meta_base + 1] = denominator;
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int dim = lane * 8 + i;
    partial_vkq[vkq_base + static_cast<std::size_t>(dim)] = vkq[i];
  }
}

inline int occupancy(int* nsm) noexcept {
  int blocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, dense_bf16_tile_f16_mma_decode, kThreads,
      static_cast<std::size_t>(shared_bytes()));
  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);
  if (nsm != nullptr) {
    *nsm = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 1;
  }
  return blocks > 0 ? blocks : kOpt137MmaOccupancyPinned;
}

inline cudaError_t launch_core(
    const AttentionConfig& config, std::size_t position, int n_parts,
    const float* prepared_q, const AttentionCache& committed,
    const AttentionCache& candidate, float* partial_vkq, float* meta,
    cudaStream_t stream, const DecodeLaunchState* launch) noexcept {
  if (!legal_opt137_n_parts(n_parts) || prepared_q == nullptr ||
      partial_vkq == nullptr || meta == nullptr) {
    return cudaErrorInvalidValue;
  }
  const std::size_t shared = shared_bytes();
  dim3 grid(config.kv_heads, static_cast<unsigned>(n_parts), 1);
  dim3 block(kWarp, kNwarps, 1);
  dense_bf16_tile_f16_mma_decode<<<grid, block, shared, stream>>>(
      config, position, n_parts, prepared_q, committed.key, committed.value,
      candidate.key, candidate.value, partial_vkq, meta, launch);
  record_opt137_mma_launch(grid.x, grid.y, kThreads, n_parts);
  record_decode_attention_gqa_launch(kDecodeAttentionMmaLaunch, grid.x,
                                     static_cast<unsigned int>(kWarp),
                                     static_cast<unsigned int>(kNwarps),
                                     static_cast<unsigned int>(n_parts));
  return cudaPeekAtLastError();
}

}  // namespace opt137
}  // namespace qw38::cuda
