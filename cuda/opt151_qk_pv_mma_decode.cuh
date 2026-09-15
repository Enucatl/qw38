#pragma once

// OPT-151 decode MMA consumer: both QK and PV use output-producing MMA.
// Adapted from pinned llama.cpp fattn-mma-f16 (MIT, The ggml authors,
// revision cc83d7b4824f73cfdda4dfbb47ee39804f71b328), Ampere specialization
// DKQ=DV=256, ncols1=1, ncols2=8, nthreads=64, occupancy pin 4, nbatch_fa=64,
// nstages target 2. Not a vendor of ggml tensor/runtime.
//
// Authoritative KV stays dense BF16. Each loaded tile converts once via FP32
// to F16 using OPT-111 rounding (__float2half_rn(__bfloat162float)) and is
// reused across the padded eight GQA columns (live heads 0-5, mask 6-7).
// Q is scaled by 1/sqrt(256) then rounded to F16 (llama Q-in-tile scale).
// Softmax is tiled over 16-row MMA chunks with FP32 max/sum; probabilities
// round to F16 for the PV MMA. VKQ combine stays FP32.
// OPT-137's scalar QK/PV kernel is unchanged; this is the v2 candidate.

#include "attention_decode.h"
#include "mma.cuh"

#include <cfloat>
#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace qw38::cuda {
namespace opt151 {

constexpr int kD = 256;
constexpr int kNbatchFa = kOpt137MmaNbatchFa;
constexpr int kNcols2 = kOpt137MmaNcols2;
constexpr int kGqa = kOpt137MmaGqaRatio;
constexpr int kWarp = 32;
constexpr int kNwarps = 2;
constexpr int kThreads = kWarp * kNwarps;
constexpr int kKvStride = kD + 8;
constexpr int kMmaRows = 16;
constexpr int kMmaK = 16;
constexpr float kScale = 0.0625F;
constexpr char kKernelName[] = "decode_attention_qk_pv_mma_v2";
constexpr char kCandidateId[] = "decode_attention_qk_pv_mma_v2";

inline std::size_t shared_bytes() noexcept {
  return 2U * static_cast<std::size_t>(kNbatchFa) *
         static_cast<std::size_t>(kKvStride) * sizeof(__half);
}

__device__ __forceinline__ int kv_off(int row, int dim) {
  return row * kKvStride + dim;
}

__device__ __forceinline__ std::uint32_t pack_half2(__half a, __half b) {
  const __half2 pair = __halves2half2(a, b);
  return *reinterpret_cast<const std::uint32_t*>(&pair);
}

__device__ __forceinline__ __half bf16_to_f16_opt111(__nv_bfloat16 value) {
  return __float2half_rn(__bfloat162float(value));
}

// Load one BF16 K/V tile into F16. Convert once per element; reuse across
// the eight GQA columns. Physical committed rows use packed_kv_load; the
// in-flight candidate row stays BF16 and is converted in the same tile.
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
  for (int elem = tid; elem < kNbatchFa * kD; elem += kThreads) {
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

// QK: C[16x8] += K[16x256] x Q^T[256x8] via m16n8k16. Accumulators are the
// scores; there is no scalar QK path and no zero-weight helper.
__device__ void mma_qk_16x8(const __half* q_cols, const __half* k_tile,
                            int k_row0, float scores[4]) {
  const int lane = static_cast<int>(threadIdx.x);
  const int group = lane / 4;
  const int tid = lane % 4;
  float c[4] = {0.0F, 0.0F, 0.0F, 0.0F};
  for (int k0 = 0; k0 < kD; k0 += kMmaK) {
    std::uint32_t a[4];
    std::uint32_t b[2];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const int row = ((l % 2) * 8) + group;
      const int col = k0 + ((l / 2) * 8) + tid * 2;
      const __half x = k_tile[kv_off(k_row0 + row, col)];
      const __half y = k_tile[kv_off(k_row0 + row, col + 1)];
      a[l] = pack_half2(x, y);
    }
#pragma unroll
    for (int l = 0; l < 2; ++l) {
      const int kdim = k0 + l * 8 + tid * 2;
      b[l] = pack_half2(q_cols[group * kD + kdim],
                        q_cols[group * kD + kdim + 1]);
    }
    mma::mma_m16n8k16_f16_f32(c, a, b);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) scores[i] = c[i];
}

// PV: C[16x8] += V^T[16x16] x W[16x8] for one 16-wide V slice. Weights are
// F16-rounded softmax probabilities. Accumulators feed the output VKQ.
__device__ void mma_pv_16x8(const __half* v_tile, int v_row0, int v_base,
                            const __half* weights, float acc[4]) {
  const int lane = static_cast<int>(threadIdx.x);
  const int group = lane / 4;
  const int tid = lane % 4;
  std::uint32_t a[4];
  std::uint32_t b[2];
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const int vrow = ((l % 2) * 8) + group;
    const int krow = ((l / 2) * 8) + tid * 2;
    const __half x = v_tile[kv_off(v_row0 + krow, v_base + vrow)];
    const __half y = v_tile[kv_off(v_row0 + krow + 1, v_base + vrow)];
    a[l] = pack_half2(x, y);
  }
#pragma unroll
  for (int l = 0; l < 2; ++l) {
    const int krow = l * 8 + tid * 2;
    b[l] = pack_half2(weights[group * kMmaRows + krow],
                      weights[group * kMmaRows + krow + 1]);
  }
  mma::mma_m16n8k16_f16_f32(acc, a, b);
}

__global__ void __launch_bounds__(kThreads, kOpt151MmaOccupancyPinned)
decode_attention_qk_pv_mma_v2(
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
  const int lane = static_cast<int>(threadIdx.x);
  const int warp = static_cast<int>(threadIdx.y);
  const int tid = warp * kWarp + lane;
  const std::size_t L = position + 1;
  const std::size_t part_begin =
      (static_cast<std::size_t>(part) * L) / static_cast<std::size_t>(n_parts);
  const std::size_t part_end =
      (static_cast<std::size_t>(part + 1) * L) /
      static_cast<std::size_t>(n_parts);

  __shared__ __half q_shared[kNcols2 * kD];
  __shared__ float qk_sh[kNwarps][kMmaRows * kNcols2];
  __shared__ __half w_sh[kNwarps][kNcols2 * kMmaRows];
  __shared__ float warp_max[kNwarps][kNcols2];
  __shared__ float warp_denom[kNwarps][kNcols2];
  __shared__ float warp_vkq[kNwarps][kNcols2][kD / kMmaRows][4];

  const std::uint32_t q_heads_base =
      kv_head * static_cast<std::uint32_t>(kGqa);
  for (int elem = tid; elem < kNcols2 * kD; elem += kThreads) {
    const int col = elem / kD;
    const int dim = elem % kD;
    float qv = 0.0F;
    if (col < kGqa) {
      const std::size_t qbase =
          static_cast<std::size_t>(q_heads_base + static_cast<std::uint32_t>(col)) *
          config.head_width;
      qv = prepared_q[qbase + static_cast<std::size_t>(dim)] * kScale;
    }
    q_shared[col * kD + dim] = __float2half_rn(qv);
  }
  __syncthreads();

  float maximum[kNcols2];
  float denominator[kNcols2];
  float vkq_frag[kD / kMmaRows][4];
#pragma unroll
  for (int h = 0; h < kNcols2; ++h) {
    maximum[h] = -INFINITY;
    denominator[h] = 0.0F;
  }
#pragma unroll
  for (int t = 0; t < kD / kMmaRows; ++t) {
#pragma unroll
    for (int l = 0; l < 4; ++l) vkq_frag[t][l] = 0.0F;
  }

  const int group = lane / 4;
  const int tid4 = lane % 4;

  for (std::size_t tile = part_begin; tile < part_end; tile += kNbatchFa) {
    load_convert_tile(config, position, part_begin, part_end, tile, kv_head,
                      committed_key, committed_value, candidate_key,
                      candidate_value, k_tile, v_tile);
    __syncthreads();

    for (int chunk = 0; chunk < 2; ++chunk) {
      const int row0 = warp * 32 + chunk * kMmaRows;
      float scores[4];
      mma_qk_16x8(q_shared, k_tile, row0, scores);
#pragma unroll
      for (int l = 0; l < 4; ++l) {
        const int row = ((l / 2) * 8) + group;
        const int col = tid4 * 2 + (l % 2);
        const std::size_t token = tile + static_cast<std::size_t>(row0 + row);
        float score = scores[l];
        if (token >= part_end || token > position) score = -INFINITY;
        qk_sh[warp][row * kNcols2 + col] = score;
      }
      __syncwarp();

      float tile_max[kNcols2];
      float tile_sum[kNcols2];
#pragma unroll
      for (int h = 0; h < kNcols2; ++h) {
        tile_max[h] = -INFINITY;
        tile_sum[h] = 0.0F;
      }
      if (lane < kMmaRows) {
#pragma unroll
        for (int h = 0; h < kNcols2; ++h) {
          const float s = qk_sh[warp][lane * kNcols2 + h];
          tile_max[h] = s;
        }
      }
#pragma unroll
      for (int h = 0; h < kNcols2; ++h) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          tile_max[h] = fmaxf(tile_max[h],
                              __shfl_xor_sync(0xffffffffU, tile_max[h], off));
        }
      }

      bool any_valid = false;
#pragma unroll
      for (int h = 0; h < kNcols2; ++h) {
        if (tile_max[h] > -INFINITY) any_valid = true;
      }
      if (any_valid) {
        float rescale[kNcols2];
#pragma unroll
        for (int h = 0; h < kNcols2; ++h) {
          const float old_max = maximum[h];
          const float new_max = fmaxf(old_max, tile_max[h]);
          rescale[h] =
              old_max == -INFINITY ? 0.0F : expf(old_max - new_max);
          maximum[h] = new_max;
        }
#pragma unroll
        for (int t = 0; t < kD / kMmaRows; ++t) {
#pragma unroll
          for (int l = 0; l < 4; ++l) {
            const int col = tid4 * 2 + (l % 2);
            vkq_frag[t][l] *= rescale[col];
          }
        }
        if (lane < kMmaRows) {
#pragma unroll
          for (int h = 0; h < kNcols2; ++h) {
            const float s = qk_sh[warp][lane * kNcols2 + h];
            const float w =
                (s > -INFINITY) ? expf(s - maximum[h]) : 0.0F;
            tile_sum[h] = w;
            w_sh[warp][h * kMmaRows + lane] = __float2half_rn(w);
          }
        } else {
#pragma unroll
          for (int h = 0; h < kNcols2; ++h) tile_sum[h] = 0.0F;
        }
#pragma unroll
        for (int h = 0; h < kNcols2; ++h) {
#pragma unroll
          for (int off = 16; off > 0; off >>= 1) {
            tile_sum[h] += __shfl_xor_sync(0xffffffffU, tile_sum[h], off);
          }
          denominator[h] = denominator[h] * rescale[h] + tile_sum[h];
        }
        __syncwarp();
#pragma unroll
        for (int t = 0; t < kD / kMmaRows; ++t) {
          mma_pv_16x8(v_tile, row0, t * kMmaRows, w_sh[warp], vkq_frag[t]);
        }
      }
      __syncwarp();
    }
    __syncthreads();
  }

#pragma unroll
  for (int h = 0; h < kNcols2; ++h) {
    if (lane == 0) {
      warp_max[warp][h] = maximum[h];
      warp_denom[warp][h] = denominator[h];
    }
  }
#pragma unroll
  for (int t = 0; t < kD / kMmaRows; ++t) {
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const int col = tid4 * 2 + (l % 2);
      warp_vkq[warp][col][t][l] = vkq_frag[t][l];
    }
  }
  __syncthreads();

  if (warp != 0) return;

  float scale0[kNcols2];
  float scale1[kNcols2];
#pragma unroll
  for (int h = 0; h < kNcols2; ++h) {
    const float m0 = warp_max[0][h];
    const float m1 = warp_max[1][h];
    const float merged_max = fmaxf(m0, m1);
    scale0[h] = m0 == -INFINITY ? 0.0F : expf(m0 - merged_max);
    scale1[h] = m1 == -INFINITY ? 0.0F : expf(m1 - merged_max);
    maximum[h] = merged_max;
    denominator[h] = warp_denom[0][h] * scale0[h] + warp_denom[1][h] * scale1[h];
  }
#pragma unroll
  for (int t = 0; t < kD / kMmaRows; ++t) {
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const int col = tid4 * 2 + (l % 2);
      vkq_frag[t][l] = warp_vkq[0][col][t][l] * scale0[col] +
                       warp_vkq[1][col][t][l] * scale1[col];
    }
  }

  for (int h = 0; h < kGqa; ++h) {
    const std::uint32_t query_head =
        q_heads_base + static_cast<std::uint32_t>(h);
    const std::size_t vkq_base =
        (static_cast<std::size_t>(query_head) *
             static_cast<std::size_t>(n_parts) +
         static_cast<std::size_t>(part)) *
        config.head_width;
    const std::size_t meta_base =
        (static_cast<std::size_t>(query_head) *
             static_cast<std::size_t>(n_parts) +
         static_cast<std::size_t>(part)) *
        2U;
    if (lane == 0) {
      meta[meta_base] = maximum[h];
      meta[meta_base + 1] = denominator[h];
    }
#pragma unroll
    for (int t = 0; t < kD / kMmaRows; ++t) {
#pragma unroll
      for (int l = 0; l < 4; ++l) {
        const int row = ((l / 2) * 8) + group;
        const int col = tid4 * 2 + (l % 2);
        if (col == h) {
          partial_vkq[vkq_base + static_cast<std::size_t>(t * kMmaRows + row)] =
              vkq_frag[t][l];
        }
      }
    }
  }
}

// Isolated MMA products for the dependency check: nontrivial Q/K/V, both
// QK and PV accumulators written to output with no scalar substitute.
__global__ void mma_qk_pv_products_kernel(const __half* q_cols,
                                          const __half* k_tile,
                                          const __half* v_tile,
                                          float* scores_out,
                                          float* pv_out) {
  __shared__ __half q_shared[kNcols2 * kD];
  __shared__ __half w_sh[kNcols2 * kMmaRows];
  const int lane = static_cast<int>(threadIdx.x);
  const int tid = static_cast<int>(threadIdx.y) * kWarp + lane;
  for (int elem = tid; elem < kNcols2 * kD; elem += kThreads) {
    q_shared[elem] = q_cols[elem];
  }
  __syncthreads();
  if (threadIdx.y != 0) return;
  float scores[4];
  mma_qk_16x8(q_shared, k_tile, 0, scores);
  const int group = lane / 4;
  const int tid4 = lane % 4;
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const int row = ((l / 2) * 8) + group;
    const int col = tid4 * 2 + (l % 2);
    scores_out[row * kNcols2 + col] = scores[l];
  }
  __syncwarp();
  if (lane < kMmaRows) {
#pragma unroll
    for (int h = 0; h < kNcols2; ++h) {
      w_sh[h * kMmaRows + lane] = __float2half_rn(1.0F / 16.0F);
    }
  }
  __syncwarp();
  float acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
  mma_pv_16x8(v_tile, 0, 0, w_sh, acc);
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const int row = ((l / 2) * 8) + group;
    const int col = tid4 * 2 + (l % 2);
    pv_out[row * kNcols2 + col] = acc[l];
  }
}

inline int occupancy(int* nsm) noexcept {
  int blocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, decode_attention_qk_pv_mma_v2, kThreads,
      static_cast<std::size_t>(shared_bytes()));
  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);
  if (nsm != nullptr) {
    *nsm = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 1;
  }
  return blocks > 0 ? blocks : kOpt151MmaOccupancyPinned;
}

inline void prepare_runtime() noexcept {
  static bool shared_ready = false;
  if (!shared_ready) {
    cudaFuncSetAttribute(
        decode_attention_qk_pv_mma_v2,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes()));
    shared_ready = true;
  }
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
  prepare_runtime();
  const std::size_t shared = shared_bytes();
  dim3 grid(config.kv_heads, static_cast<unsigned>(n_parts), 1);
  dim3 block(kWarp, kNwarps, 1);
  decode_attention_qk_pv_mma_v2<<<grid, block, shared, stream>>>(
      config, position, n_parts, prepared_q, committed.key, committed.value,
      candidate.key, candidate.value, partial_vkq, meta, launch);
  record_opt151_mma_launch(grid.x, grid.y, kThreads, n_parts);
  record_opt137_mma_launch(grid.x, grid.y, kThreads, n_parts);
  record_decode_attention_gqa_launch(kDecodeAttentionQkPvMmaV2Launch, grid.x,
                                     static_cast<unsigned int>(kWarp),
                                     static_cast<unsigned int>(kNwarps),
                                     static_cast<unsigned int>(n_parts));
  return cudaPeekAtLastError();
}

inline cudaError_t launch_mma_products(const __half* q_cols,
                                       const __half* k_tile,
                                       const __half* v_tile, float* scores_out,
                                       float* pv_out,
                                       cudaStream_t stream) noexcept {
  dim3 block(kWarp, kNwarps, 1);
  mma_qk_pv_products_kernel<<<1, block, 0, stream>>>(q_cols, k_tile, v_tile,
                                                     scores_out, pv_out);
  return cudaPeekAtLastError();
}

}  // namespace opt151
}  // namespace qw38::cuda
