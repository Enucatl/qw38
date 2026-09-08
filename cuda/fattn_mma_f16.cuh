#pragma once

// Focused Ampere fattn-mma-f16 analog for Quartz OPT-019 (DKQ=DV=256, GQA 6).
// Adapted from llama.cpp ggml/src/ggml-cuda/fattn.cu, fattn-mma-f16.cuh, and
// mma.cuh at cc83d7b4824f73cfdda4dfbb47ee39804f71b328 (MIT, The ggml authors).
// Not a vendor of those files and not a copy of ../ds4. ds4 is mentioned only
// for multi-row online structure. Explicit mma.sync PTX is not compiler FMA
// contraction; Makefile NVCCFLAGS --fmad=false stays unchanged.

#include "attention_decode.h"
#include "mma.cuh"

#include <cstdint>

#include <cuda_fp16.h>

namespace qw38::cuda {

constexpr int kFattnNcols2 = 2;
constexpr int kFattnNbatchFa = 32;
constexpr int kFattnHeadWidth = 256;
constexpr int kFattnSubgroups = 3;
constexpr float kFattnRmsEpsilon = 1.0e-6F;
constexpr float kFattnRopeTheta = 10000000.0F;

// Pinned after the 2048-row ncols1 sweep. Legal values: 8, 16, 32.
constexpr int kSelectedAttentionMmaQueryRows = 16;

inline __device__ float fattn_block_sum(float local, float* scratch, int tid,
                                       int nthreads) {
  float value = local;
  for (int mask = 16; mask > 0; mask >>= 1) {
    value = __fadd_rn(value, __shfl_xor_sync(0xffffffffU, value, mask));
  }
  const int lane = tid % 32;
  const int warp = tid / 32;
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  if (tid == 0) {
    float total = 0.0F;
    const int nwarps = nthreads / 32;
    for (int other = 0; other < nwarps; ++other)
      total = __fadd_rn(total, scratch[other]);
    scratch[0] = total;
  }
  __syncthreads();
  const float total = scratch[0];
  __syncthreads();
  return total;
}

constexpr int fattn_nthreads_for_ncols1(int ncols1) {
  return ncols1 <= 8 ? 64 : 128;
}

constexpr int fattn_occupancy_target_for_ncols1(int ncols1) {
  return ncols1 <= 8 ? 4 : 2;
}

constexpr std::size_t fattn_quality_shared_bytes(int ncols1, bool dual_f16) {
  const int ncols = ncols1 * kFattnNcols2;
  const int nwarps = fattn_nthreads_for_ncols1(ncols1) / 32;
  const std::size_t kv = 2ull * kFattnNbatchFa * kFattnHeadWidth *
                         sizeof(__nv_bfloat16);
  const std::size_t q = static_cast<std::size_t>(ncols) * kFattnHeadWidth *
                        sizeof(__half) * (dual_f16 ? 2ull : 1ull);
  const std::size_t scores = static_cast<std::size_t>(ncols) * kFattnNbatchFa *
                             sizeof(float);
  const std::size_t stats = static_cast<std::size_t>(ncols) * 2ull *
                            sizeof(float);
  const std::size_t scratch = kFattnHeadWidth * sizeof(float);
  const std::size_t cparts = static_cast<std::size_t>(nwarps) * 32ull * 4ull *
                             sizeof(float);
  const std::size_t rescale = static_cast<std::size_t>(ncols) * sizeof(float);
  return kv + q + scores + stats + scratch + cparts + rescale;
}

template <int Ncols1, bool DualF16>
__global__ void __launch_bounds__(Ncols1 <= 8 ? 64 : 128, 1)
fattn_mma_quality_kernel(
    AttentionConfig config, std::size_t start_position, std::size_t token_count,
    const float* query, const float* query_scale, const float* gate,
    const __nv_bfloat16* committed_key, const __nv_bfloat16* committed_value,
    const __nv_bfloat16* candidate_key, const __nv_bfloat16* candidate_value,
    float* output, float* normalized_query) {
  constexpr int kNcols = Ncols1 * kFattnNcols2;
  constexpr int kNthreads = Ncols1 <= 8 ? 64 : 128;
  constexpr int kNwarps = kNthreads / 32;
  constexpr int kWidth = kFattnHeadWidth;

  const std::uint32_t kv_head = blockIdx.x;
  const std::size_t first_row =
      static_cast<std::size_t>(Ncols1) * blockIdx.y;
  if (first_row >= token_count) return;
  const int active_rows = static_cast<int>(
      token_count - first_row < static_cast<std::size_t>(Ncols1)
          ? token_count - first_row
          : static_cast<std::size_t>(Ncols1));
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid % 32;
  const std::uint32_t first_head =
      kv_head * (config.query_heads / config.kv_heads);
  const std::size_t width = config.head_width;
  const std::size_t row_values =
      static_cast<std::size_t>(config.kv_heads) * width;
  const std::uint32_t half = config.rotary_width / 2;
  const float attn_scale = 1.0F / sqrtf(static_cast<float>(width));

  extern __shared__ unsigned char raw[];
  float* scratch = reinterpret_cast<float*>(raw);
  __nv_bfloat16* keys = reinterpret_cast<__nv_bfloat16*>(scratch + kWidth);
  __nv_bfloat16* values = keys + kFattnNbatchFa * kWidth;
  __half* q_f16 = reinterpret_cast<__half*>(values + kFattnNbatchFa * kWidth);
  __half* q_lo = q_f16 + kNcols * kWidth;
  float* scores = reinterpret_cast<float*>(
      DualF16 ? reinterpret_cast<__half*>(q_lo + kNcols * kWidth)
              : q_lo);
  float* qmax = scores + kNcols * kFattnNbatchFa;
  float* qden = qmax + kNcols;
  float* rescale = qden + kNcols;
  float* cparts = rescale + kNcols;

  const std::size_t last_position =
      start_position + first_row + static_cast<std::size_t>(active_rows) - 1U;

  for (int subgroup = 0; subgroup < kFattnSubgroups; ++subgroup) {
    for (int member = 0; member < kFattnNcols2; ++member) {
      const std::uint32_t query_head =
          first_head + static_cast<std::uint32_t>(subgroup * kFattnNcols2 +
                                                  member);
      for (int row = 0; row < Ncols1; ++row) {
        const int qcol = member * Ncols1 + row;
        const bool live = row < active_rows;
        const std::size_t qbase =
            live ? (first_row + static_cast<std::size_t>(row)) *
                         config.query_heads * width +
                     query_head * width
                 : 0;
        float local = 0.0F;
        if (live) {
          for (int dim = tid; dim < static_cast<int>(width); dim += kNthreads) {
            const float item = query[qbase + static_cast<std::size_t>(dim)];
            local = __fadd_rn(local, __fmul_rn(item, item));
          }
        }
        const float sum = fattn_block_sum(local, scratch, tid, kNthreads);
        const float inverse =
            1.0F / sqrtf(sum / static_cast<float>(width) + kFattnRmsEpsilon);
        for (int dim = tid; dim < static_cast<int>(width); dim += kNthreads) {
          scratch[dim] =
              live ? query[qbase + static_cast<std::size_t>(dim)] * inverse *
                         query_scale[dim]
                   : 0.0F;
        }
        __syncthreads();
        if (tid < static_cast<int>(half)) {
          const float first = scratch[tid];
          const float second = scratch[half + tid];
          const float exponent = static_cast<float>(tid * 2) /
                                 static_cast<float>(config.rotary_width);
          const float angle =
              live ? static_cast<float>(start_position + first_row +
                                        static_cast<std::size_t>(row)) /
                         powf(kFattnRopeTheta, exponent)
                   : 0.0F;
          const float cosine = cosf(angle);
          const float sine = sinf(angle);
          scratch[tid] = first * cosine - second * sine;
          scratch[half + tid] = second * cosine + first * sine;
        }
        __syncthreads();
        for (int dim = tid; dim < static_cast<int>(width); dim += kNthreads) {
          const float value = scratch[dim];
          const __half hi = __float2half_rn(value);
          q_f16[qcol * kWidth + dim] = hi;
          if constexpr (DualF16) {
            q_lo[qcol * kWidth + dim] =
                __float2half_rn(value - __half2float(hi));
          }
          if (token_count == 1 && live)
            normalized_query[query_head * width + dim] = value;
          if (live)
            output[qbase + static_cast<std::size_t>(dim)] = 0.0F;
        }
        __syncthreads();
      }
    }

    for (int qcol = tid; qcol < kNcols; qcol += kNthreads) {
      qmax[qcol] = -INFINITY;
      qden[qcol] = 0.0F;
    }
    __syncthreads();

    for (std::size_t tile = 0; tile <= last_position; tile += kFattnNbatchFa) {
      const int rows = static_cast<int>(
          (last_position + 1 - tile) < static_cast<std::size_t>(kFattnNbatchFa)
              ? (last_position + 1 - tile)
              : static_cast<std::size_t>(kFattnNbatchFa));
      for (int row = 0; row < kFattnNbatchFa; ++row) {
        const bool in_tile = row < rows;
        const std::size_t absolute = tile + static_cast<std::size_t>(row);
        const __nv_bfloat16* ksrc = nullptr;
        const __nv_bfloat16* vsrc = nullptr;
        if (in_tile) {
          if (absolute < start_position) {
            const std::size_t packed = attention_kv_physical_index(
                absolute, kv_head, 0, config.capacity, config.head_width);
            ksrc = committed_key + packed;
            vsrc = committed_value + packed;
          } else {
            ksrc = candidate_key + (absolute - start_position) * row_values +
                   kv_head * width;
            vsrc = candidate_value + (absolute - start_position) * row_values +
                   kv_head * width;
          }
        }
        for (int dim = tid; dim < static_cast<int>(width); dim += kNthreads) {
          keys[row * kWidth + dim] =
              in_tile ? ksrc[dim] : __float2bfloat16_rn(0.0F);
          values[row * kWidth + dim] =
              in_tile ? vsrc[dim] : __float2bfloat16_rn(0.0F);
        }
      }
      __syncthreads();

      for (int m0 = 0; m0 < kNcols; m0 += 16) {
        for (int n0 = 0; n0 < kFattnNbatchFa; n0 += 8) {
          float cfrag[4] = {0.0F, 0.0F, 0.0F, 0.0F};
          float cfrag_lo[4] = {0.0F, 0.0F, 0.0F, 0.0F};
          for (int k0 = warp * 16; k0 < kWidth; k0 += kNwarps * 16) {
            std::uint32_t a[4];
            std::uint32_t a_lo[4] = {0, 0, 0, 0};
            std::uint32_t b[2];
#pragma unroll
            for (int item = 0; item < 4; ++item) {
              const int i = mma::tile16x8_half2_i(lane, item);
              const int j = mma::tile16x8_half2_j(lane, item);
              const int qcol = m0 + i;
              __half pair[2] = {__float2half_rn(0.0F), __float2half_rn(0.0F)};
              __half pair_lo[2] = {__float2half_rn(0.0F),
                                   __float2half_rn(0.0F)};
              if (qcol < kNcols) {
                const int offset = qcol * kWidth + k0 + j * 2;
                pair[0] = q_f16[offset];
                pair[1] = q_f16[offset + 1];
                if constexpr (DualF16) {
                  pair_lo[0] = q_lo[offset];
                  pair_lo[1] = q_lo[offset + 1];
                }
              }
              a[item] = *reinterpret_cast<std::uint32_t*>(pair);
              if constexpr (DualF16) {
                a_lo[item] = *reinterpret_cast<std::uint32_t*>(pair_lo);
              }
            }
#pragma unroll
            for (int item = 0; item < 2; ++item) {
              const int n = mma::tile8x8_i(lane, item);
              const int j = mma::tile8x8_j(lane, item);
              const int krow = n0 + n;
              __half pair[2] = {__float2half_rn(0.0F), __float2half_rn(0.0F)};
              if (krow < rows) {
                pair[0] = __float2half_rn(__bfloat162float(
                    keys[krow * kWidth + k0 + j * 2]));
                pair[1] = __float2half_rn(__bfloat162float(
                    keys[krow * kWidth + k0 + j * 2 + 1]));
              }
              b[item] = *reinterpret_cast<std::uint32_t*>(pair);
            }
            mma::mma_m16n8k16_f16_f32(cfrag, a, b);
            if constexpr (DualF16) {
              mma::mma_m16n8k16_f16_f32(cfrag_lo, a_lo, b);
            }
          }
#pragma unroll
          for (int item = 0; item < 4; ++item)
            cparts[(warp * 32 + lane) * 4 + item] =
                cfrag[item] + cfrag_lo[item];
          __syncthreads();
          if (warp == 0) {
            float reduced[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            for (int other = 0; other < kNwarps; ++other) {
#pragma unroll
              for (int item = 0; item < 4; ++item)
                reduced[item] += cparts[(other * 32 + lane) * 4 + item];
            }
#pragma unroll
            for (int item = 0; item < 4; ++item) {
              const int i = mma::tile16x8_i(lane, item);
              const int j = mma::tile16x8_j(lane, item);
              const int qcol = m0 + i;
              const int krow = n0 + j;
              if (qcol < kNcols && krow < kFattnNbatchFa)
                scores[qcol * kFattnNbatchFa + krow] =
                    reduced[item] * attn_scale;
            }
          }
          __syncthreads();
        }
      }

      for (int qcol = tid; qcol < kNcols; qcol += kNthreads) {
        const int row = qcol % Ncols1;
        const std::size_t qpos =
            start_position + first_row + static_cast<std::size_t>(row);
        float tile_max = -INFINITY;
        for (int krow = 0; krow < rows; ++krow) {
          const std::size_t absolute = tile + static_cast<std::size_t>(krow);
          float score = scores[qcol * kFattnNbatchFa + krow];
          if (row >= active_rows || absolute > qpos) score = -INFINITY;
          scores[qcol * kFattnNbatchFa + krow] = score;
          tile_max = fmaxf(tile_max, score);
        }
        const float old_max = qmax[qcol];
        const float new_max = fmaxf(old_max, tile_max);
        const float factor =
            old_max == -INFINITY ? 0.0F : expf(old_max - new_max);
        float denominator = qden[qcol] * factor;
        for (int krow = 0; krow < rows; ++krow) {
          const float score = scores[qcol * kFattnNbatchFa + krow];
          const float weight =
              score == -INFINITY ? 0.0F : expf(score - new_max);
          scores[qcol * kFattnNbatchFa + krow] = weight;
          denominator += weight;
        }
        qmax[qcol] = new_max;
        qden[qcol] = denominator;
        rescale[qcol] = factor;
      }
      __syncthreads();

      for (int member = 0; member < kFattnNcols2; ++member) {
        const std::uint32_t query_head =
            first_head + static_cast<std::uint32_t>(subgroup * kFattnNcols2 +
                                                    member);
        for (int row = 0; row < active_rows; ++row) {
          const int qcol = member * Ncols1 + row;
          const std::size_t qbase =
              (first_row + static_cast<std::size_t>(row)) *
                  config.query_heads * width +
              query_head * width;
          const float factor = rescale[qcol];
          for (int dim = tid; dim < static_cast<int>(width); dim += kNthreads) {
            float total = output[qbase + static_cast<std::size_t>(dim)] * factor;
            for (int krow = 0; krow < rows; ++krow) {
              total += scores[qcol * kFattnNbatchFa + krow] *
                       __bfloat162float(values[krow * kWidth + dim]);
            }
            output[qbase + static_cast<std::size_t>(dim)] = total;
          }
        }
      }
      __syncthreads();
    }

    for (int member = 0; member < kFattnNcols2; ++member) {
      const std::uint32_t query_head =
          first_head + static_cast<std::uint32_t>(subgroup * kFattnNcols2 +
                                                  member);
      for (int row = 0; row < active_rows; ++row) {
        const int qcol = member * Ncols1 + row;
        const std::size_t qbase =
            (first_row + static_cast<std::size_t>(row)) * config.query_heads *
                width +
            query_head * width;
        const float denominator = qden[qcol];
        for (int dim = tid; dim < static_cast<int>(width); dim += kNthreads) {
          const float g = gate[qbase + static_cast<std::size_t>(dim)];
          const float sigmoid = g >= 0.0F ? 1.0F / (1.0F + expf(-g))
                                          : expf(g) / (1.0F + expf(g));
          const float attn =
              denominator == 0.0F
                  ? 0.0F
                  : output[qbase + static_cast<std::size_t>(dim)] / denominator;
          output[qbase + static_cast<std::size_t>(dim)] = attn * sigmoid;
        }
      }
    }
    __syncthreads();
  }
}

template <int Ncols1, bool DualF16>
inline cudaError_t launch_fattn_mma_quality_typed(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* query_scale,
    const float* gate, const __nv_bfloat16* committed_key,
    const __nv_bfloat16* committed_value, const __nv_bfloat16* candidate_key,
    const __nv_bfloat16* candidate_value, float* output,
    float* normalized_query, cudaStream_t stream) noexcept {
  constexpr int kNthreads = Ncols1 <= 8 ? 64 : 128;
  const std::size_t shared = fattn_quality_shared_bytes(Ncols1, DualF16);
  dim3 grid(config.kv_heads,
            static_cast<unsigned>((token_count + Ncols1 - 1) / Ncols1), 1);
  cudaError_t error = cudaFuncSetAttribute(
      fattn_mma_quality_kernel<Ncols1, DualF16>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
  if (error != cudaSuccess) return error;
  fattn_mma_quality_kernel<Ncols1, DualF16>
      <<<grid, kNthreads, shared, stream>>>(
          config, start_position, token_count, query, query_scale, gate,
          committed_key, committed_value, candidate_key, candidate_value,
          output, normalized_query);
  return cudaPeekAtLastError();
}

inline cudaError_t launch_fattn_mma_quality(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* query_scale,
    const float* gate, const __nv_bfloat16* committed_key,
    const __nv_bfloat16* committed_value, const __nv_bfloat16* candidate_key,
    const __nv_bfloat16* candidate_value, float* output,
    float* normalized_query, int ncols1, cudaStream_t stream) noexcept {
  switch (ncols1) {
    case 8:
      return launch_fattn_mma_quality_typed<8, true>(
          config, start_position, token_count, query, query_scale, gate,
          committed_key, committed_value, candidate_key, candidate_value,
          output, normalized_query, stream);
    case 16:
      return launch_fattn_mma_quality_typed<16, true>(
          config, start_position, token_count, query, query_scale, gate,
          committed_key, committed_value, candidate_key, candidate_value,
          output, normalized_query, stream);
    case 32:
      return launch_fattn_mma_quality_typed<32, false>(
          config, start_position, token_count, query, query_scale, gate,
          committed_key, committed_value, candidate_key, candidate_value,
          output, normalized_query, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

inline int fattn_occupancy_for(int ncols1) noexcept {
  int occupancy = 0;
  cudaError_t error = cudaErrorInvalidValue;
  const int threads = fattn_nthreads_for_ncols1(ncols1);
  const bool dual = ncols1 != 32;
  const std::size_t shared = fattn_quality_shared_bytes(ncols1, dual);
  if (ncols1 == 8) {
    error = cudaFuncSetAttribute(
        fattn_mma_quality_kernel<8, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
    if (error == cudaSuccess)
      error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, fattn_mma_quality_kernel<8, true>, threads, shared);
  } else if (ncols1 == 16) {
    error = cudaFuncSetAttribute(
        fattn_mma_quality_kernel<16, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
    if (error == cudaSuccess)
      error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, fattn_mma_quality_kernel<16, true>, threads, shared);
  } else if (ncols1 == 32) {
    error = cudaFuncSetAttribute(
        fattn_mma_quality_kernel<32, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
    if (error == cudaSuccess)
      error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, fattn_mma_quality_kernel<32, false>, threads, shared);
  }
  return error == cudaSuccess ? occupancy : 0;
}

}  // namespace qw38::cuda
