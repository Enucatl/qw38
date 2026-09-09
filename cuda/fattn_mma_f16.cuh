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

// OPT-026 A/B winner. Legal values: baseline, occ2, stream_k.
constexpr char kSelectedFattnPath[] = "stream_k";

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

template <int Ncols1, bool DualF16, int Occupancy, int KvParts>
__global__ void __launch_bounds__(Ncols1 <= 8 ? 64 : 128, Occupancy)
fattn_mma_quality_kernel(
    AttentionConfig config, std::size_t start_position, std::size_t token_count,
    const float* query, const float* query_scale, const float* gate,
    const __nv_bfloat16* committed_key, const __nv_bfloat16* committed_value,
    const __nv_bfloat16* candidate_key, const __nv_bfloat16* candidate_value,
    float* output, float* normalized_query, float* partial, float* meta) {
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

  const int kv_z = KvParts > 1 ? static_cast<int>(blockIdx.z) : 0;
  float* vkq = (KvParts > 1 && kv_z > 0) ? partial : output;
  const std::size_t last_position =
      start_position + first_row + static_cast<std::size_t>(active_rows) - 1U;
  const std::size_t kv_span = last_position + 1U;
  const std::size_t kv_tiles =
      (kv_span + static_cast<std::size_t>(kFattnNbatchFa) - 1U) /
      static_cast<std::size_t>(kFattnNbatchFa);
  const std::size_t tiles_per_part =
      KvParts > 1 ? (kv_tiles + static_cast<std::size_t>(KvParts) - 1U) /
                        static_cast<std::size_t>(KvParts)
                  : kv_tiles;
  const std::size_t kv_begin =
      tiles_per_part * static_cast<std::size_t>(kv_z) *
      static_cast<std::size_t>(kFattnNbatchFa);
  const std::size_t kv_end = kv_begin + tiles_per_part *
                                           static_cast<std::size_t>(kFattnNbatchFa);
  const std::size_t kv_stop = kv_end < kv_span ? kv_end : kv_span;

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
            vkq[qbase + static_cast<std::size_t>(dim)] = 0.0F;
        }
        __syncthreads();
      }
    }

    for (int qcol = tid; qcol < kNcols; qcol += kNthreads) {
      qmax[qcol] = -INFINITY;
      qden[qcol] = 0.0F;
    }
    __syncthreads();

    for (std::size_t tile = kv_begin; tile < kv_stop; tile += kFattnNbatchFa) {
      const int rows = static_cast<int>(
          (kv_span - tile) < static_cast<std::size_t>(kFattnNbatchFa)
              ? (kv_span - tile)
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
            float total = vkq[qbase + static_cast<std::size_t>(dim)] * factor;
            for (int krow = 0; krow < rows; ++krow) {
              total += scores[qcol * kFattnNbatchFa + krow] *
                       __bfloat162float(values[krow * kWidth + dim]);
            }
            vkq[qbase + static_cast<std::size_t>(dim)] = total;
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
        if constexpr (KvParts == 1) {
          for (int dim = tid; dim < static_cast<int>(width); dim += kNthreads) {
            const float g = gate[qbase + static_cast<std::size_t>(dim)];
            const float sigmoid = g >= 0.0F ? 1.0F / (1.0F + expf(-g))
                                            : expf(g) / (1.0F + expf(g));
            const float attn =
                denominator == 0.0F
                    ? 0.0F
                    : vkq[qbase + static_cast<std::size_t>(dim)] / denominator;
            output[qbase + static_cast<std::size_t>(dim)] = attn * sigmoid;
          }
        } else if (tid == 0) {
          const std::size_t meta_index =
              (static_cast<std::size_t>(kv_z) * token_count + first_row +
               static_cast<std::size_t>(row)) *
                  config.query_heads +
              query_head;
          meta[meta_index * 2U] = qmax[qcol];
          meta[meta_index * 2U + 1U] = denominator;
        }
      }
    }
    __syncthreads();
  }
}

__global__ void fattn_stream_k_combine_kernel(
    AttentionConfig config, std::size_t token_count, const float* gate,
    float* output, const float* partial, const float* meta) {
  const std::size_t width = config.head_width;
  const std::size_t values =
      token_count * static_cast<std::size_t>(config.query_heads) * width;
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) *
                                static_cast<std::size_t>(blockDim.x) +
                            static_cast<std::size_t>(threadIdx.x);
  if (index >= values) return;
  const std::size_t row_values =
      static_cast<std::size_t>(config.query_heads) * width;
  const std::size_t row = index / row_values;
  const std::size_t in_row = index % row_values;
  const std::uint32_t head = static_cast<std::uint32_t>(in_row / width);
  const std::size_t meta0 =
      (row * static_cast<std::size_t>(config.query_heads) + head) * 2U;
  const std::size_t meta1 =
      ((token_count + row) * static_cast<std::size_t>(config.query_heads) +
       head) *
      2U;
  const float max0 = meta[meta0];
  const float den0 = meta[meta0 + 1U];
  const float max1 = meta[meta1];
  const float den1 = meta[meta1 + 1U];
  const float new_max = fmaxf(max0, max1);
  const float scale0 = max0 == -INFINITY ? 0.0F : expf(max0 - new_max);
  const float scale1 = max1 == -INFINITY ? 0.0F : expf(max1 - new_max);
  const float vkq = output[index] * scale0 + partial[index] * scale1;
  const float denominator = den0 * scale0 + den1 * scale1;
  const float g = gate[index];
  const float sigmoid =
      g >= 0.0F ? 1.0F / (1.0F + expf(-g)) : expf(g) / (1.0F + expf(g));
  const float attn = denominator == 0.0F ? 0.0F : vkq / denominator;
  output[index] = attn * sigmoid;
}

template <int Ncols1, bool DualF16, int Occupancy, int KvParts>
inline cudaError_t launch_fattn_mma_quality_typed(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* query_scale,
    const float* gate, const __nv_bfloat16* committed_key,
    const __nv_bfloat16* committed_value, const __nv_bfloat16* candidate_key,
    const __nv_bfloat16* candidate_value, float* output,
    float* normalized_query, float* partial, float* meta,
    cudaStream_t stream) noexcept {
  constexpr int kNthreads = Ncols1 <= 8 ? 64 : 128;
  const std::size_t shared = fattn_quality_shared_bytes(Ncols1, DualF16);
  dim3 grid(config.kv_heads,
            static_cast<unsigned>((token_count + Ncols1 - 1) / Ncols1),
            KvParts > 1 ? 2U : 1U);
  cudaError_t error = cudaFuncSetAttribute(
      fattn_mma_quality_kernel<Ncols1, DualF16, Occupancy, KvParts>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
  if (error != cudaSuccess) return error;
  fattn_mma_quality_kernel<Ncols1, DualF16, Occupancy, KvParts>
      <<<grid, kNthreads, shared, stream>>>(
          config, start_position, token_count, query, query_scale, gate,
          committed_key, committed_value, candidate_key, candidate_value,
          output, normalized_query, partial, meta);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess || KvParts == 1) return error;
  const std::size_t values =
      token_count * static_cast<std::size_t>(config.query_heads) *
      config.head_width;
  const unsigned threads = 256;
  const unsigned blocks =
      static_cast<unsigned>((values + threads - 1) / threads);
  fattn_stream_k_combine_kernel<<<blocks, threads, 0, stream>>>(
      config, token_count, gate, output, partial, meta);
  return cudaPeekAtLastError();
}

inline cudaError_t launch_fattn_mma_quality_path(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* query_scale,
    const float* gate, const __nv_bfloat16* committed_key,
    const __nv_bfloat16* committed_value, const __nv_bfloat16* candidate_key,
    const __nv_bfloat16* candidate_value, float* output,
    float* normalized_query, int ncols1, const char* path, float* partial,
    float* meta, cudaStream_t stream) noexcept {
  const bool stream_k = path != nullptr && path[0] == 's';
  const bool occ2 = stream_k || (path != nullptr && path[0] == 'o');
  if (stream_k && (partial == nullptr || meta == nullptr))
    return cudaErrorInvalidValue;
  if (ncols1 == 16 && stream_k) {
    return launch_fattn_mma_quality_typed<16, true, 2, 2>(
        config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value, output,
        normalized_query, partial, meta, stream);
  }
  if (ncols1 == 16 && occ2) {
    return launch_fattn_mma_quality_typed<16, true, 2, 1>(
        config, start_position, token_count, query, query_scale, gate,
        committed_key, committed_value, candidate_key, candidate_value, output,
        normalized_query, nullptr, nullptr, stream);
  }
  switch (ncols1) {
    case 8:
      return launch_fattn_mma_quality_typed<8, true, 1, 1>(
          config, start_position, token_count, query, query_scale, gate,
          committed_key, committed_value, candidate_key, candidate_value,
          output, normalized_query, nullptr, nullptr, stream);
    case 16:
      return launch_fattn_mma_quality_typed<16, true, 1, 1>(
          config, start_position, token_count, query, query_scale, gate,
          committed_key, committed_value, candidate_key, candidate_value,
          output, normalized_query, nullptr, nullptr, stream);
    case 32:
      return launch_fattn_mma_quality_typed<32, false, 1, 1>(
          config, start_position, token_count, query, query_scale, gate,
          committed_key, committed_value, candidate_key, candidate_value,
          output, normalized_query, nullptr, nullptr, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

inline cudaError_t launch_fattn_mma_quality(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* query_scale,
    const float* gate, const __nv_bfloat16* committed_key,
    const __nv_bfloat16* committed_value, const __nv_bfloat16* candidate_key,
    const __nv_bfloat16* candidate_value, float* output,
    float* normalized_query, int ncols1, cudaStream_t stream) noexcept {
  const char* path = kSelectedFattnPath;
  if (path[0] == 's') path = "occ2";
  return launch_fattn_mma_quality_path(
      config, start_position, token_count, query, query_scale, gate,
      committed_key, committed_value, candidate_key, candidate_value, output,
      normalized_query, ncols1, path, nullptr, nullptr, stream);
}

inline cudaError_t launch_fattn_mma_stream_k(
    const AttentionConfig& config, std::size_t start_position,
    std::size_t token_count, const float* query, const float* query_scale,
    const float* gate, const __nv_bfloat16* committed_key,
    const __nv_bfloat16* committed_value, const __nv_bfloat16* candidate_key,
    const __nv_bfloat16* candidate_value, float* output,
    float* normalized_query, float* partial, float* meta,
    cudaStream_t stream) noexcept {
  return launch_fattn_mma_quality_path(
      config, start_position, token_count, query, query_scale, gate,
      committed_key, committed_value, candidate_key, candidate_value, output,
      normalized_query, kSelectedAttentionMmaQueryRows, "stream_k", partial,
      meta, stream);
}

template <int Ncols1, bool DualF16, int Occupancy, int KvParts>
inline int fattn_occupancy_typed() noexcept {
  int occupancy = 0;
  const int threads = fattn_nthreads_for_ncols1(Ncols1);
  const std::size_t shared = fattn_quality_shared_bytes(Ncols1, DualF16);
  cudaError_t error = cudaFuncSetAttribute(
      fattn_mma_quality_kernel<Ncols1, DualF16, Occupancy, KvParts>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
  if (error == cudaSuccess)
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy,
        fattn_mma_quality_kernel<Ncols1, DualF16, Occupancy, KvParts>, threads,
        shared);
  return error == cudaSuccess ? occupancy : 0;
}

inline int fattn_occupancy_for(int ncols1) noexcept {
  if (ncols1 == 8) return fattn_occupancy_typed<8, true, 1, 1>();
  if (ncols1 == 16) {
    if (kSelectedFattnPath[0] == 's' || kSelectedFattnPath[0] == 'o')
      return fattn_occupancy_typed<16, true, 2, 1>();
    return fattn_occupancy_typed<16, true, 1, 1>();
  }
  if (ncols1 == 32) return fattn_occupancy_typed<32, false, 1, 1>();
  return 0;
}

inline int fattn_occupancy_for_path(const char* path) noexcept {
  if (path != nullptr && path[0] == 's')
    return fattn_occupancy_typed<16, true, 2, 2>();
  if (path != nullptr && path[0] == 'o')
    return fattn_occupancy_typed<16, true, 2, 1>();
  return fattn_occupancy_typed<16, true, 1, 1>();
}

}  // namespace qw38::cuda
