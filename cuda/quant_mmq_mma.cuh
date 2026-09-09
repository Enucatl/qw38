#pragma once

// Q4_K / Q6_K / Q8_0 prompt MMQ with Ampere-style MMA tiles for Quartz.
// OPT-017 Rank-1 fused Q8_0 (1D 256 threads, K-step 32, SRAM Q8_1 staging,
// no q8_workspace) stays callable. OPT-018 production Q4_K/Q6_K is the
// quality kernel: MMQ_ITER_K=256, packed load-tiles, Q8_1 MMQ Y tiles in
// the existing prompt workspace, block dim3(32, 8). OPT-022 production
// mixer Q8_0 (keep) is the same quality stack with D4 quantize_mmq_q8_1
// and packed Q8_0 load-tiles; Rank-1 remains the visible fused reference
// and the reject restore path. Output is token-major FP32
// [prompt_rows, output_rows]. Production Q8_0 is launched via
// launch_q8_mmq_bf16, not launch_quant_mmq.
//
// Provenance: llama.cpp mmq.cuh / mma.cuh / mmq-config-ampere.cuh /
// mmq-load-tiles.cuh / mmq-vec-dot.cuh / quantize.cu at
// cc83d7b4824f73cfdda4dfbb47ee39804f71b328 (MIT, The ggml authors).
// ds4 cuda/mmq is the ggml-free launcher pattern only; this file does not
// copy ../ds4/cuda/mmq/ and does not include ggml headers.

#include "mma.cuh"
#include "quant_mmv.h"

#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace qw38::cuda {
namespace {

constexpr int kMmaThreads = 256;
constexpr int kMmaOutputRows = 128;
constexpr int kMmaK = 32;
constexpr std::size_t kMmaQ4Bytes = 144;
constexpr std::size_t kMmaQ6Bytes = 210;
constexpr std::size_t kMmaQ8Bytes = 34;
constexpr std::size_t kMmaBlockValues = 256;
constexpr std::size_t kMmaQ8Values = 32;

__device__ float mma_read_half(const std::uint8_t* bytes) {
  const unsigned short bits = static_cast<unsigned short>(bytes[0]) |
                              (static_cast<unsigned short>(bytes[1]) << 8U);
  return __half2float(__ushort_as_half(bits));
}

__device__ void mma_q4_scale_min(const std::uint8_t* packed, int index,
                                 int* scale, int* minimum) {
  if (index < 4) {
    *scale = packed[index] & 63;
    *minimum = packed[index + 4] & 63;
    return;
  }
  *scale = (packed[index + 4] & 15) | ((packed[index - 4] >> 6) << 4);
  *minimum = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4);
}

__device__ int mma_pack_i8x4(int v0, int v1, int v2, int v3) {
  return (v0 & 0xFF) | ((v1 & 0xFF) << 8) | ((v2 & 0xFF) << 16) |
         ((v3 & 0xFF) << 24);
}

__device__ void mma_decode_q4_group32(const std::uint8_t* block, int index0,
                                      int values[32], float* d_scale,
                                      float* neg_dmin) {
  const int group = index0 / 32;
  int scale = 0;
  int minimum = 0;
  mma_q4_scale_min(block + 4, group, &scale, &minimum);
  const int group64 = index0 / 64;
  const int high = (index0 % 64) / 32;
  const std::uint8_t* qs = block + 16 + group64 * 32;
#pragma unroll
  for (int lane = 0; lane < 32; ++lane) {
    const int packed = qs[lane];
    values[lane] = high == 0 ? (packed & 15) : (packed >> 4);
  }
  *d_scale = mma_read_half(block) * static_cast<float>(scale);
  *neg_dmin = -mma_read_half(block + 2) * static_cast<float>(minimum);
}

__device__ void mma_decode_q8_group32(const std::uint8_t* block, int values[32],
                                      float* d_scale) {
  *d_scale = mma_read_half(block);
#pragma unroll
  for (int lane = 0; lane < 32; ++lane) {
    values[lane] =
        static_cast<int>(static_cast<std::int8_t>(block[2 + lane]));
  }
}

__device__ void mma_decode_q6_group32(const std::uint8_t* block, int index0,
                                      int values[32], float scales[2]) {
  const int half = index0 / 128;
  const int within = index0 % 128;
  const int group0 = within / 32;
#pragma unroll
  for (int lane = 0; lane < 32; ++lane) {
    const int group = group0;
    const int low_offset = half * 64;
    const int high_offset = 128 + half * 32;
    const std::uint8_t low =
        block[low_offset + lane + ((group & 1) != 0 ? 32 : 0)];
    const int low_four = group < 2 ? low & 15 : low >> 4;
    const int high_two = (block[high_offset + lane] >> (group * 2)) & 3;
    values[lane] = (low_four | (high_two << 4)) - 32;
  }
  const float d = mma_read_half(block + 208);
#pragma unroll
  for (int part = 0; part < 2; ++part) {
    const int lane = part * 16;
    const std::uint8_t scale_byte =
        block[192 + half * 8 + (lane / 16) + group0 * 2];
    const int scale = scale_byte < 128 ? static_cast<int>(scale_byte)
                                       : static_cast<int>(scale_byte) - 256;
    scales[part] = d * static_cast<float>(scale);
  }
}

__device__ float mma_warp_max(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFFU, value, offset, 32));
  }
  return value;
}

__device__ float mma_warp_sum(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xFFFFFFFFU, value, offset, 32);
  }
  return value;
}

template <QuantKind Kind, int PromptTile>
__global__ void __launch_bounds__(kMmaThreads, 2)
    quant_mmq_mma_kernel(const std::uint8_t* weights, std::size_t output_rows,
                         std::size_t columns, const __nv_bfloat16* prompt,
                         std::size_t prompt_rows, float* output) {
  constexpr int kNTiles = PromptTile / 8;
  const int warp = threadIdx.x / mma::kWarpSize;
  const int lane = threadIdx.x & (mma::kWarpSize - 1);
  extern __shared__ std::uint8_t shared[];
  int* x_qs = reinterpret_cast<int*>(shared);
  float2* x_dm = reinterpret_cast<float2*>(x_qs + kMmaOutputRows * 8);
  float2* x_sc = reinterpret_cast<float2*>(x_dm + kMmaOutputRows);
  int* y_qs = reinterpret_cast<int*>(x_sc + kMmaOutputRows);
  float2* y_ds = reinterpret_cast<float2*>(y_qs + PromptTile * 8);

  const std::size_t out0 =
      static_cast<std::size_t>(blockIdx.x) * kMmaOutputRows;
  const std::size_t prompt0 =
      static_cast<std::size_t>(blockIdx.y) * PromptTile;
  constexpr std::size_t kWeightBytes =
      Kind == QuantKind::kQ4K ? kMmaQ4Bytes
      : Kind == QuantKind::kQ6K ? kMmaQ6Bytes
                               : kMmaQ8Bytes;
  constexpr std::size_t kWeightValues =
      Kind == QuantKind::kQ8_0 ? kMmaQ8Values : kMmaBlockValues;
  const std::size_t row_stride = (columns / kWeightValues) * kWeightBytes;

  float acc[kNTiles][4];
#pragma unroll
  for (int nt = 0; nt < kNTiles; ++nt) {
#pragma unroll
    for (int l = 0; l < 4; ++l) acc[nt][l] = 0.0F;
  }

  for (std::size_t k = 0; k < columns; k += kMmaK) {
    const int load_row = threadIdx.x % kMmaOutputRows;
    const int load_half = threadIdx.x / kMmaOutputRows;
    const std::size_t global_row = out0 + static_cast<std::size_t>(load_row);
    int decoded[32];
    float d_scale = 0.0F;
    float neg_dmin = 0.0F;
    float q6_scales[2] = {0.0F, 0.0F};
    if (global_row < output_rows) {
      const std::uint8_t* block =
          weights + global_row * row_stride +
          (k / kWeightValues) * kWeightBytes;
      if constexpr (Kind == QuantKind::kQ8_0) {
        mma_decode_q8_group32(block, decoded, &d_scale);
      } else if constexpr (Kind == QuantKind::kQ4K) {
        mma_decode_q4_group32(block, static_cast<int>(k % kWeightValues),
                              decoded, &d_scale, &neg_dmin);
      } else {
        mma_decode_q6_group32(block, static_cast<int>(k % kWeightValues),
                              decoded, q6_scales);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 32; ++i) decoded[i] = 0;
    }
    const int pack0 = load_half * 4;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const int base = (pack0 + p) * 4;
      x_qs[load_row * 8 + pack0 + p] = mma_pack_i8x4(
          decoded[base], decoded[base + 1], decoded[base + 2],
          decoded[base + 3]);
    }
    if (load_half == 0) {
      x_dm[load_row] = make_float2(d_scale, neg_dmin);
      x_sc[load_row] = make_float2(q6_scales[0], q6_scales[1]);
    }

    for (int prow = warp; prow < PromptTile; prow += 8) {
      const std::size_t global_prompt = prompt0 + static_cast<std::size_t>(prow);
      float value = 0.0F;
      if (global_prompt < prompt_rows && k + static_cast<std::size_t>(lane) <
                                             columns) {
        value = __bfloat162float(
            prompt[global_prompt * columns + k + static_cast<std::size_t>(lane)]);
      }
      const float maximum = mma_warp_max(fabsf(value));
      const float scale = maximum == 0.0F ? 0.0F : maximum / 127.0F;
      const int q = scale == 0.0F ? 0 : __float2int_rn(value / scale);
      const int q8 = q & 0xFF;
      const int base_lane = lane & ~3;
      int packed = q8;
      packed |= (__shfl_sync(0xFFFFFFFFU, q8, base_lane + 1, 32) & 0xFF) << 8;
      packed |= (__shfl_sync(0xFFFFFFFFU, q8, base_lane + 2, 32) & 0xFF) << 16;
      packed |= (__shfl_sync(0xFFFFFFFFU, q8, base_lane + 3, 32) & 0xFF) << 24;
      if ((lane & 3) == 0) {
        y_qs[prow * 8 + (lane / 4)] = packed;
      }
      const float qsum = mma_warp_sum(static_cast<float>(q));
      if (lane == 0) y_ds[prow] = make_float2(scale, scale * qsum);
    }
    __syncthreads();

    const int row0 = warp * 16;
    int A[4];
    {
      const int* xs =
          x_qs + (row0 + (lane % 16)) * 8 + (lane / 16) * 4;
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                   : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3])
                   : "l"(xs));
    }

#pragma unroll
    for (int nt = 0; nt < kNTiles; ++nt) {
      int B[2];
#pragma unroll
      for (int l = 0; l < 2; ++l) {
        const int n = mma::tile8x8_i(lane, l);
        const int j = mma::tile8x8_j(lane, l);
        B[l] = y_qs[(nt * 8 + n) * 8 + j];
      }
      if constexpr (Kind == QuantKind::kQ6K) {
        int A0[2];
        int A1[2];
        {
          const int* xs0 = x_qs + (row0 + (lane % 16)) * 8;
          asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                       : "=r"(A0[0]), "=r"(A0[1])
                       : "l"(xs0));
          const int* xs1 = x_qs + (row0 + (lane % 16)) * 8 + 4;
          asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                       : "=r"(A1[0]), "=r"(A1[1])
                       : "l"(xs1));
        }
        int C0[4] = {0, 0, 0, 0};
        int C1[4] = {0, 0, 0, 0};
        mma::mma_m16n8k16_s8(C0, A0, B[0]);
        mma::mma_m16n8k16_s8(C1, A1, B[1]);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
          const int i = mma::tile16x8_i(lane, l);
          const int j = mma::tile16x8_j(lane, l);
          const float2 sc = x_sc[row0 + i];
          const float2 dsB = y_ds[nt * 8 + j];
          acc[nt][l] += dsB.x * (sc.x * static_cast<float>(C0[l]) +
                                 sc.y * static_cast<float>(C1[l]));
        }
      } else {
        int C[4] = {0, 0, 0, 0};
        mma::mma_m16n8k32_s8(C, A, B);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
          const int i = mma::tile16x8_i(lane, l);
          const int j = mma::tile16x8_j(lane, l);
          const float2 dmA = x_dm[row0 + i];
          const float2 dsB = y_ds[nt * 8 + j];
          if constexpr (Kind == QuantKind::kQ8_0) {
            acc[nt][l] += dmA.x * dsB.x * static_cast<float>(C[l]);
          } else {
            acc[nt][l] += dmA.x * dsB.x * static_cast<float>(C[l]) +
                          dmA.y * dsB.y;
          }
        }
      }
    }
    __syncthreads();
  }

  const int row0 = warp * 16;
#pragma unroll
  for (int nt = 0; nt < kNTiles; ++nt) {
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const int i = mma::tile16x8_i(lane, l);
      const int j = mma::tile16x8_j(lane, l);
      const std::size_t out_row = out0 + static_cast<std::size_t>(row0 + i);
      const std::size_t prompt_row =
          prompt0 + static_cast<std::size_t>(nt * 8 + j);
      if (out_row < output_rows && prompt_row < prompt_rows) {
        output[prompt_row * output_rows + out_row] = acc[nt][l];
      }
    }
  }
}

inline std::size_t q8_mma_mmq_shared_bytes(int prompt_tile) noexcept {
  return static_cast<std::size_t>(kMmaOutputRows) * 8 * sizeof(int) +
         static_cast<std::size_t>(kMmaOutputRows) * sizeof(float2) +
         static_cast<std::size_t>(kMmaOutputRows) * sizeof(float2) +
         static_cast<std::size_t>(prompt_tile) * 8 * sizeof(int) +
         static_cast<std::size_t>(prompt_tile) * sizeof(float2);
}

template <QuantKind Kind, int PromptTile>
cudaError_t launch_mma_tile(const std::uint8_t* weights,
                            std::size_t output_rows, std::size_t columns,
                            const __nv_bfloat16* prompt,
                            std::size_t prompt_rows, float* output,
                            cudaStream_t stream) noexcept {
  const dim3 grid(
      static_cast<unsigned int>((output_rows + kMmaOutputRows - 1) /
                                kMmaOutputRows),
      static_cast<unsigned int>((prompt_rows + PromptTile - 1) / PromptTile));
  const std::size_t shared = q8_mma_mmq_shared_bytes(PromptTile);
  quant_mmq_mma_kernel<Kind, PromptTile>
      <<<grid, kMmaThreads, shared, stream>>>(
          weights, output_rows, columns, prompt, prompt_rows, output);
  return cudaPeekAtLastError();
}

constexpr int kQualityI = 128;
constexpr int kQualityNwarps = 8;
constexpr int kMmqTileNeK = 32;
constexpr int kMmqTileYk = 36;
constexpr int kQi81 = 8;
constexpr int kQi80 = 8;
constexpr int kQi6K = 32;
constexpr int kQualitySramQ4 = 76;  // 2*32 + 2*32/QI8_1 + 4; llama.cpp Q8_1
constexpr int kQualitySramQ6 = 76;  // 2*32 + 32/QI6_K + 32/8 + 7; 76 % 8 == 4
constexpr int kQualitySramQ8 = 76;  // 2*32 + 2*32/QI8_0 + 4; llama.cpp Q8_0
constexpr int kQualitySramMax = kQualitySramQ4;
static_assert(kQualitySramQ4 % 8 == 4, "Q4 MMA SRAM stride must pad K%8==4");
static_assert(kQualitySramQ6 % 8 == 4, "Q6 MMA SRAM stride must pad K%8==4");
static_assert(kQualitySramQ8 % 8 == 4, "Q8 MMA SRAM stride must pad K%8==4");
constexpr int kQualityQuantThreads = 256;

__device__ int quality_get_int_b4(const std::uint8_t* bytes, int index) {
  return reinterpret_cast<const int*>(bytes)[index];
}

__device__ int quality_get_int_b2(const std::uint8_t* bytes, int index) {
  const std::uint16_t* words = reinterpret_cast<const std::uint16_t*>(bytes);
  return static_cast<int>(words[2 * index]) |
         (static_cast<int>(words[2 * index + 1]) << 16);
}

__device__ int unpack_scales_q45_K(const int* scales, int ksc) {
  return ((scales[(ksc % 2) + (ksc != 0)] >> (4 * (ksc & (ksc / 2)))) &
          0x0F0F0F0F) |
         ((scales[ksc / 2] >> (2 * (ksc % 2))) & 0x30303030);
}

constexpr std::size_t quality_pad_ints(std::size_t count) noexcept {
  return ((count + 255) / 256) * 256;
}

constexpr std::size_t quality_shared_ints(int prompt_tile) noexcept {
  return static_cast<std::size_t>(prompt_tile) +
         quality_pad_ints(static_cast<std::size_t>(prompt_tile) *
                           kMmqTileYk) +
         static_cast<std::size_t>(kQualityI) * kQualitySramMax;
}

constexpr int quality_rows_per_warp(int prompt_tile) {
  return prompt_tile >= 48 && prompt_tile % 16 == 0 ? 32 : 16;
}

template <QuantKind Kind>
__global__ void quantize_mmq_q8_1_bf16(const __nv_bfloat16* prompt,
                                        std::size_t prompt_rows,
                                        std::size_t columns, std::uint8_t* y) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x & 31;
  const std::size_t count = prompt_rows * columns;
  float value = index < count ? __bfloat162float(prompt[index]) : 0.0F;
  float maximum = fabsf(value);
  for (int offset = 16; offset > 0; offset /= 2) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xFFFFFFFFU, maximum, offset, 32));
  }
  const float scale = maximum == 0.0F ? 0.0F : maximum / 127.0F;
  float prequant_sum = 0.0F;
  if constexpr (Kind == QuantKind::kQ4K) {
    prequant_sum = value;
    for (int offset = 16; offset > 0; offset /= 2) {
      prequant_sum += __shfl_xor_sync(0xFFFFFFFFU, prequant_sum, offset, 32);
    }
  }
  const int q =
      scale == 0.0F ? 0 : static_cast<int>(roundf(value / scale));
  if (index >= count) return;
  const std::size_t row = index / columns;
  const std::size_t col = index % columns;
  const std::size_t k32 = col / 32;
  const std::size_t k_block = k32 / 4;
  const std::size_t group = k32 % 4;
  std::uint8_t* block =
      y + (k_block * prompt_rows + row) * (kMmqTileYk * sizeof(int));
  reinterpret_cast<std::int8_t*>(block + 16)[group * 32 + lane] =
      static_cast<std::int8_t>(q);
  if (lane == 0) {
    if constexpr (Kind == QuantKind::kQ4K) {
      reinterpret_cast<half2*>(block)[group] = make_half2(scale, prequant_sum);
    } else {
      reinterpret_cast<float*>(block)[group] = scale;
    }
  }
}

template <QuantKind Kind, int PromptTile, bool Fallback>
__global__ void __launch_bounds__(256, 1) quant_mmq_mma_quality_kernel(
    const std::uint8_t* weights, std::size_t output_rows, std::size_t columns,
    const int* y, std::size_t prompt_rows, float* output) {
  constexpr int kNtx = quality_rows_per_warp(PromptTile) / 16;
  constexpr int kRowsPerWarp = quality_rows_per_warp(PromptTile);
  constexpr int kSram =
      Kind == QuantKind::kQ4K ? kQualitySramQ4
      : Kind == QuantKind::kQ6K ? kQualitySramQ6
                               : kQualitySramQ8;
  constexpr std::size_t kWeightBytes =
      Kind == QuantKind::kQ4K ? kMmaQ4Bytes
      : Kind == QuantKind::kQ6K ? kMmaQ6Bytes
                               : kMmaQ8Bytes;
  constexpr std::size_t kWeightValues =
      Kind == QuantKind::kQ8_0 ? kMmaQ8Values : kMmaBlockValues;
  const std::size_t row_stride = (columns / kWeightValues) * kWeightBytes;
  const std::size_t out0 =
      static_cast<std::size_t>(blockIdx.x) * kQualityI;
  const std::size_t prompt0 =
      static_cast<std::size_t>(blockIdx.y) * PromptTile;
  const int i_max =
      Fallback ? static_cast<int>(output_rows - out0) - 1 : kQualityI - 1;
  const int j_max =
      Fallback ? static_cast<int>(prompt_rows - prompt0) - 1 : PromptTile - 1;

  extern __shared__ std::uint8_t shared[];
  int* quality_shared = reinterpret_cast<int*>(shared);
  int* tile_y = quality_shared + PromptTile;
  int* tile_x = tile_y + quality_pad_ints(PromptTile * kMmqTileYk);
  int* x_qs = tile_x;
  [[maybe_unused]] float* x_df = nullptr;
  [[maybe_unused]] int* x_sc = nullptr;
  if constexpr (Kind != QuantKind::kQ4K) {
    x_df = reinterpret_cast<float*>(x_qs + 2 * kMmqTileNeK);
    if constexpr (Kind == QuantKind::kQ6K) {
      x_sc = reinterpret_cast<int*>(x_df + kMmqTileNeK / kQi6K);
    }
  }

  constexpr int kSum = PromptTile * kQualityI / (kQualityNwarps * 32);
  float sum[kSum];
#pragma unroll
  for (int s = 0; s < kSum; ++s) sum[s] = 0.0F;

  const int tid = threadIdx.y * 32 + threadIdx.x;
  const int lane = threadIdx.x;

  for (std::size_t kb0 = 0; kb0 < columns / kMmaBlockValues; ++kb0) {
    if constexpr (Kind == QuantKind::kQ4K) {
#pragma unroll
      for (int i0 = 0; i0 < kQualityI; i0 += kQualityNwarps) {
        int i = i0 + threadIdx.y;
        if (Fallback) i = min(i, max(i_max, 0));
        const std::size_t global_row = out0 + static_cast<std::size_t>(i);
        int qs0 = 0;
        if (!Fallback || global_row < output_rows) {
          const std::uint8_t* block =
              weights + global_row * row_stride + kb0 * kWeightBytes;
          qs0 = quality_get_int_b4(block + 16, lane);
        }
        x_qs[i * kSram + 16 * (lane / 8) + (lane % 8) + 0] =
            (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i * kSram + 16 * (lane / 8) + (lane % 8) + 8] =
            (qs0 >> 4) & 0x0F0F0F0F;
      }
#pragma unroll
      for (int i0 = 0; i0 < kQualityI; i0 += kQualityNwarps * 16) {
        int i = (i0 + threadIdx.y * 16 + lane / 2) % kQualityI;
        if (Fallback) i = min(i, max(i_max, 0));
        const std::size_t global_row = out0 + static_cast<std::size_t>(i);
        half2* dm = reinterpret_cast<half2*>(
            x_qs + i * kSram + 2 * kMmqTileNeK);
        const int ksc = lane % 2;
        if (!Fallback || global_row < output_rows) {
          const std::uint8_t* block =
              weights + global_row * row_stride + kb0 * kWeightBytes;
          const int* scales = reinterpret_cast<const int*>(block + 4);
          const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
          const int m32 = unpack_scales_q45_K(scales, ksc + 2);
          const std::uint8_t* sc8 = reinterpret_cast<const std::uint8_t*>(&sc32);
          const std::uint8_t* m8 = reinterpret_cast<const std::uint8_t*>(&m32);
          const float d = mma_read_half(block);
          const float dmin = mma_read_half(block + 2);
#pragma unroll
          for (int l = 0; l < 4; ++l) {
            dm[4 * ksc + l] = make_half2(d * static_cast<float>(sc8[l]),
                                        -dmin * static_cast<float>(m8[l]));
          }
        } else {
#pragma unroll
          for (int l = 0; l < 4; ++l) {
            dm[4 * ksc + l] = make_half2(0.0F, 0.0F);
          }
        }
      }
    } else if constexpr (Kind == QuantKind::kQ6K) {
#pragma unroll
      for (int i0 = 0; i0 < kQualityI; i0 += kQualityNwarps) {
        int i = i0 + threadIdx.y;
        if (Fallback) i = min(i, max(i_max, 0));
        const std::size_t global_row = out0 + static_cast<std::size_t>(i);
        int ql0 = 0;
        int ql1 = 0;
        int qh0 = 0;
        int qh1 = 0;
        if (!Fallback || global_row < output_rows) {
          const std::uint8_t* block =
              weights + global_row * row_stride + kb0 * kWeightBytes;
          const int ql = quality_get_int_b2(block, lane);
          ql0 = (ql >> 0) & 0x0F0F0F0F;
          ql1 = (ql >> 4) & 0x0F0F0F0F;
          const int qh = quality_get_int_b2(
              block + 128,
              (kQi6K / 4) * (lane / (kQi6K / 2)) + (lane % (kQi6K / 4)));
          qh0 = ((qh >> ((lane & 0x08) >> 2)) << 4) & 0x30303030;
          qh1 = (qh >> ((lane & 0x08) >> 2)) & 0x30303030;
        }
        const int kq0 = 2 * lane - (lane % (kQi6K / 2));
        const int kq1 = kq0 + (kQi6K / 2);
        x_qs[i * kSram + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i * kSram + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
      }
#pragma unroll
      for (int i0 = 0; i0 < kQualityI; i0 += kQualityNwarps * 32) {
        int i = (i0 + threadIdx.y * 32 + lane) % kQualityI;
        if (Fallback) i = min(i, max(i_max, 0));
        const std::size_t global_row = out0 + static_cast<std::size_t>(i);
        float d = 0.0F;
        if (!Fallback || global_row < output_rows) {
          const std::uint8_t* block =
              weights + global_row * row_stride + kb0 * kWeightBytes;
          d = mma_read_half(block + 208);
        }
        x_df[i * kSram] = d;
      }
#pragma unroll
      for (int i0 = 0; i0 < kQualityI; i0 += kQualityNwarps * 8) {
        int i = (i0 + threadIdx.y * 8 + lane / 4) % kQualityI;
        if (Fallback) i = min(i, max(i_max, 0));
        const std::size_t global_row = out0 + static_cast<std::size_t>(i);
        int sc = 0;
        if (!Fallback || global_row < output_rows) {
          const std::uint8_t* block =
              weights + global_row * row_stride + kb0 * kWeightBytes;
          sc = quality_get_int_b2(block + 192, lane % 4);
        }
        x_sc[i * kSram + (lane % 4)] = sc;
      }
    } else {
      const int kbx = lane / kQi80;
      const int kqsx = lane % kQi80;
#pragma unroll
      for (int i0 = 0; i0 < kQualityI; i0 += kQualityNwarps) {
        int i = i0 + threadIdx.y;
        if (Fallback) i = min(i, max(i_max, 0));
        const std::size_t global_row = out0 + static_cast<std::size_t>(i);
        int qs0 = 0;
        int qs1 = 0;
        if (!Fallback || global_row < output_rows) {
          const std::uint8_t* row =
              weights + global_row * row_stride +
              kb0 * 8 * kWeightBytes;
          qs0 = quality_get_int_b2(row + kbx * kWeightBytes + 2, kqsx);
          qs1 = quality_get_int_b2(
              row + (4 + kbx) * kWeightBytes + 2, kqsx);
        }
        x_qs[i * kSram + lane] = qs0;
        x_qs[i * kSram + kMmqTileNeK + lane] = qs1;
      }
#pragma unroll
      for (int i0 = 0; i0 < kQualityI; i0 += kQualityNwarps * 4) {
        int i = i0 + threadIdx.y * 4 + lane / 8;
        if (Fallback) i = min(i, max(i_max, 0));
        const std::size_t global_row = out0 + static_cast<std::size_t>(i);
        const int kbxd = lane % 8;
        float scale = 0.0F;
        if (!Fallback || global_row < output_rows) {
          const std::uint8_t* block =
              weights + global_row * row_stride +
              (kb0 * 8 + static_cast<std::size_t>(kbxd)) * kWeightBytes;
          scale = mma_read_half(block);
        }
        x_df[i * kSram + kbxd] = scale;
      }
    }

#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const std::size_t k_block = kb0 * 2 + static_cast<std::size_t>(half);
      if (Fallback) {
#pragma unroll
        for (int l0 = 0; l0 < PromptTile * kMmqTileYk;
             l0 += kQualityNwarps * 32) {
          const int l = l0 + tid;
          if (l >= PromptTile * kMmqTileYk) continue;
          const int j = l / kMmqTileYk;
          const int r = l % kMmqTileYk;
          const std::size_t prow = prompt0 + static_cast<std::size_t>(j);
          int value = 0;
          if (prow < prompt_rows) {
            value = y[(k_block * prompt_rows + prow) * kMmqTileYk + r];
          }
          tile_y[l] = value;
        }
      } else {
        const int* by0 =
            y + (k_block * prompt_rows + prompt0) * kMmqTileYk;
#pragma unroll
        for (int l0 = 0; l0 < PromptTile * kMmqTileYk;
             l0 += kQualityNwarps * 32) {
          const int l = l0 + tid;
          if (l < PromptTile * kMmqTileYk) tile_y[l] = by0[l];
        }
      }
      __syncthreads();

      const int k00 = half * kMmqTileNeK;
      const int* y_tile =
          tile_y + (threadIdx.y % kNtx) * (8 * kMmqTileYk);
      const int* y_qs = y_tile + 4;
      const int i0 = (threadIdx.y / kNtx) * kRowsPerWarp;

      if constexpr (Kind == QuantKind::kQ4K) {
        const half2* y_dm = reinterpret_cast<const half2*>(y_tile);
        int A[kNtx][4][4];
        float2 dmA[kNtx][2][4];
#pragma unroll
        for (int n = 0; n < kNtx; ++n) {
#pragma unroll
          for (int k01 = 0; k01 < kMmqTileNeK; k01 += kQi81) {
            const int k0 = k00 + k01;
            mma::load_a_m16k32(A[n][k01 / kQi81],
                               x_qs + (i0 + n * 16) * kSram + k0,
                               kSram);
          }
#pragma unroll
          for (int l = 0; l < 2; ++l) {
            const int i = i0 + n * 16 + mma::tile16x8_i(lane, 2 * l);
#pragma unroll
            for (int k01 = 0; k01 < kMmqTileNeK; k01 += kQi81) {
              const int k0 = k00 + k01;
              dmA[n][l][k01 / kQi81] = __half22float2(
                  reinterpret_cast<const half2*>(
                      x_qs + i * kSram + 2 * kMmqTileNeK)[k0 / kQi81]);
            }
          }
        }
#pragma unroll
        for (int j0 = 0; j0 < PromptTile; j0 += kNtx * 8) {
#pragma unroll
          for (int k01 = 0; k01 < kMmqTileNeK; k01 += kQi81) {
            int B[2];
            float2 dsB[2];
            mma::load_b_generic_k32(B, y_qs + j0 * kMmqTileYk + k01,
                                    kMmqTileYk);
#pragma unroll
            for (int l = 0; l < 2; ++l) {
              const int j = j0 + mma::tile16x8_j(lane, l);
              dsB[l] = __half22float2(y_dm[j * kMmqTileYk + k01 / kQi81]);
            }
#pragma unroll
            for (int n = 0; n < kNtx; ++n) {
              int C[4] = {0, 0, 0, 0};
              mma::mma_m16n8k32_s8(C, A[n][k01 / kQi81], B);
#pragma unroll
              for (int l = 0; l < 4; ++l) {
                sum[(j0 / 8 + n) * 4 + l] +=
                    dmA[n][l / 2][k01 / kQi81].x * dsB[l % 2].x *
                        static_cast<float>(C[l]) +
                    dmA[n][l / 2][k01 / kQi81].y * dsB[l % 2].y;
              }
            }
          }
        }
      } else if constexpr (Kind == QuantKind::kQ6K) {
        const float* y_df = reinterpret_cast<const float*>(y_tile);
        int A[kNtx][8][2];
        int scA[kNtx][2][8];
        float dA[kNtx][2];
#pragma unroll
        for (int n = 0; n < kNtx; ++n) {
#pragma unroll
          for (int k01 = 0; k01 < kMmqTileNeK; k01 += 8) {
            const int k0 = k00 + k01;
            mma::load_a_m16k16(A[n][k01 / 4 + 0],
                               x_qs + (i0 + n * 16) * kSram + k0,
                               kSram);
            mma::load_a_m16k16(
                A[n][k01 / 4 + 1],
                x_qs + (i0 + n * 16) * kSram + k0 + 4,
                kSram);
          }
#pragma unroll
          for (int k01 = 0; k01 < kMmqTileNeK; k01 += 16) {
            const int k0 = k00 + k01;
#pragma unroll
            for (int l = 0; l < 2; ++l) {
              const int i = i0 + n * 16 + mma::tile16x8_i(lane, 2 * l);
              const int sc_packed = x_sc[i * kSram + k0 / 16];
              const std::int8_t* sc =
                  reinterpret_cast<const std::int8_t*>(&sc_packed);
#pragma unroll
              for (int ksc = 0; ksc < 4; ++ksc) {
                scA[n][l][k01 / 4 + ksc] = sc[ksc];
              }
            }
          }
#pragma unroll
          for (int l = 0; l < 2; ++l) {
            const int i = i0 + n * 16 + mma::tile16x8_i(lane, 2 * l);
            dA[n][l] = x_df[i * kSram];
          }
        }
#pragma unroll
        for (int j0 = 0; j0 < PromptTile; j0 += kNtx * 8) {
          float tmp[kNtx][4];
#pragma unroll
          for (int n = 0; n < kNtx; ++n) {
#pragma unroll
            for (int l = 0; l < 4; ++l) tmp[n][l] = 0.0F;
          }
#pragma unroll
          for (int k01 = 0; k01 < kMmqTileNeK; k01 += 8) {
            int B0 = 0;
            int B1 = 0;
            float dB[2];
            mma::load_b_generic_k16(&B0, y_qs + j0 * kMmqTileYk + k01,
                                   kMmqTileYk);
            mma::load_b_generic_k16(&B1, y_qs + j0 * kMmqTileYk + k01 + 4,
                                   kMmqTileYk);
#pragma unroll
            for (int l = 0; l < 2; ++l) {
              const int j = j0 + mma::tile16x8_j(lane, l);
              dB[l] = y_df[j * kMmqTileYk + k01 / kQi81];
            }
#pragma unroll
            for (int n = 0; n < kNtx; ++n) {
              int C0[4] = {0, 0, 0, 0};
              int C1[4] = {0, 0, 0, 0};
              mma::mma_m16n8k16_s8(C0, A[n][k01 / 4 + 0], B0);
              mma::mma_m16n8k16_s8(C1, A[n][k01 / 4 + 1], B1);
#pragma unroll
              for (int l = 0; l < 4; ++l) {
                tmp[n][l] +=
                    static_cast<float>(C0[l] * scA[n][l / 2][k01 / 4 + 0] +
                                       C1[l] * scA[n][l / 2][k01 / 4 + 1]) *
                    dB[l % 2];
              }
            }
          }
#pragma unroll
          for (int n = 0; n < kNtx; ++n) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
              sum[(j0 / 8 + n) * 4 + l] += tmp[n][l] * dA[n][l / 2];
            }
          }
        }
      } else {
        const float* y_df = reinterpret_cast<const float*>(y_tile);
        int A[kNtx][4][4];
        float dA[kNtx][2][4];
#pragma unroll
        for (int n = 0; n < kNtx; ++n) {
#pragma unroll
          for (int k01 = 0; k01 < kMmqTileNeK; k01 += kQi80) {
            const int k0 = k00 + k01;
            mma::load_a_m16k32(A[n][k01 / kQi80],
                               x_qs + (i0 + n * 16) * kSram + k0,
                               kSram);
          }
#pragma unroll
          for (int l = 0; l < 2; ++l) {
            const int i = i0 + n * 16 + mma::tile16x8_i(lane, 2 * l);
#pragma unroll
            for (int k01 = 0; k01 < kMmqTileNeK; k01 += kQi80) {
              const int k0 = k00 + k01;
              dA[n][l][k01 / kQi80] = x_df[i * kSram + k0 / kQi80];
            }
          }
        }
#pragma unroll
        for (int j0 = 0; j0 < PromptTile; j0 += kNtx * 8) {
#pragma unroll
          for (int k01 = 0; k01 < kMmqTileNeK; k01 += kQi80) {
            int B[2];
            float dB[2];
            mma::load_b_generic_k32(B, y_qs + j0 * kMmqTileYk + k01,
                                    kMmqTileYk);
#pragma unroll
            for (int l = 0; l < 2; ++l) {
              const int j = j0 + mma::tile16x8_j(lane, l);
              dB[l] = y_df[j * kMmqTileYk + k01 / kQi81];
            }
#pragma unroll
            for (int n = 0; n < kNtx; ++n) {
              int C[4] = {0, 0, 0, 0};
              mma::mma_m16n8k32_s8(C, A[n][k01 / kQi80], B);
#pragma unroll
              for (int l = 0; l < 4; ++l) {
                sum[(j0 / 8 + n) * 4 + l] +=
                    static_cast<float>(C[l]) * dA[n][l / 2][k01 / kQi80] *
                    dB[l % 2];
              }
            }
          }
        }
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (int j0 = 0; j0 < PromptTile; j0 += kNtx * 8) {
#pragma unroll
    for (int n = 0; n < kNtx; ++n) {
#pragma unroll
      for (int l = 0; l < 4; ++l) {
        const int j =
            j0 + (threadIdx.y % kNtx) * 8 + mma::tile16x8_j(lane, l);
        if (j > j_max) continue;
        const int row = (threadIdx.y / kNtx) * kRowsPerWarp + n * 16 +
                        mma::tile16x8_i(lane, l);
        if (Fallback && row > i_max) continue;
        const std::size_t out_row = out0 + static_cast<std::size_t>(row);
        const std::size_t prompt_row =
            prompt0 + static_cast<std::size_t>(j);
        if (out_row < output_rows && prompt_row < prompt_rows) {
          output[prompt_row * output_rows + out_row] =
              sum[(j0 / 8 + n) * 4 + l];
        }
      }
    }
  }
}

template <QuantKind Kind, int PromptTile, bool Fallback>
cudaError_t launch_quality_mma(
    const std::uint8_t* weights, std::size_t output_rows, std::size_t columns,
    const int* y, std::size_t prompt_rows, float* output,
    cudaStream_t stream) noexcept {
  const dim3 grid(
      static_cast<unsigned int>((output_rows + kQualityI - 1) / kQualityI),
      static_cast<unsigned int>((prompt_rows + PromptTile - 1) / PromptTile));
  const dim3 block(32, 8);
  const std::size_t shared =
      quality_shared_ints(PromptTile) * sizeof(int);
  cudaError_t error = cudaFuncSetAttribute(
      quant_mmq_mma_quality_kernel<Kind, PromptTile, Fallback>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shared));
  if (error != cudaSuccess) return error;
  if constexpr (Kind == QuantKind::kQ4K && PromptTile == 128) {
    static bool printed_q4_attrs = false;
    if (!printed_q4_attrs) {
      cudaFuncAttributes attr{};
      if (cudaFuncGetAttributes(
              &attr, quant_mmq_mma_quality_kernel<Kind, PromptTile, Fallback>) ==
          cudaSuccess) {
        std::printf(
            "mma_quality_q4_attrs fallback=%d regs=%d local=%zu const=%zu "
            "shared=%zu maxThreadsPerBlock=%d\n",
            Fallback ? 1 : 0, attr.numRegs, attr.localSizeBytes,
            attr.constSizeBytes, attr.sharedSizeBytes, attr.maxThreadsPerBlock);
      }
      printed_q4_attrs = true;
    }
  }
  quant_mmq_mma_quality_kernel<Kind, PromptTile, Fallback>
      <<<grid, block, shared, stream>>>(weights, output_rows, columns, y,
                                       prompt_rows, output);
  return cudaPeekAtLastError();
}

}  // namespace

unsigned int selected_mma_mmq_prompt_tile() noexcept { return 128U; }

std::size_t mma_mmq_shared_bytes(unsigned int prompt_tile) noexcept {
  if (prompt_tile != 32 && prompt_tile != 64 && prompt_tile != 128) return 0;
  return quality_shared_ints(static_cast<int>(prompt_tile)) * sizeof(int);
}

int mma_mmq_occupancy(QuantKind kind, unsigned int prompt_tile) noexcept {
  int occupancy = 0;
  cudaError_t error = cudaErrorInvalidValue;
  const std::size_t shared = mma_mmq_shared_bytes(prompt_tile);
  if (shared == 0) return 0;
  auto query = [&](auto kernel) -> cudaError_t {
    cudaError_t set = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared));
    if (set != cudaSuccess) return set;
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, kMmaThreads, shared);
  };
  if (kind == QuantKind::kQ4K) {
    if (prompt_tile == 32) {
      error = query(quant_mmq_mma_quality_kernel<QuantKind::kQ4K, 32, true>);
    } else if (prompt_tile == 64) {
      error = query(quant_mmq_mma_quality_kernel<QuantKind::kQ4K, 64, true>);
    } else if (prompt_tile == 128) {
      error = query(quant_mmq_mma_quality_kernel<QuantKind::kQ4K, 128, true>);
    }
  } else if (kind == QuantKind::kQ6K) {
    if (prompt_tile == 32) {
      error = query(quant_mmq_mma_quality_kernel<QuantKind::kQ6K, 32, true>);
    } else if (prompt_tile == 64) {
      error = query(quant_mmq_mma_quality_kernel<QuantKind::kQ6K, 64, true>);
    } else if (prompt_tile == 128) {
      error = query(quant_mmq_mma_quality_kernel<QuantKind::kQ6K, 128, true>);
    }
  } else if (kind == QuantKind::kQ8_0) {
    if (prompt_tile == 32) {
      error = query(quant_mmq_mma_quality_kernel<QuantKind::kQ8_0, 32, true>);
    } else if (prompt_tile == 64) {
      error = query(quant_mmq_mma_quality_kernel<QuantKind::kQ8_0, 64, true>);
    } else if (prompt_tile == 128) {
      error = query(quant_mmq_mma_quality_kernel<QuantKind::kQ8_0, 128, true>);
    }
  }
  return error == cudaSuccess ? occupancy : 0;
}

template <QuantKind Kind, int PromptTile>
cudaError_t launch_quality_dispatch(
    const std::uint8_t* weights, std::size_t output_rows, std::size_t columns,
    const int* y, std::size_t prompt_rows, float* output,
    cudaStream_t stream) noexcept {
  if (output_rows % kQualityI == 0 && prompt_rows % PromptTile == 0) {
    return launch_quality_mma<Kind, PromptTile, false>(
        weights, output_rows, columns, y, prompt_rows, output, stream);
  }
  return launch_quality_mma<Kind, PromptTile, true>(
      weights, output_rows, columns, y, prompt_rows, output, stream);
}

cudaError_t launch_quantize_mmq_q8_1(QuantKind kind, const __nv_bfloat16* prompt,
                                     std::size_t prompt_rows,
                                     std::size_t columns, Q8Block* workspace,
                                     cudaStream_t stream) noexcept {
  if (prompt == nullptr || workspace == nullptr || prompt_rows == 0 ||
      columns == 0 ||
      (kind != QuantKind::kQ4K && kind != QuantKind::kQ6K &&
       kind != QuantKind::kQ8_0)) {
    return cudaErrorInvalidValue;
  }
  if (kind == QuantKind::kQ8_0 && columns % kMmaBlockValues != 0) {
    return cudaErrorInvalidValue;
  }
  const std::size_t count = prompt_rows * columns;
  const unsigned int blocks = static_cast<unsigned int>(
      (count + static_cast<std::size_t>(kQualityQuantThreads) - 1) /
      static_cast<std::size_t>(kQualityQuantThreads));
  if (kind == QuantKind::kQ4K) {
    quantize_mmq_q8_1_bf16<QuantKind::kQ4K>
        <<<blocks, kQualityQuantThreads, 0, stream>>>(
            prompt, prompt_rows, columns,
            reinterpret_cast<std::uint8_t*>(workspace));
  } else if (kind == QuantKind::kQ6K) {
    quantize_mmq_q8_1_bf16<QuantKind::kQ6K>
        <<<blocks, kQualityQuantThreads, 0, stream>>>(
            prompt, prompt_rows, columns,
            reinterpret_cast<std::uint8_t*>(workspace));
  } else {
    quantize_mmq_q8_1_bf16<QuantKind::kQ8_0>
        <<<blocks, kQualityQuantThreads, 0, stream>>>(
            prompt, prompt_rows, columns,
            reinterpret_cast<std::uint8_t*>(workspace));
  }
  return cudaPeekAtLastError();
}

cudaError_t launch_quant_mmq_mma_tile(
    QuantKind kind, const std::uint8_t* weights, std::size_t output_rows,
    std::size_t columns, const __nv_bfloat16* prompt, std::size_t prompt_rows,
    Q8Block* q8_workspace, float* output, unsigned int prompt_tile,
    cudaStream_t stream) noexcept {
  if (weights == nullptr || prompt == nullptr || q8_workspace == nullptr ||
      output == nullptr || output_rows == 0 || columns == 0 ||
      prompt_rows == 0 || columns % kMmaBlockValues != 0 ||
      (kind != QuantKind::kQ4K && kind != QuantKind::kQ6K) ||
      (prompt_tile != 32 && prompt_tile != 64 && prompt_tile != 128)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t error = launch_quantize_mmq_q8_1(
      kind, prompt, prompt_rows, columns, q8_workspace, stream);
  if (error != cudaSuccess) return error;
  const int* y = reinterpret_cast<const int*>(q8_workspace);
  if (kind == QuantKind::kQ4K) {
    if (prompt_tile == 32) {
      error = launch_quality_dispatch<QuantKind::kQ4K, 32>(
          weights, output_rows, columns, y, prompt_rows, output, stream);
    } else if (prompt_tile == 128) {
      error = launch_quality_dispatch<QuantKind::kQ4K, 128>(
          weights, output_rows, columns, y, prompt_rows, output, stream);
    } else {
      error = launch_quality_dispatch<QuantKind::kQ4K, 64>(
          weights, output_rows, columns, y, prompt_rows, output, stream);
    }
  } else if (prompt_tile == 32) {
    error = launch_quality_dispatch<QuantKind::kQ6K, 32>(
        weights, output_rows, columns, y, prompt_rows, output, stream);
  } else if (prompt_tile == 128) {
    error = launch_quality_dispatch<QuantKind::kQ6K, 128>(
        weights, output_rows, columns, y, prompt_rows, output, stream);
  } else {
    error = launch_quality_dispatch<QuantKind::kQ6K, 64>(
        weights, output_rows, columns, y, prompt_rows, output, stream);
  }
  return error;
}

cudaError_t launch_quant_mmq_mma(
    QuantKind kind, const std::uint8_t* weights, std::size_t output_rows,
    std::size_t columns, const __nv_bfloat16* prompt, std::size_t prompt_rows,
    Q8Block* q8_workspace, float* output, cudaStream_t stream) noexcept {
  return launch_quant_mmq_mma_tile(
      kind, weights, output_rows, columns, prompt, prompt_rows, q8_workspace,
      output, selected_mma_mmq_prompt_tile(), stream);
}

unsigned int selected_q8_mma_mmq_prompt_tile() noexcept { return 128U; }

int q8_mma_mmq_occupancy(unsigned int prompt_tile) noexcept {
  int occupancy = 0;
  cudaError_t error = cudaErrorInvalidValue;
  if (prompt_tile == 32) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, quant_mmq_mma_kernel<QuantKind::kQ8_0, 32>, kMmaThreads,
        q8_mma_mmq_shared_bytes(32));
  } else if (prompt_tile == 64) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, quant_mmq_mma_kernel<QuantKind::kQ8_0, 64>, kMmaThreads,
        q8_mma_mmq_shared_bytes(64));
  } else if (prompt_tile == 128) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, quant_mmq_mma_kernel<QuantKind::kQ8_0, 128>, kMmaThreads,
        q8_mma_mmq_shared_bytes(128));
  }
  return error == cudaSuccess ? occupancy : 0;
}

cudaError_t launch_q8_mmq_mma_tile(
    const std::uint8_t* weights, std::size_t output_rows, std::size_t columns,
    const __nv_bfloat16* prompt, std::size_t prompt_rows, float* output,
    unsigned int prompt_tile, cudaStream_t stream) noexcept {
  if (weights == nullptr || prompt == nullptr || output == nullptr ||
      output_rows == 0 || columns == 0 || prompt_rows == 0 ||
      columns % kMmaQ8Values != 0 ||
      (prompt_tile != 32 && prompt_tile != 64 && prompt_tile != 128)) {
    return cudaErrorInvalidValue;
  }
  if (prompt_tile == 32) {
    return launch_mma_tile<QuantKind::kQ8_0, 32>(
        weights, output_rows, columns, prompt, prompt_rows, output, stream);
  }
  if (prompt_tile == 128) {
    return launch_mma_tile<QuantKind::kQ8_0, 128>(
        weights, output_rows, columns, prompt, prompt_rows, output, stream);
  }
  return launch_mma_tile<QuantKind::kQ8_0, 64>(
      weights, output_rows, columns, prompt, prompt_rows, output, stream);
}

cudaError_t launch_q8_mmq_mma(const std::uint8_t* weights,
                              std::size_t output_rows, std::size_t columns,
                              const __nv_bfloat16* prompt,
                              std::size_t prompt_rows, float* output,
                              cudaStream_t stream) noexcept {
  return launch_q8_mmq_mma_tile(weights, output_rows, columns, prompt,
                                prompt_rows, output,
                                selected_q8_mma_mmq_prompt_tile(), stream);
}

unsigned int selected_q8_quality_mmq_prompt_tile() noexcept { return 128U; }

int q8_quality_mmq_occupancy(unsigned int prompt_tile) noexcept {
  return mma_mmq_occupancy(QuantKind::kQ8_0, prompt_tile);
}

cudaError_t launch_q8_mmq_quality_mma(const std::uint8_t* weights,
                                      std::size_t output_rows,
                                      std::size_t columns, const Q8Block* y,
                                      std::size_t prompt_rows, float* output,
                                      cudaStream_t stream) noexcept {
  if (weights == nullptr || y == nullptr || output == nullptr ||
      output_rows == 0 || columns == 0 || prompt_rows == 0 ||
      columns % kMmaBlockValues != 0) {
    return cudaErrorInvalidValue;
  }
  const unsigned int prompt_tile = selected_q8_quality_mmq_prompt_tile();
  const int* packed = reinterpret_cast<const int*>(y);
  if (prompt_tile == 32) {
    return launch_quality_dispatch<QuantKind::kQ8_0, 32>(
        weights, output_rows, columns, packed, prompt_rows, output, stream);
  }
  if (prompt_tile == 64) {
    return launch_quality_dispatch<QuantKind::kQ8_0, 64>(
        weights, output_rows, columns, packed, prompt_rows, output, stream);
  }
  return launch_quality_dispatch<QuantKind::kQ8_0, 128>(
      weights, output_rows, columns, packed, prompt_rows, output, stream);
}

cudaError_t launch_q8_mmq_quality(const std::uint8_t* weights,
                                  std::size_t output_rows, std::size_t columns,
                                  const __nv_bfloat16* prompt,
                                  std::size_t prompt_rows, Q8Block* q8_workspace,
                                  float* output, cudaStream_t stream) noexcept {
  if (prompt == nullptr || q8_workspace == nullptr) {
    return cudaErrorInvalidValue;
  }
  cudaError_t error = launch_quantize_mmq_q8_1(
      QuantKind::kQ8_0, prompt, prompt_rows, columns, q8_workspace, stream);
  if (error != cudaSuccess) return error;
  return launch_q8_mmq_quality_mma(weights, output_rows, columns, q8_workspace,
                                   prompt_rows, output, stream);
}

}  // namespace qw38::cuda
