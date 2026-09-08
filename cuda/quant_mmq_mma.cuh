#pragma once

// Q4_K / Q6_K / Q8_0 prompt MMQ with Ampere-style MMA tiles for Quartz
// OPT-016 (Q4_K/Q6_K) and OPT-017 (mixer Q8_0). Geometry: 256 threads,
// 128 output rows per block, prompt tile J in {32, 64, 128} (llama.cpp
// Ampere J set intersected with sm_120 shared memory). Internal activation
// staging is llama.cpp Q8_1 in shared memory from token-major BF16; this
// layout is not Quartz Q8Block and uses no extra cudaMalloc. Output is
// token-major FP32 [prompt_rows, output_rows]. Q8_0 uses GGUF 34-byte /
// 32-value blocks (FP16 scale, signed int8 values); each K-step of 32
// columns is one block. Production Q8_0 is launched via launch_q8_mmq_mma,
// not launch_quant_mmq.
//
// Provenance: llama.cpp mmq.cuh / mma.cuh / mmq-config-ampere.cuh /
// mmq-load-tiles.cuh / mmq-vec-dot.cuh at
// cc83d7b4824f73cfdda4dfbb47ee39804f71b328 (MIT, The ggml authors).
// ds4 cuda/mmq is the ggml-free launcher pattern only; this file does not
// copy ../ds4/cuda/mmq/ and does not include ggml headers.

#include "mma.cuh"
#include "quant_mmv.h"

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

inline std::size_t mma_mmq_shared_bytes(int prompt_tile) noexcept {
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
  const std::size_t shared = mma_mmq_shared_bytes(PromptTile);
  quant_mmq_mma_kernel<Kind, PromptTile>
      <<<grid, kMmaThreads, shared, stream>>>(
          weights, output_rows, columns, prompt, prompt_rows, output);
  return cudaPeekAtLastError();
}

}  // namespace

unsigned int selected_mma_mmq_prompt_tile() noexcept { return 128U; }

cudaError_t launch_quant_mmq_mma(
    QuantKind kind, const std::uint8_t* weights, std::size_t output_rows,
    std::size_t columns, const __nv_bfloat16* prompt, std::size_t prompt_rows,
    float* output, cudaStream_t stream) noexcept {
  if (weights == nullptr || prompt == nullptr || output == nullptr ||
      output_rows == 0 || columns == 0 || prompt_rows == 0 ||
      columns % kMmaBlockValues != 0 ||
      (kind != QuantKind::kQ4K && kind != QuantKind::kQ6K)) {
    return cudaErrorInvalidValue;
  }
  const unsigned int tile = selected_mma_mmq_prompt_tile();
  if (kind == QuantKind::kQ4K) {
    if (tile == 32) {
      return launch_mma_tile<QuantKind::kQ4K, 32>(
          weights, output_rows, columns, prompt, prompt_rows, output, stream);
    }
    if (tile == 128) {
      return launch_mma_tile<QuantKind::kQ4K, 128>(
          weights, output_rows, columns, prompt, prompt_rows, output, stream);
    }
    return launch_mma_tile<QuantKind::kQ4K, 64>(
        weights, output_rows, columns, prompt, prompt_rows, output, stream);
  }
  if (tile == 32) {
    return launch_mma_tile<QuantKind::kQ6K, 32>(
        weights, output_rows, columns, prompt, prompt_rows, output, stream);
  }
  if (tile == 128) {
    return launch_mma_tile<QuantKind::kQ6K, 128>(
        weights, output_rows, columns, prompt, prompt_rows, output, stream);
  }
  return launch_mma_tile<QuantKind::kQ6K, 64>(
      weights, output_rows, columns, prompt, prompt_rows, output, stream);
}

cudaError_t launch_quant_mmq_mma_tile(
    QuantKind kind, const std::uint8_t* weights, std::size_t output_rows,
    std::size_t columns, const __nv_bfloat16* prompt, std::size_t prompt_rows,
    float* output, unsigned int prompt_tile, cudaStream_t stream) noexcept {
  if (weights == nullptr || prompt == nullptr || output == nullptr ||
      output_rows == 0 || columns == 0 || prompt_rows == 0 ||
      columns % kMmaBlockValues != 0 ||
      (kind != QuantKind::kQ4K && kind != QuantKind::kQ6K) ||
      (prompt_tile != 32 && prompt_tile != 64 && prompt_tile != 128)) {
    return cudaErrorInvalidValue;
  }
  if (kind == QuantKind::kQ4K) {
    if (prompt_tile == 32) {
      return launch_mma_tile<QuantKind::kQ4K, 32>(
          weights, output_rows, columns, prompt, prompt_rows, output, stream);
    }
    if (prompt_tile == 128) {
      return launch_mma_tile<QuantKind::kQ4K, 128>(
          weights, output_rows, columns, prompt, prompt_rows, output, stream);
    }
    return launch_mma_tile<QuantKind::kQ4K, 64>(
        weights, output_rows, columns, prompt, prompt_rows, output, stream);
  }
  if (prompt_tile == 32) {
    return launch_mma_tile<QuantKind::kQ6K, 32>(
        weights, output_rows, columns, prompt, prompt_rows, output, stream);
  }
  if (prompt_tile == 128) {
    return launch_mma_tile<QuantKind::kQ6K, 128>(
        weights, output_rows, columns, prompt, prompt_rows, output, stream);
  }
  return launch_mma_tile<QuantKind::kQ6K, 64>(
      weights, output_rows, columns, prompt, prompt_rows, output, stream);
}

unsigned int selected_q8_mma_mmq_prompt_tile() noexcept { return 128U; }

int q8_mma_mmq_occupancy(unsigned int prompt_tile) noexcept {
  int occupancy = 0;
  cudaError_t error = cudaErrorInvalidValue;
  if (prompt_tile == 32) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, quant_mmq_mma_kernel<QuantKind::kQ8_0, 32>, kMmaThreads,
        mma_mmq_shared_bytes(32));
  } else if (prompt_tile == 64) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, quant_mmq_mma_kernel<QuantKind::kQ8_0, 64>, kMmaThreads,
        mma_mmq_shared_bytes(64));
  } else if (prompt_tile == 128) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, quant_mmq_mma_kernel<QuantKind::kQ8_0, 128>, kMmaThreads,
        mma_mmq_shared_bytes(128));
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

}  // namespace qw38::cuda
