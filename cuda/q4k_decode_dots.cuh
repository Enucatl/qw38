#pragma once

// OPT-046 cooperative packed Q4_K integer decode dots.
// Technique from pinned llama.cpp mmvq.cu::mul_mat_vec_q and
// vecdotq.cuh::vec_dot_q4_K_q8_1 (cc83d7b, MIT). Quartz-owned rewrite:
// 32-bit qs loads, DP4A of four int8 products, K distributed across warps
// per row, independent lane accumulators until row completion, Q8 group
// sums staged once. Do not vendor ggml headers.

#include <cstring>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "q4k_decode_path.cuh"
#include "quant_mmv.h"

namespace qw38::cuda {

// OPT-046 llama-style Q8_1 staging. Distinct from FP32-scale Q8Block.
// Layout matches pinned llama block_q8_1: half d, half s, int8 qs[32].
// Kept out of quant_mmv.h so full_scheduler.cu does not pull cuda_fp16.h
// through the packed MMV public header (that TU is codegen-sensitive).
struct Q8_1Block {
  __half scale;
  __half q8_sum;
  std::int8_t values[32];
};
static_assert(sizeof(Q8_1Block) == 36, "unexpected Q8_1 block padding");

cudaError_t launch_quantize_bf16_q8_1(const __nv_bfloat16* activation,
                                      Q8_1Block* q8, std::size_t columns,
                                      cudaStream_t stream) noexcept;

#if defined(__CUDACC__) && !defined(QW38_SKIP_Q4K_DOT_KERNELS)

namespace q4k_dots {

constexpr int kWarp = 32;
constexpr int kQ4KBytes = 144;
constexpr int kValuesPerBlock = 256;

__device__ __forceinline__ float read_half_bytes(const std::uint8_t* bytes) {
  const unsigned short bits = static_cast<unsigned short>(bytes[0]) |
                              (static_cast<unsigned short>(bytes[1]) << 8U);
  return __half2float(__ushort_as_half(bits));
}

// Same 6-bit scale/min unpack as packed decode_q4 / OPT-042.
__device__ __forceinline__ void q4k_scale_min(const std::uint8_t* packed,
                                              int index, int* scale,
                                              int* minimum) {
  if (index < 4) {
    *scale = packed[index] & 63;
    *minimum = packed[index + 4] & 63;
    return;
  }
  *scale = (packed[index + 4] & 15) | ((packed[index - 4] >> 6) << 4);
  *minimum = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4);
}

__device__ __forceinline__ int pack4_bytes(const std::int8_t* bytes) {
  return static_cast<int>(static_cast<std::uint8_t>(bytes[0])) |
         (static_cast<int>(static_cast<std::uint8_t>(bytes[1])) << 8) |
         (static_cast<int>(static_cast<std::uint8_t>(bytes[2])) << 16) |
         (static_cast<int>(static_cast<std::uint8_t>(bytes[3])) << 24);
}

template <bool UseQ81>
__device__ __forceinline__ float vec_dot_q4k_group(const std::uint8_t* block,
                                                   const void* block_staged,
                                                   int group, int pack) {
  const float d = read_half_bytes(block);
  const float dmin = read_half_bytes(block + 2);
  int scale = 0;
  int minimum = 0;
  q4k_scale_min(block + 4, group, &scale, &minimum);
  const std::uint8_t* qs = block + 16 + (group / 2) * 32;
  const int high = group & 1;
  std::int8_t q4_lo[4];
  std::int8_t q4_hi[4];
  std::int8_t u_lo[4];
  std::int8_t u_hi[4];
  float d8 = 0.0F;
  int stored_q8_sum = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int e0 = pack * 4 + i;
    const int e1 = 16 + pack * 4 + i;
    const std::uint8_t b0 = qs[e0];
    const std::uint8_t b1 = qs[e1];
    q4_lo[i] = static_cast<std::int8_t>(high == 0 ? (b0 & 15) : (b0 >> 4));
    q4_hi[i] = static_cast<std::int8_t>(high == 0 ? (b1 & 15) : (b1 >> 4));
  }
  if constexpr (UseQ81) {
    const Q8_1Block* q8 =
        static_cast<const Q8_1Block*>(block_staged) + group;
    d8 = __half2float(q8->scale);
    stored_q8_sum = static_cast<int>(rintf(__half2float(q8->q8_sum)));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      u_lo[i] = q8->values[pack * 4 + i];
      u_hi[i] = q8->values[16 + pack * 4 + i];
    }
  } else {
    const Q8Block* q8 = static_cast<const Q8Block*>(block_staged) + group;
    d8 = q8->scale;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      u_lo[i] = q8->values[pack * 4 + i];
      u_hi[i] = q8->values[16 + pack * 4 + i];
    }
  }
  const int v0 = pack4_bytes(q4_lo) & 0x0F0F0F0F;
  const int v1 = pack4_bytes(q4_hi) & 0x0F0F0F0F;
  const int u0 = pack4_bytes(u_lo);
  const int u1 = pack4_bytes(u_hi);
  int dot = __dp4a(v1, u1, __dp4a(v0, u0, 0));
  int q8_sum = 0;
  if constexpr (UseQ81) {
    q8_sum = pack == 0 ? stored_q8_sum : 0;
  } else {
    q8_sum = __dp4a(0x01010101, u1, __dp4a(0x01010101, u0, 0));
  }
  dot += __shfl_xor_sync(0xFFFFFFFFU, dot, 1, kWarp);
  dot += __shfl_xor_sync(0xFFFFFFFFU, dot, 2, kWarp);
  if constexpr (!UseQ81) {
    q8_sum += __shfl_xor_sync(0xFFFFFFFFU, q8_sum, 1, kWarp);
    q8_sum += __shfl_xor_sync(0xFFFFFFFFU, q8_sum, 2, kWarp);
  }
  if (pack != 0) return 0.0F;
  const float sd = __fmul_rn(__fmul_rn(d, static_cast<float>(scale)),
                             static_cast<float>(dot));
  const float md = __fmul_rn(__fmul_rn(dmin, static_cast<float>(minimum)),
                             static_cast<float>(q8_sum));
  return __fmul_rn(d8, __fsub_rn(sd, md));
}

template <int WarpsPerRow, bool UseQ81>
__global__ void q4k_coop_mmv(const std::uint8_t* weights, std::size_t rows,
                             std::size_t columns, const void* staged,
                             float* output) {
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) return;

  const int group = lane / 4;
  const int pack = lane % 4;
  const std::size_t n_blocks = columns / kValuesPerBlock;
  const std::uint8_t* row_weights = weights + row * n_blocks * kQ4KBytes;
  const std::size_t group_bytes =
      UseQ81 ? sizeof(Q8_1Block) : sizeof(Q8Block);
  const char* staged_bytes = static_cast<const char*>(staged);
  float acc = 0.0F;
  for (std::size_t kbx = static_cast<std::size_t>(warp); kbx < n_blocks;
       kbx += static_cast<std::size_t>(WarpsPerRow)) {
    const std::uint8_t* block = row_weights + kbx * kQ4KBytes;
    const void* block_staged =
        staged_bytes + kbx * 8U * group_bytes;
    acc += vec_dot_q4k_group<UseQ81>(block, block_staged, group, pack);
  }

  __shared__ float partial[WarpsPerRow][kWarp];
  partial[warp][lane] = acc;
  __syncthreads();
  if (warp == 0) {
    float sum = 0.0F;
#pragma unroll
    for (int other = 0; other < WarpsPerRow; ++other) {
      sum += partial[other][lane];
    }
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset, kWarp);
    }
    if (lane == 0) output[row] = sum;
  }
}

__global__ void quantize_bf16_q8_1(const __nv_bfloat16* input, Q8_1Block* output,
                                   std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x & (kWarp - 1);
  const std::size_t q8_index = index / kWarp;
  const float value = index < count ? __bfloat162float(input[index]) : 0.0F;
  float maximum = fabsf(value);
  float sum = value;
  for (int offset = 16; offset > 0; offset /= 2) {
    maximum = fmaxf(maximum,
                    __shfl_down_sync(0xFFFFFFFFU, maximum, offset, kWarp));
    sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset, kWarp);
  }
  maximum = __shfl_sync(0xFFFFFFFFU, maximum, 0, kWarp);
  const float scale = maximum == 0.0F ? 0.0F : maximum / 127.0F;
  const std::int8_t quant =
      scale == 0.0F ? 0 : static_cast<std::int8_t>(roundf(value / scale));
  int q_sum = static_cast<int>(quant);
  for (int offset = 16; offset > 0; offset /= 2) {
    q_sum += __shfl_down_sync(0xFFFFFFFFU, q_sum, offset, kWarp);
  }
  if (index < count) output[q8_index].values[lane] = quant;
  if (lane == 0 && q8_index * kWarp < count) {
    output[q8_index].scale = __float2half_rn(scale);
    output[q8_index].q8_sum =
        __float2half_rn(static_cast<float>(q_sum));
    (void)sum;
  }
}

template <int WarpsPerRow, bool UseQ81>
cudaError_t launch_coop(const std::uint8_t* weights, std::size_t rows,
                        std::size_t columns, const void* staged, float* output,
                        cudaStream_t stream) {
  dim3 block(kWarp, WarpsPerRow);
  const unsigned int grid = static_cast<unsigned int>(rows);
  q4k_coop_mmv<WarpsPerRow, UseQ81>
      <<<grid, block, 0, stream>>>(weights, rows, columns, staged, output);
  return cudaPeekAtLastError();
}

}  // namespace q4k_dots

#endif  // __CUDACC__

}  // namespace qw38::cuda
