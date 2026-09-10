#pragma once

// OPT-047 Q8_0 × Q8_1 cooperative DP4A mixer decode dots.
// Technique from pinned llama.cpp mmvq.cu::mul_mat_vec_q and
// vecdotq.cuh::vec_dot_q8_0_q8_1 (cc83d7b, MIT). Quartz-owned rewrite:
// 2-byte-aligned Q8_0 qs loads, DP4A of four int8 products, K distributed
// across 1/2/4/8 warps per row, OPT-046 Q8_1 staging reused. No ggml.

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "q8_decode_path.cuh"
#include "quant_mmv.h"

namespace qw38::cuda {

cudaError_t launch_q8_mmv_bf16(const std::uint8_t* weights, std::size_t rows,
                               std::size_t columns,
                               const __nv_bfloat16* activation, float* output,
                               cudaStream_t stream) noexcept;

cudaError_t launch_q8_coop_mmv_prequant(
    const std::uint8_t* weights, std::size_t rows, std::size_t columns,
    const void* staged, float* output, unsigned int warps_per_row,
    cudaStream_t stream) noexcept;

cudaError_t launch_q8_coop_mmv(const std::uint8_t* weights, std::size_t rows,
                               std::size_t columns,
                               const __nv_bfloat16* activation, void* workspace,
                               float* output, unsigned int warps_per_row,
                               cudaStream_t stream) noexcept;

int q8_coop_occupancy(unsigned int warps_per_row) noexcept;
void q8_coop_kernel_attributes(unsigned int warps_per_row, int* registers,
                               std::size_t* local_bytes,
                               int* occupancy) noexcept;

int q8_mmv_bf16_occupancy() noexcept;
void q8_mmv_bf16_kernel_attributes(int* registers, std::size_t* local_bytes,
                                   int* occupancy) noexcept;

}  // namespace qw38::cuda

#if defined(__CUDACC__) && !defined(QW38_SKIP_Q8_DOT_KERNELS)

#define QW38_SKIP_Q4K_DOT_KERNELS
#include "q4k_decode_dots.cuh"

namespace qw38::cuda {
namespace q8_dots {

constexpr int kWarp = 32;
constexpr int kQ80Bytes = 34;
constexpr int kQ80Values = 32;
constexpr int kQI8 = 8;
constexpr int kVdr = 2;
constexpr int kBf16Threads = 256;

__device__ __forceinline__ float read_q8_half(const std::uint8_t* bytes) {
  const unsigned short bits = static_cast<unsigned short>(bytes[0]) |
                              (static_cast<unsigned short>(bytes[1]) << 8U);
  return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ int get_int_b2(const void* x, int i32) {
  const std::uint16_t* x16 = static_cast<const std::uint16_t*>(x);
  return static_cast<int>(x16[2 * i32]) |
         (static_cast<int>(x16[2 * i32 + 1]) << 16);
}

__device__ __forceinline__ float vec_dot_q8_0_q8_1(const std::uint8_t* block,
                                                   const Q8_1Block* q8,
                                                   int iqs) {
  const int v0 = get_int_b2(block + 2, iqs);
  const int v1 = get_int_b2(block + 2, iqs + 1);
  const int u0 =
      *reinterpret_cast<const int*>(q8->values + static_cast<std::size_t>(iqs) * 4);
  const int u1 = *reinterpret_cast<const int*>(
      q8->values + static_cast<std::size_t>(iqs + 1) * 4);
  const int sumi = __dp4a(v1, u1, __dp4a(v0, u0, 0));
  const float d0 = read_q8_half(block);
  const float d1 = __half2float(q8->scale);
  return d0 * d1 * static_cast<float>(sumi);
}

template <int WarpsPerRow>
__global__ void q8_coop_mmv(const std::uint8_t* weights, std::size_t rows,
                            std::size_t columns, const Q8_1Block* staged,
                            float* output) {
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) return;

  const int tid = kWarp * warp + lane;
  const int n_blocks = static_cast<int>(columns / kQ80Values);
  constexpr int blocks_per_iter = kVdr * WarpsPerRow * kWarp / kQI8;
  const std::uint8_t* row_weights =
      weights + row * static_cast<std::size_t>(n_blocks) * kQ80Bytes;
  float acc = 0.0F;
  for (int kbx = tid / (kQI8 / kVdr); kbx < n_blocks; kbx += blocks_per_iter) {
    const int iqs = kVdr * (tid % (kQI8 / kVdr));
    acc += vec_dot_q8_0_q8_1(row_weights + static_cast<std::size_t>(kbx) * kQ80Bytes,
                             staged + kbx, iqs);
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

__global__ void q8_mmv_bf16_ref(const std::uint8_t* weights, std::size_t rows,
                                std::size_t columns,
                                const __nv_bfloat16* activation, float* output) {
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x & (kWarp - 1);
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * (kBf16Threads / kWarp) +
      static_cast<std::size_t>(warp);
  if (row >= rows) return;
  const std::uint8_t* row_weights = weights + row * (columns / 32) * 34;
  float sum = 0.0F;
  for (std::size_t column = static_cast<std::size_t>(lane); column < columns;
       column += kWarp) {
    const std::uint8_t* block = row_weights + (column / 32) * 34;
    const float weight =
        read_q8_half(block) *
        static_cast<float>(static_cast<std::int8_t>(block[2 + column % 32]));
    sum = __fadd_rn(sum,
                    __fmul_rn(weight, __bfloat162float(activation[column])));
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    sum = __fadd_rn(sum, __shfl_down_sync(0xFFFFFFFFU, sum, offset, kWarp));
  }
  if (lane == 0) output[row] = sum;
}

template <int WarpsPerRow>
cudaError_t launch_coop(const std::uint8_t* weights, std::size_t rows,
                        std::size_t columns, const void* staged, float* output,
                        cudaStream_t stream) {
  dim3 block(kWarp, WarpsPerRow);
  const unsigned int grid = static_cast<unsigned int>(rows);
  q8_coop_mmv<WarpsPerRow><<<grid, block, 0, stream>>>(
      weights, rows, columns, static_cast<const Q8_1Block*>(staged), output);
  return cudaPeekAtLastError();
}

}  // namespace q8_dots

}  // namespace qw38::cuda

#endif  // __CUDACC__
