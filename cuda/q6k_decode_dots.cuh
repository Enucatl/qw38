#pragma once

// OPT-048 cooperative packed Q6_K integer vocabulary dots.
// Technique from pinned llama.cpp mmvq.cu::mul_mat_vec_q and
// vecdotq.cuh::vec_dot_q6_K_q8_1 (cc83d7b, MIT). Quartz-owned rewrite:
// 2-byte-aligned ql/qh/d loads (210-byte blocks are not 16-byte aligned),
// DP4A of four int8 products with 6-bit offset -32, signed int8 subscales
// on the 16-value boundary, K distributed across warps per row. Do not
// vendor ggml headers. Packed FP32 quant_mmv remains the strict path.

#include <cstring>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "q6k_decode_path.cuh"
#include "quant_mmv.h"

#define QW38_SKIP_Q4K_DOT_KERNELS
#include "q4k_decode_dots.cuh"

namespace qw38::cuda {

#if defined(__CUDACC__) && !defined(QW38_SKIP_Q6K_DOT_KERNELS)

namespace q6k_dots {

constexpr int kWarp = 32;
constexpr int kQ6KBytes = 210;
constexpr int kValuesPerBlock = 256;
constexpr int kQR6 = 2;
constexpr int kQI6 = 32;
constexpr int kQI8 = 8;

// 210-byte Q6_K blocks are 2-byte aligned, not 16-byte or necessarily
// 4-byte aligned. Match llama get_int_b2: two 16-bit loads.
__device__ __forceinline__ int get_int_b2(const void* x, int i32) {
  const std::uint16_t* x16 = static_cast<const std::uint16_t*>(x);
  return static_cast<int>(x16[2 * i32]) |
         (static_cast<int>(x16[2 * i32 + 1]) << 16);
}

__device__ __forceinline__ float read_half_b2(const std::uint8_t* bytes) {
  const unsigned short bits =
      *reinterpret_cast<const unsigned short*>(bytes);
  return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ int get_int_aligned(const void* x, int i32) {
  return static_cast<const int*>(x)[i32];
}

// llama vec_dot_q6_K_q8_1 / vec_dot_q6_K_q8_1_impl_mmvq. iqs is the
// thread's 32-bit ql index (0..31). Packed signedness: 6-bit values are
// (low4 | high2<<4) - 32; subscales are signed int8, one per 16 values.
template <bool UseQ81>
__device__ __forceinline__ float vec_dot_q6k_q8(const std::uint8_t* block,
                                                const void* staged, int iqs) {
  const int vl = get_int_b2(block, iqs);
  const int vh_index = (kQI6 / 4) * (iqs / (kQI6 / 2)) + iqs % (kQI6 / 4);
  const int vh_shift = 2 * ((iqs % (kQI6 / 2)) / (kQI6 / 4));
  const int vh = get_int_b2(block + 128, vh_index) >> vh_shift;
  const int scale_offset =
      (kQI6 / 4) * (iqs / (kQI6 / 2)) + (iqs % (kQI6 / 2)) / (kQI6 / 8);
  const std::int8_t* scales =
      reinterpret_cast<const std::int8_t*>(block + 192) + scale_offset;
  const float d = read_half_b2(block + 208);
  const int bq8_offset =
      2 * kQR6 * (iqs / (kQI6 / 2)) + (iqs % (kQI6 / 2)) / (kQI6 / 4);
  float sumf = 0.0F;
#pragma unroll
  for (int i = 0; i < kQR6; ++i) {
    const int sc = static_cast<int>(scales[4 * i]);
    const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
    const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
    const int vi = static_cast<int>(
        __vsubss4(static_cast<unsigned int>(vil | vih), 0x20202020U));
    int u = 0;
    float d8 = 0.0F;
    if constexpr (UseQ81) {
      const Q8_1Block* q8 =
          static_cast<const Q8_1Block*>(staged) + bq8_offset + 2 * i;
      u = get_int_aligned(q8->values, iqs % kQI8);
      d8 = __half2float(q8->scale);
    } else {
      const Q8Block* q8 =
          static_cast<const Q8Block*>(staged) + bq8_offset + 2 * i;
      u = get_int_aligned(q8->values, iqs % kQI8);
      d8 = q8->scale;
    }
    sumf += d8 * static_cast<float>(__dp4a(vi, u, 0) * sc);
  }
  return d * sumf;
}

template <int WarpsPerRow, bool UseQ81>
__global__ void q6k_coop_mmv(const std::uint8_t* weights, std::size_t rows,
                             std::size_t columns, const void* staged,
                             float* output) {
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) return;

  const std::size_t n_blocks = columns / kValuesPerBlock;
  const std::uint8_t* row_weights = weights + row * n_blocks * kQ6KBytes;
  const std::size_t group_bytes =
      UseQ81 ? sizeof(Q8_1Block) : sizeof(Q8Block);
  const char* staged_bytes = static_cast<const char*>(staged);
  float acc = 0.0F;
  for (std::size_t kbx = static_cast<std::size_t>(warp); kbx < n_blocks;
       kbx += static_cast<std::size_t>(WarpsPerRow)) {
    const std::uint8_t* block = row_weights + kbx * kQ6KBytes;
    const void* block_staged = staged_bytes + kbx * 8U * group_bytes;
    acc += vec_dot_q6k_q8<UseQ81>(block, block_staged, lane);
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

template <int WarpsPerRow, bool UseQ81>
cudaError_t launch_coop(const std::uint8_t* weights, std::size_t rows,
                        std::size_t columns, const void* staged, float* output,
                        cudaStream_t stream) {
  dim3 block(kWarp, WarpsPerRow);
  const unsigned int grid = static_cast<unsigned int>(rows);
  q6k_coop_mmv<WarpsPerRow, UseQ81>
      <<<grid, block, 0, stream>>>(weights, rows, columns, staged, output);
  return cudaPeekAtLastError();
}

}  // namespace q6k_dots

#endif  // __CUDACC__

}  // namespace qw38::cuda
