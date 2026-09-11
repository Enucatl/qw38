#pragma once

// OPT-046 cooperative packed Q4_K integer decode dots.
// Technique from pinned llama.cpp mmvq.cu::mul_mat_vec_q and
// vecdotq.cuh::vec_dot_q4_K_q8_1 (cc83d7b, MIT). Quartz-owned rewrite:
// 32-bit qs loads, DP4A of four int8 products, K distributed across warps
// per row, independent lane accumulators until row completion, Q8 group
// sums staged once. Do not vendor ggml headers.

#include <cstdint>
#include <cstring>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "q4k_decode_path.cuh"
#include "quant_mmv.h"

namespace qw38::cuda {

// OPT-046/OPT-059 Q8_1 staging. Distinct from FP32-scale Q8Block.
// Layout matches pinned llama block_q8_1: half d, half s, int8 qs[32].
// Existing Quartz kernel stores half(sum(integer quants)) in q8_sum
// (format id quartz_q8_1_sum_q). Pinned llama GPU quantize_q8_1 stores
// half(sum(original x)). Do not reinterpret existing staged buffers.
// Integers above 2048 lose units when stored as half. Kept out of
// quant_mmv.h so full_scheduler.cu does not pull cuda_fp16.h through the
// packed MMV public header (that TU is codegen-sensitive).
struct Q8_1Block {
  __half scale;
  __half q8_sum;
  std::int8_t values[32];
};
static_assert(sizeof(Q8_1Block) == 36, "unexpected Q8_1 block padding");

cudaError_t launch_quantize_bf16_q8_1(const __nv_bfloat16* activation,
                                      void* q8, std::size_t columns,
                                      cudaStream_t stream) noexcept;
cudaError_t launch_quantize_bf16_q8_1_sum_x(const __nv_bfloat16* activation,
                                            void* q8, std::size_t columns,
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

struct Q8PackRegs {
  float d8;
  int u0;
  int u1;
  int stored_q8_sum;
};

// Shared activation pack. Two Q4 weight paths reuse this load (llama.cpp
// mmvq.cu::mul_mat_vec_q has_fusion, cc83d7b, MIT): one y, two vx pointers.
template <bool UseQ81>
__device__ __forceinline__ Q8PackRegs load_q8_pack(const void* block_staged,
                                                  int group, int pack) {
  Q8PackRegs pack_regs{};
  std::int8_t u_lo[4];
  std::int8_t u_hi[4];
  if constexpr (UseQ81) {
    const Q8_1Block* q8 =
        static_cast<const Q8_1Block*>(block_staged) + group;
    pack_regs.d8 = __half2float(q8->scale);
    pack_regs.stored_q8_sum =
        static_cast<int>(rintf(__half2float(q8->q8_sum)));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      u_lo[i] = q8->values[pack * 4 + i];
      u_hi[i] = q8->values[16 + pack * 4 + i];
    }
  } else {
    const Q8Block* q8 = static_cast<const Q8Block*>(block_staged) + group;
    pack_regs.d8 = q8->scale;
    pack_regs.stored_q8_sum = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      u_lo[i] = q8->values[pack * 4 + i];
      u_hi[i] = q8->values[16 + pack * 4 + i];
    }
  }
  pack_regs.u0 = pack4_bytes(u_lo);
  pack_regs.u1 = pack4_bytes(u_hi);
  return pack_regs;
}

template <bool UseQ81>
__device__ __forceinline__ float vec_dot_q4k_group_loaded(
    const std::uint8_t* block, const Q8PackRegs& q8, int group, int pack) {
  const float d = read_half_bytes(block);
  const float dmin = read_half_bytes(block + 2);
  int scale = 0;
  int minimum = 0;
  q4k_scale_min(block + 4, group, &scale, &minimum);
  const std::uint8_t* qs = block + 16 + (group / 2) * 32;
  const int high = group & 1;
  std::int8_t q4_lo[4];
  std::int8_t q4_hi[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int e0 = pack * 4 + i;
    const int e1 = 16 + pack * 4 + i;
    const std::uint8_t b0 = qs[e0];
    const std::uint8_t b1 = qs[e1];
    q4_lo[i] = static_cast<std::int8_t>(high == 0 ? (b0 & 15) : (b0 >> 4));
    q4_hi[i] = static_cast<std::int8_t>(high == 0 ? (b1 & 15) : (b1 >> 4));
  }
  const int v0 = pack4_bytes(q4_lo) & 0x0F0F0F0F;
  const int v1 = pack4_bytes(q4_hi) & 0x0F0F0F0F;
  int dot = __dp4a(v1, q8.u1, __dp4a(v0, q8.u0, 0));
  int q8_sum = 0;
  if constexpr (UseQ81) {
    q8_sum = pack == 0 ? q8.stored_q8_sum : 0;
  } else {
    q8_sum = __dp4a(0x01010101, q8.u1, __dp4a(0x01010101, q8.u0, 0));
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
  return __fmul_rn(q8.d8, __fsub_rn(sd, md));
}

template <bool UseQ81>
__device__ __forceinline__ float vec_dot_q4k_group(const std::uint8_t* block,
                                                   const void* block_staged,
                                                   int group, int pack) {
  return vec_dot_q4k_group_loaded<UseQ81>(
      block, load_q8_pack<UseQ81>(block_staged, group, pack), group, pack);
}

// OPT-076 Q8Block sibling: 32-bit qs/q8 loads, nibble mask/shift, exact
// integer dot and min-correction sums, FP32 scale applied per same-scale
// subgroup. Lanes keep their contribution; the K loop does not shuffle.
// Unaligned qs falls back to byte unpack without changing the association.
__device__ __forceinline__ Q8PackRegs load_q8_pack_late(const void* block_staged,
                                                       int group, int pack) {
  Q8PackRegs pack_regs{};
  const Q8Block* q8 = static_cast<const Q8Block*>(block_staged) + group;
  pack_regs.d8 = q8->scale;
  pack_regs.stored_q8_sum = 0;
  const std::uintptr_t values_addr =
      reinterpret_cast<std::uintptr_t>(q8->values);
  if ((values_addr & 3U) == 0U) {
    const int* words = reinterpret_cast<const int*>(q8->values);
    pack_regs.u0 = __ldg(words + pack);
    pack_regs.u1 = __ldg(words + pack + 4);
  } else {
    std::int8_t u_lo[4];
    std::int8_t u_hi[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      u_lo[i] = q8->values[pack * 4 + i];
      u_hi[i] = q8->values[16 + pack * 4 + i];
    }
    pack_regs.u0 = pack4_bytes(u_lo);
    pack_regs.u1 = pack4_bytes(u_hi);
  }
  return pack_regs;
}

__device__ __forceinline__ float vec_dot_q4k_q8block_late(
    const std::uint8_t* block, const Q8PackRegs& q8, int group, int pack) {
  float d = 0.0F;
  float dmin = 0.0F;
  const std::uintptr_t block_addr = reinterpret_cast<std::uintptr_t>(block);
  if ((block_addr & 3U) == 0U) {
    const unsigned dm = __ldg(reinterpret_cast<const unsigned*>(block));
    d = __half2float(
        __ushort_as_half(static_cast<unsigned short>(dm & 0xFFFFU)));
    dmin = __half2float(
        __ushort_as_half(static_cast<unsigned short>(dm >> 16U)));
  } else {
    d = read_half_bytes(block);
    dmin = read_half_bytes(block + 2);
  }
  int scale = 0;
  int minimum = 0;
  q4k_scale_min(block + 4, group, &scale, &minimum);
  const std::uint8_t* qs = block + 16 + (group / 2) * 32;
  const int high = group & 1;
  int v0 = 0;
  int v1 = 0;
  const std::uintptr_t qs_addr = reinterpret_cast<std::uintptr_t>(qs);
  if ((qs_addr & 3U) == 0U) {
    const int* words = reinterpret_cast<const int*>(qs);
    const int word0 = __ldg(words + pack);
    const int word1 = __ldg(words + pack + 4);
    const int shift = high << 2;
    v0 = (word0 >> shift) & 0x0F0F0F0F;
    v1 = (word1 >> shift) & 0x0F0F0F0F;
  } else {
    std::int8_t q4_lo[4];
    std::int8_t q4_hi[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const std::uint8_t b0 = qs[pack * 4 + i];
      const std::uint8_t b1 = qs[16 + pack * 4 + i];
      q4_lo[i] = static_cast<std::int8_t>(high == 0 ? (b0 & 15) : (b0 >> 4));
      q4_hi[i] = static_cast<std::int8_t>(high == 0 ? (b1 & 15) : (b1 >> 4));
    }
    v0 = pack4_bytes(q4_lo) & 0x0F0F0F0F;
    v1 = pack4_bytes(q4_hi) & 0x0F0F0F0F;
  }
  const int dot = __dp4a(v1, q8.u1, __dp4a(v0, q8.u0, 0));
  const int q8_sum = __dp4a(0x01010101, q8.u1, __dp4a(0x01010101, q8.u0, 0));
  const float sd = __fmul_rn(__fmul_rn(d, static_cast<float>(scale)),
                             static_cast<float>(dot));
  const float md = __fmul_rn(__fmul_rn(dmin, static_cast<float>(minimum)),
                             static_cast<float>(q8_sum));
  return __fmul_rn(q8.d8, __fsub_rn(sd, md));
}

__device__ __forceinline__ __nv_bfloat16 admitted_swiglu(float gate, float up) {
  const float activated = gate / (1.0F + expf(-gate));
  return __float2bfloat16_rn(__fmul_rn(activated, up));
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

// Paired integer gate/up + SwiGLU. Two original GGUF pointers, one Q8Block
// pack load per K/group, separate FP32 accumulators, same row-completion
// reduction as q4k_coop_mmv, admitted silu(gate)*up rounded once to BF16.
// Does not fuse the following 32-value Q8 quant for down.
template <int WarpsPerRow>
__global__ void q4k_coop_gate_up_swiglu(const std::uint8_t* gate_weights,
                                        const std::uint8_t* up_weights,
                                        std::size_t rows, std::size_t columns,
                                        const void* staged,
                                        __nv_bfloat16* output) {
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  const bool live = row < rows;

  const int group = lane / 4;
  const int pack = lane % 4;
  const std::size_t n_blocks = columns / kValuesPerBlock;
  const std::size_t group_bytes = sizeof(Q8Block);
  const char* staged_bytes = static_cast<const char*>(staged);
  float acc_g = 0.0F;
  float acc_u = 0.0F;
  if (live) {
    const std::uint8_t* gate_row = gate_weights + row * n_blocks * kQ4KBytes;
    const std::uint8_t* up_row = up_weights + row * n_blocks * kQ4KBytes;
    for (std::size_t kbx = static_cast<std::size_t>(warp); kbx < n_blocks;
         kbx += static_cast<std::size_t>(WarpsPerRow)) {
      const void* block_staged = staged_bytes + kbx * 8U * group_bytes;
      const Q8PackRegs q8 = load_q8_pack<false>(block_staged, group, pack);
      acc_g += vec_dot_q4k_group_loaded<false>(gate_row + kbx * kQ4KBytes, q8,
                                               group, pack);
      acc_u += vec_dot_q4k_group_loaded<false>(up_row + kbx * kQ4KBytes, q8,
                                               group, pack);
    }
  }

  __shared__ float partial_g[WarpsPerRow][kWarp];
  __shared__ float partial_u[WarpsPerRow][kWarp];
  partial_g[warp][lane] = acc_g;
  partial_u[warp][lane] = acc_u;
  __syncthreads();
  if (live && warp == 0) {
    float sum_g = 0.0F;
    float sum_u = 0.0F;
#pragma unroll
    for (int other = 0; other < WarpsPerRow; ++other) {
      sum_g += partial_g[other][lane];
      sum_u += partial_u[other][lane];
    }
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
      sum_g += __shfl_down_sync(0xFFFFFFFFU, sum_g, offset, kWarp);
      sum_u += __shfl_down_sync(0xFFFFFFFFU, sum_u, offset, kWarp);
    }
    if (lane == 0) output[row] = admitted_swiglu(sum_g, sum_u);
  }
}

template <int WarpsPerRow>
__global__ void q4k_coop_mmv_late(const std::uint8_t* weights, std::size_t rows,
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
  const std::size_t group_bytes = sizeof(Q8Block);
  const char* staged_bytes = static_cast<const char*>(staged);
  float acc = 0.0F;
  for (std::size_t kbx = static_cast<std::size_t>(warp); kbx < n_blocks;
       kbx += static_cast<std::size_t>(WarpsPerRow)) {
    const std::uint8_t* block = row_weights + kbx * kQ4KBytes;
    const void* block_staged = staged_bytes + kbx * 8U * group_bytes;
    acc += vec_dot_q4k_q8block_late(
        block, load_q8_pack_late(block_staged, group, pack), group, pack);
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

template <int WarpsPerRow>
__global__ void q4k_coop_gate_up_swiglu_late(const std::uint8_t* gate_weights,
                                             const std::uint8_t* up_weights,
                                             std::size_t rows,
                                             std::size_t columns,
                                             const void* staged,
                                             __nv_bfloat16* output) {
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  const bool live = row < rows;

  const int group = lane / 4;
  const int pack = lane % 4;
  const std::size_t n_blocks = columns / kValuesPerBlock;
  const std::size_t group_bytes = sizeof(Q8Block);
  const char* staged_bytes = static_cast<const char*>(staged);
  float acc_g = 0.0F;
  float acc_u = 0.0F;
  if (live) {
    const std::uint8_t* gate_row = gate_weights + row * n_blocks * kQ4KBytes;
    const std::uint8_t* up_row = up_weights + row * n_blocks * kQ4KBytes;
    for (std::size_t kbx = static_cast<std::size_t>(warp); kbx < n_blocks;
         kbx += static_cast<std::size_t>(WarpsPerRow)) {
      const void* block_staged = staged_bytes + kbx * 8U * group_bytes;
      const Q8PackRegs q8 = load_q8_pack_late(block_staged, group, pack);
      acc_g += vec_dot_q4k_q8block_late(gate_row + kbx * kQ4KBytes, q8, group,
                                        pack);
      acc_u += vec_dot_q4k_q8block_late(up_row + kbx * kQ4KBytes, q8, group,
                                        pack);
    }
  }

  __shared__ float partial_g[WarpsPerRow][kWarp];
  __shared__ float partial_u[WarpsPerRow][kWarp];
  partial_g[warp][lane] = acc_g;
  partial_u[warp][lane] = acc_u;
  __syncthreads();
  if (live && warp == 0) {
    float sum_g = 0.0F;
    float sum_u = 0.0F;
#pragma unroll
    for (int other = 0; other < WarpsPerRow; ++other) {
      sum_g += partial_g[other][lane];
      sum_u += partial_u[other][lane];
    }
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
      sum_g += __shfl_down_sync(0xFFFFFFFFU, sum_g, offset, kWarp);
      sum_u += __shfl_down_sync(0xFFFFFFFFU, sum_u, offset, kWarp);
    }
    if (lane == 0) output[row] = admitted_swiglu(sum_g, sum_u);
  }
}

template <int WarpsPerRow>
cudaError_t launch_coop_late(const std::uint8_t* weights, std::size_t rows,
                             std::size_t columns, const void* staged,
                             float* output, cudaStream_t stream) {
  dim3 block(kWarp, WarpsPerRow);
  const unsigned int grid = static_cast<unsigned int>(rows);
  q4k_coop_mmv_late<WarpsPerRow>
      <<<grid, block, 0, stream>>>(weights, rows, columns, staged, output);
  return cudaPeekAtLastError();
}

template <int WarpsPerRow>
cudaError_t launch_coop_gate_up_late(const std::uint8_t* gate_weights,
                                     const std::uint8_t* up_weights,
                                     std::size_t rows, std::size_t columns,
                                     const void* staged, __nv_bfloat16* output,
                                     cudaStream_t stream) {
  dim3 block(kWarp, WarpsPerRow);
  const unsigned int grid = static_cast<unsigned int>(rows);
  q4k_coop_gate_up_swiglu_late<WarpsPerRow>
      <<<grid, block, 0, stream>>>(gate_weights, up_weights, rows, columns,
                                   staged, output);
  return cudaPeekAtLastError();
}

template <int WarpsPerRow>
cudaError_t launch_coop_gate_up(const std::uint8_t* gate_weights,
                                const std::uint8_t* up_weights,
                                std::size_t rows, std::size_t columns,
                                const void* staged, __nv_bfloat16* output,
                                cudaStream_t stream) {
  dim3 block(kWarp, WarpsPerRow);
  const unsigned int grid = static_cast<unsigned int>(rows);
  q4k_coop_gate_up_swiglu<WarpsPerRow>
      <<<grid, block, 0, stream>>>(gate_weights, up_weights, rows, columns,
                                   staged, output);
  return cudaPeekAtLastError();
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
    // quartz_q8_1_sum_q: half(sum(q)). The reduced original-x sum is unused.
    output[q8_index].q8_sum =
        __float2half_rn(static_cast<float>(q_sum));
    (void)sum;
  }
}

// Distinct llama GPU staging: half(sum(original x)), matching pinned
// ggml-cuda/quantize.cu::quantize_q8_1. Does not rewrite existing buffers.
__global__ void quantize_bf16_q8_1_sum_x(const __nv_bfloat16* input,
                                         Q8_1Block* output,
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
  if (index < count) output[q8_index].values[lane] = quant;
  if (lane == 0 && q8_index * kWarp < count) {
    output[q8_index].scale = __float2half_rn(scale);
    output[q8_index].q8_sum = __float2half_rn(sum);
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
