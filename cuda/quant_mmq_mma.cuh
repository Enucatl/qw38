#pragma once

// Q4_K / Q6_K / Q8_0 prompt MMQ with Ampere-style MMA tiles for Quartz.
// OPT-017 Rank-1 fused Q8_0 (1D 256 threads, K-step 32, SRAM Q8_1 staging,
// no q8_workspace) stays callable. OPT-018 production Q4_K/Q6_K is the
// quality kernel: MMQ_ITER_K=256, packed load-tiles, Q8_1 MMQ Y tiles in
// the existing prompt workspace, block dim3(32, 8). OPT-022 production
// mixer Q8_0 (keep) is the same quality stack with D4 quantize_mmq_q8_1
// and packed Q8_0 load-tiles; Rank-1 remains the visible fused reference
// and the reject restore path. OPT-023 may dispatch skinny mixer Q8_0
// (output_rows < 128) through I=32 or I=64 quality MMA when a paired
// CUDA-event A/B strictly beats I=128. OPT-024 may dispatch large mixer
// Q8_0 (output_rows >= 128) through an aligned-SoA D2R / int8 MMA path
// when a paired CUDA-event A/B strictly beats quality MMA; otherwise
// large GEMMs stay I=128 quality MMA. OPT-025 may reuse one Q4_K DS4 Q8_1
// of the FFN input for gate and up and write down-leg Y from SwiGLU
// without a global BF16 mid store when a paired CUDA-event A/B strictly
// beats per-GEMM quantize. OPT-028 may dispatch Q4_K/Q6_K quality MMA
// through llama.cpp-style stream-K tile decomposition plus optional
// mul_mat_q_stream_k_fixup when a paired CUDA-event A/B strictly beats
// 2D tiling; mixer Q8_0 stays on quality MMA. OPT-053 may dispatch
// quality MMA through explicit FMA scale accumulation and/or a two-stage
// cp.async packed-Y loader when a paired CUDA-event A/B strictly beats
// the production 2D tile; tails and misaligned tiles keep the
// synchronous loader. Output is token-major FP32
// [prompt_rows, output_rows]. Production Q8_0 is launched via
// launch_q8_mmq_bf16, not launch_quant_mmq.
//
// Provenance: llama.cpp mmq.cuh / mma.cuh / mmq-config-ampere.cuh /
// mmq-load-tiles.cuh / mmq-vec-dot.cuh / quantize.cu at
// cc83d7b4824f73cfdda4dfbb47ee39804f71b328 (MIT, The ggml authors),
// including mul_mat_q kbc walk and mul_mat_q_stream_k_fixup.
// ds4 cuda/mmq is the ggml-free launcher pattern only; this file does not
// copy ../ds4/cuda/mmq/ and does not include ggml headers.

#include "mma.cuh"
#include "pdl_launch.cuh"
#include "quant_mmv.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
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

constexpr std::size_t quality_shared_ints(int prompt_tile,
                                         int quality_i = kQualityI,
                                         bool async_y = false) noexcept {
  const std::size_t y_tile = quality_pad_ints(
      static_cast<std::size_t>(prompt_tile) * kMmqTileYk);
  return static_cast<std::size_t>(prompt_tile) +
         y_tile * (async_y ? 2U : 1U) +
         static_cast<std::size_t>(quality_i) * kQualitySramMax;
}

inline __device__ void mmq_cp_async16(void* dst, const void* src) {
  const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem), "l"(src));
}

inline __device__ void mmq_cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}

inline __device__ void mmq_cp_async_wait() {
  asm volatile("cp.async.wait_all;");
}

template <int PromptTile>
__device__ void mmq_load_y_async(int* dst, const int* src) {
  constexpr int kBytes = PromptTile * kMmqTileYk * static_cast<int>(sizeof(int));
  constexpr int kChunks = kBytes / 16;
  const int tid = threadIdx.y * 32 + threadIdx.x;
  for (int chunk = tid; chunk < kChunks; chunk += kQualityNwarps * 32) {
    mmq_cp_async16(reinterpret_cast<char*>(dst) + chunk * 16,
                   reinterpret_cast<const char*>(src) + chunk * 16);
  }
}

constexpr std::size_t soa_dq_bytes(std::size_t output_rows,
                                   std::size_t columns) noexcept {
  return output_rows * (columns / 32) * sizeof(half);
}

constexpr std::size_t soa_pad_bytes(std::size_t dq_bytes) noexcept {
  return (64U - (dq_bytes % 64U)) % 64U;
}

constexpr std::size_t soa_total_bytes(std::size_t output_rows,
                                      std::size_t columns) noexcept {
  const std::size_t dq = soa_dq_bytes(output_rows, columns);
  return dq + soa_pad_bytes(dq) + output_rows * columns;
}

__global__ void repack_q8_0_aligned_kernel(const std::uint8_t* weights,
                                           std::size_t output_rows,
                                           std::size_t columns,
                                           std::uint8_t* soa) {
  const std::size_t nb = columns / 32;
  const std::size_t nblk = output_rows * nb;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= nblk) return;
  const std::size_t row = index / nb;
  const std::size_t kb = index % nb;
  const std::uint8_t* src = weights + index * kMmaQ8Bytes;
  const std::size_t dq = soa_dq_bytes(output_rows, columns);
  const std::size_t pad = soa_pad_bytes(dq);
  reinterpret_cast<half*>(soa)[index] = *reinterpret_cast<const half*>(src);
  std::int8_t* qs = reinterpret_cast<std::int8_t*>(soa + dq + pad) +
                    row * columns + kb * 32;
#pragma unroll
  for (int lane = 0; lane < 32; ++lane) {
    qs[lane] = static_cast<std::int8_t>(src[2 + lane]);
  }
}

constexpr int quality_rows_per_warp(int prompt_tile) {
  return prompt_tile >= 48 && prompt_tile % 16 == 0 ? 32 : 16;
}

template <QuantKind Kind>
__global__ void quantize_mmq_q8_1_bf16(const __nv_bfloat16* prompt,
                                        std::size_t prompt_rows,
                                        std::size_t columns, std::uint8_t* y) {
  quartz_pdl_sync();
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
  if (index >= count) {
    quartz_pdl_lc();
    return;
  }
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
  quartz_pdl_lc();
}

template <QuantKind Kind>
__global__ void quantize_mmq_q8_1_swiglu(const float* gate, const float* up,
                                         std::size_t prompt_rows,
                                         std::size_t columns, std::uint8_t* y) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x & 31;
  const std::size_t count = prompt_rows * columns;
  float activated = 0.0F;
  if (index < count) {
    const float g = gate[index];
    activated = (g / (1.0F + expf(-g))) * up[index];
  }
  const float value =
      index < count ? __bfloat162float(__float2bfloat16_rn(activated)) : 0.0F;
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
  const int q = scale == 0.0F ? 0 : static_cast<int>(roundf(value / scale));
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

template <QuantKind Kind, int PromptTile, bool Fallback, int QualityI = kQualityI,
          bool SoaWeights = false, bool UseFma = false, bool UseAsyncY = false>
__device__ __forceinline__ void quality_mma_process_tile(
    const std::uint8_t* weights, std::size_t output_rows, std::size_t columns,
    const int* y, std::size_t prompt_rows, float* output, float* tmp_fixup,
    std::size_t out0, std::size_t prompt0, int i_max, int j_max, int kb0_start,
    int kb0_stop, bool write_fixup) {
  constexpr int kRowsPerWarp = quality_rows_per_warp(PromptTile);
  constexpr int kNtx = kQualityNwarps * kRowsPerWarp / QualityI;
  constexpr int kNi = kRowsPerWarp / 16;
  static_assert(QualityI > 0 && QualityI % 16 == 0,
                "QualityI must be a multiple of 16");
  static_assert(kNtx > 0 && (kQualityNwarps * kRowsPerWarp) % QualityI == 0,
                "QualityI must tile the warp I mapping");
  static_assert(kNi > 0 && kRowsPerWarp % 16 == 0,
                "warp I coverage must be 16-row MMA tiles");
  static_assert((PromptTile * QualityI) % (kQualityNwarps * 32) == 0,
                "kSum must be integral");
  static_assert(!SoaWeights || Kind == QuantKind::kQ8_0,
                "aligned SoA D2R is Q8_0 only");
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
  const half* soa_d = nullptr;
  const std::int8_t* soa_q = nullptr;
  if constexpr (SoaWeights) {
    const std::size_t dq = soa_dq_bytes(output_rows, columns);
    soa_d = reinterpret_cast<const half*>(weights);
    soa_q = reinterpret_cast<const std::int8_t*>(
        weights + dq + soa_pad_bytes(dq));
  }

  extern __shared__ std::uint8_t shared[];
  int* quality_shared = reinterpret_cast<int*>(shared);
  constexpr std::size_t kYInts = quality_pad_ints(
      static_cast<std::size_t>(PromptTile) * kMmqTileYk);
  constexpr bool kAsyncY = UseAsyncY && !Fallback;
  int* tile_y0 = quality_shared + PromptTile;
  int* tile_y = tile_y0;
  int* tile_x = tile_y0 + kYInts * (kAsyncY ? 2 : 1);
  int* x_qs = tile_x;
  [[maybe_unused]] float* x_df = nullptr;
  [[maybe_unused]] int* x_sc = nullptr;
  if constexpr (Kind != QuantKind::kQ4K) {
    x_df = reinterpret_cast<float*>(x_qs + 2 * kMmqTileNeK);
    if constexpr (Kind == QuantKind::kQ6K) {
      x_sc = reinterpret_cast<int*>(x_df + kMmqTileNeK / kQi6K);
    }
  }

  constexpr int kSum = PromptTile * QualityI / (kQualityNwarps * 32);
  float sum[kSum];
#pragma unroll
  for (int s = 0; s < kSum; ++s) sum[s] = 0.0F;

  const int tid = threadIdx.y * 32 + threadIdx.x;
  const int lane = threadIdx.x;
  const std::uint8_t* weight_row0 =
      weights + out0 * row_stride;
  [[maybe_unused]] int y_stage = 0;
  __syncthreads();

  if constexpr (kAsyncY) {
    const int* y0 =
        y + (static_cast<std::size_t>(kb0_start) * 2 * prompt_rows + prompt0) *
                kMmqTileYk;
    mmq_load_y_async<PromptTile>(tile_y0, y0);
    mmq_cp_async_commit();
    mmq_cp_async_wait();
    __syncthreads();
  }

  for (int kb0 = kb0_start; kb0 < kb0_stop; ++kb0) {
    if constexpr (Kind == QuantKind::kQ4K) {
#pragma unroll
      for (int i0 = 0; i0 < QualityI; i0 += kQualityNwarps) {
        int i = i0 + threadIdx.y;
        if (Fallback) i = min(i, max(i_max, 0));
        const std::size_t global_row = out0 + static_cast<std::size_t>(i);
        int qs0 = 0;
        if (!Fallback || global_row < output_rows) {
          const std::uint8_t* block =
              weight_row0 + static_cast<std::size_t>(i) * row_stride +
              kb0 * kWeightBytes;
          qs0 = quality_get_int_b4(block + 16, lane);
        }
        x_qs[i * kSram + 16 * (lane / 8) + (lane % 8) + 0] =
            (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i * kSram + 16 * (lane / 8) + (lane % 8) + 8] =
            (qs0 >> 4) & 0x0F0F0F0F;
      }
#pragma unroll
      for (int i0 = 0; i0 < QualityI; i0 += kQualityNwarps * 16) {
        int i = (i0 + threadIdx.y * 16 + lane / 2) % QualityI;
        if (Fallback) i = min(i, max(i_max, 0));
        const std::size_t global_row = out0 + static_cast<std::size_t>(i);
        half2* dm = reinterpret_cast<half2*>(
            x_qs + i * kSram + 2 * kMmqTileNeK);
        const int ksc = lane % 2;
        if (!Fallback || global_row < output_rows) {
          const std::uint8_t* block =
              weight_row0 + static_cast<std::size_t>(i) * row_stride +
              kb0 * kWeightBytes;
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
      for (int i0 = 0; i0 < QualityI; i0 += kQualityNwarps) {
        int i = i0 + threadIdx.y;
        if (Fallback) i = min(i, max(i_max, 0));
        const std::size_t global_row = out0 + static_cast<std::size_t>(i);
        int ql0 = 0;
        int ql1 = 0;
        int qh0 = 0;
        int qh1 = 0;
        if (!Fallback || global_row < output_rows) {
          const std::uint8_t* block =
              weight_row0 + static_cast<std::size_t>(i) * row_stride +
              kb0 * kWeightBytes;
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
      for (int i0 = 0; i0 < QualityI; i0 += kQualityNwarps * 32) {
        int i = (i0 + threadIdx.y * 32 + lane) % QualityI;
        if (Fallback) i = min(i, max(i_max, 0));
        const std::size_t global_row = out0 + static_cast<std::size_t>(i);
        float d = 0.0F;
        if (!Fallback || global_row < output_rows) {
          const std::uint8_t* block =
              weight_row0 + static_cast<std::size_t>(i) * row_stride +
              kb0 * kWeightBytes;
          d = mma_read_half(block + 208);
        }
        x_df[i * kSram] = d;
      }
#pragma unroll
      for (int i0 = 0; i0 < QualityI; i0 += kQualityNwarps * 8) {
        int i = (i0 + threadIdx.y * 8 + lane / 4) % QualityI;
        if (Fallback) i = min(i, max(i_max, 0));
        const std::size_t global_row = out0 + static_cast<std::size_t>(i);
        int sc = 0;
        if (!Fallback || global_row < output_rows) {
          const std::uint8_t* block =
              weight_row0 + static_cast<std::size_t>(i) * row_stride +
              kb0 * kWeightBytes;
          sc = quality_get_int_b2(block + 192, lane % 4);
        }
        x_sc[i * kSram + (lane % 4)] = sc;
      }
    } else {
      const int kbx = lane / kQi80;
      const int kqsx = lane % kQi80;
#pragma unroll
      for (int i0 = 0; i0 < QualityI; i0 += kQualityNwarps) {
        int i = i0 + threadIdx.y;
        if (Fallback) i = min(i, max(i_max, 0));
        const std::size_t global_row = out0 + static_cast<std::size_t>(i);
        int qs0 = 0;
        int qs1 = 0;
        if (!Fallback || global_row < output_rows) {
          if constexpr (SoaWeights) {
            const std::int8_t* row0 =
                soa_q + global_row * columns +
                (kb0 * 8 + static_cast<std::size_t>(kbx)) * 32;
            const std::int8_t* row1 =
                soa_q + global_row * columns +
                (kb0 * 8 + 4 + static_cast<std::size_t>(kbx)) * 32;
            qs0 = reinterpret_cast<const int*>(row0)[kqsx];
            qs1 = reinterpret_cast<const int*>(row1)[kqsx];
          } else {
            const std::uint8_t* row =
                weight_row0 + static_cast<std::size_t>(i) * row_stride +
                kb0 * 8 * kWeightBytes;
            qs0 = quality_get_int_b2(row + kbx * kWeightBytes + 2, kqsx);
            qs1 = quality_get_int_b2(
                row + (4 + kbx) * kWeightBytes + 2, kqsx);
          }
        }
        x_qs[i * kSram + lane] = qs0;
        x_qs[i * kSram + kMmqTileNeK + lane] = qs1;
      }
#pragma unroll
      for (int i0 = 0; i0 < QualityI; i0 += kQualityNwarps * 4) {
        int i = i0 + threadIdx.y * 4 + lane / 8;
        if (Fallback) i = min(i, max(i_max, 0));
        const std::size_t global_row = out0 + static_cast<std::size_t>(i);
        const int kbxd = lane % 8;
        float scale = 0.0F;
        if (!Fallback || global_row < output_rows) {
          if constexpr (SoaWeights) {
            scale = __half2float(
                soa_d[global_row * (columns / 32) + kb0 * 8 +
                      static_cast<std::size_t>(kbxd)]);
          } else {
            const std::uint8_t* block =
                weight_row0 + static_cast<std::size_t>(i) * row_stride +
                (kb0 * 8 + static_cast<std::size_t>(kbxd)) * kWeightBytes;
            scale = mma_read_half(block);
          }
        }
        x_df[i * kSram + kbxd] = scale;
      }
    }
    if constexpr (kAsyncY) __syncthreads();

#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const std::size_t k_block = kb0 * 2 + static_cast<std::size_t>(half);
      if constexpr (kAsyncY) {
        tile_y = tile_y0 + static_cast<std::size_t>(y_stage) * kYInts;
        const int next_half = half == 0 ? 1 : 0;
        const int next_kb0 = half == 0 ? kb0 : kb0 + 1;
        if (next_kb0 < kb0_stop) {
          const std::size_t next_block =
              static_cast<std::size_t>(next_kb0) * 2 +
              static_cast<std::size_t>(next_half);
          mmq_load_y_async<PromptTile>(
              tile_y0 + static_cast<std::size_t>(1 - y_stage) * kYInts,
              y + (next_block * prompt_rows + prompt0) * kMmqTileYk);
          mmq_cp_async_commit();
        }
      } else if (Fallback) {
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
      if constexpr (!kAsyncY) __syncthreads();

      const int k00 = half * kMmqTileNeK;
      const int* y_tile =
          tile_y + (threadIdx.y % kNtx) * (8 * kMmqTileYk);
      const int* y_qs = y_tile + 4;
      const int i0 = (threadIdx.y / kNtx) * kRowsPerWarp;

      if constexpr (Kind == QuantKind::kQ4K) {
        const half2* y_dm = reinterpret_cast<const half2*>(y_tile);
        int A[kNi][4][4];
        float2 dmA[kNi][2][4];
#pragma unroll
        for (int n = 0; n < kNi; ++n) {
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
            for (int n = 0; n < kNi; ++n) {
              int C[4] = {0, 0, 0, 0};
              mma::mma_m16n8k32_s8(C, A[n][k01 / kQi81], B);
#pragma unroll
              for (int l = 0; l < 4; ++l) {
                float& acc = sum[(j0 / (kNtx * 8) * kNi + n) * 4 + l];
                const float mag = static_cast<float>(C[l]);
                const float sx = dmA[n][l / 2][k01 / kQi81].x * dsB[l % 2].x;
                const float sy = dmA[n][l / 2][k01 / kQi81].y;
                const float dy = dsB[l % 2].y;
                if constexpr (UseFma) {
                  acc = __fmaf_rn(sx, mag, __fmaf_rn(sy, dy, acc));
                } else {
                  acc += sx * mag + sy * dy;
                }
              }
            }
          }
        }
      } else if constexpr (Kind == QuantKind::kQ6K) {
        const float* y_df = reinterpret_cast<const float*>(y_tile);
        int A[kNi][8][2];
        int scA[kNi][2][8];
        float dA[kNi][2];
#pragma unroll
        for (int n = 0; n < kNi; ++n) {
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
          float tmp[kNi][4];
#pragma unroll
          for (int n = 0; n < kNi; ++n) {
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
            for (int n = 0; n < kNi; ++n) {
              int C0[4] = {0, 0, 0, 0};
              int C1[4] = {0, 0, 0, 0};
              mma::mma_m16n8k16_s8(C0, A[n][k01 / 4 + 0], B0);
              mma::mma_m16n8k16_s8(C1, A[n][k01 / 4 + 1], B1);
#pragma unroll
              for (int l = 0; l < 4; ++l) {
                const float mag = static_cast<float>(
                    C0[l] * scA[n][l / 2][k01 / 4 + 0] +
                    C1[l] * scA[n][l / 2][k01 / 4 + 1]);
                if constexpr (UseFma) {
                  tmp[n][l] = __fmaf_rn(mag, dB[l % 2], tmp[n][l]);
                } else {
                  tmp[n][l] += mag * dB[l % 2];
                }
              }
            }
          }
#pragma unroll
          for (int n = 0; n < kNi; ++n) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
              float& acc = sum[(j0 / (kNtx * 8) * kNi + n) * 4 + l];
              if constexpr (UseFma) {
                acc = __fmaf_rn(tmp[n][l], dA[n][l / 2], acc);
              } else {
                acc += tmp[n][l] * dA[n][l / 2];
              }
            }
          }
        }
      } else {
        const float* y_df = reinterpret_cast<const float*>(y_tile);
        int A[kNi][4][4];
        float dA[kNi][2][4];
#pragma unroll
        for (int n = 0; n < kNi; ++n) {
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
            for (int n = 0; n < kNi; ++n) {
              int C[4] = {0, 0, 0, 0};
              mma::mma_m16n8k32_s8(C, A[n][k01 / kQi80], B);
#pragma unroll
              for (int l = 0; l < 4; ++l) {
                float& acc = sum[(j0 / (kNtx * 8) * kNi + n) * 4 + l];
                const float mag = static_cast<float>(C[l]) *
                                  dA[n][l / 2][k01 / kQi80];
                if constexpr (UseFma) {
                  acc = __fmaf_rn(mag, dB[l % 2], acc);
                } else {
                  acc += mag * dB[l % 2];
                }
              }
            }
          }
        }
      }
      if constexpr (kAsyncY) {
        const int next_kb0 = half == 0 ? kb0 : kb0 + 1;
        if (next_kb0 < kb0_stop) mmq_cp_async_wait();
        __syncthreads();
        y_stage = 1 - y_stage;
      } else {
        __syncthreads();
      }
    }
  }

#pragma unroll
  for (int j0 = 0; j0 < PromptTile; j0 += kNtx * 8) {
#pragma unroll
    for (int n = 0; n < kNi; ++n) {
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
        if (write_fixup) {
          tmp_fixup[static_cast<std::size_t>(blockIdx.x) *
                        (PromptTile * QualityI) +
                    static_cast<std::size_t>(j) * QualityI +
                    static_cast<std::size_t>(row)] =
              sum[(j0 / (kNtx * 8) * kNi + n) * 4 + l];
        } else if (out_row < output_rows && prompt_row < prompt_rows) {
          output[prompt_row * output_rows + out_row] =
              sum[(j0 / (kNtx * 8) * kNi + n) * 4 + l];
        }
      }
    }
  }
}

template <QuantKind Kind, int PromptTile, bool Fallback, int QualityI = kQualityI,
          bool SoaWeights = false, bool UseFma = false, bool UseAsyncY = false>
__global__ void __launch_bounds__(256, 1) quant_mmq_mma_quality_kernel(
    const std::uint8_t* weights, std::size_t output_rows, std::size_t columns,
    const int* y, std::size_t prompt_rows, float* output) {
  quartz_pdl_sync();
  const std::size_t out0 =
      static_cast<std::size_t>(blockIdx.x) * QualityI;
  const std::size_t prompt0 =
      static_cast<std::size_t>(blockIdx.y) * PromptTile;
  const int i_max =
      Fallback ? static_cast<int>(output_rows - out0) - 1 : QualityI - 1;
  const int j_max =
      Fallback ? static_cast<int>(prompt_rows - prompt0) - 1 : PromptTile - 1;
  quality_mma_process_tile<Kind, PromptTile, Fallback, QualityI, SoaWeights,
                           UseFma, UseAsyncY>(
      weights, output_rows, columns, y, prompt_rows, output, nullptr, out0,
      prompt0, i_max, j_max, 0, static_cast<int>(columns / kMmaBlockValues),
      false);
  quartz_pdl_lc();
}

template <QuantKind Kind, int PromptTile, bool Fallback, int QualityI = kQualityI>
__global__ void __launch_bounds__(256, 1)
    quant_mmq_mma_quality_stream_k_kernel(
        const std::uint8_t* weights, std::size_t output_rows,
        std::size_t columns, const int* y, std::size_t prompt_rows,
        float* output, float* tmp_fixup, int ntx, int nty, int nblocks) {
  const int blocks_per_ne00 = static_cast<int>(columns / kMmaBlockValues);
  const std::int64_t total = static_cast<std::int64_t>(ntx) *
                             static_cast<std::int64_t>(nty) *
                             static_cast<std::int64_t>(blocks_per_ne00);
  int kbc = static_cast<int>(static_cast<std::int64_t>(blockIdx.x) * total /
                             static_cast<std::int64_t>(nblocks));
  int kbc_stop =
      static_cast<int>(static_cast<std::int64_t>(blockIdx.x + 1) * total /
                       static_cast<std::int64_t>(nblocks));
  int kb0_start = kbc % blocks_per_ne00;
  int kb0_stop = min(blocks_per_ne00, kb0_start + kbc_stop - kbc);
  while (kbc < kbc_stop && kb0_stop == blocks_per_ne00) {
    const int tile = kbc / blocks_per_ne00;
    const int jt = tile % ntx;
    const int it = tile / ntx;
    const std::size_t out0 = static_cast<std::size_t>(it) * QualityI;
    const std::size_t prompt0 = static_cast<std::size_t>(jt) * PromptTile;
    const int i_max =
        Fallback ? static_cast<int>(output_rows - out0) - 1 : QualityI - 1;
    const int j_max =
        Fallback ? static_cast<int>(prompt_rows - prompt0) - 1 : PromptTile - 1;
    quality_mma_process_tile<Kind, PromptTile, Fallback, QualityI, false>(
        weights, output_rows, columns, y, prompt_rows, output, tmp_fixup, out0,
        prompt0, i_max, j_max, kb0_start, kb0_stop, false);
    kbc += blocks_per_ne00;
    kbc -= kbc % blocks_per_ne00;
    kb0_start = 0;
    kb0_stop = min(blocks_per_ne00, kbc_stop - kbc);
  }
  if (kbc >= kbc_stop) return;
  const int tile = kbc / blocks_per_ne00;
  const int jt = tile % ntx;
  const int it = tile / ntx;
  const std::size_t out0 = static_cast<std::size_t>(it) * QualityI;
  const std::size_t prompt0 = static_cast<std::size_t>(jt) * PromptTile;
  const int i_max =
      Fallback ? static_cast<int>(output_rows - out0) - 1 : QualityI - 1;
  const int j_max =
      Fallback ? static_cast<int>(prompt_rows - prompt0) - 1 : PromptTile - 1;
  quality_mma_process_tile<Kind, PromptTile, Fallback, QualityI, false>(
      weights, output_rows, columns, y, prompt_rows, output, tmp_fixup, out0,
      prompt0, i_max, j_max, kb0_start, kb0_stop, true);
}

template <int PromptTile, bool Fallback, int QualityI = kQualityI>
__global__ void __launch_bounds__(128, 1) quant_mmq_mma_stream_k_fixup_kernel(
    float* dst, const float* tmp_fixup, std::size_t output_rows,
    std::size_t prompt_rows, int ntx, int nty, int nblocks,
    int blocks_per_ne00) {
  constexpr int kNwarps = 4;
  float sum[PromptTile / kNwarps];
#pragma unroll
  for (int s = 0; s < PromptTile / kNwarps; ++s) sum[s] = 0.0F;
  const int i = static_cast<int>(blockIdx.y) * 32 + static_cast<int>(threadIdx.x);
  const std::int64_t total = static_cast<std::int64_t>(ntx) *
                             static_cast<std::int64_t>(nty) *
                             static_cast<std::int64_t>(blocks_per_ne00);
  const int bidx0 = static_cast<int>(blockIdx.x);
  int kbc0 = static_cast<int>(static_cast<std::int64_t>(blockIdx.x) * total /
                              static_cast<std::int64_t>(nblocks));
  int kbc0_stop =
      static_cast<int>(static_cast<std::int64_t>(blockIdx.x + 1) * total /
                       static_cast<std::int64_t>(nblocks));
  const bool did_not_have_any_data = kbc0 == kbc0_stop;
  const bool wrote_beginning_of_tile = (kbc0 % blocks_per_ne00) == 0;
  const bool did_not_write_last =
      (kbc0 / blocks_per_ne00) == (kbc0_stop / blocks_per_ne00) &&
      (kbc0_stop % blocks_per_ne00) != 0;
  if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
    return;
  }
  bool any_fixup = false;
  int bidx = bidx0 - 1;
  int kbc_stop = kbc0;
  int kbc = 0;
  while (true) {
    kbc = static_cast<int>(static_cast<std::int64_t>(bidx) * total /
                           static_cast<std::int64_t>(nblocks));
    if (kbc == kbc_stop) {
      --bidx;
      kbc_stop = kbc;
      continue;
    }
    any_fixup = true;
#pragma unroll
    for (int j0 = 0; j0 < PromptTile; j0 += kNwarps) {
      const int j = j0 + static_cast<int>(threadIdx.y);
      sum[j0 / kNwarps] +=
          tmp_fixup[static_cast<std::size_t>(bidx) * (PromptTile * QualityI) +
                    static_cast<std::size_t>(j) * QualityI +
                    static_cast<std::size_t>(i)];
    }
    if ((kbc % blocks_per_ne00) == 0 ||
        (kbc / blocks_per_ne00) < (kbc0 / blocks_per_ne00)) {
      break;
    }
    --bidx;
    kbc_stop = kbc;
  }
  if (!any_fixup) return;
  const int tile = kbc0 / blocks_per_ne00;
  const int jt = tile % ntx;
  const int it = tile / ntx;
  const std::size_t out0 = static_cast<std::size_t>(it) * QualityI;
  const std::size_t prompt0 = static_cast<std::size_t>(jt) * PromptTile;
  const int i_max =
      Fallback ? static_cast<int>(output_rows - out0) - 1 : QualityI - 1;
  const int j_max =
      Fallback ? static_cast<int>(prompt_rows - prompt0) - 1 : PromptTile - 1;
  if (Fallback && i > i_max) return;
#pragma unroll
  for (int j0 = 0; j0 < PromptTile; j0 += kNwarps) {
    const int j = j0 + static_cast<int>(threadIdx.y);
    if (j > j_max) return;
    const std::size_t out_row = out0 + static_cast<std::size_t>(i);
    const std::size_t prompt_row = prompt0 + static_cast<std::size_t>(j);
    if (out_row < output_rows && prompt_row < prompt_rows) {
      dst[prompt_row * output_rows + out_row] += sum[j0 / kNwarps];
    }
  }
}

template <QuantKind Kind, int PromptTile, bool Fallback, int QualityI = kQualityI>
cudaError_t launch_quality_mma_stream_k(
    const std::uint8_t* weights, std::size_t output_rows, std::size_t columns,
    const int* y, std::size_t prompt_rows, float* output, float* tmp_fixup,
    int nblocks, bool fixup_needed, cudaStream_t stream) noexcept {
  const int nty =
      static_cast<int>((output_rows + QualityI - 1) / QualityI);
  const int ntx =
      static_cast<int>((prompt_rows + PromptTile - 1) / PromptTile);
  const dim3 grid(static_cast<unsigned int>(nblocks), 1);
  const dim3 block(32, 8);
  const std::size_t shared =
      quality_shared_ints(PromptTile, QualityI) * sizeof(int);
  auto kernel =
      quant_mmq_mma_quality_stream_k_kernel<Kind, PromptTile, Fallback,
                                            QualityI>;
  cudaError_t error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shared));
  if (error != cudaSuccess) return error;
  kernel<<<grid, block, shared, stream>>>(weights, output_rows, columns, y,
                                          prompt_rows, output, tmp_fixup, ntx,
                                          nty, nblocks);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess || !fixup_needed) return error;
  const dim3 fixup_grid(static_cast<unsigned int>(nblocks),
                        static_cast<unsigned int>(QualityI / 32));
  const dim3 fixup_block(32, 4);
  quant_mmq_mma_stream_k_fixup_kernel<PromptTile, Fallback, QualityI>
      <<<fixup_grid, fixup_block, 0, stream>>>(
          output, tmp_fixup, output_rows, prompt_rows, ntx, nty, nblocks,
          static_cast<int>(columns / kMmaBlockValues));
  return cudaPeekAtLastError();
}

bool mmq_stream_k_path_on(const char* path) noexcept {
  return path != nullptr &&
         (std::strcmp(path, "stream_k") == 0 ||
          std::strcmp(path, "stream_k_nsm") == 0);
}

template <QuantKind Kind, int PromptTile, bool Fallback, int QualityI = kQualityI,
          bool SoaWeights = false, bool UseFma = false, bool UseAsyncY = false>
cudaError_t launch_quality_mma(
    const std::uint8_t* weights, std::size_t output_rows, std::size_t columns,
    const int* y, std::size_t prompt_rows, float* output,
    cudaStream_t stream) noexcept;

template <QuantKind Kind, int PromptTile, int QualityI = kQualityI>
cudaError_t launch_quality_mma_maybe_stream_k(
    const std::uint8_t* weights, std::size_t output_rows, std::size_t columns,
    const int* y, std::size_t prompt_rows, float* output, const char* path,
    float* tmp_fixup, std::size_t tmp_fixup_floats,
    cudaStream_t stream) noexcept {
  if (!mmq_stream_k_path_on(path) || PromptTile != 128 ||
      QualityI != kQualityI) {
    if (output_rows % QualityI == 0 && prompt_rows % PromptTile == 0) {
      return launch_quality_mma<Kind, PromptTile, false, QualityI>(
          weights, output_rows, columns, y, prompt_rows, output, stream);
    }
    return launch_quality_mma<Kind, PromptTile, true, QualityI>(
        weights, output_rows, columns, y, prompt_rows, output, stream);
  }
  const int nty =
      static_cast<int>((output_rows + QualityI - 1) / QualityI);
  const int ntx =
      static_cast<int>((prompt_rows + PromptTile - 1) / PromptTile);
  const int ntiles = ntx * nty;
  cudaDeviceProp prop{};
  int device = 0;
  int nsm = 0;
  if (cudaGetDevice(&device) == cudaSuccess &&
      cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
    nsm = prop.multiProcessorCount;
  }
  int nblocks = 0;
  if (nsm > 0 && ntiles > 0) {
    if (std::strcmp(path, "stream_k_nsm") == 0) {
      nblocks = nsm;
    } else {
      const int tiles_nwaves = (ntiles + nsm - 1) / nsm;
      const int efficiency = 100 * ntiles / (nsm * tiles_nwaves);
      nblocks = efficiency >= 90 ? ntiles : nsm;
    }
  }
  const bool fixup_needed = nblocks > 0 && ntiles % nblocks != 0;
  const std::size_t needed =
      fixup_needed ? static_cast<std::size_t>(nblocks) * PromptTile * QualityI
                   : 0;
  if (nblocks <= 0 || (fixup_needed && (tmp_fixup == nullptr ||
                                        tmp_fixup_floats < needed))) {
    if (output_rows % QualityI == 0 && prompt_rows % PromptTile == 0) {
      return launch_quality_mma<Kind, PromptTile, false, QualityI>(
          weights, output_rows, columns, y, prompt_rows, output, stream);
    }
    return launch_quality_mma<Kind, PromptTile, true, QualityI>(
        weights, output_rows, columns, y, prompt_rows, output, stream);
  }
  if (output_rows % QualityI == 0 && prompt_rows % PromptTile == 0) {
    return launch_quality_mma_stream_k<Kind, PromptTile, false, QualityI>(
        weights, output_rows, columns, y, prompt_rows, output, tmp_fixup,
        nblocks, fixup_needed, stream);
  }
  return launch_quality_mma_stream_k<Kind, PromptTile, true, QualityI>(
      weights, output_rows, columns, y, prompt_rows, output, tmp_fixup, nblocks,
      fixup_needed, stream);
}

template <typename Kernel>
cudaError_t prepare_mmq_shared(Kernel kernel, std::size_t shared,
                               int* occupancy) noexcept;

template <QuantKind Kind, int PromptTile, bool Fallback, int QualityI,
          bool SoaWeights, bool UseFma, bool UseAsyncY>
cudaError_t launch_quality_mma(
    const std::uint8_t* weights, std::size_t output_rows, std::size_t columns,
    const int* y, std::size_t prompt_rows, float* output,
    cudaStream_t stream) noexcept {
  const dim3 grid(
      static_cast<unsigned int>((output_rows + QualityI - 1) / QualityI),
      static_cast<unsigned int>((prompt_rows + PromptTile - 1) / PromptTile));
  const dim3 block(32, 8);
  constexpr bool kAsyncY = UseAsyncY && !Fallback;
  const std::size_t shared =
      quality_shared_ints(PromptTile, QualityI, kAsyncY) * sizeof(int);
  auto kernel = quant_mmq_mma_quality_kernel<Kind, PromptTile, Fallback,
                                             QualityI, SoaWeights, UseFma,
                                             UseAsyncY>;
  cudaError_t error = prepare_mmq_shared(kernel, shared, nullptr);
  if (error != cudaSuccess) return error;
  if constexpr (Kind == QuantKind::kQ4K && PromptTile == 128 &&
                QualityI == kQualityI && !SoaWeights && !UseFma &&
                !UseAsyncY) {
    static bool printed_q4_attrs = false;
    if (!printed_q4_attrs) {
      cudaFuncAttributes attr{};
      if (cudaFuncGetAttributes(&attr, kernel) == cudaSuccess) {
        std::printf(
            "mma_quality_q4_attrs fallback=%d regs=%d local=%zu const=%zu "
            "shared=%zu maxThreadsPerBlock=%d\n",
            Fallback ? 1 : 0, attr.numRegs, attr.localSizeBytes,
            attr.constSizeBytes, attr.sharedSizeBytes, attr.maxThreadsPerBlock);
      }
      printed_q4_attrs = true;
    }
  }
  return quartz_launch_kernel(kernel, grid, block, shared, stream, weights,
                              output_rows, columns, y, prompt_rows, output);
}

template <QuantKind Kind, int PromptTile, int QualityI>
cudaError_t launch_quality_mma_pipeline_aligned(
    const std::uint8_t* weights, std::size_t output_rows, std::size_t columns,
    const int* y, std::size_t prompt_rows, float* output, const char* path,
    cudaStream_t stream) noexcept {
  if (path != nullptr && std::strcmp(path, "fma_async") == 0) {
    return launch_quality_mma<Kind, PromptTile, false, QualityI, false, true,
                              true>(weights, output_rows, columns, y,
                                    prompt_rows, output, stream);
  }
  if (path != nullptr && std::strcmp(path, "async_y") == 0) {
    return launch_quality_mma<Kind, PromptTile, false, QualityI, false, false,
                              true>(weights, output_rows, columns, y,
                                    prompt_rows, output, stream);
  }
  if (path != nullptr && std::strcmp(path, "fma") == 0) {
    return launch_quality_mma<Kind, PromptTile, false, QualityI, false, true,
                              false>(weights, output_rows, columns, y,
                                     prompt_rows, output, stream);
  }
  return launch_quality_mma<Kind, PromptTile, false, QualityI>(
      weights, output_rows, columns, y, prompt_rows, output, stream);
}

template <typename Kernel>
cudaError_t prepare_mmq_shared(Kernel kernel, std::size_t shared,
                               int* occupancy) noexcept {
  int device = 0;
  cudaDeviceProp prop{};
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return error;
  error = cudaGetDeviceProperties(&prop, device);
  if (error != cudaSuccess) return error;
  if (shared > static_cast<std::size_t>(prop.sharedMemPerBlockOptin)) {
    return cudaErrorInvalidValue;
  }
  error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shared));
  if (error != cudaSuccess) return error;
  int occ = 0;
  error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel,
                                                        kMmaThreads, shared);
  if (error != cudaSuccess) return error;
  if (occ <= 0) return cudaErrorLaunchOutOfResources;
  if (occupancy != nullptr) *occupancy = occ;
  return cudaSuccess;
}

template <QuantKind Kind, int PromptTile, int QualityI>
cudaError_t query_pipeline_kernel(bool fma, bool async_y, int* occupancy,
                                  int* registers, std::size_t* local_bytes,
                                  std::size_t* shared_bytes) noexcept {
  const std::size_t shared =
      quality_shared_ints(PromptTile, QualityI, async_y) * sizeof(int);
  cudaError_t error = cudaErrorInvalidValue;
  int occ = 0;
  cudaFuncAttributes attr{};
  auto query = [&](auto kernel) -> cudaError_t {
    const cudaError_t prepared = prepare_mmq_shared(kernel, shared, &occ);
    if (prepared != cudaSuccess) return prepared;
    return cudaFuncGetAttributes(&attr, kernel);
  };
  if (fma && async_y) {
    error = query(quant_mmq_mma_quality_kernel<Kind, PromptTile, false,
                                               QualityI, false, true, true>);
  } else if (async_y) {
    error = query(quant_mmq_mma_quality_kernel<Kind, PromptTile, false,
                                               QualityI, false, false, true>);
  } else if (fma) {
    error = query(quant_mmq_mma_quality_kernel<Kind, PromptTile, false,
                                               QualityI, false, true, false>);
  } else {
    error = query(
        quant_mmq_mma_quality_kernel<Kind, PromptTile, false, QualityI>);
  }
  if (error != cudaSuccess) return error;
  if (occupancy != nullptr) *occupancy = occ;
  if (registers != nullptr) *registers = attr.numRegs;
  if (local_bytes != nullptr) {
    *local_bytes = static_cast<std::size_t>(attr.localSizeBytes);
  }
  if (shared_bytes != nullptr) *shared_bytes = shared;
  return cudaSuccess;
}

cudaError_t query_q4_pipeline_kernel(unsigned int prompt_tile,
                                     unsigned int quality_i, bool fma,
                                     bool async_y, int* occupancy,
                                     int* registers, std::size_t* local_bytes,
                                     std::size_t* shared_bytes) noexcept {
  if (quality_i == 64 && prompt_tile == 64) {
    return query_pipeline_kernel<QuantKind::kQ4K, 64, 64>(
        fma, async_y, occupancy, registers, local_bytes, shared_bytes);
  }
  if (quality_i == 64 && prompt_tile == 128) {
    return query_pipeline_kernel<QuantKind::kQ4K, 128, 64>(
        fma, async_y, occupancy, registers, local_bytes, shared_bytes);
  }
  if (quality_i == 128 && prompt_tile == 64) {
    return query_pipeline_kernel<QuantKind::kQ4K, 64, 128>(
        fma, async_y, occupancy, registers, local_bytes, shared_bytes);
  }
  if (quality_i == 128 && prompt_tile == 128) {
    return query_pipeline_kernel<QuantKind::kQ4K, 128, 128>(
        fma, async_y, occupancy, registers, local_bytes, shared_bytes);
  }
  return cudaErrorInvalidValue;
}

cudaError_t launch_q4_pipeline_aligned_ij(
    const std::uint8_t* weights, std::size_t output_rows, std::size_t columns,
    const int* y, std::size_t prompt_rows, float* output,
    unsigned int quality_i, unsigned int prompt_tile, const char* path,
    cudaStream_t stream) noexcept {
  if (quality_i == 64 && prompt_tile == 64) {
    return launch_quality_mma_pipeline_aligned<QuantKind::kQ4K, 64, 64>(
        weights, output_rows, columns, y, prompt_rows, output, path, stream);
  }
  if (quality_i == 64 && prompt_tile == 128) {
    return launch_quality_mma_pipeline_aligned<QuantKind::kQ4K, 128, 64>(
        weights, output_rows, columns, y, prompt_rows, output, path, stream);
  }
  if (quality_i == 128 && prompt_tile == 64) {
    return launch_quality_mma_pipeline_aligned<QuantKind::kQ4K, 64, 128>(
        weights, output_rows, columns, y, prompt_rows, output, path, stream);
  }
  if (quality_i == 128 && prompt_tile == 128) {
    return launch_quality_mma_pipeline_aligned<QuantKind::kQ4K, 128, 128>(
        weights, output_rows, columns, y, prompt_rows, output, path, stream);
  }
  return cudaErrorInvalidValue;
}

}  // namespace

constexpr const char kSelectedMmqStreamKPath[] = "off";
constexpr const char kSelectedMmqPipelinePath[] = "fma_async";
inline thread_local const char* g_mmq_pipeline_path_override = nullptr;
inline thread_local MmqTileDispatch g_last_mmq_tile_dispatch{};
inline thread_local bool g_ffn_tile_override = false;
inline thread_local unsigned int g_ffn_gate_i = 0;
inline thread_local unsigned int g_ffn_gate_j = 0;
inline thread_local unsigned int g_ffn_up_i = 0;
inline thread_local unsigned int g_ffn_up_j = 0;
inline thread_local unsigned int g_ffn_down_i = 0;
inline thread_local unsigned int g_ffn_down_j = 0;

unsigned int selected_mma_mmq_prompt_tile() noexcept { return 128U; }

bool mmq_q4_pipeline_tile(unsigned int quality_i,
                          unsigned int prompt_tile) noexcept {
  return (quality_i == 64 || quality_i == 128) &&
         (prompt_tile == 64 || prompt_tile == 128);
}

const char* mmq_q4_tile_ident(unsigned int quality_i,
                              unsigned int prompt_tile) noexcept {
  if (quality_i == 128 && prompt_tile == 128) return "i128_j128";
  if (quality_i == 64 && prompt_tile == 128) return "i64_j128";
  if (quality_i == 128 && prompt_tile == 64) return "i128_j64";
  if (quality_i == 64 && prompt_tile == 64) return "i64_j64";
  if (quality_i == 128 && prompt_tile == 32) return "i128_j32";
  if (quality_i == 64 && prompt_tile == 32) return "i64_j32";
  return "unknown";
}

const char* mmq_q4_kernel_ident(unsigned int quality_i, unsigned int prompt_tile,
                                bool pipeline, bool fallback, bool fma,
                                bool async_y) noexcept {
  if (fallback || !pipeline) {
    if (quality_i == 128 && prompt_tile == 128) return "q4_i128_j128_sync_fallback";
    if (quality_i == 64 && prompt_tile == 128) return "q4_i64_j128_sync_fallback";
    if (quality_i == 128 && prompt_tile == 64) return "q4_i128_j64_sync_fallback";
    if (quality_i == 64 && prompt_tile == 64) return "q4_i64_j64_sync_fallback";
    return "q4_sync_fallback";
  }
  if (fma && async_y) {
    if (quality_i == 128 && prompt_tile == 128) return "q4_i128_j128_fma_async";
    if (quality_i == 64 && prompt_tile == 128) return "q4_i64_j128_fma_async";
    if (quality_i == 128 && prompt_tile == 64) return "q4_i128_j64_fma_async";
    if (quality_i == 64 && prompt_tile == 64) return "q4_i64_j64_fma_async";
  }
  return "q4_pipeline";
}

void record_q4_tile_dispatch(unsigned int quality_i, unsigned int prompt_tile,
                             bool aligned, const char* path) noexcept {
  const bool pipe_on = mmq_pipeline_path_on(path) &&
                       mmq_q4_pipeline_tile(quality_i, prompt_tile) && aligned;
  const bool fma = pipe_on && (std::strcmp(path, "fma") == 0 ||
                               std::strcmp(path, "fma_async") == 0);
  const bool async_y = pipe_on && (std::strcmp(path, "async_y") == 0 ||
                                   std::strcmp(path, "fma_async") == 0);
  g_last_mmq_tile_dispatch.quality_i = quality_i;
  g_last_mmq_tile_dispatch.prompt_tile = prompt_tile;
  g_last_mmq_tile_dispatch.aligned = aligned;
  g_last_mmq_tile_dispatch.pipeline = pipe_on;
  g_last_mmq_tile_dispatch.fallback = !aligned;
  g_last_mmq_tile_dispatch.fma = fma;
  g_last_mmq_tile_dispatch.async_y = async_y;
  g_last_mmq_tile_dispatch.ident = mmq_q4_tile_ident(quality_i, prompt_tile);
  g_last_mmq_tile_dispatch.path = path != nullptr ? path : "";
  g_last_mmq_tile_dispatch.kernel = mmq_q4_kernel_ident(
      quality_i, prompt_tile, pipe_on, !aligned, fma, async_y);
}

const MmqTileDispatch& last_mmq_tile_dispatch() noexcept {
  return g_last_mmq_tile_dispatch;
}

void clear_mmq_tile_dispatch() noexcept { g_last_mmq_tile_dispatch = {}; }

void set_ffn_tile_override(unsigned int gate_i, unsigned int gate_j,
                           unsigned int up_i, unsigned int up_j,
                           unsigned int down_i, unsigned int down_j) noexcept {
  g_ffn_tile_override = true;
  g_ffn_gate_i = gate_i;
  g_ffn_gate_j = gate_j;
  g_ffn_up_i = up_i;
  g_ffn_up_j = up_j;
  g_ffn_down_i = down_i;
  g_ffn_down_j = down_j;
}

void clear_ffn_tile_override() noexcept {
  g_ffn_tile_override = false;
  g_ffn_gate_i = 0;
  g_ffn_gate_j = 0;
  g_ffn_up_i = 0;
  g_ffn_up_j = 0;
  g_ffn_down_i = 0;
  g_ffn_down_j = 0;
}

bool legal_mmq_pipeline_path(const char* path) noexcept {
  return path != nullptr &&
         (std::strcmp(path, "off") == 0 || std::strcmp(path, "fma") == 0 ||
          std::strcmp(path, "async_y") == 0 ||
          std::strcmp(path, "fma_async") == 0);
}

const char* selected_mmq_pipeline_path() noexcept {
  return kSelectedMmqPipelinePath;
}

const char* effective_mmq_pipeline_path() noexcept {
  return g_mmq_pipeline_path_override != nullptr ? g_mmq_pipeline_path_override
                                                 : kSelectedMmqPipelinePath;
}

void set_mmq_pipeline_path_override(const char* path) noexcept {
  g_mmq_pipeline_path_override = path;
}

bool mmq_pipeline_path_on(const char* path) noexcept {
  return legal_mmq_pipeline_path(path) && std::strcmp(path, "off") != 0;
}

std::size_t mmq_pipeline_extra_shared_bytes(unsigned int prompt_tile,
                                            unsigned int quality_i,
                                            const char* path) noexcept {
  if (!mmq_pipeline_path_on(path)) return 0;
  if (std::strcmp(path, "async_y") != 0 &&
      std::strcmp(path, "fma_async") != 0) {
    return 0;
  }
  const std::size_t y_tile = quality_pad_ints(
      static_cast<std::size_t>(prompt_tile) * kMmqTileYk);
  (void)quality_i;
  return y_tile * sizeof(int);
}

int mmq_pipeline_occupancy(QuantKind kind, unsigned int prompt_tile,
                           unsigned int quality_i, const char* path) noexcept {
  if (!legal_mmq_pipeline_path(path)) return 0;
  const bool async_y = std::strcmp(path, "async_y") == 0 ||
                       std::strcmp(path, "fma_async") == 0;
  const bool fma = std::strcmp(path, "fma") == 0 ||
                   std::strcmp(path, "fma_async") == 0;
  int occupancy = 0;
  cudaError_t error = cudaErrorInvalidValue;
  if (kind == QuantKind::kQ4K && mmq_q4_pipeline_tile(quality_i, prompt_tile)) {
    error = query_q4_pipeline_kernel(prompt_tile, quality_i, fma, async_y,
                                     &occupancy, nullptr, nullptr, nullptr);
    return error == cudaSuccess ? occupancy : 0;
  }
  const std::size_t shared =
      quality_shared_ints(static_cast<int>(prompt_tile),
                         static_cast<int>(quality_i), async_y) *
      sizeof(int);
  auto query = [&](auto kernel) -> cudaError_t {
    cudaError_t set = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared));
    if (set != cudaSuccess) return set;
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, kMmaThreads, shared);
  };
  if (kind == QuantKind::kQ8_0 && prompt_tile == 128 &&
             (quality_i == 32 || quality_i == 64 || quality_i == 128)) {
    if (quality_i == 32) {
      if (fma && async_y) {
        error = query(
            quant_mmq_mma_quality_kernel<QuantKind::kQ8_0, 128, false, 32, false,
                                         true, true>);
      } else if (async_y) {
        error = query(
            quant_mmq_mma_quality_kernel<QuantKind::kQ8_0, 128, false, 32, false,
                                         false, true>);
      } else if (fma) {
        error = query(
            quant_mmq_mma_quality_kernel<QuantKind::kQ8_0, 128, false, 32, false,
                                         true, false>);
      } else {
        error = query(
            quant_mmq_mma_quality_kernel<QuantKind::kQ8_0, 128, false, 32>);
      }
    } else if (quality_i == 64) {
      if (fma && async_y) {
        error = query(
            quant_mmq_mma_quality_kernel<QuantKind::kQ8_0, 128, false, 64, false,
                                         true, true>);
      } else if (async_y) {
        error = query(
            quant_mmq_mma_quality_kernel<QuantKind::kQ8_0, 128, false, 64, false,
                                         false, true>);
      } else if (fma) {
        error = query(
            quant_mmq_mma_quality_kernel<QuantKind::kQ8_0, 128, false, 64, false,
                                         true, false>);
      } else {
        error = query(
            quant_mmq_mma_quality_kernel<QuantKind::kQ8_0, 128, false, 64>);
      }
    } else {
      if (fma && async_y) {
        error = query(
            quant_mmq_mma_quality_kernel<QuantKind::kQ8_0, 128, false, 128, false,
                                         true, true>);
      } else if (async_y) {
        error = query(
            quant_mmq_mma_quality_kernel<QuantKind::kQ8_0, 128, false, 128, false,
                                         false, true>);
      } else if (fma) {
        error = query(
            quant_mmq_mma_quality_kernel<QuantKind::kQ8_0, 128, false, 128, false,
                                         true, false>);
      } else {
        error = query(
            quant_mmq_mma_quality_kernel<QuantKind::kQ8_0, 128, false, 128>);
      }
    }
  } else {
    return mma_mmq_occupancy_ij(kind, prompt_tile, quality_i);
  }
  return error == cudaSuccess ? occupancy : 0;
}

template <QuantKind Kind, int PromptTile, int QualityI = kQualityI,
          bool SoaWeights = false>
cudaError_t launch_quality_dispatch(
    const std::uint8_t* weights, std::size_t output_rows, std::size_t columns,
    const int* y, std::size_t prompt_rows, float* output,
    cudaStream_t stream) noexcept;

cudaError_t launch_q4_quality_sync_ij(
    const std::uint8_t* weights, std::size_t output_rows, std::size_t columns,
    const int* packed, std::size_t prompt_rows, float* output,
    unsigned int quality_i, unsigned int prompt_tile,
    cudaStream_t stream) noexcept {
  if (quality_i == 64) {
    if (prompt_tile == 32) {
      return launch_quality_dispatch<QuantKind::kQ4K, 32, 64>(
          weights, output_rows, columns, packed, prompt_rows, output, stream);
    }
    if (prompt_tile == 64) {
      return launch_quality_dispatch<QuantKind::kQ4K, 64, 64>(
          weights, output_rows, columns, packed, prompt_rows, output, stream);
    }
    return launch_quality_dispatch<QuantKind::kQ4K, 128, 64>(
        weights, output_rows, columns, packed, prompt_rows, output, stream);
  }
  if (prompt_tile == 32) {
    return launch_quality_dispatch<QuantKind::kQ4K, 32>(
        weights, output_rows, columns, packed, prompt_rows, output, stream);
  }
  if (prompt_tile == 64) {
    return launch_quality_dispatch<QuantKind::kQ4K, 64>(
        weights, output_rows, columns, packed, prompt_rows, output, stream);
  }
  return launch_quality_dispatch<QuantKind::kQ4K, 128>(
      weights, output_rows, columns, packed, prompt_rows, output, stream);
}

cudaError_t launch_quant_mmq_mma_y_pipeline(
    QuantKind kind, const std::uint8_t* weights, std::size_t output_rows,
    std::size_t columns, const Q8Block* y, std::size_t prompt_rows,
    float* output, unsigned int quality_i, unsigned int prompt_tile,
    const char* path, cudaStream_t stream) noexcept {
  if (!legal_mmq_pipeline_path(path) ||
      (kind != QuantKind::kQ4K && kind != QuantKind::kQ6K) ||
      !legal_ffn_quality_i(quality_i) || !legal_ffn_prompt_tile(prompt_tile) ||
      columns % kMmaBlockValues != 0) {
    return cudaErrorInvalidValue;
  }
  const int* packed = reinterpret_cast<const int*>(y);
  const bool aligned =
      output_rows % quality_i == 0 && prompt_rows % prompt_tile == 0;
  if (kind == QuantKind::kQ4K) {
    record_q4_tile_dispatch(quality_i, prompt_tile, aligned, path);
    if (mmq_pipeline_path_on(path) && aligned &&
        mmq_q4_pipeline_tile(quality_i, prompt_tile)) {
      return launch_q4_pipeline_aligned_ij(
          weights, output_rows, columns, packed, prompt_rows, output, quality_i,
          prompt_tile, path, stream);
    }
    return launch_q4_quality_sync_ij(weights, output_rows, columns, packed,
                                     prompt_rows, output, quality_i,
                                     prompt_tile, stream);
  }
  if (prompt_tile == 32) {
    return launch_quality_dispatch<QuantKind::kQ6K, 32>(
        weights, output_rows, columns, packed, prompt_rows, output, stream);
  }
  if (prompt_tile == 64) {
    return launch_quality_dispatch<QuantKind::kQ6K, 64>(
        weights, output_rows, columns, packed, prompt_rows, output, stream);
  }
  return launch_quality_dispatch<QuantKind::kQ6K, 128>(
      weights, output_rows, columns, packed, prompt_rows, output, stream);
}

cudaError_t launch_q8_mmq_quality_mma_pipeline(
    const std::uint8_t* weights, std::size_t output_rows, std::size_t columns,
    const Q8Block* y, std::size_t prompt_rows, float* output,
    unsigned int quality_i, const char* path, cudaStream_t stream) noexcept {
  if (!legal_mmq_pipeline_path(path) ||
      (quality_i != 32 && quality_i != 64 && quality_i != 128) ||
      columns % kMmaBlockValues != 0) {
    return cudaErrorInvalidValue;
  }
  const int* packed = reinterpret_cast<const int*>(y);
  const bool aligned =
      output_rows % quality_i == 0 && prompt_rows % 128U == 0;
  if (mmq_pipeline_path_on(path) && aligned) {
    if (quality_i == 32) {
      return launch_quality_mma_pipeline_aligned<QuantKind::kQ8_0, 128, 32>(
          weights, output_rows, columns, packed, prompt_rows, output, path,
          stream);
    }
    if (quality_i == 64) {
      return launch_quality_mma_pipeline_aligned<QuantKind::kQ8_0, 128, 64>(
          weights, output_rows, columns, packed, prompt_rows, output, path,
          stream);
    }
    return launch_quality_mma_pipeline_aligned<QuantKind::kQ8_0, 128, 128>(
        weights, output_rows, columns, packed, prompt_rows, output, path,
        stream);
  }
  if (quality_i == 32) {
    return launch_quality_dispatch<QuantKind::kQ8_0, 128, 32>(
        weights, output_rows, columns, packed, prompt_rows, output, stream);
  }
  if (quality_i == 64) {
    return launch_quality_dispatch<QuantKind::kQ8_0, 128, 64>(
        weights, output_rows, columns, packed, prompt_rows, output, stream);
  }
  return launch_quality_dispatch<QuantKind::kQ8_0, 128, 128>(
      weights, output_rows, columns, packed, prompt_rows, output, stream);
}

const char* selected_mmq_stream_k_path() noexcept {
  return kSelectedMmqStreamKPath;
}

bool mmq_uses_stream_k() noexcept {
  return std::strcmp(kSelectedMmqStreamKPath, "stream_k") == 0 ||
         std::strcmp(kSelectedMmqStreamKPath, "stream_k_nsm") == 0;
}

int mmq_stream_k_nsm() noexcept {
  int device = 0;
  cudaDeviceProp prop{};
  if (cudaGetDevice(&device) != cudaSuccess) return 0;
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return 0;
  return prop.multiProcessorCount;
}

int mmq_stream_k_nblocks(int nsm, int ntiles_dst, const char* path) noexcept {
  if (path == nullptr || nsm <= 0 || ntiles_dst <= 0) return 0;
  if (std::strcmp(path, "off") == 0) return 0;
  if (std::strcmp(path, "stream_k_nsm") == 0) return nsm;
  if (std::strcmp(path, "stream_k") != 0) return 0;
  const int tiles_nwaves = (ntiles_dst + nsm - 1) / nsm;
  const int efficiency = 100 * ntiles_dst / (nsm * tiles_nwaves);
  return efficiency >= 90 ? ntiles_dst : nsm;
}

bool mmq_stream_k_fixup_needed(int ntiles_dst, int nblocks) noexcept {
  return nblocks > 0 && ntiles_dst % nblocks != 0;
}

std::size_t mmq_stream_k_fixup_floats(int nblocks, unsigned int i,
                                     unsigned int j) noexcept {
  if (nblocks <= 0 || i == 0 || j == 0) return 0;
  return static_cast<std::size_t>(nblocks) * static_cast<std::size_t>(j) *
         static_cast<std::size_t>(i);
}

std::size_t mma_mmq_shared_bytes(unsigned int prompt_tile) noexcept {
  if (prompt_tile != 32 && prompt_tile != 64 && prompt_tile != 128) return 0;
  return quality_shared_ints(static_cast<int>(prompt_tile)) * sizeof(int);
}

int mmq_stream_k_occupancy(QuantKind kind) noexcept {
  int occupancy = 0;
  cudaError_t error = cudaErrorInvalidValue;
  const std::size_t shared = mma_mmq_shared_bytes(128);
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
    error = query(
        quant_mmq_mma_quality_stream_k_kernel<QuantKind::kQ4K, 128, true>);
  } else if (kind == QuantKind::kQ6K) {
    error = query(
        quant_mmq_mma_quality_stream_k_kernel<QuantKind::kQ6K, 128, true>);
  }
  return error == cudaSuccess ? occupancy : 0;
}

int mmq_stream_k_fixup_occupancy() noexcept {
  int occupancy = 0;
  const cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, quant_mmq_mma_stream_k_fixup_kernel<128, true>, 128, 0);
  return error == cudaSuccess ? occupancy : 0;
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

template <QuantKind Kind, int PromptTile, int QualityI, bool SoaWeights>
cudaError_t launch_quality_dispatch(
    const std::uint8_t* weights, std::size_t output_rows, std::size_t columns,
    const int* y, std::size_t prompt_rows, float* output,
    cudaStream_t stream) noexcept {
  if (output_rows % QualityI == 0 && prompt_rows % PromptTile == 0) {
    return launch_quality_mma<Kind, PromptTile, false, QualityI, SoaWeights>(
        weights, output_rows, columns, y, prompt_rows, output, stream);
  }
  return launch_quality_mma<Kind, PromptTile, true, QualityI, SoaWeights>(
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
    return quartz_launch_kernel(quantize_mmq_q8_1_bf16<QuantKind::kQ4K>,
                                dim3(blocks), dim3(kQualityQuantThreads), 0,
                                stream, prompt, prompt_rows, columns,
                                reinterpret_cast<std::uint8_t*>(workspace));
  } else if (kind == QuantKind::kQ6K) {
    return quartz_launch_kernel(quantize_mmq_q8_1_bf16<QuantKind::kQ6K>,
                                dim3(blocks), dim3(kQualityQuantThreads), 0,
                                stream, prompt, prompt_rows, columns,
                                reinterpret_cast<std::uint8_t*>(workspace));
  }
  return quartz_launch_kernel(quantize_mmq_q8_1_bf16<QuantKind::kQ8_0>,
                              dim3(blocks), dim3(kQualityQuantThreads), 0,
                              stream, prompt, prompt_rows, columns,
                              reinterpret_cast<std::uint8_t*>(workspace));
}

cudaError_t launch_quant_mmq_mma_y_path(
    QuantKind kind, const std::uint8_t* weights, std::size_t output_rows,
    std::size_t columns, const Q8Block* y, std::size_t prompt_rows,
    float* output, cudaStream_t stream, const char* path, float* tmp_fixup,
    std::size_t tmp_fixup_floats) noexcept {
  if (weights == nullptr || y == nullptr || output == nullptr ||
      output_rows == 0 || columns == 0 || prompt_rows == 0 ||
      columns % kMmaBlockValues != 0 ||
      (kind != QuantKind::kQ4K && kind != QuantKind::kQ6K)) {
    return cudaErrorInvalidValue;
  }
  const unsigned int prompt_tile = selected_mma_mmq_prompt_tile();
  const int* packed = reinterpret_cast<const int*>(y);
  const char* resolved =
      path == nullptr || path[0] == '\0' ? kSelectedMmqStreamKPath : path;
  if (prompt_tile != 128) {
    if (kind == QuantKind::kQ4K) {
      if (prompt_tile == 32) {
        return launch_quality_dispatch<QuantKind::kQ4K, 32>(
            weights, output_rows, columns, packed, prompt_rows, output, stream);
      }
      return launch_quality_dispatch<QuantKind::kQ4K, 64>(
          weights, output_rows, columns, packed, prompt_rows, output, stream);
    }
    if (prompt_tile == 32) {
      return launch_quality_dispatch<QuantKind::kQ6K, 32>(
          weights, output_rows, columns, packed, prompt_rows, output, stream);
    }
    return launch_quality_dispatch<QuantKind::kQ6K, 64>(
        weights, output_rows, columns, packed, prompt_rows, output, stream);
  }
  if (kind == QuantKind::kQ4K) {
    return launch_quality_mma_maybe_stream_k<QuantKind::kQ4K, 128>(
        weights, output_rows, columns, packed, prompt_rows, output, resolved,
        tmp_fixup, tmp_fixup_floats, stream);
  }
  return launch_quality_mma_maybe_stream_k<QuantKind::kQ6K, 128>(
      weights, output_rows, columns, packed, prompt_rows, output, resolved,
      tmp_fixup, tmp_fixup_floats, stream);
}

cudaError_t launch_quant_mmq_mma_y(
    QuantKind kind, const std::uint8_t* weights, std::size_t output_rows,
    std::size_t columns, const Q8Block* y, std::size_t prompt_rows,
    float* output, cudaStream_t stream, float* tmp_fixup,
    std::size_t tmp_fixup_floats) noexcept {
  return launch_quant_mmq_mma_y_path(
      kind, weights, output_rows, columns, y, prompt_rows, output, stream,
      kSelectedMmqStreamKPath, tmp_fixup, tmp_fixup_floats);
}

cudaError_t launch_swiglu_quantize_mmq_q8_1(const float* gate, const float* up,
                                            std::size_t prompt_rows,
                                            std::size_t columns, Q8Block* y,
                                            cudaStream_t stream) noexcept {
  if (gate == nullptr || up == nullptr || y == nullptr || prompt_rows == 0 ||
      columns == 0 || columns % kMmaBlockValues != 0) {
    return cudaErrorInvalidValue;
  }
  const std::size_t count = prompt_rows * columns;
  const unsigned int blocks = static_cast<unsigned int>(
      (count + static_cast<std::size_t>(kQualityQuantThreads) - 1) /
      static_cast<std::size_t>(kQualityQuantThreads));
  quantize_mmq_q8_1_swiglu<QuantKind::kQ4K>
      <<<blocks, kQualityQuantThreads, 0, stream>>>(
          gate, up, prompt_rows, columns, reinterpret_cast<std::uint8_t*>(y));
  return cudaPeekAtLastError();
}

cudaError_t launch_quant_mmq_mma_tile(
    QuantKind kind, const std::uint8_t* weights, std::size_t output_rows,
    std::size_t columns, const __nv_bfloat16* prompt, std::size_t prompt_rows,
    Q8Block* q8_workspace, float* output, unsigned int prompt_tile,
    cudaStream_t stream, float* tmp_fixup,
    std::size_t tmp_fixup_floats) noexcept {
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
  if (prompt_tile != selected_mma_mmq_prompt_tile()) {
    const int* packed = reinterpret_cast<const int*>(q8_workspace);
    if (kind == QuantKind::kQ4K) {
      if (prompt_tile == 32) {
        return launch_quality_dispatch<QuantKind::kQ4K, 32>(
            weights, output_rows, columns, packed, prompt_rows, output, stream);
      }
      if (prompt_tile == 128) {
        return launch_quality_dispatch<QuantKind::kQ4K, 128>(
            weights, output_rows, columns, packed, prompt_rows, output, stream);
      }
      return launch_quality_dispatch<QuantKind::kQ4K, 64>(
          weights, output_rows, columns, packed, prompt_rows, output, stream);
    }
    if (prompt_tile == 32) {
      return launch_quality_dispatch<QuantKind::kQ6K, 32>(
          weights, output_rows, columns, packed, prompt_rows, output, stream);
    }
    if (prompt_tile == 128) {
      return launch_quality_dispatch<QuantKind::kQ6K, 128>(
          weights, output_rows, columns, packed, prompt_rows, output, stream);
    }
    return launch_quality_dispatch<QuantKind::kQ6K, 64>(
        weights, output_rows, columns, packed, prompt_rows, output, stream);
  }
  return launch_quant_mmq_mma_y(kind, weights, output_rows, columns,
                                q8_workspace, prompt_rows, output, stream,
                                tmp_fixup, tmp_fixup_floats);
}

cudaError_t launch_quant_mmq_mma(
    QuantKind kind, const std::uint8_t* weights, std::size_t output_rows,
    std::size_t columns, const __nv_bfloat16* prompt, std::size_t prompt_rows,
    Q8Block* q8_workspace, float* output, cudaStream_t stream, float* tmp_fixup,
    std::size_t tmp_fixup_floats) noexcept {
  return launch_quant_mmq_mma_tile(
      kind, weights, output_rows, columns, prompt, prompt_rows, q8_workspace,
      output, selected_mma_mmq_prompt_tile(), stream, tmp_fixup,
      tmp_fixup_floats);
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

constexpr const char kSelectedSkinnyMixerPath[] = "mma_i32_j128";

bool skinny_mixer_q8_output_rows(std::size_t output_rows) noexcept {
  return output_rows > 0 && output_rows < static_cast<std::size_t>(kQualityI);
}

const char* selected_skinny_mixer_path() noexcept {
  return kSelectedSkinnyMixerPath;
}

int q8_quality_mmq_occupancy_i(unsigned int prompt_tile,
                               unsigned int quality_i) noexcept {
  int occupancy = 0;
  cudaError_t error = cudaErrorInvalidValue;
  const std::size_t shared =
      quality_shared_ints(static_cast<int>(prompt_tile),
                         static_cast<int>(quality_i)) *
      sizeof(int);
  if (shared == 0) return 0;
  auto query = [&](auto kernel) -> cudaError_t {
    cudaError_t set = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared));
    if (set != cudaSuccess) return set;
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, kMmaThreads, shared);
  };
  if (prompt_tile == 128 && quality_i == 32) {
    error = query(quant_mmq_mma_quality_kernel<QuantKind::kQ8_0, 128, true, 32>);
  } else if (prompt_tile == 128 && quality_i == 64) {
    error = query(quant_mmq_mma_quality_kernel<QuantKind::kQ8_0, 128, true, 64>);
  } else if (quality_i == 128) {
    return mma_mmq_occupancy(QuantKind::kQ8_0, prompt_tile);
  }
  return error == cudaSuccess ? occupancy : 0;
}

int q8_quality_mmq_occupancy(unsigned int prompt_tile) noexcept {
  return q8_quality_mmq_occupancy_i(prompt_tile, kQualityI);
}

cudaError_t launch_q8_mmq_quality_mma_i(const std::uint8_t* weights,
                                        std::size_t output_rows,
                                        std::size_t columns, const Q8Block* y,
                                        std::size_t prompt_rows, float* output,
                                        unsigned int quality_i,
                                        cudaStream_t stream) noexcept {
  if (weights == nullptr || y == nullptr || output == nullptr ||
      output_rows == 0 || columns == 0 || prompt_rows == 0 ||
      columns % kMmaBlockValues != 0 ||
      (quality_i != 32 && quality_i != 64 && quality_i != 128)) {
    return cudaErrorInvalidValue;
  }
  const char* pipe = effective_mmq_pipeline_path();
  if (mmq_pipeline_path_on(pipe) && output_rows % quality_i == 0 &&
      prompt_rows % 128U == 0) {
    return launch_q8_mmq_quality_mma_pipeline(weights, output_rows, columns, y,
                                              prompt_rows, output, quality_i,
                                              pipe, stream);
  }
  const int* packed = reinterpret_cast<const int*>(y);
  if (quality_i == 32) {
    return launch_quality_dispatch<QuantKind::kQ8_0, 128, 32>(
        weights, output_rows, columns, packed, prompt_rows, output, stream);
  }
  if (quality_i == 64) {
    return launch_quality_dispatch<QuantKind::kQ8_0, 128, 64>(
        weights, output_rows, columns, packed, prompt_rows, output, stream);
  }
  return launch_quality_dispatch<QuantKind::kQ8_0, 128, 128>(
      weights, output_rows, columns, packed, prompt_rows, output, stream);
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
  if (skinny_mixer_q8_output_rows(output_rows) && prompt_rows >= 8) {
    const char* path = selected_skinny_mixer_path();
    if (std::strcmp(path, "mma_i32_j128") == 0) {
      return launch_q8_mmq_quality_mma_i(weights, output_rows, columns, y,
                                         prompt_rows, output, 32, stream);
    }
    if (std::strcmp(path, "mma_i64_j128") == 0) {
      return launch_q8_mmq_quality_mma_i(weights, output_rows, columns, y,
                                         prompt_rows, output, 64, stream);
    }
  }
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

constexpr const char kSelectedLargeMixerQ8Path[] = "quality_mma";

bool large_mixer_q8_output_rows(std::size_t output_rows) noexcept {
  return output_rows >= static_cast<std::size_t>(kQualityI);
}

std::size_t q8_aligned_soa_bytes(std::size_t output_rows,
                                 std::size_t columns) noexcept {
  if (output_rows == 0 || columns == 0 || columns % 32 != 0) return 0;
  return soa_total_bytes(output_rows, columns);
}

const char* selected_large_mixer_q8_path() noexcept {
  return kSelectedLargeMixerQ8Path;
}

int q8_d2r_occupancy(unsigned int prompt_tile) noexcept {
  int occupancy = 0;
  cudaError_t error = cudaErrorInvalidValue;
  const std::size_t shared =
      quality_shared_ints(static_cast<int>(prompt_tile), kQualityI) *
      sizeof(int);
  if (shared == 0 || prompt_tile != 128) return 0;
  auto kernel =
      quant_mmq_mma_quality_kernel<QuantKind::kQ8_0, 128, true, kQualityI, true>;
  error = cudaFuncSetAttribute(kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(shared));
  if (error != cudaSuccess) return 0;
  error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, kernel, kMmaThreads, shared);
  return error == cudaSuccess ? occupancy : 0;
}

cudaError_t launch_repack_q8_0_aligned(const std::uint8_t* weights,
                                       std::size_t output_rows,
                                       std::size_t columns, std::uint8_t* soa,
                                       cudaStream_t stream) noexcept {
  if (weights == nullptr || soa == nullptr || output_rows == 0 ||
      columns == 0 || columns % 32 != 0) {
    return cudaErrorInvalidValue;
  }
  const std::size_t nblk = output_rows * (columns / 32);
  const unsigned int blocks = static_cast<unsigned int>(
      (nblk + static_cast<std::size_t>(kQualityQuantThreads) - 1) /
      static_cast<std::size_t>(kQualityQuantThreads));
  repack_q8_0_aligned_kernel<<<blocks, kQualityQuantThreads, 0, stream>>>(
      weights, output_rows, columns, soa);
  return cudaPeekAtLastError();
}

cudaError_t launch_q8_mmq_d2r(const std::uint8_t* soa, std::size_t output_rows,
                              std::size_t columns, const Q8Block* y,
                              std::size_t prompt_rows, float* output,
                              cudaStream_t stream) noexcept {
  if (soa == nullptr || y == nullptr || output == nullptr ||
      output_rows == 0 || columns == 0 || prompt_rows == 0 ||
      columns % kMmaBlockValues != 0) {
    return cudaErrorInvalidValue;
  }
  const int* packed = reinterpret_cast<const int*>(y);
  return launch_quality_dispatch<QuantKind::kQ8_0, 128, kQualityI, true>(
      soa, output_rows, columns, packed, prompt_rows, output, stream);
}

constexpr unsigned int kSelectedFfnGateQualityI = 128U;
constexpr unsigned int kSelectedFfnGatePromptTile = 128U;
constexpr unsigned int kSelectedFfnUpQualityI = 128U;
constexpr unsigned int kSelectedFfnUpPromptTile = 128U;
constexpr unsigned int kSelectedFfnDownQualityI = 128U;
constexpr unsigned int kSelectedFfnDownPromptTile = 128U;

bool legal_ffn_quality_i(unsigned int quality_i) noexcept {
  return quality_i == 64 || quality_i == 128;
}

bool legal_ffn_prompt_tile(unsigned int prompt_tile) noexcept {
  return prompt_tile == 32 || prompt_tile == 64 || prompt_tile == 128;
}

unsigned int selected_ffn_gate_quality_i() noexcept {
  return g_ffn_tile_override ? g_ffn_gate_i : kSelectedFfnGateQualityI;
}

unsigned int selected_ffn_gate_prompt_tile() noexcept {
  return g_ffn_tile_override ? g_ffn_gate_j : kSelectedFfnGatePromptTile;
}

unsigned int selected_ffn_up_quality_i() noexcept {
  return g_ffn_tile_override ? g_ffn_up_i : kSelectedFfnUpQualityI;
}

unsigned int selected_ffn_up_prompt_tile() noexcept {
  return g_ffn_tile_override ? g_ffn_up_j : kSelectedFfnUpPromptTile;
}

unsigned int selected_ffn_down_quality_i() noexcept {
  return g_ffn_tile_override ? g_ffn_down_i : kSelectedFfnDownQualityI;
}

unsigned int selected_ffn_down_prompt_tile() noexcept {
  return g_ffn_tile_override ? g_ffn_down_j : kSelectedFfnDownPromptTile;
}

int mma_mmq_occupancy_ij(QuantKind kind, unsigned int prompt_tile,
                         unsigned int quality_i) noexcept {
  if (!legal_ffn_prompt_tile(prompt_tile) || !legal_ffn_quality_i(quality_i)) {
    return 0;
  }
  if (quality_i == 128) {
    return mma_mmq_occupancy(kind, prompt_tile);
  }
  if (kind != QuantKind::kQ4K) return 0;
  int occupancy = 0;
  cudaError_t error = cudaErrorInvalidValue;
  const std::size_t shared =
      quality_shared_ints(static_cast<int>(prompt_tile),
                         static_cast<int>(quality_i)) *
      sizeof(int);
  auto query = [&](auto kernel) -> cudaError_t {
    cudaError_t set = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared));
    if (set != cudaSuccess) return set;
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, kMmaThreads, shared);
  };
  if (prompt_tile == 32) {
    error = query(quant_mmq_mma_quality_kernel<QuantKind::kQ4K, 32, true, 64>);
  } else if (prompt_tile == 64) {
    error = query(quant_mmq_mma_quality_kernel<QuantKind::kQ4K, 64, true, 64>);
  } else {
    error = query(quant_mmq_mma_quality_kernel<QuantKind::kQ4K, 128, true, 64>);
  }
  return error == cudaSuccess ? occupancy : 0;
}

cudaError_t launch_quant_mmq_mma_y_ij(
    QuantKind kind, const std::uint8_t* weights, std::size_t output_rows,
    std::size_t columns, const Q8Block* y, std::size_t prompt_rows,
    float* output, unsigned int quality_i, unsigned int prompt_tile,
    cudaStream_t stream, float* tmp_fixup,
    std::size_t tmp_fixup_floats) noexcept {
  (void)tmp_fixup;
  (void)tmp_fixup_floats;
  if (weights == nullptr || y == nullptr || output == nullptr ||
      output_rows == 0 || columns == 0 || prompt_rows == 0 ||
      columns % kMmaBlockValues != 0 || !legal_ffn_quality_i(quality_i) ||
      !legal_ffn_prompt_tile(prompt_tile) ||
      (kind != QuantKind::kQ4K && kind != QuantKind::kQ6K) ||
      (kind == QuantKind::kQ6K && quality_i != 128)) {
    return cudaErrorInvalidValue;
  }
  const char* pipe = effective_mmq_pipeline_path();
  const bool aligned =
      output_rows % quality_i == 0 && prompt_rows % prompt_tile == 0;
  if (kind == QuantKind::kQ4K) {
    record_q4_tile_dispatch(quality_i, prompt_tile, aligned, pipe);
    if (mmq_pipeline_path_on(pipe) && aligned &&
        mmq_q4_pipeline_tile(quality_i, prompt_tile)) {
      return launch_quant_mmq_mma_y_pipeline(
          kind, weights, output_rows, columns, y, prompt_rows, output, quality_i,
          prompt_tile, pipe, stream);
    }
    const int* packed = reinterpret_cast<const int*>(y);
    return launch_q4_quality_sync_ij(weights, output_rows, columns, packed,
                                     prompt_rows, output, quality_i,
                                     prompt_tile, stream);
  }
  const int* packed = reinterpret_cast<const int*>(y);
  if (prompt_tile == 32) {
    return launch_quality_dispatch<QuantKind::kQ6K, 32>(
        weights, output_rows, columns, packed, prompt_rows, output, stream);
  }
  if (prompt_tile == 64) {
    return launch_quality_dispatch<QuantKind::kQ6K, 64>(
        weights, output_rows, columns, packed, prompt_rows, output, stream);
  }
  return launch_quality_dispatch<QuantKind::kQ6K, 128>(
      weights, output_rows, columns, packed, prompt_rows, output, stream);
}

cudaError_t mmq_pipeline_kernel_attributes(
    QuantKind kind, unsigned int prompt_tile, unsigned int quality_i,
    const char* path, int* occupancy, int* registers,
    std::size_t* local_bytes, std::size_t* shared_bytes) noexcept {
  if (!legal_mmq_pipeline_path(path)) return cudaErrorInvalidValue;
  const bool async_y = std::strcmp(path, "async_y") == 0 ||
                       std::strcmp(path, "fma_async") == 0;
  const bool fma = std::strcmp(path, "fma") == 0 ||
                   std::strcmp(path, "fma_async") == 0;
  if (kind == QuantKind::kQ4K && mmq_q4_pipeline_tile(quality_i, prompt_tile)) {
    return query_q4_pipeline_kernel(prompt_tile, quality_i, fma, async_y,
                                    occupancy, registers, local_bytes,
                                    shared_bytes);
  }
  if (kind == QuantKind::kQ8_0 && prompt_tile == 128 &&
      (quality_i == 32 || quality_i == 64 || quality_i == 128)) {
    if (quality_i == 32) {
      return query_pipeline_kernel<QuantKind::kQ8_0, 128, 32>(
          fma, async_y, occupancy, registers, local_bytes, shared_bytes);
    }
    if (quality_i == 64) {
      return query_pipeline_kernel<QuantKind::kQ8_0, 128, 64>(
          fma, async_y, occupancy, registers, local_bytes, shared_bytes);
    }
    return query_pipeline_kernel<QuantKind::kQ8_0, 128, 128>(
        fma, async_y, occupancy, registers, local_bytes, shared_bytes);
  }
  return cudaErrorInvalidValue;
}

constexpr const char kSelectedFfnPath[] = "shared_y_swiglu_q8";

const char* selected_ffn_path() noexcept { return kSelectedFfnPath; }

bool ffn_shares_gate_up_y() noexcept {
  return std::strcmp(kSelectedFfnPath, "shared_y") == 0 ||
         std::strcmp(kSelectedFfnPath, "shared_y_swiglu_q8") == 0;
}

bool ffn_swiglu_writes_q8() noexcept {
  return std::strcmp(kSelectedFfnPath, "swiglu_q8") == 0 ||
         std::strcmp(kSelectedFfnPath, "shared_y_swiglu_q8") == 0;
}

}  // namespace qw38::cuda
