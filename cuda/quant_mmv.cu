#include "quant_mmv.h"
#include "quant_mmq_mma.cuh"
#include "ffn_decode_path.cuh"
#include "q4k_decode_path.cuh"
#include "q6k_decode_path.cuh"

#include <cstring>

#include <cuda_fp16.h>

QW38_PDL_REGISTER_DEVICE_OPS()

namespace qw38::cuda {
namespace {

constexpr int kWarpSize = 32;
constexpr int kThreads = 256;
constexpr std::size_t kValuesPerWeightBlock = 256;
constexpr std::size_t kQ4KBytes = 144;
constexpr std::size_t kQ6KBytes = 210;
constexpr std::size_t kQ80Bytes = 34;

__device__ float read_half(const std::uint8_t* bytes) {
  const unsigned short bits = static_cast<unsigned short>(bytes[0]) |
                              (static_cast<unsigned short>(bytes[1]) << 8U);
  return __half2float(__ushort_as_half(bits));
}

__device__ void q4_scale_min(const std::uint8_t* packed, int index,
                             int* scale, int* minimum) {
  if (index < 4) {
    *scale = packed[index] & 63;
    *minimum = packed[index + 4] & 63;
    return;
  }
  *scale = (packed[index + 4] & 15) | ((packed[index - 4] >> 6) << 4);
  *minimum = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4);
}

__device__ float decode_q4(const std::uint8_t* block, int index) {
  const int group = index / 64;
  const int within = index % 64;
  const int high = within / 32;
  const int lane = within % 32;
  int scale = 0;
  int minimum = 0;
  q4_scale_min(block + 4, group * 2 + high, &scale, &minimum);
  const std::uint8_t packed = block[16 + group * 32 + lane];
  const int quant = high == 0 ? packed & 15 : packed >> 4;
  return read_half(block) * static_cast<float>(scale * quant) -
         read_half(block + 2) * static_cast<float>(minimum);
}

__device__ float decode_q6(const std::uint8_t* block, int index) {
  const int half = index / 128;
  const int within = index % 128;
  const int group = within / 32;
  const int lane = within % 32;
  const int low_offset = half * 64;
  const int high_offset = 128 + half * 32;
  const std::uint8_t low =
      block[low_offset + lane + ((group & 1) != 0 ? 32 : 0)];
  const int low_four = group < 2 ? low & 15 : low >> 4;
  const int high_two = (block[high_offset + lane] >> (group * 2)) & 3;
  const int quant = (low_four | (high_two << 4)) - 32;
  const std::uint8_t scale_byte =
      block[192 + half * 8 + (lane / 16) + group * 2];
  const int scale = scale_byte < 128 ? static_cast<int>(scale_byte)
                                     : static_cast<int>(scale_byte) - 256;
  return read_half(block + 208) * static_cast<float>(scale * quant);
}

__device__ float decode_q8(const std::uint8_t* block, int index) {
  return read_half(block) *
         static_cast<float>(static_cast<std::int8_t>(block[2 + index]));
}

template <QuantKind Kind>
__device__ float decode_weight(const std::uint8_t* block, int index) {
  if constexpr (Kind == QuantKind::kQ4K) return decode_q4(block, index);
  if constexpr (Kind == QuantKind::kQ6K) return decode_q6(block, index);
  return decode_q8(block, index);
}

__global__ void quantize_bf16_q8(const __nv_bfloat16* input, Q8Block* output,
                                 std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const std::size_t q8_index = index / kWarpSize;
  float value = index < count ? __bfloat162float(input[index]) : 0.0F;
  float maximum = fabsf(value);
  for (int offset = 16; offset > 0; offset /= 2) {
    maximum = fmaxf(maximum,
                    __shfl_down_sync(0xFFFFFFFFU, maximum, offset, kWarpSize));
  }
  maximum = __shfl_sync(0xFFFFFFFFU, maximum, 0, kWarpSize);
  const float scale = maximum == 0.0F ? 0.0F : maximum / 127.0F;
  if (lane == 0) output[q8_index].scale = scale;
  if (index < count) {
    output[q8_index].values[lane] =
        scale == 0.0F ? 0 : static_cast<std::int8_t>(roundf(value / scale));
  }
}

template <QuantKind Kind, int Warps, bool PackedLoads = false>
__global__ void quant_mmv(const std::uint8_t* weights, std::size_t rows,
                          std::size_t columns, const Q8Block* activation,
                          float* output) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * Warps + warp;
  if (row >= rows) return;

  constexpr std::size_t kWeightBytes =
      Kind == QuantKind::kQ4K ? kQ4KBytes
      : Kind == QuantKind::kQ6K ? kQ6KBytes
                               : kQ80Bytes;
  constexpr std::size_t kWeightValues =
      Kind == QuantKind::kQ8_0 ? kWarpSize : kValuesPerWeightBlock;
  const std::uint8_t* row_weights =
      weights + row * (columns / kWeightValues) * kWeightBytes;
  float sum = 0.0F;
  if constexpr (PackedLoads && Kind == QuantKind::kQ4K) {
    const std::size_t n_blocks = columns / kValuesPerWeightBlock;
    for (std::size_t weight_block = 0; weight_block < n_blocks;
         ++weight_block) {
      const std::uint8_t* block =
          row_weights + weight_block * kQ4KBytes;
      const float d = read_half(block);
      const float dmin = read_half(block + 2);
      int scales[8];
      int mins[8];
#pragma unroll
      for (int smi = 0; smi < 8; ++smi) {
        q4_scale_min(block + 4, smi, &scales[smi], &mins[smi]);
      }
      std::uint8_t qs[4];
#pragma unroll
      for (int group = 0; group < 4; ++group) {
        qs[group] = block[16 + group * 32 + lane];
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int group = i / 2;
        const int high = i & 1;
        const int quant = high == 0 ? qs[group] & 15 : qs[group] >> 4;
        const float weight = d * static_cast<float>(scales[i] * quant) -
                             dmin * static_cast<float>(mins[i]);
        const std::size_t column =
            weight_block * kValuesPerWeightBlock +
            static_cast<std::size_t>(lane + i * kWarpSize);
        const Q8Block& q8 = activation[column / kWarpSize];
        const float value =
            q8.scale * static_cast<float>(q8.values[column % kWarpSize]);
        sum = __fadd_rn(sum, __fmul_rn(weight, value));
      }
    }
  } else if constexpr (PackedLoads && Kind == QuantKind::kQ6K) {
    const std::size_t n_blocks = columns / kValuesPerWeightBlock;
    for (std::size_t weight_block = 0; weight_block < n_blocks;
         ++weight_block) {
      const std::uint8_t* block =
          row_weights + weight_block * kQ6KBytes;
      const float d = read_half(block + 208);
      int scales[16];
#pragma unroll
      for (int s = 0; s < 16; ++s) {
        const std::uint8_t scale_byte = block[192 + s];
        scales[s] = scale_byte < 128 ? static_cast<int>(scale_byte)
                                     : static_cast<int>(scale_byte) - 256;
      }
      const std::uint8_t ql[4] = {
          block[lane], block[32 + lane], block[64 + lane],
          block[96 + lane]};
      const std::uint8_t qh[2] = {block[128 + lane], block[160 + lane]};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int half = i / 4;
        const int group = i % 4;
        const std::uint8_t low = ql[half * 2 + (group & 1)];
        const int low_four = group < 2 ? low & 15 : low >> 4;
        const int high_two = (qh[half] >> (group * 2)) & 3;
        const int quant = (low_four | (high_two << 4)) - 32;
        const int scale = scales[half * 8 + (lane / 16) + group * 2];
        const float weight = d * static_cast<float>(scale * quant);
        const std::size_t column =
            weight_block * kValuesPerWeightBlock +
            static_cast<std::size_t>(lane + i * kWarpSize);
        const Q8Block& q8 = activation[column / kWarpSize];
        const float value =
            q8.scale * static_cast<float>(q8.values[column % kWarpSize]);
        sum = __fadd_rn(sum, __fmul_rn(weight, value));
      }
    }
  } else {
    for (std::size_t column = lane; column < columns; column += kWarpSize) {
      const std::size_t weight_block = column / kWeightValues;
      const int within = static_cast<int>(column % kWeightValues);
      const std::uint8_t* block = row_weights + weight_block * kWeightBytes;
      const float weight = decode_weight<Kind>(block, within);
      const Q8Block& q8 = activation[column / kWarpSize];
      const float value =
          q8.scale * static_cast<float>(q8.values[column % kWarpSize]);
      sum = __fadd_rn(sum, __fmul_rn(weight, value));
    }
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    sum = __fadd_rn(
        sum, __shfl_down_sync(0xFFFFFFFFU, sum, offset, kWarpSize));
  }
  if (lane == 0) output[row] = sum;
}

__device__ __forceinline__ __nv_bfloat16 admitted_swiglu(float gate, float up) {
  const float activated = gate / (1.0F + expf(-gate));
  return __float2bfloat16_rn(__fmul_rn(activated, up));
}

// Two-pointer packed Q4_K gate/up: independent row pointers, one activation
// load, separate accumulators, admitted SwiGLU in the epilogue. Gate and up
// stay as distinct weight pointers.
template <int Warps, bool UseStagedQ8>
__global__ void quant_mmv_q4k_gate_up_swiglu(
    const std::uint8_t* gate_weights, const std::uint8_t* up_weights,
    std::size_t rows, std::size_t columns, const void* activation,
    __nv_bfloat16* output) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * Warps + warp;
  if (row >= rows) return;

  const std::size_t n_blocks = columns / kValuesPerWeightBlock;
  const std::uint8_t* gate_row =
      gate_weights + row * n_blocks * kQ4KBytes;
  const std::uint8_t* up_row = up_weights + row * n_blocks * kQ4KBytes;
  float sum_g = 0.0F;
  float sum_u = 0.0F;
  for (std::size_t weight_block = 0; weight_block < n_blocks; ++weight_block) {
    const std::uint8_t* gblock = gate_row + weight_block * kQ4KBytes;
    const std::uint8_t* ublock = up_row + weight_block * kQ4KBytes;
    const float gd = read_half(gblock);
    const float gdmin = read_half(gblock + 2);
    const float ud = read_half(ublock);
    const float udmin = read_half(ublock + 2);
    int gscales[8];
    int gmins[8];
    int uscales[8];
    int umins[8];
#pragma unroll
    for (int smi = 0; smi < 8; ++smi) {
      q4_scale_min(gblock + 4, smi, &gscales[smi], &gmins[smi]);
      q4_scale_min(ublock + 4, smi, &uscales[smi], &umins[smi]);
    }
    std::uint8_t gqs[4];
    std::uint8_t uqs[4];
#pragma unroll
    for (int group = 0; group < 4; ++group) {
      gqs[group] = gblock[16 + group * 32 + lane];
      uqs[group] = ublock[16 + group * 32 + lane];
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int group = i / 2;
      const int high = i & 1;
      const int gquant = high == 0 ? gqs[group] & 15 : gqs[group] >> 4;
      const int uquant = high == 0 ? uqs[group] & 15 : uqs[group] >> 4;
      const float gweight = gd * static_cast<float>(gscales[i] * gquant) -
                            gdmin * static_cast<float>(gmins[i]);
      const float uweight = ud * static_cast<float>(uscales[i] * uquant) -
                            udmin * static_cast<float>(umins[i]);
      const std::size_t column =
          weight_block * kValuesPerWeightBlock +
          static_cast<std::size_t>(lane + i * kWarpSize);
      float value = 0.0F;
      if constexpr (UseStagedQ8) {
        const Q8Block* q8 = static_cast<const Q8Block*>(activation);
        const Q8Block& block = q8[column / kWarpSize];
        value = block.scale * static_cast<float>(block.values[column % kWarpSize]);
      } else {
        const __nv_bfloat16* act =
            static_cast<const __nv_bfloat16*>(activation);
        value = __bfloat162float(act[column]);
      }
      sum_g = __fadd_rn(sum_g, __fmul_rn(gweight, value));
      sum_u = __fadd_rn(sum_u, __fmul_rn(uweight, value));
    }
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    sum_g = __fadd_rn(
        sum_g, __shfl_down_sync(0xFFFFFFFFU, sum_g, offset, kWarpSize));
    sum_u = __fadd_rn(
        sum_u, __shfl_down_sync(0xFFFFFFFFU, sum_u, offset, kWarpSize));
  }
  if (lane == 0) output[row] = admitted_swiglu(sum_g, sum_u);
}

template <QuantKind Kind, int PromptRowsPerTile>
__global__ void quant_mmq(const std::uint8_t* weights, std::size_t output_rows,
                          std::size_t columns, const Q8Block* prompt,
                          std::size_t prompt_rows, float* output) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const std::size_t output_row =
      static_cast<std::size_t>(blockIdx.x) * (kThreads / kWarpSize) + warp;
  const std::size_t prompt_start =
      static_cast<std::size_t>(blockIdx.y) * PromptRowsPerTile;
  if (output_row >= output_rows) return;

  constexpr std::size_t kWeightBytes =
      Kind == QuantKind::kQ4K ? kQ4KBytes
      : Kind == QuantKind::kQ6K ? kQ6KBytes
                               : kQ80Bytes;
  constexpr std::size_t kWeightValues =
      Kind == QuantKind::kQ8_0 ? kWarpSize : kValuesPerWeightBlock;
  const std::size_t q8_blocks_per_row = columns / kWarpSize;
  const std::uint8_t* row_weights =
      weights + output_row * (columns / kWeightValues) * kWeightBytes;
  float sums[PromptRowsPerTile] = {};
  for (std::size_t column = lane; column < columns; column += kWarpSize) {
    const std::size_t weight_block = column / kWeightValues;
    const int within = static_cast<int>(column % kWeightValues);
    const std::uint8_t* block = row_weights + weight_block * kWeightBytes;
    const float weight = decode_weight<Kind>(block, within);
#pragma unroll
    for (int prompt_offset = 0; prompt_offset < PromptRowsPerTile;
         ++prompt_offset) {
      const std::size_t prompt_row = prompt_start + prompt_offset;
      if (prompt_row < prompt_rows) {
        const Q8Block& q8 =
            prompt[prompt_row * q8_blocks_per_row + column / kWarpSize];
        const float value =
            q8.scale * static_cast<float>(q8.values[column % kWarpSize]);
        sums[prompt_offset] = __fadd_rn(
            sums[prompt_offset], __fmul_rn(weight, value));
      }
    }
  }
#pragma unroll
  for (int prompt_offset = 0; prompt_offset < PromptRowsPerTile;
       ++prompt_offset) {
    for (int offset = 16; offset > 0; offset /= 2) {
      sums[prompt_offset] = __fadd_rn(
          sums[prompt_offset],
          __shfl_down_sync(0xFFFFFFFFU, sums[prompt_offset], offset,
                           kWarpSize));
    }
    const std::size_t prompt_row = prompt_start + prompt_offset;
    if (lane == 0 && prompt_row < prompt_rows) {
      output[prompt_row * output_rows + output_row] = sums[prompt_offset];
    }
  }
}

__global__ void q8_mmq_bf16_reference(const std::uint8_t* weights,
                                      std::size_t rows, std::size_t columns,
                                      const __nv_bfloat16* activation,
                                      std::size_t prompt_rows, float* output) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * (kThreads / kWarpSize) + warp;
  const std::size_t prompt_row = blockIdx.y;
  if (row >= rows || prompt_row >= prompt_rows) return;
  const std::uint8_t* row_weights = weights + row * (columns / 32) * 34;
  const __nv_bfloat16* row_activation = activation + prompt_row * columns;
  float sum = 0.0F;
  for (std::size_t column = lane; column < columns; column += kWarpSize) {
    const std::uint8_t* block = row_weights + (column / 32) * 34;
    const float weight =
        read_half(block) *
        static_cast<float>(static_cast<std::int8_t>(block[2 + column % 32]));
    sum = __fadd_rn(
        sum, __fmul_rn(weight, __bfloat162float(row_activation[column])));
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    sum = __fadd_rn(
        sum, __shfl_down_sync(0xFFFFFFFFU, sum, offset, kWarpSize));
  }
  if (lane == 0) output[prompt_row * rows + row] = sum;
}

template <int PromptTile>
__global__ void q8_mmq_bf16_tiled(const std::uint8_t* weights, std::size_t rows,
                                  std::size_t columns,
                                  const __nv_bfloat16* activation,
                                  std::size_t prompt_rows, float* output) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const std::size_t output_row =
      static_cast<std::size_t>(blockIdx.x) * (kThreads / kWarpSize) + warp;
  const std::size_t prompt_start =
      static_cast<std::size_t>(blockIdx.y) * PromptTile;
  if (output_row >= rows) return;

  const std::uint8_t* row_weights =
      weights + output_row * (columns / 32) * 34;
  float sums[PromptTile] = {};
  for (std::size_t column = lane; column < columns; column += kWarpSize) {
    const std::uint8_t* block = row_weights + (column / 32) * 34;
    const float weight =
        read_half(block) *
        static_cast<float>(static_cast<std::int8_t>(block[2 + column % 32]));
#pragma unroll
    for (int prompt_offset = 0; prompt_offset < PromptTile; ++prompt_offset) {
      const std::size_t prompt_row = prompt_start + prompt_offset;
      if (prompt_row < prompt_rows) {
        sums[prompt_offset] = __fadd_rn(
            sums[prompt_offset],
            __fmul_rn(weight, __bfloat162float(
                                  activation[prompt_row * columns + column])));
      }
    }
  }
#pragma unroll
  for (int prompt_offset = 0; prompt_offset < PromptTile; ++prompt_offset) {
    for (int offset = 16; offset > 0; offset /= 2) {
      sums[prompt_offset] = __fadd_rn(
          sums[prompt_offset],
          __shfl_down_sync(0xFFFFFFFFU, sums[prompt_offset], offset,
                           kWarpSize));
    }
    const std::size_t prompt_row = prompt_start + prompt_offset;
    if (lane == 0 && prompt_row < prompt_rows) {
      output[prompt_row * rows + output_row] = sums[prompt_offset];
    }
  }
}

template <QuantKind Kind>
__global__ void quant_row_decode(const std::uint8_t* weights,
                                 std::size_t columns, std::size_t row,
                                 __nv_bfloat16* output) {
  const std::size_t column =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= columns) return;
  constexpr std::size_t kWeightBytes =
      Kind == QuantKind::kQ4K ? kQ4KBytes
      : Kind == QuantKind::kQ6K ? kQ6KBytes
                               : kQ80Bytes;
  constexpr std::size_t kWeightValues =
      Kind == QuantKind::kQ8_0 ? kWarpSize : kValuesPerWeightBlock;
  const std::size_t row_bytes = columns / kWeightValues * kWeightBytes;
  const std::uint8_t* block =
      weights + row * row_bytes + column / kWeightValues * kWeightBytes;
  output[column] = __float2bfloat16_rn(
      decode_weight<Kind>(block, static_cast<int>(column % kWeightValues)));
}

template <QuantKind Kind>
__global__ void quant_rows_decode_widen(const std::uint8_t* weights,
                                        std::size_t columns,
                                        const std::size_t* token_ids,
                                        float* output) {
  const std::size_t column =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= columns) return;
  const std::size_t row = blockIdx.y;
  constexpr std::size_t kWeightBytes =
      Kind == QuantKind::kQ4K ? kQ4KBytes
      : Kind == QuantKind::kQ6K ? kQ6KBytes
                               : kQ80Bytes;
  constexpr std::size_t kWeightValues =
      Kind == QuantKind::kQ8_0 ? kWarpSize : kValuesPerWeightBlock;
  const std::size_t row_bytes = columns / kWeightValues * kWeightBytes;
  const std::size_t token = token_ids[row];
  const std::uint8_t* block =
      weights + token * row_bytes + column / kWeightValues * kWeightBytes;
  const __nv_bfloat16 rounded = __float2bfloat16_rn(
      decode_weight<Kind>(block, static_cast<int>(column % kWeightValues)));
  output[row * columns + column] = __bfloat162float(rounded);
}

}  // namespace

std::size_t q8_workspace_bytes(std::size_t columns) noexcept {
  return columns % kValuesPerWeightBlock == 0
             ? (columns / kWarpSize) * sizeof(Q8Block)
             : 0;
}

std::size_t q8_1_workspace_bytes(std::size_t columns) noexcept {
  return q8_workspace_bytes(columns);
}

std::size_t q8_prompt_workspace_bytes(std::size_t prompt_rows,
                                      std::size_t columns) noexcept {
  if (prompt_rows == 0 || columns % kValuesPerWeightBlock != 0) return 0;
  return prompt_rows * (columns / kWarpSize) * sizeof(Q8Block);
}

constexpr const char kSelectedMmvLoadPath[] = "packed";
// Unrepresented family/shape fallback. Engine summary is computed from the
// admission table because Q8/Q6 production pins already use approximations.
constexpr const char kSelectedProductionNumericsPath[] = "strict";

constexpr ProductionAdmissionEntry kProductionAdmission[] = {
    {"q4_k_k5120_packed_q8_fp32", "q4_k", 5120, kStagingQ8Fp32, "packed_fp32",
     kLegalProductionNumericsPathStrict, true, false, true},
    {"q4_k_k17408_packed_q8_fp32", "q4_k", 17408, kStagingQ8Fp32, "packed_fp32",
     kLegalProductionNumericsPathStrict, true, false, true},
    {"q4_k_integer_q8", "q4_k", 0, kStagingQ8Fp32, "integer_dp4a_q8",
     kLegalProductionNumericsPathStrict, false, false, true},
    {"q4_k_integer_quartz_q8_1_sum_q", "q4_k", 0, kStagingQuartzQ81SumQ,
     "integer_dp4a_q8_1", kLegalProductionNumericsPathStrict, false, false,
     true},
    {"q4_k_integer_llama_q8_1_sum_x", "q4_k", 0, kStagingLlamaQ81SumX,
     "integer_dp4a_q8_1_sum_x", kLegalProductionNumericsPathStrict, false, false,
     true},
    {"q8_0_k5120_dp4a_q8_1", "q8_0", 5120, kStagingQuartzQ81SumQ, "dp4a_q8_1",
     kLegalProductionNumericsPathOptimized, true, false, true},
    {"q8_0_k6144_dp4a_q8_1", "q8_0", 6144, kStagingQuartzQ81SumQ, "dp4a_q8_1",
     kLegalProductionNumericsPathOptimized, true, false, true},
    {"q8_0_direct_bf16", "q8_0", 0, kStagingBf16Direct, "direct_bf16",
     kLegalProductionNumericsPathStrict, false, false, true},
    {"q6_k_k5120_integer_q8_1", "q6_k", 5120, kStagingQuartzQ81SumQ,
     "integer_q8_1", kLegalProductionNumericsPathOptimized, true, false, true},
    {"q6_k_packed_fp32", "q6_k", 256, kStagingQ8Fp32, "packed_fp32",
     kLegalProductionNumericsPathStrict, true, false, true},
};

const ProductionAdmissionEntry* match_admission(const char* family,
                                                std::size_t columns,
                                                const char* staging,
                                                const char* variant) noexcept {
  const ProductionAdmissionEntry* wildcard = nullptr;
  for (const ProductionAdmissionEntry& entry : kProductionAdmission) {
    if (family == nullptr || std::strcmp(entry.family, family) != 0) continue;
    if (staging != nullptr && std::strcmp(entry.staging, staging) != 0) continue;
    if (variant != nullptr && std::strcmp(entry.variant, variant) != 0) continue;
    if (entry.columns == columns) return &entry;
    if (entry.columns == 0) wildcard = &entry;
  }
  return wildcard;
}

unsigned int selected_mmv_warps(std::size_t rows) noexcept {
  if (rows <= 48) return 4;
  if (rows <= 1024) return 8;
  if (rows <= 5120) return 16;
  if (rows <= 10240) return 8;
  if (rows <= 12288) return 4;
  if (rows <= 17408) return 8;
  return 4;
}

const char* selected_mmv_load_path() noexcept { return kSelectedMmvLoadPath; }

const char* unrepresented_production_numerics_path() noexcept {
  return kSelectedProductionNumericsPath;
}

const char* selected_production_numerics_path() noexcept {
  bool saw_strict = false;
  bool saw_optimized = false;
  for (const ProductionAdmissionEntry& entry : kProductionAdmission) {
    if (!entry.currently_selected) continue;
    if (std::strcmp(entry.production_dispatch,
                    kLegalProductionNumericsPathOptimized) == 0) {
      saw_optimized = true;
    } else {
      saw_strict = true;
    }
  }
  if (saw_optimized && saw_strict) return kLegalProductionNumericsPathMixed;
  if (saw_optimized) return kLegalProductionNumericsPathOptimized;
  return kSelectedProductionNumericsPath;
}

bool production_numerics_optimized_admitted() noexcept {
  for (const ProductionAdmissionEntry& entry : kProductionAdmission) {
    if (entry.currently_selected &&
        std::strcmp(entry.production_dispatch,
                    kLegalProductionNumericsPathOptimized) == 0) {
      return true;
    }
  }
  return false;
}

bool mmv_uses_packed_loads() noexcept {
  return std::strcmp(kSelectedMmvLoadPath, "packed") == 0;
}

const char* production_numerics_path_for_family(const char* family) noexcept {
  const ProductionAdmissionEntry* selected = nullptr;
  for (const ProductionAdmissionEntry& entry : kProductionAdmission) {
    if (family == nullptr || std::strcmp(entry.family, family) != 0) continue;
    if (!entry.currently_selected) continue;
    selected = &entry;
    break;
  }
  if (selected == nullptr) return kSelectedProductionNumericsPath;
  return selected->production_dispatch;
}

const char* production_numerics_path_for(const char* family, std::size_t columns,
                                         const char* staging,
                                         const char* variant) noexcept {
  const ProductionAdmissionEntry* entry =
      match_admission(family, columns, staging, variant);
  if (entry == nullptr) return kSelectedProductionNumericsPath;
  return entry->production_dispatch;
}

bool production_numerics_testing_admitted(const char* family,
                                          std::size_t columns,
                                          const char* staging,
                                          const char* variant) noexcept {
  const ProductionAdmissionEntry* entry =
      match_admission(family, columns, staging, variant);
  return entry != nullptr && entry->testing_admitted;
}

bool production_numerics_v2_admitted(const char* family, std::size_t columns,
                                     const char* staging,
                                     const char* variant) noexcept {
  const ProductionAdmissionEntry* entry =
      match_admission(family, columns, staging, variant);
  return entry != nullptr && entry->v2_admitted;
}

std::size_t production_numerics_admission_count() noexcept {
  return sizeof(kProductionAdmission) / sizeof(kProductionAdmission[0]);
}

const ProductionAdmissionEntry* production_numerics_admission_entry(
    std::size_t index) noexcept {
  if (index >= production_numerics_admission_count()) return nullptr;
  return &kProductionAdmission[index];
}

bool legal_mmv_load_path(const char* load_path) noexcept {
  return load_path != nullptr &&
         (std::strcmp(load_path, "elementwise") == 0 ||
          std::strcmp(load_path, "packed") == 0);
}

bool mmv_path_is_packed(const char* load_path) noexcept {
  return load_path != nullptr && std::strcmp(load_path, "packed") == 0;
}

int mmv_packed_occupancy(QuantKind kind, unsigned int warps) noexcept {
  int occupancy = 0;
  cudaError_t error = cudaErrorInvalidValue;
  if (kind == QuantKind::kQ4K && warps == 4) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, quant_mmv<QuantKind::kQ4K, 4, true>, 128, 0);
  } else if (kind == QuantKind::kQ4K && warps == 8) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, quant_mmv<QuantKind::kQ4K, 8, true>, 256, 0);
  } else if (kind == QuantKind::kQ4K && warps == 16) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, quant_mmv<QuantKind::kQ4K, 16, true>, 512, 0);
  } else if (kind == QuantKind::kQ6K && warps == 4) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, quant_mmv<QuantKind::kQ6K, 4, true>, 128, 0);
  } else if (kind == QuantKind::kQ6K && warps == 8) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, quant_mmv<QuantKind::kQ6K, 8, true>, 256, 0);
  } else if (kind == QuantKind::kQ6K && warps == 16) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, quant_mmv<QuantKind::kQ6K, 16, true>, 512, 0);
  }
  if (error != cudaSuccess) return 0;
  return occupancy;
}

bool legal_mmq_prompt_tile(unsigned int prompt_tile) noexcept {
  return prompt_tile == 1 || prompt_tile == 2 || prompt_tile == 4 ||
         prompt_tile == 8 || prompt_tile == 16 || prompt_tile == 32 ||
         prompt_tile == 64;
}

unsigned int selected_mmq_prompt_tile(QuantKind kind,
                                      std::size_t prompt_rows) noexcept {
  if (kind == QuantKind::kQ8_0) {
    if (prompt_rows <= 1) return 1;
    if (prompt_rows <= 2) return 2;
    return 4;
  }
  if (kind == QuantKind::kQ6K) {
    if (prompt_rows <= 1) return 1;
    if (prompt_rows <= 2) return 2;
    if (prompt_rows <= 4) return 4;
    return 8;
  }
  if (prompt_rows <= 1) return 1;
  if (prompt_rows <= 2) return 2;
  if (prompt_rows <= 8) return 4;
  if (prompt_rows <= 256) return 8;
  return 4;
}

unsigned int selected_mmq_prompt_tile(std::size_t prompt_rows) noexcept {
  return selected_mmq_prompt_tile(QuantKind::kQ4K, prompt_rows);
}

template <int Warps, bool PackedLoads>
cudaError_t launch_mmv_kernel(QuantKind kind, const std::uint8_t* weights,
                              std::size_t rows, std::size_t columns,
                              const Q8Block* activation, float* output,
                              cudaStream_t stream) noexcept {
  constexpr int kLaunchThreads = Warps * kWarpSize;
  const unsigned int row_blocks =
      static_cast<unsigned int>((rows + Warps - 1) / Warps);
  if (kind == QuantKind::kQ4K) {
    quant_mmv<QuantKind::kQ4K, Warps, PackedLoads>
        <<<row_blocks, kLaunchThreads, 0, stream>>>(
            weights, rows, columns, activation, output);
  } else if (kind == QuantKind::kQ6K) {
    quant_mmv<QuantKind::kQ6K, Warps, PackedLoads>
        <<<row_blocks, kLaunchThreads, 0, stream>>>(
            weights, rows, columns, activation, output);
  } else {
    quant_mmv<QuantKind::kQ8_0, Warps, false>
        <<<row_blocks, kLaunchThreads, 0, stream>>>(
            weights, rows, columns, activation, output);
  }
  return cudaPeekAtLastError();
}

cudaError_t launch_mmv_after_quant(QuantKind kind, const std::uint8_t* weights,
                                   std::size_t rows, std::size_t columns,
                                   const Q8Block* activation, float* output,
                                   unsigned int warps, bool packed,
                                   cudaStream_t stream) noexcept {
  if (packed) {
    if (warps == 4) {
      return launch_mmv_kernel<4, true>(kind, weights, rows, columns,
                                        activation, output, stream);
    }
    if (warps == 8) {
      return launch_mmv_kernel<8, true>(kind, weights, rows, columns,
                                        activation, output, stream);
    }
    return launch_mmv_kernel<16, true>(kind, weights, rows, columns, activation,
                                       output, stream);
  }
  if (warps == 4) {
    return launch_mmv_kernel<4, false>(kind, weights, rows, columns, activation,
                                       output, stream);
  }
  if (warps == 8) {
    return launch_mmv_kernel<8, false>(kind, weights, rows, columns, activation,
                                       output, stream);
  }
  return launch_mmv_kernel<16, false>(kind, weights, rows, columns, activation,
                                      output, stream);
}

cudaError_t launch_quant_mmv_variant(
    QuantKind kind, const std::uint8_t* weights, std::size_t rows,
    std::size_t columns, const __nv_bfloat16* activation,
    Q8Block* q8_workspace, float* output, unsigned int warps,
    cudaStream_t stream) noexcept {
  return launch_quant_mmv_path(kind, weights, rows, columns, activation,
                               q8_workspace, output, kSelectedMmvLoadPath,
                               stream, warps);
}

cudaError_t launch_quant_mmv(QuantKind kind, const std::uint8_t* weights,
                             std::size_t rows, std::size_t columns,
                             const __nv_bfloat16* activation,
                             Q8Block* q8_workspace, float* output,
                             cudaStream_t stream) noexcept {
  if (kind == QuantKind::kQ4K && q4_decode_uses_integer()) {
    return launch_q4k_coop_mmv(
        weights, rows, columns, activation, q8_workspace, output,
        effective_q4_decode_warps_per_row(), q4_decode_uses_q8_1(), stream);
  }
  if (kind == QuantKind::kQ4K) {
    record_q4_launch_variant(kQ4LaunchVariantPacked);
  }
  if (kind == QuantKind::kQ6K && q6_decode_uses_integer_for_rows(rows)) {
    return launch_q6k_coop_mmv(
        weights, rows, columns, activation, q8_workspace, output,
        effective_q6_decode_warps_per_row(), q6_decode_uses_q8_1(), stream);
  }
  return launch_quant_mmv_path(kind, weights, rows, columns, activation,
                               q8_workspace, output, kSelectedMmvLoadPath,
                               stream);
}

cudaError_t launch_quantize_bf16_q8(const __nv_bfloat16* activation, Q8Block* q8,
                                    std::size_t columns,
                                    cudaStream_t stream) noexcept {
  if (activation == nullptr || q8 == nullptr || columns == 0 ||
      columns % kValuesPerWeightBlock != 0) {
    return cudaErrorInvalidValue;
  }
  const unsigned int quant_blocks =
      static_cast<unsigned int>((columns + kThreads - 1) / kThreads);
  quantize_bf16_q8<<<quant_blocks, kThreads, 0, stream>>>(activation, q8,
                                                          columns);
  return cudaPeekAtLastError();
}

cudaError_t launch_quant_mmv_prequant(QuantKind kind, const std::uint8_t* weights,
                                      std::size_t rows, std::size_t columns,
                                      const Q8Block* q8, float* output,
                                      cudaStream_t stream) noexcept {
  if (weights == nullptr || q8 == nullptr || output == nullptr || rows == 0 ||
      columns == 0 || columns % kValuesPerWeightBlock != 0 ||
      (kind != QuantKind::kQ4K && kind != QuantKind::kQ6K &&
       kind != QuantKind::kQ8_0)) {
    return cudaErrorInvalidValue;
  }
  // Q8Block* may dispatch only Q8Block-compatible kernels. Never reinterpret
  // this pointer as Q8_1 because kSelectedQ4DecodePath became integer_q8_1.
  if (kind == QuantKind::kQ4K && q4_decode_uses_integer_q8block()) {
    return launch_q4k_coop_mmv_prequant_q8(
        weights, rows, columns, q8, output, effective_q4_decode_warps_per_row(),
        stream);
  }
  const unsigned int warps = selected_mmv_warps(rows);
  if (warps != 4 && warps != 8 && warps != 16) return cudaErrorInvalidValue;
  if (kind == QuantKind::kQ4K) {
    record_q4_launch_variant(kQ4LaunchVariantPackedPrequant);
  }
  const bool packed = kind != QuantKind::kQ8_0;
  return launch_mmv_after_quant(kind, weights, rows, columns, q8, output, warps,
                                packed, stream);
}

template <int Warps, bool UseStagedQ8>
cudaError_t launch_gate_up_kernel(const std::uint8_t* gate_weights,
                                  const std::uint8_t* up_weights,
                                  std::size_t rows, std::size_t columns,
                                  const void* activation,
                                  __nv_bfloat16* output,
                                  cudaStream_t stream) noexcept {
  constexpr int kLaunchThreads = Warps * kWarpSize;
  const unsigned int row_blocks =
      static_cast<unsigned int>((rows + Warps - 1) / Warps);
  quant_mmv_q4k_gate_up_swiglu<Warps, UseStagedQ8>
      <<<row_blocks, kLaunchThreads, 0, stream>>>(
          gate_weights, up_weights, rows, columns, activation, output);
  return cudaPeekAtLastError();
}

template <bool UseStagedQ8>
cudaError_t launch_gate_up_warps(const std::uint8_t* gate_weights,
                                 const std::uint8_t* up_weights,
                                 std::size_t rows, std::size_t columns,
                                 const void* activation, __nv_bfloat16* output,
                                 cudaStream_t stream) noexcept {
  const unsigned int warps = selected_mmv_warps(rows);
  if (warps == 4) {
    return launch_gate_up_kernel<4, UseStagedQ8>(
        gate_weights, up_weights, rows, columns, activation, output, stream);
  }
  if (warps == 8) {
    return launch_gate_up_kernel<8, UseStagedQ8>(
        gate_weights, up_weights, rows, columns, activation, output, stream);
  }
  return launch_gate_up_kernel<16, UseStagedQ8>(
      gate_weights, up_weights, rows, columns, activation, output, stream);
}

cudaError_t launch_q4k_gate_up_swiglu_prequant(
    const std::uint8_t* gate_weights, const std::uint8_t* up_weights,
    std::size_t rows, std::size_t columns, const Q8Block* staged,
    __nv_bfloat16* output, cudaStream_t stream) noexcept {
  const unsigned int warps = selected_mmv_warps(rows);
  if (gate_weights == nullptr || up_weights == nullptr || staged == nullptr ||
      output == nullptr || rows == 0 || columns == 0 ||
      columns % kValuesPerWeightBlock != 0 ||
      (warps != 4 && warps != 8 && warps != 16)) {
    return cudaErrorInvalidValue;
  }
  record_q4_launch_variant(kQ4LaunchVariantPairedStaged);
  return launch_gate_up_warps<true>(gate_weights, up_weights, rows, columns,
                                    staged, output, stream);
}

cudaError_t launch_q4k_gate_up_swiglu(
    const std::uint8_t* gate_weights, const std::uint8_t* up_weights,
    std::size_t rows, std::size_t columns, const __nv_bfloat16* activation,
    Q8Block* workspace, __nv_bfloat16* output, cudaStream_t stream) noexcept {
  (void)workspace;
  const unsigned int warps = selected_mmv_warps(rows);
  if (gate_weights == nullptr || up_weights == nullptr ||
      activation == nullptr || output == nullptr || rows == 0 ||
      columns == 0 || columns % kValuesPerWeightBlock != 0 ||
      (warps != 4 && warps != 8 && warps != 16)) {
    return cudaErrorInvalidValue;
  }
  record_q4_launch_variant(kQ4LaunchVariantPaired);
  return launch_gate_up_warps<false>(gate_weights, up_weights, rows, columns,
                                     activation, output, stream);
}

int q4k_gate_up_swiglu_occupancy(unsigned int warps, bool staged) noexcept {
  int occupancy = 0;
  cudaError_t error = cudaErrorInvalidValue;
  if (staged) {
    if (warps == 4) {
      error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, quant_mmv_q4k_gate_up_swiglu<4, true>, 128, 0);
    } else if (warps == 8) {
      error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, quant_mmv_q4k_gate_up_swiglu<8, true>, 256, 0);
    } else if (warps == 16) {
      error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, quant_mmv_q4k_gate_up_swiglu<16, true>, 512, 0);
    }
  } else if (warps == 4) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, quant_mmv_q4k_gate_up_swiglu<4, false>, 128, 0);
  } else if (warps == 8) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, quant_mmv_q4k_gate_up_swiglu<8, false>, 256, 0);
  } else if (warps == 16) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, quant_mmv_q4k_gate_up_swiglu<16, false>, 512, 0);
  }
  if (error != cudaSuccess) return 0;
  return occupancy;
}

void q4k_gate_up_swiglu_kernel_attributes(unsigned int warps, bool staged,
                                          int* registers,
                                          std::size_t* local_bytes,
                                          int* occupancy) noexcept {
  cudaFuncAttributes attrs{};
  cudaError_t error = cudaErrorInvalidValue;
  if (staged) {
    if (warps == 4) {
      error = cudaFuncGetAttributes(&attrs,
                                    quant_mmv_q4k_gate_up_swiglu<4, true>);
    } else if (warps == 8) {
      error = cudaFuncGetAttributes(&attrs,
                                    quant_mmv_q4k_gate_up_swiglu<8, true>);
    } else if (warps == 16) {
      error = cudaFuncGetAttributes(&attrs,
                                    quant_mmv_q4k_gate_up_swiglu<16, true>);
    }
  } else if (warps == 4) {
    error = cudaFuncGetAttributes(&attrs,
                                  quant_mmv_q4k_gate_up_swiglu<4, false>);
  } else if (warps == 8) {
    error = cudaFuncGetAttributes(&attrs,
                                  quant_mmv_q4k_gate_up_swiglu<8, false>);
  } else if (warps == 16) {
    error = cudaFuncGetAttributes(&attrs,
                                  quant_mmv_q4k_gate_up_swiglu<16, false>);
  }
  if (registers != nullptr) *registers = error == cudaSuccess ? attrs.numRegs : 0;
  if (local_bytes != nullptr) {
    *local_bytes =
        error == cudaSuccess ? static_cast<std::size_t>(attrs.localSizeBytes)
                             : 0;
  }
  if (occupancy != nullptr) {
    *occupancy = q4k_gate_up_swiglu_occupancy(warps, staged);
  }
}

cudaError_t launch_quant_mmv_path(
    QuantKind kind, const std::uint8_t* weights, std::size_t rows,
    std::size_t columns, const __nv_bfloat16* activation,
    Q8Block* q8_workspace, float* output, const char* load_path,
    cudaStream_t stream, unsigned int warps) noexcept {
  const unsigned int selected_warps =
      warps == 0 ? selected_mmv_warps(rows) : warps;
  if (weights == nullptr || activation == nullptr || q8_workspace == nullptr ||
      output == nullptr || rows == 0 || columns == 0 ||
      columns % kValuesPerWeightBlock != 0 ||
      (selected_warps != 4 && selected_warps != 8 && selected_warps != 16) ||
      (kind != QuantKind::kQ4K && kind != QuantKind::kQ6K &&
       kind != QuantKind::kQ8_0) ||
      !legal_mmv_load_path(load_path)) {
    return cudaErrorInvalidValue;
  }
  const unsigned int quant_blocks =
      static_cast<unsigned int>((columns + kThreads - 1) / kThreads);
  quantize_bf16_q8<<<quant_blocks, kThreads, 0, stream>>>(
      activation, q8_workspace, columns);
  cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  const bool packed =
      mmv_path_is_packed(load_path) && kind != QuantKind::kQ8_0;
  return launch_mmv_after_quant(kind, weights, rows, columns, q8_workspace,
                                output, selected_warps, packed, stream);
}

template <int PromptTile>
cudaError_t launch_mmq_kernel(QuantKind kind, const std::uint8_t* weights,
                              std::size_t output_rows, std::size_t columns,
                              const Q8Block* prompt, std::size_t prompt_rows,
                              float* output, cudaStream_t stream) noexcept {
  const dim3 grid(
      static_cast<unsigned int>((output_rows + 7) / 8),
      static_cast<unsigned int>((prompt_rows + PromptTile - 1) / PromptTile));
  if (kind == QuantKind::kQ4K) {
    quant_mmq<QuantKind::kQ4K, PromptTile><<<grid, kThreads, 0, stream>>>(
        weights, output_rows, columns, prompt, prompt_rows, output);
  } else if (kind == QuantKind::kQ6K) {
    quant_mmq<QuantKind::kQ6K, PromptTile><<<grid, kThreads, 0, stream>>>(
        weights, output_rows, columns, prompt, prompt_rows, output);
  } else {
    quant_mmq<QuantKind::kQ8_0, PromptTile><<<grid, kThreads, 0, stream>>>(
        weights, output_rows, columns, prompt, prompt_rows, output);
  }
  return cudaPeekAtLastError();
}

cudaError_t launch_quant_mmq_variant(
    QuantKind kind, const std::uint8_t* weights, std::size_t output_rows,
    std::size_t columns, const __nv_bfloat16* prompt,
    std::size_t prompt_rows, Q8Block* q8_workspace, float* output,
    unsigned int prompt_tile, cudaStream_t stream) noexcept {
  if (weights == nullptr || prompt == nullptr || q8_workspace == nullptr ||
      output == nullptr || output_rows == 0 || columns == 0 ||
      prompt_rows == 0 || columns % kValuesPerWeightBlock != 0 ||
      !legal_mmq_prompt_tile(prompt_tile) ||
      (kind != QuantKind::kQ4K && kind != QuantKind::kQ6K &&
       kind != QuantKind::kQ8_0)) {
    return cudaErrorInvalidValue;
  }
  const std::size_t prompt_values = prompt_rows * columns;
  const unsigned int quant_blocks = static_cast<unsigned int>(
      (prompt_values + kThreads - 1) / kThreads);
  quantize_bf16_q8<<<quant_blocks, kThreads, 0, stream>>>(
      prompt, q8_workspace, prompt_values);
  cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  if (prompt_tile == 1) {
    return launch_mmq_kernel<1>(kind, weights, output_rows, columns,
                                q8_workspace, prompt_rows, output, stream);
  }
  if (prompt_tile == 2) {
    return launch_mmq_kernel<2>(kind, weights, output_rows, columns,
                                q8_workspace, prompt_rows, output, stream);
  }
  if (prompt_tile == 4) {
    return launch_mmq_kernel<4>(kind, weights, output_rows, columns,
                                q8_workspace, prompt_rows, output, stream);
  }
  if (prompt_tile == 8) {
    return launch_mmq_kernel<8>(kind, weights, output_rows, columns,
                                q8_workspace, prompt_rows, output, stream);
  }
  if (prompt_tile == 16) {
    return launch_mmq_kernel<16>(kind, weights, output_rows, columns,
                                 q8_workspace, prompt_rows, output, stream);
  }
  if (prompt_tile == 32) {
    return launch_mmq_kernel<32>(kind, weights, output_rows, columns,
                                 q8_workspace, prompt_rows, output, stream);
  }
  return launch_mmq_kernel<64>(kind, weights, output_rows, columns,
                               q8_workspace, prompt_rows, output, stream);
}

cudaError_t launch_quant_mmq(QuantKind kind, const std::uint8_t* weights,
                             std::size_t output_rows, std::size_t columns,
                             const __nv_bfloat16* prompt,
                             std::size_t prompt_rows, Q8Block* q8_workspace,
                             float* output, cudaStream_t stream,
                             float* tmp_fixup,
                             std::size_t tmp_fixup_floats) noexcept {
  if ((kind == QuantKind::kQ4K || kind == QuantKind::kQ6K) &&
      prompt_rows >= 8) {
    return launch_quant_mmq_mma(kind, weights, output_rows, columns, prompt,
                                prompt_rows, q8_workspace, output, stream,
                                tmp_fixup, tmp_fixup_floats);
  }
  return launch_quant_mmq_variant(
      kind, weights, output_rows, columns, prompt, prompt_rows, q8_workspace,
      output, selected_mmq_prompt_tile(kind, prompt_rows), stream);
}

template <int PromptTile>
cudaError_t launch_q8_mmq_kernel(const std::uint8_t* weights,
                                 std::size_t output_rows, std::size_t columns,
                                 const __nv_bfloat16* prompt,
                                 std::size_t prompt_rows, float* output,
                                 cudaStream_t stream) noexcept {
  const dim3 grid(
      static_cast<unsigned int>((output_rows + 7) / 8),
      static_cast<unsigned int>((prompt_rows + PromptTile - 1) / PromptTile));
  q8_mmq_bf16_tiled<PromptTile><<<grid, kThreads, 0, stream>>>(
      weights, output_rows, columns, prompt, prompt_rows, output);
  return cudaPeekAtLastError();
}

cudaError_t launch_q8_mmq_bf16_variant(
    const std::uint8_t* weights, std::size_t output_rows, std::size_t columns,
    const __nv_bfloat16* prompt, std::size_t prompt_rows, float* output,
    unsigned int prompt_tile, cudaStream_t stream) noexcept {
  if (weights == nullptr || prompt == nullptr || output == nullptr ||
      output_rows == 0 || columns == 0 || prompt_rows == 0 ||
      columns % kValuesPerWeightBlock != 0 ||
      !legal_mmq_prompt_tile(prompt_tile)) {
    return cudaErrorInvalidValue;
  }
  if (prompt_tile == 1) {
    return launch_q8_mmq_kernel<1>(weights, output_rows, columns, prompt,
                                   prompt_rows, output, stream);
  }
  if (prompt_tile == 2) {
    return launch_q8_mmq_kernel<2>(weights, output_rows, columns, prompt,
                                   prompt_rows, output, stream);
  }
  if (prompt_tile == 4) {
    return launch_q8_mmq_kernel<4>(weights, output_rows, columns, prompt,
                                   prompt_rows, output, stream);
  }
  if (prompt_tile == 8) {
    return launch_q8_mmq_kernel<8>(weights, output_rows, columns, prompt,
                                   prompt_rows, output, stream);
  }
  if (prompt_tile == 16) {
    return launch_q8_mmq_kernel<16>(weights, output_rows, columns, prompt,
                                    prompt_rows, output, stream);
  }
  if (prompt_tile == 32) {
    return launch_q8_mmq_kernel<32>(weights, output_rows, columns, prompt,
                                    prompt_rows, output, stream);
  }
  return launch_q8_mmq_kernel<64>(weights, output_rows, columns, prompt,
                                  prompt_rows, output, stream);
}

cudaError_t launch_q8_mmq_bf16(const std::uint8_t* weights,
                               std::size_t output_rows, std::size_t columns,
                               const __nv_bfloat16* prompt,
                               std::size_t prompt_rows, Q8Block* q8_workspace,
                               float* output, cudaStream_t stream) noexcept {
  if (weights == nullptr || prompt == nullptr || output == nullptr ||
      output_rows == 0 || columns == 0 || prompt_rows == 0 ||
      columns % kValuesPerWeightBlock != 0) {
    return cudaErrorInvalidValue;
  }
  if (prompt_rows >= 8) {
    return launch_q8_mmq_quality(weights, output_rows, columns, prompt,
                                 prompt_rows, q8_workspace, output, stream);
  }
  return launch_q8_mmq_bf16_variant(
      weights, output_rows, columns, prompt, prompt_rows, output,
      selected_mmq_prompt_tile(QuantKind::kQ8_0, prompt_rows), stream);
}

cudaError_t launch_q8_mmq_bf16_reference(
    const std::uint8_t* weights, std::size_t output_rows, std::size_t columns,
    const __nv_bfloat16* prompt, std::size_t prompt_rows, float* output,
    cudaStream_t stream) noexcept {
  if (weights == nullptr || prompt == nullptr || output == nullptr ||
      output_rows == 0 || columns == 0 || prompt_rows == 0 ||
      columns % kValuesPerWeightBlock != 0) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid(static_cast<unsigned int>((output_rows + 7) / 8),
                  static_cast<unsigned int>(prompt_rows));
  q8_mmq_bf16_reference<<<grid, kThreads, 0, stream>>>(
      weights, output_rows, columns, prompt, prompt_rows, output);
  return cudaPeekAtLastError();
}

cudaError_t launch_quant_row_decode(QuantKind kind,
                                    const std::uint8_t* weights,
                                    std::size_t rows, std::size_t columns,
                                    std::size_t row, __nv_bfloat16* output,
                                    cudaStream_t stream) noexcept {
  if (weights == nullptr || output == nullptr || rows == 0 || columns == 0 ||
      row >= rows || columns % kValuesPerWeightBlock != 0 ||
      (kind != QuantKind::kQ4K && kind != QuantKind::kQ6K &&
       kind != QuantKind::kQ8_0)) {
    return cudaErrorInvalidValue;
  }
  const unsigned int blocks =
      static_cast<unsigned int>((columns + kThreads - 1) / kThreads);
  if (kind == QuantKind::kQ4K) {
    quant_row_decode<QuantKind::kQ4K><<<blocks, kThreads, 0, stream>>>(
        weights, columns, row, output);
  } else if (kind == QuantKind::kQ6K) {
    quant_row_decode<QuantKind::kQ6K><<<blocks, kThreads, 0, stream>>>(
        weights, columns, row, output);
  } else {
    quant_row_decode<QuantKind::kQ8_0><<<blocks, kThreads, 0, stream>>>(
        weights, columns, row, output);
  }
  return cudaPeekAtLastError();
}

cudaError_t launch_quant_rows_decode_widen(QuantKind kind,
                                           const std::uint8_t* weights,
                                           std::size_t rows, std::size_t columns,
                                           const std::size_t* token_ids,
                                           std::size_t token_count,
                                           float* output,
                                           cudaStream_t stream) noexcept {
  if (weights == nullptr || token_ids == nullptr || output == nullptr ||
      rows == 0 || columns == 0 || token_count == 0 || token_count > rows ||
      columns % kValuesPerWeightBlock != 0 ||
      (kind != QuantKind::kQ4K && kind != QuantKind::kQ6K &&
       kind != QuantKind::kQ8_0)) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid(static_cast<unsigned int>((columns + kThreads - 1) / kThreads),
                  static_cast<unsigned int>(token_count));
  if (kind == QuantKind::kQ4K) {
    return quartz_launch_kernel(quant_rows_decode_widen<QuantKind::kQ4K>, grid,
                                dim3(kThreads), 0, stream, weights, columns,
                                token_ids, output);
  }
  if (kind == QuantKind::kQ6K) {
    return quartz_launch_kernel(quant_rows_decode_widen<QuantKind::kQ6K>, grid,
                                dim3(kThreads), 0, stream, weights, columns,
                                token_ids, output);
  }
  return quartz_launch_kernel(quant_rows_decode_widen<QuantKind::kQ8_0>, grid,
                              dim3(kThreads), 0, stream, weights, columns,
                              token_ids, output);
}

}  // namespace qw38::cuda
