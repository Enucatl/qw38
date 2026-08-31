#include "quant_mmv.h"

#include <cuda_fp16.h>

namespace qw38::cuda {
namespace {

constexpr int kWarpSize = 32;
constexpr int kThreads = 256;
constexpr int kPromptRowsPerTile = 4;
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

template <QuantKind Kind>
__global__ void quant_mmv(const std::uint8_t* weights, std::size_t rows,
                          std::size_t columns, const Q8Block* activation,
                          float* output) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * (kThreads / kWarpSize) + warp;
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
  for (int offset = 16; offset > 0; offset /= 2) {
    sum = __fadd_rn(
        sum, __shfl_down_sync(0xFFFFFFFFU, sum, offset, kWarpSize));
  }
  if (lane == 0) output[row] = sum;
}

template <QuantKind Kind>
__global__ void quant_mmq(const std::uint8_t* weights, std::size_t output_rows,
                          std::size_t columns, const Q8Block* prompt,
                          std::size_t prompt_rows, float* output) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const std::size_t output_row =
      static_cast<std::size_t>(blockIdx.x) * (kThreads / kWarpSize) + warp;
  const std::size_t prompt_start =
      static_cast<std::size_t>(blockIdx.y) * kPromptRowsPerTile;
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
  float sums[kPromptRowsPerTile] = {0.0F, 0.0F, 0.0F, 0.0F};
  for (std::size_t column = lane; column < columns; column += kWarpSize) {
    const std::size_t weight_block = column / kWeightValues;
    const int within = static_cast<int>(column % kWeightValues);
    const std::uint8_t* block = row_weights + weight_block * kWeightBytes;
    const float weight = decode_weight<Kind>(block, within);
#pragma unroll
    for (int prompt_offset = 0; prompt_offset < kPromptRowsPerTile;
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
  for (int prompt_offset = 0; prompt_offset < kPromptRowsPerTile;
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

}  // namespace

std::size_t q8_workspace_bytes(std::size_t columns) noexcept {
  return columns % kValuesPerWeightBlock == 0
             ? (columns / kWarpSize) * sizeof(Q8Block)
             : 0;
}

std::size_t q8_prompt_workspace_bytes(std::size_t prompt_rows,
                                      std::size_t columns) noexcept {
  if (prompt_rows == 0 || columns % kValuesPerWeightBlock != 0) return 0;
  return prompt_rows * (columns / kWarpSize) * sizeof(Q8Block);
}

cudaError_t launch_quant_mmv(QuantKind kind, const std::uint8_t* weights,
                             std::size_t rows, std::size_t columns,
                             const __nv_bfloat16* activation,
                             Q8Block* q8_workspace, float* output,
                             cudaStream_t stream) noexcept {
  if (weights == nullptr || activation == nullptr || q8_workspace == nullptr ||
      output == nullptr || rows == 0 || columns == 0 ||
      columns % kValuesPerWeightBlock != 0 ||
      (kind != QuantKind::kQ4K && kind != QuantKind::kQ6K &&
       kind != QuantKind::kQ8_0)) {
    return cudaErrorInvalidValue;
  }
  const unsigned int quant_blocks =
      static_cast<unsigned int>((columns + kThreads - 1) / kThreads);
  quantize_bf16_q8<<<quant_blocks, kThreads, 0, stream>>>(
      activation, q8_workspace, columns);
  cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  const unsigned int row_blocks = static_cast<unsigned int>(
      (rows + (kThreads / kWarpSize) - 1) / (kThreads / kWarpSize));
  if (kind == QuantKind::kQ4K) {
    quant_mmv<QuantKind::kQ4K><<<row_blocks, kThreads, 0, stream>>>(
        weights, rows, columns, q8_workspace, output);
  } else if (kind == QuantKind::kQ6K) {
    quant_mmv<QuantKind::kQ6K><<<row_blocks, kThreads, 0, stream>>>(
        weights, rows, columns, q8_workspace, output);
  } else {
    quant_mmv<QuantKind::kQ8_0><<<row_blocks, kThreads, 0, stream>>>(
        weights, rows, columns, q8_workspace, output);
  }
  return cudaPeekAtLastError();
}

cudaError_t launch_quant_mmq(QuantKind kind, const std::uint8_t* weights,
                             std::size_t output_rows, std::size_t columns,
                             const __nv_bfloat16* prompt,
                             std::size_t prompt_rows, Q8Block* q8_workspace,
                             float* output, cudaStream_t stream) noexcept {
  if (weights == nullptr || prompt == nullptr || q8_workspace == nullptr ||
      output == nullptr || output_rows == 0 || columns == 0 ||
      prompt_rows == 0 || columns % kValuesPerWeightBlock != 0 ||
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
  const dim3 grid(
      static_cast<unsigned int>(
          (output_rows + (kThreads / kWarpSize) - 1) /
          (kThreads / kWarpSize)),
      static_cast<unsigned int>((prompt_rows + kPromptRowsPerTile - 1) /
                                kPromptRowsPerTile));
  if (kind == QuantKind::kQ4K) {
    quant_mmq<QuantKind::kQ4K><<<grid, kThreads, 0, stream>>>(
        weights, output_rows, columns, q8_workspace, prompt_rows, output);
  } else if (kind == QuantKind::kQ6K) {
    quant_mmq<QuantKind::kQ6K><<<grid, kThreads, 0, stream>>>(
        weights, output_rows, columns, q8_workspace, prompt_rows, output);
  } else {
    quant_mmq<QuantKind::kQ8_0><<<grid, kThreads, 0, stream>>>(
        weights, output_rows, columns, q8_workspace, prompt_rows, output);
  }
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

}  // namespace qw38::cuda
