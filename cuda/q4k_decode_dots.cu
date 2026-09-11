#include "q4k_decode_dots.cuh"

#include "quant_mmv.h"

namespace qw38::cuda {
namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr std::size_t kValuesPerWeightBlock = 256;

}  // namespace

cudaError_t launch_quantize_bf16_q8_1(const __nv_bfloat16* activation,
                                      void* q8, std::size_t columns,
                                      cudaStream_t stream) noexcept {
  if (activation == nullptr || q8 == nullptr || columns == 0 ||
      columns % kWarpSize != 0) {
    return cudaErrorInvalidValue;
  }
  const unsigned int blocks =
      static_cast<unsigned int>((columns + kThreads - 1) / kThreads);
  q4k_dots::quantize_bf16_q8_1<<<blocks, kThreads, 0, stream>>>(
      activation, static_cast<Q8_1Block*>(q8), columns);
  return cudaPeekAtLastError();
}

cudaError_t launch_quantize_bf16_q8_1_sum_x(const __nv_bfloat16* activation,
                                            void* q8, std::size_t columns,
                                            cudaStream_t stream) noexcept {
  if (activation == nullptr || q8 == nullptr || columns == 0 ||
      columns % kWarpSize != 0) {
    return cudaErrorInvalidValue;
  }
  const unsigned int blocks =
      static_cast<unsigned int>((columns + kThreads - 1) / kThreads);
  q4k_dots::quantize_bf16_q8_1_sum_x<<<blocks, kThreads, 0, stream>>>(
      activation, static_cast<Q8_1Block*>(q8), columns);
  return cudaPeekAtLastError();
}

cudaError_t launch_q4k_coop_mmv_prequant(
    const std::uint8_t* weights, std::size_t rows, std::size_t columns,
    const void* staged, float* output, unsigned int warps_per_row, bool q8_1,
    cudaStream_t stream) noexcept {
  if (weights == nullptr || staged == nullptr || output == nullptr ||
      rows == 0 || columns == 0 || columns % kValuesPerWeightBlock != 0 ||
      !legal_q4_decode_warps_per_row(warps_per_row)) {
    return cudaErrorInvalidValue;
  }
  record_q4_launch_variant(q8_1 ? kQ4LaunchVariantCoopQ81Prequant
                                : kQ4LaunchVariantCoopQ8Prequant);
  if (q8_1) {
    if (warps_per_row == 1) {
      return q4k_dots::launch_coop<1, true>(weights, rows, columns, staged,
                                            output, stream);
    }
    if (warps_per_row == 2) {
      return q4k_dots::launch_coop<2, true>(weights, rows, columns, staged,
                                            output, stream);
    }
    if (warps_per_row == 4) {
      return q4k_dots::launch_coop<4, true>(weights, rows, columns, staged,
                                            output, stream);
    }
    return q4k_dots::launch_coop<8, true>(weights, rows, columns, staged,
                                          output, stream);
  }
  if (warps_per_row == 1) {
    return q4k_dots::launch_coop<1, false>(weights, rows, columns, staged,
                                           output, stream);
  }
  if (warps_per_row == 2) {
    return q4k_dots::launch_coop<2, false>(weights, rows, columns, staged,
                                           output, stream);
  }
  if (warps_per_row == 4) {
    return q4k_dots::launch_coop<4, false>(weights, rows, columns, staged,
                                           output, stream);
  }
  return q4k_dots::launch_coop<8, false>(weights, rows, columns, staged, output,
                                         stream);
}

cudaError_t launch_q4k_coop_mmv(
    const std::uint8_t* weights, std::size_t rows, std::size_t columns,
    const __nv_bfloat16* activation, void* workspace, float* output,
    unsigned int warps_per_row, bool q8_1, cudaStream_t stream) noexcept {
  if (activation == nullptr || workspace == nullptr) {
    return cudaErrorInvalidValue;
  }
  cudaError_t error = cudaSuccess;
  if (q8_1) {
    error = launch_quantize_bf16_q8_1(
        activation, static_cast<Q8_1Block*>(workspace), columns, stream);
  } else {
    error = launch_quantize_bf16_q8(
        activation, static_cast<Q8Block*>(workspace), columns, stream);
  }
  if (error != cudaSuccess) return error;
  error = launch_q4k_coop_mmv_prequant(weights, rows, columns, workspace, output,
                                       warps_per_row, q8_1, stream);
  if (error == cudaSuccess) {
    record_q4_launch_variant(q8_1 ? kQ4LaunchVariantCoopQ81
                                  : kQ4LaunchVariantCoopQ8);
  }
  return error;
}

cudaError_t launch_q4k_coop_mmv_prequant_q8(
    const std::uint8_t* weights, std::size_t rows, std::size_t columns,
    const Q8Block* q8, float* output, unsigned int warps_per_row,
    cudaStream_t stream) noexcept {
  return launch_q4k_coop_mmv_prequant(weights, rows, columns, q8, output,
                                      warps_per_row, false, stream);
}

int q4k_coop_occupancy(unsigned int warps_per_row, bool q8_1) noexcept {
  int occupancy = 0;
  cudaError_t error = cudaErrorInvalidValue;
  const int threads = static_cast<int>(warps_per_row) * kWarpSize;
  if (q8_1) {
    if (warps_per_row == 1) {
      error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, q4k_dots::q4k_coop_mmv<1, true>, threads, 0);
    } else if (warps_per_row == 2) {
      error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, q4k_dots::q4k_coop_mmv<2, true>, threads, 0);
    } else if (warps_per_row == 4) {
      error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, q4k_dots::q4k_coop_mmv<4, true>, threads, 0);
    } else if (warps_per_row == 8) {
      error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, q4k_dots::q4k_coop_mmv<8, true>, threads, 0);
    }
  } else if (warps_per_row == 1) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, q4k_dots::q4k_coop_mmv<1, false>, threads, 0);
  } else if (warps_per_row == 2) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, q4k_dots::q4k_coop_mmv<2, false>, threads, 0);
  } else if (warps_per_row == 4) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, q4k_dots::q4k_coop_mmv<4, false>, threads, 0);
  } else if (warps_per_row == 8) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, q4k_dots::q4k_coop_mmv<8, false>, threads, 0);
  }
  if (error != cudaSuccess) return 0;
  return occupancy;
}

void q4k_coop_kernel_attributes(unsigned int warps_per_row, bool q8_1,
                                int* registers, std::size_t* local_bytes,
                                int* occupancy) noexcept {
  cudaFuncAttributes attrs{};
  cudaError_t error = cudaErrorInvalidValue;
  if (q8_1) {
    if (warps_per_row == 1) {
      error = cudaFuncGetAttributes(&attrs, q4k_dots::q4k_coop_mmv<1, true>);
    } else if (warps_per_row == 2) {
      error = cudaFuncGetAttributes(&attrs, q4k_dots::q4k_coop_mmv<2, true>);
    } else if (warps_per_row == 4) {
      error = cudaFuncGetAttributes(&attrs, q4k_dots::q4k_coop_mmv<4, true>);
    } else if (warps_per_row == 8) {
      error = cudaFuncGetAttributes(&attrs, q4k_dots::q4k_coop_mmv<8, true>);
    }
  } else if (warps_per_row == 1) {
    error = cudaFuncGetAttributes(&attrs, q4k_dots::q4k_coop_mmv<1, false>);
  } else if (warps_per_row == 2) {
    error = cudaFuncGetAttributes(&attrs, q4k_dots::q4k_coop_mmv<2, false>);
  } else if (warps_per_row == 4) {
    error = cudaFuncGetAttributes(&attrs, q4k_dots::q4k_coop_mmv<4, false>);
  } else if (warps_per_row == 8) {
    error = cudaFuncGetAttributes(&attrs, q4k_dots::q4k_coop_mmv<8, false>);
  }
  if (registers != nullptr) {
    *registers = error == cudaSuccess ? attrs.numRegs : 0;
  }
  if (local_bytes != nullptr) {
    *local_bytes = error == cudaSuccess
                       ? static_cast<std::size_t>(attrs.localSizeBytes)
                       : 0;
  }
  if (occupancy != nullptr) *occupancy = q4k_coop_occupancy(warps_per_row, q8_1);
}

}  // namespace qw38::cuda
