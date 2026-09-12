#include "q6k_decode_dots.cuh"

#include "quant_mmv.h"

namespace qw38::cuda {
namespace {

constexpr int kWarpSize = 32;
constexpr std::size_t kValuesPerWeightBlock = 256;

}  // namespace

namespace {

template <bool UseQ81>
cudaError_t launch_coop_dispatch(const std::uint8_t* weights, std::size_t rows,
                                 std::size_t columns, const void* staged,
                                 float* output, unsigned int warps_per_row,
                                 cudaStream_t stream) noexcept {
  const bool aligned = q6_device_uses_aligned_soa();
  if (aligned) {
    if (warps_per_row == 1) {
      return q6k_dots::launch_coop_aligned<1, UseQ81>(
          weights, rows, columns, staged, output, stream);
    }
    if (warps_per_row == 2) {
      return q6k_dots::launch_coop_aligned<2, UseQ81>(
          weights, rows, columns, staged, output, stream);
    }
    if (warps_per_row == 4) {
      return q6k_dots::launch_coop_aligned<4, UseQ81>(
          weights, rows, columns, staged, output, stream);
    }
    return q6k_dots::launch_coop_aligned<8, UseQ81>(weights, rows, columns,
                                                    staged, output, stream);
  }
  if (warps_per_row == 1) {
    return q6k_dots::launch_coop<1, UseQ81>(weights, rows, columns, staged,
                                            output, stream);
  }
  if (warps_per_row == 2) {
    return q6k_dots::launch_coop<2, UseQ81>(weights, rows, columns, staged,
                                            output, stream);
  }
  if (warps_per_row == 4) {
    return q6k_dots::launch_coop<4, UseQ81>(weights, rows, columns, staged,
                                            output, stream);
  }
  return q6k_dots::launch_coop<8, UseQ81>(weights, rows, columns, staged, output,
                                          stream);
}

}  // namespace

cudaError_t launch_q6k_coop_mmv_prequant(
    const std::uint8_t* weights, std::size_t rows, std::size_t columns,
    const void* staged, float* output, unsigned int warps_per_row, bool q8_1,
    cudaStream_t stream) noexcept {
  if (weights == nullptr || staged == nullptr || output == nullptr ||
      rows == 0 || columns == 0 || columns % kValuesPerWeightBlock != 0 ||
      !legal_q6_decode_warps_per_row(warps_per_row)) {
    return cudaErrorInvalidValue;
  }
  if (q8_1) {
    return launch_coop_dispatch<true>(weights, rows, columns, staged, output,
                                      warps_per_row, stream);
  }
  return launch_coop_dispatch<false>(weights, rows, columns, staged, output,
                                     warps_per_row, stream);
}

cudaError_t launch_q6k_coop_mmv(
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
  return launch_q6k_coop_mmv_prequant(weights, rows, columns, workspace, output,
                                      warps_per_row, q8_1, stream);
}

int q6k_coop_occupancy(unsigned int warps_per_row, bool q8_1) noexcept {
  int occupancy = 0;
  cudaError_t error = cudaErrorInvalidValue;
  const int threads = static_cast<int>(warps_per_row) * kWarpSize;
  const bool aligned = q6_device_uses_aligned_soa();
  if (q8_1) {
    if (aligned) {
      if (warps_per_row == 1) {
        error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occupancy, q6k_dots::q6k_coop_mmv_aligned<1, true>, threads, 0);
      } else if (warps_per_row == 2) {
        error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occupancy, q6k_dots::q6k_coop_mmv_aligned<2, true>, threads, 0);
      } else if (warps_per_row == 4) {
        error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occupancy, q6k_dots::q6k_coop_mmv_aligned<4, true>, threads, 0);
      } else if (warps_per_row == 8) {
        error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occupancy, q6k_dots::q6k_coop_mmv_aligned<8, true>, threads, 0);
      }
    } else if (warps_per_row == 1) {
      error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, q6k_dots::q6k_coop_mmv<1, true>, threads, 0);
    } else if (warps_per_row == 2) {
      error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, q6k_dots::q6k_coop_mmv<2, true>, threads, 0);
    } else if (warps_per_row == 4) {
      error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, q6k_dots::q6k_coop_mmv<4, true>, threads, 0);
    } else if (warps_per_row == 8) {
      error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, q6k_dots::q6k_coop_mmv<8, true>, threads, 0);
    }
  } else if (aligned) {
    if (warps_per_row == 1) {
      error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, q6k_dots::q6k_coop_mmv_aligned<1, false>, threads, 0);
    } else if (warps_per_row == 2) {
      error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, q6k_dots::q6k_coop_mmv_aligned<2, false>, threads, 0);
    } else if (warps_per_row == 4) {
      error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, q6k_dots::q6k_coop_mmv_aligned<4, false>, threads, 0);
    } else if (warps_per_row == 8) {
      error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, q6k_dots::q6k_coop_mmv_aligned<8, false>, threads, 0);
    }
  } else if (warps_per_row == 1) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, q6k_dots::q6k_coop_mmv<1, false>, threads, 0);
  } else if (warps_per_row == 2) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, q6k_dots::q6k_coop_mmv<2, false>, threads, 0);
  } else if (warps_per_row == 4) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, q6k_dots::q6k_coop_mmv<4, false>, threads, 0);
  } else if (warps_per_row == 8) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, q6k_dots::q6k_coop_mmv<8, false>, threads, 0);
  }
  if (error != cudaSuccess) return 0;
  return occupancy;
}

void q6k_coop_kernel_attributes(unsigned int warps_per_row, bool q8_1,
                                int* registers, std::size_t* local_bytes,
                                int* occupancy) noexcept {
  cudaFuncAttributes attrs{};
  cudaError_t error = cudaErrorInvalidValue;
  const bool aligned = q6_device_uses_aligned_soa();
  if (q8_1) {
    if (aligned) {
      if (warps_per_row == 1) {
        error = cudaFuncGetAttributes(&attrs,
                                      q6k_dots::q6k_coop_mmv_aligned<1, true>);
      } else if (warps_per_row == 2) {
        error = cudaFuncGetAttributes(&attrs,
                                      q6k_dots::q6k_coop_mmv_aligned<2, true>);
      } else if (warps_per_row == 4) {
        error = cudaFuncGetAttributes(&attrs,
                                      q6k_dots::q6k_coop_mmv_aligned<4, true>);
      } else if (warps_per_row == 8) {
        error = cudaFuncGetAttributes(&attrs,
                                      q6k_dots::q6k_coop_mmv_aligned<8, true>);
      }
    } else if (warps_per_row == 1) {
      error = cudaFuncGetAttributes(&attrs, q6k_dots::q6k_coop_mmv<1, true>);
    } else if (warps_per_row == 2) {
      error = cudaFuncGetAttributes(&attrs, q6k_dots::q6k_coop_mmv<2, true>);
    } else if (warps_per_row == 4) {
      error = cudaFuncGetAttributes(&attrs, q6k_dots::q6k_coop_mmv<4, true>);
    } else if (warps_per_row == 8) {
      error = cudaFuncGetAttributes(&attrs, q6k_dots::q6k_coop_mmv<8, true>);
    }
  } else if (aligned) {
    if (warps_per_row == 1) {
      error = cudaFuncGetAttributes(&attrs,
                                    q6k_dots::q6k_coop_mmv_aligned<1, false>);
    } else if (warps_per_row == 2) {
      error = cudaFuncGetAttributes(&attrs,
                                    q6k_dots::q6k_coop_mmv_aligned<2, false>);
    } else if (warps_per_row == 4) {
      error = cudaFuncGetAttributes(&attrs,
                                    q6k_dots::q6k_coop_mmv_aligned<4, false>);
    } else if (warps_per_row == 8) {
      error = cudaFuncGetAttributes(&attrs,
                                    q6k_dots::q6k_coop_mmv_aligned<8, false>);
    }
  } else if (warps_per_row == 1) {
    error = cudaFuncGetAttributes(&attrs, q6k_dots::q6k_coop_mmv<1, false>);
  } else if (warps_per_row == 2) {
    error = cudaFuncGetAttributes(&attrs, q6k_dots::q6k_coop_mmv<2, false>);
  } else if (warps_per_row == 4) {
    error = cudaFuncGetAttributes(&attrs, q6k_dots::q6k_coop_mmv<4, false>);
  } else if (warps_per_row == 8) {
    error = cudaFuncGetAttributes(&attrs, q6k_dots::q6k_coop_mmv<8, false>);
  }
  if (registers != nullptr) {
    *registers = error == cudaSuccess ? attrs.numRegs : 0;
  }
  if (local_bytes != nullptr) {
    *local_bytes = error == cudaSuccess
                       ? static_cast<std::size_t>(attrs.localSizeBytes)
                       : 0;
  }
  if (occupancy != nullptr) *occupancy = q6k_coop_occupancy(warps_per_row, q8_1);
}

}  // namespace qw38::cuda
