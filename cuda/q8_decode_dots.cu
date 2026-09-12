#include "q8_decode_dots.cuh"

#include "q4k_decode_dots.cuh"

namespace qw38::cuda {
namespace {

constexpr int kBf16Threads = 256;
constexpr int kWarpSize = 32;
constexpr std::size_t kQ80Values = 32;

cudaError_t launch_layout(const std::uint8_t* weights, std::size_t rows,
                          std::size_t columns, const void* staged,
                          float* output, unsigned int rows_per_cta,
                          unsigned int warps_per_row, cudaStream_t stream) {
  if (rows_per_cta == 1 && warps_per_row == 1) {
    return q8_dots::launch_coop<1, 1>(weights, rows, columns, staged, output,
                                      stream);
  }
  if (rows_per_cta == 1 && warps_per_row == 2) {
    return q8_dots::launch_coop<1, 2>(weights, rows, columns, staged, output,
                                      stream);
  }
  if (rows_per_cta == 1 && warps_per_row == 4) {
    return q8_dots::launch_coop<1, 4>(weights, rows, columns, staged, output,
                                      stream);
  }
  if (rows_per_cta == 1 && warps_per_row == 8) {
    return q8_dots::launch_coop<1, 8>(weights, rows, columns, staged, output,
                                      stream);
  }
  if (rows_per_cta == 2 && warps_per_row == 2) {
    return q8_dots::launch_coop<2, 2>(weights, rows, columns, staged, output,
                                      stream);
  }
  if (rows_per_cta == 4 && warps_per_row == 1) {
    return q8_dots::launch_coop<4, 1>(weights, rows, columns, staged, output,
                                      stream);
  }
  if (rows_per_cta == 8 && warps_per_row == 1) {
    return q8_dots::launch_coop<8, 1>(weights, rows, columns, staged, output,
                                      stream);
  }
  return cudaErrorInvalidValue;
}

}  // namespace

cudaError_t launch_q8_mmv_bf16(const std::uint8_t* weights, std::size_t rows,
                               std::size_t columns,
                               const __nv_bfloat16* activation, float* output,
                               cudaStream_t stream) noexcept {
  if (weights == nullptr || activation == nullptr || output == nullptr ||
      rows == 0 || columns == 0 || columns % kQ80Values != 0) {
    return cudaErrorInvalidValue;
  }
  const unsigned int blocks = static_cast<unsigned int>(
      (rows + (kBf16Threads / kWarpSize) - 1) / (kBf16Threads / kWarpSize));
  q8_dots::q8_mmv_bf16_ref<<<blocks, kBf16Threads, 0, stream>>>(
      weights, rows, columns, activation, output);
  return cudaPeekAtLastError();
}

cudaError_t launch_q8_coop_mmv_prequant(
    const std::uint8_t* weights, std::size_t rows, std::size_t columns,
    const void* staged, float* output, unsigned int rows_per_cta,
    unsigned int warps_per_row, cudaStream_t stream) noexcept {
  if (weights == nullptr || staged == nullptr || output == nullptr ||
      rows == 0 || columns == 0 || columns % kQ80Values != 0 ||
      !legal_q8_decode_layout(rows_per_cta, warps_per_row)) {
    return cudaErrorInvalidValue;
  }
  record_q8_decode_dispatch(rows, columns, rows_per_cta, warps_per_row);
  return launch_layout(weights, rows, columns, staged, output, rows_per_cta,
                       warps_per_row, stream);
}

cudaError_t launch_q8_coop_mmv_prequant(
    const std::uint8_t* weights, std::size_t rows, std::size_t columns,
    const void* staged, float* output, unsigned int warps_per_row,
    cudaStream_t stream) noexcept {
  return launch_q8_coop_mmv_prequant(weights, rows, columns, staged, output, 1U,
                                     warps_per_row, stream);
}

cudaError_t launch_q8_coop_mmv(const std::uint8_t* weights, std::size_t rows,
                               std::size_t columns,
                               const __nv_bfloat16* activation, void* workspace,
                               float* output, unsigned int rows_per_cta,
                               unsigned int warps_per_row,
                               cudaStream_t stream) noexcept {
  if (activation == nullptr || workspace == nullptr) {
    return cudaErrorInvalidValue;
  }
  const cudaError_t error = launch_quantize_bf16_q8_1(
      activation, static_cast<Q8_1Block*>(workspace), columns, stream);
  if (error != cudaSuccess) return error;
  return launch_q8_coop_mmv_prequant(weights, rows, columns, workspace, output,
                                     rows_per_cta, warps_per_row, stream);
}

cudaError_t launch_q8_coop_mmv(const std::uint8_t* weights, std::size_t rows,
                               std::size_t columns,
                               const __nv_bfloat16* activation, void* workspace,
                               float* output, unsigned int warps_per_row,
                               cudaStream_t stream) noexcept {
  return launch_q8_coop_mmv(weights, rows, columns, activation, workspace,
                            output, 1U, warps_per_row, stream);
}

cudaError_t upload_q8_grouped_descriptors(Q8GroupedProjDesc* device,
                                          const Q8GroupedProjDesc host[4],
                                          cudaStream_t stream) noexcept {
  if (device == nullptr || host == nullptr) return cudaErrorInvalidValue;
  return cudaMemcpyAsync(device, host, sizeof(Q8GroupedProjDesc) * 4,
                         cudaMemcpyHostToDevice, stream);
}

cudaError_t launch_q8_coop_mmv_grouped_r1_w4(
    const Q8GroupedProjDesc* descriptors, const Q8GroupedProjDesc host[4],
    std::size_t columns, const void* staged, cudaStream_t stream) noexcept {
  if (descriptors == nullptr || host == nullptr || staged == nullptr ||
      columns == 0 || columns % kQ80Values != 0) {
    return cudaErrorInvalidValue;
  }
  unsigned int grid = 0;
  for (int index = 0; index < kQ8GroupedDescCount; ++index) {
    grid += static_cast<unsigned int>(host[index].rows);
  }
  record_q8_grouped_dispatch(static_cast<std::size_t>(grid), columns);
  if (grid == 0) return cudaSuccess;
  dim3 block(kWarpSize, 4);
  q8_dots::q8_coop_mmv_grouped_r1_w4<<<grid, block, 0, stream>>>(
      descriptors, columns, static_cast<const Q8_1Block*>(staged));
  return cudaPeekAtLastError();
}

int q8_coop_occupancy(unsigned int rows_per_cta,
                      unsigned int warps_per_row) noexcept {
  int occupancy = 0;
  cudaError_t error = cudaErrorInvalidValue;
  const int threads =
      static_cast<int>(rows_per_cta * warps_per_row) * kWarpSize;
  if (rows_per_cta == 1 && warps_per_row == 1) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, q8_dots::q8_coop_mmv<1, 1>, threads, 0);
  } else if (rows_per_cta == 1 && warps_per_row == 2) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, q8_dots::q8_coop_mmv<1, 2>, threads, 0);
  } else if (rows_per_cta == 1 && warps_per_row == 4) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, q8_dots::q8_coop_mmv<1, 4>, threads, 0);
  } else if (rows_per_cta == 1 && warps_per_row == 8) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, q8_dots::q8_coop_mmv<1, 8>, threads, 0);
  } else if (rows_per_cta == 2 && warps_per_row == 2) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, q8_dots::q8_coop_mmv<2, 2>, threads, 0);
  } else if (rows_per_cta == 4 && warps_per_row == 1) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, q8_dots::q8_coop_mmv<4, 1>, threads, 0);
  } else if (rows_per_cta == 8 && warps_per_row == 1) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, q8_dots::q8_coop_mmv<8, 1>, threads, 0);
  }
  if (error != cudaSuccess) return 0;
  return occupancy;
}

int q8_coop_occupancy(unsigned int warps_per_row) noexcept {
  return q8_coop_occupancy(1U, warps_per_row);
}

void q8_coop_kernel_attributes(unsigned int rows_per_cta,
                               unsigned int warps_per_row, int* registers,
                               std::size_t* local_bytes,
                               int* occupancy) noexcept {
  cudaFuncAttributes attrs{};
  cudaError_t error = cudaErrorInvalidValue;
  if (rows_per_cta == 1 && warps_per_row == 1) {
    error = cudaFuncGetAttributes(&attrs, q8_dots::q8_coop_mmv<1, 1>);
  } else if (rows_per_cta == 1 && warps_per_row == 2) {
    error = cudaFuncGetAttributes(&attrs, q8_dots::q8_coop_mmv<1, 2>);
  } else if (rows_per_cta == 1 && warps_per_row == 4) {
    error = cudaFuncGetAttributes(&attrs, q8_dots::q8_coop_mmv<1, 4>);
  } else if (rows_per_cta == 1 && warps_per_row == 8) {
    error = cudaFuncGetAttributes(&attrs, q8_dots::q8_coop_mmv<1, 8>);
  } else if (rows_per_cta == 2 && warps_per_row == 2) {
    error = cudaFuncGetAttributes(&attrs, q8_dots::q8_coop_mmv<2, 2>);
  } else if (rows_per_cta == 4 && warps_per_row == 1) {
    error = cudaFuncGetAttributes(&attrs, q8_dots::q8_coop_mmv<4, 1>);
  } else if (rows_per_cta == 8 && warps_per_row == 1) {
    error = cudaFuncGetAttributes(&attrs, q8_dots::q8_coop_mmv<8, 1>);
  }
  if (registers != nullptr) {
    *registers = error == cudaSuccess ? attrs.numRegs : 0;
  }
  if (local_bytes != nullptr) {
    *local_bytes = error == cudaSuccess
                       ? static_cast<std::size_t>(attrs.localSizeBytes)
                       : 0;
  }
  if (occupancy != nullptr) {
    *occupancy = q8_coop_occupancy(rows_per_cta, warps_per_row);
  }
}

void q8_coop_kernel_attributes(unsigned int warps_per_row, int* registers,
                               std::size_t* local_bytes,
                               int* occupancy) noexcept {
  q8_coop_kernel_attributes(1U, warps_per_row, registers, local_bytes,
                            occupancy);
}

int q8_mmv_bf16_occupancy() noexcept {
  int occupancy = 0;
  const cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, q8_dots::q8_mmv_bf16_ref, kBf16Threads, 0);
  return error == cudaSuccess ? occupancy : 0;
}

void q8_mmv_bf16_kernel_attributes(int* registers, std::size_t* local_bytes,
                                   int* occupancy) noexcept {
  cudaFuncAttributes attrs{};
  const cudaError_t error =
      cudaFuncGetAttributes(&attrs, q8_dots::q8_mmv_bf16_ref);
  if (registers != nullptr) {
    *registers = error == cudaSuccess ? attrs.numRegs : 0;
  }
  if (local_bytes != nullptr) {
    *local_bytes = error == cudaSuccess
                       ? static_cast<std::size_t>(attrs.localSizeBytes)
                       : 0;
  }
  if (occupancy != nullptr) *occupancy = q8_mmv_bf16_occupancy();
}

}  // namespace qw38::cuda
