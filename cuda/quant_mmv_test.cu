#include "quant_mmv.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "quant.h"

namespace {

int fail_cuda(const char* operation, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
  return 1;
}

void write_u16(std::uint8_t* bytes, std::uint16_t value) {
  bytes[0] = static_cast<std::uint8_t>(value & 0xFFU);
  bytes[1] = static_cast<std::uint8_t>(value >> 8U);
}

void fill_weights(qw38::cuda::QuantKind kind, std::size_t rows,
                  std::size_t columns, std::vector<std::uint8_t>* weights) {
  const std::size_t block_bytes =
      kind == qw38::cuda::QuantKind::kQ4K ? 144
      : kind == qw38::cuda::QuantKind::kQ6K ? 210
                                           : 34;
  const std::size_t block_values =
      kind == qw38::cuda::QuantKind::kQ8_0 ? 32 : 256;
  weights->resize(rows * (columns / block_values) * block_bytes);
  for (std::size_t index = 0; index < weights->size(); ++index) {
    (*weights)[index] = static_cast<std::uint8_t>((index * 73 + 19) & 0xFFU);
  }
  for (std::size_t offset = 0; offset < weights->size();
       offset += block_bytes) {
    if (kind == qw38::cuda::QuantKind::kQ4K) {
      write_u16(weights->data() + offset, 0x2400U);      // 2^-6
      write_u16(weights->data() + offset + 2, 0x1C00U);  // 2^-8
    } else if (kind == qw38::cuda::QuantKind::kQ6K) {
      write_u16(weights->data() + offset + 208, 0x1C00U);  // 2^-8
    } else {
      write_u16(weights->data() + offset, 0x3400U);  // 2^-3
      for (std::size_t lane = 0; lane < 32; ++lane) {
        (*weights)[offset + 2 + lane] = static_cast<std::uint8_t>(
            static_cast<std::int8_t>(static_cast<int>((lane * 11 + offset) % 63) - 31));
      }
    }
  }
}

void make_activation(std::size_t columns,
                     std::vector<__nv_bfloat16>* activation,
                     std::vector<qw38::cuda::Q8Block>* staged,
                     std::size_t prompt_row = 0) {
  activation->resize(columns);
  staged->resize(columns / 32);
  for (std::size_t index = 0; index < columns; ++index) {
    const float phase = static_cast<float>(index) * 0.071F +
                        static_cast<float>(prompt_row) * 0.173F;
    const float value = std::sin(phase) * 3.0F +
                        static_cast<float>(static_cast<int>(index % 11) - 5) *
                            0.03125F +
                        static_cast<float>(static_cast<int>(prompt_row % 5) - 2) *
                            0.015625F;
    (*activation)[index] = __float2bfloat16_rn(value);
  }
  for (std::size_t block = 0; block < staged->size(); ++block) {
    float maximum = 0.0F;
    for (std::size_t lane = 0; lane < 32; ++lane) {
      maximum = std::max(
          maximum,
          std::fabs(__bfloat162float((*activation)[block * 32 + lane])));
    }
    const float scale = maximum == 0.0F ? 0.0F : maximum / 127.0F;
    (*staged)[block].scale = scale;
    for (std::size_t lane = 0; lane < 32; ++lane) {
      const float value =
          __bfloat162float((*activation)[block * 32 + lane]);
      (*staged)[block].values[lane] =
          scale == 0.0F
              ? 0
              : static_cast<std::int8_t>(std::round(value / scale));
    }
  }
}

bool reference(qw38::cuda::QuantKind kind,
               const std::vector<std::uint8_t>& weights, std::size_t rows,
               std::size_t columns,
               const std::vector<qw38::cuda::Q8Block>& activation,
               std::vector<float>* output) {
  const std::size_t block_bytes =
      kind == qw38::cuda::QuantKind::kQ4K ? 144
      : kind == qw38::cuda::QuantKind::kQ6K ? 210
                                           : 34;
  const std::size_t block_values =
      kind == qw38::cuda::QuantKind::kQ8_0 ? 32 : 256;
  output->assign(rows, 0.0F);
  std::vector<float> decoded(block_values);
  for (std::size_t row = 0; row < rows; ++row) {
    float sum = 0.0F;
    for (std::size_t block = 0; block < columns / block_values; ++block) {
      const std::uint8_t* packed =
          weights.data() +
          (row * (columns / block_values) + block) * block_bytes;
      const qw38::Status status =
          kind == qw38::cuda::QuantKind::kQ4K
              ? qw38::internal::decode_q4_k(packed, block_bytes, decoded.data(),
                                            decoded.size())
          : kind == qw38::cuda::QuantKind::kQ6K
              ? qw38::internal::decode_q6_k(packed, block_bytes, decoded.data(),
                                            decoded.size())
              : qw38::internal::decode_q8_0(packed, block_bytes, decoded.data(),
                                            decoded.size());
      if (!status.is_ok()) return false;
      for (std::size_t within = 0; within < block_values; ++within) {
        const std::size_t column = block * block_values + within;
        const auto& q8 = activation[column / 32];
        sum += decoded[within] *
               (q8.scale * static_cast<float>(q8.values[column % 32]));
      }
    }
    (*output)[row] = sum;
  }
  return true;
}

int run_case(qw38::cuda::QuantKind kind, const char* name, std::size_t rows,
             std::size_t columns) {
  std::vector<std::uint8_t> weights;
  std::vector<__nv_bfloat16> activation;
  std::vector<qw38::cuda::Q8Block> staged;
  std::vector<float> expected;
  fill_weights(kind, rows, columns, &weights);
  make_activation(columns, &activation, &staged);
  if (!reference(kind, weights, rows, columns, staged, &expected)) return 1;

  std::uint8_t* device_weights = nullptr;
  __nv_bfloat16* device_activation = nullptr;
  qw38::cuda::Q8Block* device_staged = nullptr;
  float* device_output = nullptr;
  cudaError_t error = cudaMalloc(&device_weights, weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_activation,
                       activation.size() * sizeof(activation[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_staged,
                       staged.size() * sizeof(qw38::cuda::Q8Block));
  }
  if (error == cudaSuccess) error = cudaMalloc(&device_output, rows * 4);
  if (error != cudaSuccess) return fail_cuda("cudaMalloc", error);
  error = cudaMemcpy(device_weights, weights.data(), weights.size(),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_activation, activation.data(),
                       activation.size() * sizeof(activation[0]),
                       cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("cudaMemcpy H2D", error);

  for (int warmup = 0; warmup < 3 && error == cudaSuccess; ++warmup) {
    error = qw38::cuda::launch_quant_mmv(kind, device_weights, rows, columns,
                                         device_activation, device_staged,
                                         device_output, nullptr);
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (error == cudaSuccess) error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error == cudaSuccess) error = cudaEventRecord(start);
  for (int sample = 0; sample < 30 && error == cudaSuccess; ++sample) {
    error = qw38::cuda::launch_quant_mmv(kind, device_weights, rows, columns,
                                         device_activation, device_staged,
                                         device_output, nullptr);
  }
  if (error == cudaSuccess) error = cudaEventRecord(stop);
  if (error == cudaSuccess) error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) return fail_cuda("CUDA MMV execution", error);
  float milliseconds = 0.0F;
  error = cudaEventElapsedTime(&milliseconds, start, stop);
  if (error != cudaSuccess) return fail_cuda("cudaEventElapsedTime", error);

  std::vector<float> actual(rows);
  std::vector<qw38::cuda::Q8Block> actual_staged(staged.size());
  error = cudaMemcpy(actual.data(), device_output, rows * sizeof(float),
                     cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(actual_staged.data(), device_staged,
                       staged.size() * sizeof(staged[0]),
                       cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("cudaMemcpy D2H", error);

  float maximum_absolute = 0.0F;
  float maximum_relative = 0.0F;
  double squared = 0.0;
  std::size_t nonfinite = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    if (!std::isfinite(actual[row])) ++nonfinite;
    const float absolute = std::fabs(actual[row] - expected[row]);
    maximum_absolute = std::max(maximum_absolute, absolute);
    maximum_relative = std::max(
        maximum_relative, absolute / std::max(std::fabs(expected[row]), 1.0F));
    squared += static_cast<double>(absolute) * absolute;
  }
  bool q8_equal = true;
  for (std::size_t block = 0; block < staged.size(); ++block) {
    q8_equal = q8_equal && actual_staged[block].scale == staged[block].scale;
    for (std::size_t lane = 0; lane < 32; ++lane) {
      q8_equal = q8_equal && actual_staged[block].values[lane] ==
                                 staged[block].values[lane];
    }
  }
  const float rms = static_cast<float>(std::sqrt(squared / rows));
  std::printf("case=%s rows=%zu columns=%zu max_abs=%.9g max_rel=%.9g "
              "rms=%.9g nonfinite=%zu q8_equal=%s mean_ms=%.9g\n",
              name, rows, columns, maximum_absolute, maximum_relative, rms,
              nonfinite, q8_equal ? "true" : "false", milliseconds / 30.0F);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(device_output);
  cudaFree(device_staged);
  cudaFree(device_activation);
  cudaFree(device_weights);
  return q8_equal && nonfinite == 0 && maximum_absolute <= 3.0e-4F &&
                 rms <= 2.0e-4F
             ? 0
             : 1;
}

int run_prompt_case(qw38::cuda::QuantKind kind, const char* name,
                    std::size_t output_rows, std::size_t columns,
                    std::size_t prompt_rows) {
  std::vector<std::uint8_t> weights;
  fill_weights(kind, output_rows, columns, &weights);
  std::vector<__nv_bfloat16> prompt(prompt_rows * columns);
  std::vector<qw38::cuda::Q8Block> staged(prompt_rows * (columns / 32));
  std::vector<float> expected(prompt_rows * output_rows);
  for (std::size_t prompt_row = 0; prompt_row < prompt_rows; ++prompt_row) {
    std::vector<__nv_bfloat16> row;
    std::vector<qw38::cuda::Q8Block> row_staged;
    std::vector<float> row_expected;
    make_activation(columns, &row, &row_staged, prompt_row);
    std::copy(row.begin(), row.end(), prompt.begin() + prompt_row * columns);
    std::copy(row_staged.begin(), row_staged.end(),
              staged.begin() + prompt_row * (columns / 32));
    if (!reference(kind, weights, output_rows, columns, row_staged,
                   &row_expected)) {
      return 1;
    }
    std::copy(row_expected.begin(), row_expected.end(),
              expected.begin() + prompt_row * output_rows);
  }

  std::uint8_t* device_weights = nullptr;
  __nv_bfloat16* device_prompt = nullptr;
  qw38::cuda::Q8Block* device_staged = nullptr;
  float* device_output = nullptr;
  cudaError_t error = cudaMalloc(&device_weights, weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_prompt, prompt.size() * sizeof(prompt[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_staged, staged.size() * sizeof(staged[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_output, expected.size() * sizeof(expected[0]));
  }
  if (error != cudaSuccess) return fail_cuda("MMQ cudaMalloc", error);
  error = cudaMemcpy(device_weights, weights.data(), weights.size(),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_prompt, prompt.data(),
                       prompt.size() * sizeof(prompt[0]), cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("MMQ cudaMemcpy H2D", error);

  for (int warmup = 0; warmup < 3 && error == cudaSuccess; ++warmup) {
    error = qw38::cuda::launch_quant_mmq(
        kind, device_weights, output_rows, columns, device_prompt, prompt_rows,
        device_staged, device_output, nullptr);
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (error == cudaSuccess) error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error == cudaSuccess) error = cudaEventRecord(start);
  for (int sample = 0; sample < 30 && error == cudaSuccess; ++sample) {
    error = qw38::cuda::launch_quant_mmq(
        kind, device_weights, output_rows, columns, device_prompt, prompt_rows,
        device_staged, device_output, nullptr);
  }
  if (error == cudaSuccess) error = cudaEventRecord(stop);
  if (error == cudaSuccess) error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) return fail_cuda("CUDA MMQ execution", error);
  float milliseconds = 0.0F;
  error = cudaEventElapsedTime(&milliseconds, start, stop);
  if (error != cudaSuccess) return fail_cuda("MMQ cudaEventElapsedTime", error);

  std::vector<float> actual(expected.size());
  std::vector<qw38::cuda::Q8Block> actual_staged(staged.size());
  error = cudaMemcpy(actual.data(), device_output,
                     actual.size() * sizeof(actual[0]), cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(actual_staged.data(), device_staged,
                       actual_staged.size() * sizeof(actual_staged[0]),
                       cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("MMQ cudaMemcpy D2H", error);

  float maximum_absolute = 0.0F;
  float maximum_relative = 0.0F;
  double squared = 0.0;
  std::size_t nonfinite = 0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index])) ++nonfinite;
    const float absolute = std::fabs(actual[index] - expected[index]);
    maximum_absolute = std::max(maximum_absolute, absolute);
    maximum_relative = std::max(
        maximum_relative,
        absolute / std::max(std::fabs(expected[index]), 1.0F));
    squared += static_cast<double>(absolute) * absolute;
  }
  bool q8_equal = true;
  for (std::size_t block = 0; block < staged.size(); ++block) {
    q8_equal = q8_equal && actual_staged[block].scale == staged[block].scale;
    for (std::size_t lane = 0; lane < 32; ++lane) {
      q8_equal = q8_equal && actual_staged[block].values[lane] ==
                                 staged[block].values[lane];
    }
  }
  const float rms =
      static_cast<float>(std::sqrt(squared / static_cast<double>(actual.size())));
  std::printf("mmq_case=%s prompt_rows=%zu output_rows=%zu columns=%zu "
              "max_abs=%.9g max_rel=%.9g rms=%.9g nonfinite=%zu "
              "q8_equal=%s mean_ms=%.9g\n",
              name, prompt_rows, output_rows, columns, maximum_absolute,
              maximum_relative, rms, nonfinite, q8_equal ? "true" : "false",
              milliseconds / 30.0F);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(device_output);
  cudaFree(device_staged);
  cudaFree(device_prompt);
  cudaFree(device_weights);
  return q8_equal && nonfinite == 0 && maximum_absolute <= 5.0e-4F &&
                 rms <= 2.5e-4F
             ? 0
             : 1;
}

}  // namespace

int main() {
  if (qw38::cuda::selected_mmv_warps(48) != 4 ||
      qw38::cuda::selected_mmv_warps(1024) != 8 ||
      qw38::cuda::selected_mmv_warps(5120) != 16 ||
      qw38::cuda::selected_mmv_warps(10240) != 8 ||
      qw38::cuda::selected_mmv_warps(12288) != 4 ||
      qw38::cuda::selected_mmv_warps(17408) != 8 ||
      qw38::cuda::selected_mmv_warps(248320) != 4 ||
      qw38::cuda::selected_mmq_prompt_tile(1) != 1 ||
      qw38::cuda::selected_mmq_prompt_tile(2) != 2 ||
      qw38::cuda::selected_mmq_prompt_tile(4) != 4 ||
      qw38::cuda::selected_mmq_prompt_tile(64) != 8) {
    std::fprintf(stderr, "dispatch table selection failed\n");
    return 1;
  }
  if (qw38::cuda::q8_workspace_bytes(255) != 0 ||
      qw38::cuda::q8_workspace_bytes(256) !=
          8 * sizeof(qw38::cuda::Q8Block)) {
    std::fprintf(stderr, "workspace sizing failed\n");
    return 1;
  }
  if (qw38::cuda::launch_quant_mmv(
          qw38::cuda::QuantKind::kQ4K, nullptr, 1, 256, nullptr, nullptr,
          nullptr, nullptr) != cudaErrorInvalidValue) {
    std::fprintf(stderr, "invalid launch was not rejected\n");
    return 1;
  }
  if (run_case(qw38::cuda::QuantKind::kQ4K, "q4_k_17x256", 17, 256) != 0 ||
      run_case(qw38::cuda::QuantKind::kQ4K, "q4_k_257x512", 257, 512) != 0 ||
      run_case(qw38::cuda::QuantKind::kQ6K, "q6_k_17x256", 17, 256) != 0 ||
      run_case(qw38::cuda::QuantKind::kQ6K, "q6_k_257x512", 257, 512) != 0 ||
      run_case(qw38::cuda::QuantKind::kQ8_0, "q8_0_17x256", 17, 256) != 0 ||
      run_case(qw38::cuda::QuantKind::kQ8_0, "q8_0_257x512", 257, 512) != 0) {
    return 1;
  }
  if (qw38::cuda::q8_prompt_workspace_bytes(3, 256) !=
          24 * sizeof(qw38::cuda::Q8Block) ||
      qw38::cuda::q8_prompt_workspace_bytes(0, 256) != 0 ||
      run_prompt_case(qw38::cuda::QuantKind::kQ4K, "q4_k_3x17x256", 17,
                      256, 3) != 0 ||
      run_prompt_case(qw38::cuda::QuantKind::kQ4K, "q4_k_5x257x512", 257,
                      512, 5) != 0 ||
      run_prompt_case(qw38::cuda::QuantKind::kQ6K, "q6_k_1x17x256", 17,
                      256, 1) != 0 ||
      run_prompt_case(qw38::cuda::QuantKind::kQ6K, "q6_k_9x257x512", 257,
                      512, 9) != 0 ||
      run_prompt_case(qw38::cuda::QuantKind::kQ8_0, "q8_0_3x17x256", 17,
                      256, 3) != 0) {
    return 1;
  }
  std::printf("status=passed\n");
  return 0;
}
