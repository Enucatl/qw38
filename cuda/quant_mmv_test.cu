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

// ds4-style host reference: dequantized F32 weights × BF16→float activations.
bool reference_dequant_gemm(qw38::cuda::QuantKind kind,
                            const std::vector<std::uint8_t>& weights,
                            std::size_t output_rows, std::size_t columns,
                            const std::vector<__nv_bfloat16>& prompt,
                            std::size_t prompt_rows,
                            std::vector<float>* output) {
  const std::size_t block_bytes =
      kind == qw38::cuda::QuantKind::kQ4K ? 144
      : kind == qw38::cuda::QuantKind::kQ6K ? 210
                                           : 34;
  const std::size_t block_values =
      kind == qw38::cuda::QuantKind::kQ8_0 ? 32 : 256;
  output->assign(prompt_rows * output_rows, 0.0F);
  std::vector<float> decoded(block_values);
  for (std::size_t out = 0; out < output_rows; ++out) {
    for (std::size_t prompt_row = 0; prompt_row < prompt_rows; ++prompt_row) {
      float sum = 0.0F;
      for (std::size_t block = 0; block < columns / block_values; ++block) {
        const std::uint8_t* packed =
            weights.data() +
            (out * (columns / block_values) + block) * block_bytes;
        const qw38::Status status =
            kind == qw38::cuda::QuantKind::kQ4K
                ? qw38::internal::decode_q4_k(packed, block_bytes,
                                              decoded.data(), decoded.size())
            : kind == qw38::cuda::QuantKind::kQ6K
                ? qw38::internal::decode_q6_k(packed, block_bytes,
                                              decoded.data(), decoded.size())
                : qw38::internal::decode_q8_0(packed, block_bytes,
                                              decoded.data(), decoded.size());
        if (!status.is_ok()) return false;
        for (std::size_t within = 0; within < block_values; ++within) {
          const std::size_t column = block * block_values + within;
          sum += decoded[within] * __bfloat162float(
                     prompt[prompt_row * columns + column]);
        }
      }
      (*output)[prompt_row * output_rows + out] = sum;
    }
  }
  return true;
}

// ds4 Q4_K check_close: fail only when both abs and rel exceed.
bool ds4_q4k_association_ok(const std::vector<float>& got,
                            const std::vector<float>& ref, std::size_t columns,
                            float* max_abs, float* max_rel, float* rms,
                            std::size_t* association_bad,
                            std::size_t* nonfinite) {
  constexpr float kAbsScale = 0.20F;
  constexpr float kRelTol = 0.05F;
  const float abs_tol = kAbsScale * std::sqrt(static_cast<float>(columns));
  *max_abs = 0.0F;
  *max_rel = 0.0F;
  *association_bad = 0;
  *nonfinite = 0;
  double squared = 0.0;
  for (std::size_t index = 0; index < got.size(); ++index) {
    if (!std::isfinite(got[index]) || !std::isfinite(ref[index])) {
      ++*nonfinite;
      continue;
    }
    const float absolute = std::fabs(got[index] - ref[index]);
    const float relative =
        ref[index] != 0.0F ? absolute / std::fabs(ref[index])
                           : (absolute > 0.0F ? INFINITY : 0.0F);
    *max_abs = std::max(*max_abs, absolute);
    if (std::isfinite(relative)) *max_rel = std::max(*max_rel, relative);
    squared += static_cast<double>(absolute) * absolute;
    if (absolute > abs_tol && relative > kRelTol) ++*association_bad;
  }
  *rms = got.empty()
             ? 0.0F
             : static_cast<float>(std::sqrt(squared / got.size()));
  return *nonfinite == 0 && *association_bad == 0;
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
  for (std::size_t prompt_row = 0; prompt_row < prompt_rows; ++prompt_row) {
    std::vector<__nv_bfloat16> row;
    std::vector<qw38::cuda::Q8Block> row_staged;
    make_activation(columns, &row, &row_staged, prompt_row);
    std::copy(row.begin(), row.end(), prompt.begin() + prompt_row * columns);
    std::copy(row_staged.begin(), row_staged.end(),
              staged.begin() + prompt_row * (columns / 32));
  }
  std::vector<float> expected;
  const bool q4_or_q6 = kind == qw38::cuda::QuantKind::kQ4K ||
                        kind == qw38::cuda::QuantKind::kQ6K;
  if (q4_or_q6) {
    if (!reference_dequant_gemm(kind, weights, output_rows, columns, prompt,
                                prompt_rows, &expected)) {
      return 1;
    }
  } else {
    expected.assign(prompt_rows * output_rows, 0.0F);
    for (std::size_t prompt_row = 0; prompt_row < prompt_rows; ++prompt_row) {
      std::vector<qw38::cuda::Q8Block> row_staged(
          staged.begin() + prompt_row * (columns / 32),
          staged.begin() + (prompt_row + 1) * (columns / 32));
      std::vector<float> row_expected;
      if (!reference(kind, weights, output_rows, columns, row_staged,
                     &row_expected)) {
        return 1;
      }
      std::copy(row_expected.begin(), row_expected.end(),
                expected.begin() + prompt_row * output_rows);
    }
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
    error = qw38::cuda::launch_quant_mmq_variant(
        kind, device_weights, output_rows, columns, device_prompt, prompt_rows,
        device_staged, device_output,
        qw38::cuda::selected_mmq_prompt_tile(kind, prompt_rows), nullptr);
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (error == cudaSuccess) error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error == cudaSuccess) error = cudaEventRecord(start);
  for (int sample = 0; sample < 30 && error == cudaSuccess; ++sample) {
    error = qw38::cuda::launch_quant_mmq_variant(
        kind, device_weights, output_rows, columns, device_prompt, prompt_rows,
        device_staged, device_output,
        qw38::cuda::selected_mmq_prompt_tile(kind, prompt_rows), nullptr);
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

  bool q8_equal = true;
  for (std::size_t block = 0; block < staged.size(); ++block) {
    q8_equal = q8_equal && actual_staged[block].scale == staged[block].scale;
    for (std::size_t lane = 0; lane < 32; ++lane) {
      q8_equal = q8_equal && actual_staged[block].values[lane] ==
                                 staged[block].values[lane];
    }
  }

  float maximum_absolute = 0.0F;
  float maximum_relative = 0.0F;
  float rms = 0.0F;
  std::size_t nonfinite = 0;
  std::size_t association_bad = 0;
  bool numeric_ok = false;
  if (q4_or_q6) {
    numeric_ok = ds4_q4k_association_ok(actual, expected, columns,
                                        &maximum_absolute, &maximum_relative,
                                        &rms, &association_bad, &nonfinite);
    std::printf("mmq_case=%s prompt_rows=%zu output_rows=%zu columns=%zu "
                "max_abs=%.9g max_rel=%.9g rms=%.9g association_bad=%zu "
                "nonfinite=%zu q8_equal=%s mean_ms=%.9g gate=ds4_q4k_parity "
                "ref=cpu_dequant_gemm\n",
                name, prompt_rows, output_rows, columns, maximum_absolute,
                maximum_relative, rms, association_bad, nonfinite,
                q8_equal ? "true" : "false", milliseconds / 30.0F);
  } else {
    double squared = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
      if (!std::isfinite(actual[index])) ++nonfinite;
      const float absolute = std::fabs(actual[index] - expected[index]);
      maximum_absolute = std::max(maximum_absolute, absolute);
      maximum_relative = std::max(
          maximum_relative,
          absolute / std::max(std::fabs(expected[index]), 1.0F));
      squared += static_cast<double>(absolute) * absolute;
    }
    rms = static_cast<float>(
        std::sqrt(squared / static_cast<double>(actual.size())));
    numeric_ok =
        nonfinite == 0 && maximum_absolute <= 5.0e-4F && rms <= 2.5e-4F;
    std::printf("mmq_case=%s prompt_rows=%zu output_rows=%zu columns=%zu "
                "max_abs=%.9g max_rel=%.9g rms=%.9g nonfinite=%zu "
                "q8_equal=%s mean_ms=%.9g\n",
                name, prompt_rows, output_rows, columns, maximum_absolute,
                maximum_relative, rms, nonfinite, q8_equal ? "true" : "false",
                milliseconds / 30.0F);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(device_output);
  cudaFree(device_staged);
  cudaFree(device_prompt);
  cudaFree(device_weights);
  return q8_equal && numeric_ok ? 0 : 1;
}

int run_mma_case(qw38::cuda::QuantKind kind, const char* name,
                 std::size_t output_rows, std::size_t columns,
                 std::size_t prompt_rows, unsigned int tile) {
  std::vector<std::uint8_t> weights;
  fill_weights(kind, output_rows, columns, &weights);
  std::vector<__nv_bfloat16> prompt;
  std::vector<qw38::cuda::Q8Block> staged;
  prompt.resize(prompt_rows * columns);
  staged.resize(prompt_rows * (columns / 32));
  for (std::size_t prompt_row = 0; prompt_row < prompt_rows; ++prompt_row) {
    std::vector<__nv_bfloat16> row_activation;
    std::vector<qw38::cuda::Q8Block> row_staged;
    make_activation(columns, &row_activation, &row_staged, prompt_row);
    std::copy(row_activation.begin(), row_activation.end(),
              prompt.begin() + prompt_row * columns);
    std::copy(row_staged.begin(), row_staged.end(),
              staged.begin() + prompt_row * (columns / 32));
  }
  std::vector<float> expected;
  if (!reference_dequant_gemm(kind, weights, output_rows, columns, prompt,
                              prompt_rows, &expected)) {
    return 1;
  }

  std::uint8_t* device_weights = nullptr;
  __nv_bfloat16* device_prompt = nullptr;
  qw38::cuda::Q8Block* device_staged = nullptr;
  float* device_mma = nullptr;
  float* device_variant = nullptr;
  cudaError_t error = cudaMalloc(&device_weights, weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_prompt, prompt.size() * sizeof(prompt[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_staged, staged.size() * sizeof(staged[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_mma, expected.size() * sizeof(expected[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_variant, expected.size() * sizeof(expected[0]));
  }
  if (error != cudaSuccess) return fail_cuda("MMA cudaMalloc", error);
  error = cudaMemcpy(device_weights, weights.data(), weights.size(),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_prompt, prompt.data(),
                       prompt.size() * sizeof(prompt[0]), cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("MMA cudaMemcpy H2D", error);

  error = qw38::cuda::launch_quant_mmq_variant(
      kind, device_weights, output_rows, columns, device_prompt, prompt_rows,
      device_staged, device_variant,
      qw38::cuda::selected_mmq_prompt_tile(kind, prompt_rows), nullptr);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmq_mma_tile(
        kind, device_weights, output_rows, columns, device_prompt, prompt_rows,
        device_staged, device_mma, tile, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("MMA execution", error);

  std::vector<float> mma(expected.size());
  std::vector<float> variant(expected.size());
  error = cudaMemcpy(mma.data(), device_mma, mma.size() * sizeof(mma[0]),
                     cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(variant.data(), device_variant,
                       variant.size() * sizeof(variant[0]),
                       cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("MMA cudaMemcpy D2H", error);

  // Option C: MMA and variant both vs CPU dequant×BF16 GEMM under ds4 Q4_K
  // association (plan.md). Variant-vs-MMA delta is informational only.
  float mma_abs = 0.0F;
  float mma_rel = 0.0F;
  float mma_rms = 0.0F;
  std::size_t mma_bad = 0;
  std::size_t mma_nonfinite = 0;
  const bool mma_ok = ds4_q4k_association_ok(
      mma, expected, columns, &mma_abs, &mma_rel, &mma_rms, &mma_bad,
      &mma_nonfinite);
  float var_abs = 0.0F;
  float var_rel = 0.0F;
  float var_rms = 0.0F;
  std::size_t var_bad = 0;
  std::size_t var_nonfinite = 0;
  const bool variant_ok = ds4_q4k_association_ok(
      variant, expected, columns, &var_abs, &var_rel, &var_rms, &var_bad,
      &var_nonfinite);
  std::printf("mma_case=%s prompt_rows=%zu output_rows=%zu columns=%zu tile=%u "
              "mma_max_abs=%.9g mma_max_rel=%.9g mma_rms=%.9g mma_bad=%zu "
              "mma_nonfinite=%zu variant_max_abs=%.9g variant_max_rel=%.9g "
              "variant_bad=%zu gate=ds4_q4k_parity ref=cpu_dequant_gemm\n",
              name, prompt_rows, output_rows, columns, tile, mma_abs, mma_rel,
              mma_rms, mma_bad, mma_nonfinite, var_abs, var_rel, var_bad);
  cudaFree(device_variant);
  cudaFree(device_mma);
  cudaFree(device_staged);
  cudaFree(device_prompt);
  cudaFree(device_weights);
  return mma_ok && variant_ok ? 0 : 1;
}

int time_mma_tile(qw38::cuda::QuantKind kind, std::size_t output_rows,
                  std::size_t columns, std::size_t prompt_rows,
                  unsigned int tile, float* mean_ms, int* occupancy) {
  *occupancy = qw38::cuda::mma_mmq_occupancy(kind, tile);
  if (*occupancy < 1) return 1;
  std::vector<std::uint8_t> weights;
  fill_weights(kind, output_rows, columns, &weights);
  std::vector<__nv_bfloat16> prompt(prompt_rows * columns);
  for (std::size_t index = 0; index < prompt.size(); ++index) {
    prompt[index] = __float2bfloat16_rn(
        std::sin(static_cast<float>(index) * 0.01F) * 0.25F);
  }
  std::uint8_t* device_weights = nullptr;
  __nv_bfloat16* device_prompt = nullptr;
  qw38::cuda::Q8Block* device_workspace = nullptr;
  float* device_output = nullptr;
  cudaError_t error = cudaMalloc(&device_weights, weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_prompt, prompt.size() * sizeof(prompt[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_workspace,
                       qw38::cuda::q8_prompt_workspace_bytes(prompt_rows, columns));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_output,
                       prompt_rows * output_rows * sizeof(float));
  }
  if (error != cudaSuccess) return fail_cuda("MMA sweep cudaMalloc", error);
  error = cudaMemcpy(device_weights, weights.data(), weights.size(),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_prompt, prompt.data(),
                       prompt.size() * sizeof(prompt[0]), cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("MMA sweep H2D", error);
  error = qw38::cuda::launch_quant_mmq_mma_tile(
      kind, device_weights, output_rows, columns, device_prompt, prompt_rows,
      device_workspace, device_output, tile, nullptr);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    cudaFree(device_output);
    cudaFree(device_workspace);
    cudaFree(device_prompt);
    cudaFree(device_weights);
    return fail_cuda("MMA sweep launch", error);
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  float total = 0.0F;
  for (int sample = 0; sample < 3 && error == cudaSuccess; ++sample) {
    error = cudaEventRecord(start);
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_quant_mmq_mma_tile(
          kind, device_weights, output_rows, columns, device_prompt,
          prompt_rows, device_workspace, device_output, tile, nullptr);
    }
    if (error == cudaSuccess) error = cudaEventRecord(stop);
    if (error == cudaSuccess) error = cudaEventSynchronize(stop);
    float milliseconds = 0.0F;
    if (error == cudaSuccess) {
      error = cudaEventElapsedTime(&milliseconds, start, stop);
    }
    std::printf("opt018_j_sample kind=%s output_rows=%zu columns=%zu "
                "prompt_rows=%zu tile=%u replicate=%d ms=%.9g occupancy=%d\n",
                kind == qw38::cuda::QuantKind::kQ4K ? "q4_k" : "q6_k",
                output_rows, columns, prompt_rows, tile, sample, milliseconds,
                *occupancy);
    total += milliseconds;
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(device_output);
  cudaFree(device_workspace);
  cudaFree(device_prompt);
  cudaFree(device_weights);
  if (error != cudaSuccess) return fail_cuda("MMA sweep time", error);
  *mean_ms = total / 3.0F;
  std::printf("mma_tune kind=%s output_rows=%zu columns=%zu prompt_rows=%zu "
              "tile=%u mean_ms=%.9g occupancy=%d\n",
              kind == qw38::cuda::QuantKind::kQ4K ? "q4_k" : "q6_k",
              output_rows, columns, prompt_rows, tile, *mean_ms, *occupancy);
  return 0;
}

int time_variant_tile(qw38::cuda::QuantKind kind, std::size_t output_rows,
                       std::size_t columns, std::size_t prompt_rows,
                       float* mean_ms) {
  std::vector<std::uint8_t> weights;
  fill_weights(kind, output_rows, columns, &weights);
  std::vector<__nv_bfloat16> prompt(prompt_rows * columns);
  for (std::size_t index = 0; index < prompt.size(); ++index) {
    prompt[index] = __float2bfloat16_rn(
        std::sin(static_cast<float>(index) * 0.01F) * 0.25F);
  }
  std::uint8_t* device_weights = nullptr;
  __nv_bfloat16* device_prompt = nullptr;
  qw38::cuda::Q8Block* device_workspace = nullptr;
  float* device_output = nullptr;
  cudaError_t error = cudaMalloc(&device_weights, weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_prompt, prompt.size() * sizeof(prompt[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_workspace,
                       qw38::cuda::q8_prompt_workspace_bytes(prompt_rows, columns));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_output,
                       prompt_rows * output_rows * sizeof(float));
  }
  if (error != cudaSuccess) return fail_cuda("variant sweep cudaMalloc", error);
  error = cudaMemcpy(device_weights, weights.data(), weights.size(),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_prompt, prompt.data(),
                       prompt.size() * sizeof(prompt[0]), cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("variant sweep H2D", error);
  const unsigned tile =
      qw38::cuda::selected_mmq_prompt_tile(kind, prompt_rows);
  error = qw38::cuda::launch_quant_mmq_variant(
      kind, device_weights, output_rows, columns, device_prompt, prompt_rows,
      device_workspace, device_output, tile, nullptr);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    cudaFree(device_output);
    cudaFree(device_workspace);
    cudaFree(device_prompt);
    cudaFree(device_weights);
    return fail_cuda("variant sweep launch", error);
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  float total = 0.0F;
  for (int sample = 0; sample < 3 && error == cudaSuccess; ++sample) {
    error = cudaEventRecord(start);
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_quant_mmq_variant(
          kind, device_weights, output_rows, columns, device_prompt,
          prompt_rows, device_workspace, device_output, tile, nullptr);
    }
    if (error == cudaSuccess) error = cudaEventRecord(stop);
    if (error == cudaSuccess) error = cudaEventSynchronize(stop);
    float milliseconds = 0.0F;
    if (error == cudaSuccess) {
      error = cudaEventElapsedTime(&milliseconds, start, stop);
    }
    total += milliseconds;
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(device_output);
  cudaFree(device_workspace);
  cudaFree(device_prompt);
  cudaFree(device_weights);
  if (error != cudaSuccess) return fail_cuda("variant sweep time", error);
  *mean_ms = total / 3.0F;
  return 0;
}

int time_q8_mma_tile(std::size_t output_rows, std::size_t columns,
                     std::size_t prompt_rows, unsigned int tile, float* mean_ms,
                     int* occupancy, std::size_t* nonfinite) {
  *occupancy = qw38::cuda::q8_mma_mmq_occupancy(tile);
  if (*occupancy < 1) return 1;
  std::vector<std::uint8_t> weights;
  fill_weights(qw38::cuda::QuantKind::kQ8_0, output_rows, columns, &weights);
  std::vector<__nv_bfloat16> prompt(prompt_rows * columns);
  for (std::size_t index = 0; index < prompt.size(); ++index) {
    prompt[index] = __float2bfloat16_rn(
        std::sin(static_cast<float>(index) * 0.01F) * 0.25F);
  }
  std::uint8_t* device_weights = nullptr;
  __nv_bfloat16* device_prompt = nullptr;
  float* device_output = nullptr;
  cudaError_t error = cudaMalloc(&device_weights, weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_prompt, prompt.size() * sizeof(prompt[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_output,
                       prompt_rows * output_rows * sizeof(float));
  }
  if (error != cudaSuccess) return fail_cuda("Q8 MMA sweep cudaMalloc", error);
  error = cudaMemcpy(device_weights, weights.data(), weights.size(),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_prompt, prompt.data(),
                       prompt.size() * sizeof(prompt[0]), cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("Q8 MMA sweep H2D", error);
  error = qw38::cuda::launch_q8_mmq_mma_tile(
      device_weights, output_rows, columns, device_prompt, prompt_rows,
      device_output, tile, nullptr);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    cudaFree(device_output);
    cudaFree(device_prompt);
    cudaFree(device_weights);
    return fail_cuda("Q8 MMA sweep launch", error);
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  float total = 0.0F;
  for (int sample = 0; sample < 3 && error == cudaSuccess; ++sample) {
    error = cudaEventRecord(start);
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_q8_mmq_mma_tile(
          device_weights, output_rows, columns, device_prompt, prompt_rows,
          device_output, tile, nullptr);
    }
    if (error == cudaSuccess) error = cudaEventRecord(stop);
    if (error == cudaSuccess) error = cudaEventSynchronize(stop);
    float milliseconds = 0.0F;
    if (error == cudaSuccess) {
      error = cudaEventElapsedTime(&milliseconds, start, stop);
    }
    if (error == cudaSuccess) {
      total += milliseconds;
      std::printf("q8_mma_tune output_rows=%zu columns=%zu prompt_rows=%zu "
                  "tile=%u sample=%d ms=%.9g occupancy=%d\n",
                  output_rows, columns, prompt_rows, tile, sample, milliseconds,
                  *occupancy);
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::vector<float> host(prompt_rows * output_rows);
  if (error == cudaSuccess) {
    error = cudaMemcpy(host.data(), device_output,
                       host.size() * sizeof(float), cudaMemcpyDeviceToHost);
  }
  cudaFree(device_output);
  cudaFree(device_prompt);
  cudaFree(device_weights);
  if (error != cudaSuccess) return fail_cuda("Q8 MMA sweep time", error);
  *nonfinite = 0;
  for (float value : host) {
    if (!std::isfinite(value)) ++*nonfinite;
  }
  *mean_ms = total / 3.0F;
  std::printf("q8_mma_tune_mean output_rows=%zu columns=%zu prompt_rows=%zu "
              "tile=%u mean_ms=%.9g occupancy=%d nonfinite=%zu\n",
              output_rows, columns, prompt_rows, tile, *mean_ms, *occupancy,
              *nonfinite);
  return *nonfinite == 0 ? 0 : 1;
}

int time_q8_variant_tile(std::size_t output_rows, std::size_t columns,
                         std::size_t prompt_rows, unsigned int tile,
                         float* mean_ms) {
  std::vector<std::uint8_t> weights;
  fill_weights(qw38::cuda::QuantKind::kQ8_0, output_rows, columns, &weights);
  std::vector<__nv_bfloat16> prompt(prompt_rows * columns);
  for (std::size_t index = 0; index < prompt.size(); ++index) {
    prompt[index] = __float2bfloat16_rn(
        std::sin(static_cast<float>(index) * 0.01F) * 0.25F);
  }
  std::uint8_t* device_weights = nullptr;
  __nv_bfloat16* device_prompt = nullptr;
  float* device_output = nullptr;
  cudaError_t error = cudaMalloc(&device_weights, weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_prompt, prompt.size() * sizeof(prompt[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_output,
                       prompt_rows * output_rows * sizeof(float));
  }
  if (error != cudaSuccess) {
    return fail_cuda("Q8 variant sweep cudaMalloc", error);
  }
  error = cudaMemcpy(device_weights, weights.data(), weights.size(),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_prompt, prompt.data(),
                       prompt.size() * sizeof(prompt[0]), cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("Q8 variant sweep H2D", error);
  error = qw38::cuda::launch_q8_mmq_bf16_variant(
      device_weights, output_rows, columns, device_prompt, prompt_rows,
      device_output, tile, nullptr);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    cudaFree(device_output);
    cudaFree(device_prompt);
    cudaFree(device_weights);
    return fail_cuda("Q8 variant sweep launch", error);
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  float total = 0.0F;
  for (int sample = 0; sample < 3 && error == cudaSuccess; ++sample) {
    error = cudaEventRecord(start);
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_q8_mmq_bf16_variant(
          device_weights, output_rows, columns, device_prompt, prompt_rows,
          device_output, tile, nullptr);
    }
    if (error == cudaSuccess) error = cudaEventRecord(stop);
    if (error == cudaSuccess) error = cudaEventSynchronize(stop);
    float milliseconds = 0.0F;
    if (error == cudaSuccess) {
      error = cudaEventElapsedTime(&milliseconds, start, stop);
    }
    total += milliseconds;
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(device_output);
  cudaFree(device_prompt);
  cudaFree(device_weights);
  if (error != cudaSuccess) return fail_cuda("Q8 variant sweep time", error);
  *mean_ms = total / 3.0F;
  return 0;
}

int run_q8_mma_case(const char* name, std::size_t output_rows,
                    std::size_t columns, std::size_t prompt_rows,
                    unsigned int tile) {
  std::vector<std::uint8_t> weights;
  fill_weights(qw38::cuda::QuantKind::kQ8_0, output_rows, columns, &weights);
  std::vector<__nv_bfloat16> prompt(prompt_rows * columns);
  std::vector<qw38::cuda::Q8Block> staged(prompt_rows * (columns / 32));
  std::vector<float> staged_ref(prompt_rows * output_rows);
  for (std::size_t prompt_row = 0; prompt_row < prompt_rows; ++prompt_row) {
    std::vector<__nv_bfloat16> row_activation;
    std::vector<qw38::cuda::Q8Block> row_staged;
    make_activation(columns, &row_activation, &row_staged, prompt_row);
    std::copy(row_activation.begin(), row_activation.end(),
              prompt.begin() + prompt_row * columns);
    std::copy(row_staged.begin(), row_staged.end(),
              staged.begin() + prompt_row * (columns / 32));
    std::vector<float> row_expected;
    if (!reference(qw38::cuda::QuantKind::kQ8_0, weights, output_rows, columns,
                   row_staged, &row_expected)) {
      return 1;
    }
    std::copy(row_expected.begin(), row_expected.end(),
              staged_ref.begin() + prompt_row * output_rows);
  }

  std::uint8_t* device_weights = nullptr;
  __nv_bfloat16* device_prompt = nullptr;
  qw38::cuda::Q8Block* device_staged = nullptr;
  float* device_mma = nullptr;
  float* device_variant = nullptr;
  float* device_staged_gpu = nullptr;
  const std::size_t values = prompt_rows * output_rows;
  cudaError_t error = cudaMalloc(&device_weights, weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_prompt, prompt.size() * sizeof(prompt[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_staged, staged.size() * sizeof(staged[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_mma, values * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_variant, values * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_staged_gpu, values * sizeof(float));
  }
  if (error != cudaSuccess) return fail_cuda("Q8 MMA cudaMalloc", error);
  error = cudaMemcpy(device_weights, weights.data(), weights.size(),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_prompt, prompt.data(),
                       prompt.size() * sizeof(prompt[0]), cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_staged, staged.data(),
                       staged.size() * sizeof(staged[0]), cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("Q8 MMA cudaMemcpy H2D", error);

  const unsigned variant_tile = qw38::cuda::selected_mmq_prompt_tile(
      qw38::cuda::QuantKind::kQ8_0, prompt_rows);
  error = qw38::cuda::launch_q8_mmq_bf16_variant(
      device_weights, output_rows, columns, device_prompt, prompt_rows,
      device_variant, variant_tile, nullptr);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmq_variant(
        qw38::cuda::QuantKind::kQ8_0, device_weights, output_rows, columns,
        device_prompt, prompt_rows, device_staged, device_staged_gpu,
        variant_tile, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q8_mmq_mma_tile(
        device_weights, output_rows, columns, device_prompt, prompt_rows,
        device_mma, tile, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("Q8 MMA execution", error);

  std::vector<float> mma(values);
  std::vector<float> variant(values);
  std::vector<float> staged_gpu(values);
  error = cudaMemcpy(mma.data(), device_mma, mma.size() * sizeof(mma[0]),
                     cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(variant.data(), device_variant,
                       variant.size() * sizeof(variant[0]),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(staged_gpu.data(), device_staged_gpu,
                       staged_gpu.size() * sizeof(staged_gpu[0]),
                       cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("Q8 MMA cudaMemcpy D2H", error);

  float maximum_absolute = 0.0F;
  double squared = 0.0;
  std::size_t nonfinite = 0;
  for (std::size_t index = 0; index < mma.size(); ++index) {
    if (!std::isfinite(mma[index]) || !std::isfinite(variant[index])) {
      ++nonfinite;
    }
    const float absolute = std::fabs(mma[index] - variant[index]);
    maximum_absolute = std::max(maximum_absolute, absolute);
    squared += static_cast<double>(absolute) * absolute;
  }
  const float rms =
      static_cast<float>(std::sqrt(squared / static_cast<double>(mma.size())));
  float staged_max = 0.0F;
  double staged_sq = 0.0;
  std::size_t staged_nonfinite = 0;
  for (std::size_t index = 0; index < mma.size(); ++index) {
    if (!std::isfinite(mma[index]) || !std::isfinite(staged_ref[index])) {
      ++staged_nonfinite;
    }
    const float absolute = std::fabs(mma[index] - staged_ref[index]);
    staged_max = std::max(staged_max, absolute);
    staged_sq += static_cast<double>(absolute) * absolute;
  }
  const float staged_rms = static_cast<float>(
      std::sqrt(staged_sq / static_cast<double>(mma.size())));
  float gpu_max = 0.0F;
  double gpu_sq = 0.0;
  std::size_t gpu_nonfinite = 0;
  for (std::size_t index = 0; index < mma.size(); ++index) {
    if (!std::isfinite(mma[index]) || !std::isfinite(staged_gpu[index])) {
      ++gpu_nonfinite;
    }
    const float absolute = std::fabs(mma[index] - staged_gpu[index]);
    gpu_max = std::max(gpu_max, absolute);
    gpu_sq += static_cast<double>(absolute) * absolute;
  }
  const float gpu_rms =
      static_cast<float>(std::sqrt(gpu_sq / static_cast<double>(mma.size())));
  std::printf("q8_mma_case=%s prompt_rows=%zu output_rows=%zu columns=%zu "
              "tile=%u max_abs=%.9g rms=%.9g nonfinite=%zu staged_max_abs=%.9g "
              "staged_rms=%.9g staged_nonfinite=%zu gpu_staged_max_abs=%.9g "
              "gpu_staged_rms=%.9g gpu_staged_nonfinite=%zu\n",
              name, prompt_rows, output_rows, columns, tile, maximum_absolute,
              rms, nonfinite, staged_max, staged_rms, staged_nonfinite, gpu_max,
              gpu_rms, gpu_nonfinite);
  cudaFree(device_staged_gpu);
  cudaFree(device_variant);
  cudaFree(device_mma);
  cudaFree(device_staged);
  cudaFree(device_prompt);
  cudaFree(device_weights);
  return gpu_nonfinite == 0 && gpu_max <= 5.0e-4F && gpu_rms <= 2.5e-4F ? 0
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
      qw38::cuda::selected_mmq_prompt_tile(8) != 4 ||
      qw38::cuda::selected_mmq_prompt_tile(64) != 8 ||
      qw38::cuda::selected_mmq_prompt_tile(4096) != 4 ||
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ8_0, 1) !=
          1 ||
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ8_0, 2) !=
          2 ||
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ8_0, 4) !=
          4 ||
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ8_0, 64) !=
          4 ||
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ8_0, 4096) !=
          4 ||
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ6K, 1) != 1 ||
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ6K, 4) != 4 ||
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ6K, 8) != 8 ||
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ6K, 64) !=
          8 ||
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ6K, 4096) !=
          8) {
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
  if (qw38::cuda::selected_mma_mmq_prompt_tile() != 128U ||
      qw38::cuda::mma_mmq_shared_bytes(128) == 0 ||
      qw38::cuda::launch_quant_mmq_mma(
          qw38::cuda::QuantKind::kQ4K, nullptr, 17, 256, nullptr, 8, nullptr,
          nullptr, nullptr) != cudaErrorInvalidValue ||
      run_mma_case(qw38::cuda::QuantKind::kQ4K, "q4_k_mma_8x17x256", 17, 256,
                   8, 128) != 0 ||
      run_mma_case(qw38::cuda::QuantKind::kQ4K, "q4_k_mma_9x17x256", 17, 256,
                   9, 128) != 0 ||
      run_mma_case(qw38::cuda::QuantKind::kQ4K, "q4_k_mma_64x17x256", 17, 256,
                   64, 128) != 0 ||
      run_mma_case(qw38::cuda::QuantKind::kQ4K, "q4_k_mma_65x17x256", 17, 256,
                   65, 128) != 0 ||
      run_mma_case(qw38::cuda::QuantKind::kQ4K, "q4_k_mma_9x64x256", 64, 256,
                   9, 128) != 0 ||
      run_mma_case(qw38::cuda::QuantKind::kQ4K, "q4_k_mma_17x128x256", 128,
                   256, 17, 128) != 0 ||
      run_mma_case(qw38::cuda::QuantKind::kQ4K, "q4_k_mma_8x1024x5120", 1024,
                   5120, 8, 128) != 0 ||
      run_mma_case(qw38::cuda::QuantKind::kQ6K, "q6_k_mma_8x17x256", 17, 256,
                   8, 128) != 0 ||
      run_mma_case(qw38::cuda::QuantKind::kQ6K, "q6_k_mma_9x64x256", 64, 256,
                   9, 128) != 0 ||
      run_mma_case(qw38::cuda::QuantKind::kQ6K, "q6_k_mma_17x128x256", 128,
                   256, 17, 128) != 0 ||
      run_mma_case(qw38::cuda::QuantKind::kQ6K, "q6_k_mma_64x5120x6144", 5120,
                   6144, 64, 128) != 0) {
    return 1;
  }
  unsigned int winner = 128;
  float best = 1.0e30F;
  const unsigned int tiles[3] = {32U, 64U, 128U};
  const std::size_t shapes[2][2] = {{17408, 5120}, {5120, 17408}};
  for (unsigned int tile : tiles) {
    float mean = 0.0F;
    bool eligible = true;
    if (run_mma_case(qw38::cuda::QuantKind::kQ4K, "q4_k_mma_elig_8x17x256", 17,
                      256, 8, tile) != 0) {
      std::printf("opt018_j_skip tile=%u reason=cud002\n", tile);
      continue;
    }
    for (const auto& shape : shapes) {
      float part = 0.0F;
      int occ = 0;
      if (time_mma_tile(qw38::cuda::QuantKind::kQ4K, shape[0], shape[1], 2048,
                        tile, &part, &occ) != 0 ||
          occ < 1) {
        eligible = false;
        break;
      }
      mean += part;
    }
    std::printf("opt018_j_mean tile=%u mean_ms=%.9g eligible=%s occupancy=%d\n",
                tile, mean, eligible ? "true" : "false",
                qw38::cuda::mma_mmq_occupancy(qw38::cuda::QuantKind::kQ4K, tile));
    if (eligible && mean < best) {
      best = mean;
      winner = tile;
    }
  }
  std::printf("mma_prompt_tile_2048=%u mean_ms=%.9g pinned=%u occupancy=%d\n",
              winner, best, qw38::cuda::selected_mma_mmq_prompt_tile(),
              qw38::cuda::mma_mmq_occupancy(qw38::cuda::QuantKind::kQ4K, winner));
  if (winner != qw38::cuda::selected_mma_mmq_prompt_tile()) {
    std::fprintf(stderr,
                 "selected_mma_mmq_prompt_tile=%u must match FFN J sweep "
                 "winner %u\n",
                 qw38::cuda::selected_mma_mmq_prompt_tile(), winner);
    return 1;
  }
  if (run_mma_case(qw38::cuda::QuantKind::kQ4K, "q4_k_mma_2048x17408x5120",
                   17408, 5120, 2048, winner) != 0 ||
      run_mma_case(qw38::cuda::QuantKind::kQ4K, "q4_k_mma_2048x5120x17408",
                   5120, 17408, 2048, winner) != 0 ||
      run_mma_case(qw38::cuda::QuantKind::kQ6K, "q6_k_mma_2048x5120x6144",
                   5120, 6144, 2048, winner) != 0) {
    return 1;
  }
  float q4_mma_speed = 0.0F;
  float q4_variant_speed = 0.0F;
  int q4_speed_occ = 0;
  if (time_mma_tile(qw38::cuda::QuantKind::kQ4K, 17408, 5120, 2048, winner,
                     &q4_mma_speed, &q4_speed_occ) != 0 ||
      time_variant_tile(qw38::cuda::QuantKind::kQ4K, 17408, 5120, 2048,
                        &q4_variant_speed) != 0 ||
      !(q4_mma_speed < q4_variant_speed)) {
    std::fprintf(stderr,
                 "Q4_K MMA speed gate failed mma_ms=%.9g variant_ms=%.9g "
                 "occupancy=%d\n",
                 q4_mma_speed, q4_variant_speed, q4_speed_occ);
    return 1;
  }
  std::printf("mma_speed_2048_17408x5120 mma_ms=%.9g variant_ms=%.9g "
              "occupancy=%d\n",
              q4_mma_speed, q4_variant_speed, q4_speed_occ);

  if (qw38::cuda::launch_quant_mmq_mma(
          qw38::cuda::QuantKind::kQ8_0, nullptr, 17, 256, nullptr, 8, nullptr,
          nullptr, nullptr) != cudaErrorInvalidValue) {
    std::fprintf(stderr, "launch_quant_mmq_mma must reject kQ8_0\n");
    return 1;
  }
  const unsigned q8_tile = qw38::cuda::selected_q8_mma_mmq_prompt_tile();
  if (run_q8_mma_case("q8_0_mma_8x17x256", 17, 256, 8, q8_tile) != 0 ||
      run_q8_mma_case("q8_0_mma_9x17x256", 17, 256, 9, q8_tile) != 0 ||
      run_q8_mma_case("q8_0_mma_64x17x256", 17, 256, 64, q8_tile) != 0 ||
      run_q8_mma_case("q8_0_mma_65x17x256", 17, 256, 65, q8_tile) != 0 ||
      run_q8_mma_case("q8_0_mma_8x1024x5120", 1024, 5120, 8, q8_tile) != 0 ||
      run_q8_mma_case("q8_0_mma_9x1024x5120", 1024, 5120, 9, q8_tile) != 0 ||
      run_q8_mma_case("q8_0_mma_64x1024x5120", 1024, 5120, 64, q8_tile) != 0 ||
      run_q8_mma_case("q8_0_mma_65x1024x5120", 1024, 5120, 65, q8_tile) != 0 ||
      run_q8_mma_case("q8_0_mma_9x48x256", 48, 256, 9, q8_tile) != 0) {
    return 1;
  }
  std::printf("q8_mma_admission_reference=launch_quant_mmq_variant_kQ8_0 "
              "q8_mma_bf16_delta=informational_only\n");

  unsigned int q8_winner = 128;
  float q8_best = 1.0e30F;
  const unsigned int q8_tiles[3] = {32U, 64U, 128U};
  const std::size_t q8_shapes[3][2] = {
      {12288, 5120}, {10240, 5120}, {5120, 6144}};
  for (unsigned int tile : q8_tiles) {
    const int occ = qw38::cuda::q8_mma_mmq_occupancy(tile);
    if (occ < 1) {
      std::printf("q8_mma_tune_skip tile=%u occupancy=%d\n", tile, occ);
      continue;
    }
    if (run_q8_mma_case("q8_0_mma_elig_8x17x256", 17, 256, 8, tile) != 0) {
      std::printf("q8_mma_tune_skip tile=%u reason=cud002_gpu_staged\n", tile);
      continue;
    }
    float mean = 0.0F;
    bool eligible = true;
    for (const auto& shape : q8_shapes) {
      float part = 0.0F;
      int shape_occ = 0;
      std::size_t nonfinite = 0;
      if (time_q8_mma_tile(shape[0], shape[1], 2048, tile, &part, &shape_occ,
                           &nonfinite) != 0 ||
          shape_occ < 1 || nonfinite != 0) {
        eligible = false;
        break;
      }
      mean += part;
    }
    std::printf("q8_mma_tune_shape_mean tile=%u mean_ms=%.9g occupancy=%d "
                "eligible=%s\n",
                tile, mean, occ, eligible ? "true" : "false");
    if (eligible && mean < q8_best) {
      q8_best = mean;
      q8_winner = tile;
    }
  }
  std::printf("q8_mma_prompt_tile_2048=%u mean_ms=%.9g pinned=%u occupancy=%d\n",
              q8_winner, q8_best, q8_tile,
              qw38::cuda::q8_mma_mmq_occupancy(q8_winner));
  if (q8_winner != q8_tile) {
    std::fprintf(stderr,
                 "selected_q8_mma_mmq_prompt_tile=%u must match mixer sweep "
                 "winner %u\n",
                 q8_tile, q8_winner);
    return 1;
  }

  float mma_speed = 0.0F;
  float variant_speed = 0.0F;
  int speed_occ = 0;
  std::size_t speed_nonfinite = 0;
  const unsigned variant_tile = qw38::cuda::selected_mmq_prompt_tile(
      qw38::cuda::QuantKind::kQ8_0, 2048);
  if (time_q8_mma_tile(12288, 5120, 2048, q8_winner, &mma_speed, &speed_occ,
                       &speed_nonfinite) != 0 ||
      time_q8_variant_tile(12288, 5120, 2048, variant_tile, &variant_speed) !=
          0 ||
      speed_nonfinite != 0 || !(mma_speed < variant_speed)) {
    std::fprintf(stderr,
                 "Q8 MMA speed gate failed mma_ms=%.9g variant_ms=%.9g "
                 "nonfinite=%zu occupancy=%d\n",
                 mma_speed, variant_speed, speed_nonfinite, speed_occ);
    return 1;
  }
  std::printf("q8_mma_speed_2048_12288x5120 mma_ms=%.9g variant_ms=%.9g "
              "occupancy=%d\n",
              mma_speed, variant_speed, speed_occ);
  std::printf("status=passed\n");
  return 0;
}
