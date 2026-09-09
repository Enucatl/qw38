#include "quant_mmv.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>
#include <vector>

#include <cuda_fp16.h>

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

float unit_normal(std::uint32_t index, std::uint32_t seed) {
  std::uint32_t x = index * 747796405U + seed;
  x ^= x >> 16U;
  x *= 0x7feb352dU;
  x ^= x >> 15U;
  x *= 0x846ca68bU;
  x ^= x >> 16U;
  const float u1 =
      (static_cast<float>(x & 0x00FFFFFFU) + 1.0F) / 16777217.0F;
  x = x * 1664525U + 1013904223U;
  const float u2 =
      static_cast<float>(x & 0x00FFFFFFU) / 16777216.0F;
  return std::sqrt(-2.0F * std::log(u1)) *
         std::cos(6.283185307179586F * u2);
}

std::uint16_t float_to_fp16_bits(float value) {
  const __half half = __float2half_rn(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

void quantize_row_q8_0(const float* src, std::uint8_t* dst, std::size_t n) {
  for (std::size_t block = 0; block < n / 32; ++block) {
    float amax = 0.0F;
    for (int lane = 0; lane < 32; ++lane) {
      amax = std::max(
          amax, std::fabs(src[block * 32 + static_cast<std::size_t>(lane)]));
    }
    const float scale = amax / 127.0F;
    write_u16(dst, float_to_fp16_bits(scale));
    for (int lane = 0; lane < 32; ++lane) {
      int quant = 0;
      if (scale != 0.0F) {
        quant = static_cast<int>(std::round(
            src[block * 32 + static_cast<std::size_t>(lane)] / scale));
        if (quant > 127) quant = 127;
        if (quant < -127) quant = -127;
      }
      dst[2 + lane] =
          static_cast<std::uint8_t>(static_cast<std::int8_t>(quant));
    }
    dst += 34;
  }
}

void fill_q8_quality_weights(std::size_t rows, std::size_t columns,
                             std::vector<std::uint8_t>* weights) {
  weights->assign(rows * (columns / 32) * 34, 0);
  std::vector<float> row(columns);
  for (std::size_t out = 0; out < rows; ++out) {
    for (std::size_t column = 0; column < columns; ++column) {
      row[column] = unit_normal(
          static_cast<std::uint32_t>(out * columns + column), 0xC0FFEEU);
    }
    quantize_row_q8_0(row.data(),
                      weights->data() + out * (columns / 32) * 34, columns);
  }
}

void make_q8_quality_activation(std::size_t columns,
                                std::vector<__nv_bfloat16>* activation,
                                std::size_t prompt_row) {
  activation->resize(columns);
  for (std::size_t index = 0; index < columns; ++index) {
    (*activation)[index] = __float2bfloat16_rn(unit_normal(
        static_cast<std::uint32_t>(prompt_row * columns + index), 0xA5A5A5U));
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

// ds4 Q8_0 check_close: fail only when both abs and rel exceed (abs scale 0.05).
bool ds4_q8_association_ok(const std::vector<float>& got,
                           const std::vector<float>& ref, std::size_t columns,
                           float* max_abs, float* max_rel, float* rms,
                           std::size_t* association_bad,
                           std::size_t* nonfinite) {
  constexpr float kAbsScale = 0.05F;
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

int run_packed_byte_equal(qw38::cuda::QuantKind kind, const char* name,
                          std::size_t rows, std::size_t columns) {
  std::vector<std::uint8_t> weights;
  fill_weights(kind, rows, columns, &weights);
  std::vector<__nv_bfloat16> activation(columns);
  for (std::size_t column = 0; column < columns; ++column) {
    activation[column] = __float2bfloat16_rn(
        unit_normal(static_cast<std::uint32_t>(column), 0xC0FFEEu));
  }
  std::vector<qw38::cuda::Q8Block> staged(
      columns / 32, qw38::cuda::Q8Block{0.0F, {}});
  std::uint8_t* device_weights = nullptr;
  __nv_bfloat16* device_activation = nullptr;
  qw38::cuda::Q8Block* device_staged_a = nullptr;
  qw38::cuda::Q8Block* device_staged_b = nullptr;
  float* device_elementwise = nullptr;
  float* device_packed = nullptr;
  cudaError_t error = cudaMalloc(&device_weights, weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_activation,
                       activation.size() * sizeof(activation[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_staged_a,
                       staged.size() * sizeof(qw38::cuda::Q8Block));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_staged_b,
                       staged.size() * sizeof(qw38::cuda::Q8Block));
  }
  if (error == cudaSuccess) error = cudaMalloc(&device_elementwise, rows * 4);
  if (error == cudaSuccess) error = cudaMalloc(&device_packed, rows * 4);
  if (error != cudaSuccess) return fail_cuda("packed memcmp cudaMalloc", error);
  error = cudaMemcpy(device_weights, weights.data(), weights.size(),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_activation, activation.data(),
                       activation.size() * sizeof(activation[0]),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmv_path(
        kind, device_weights, rows, columns, device_activation,
        device_staged_a, device_elementwise, "elementwise", nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmv_path(
        kind, device_weights, rows, columns, device_activation,
        device_staged_b, device_packed, "packed", nullptr);
  }
  if (error != cudaSuccess) return fail_cuda("packed memcmp launch", error);
  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("packed memcmp sync", error);
  std::vector<float> elementwise(rows);
  std::vector<float> packed(rows);
  std::vector<qw38::cuda::Q8Block> staged_a(staged.size());
  std::vector<qw38::cuda::Q8Block> staged_b(staged.size());
  error = cudaMemcpy(elementwise.data(), device_elementwise, rows * 4,
                     cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(packed.data(), device_packed, rows * 4,
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(staged_a.data(), device_staged_a,
                       staged.size() * sizeof(staged[0]),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(staged_b.data(), device_staged_b,
                       staged.size() * sizeof(staged[0]),
                       cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("packed memcmp D2H", error);
  const bool output_eq =
      std::memcmp(elementwise.data(), packed.data(), rows * 4) == 0;
  const bool q8_eq =
      std::memcmp(staged_a.data(), staged_b.data(),
                  staged.size() * sizeof(staged[0])) == 0;
  std::printf("packed_byte_equal case=%s rows=%zu columns=%zu output_eq=%s "
              "q8_eq=%s occupancy=%d\n",
              name, rows, columns, output_eq ? "true" : "false",
              q8_eq ? "true" : "false",
              qw38::cuda::mmv_packed_occupancy(
                  kind, qw38::cuda::selected_mmv_warps(rows)));
  cudaFree(device_packed);
  cudaFree(device_elementwise);
  cudaFree(device_staged_b);
  cudaFree(device_staged_a);
  cudaFree(device_activation);
  cudaFree(device_weights);
  return output_eq && q8_eq ? 0 : 1;
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

int run_q8_quality_case(const char* name, std::size_t output_rows,
                        std::size_t columns, std::size_t prompt_rows) {
  std::vector<std::uint8_t> weights;
  fill_q8_quality_weights(output_rows, columns, &weights);
  std::vector<__nv_bfloat16> prompt;
  prompt.resize(prompt_rows * columns);
  for (std::size_t prompt_row = 0; prompt_row < prompt_rows; ++prompt_row) {
    std::vector<__nv_bfloat16> row_activation;
    make_q8_quality_activation(columns, &row_activation, prompt_row);
    std::copy(row_activation.begin(), row_activation.end(),
              prompt.begin() + prompt_row * columns);
  }
  std::vector<float> expected;
  if (!reference_dequant_gemm(qw38::cuda::QuantKind::kQ8_0, weights, output_rows,
                              columns, prompt, prompt_rows, &expected)) {
    return 1;
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
    error = cudaMalloc(
        &device_workspace,
        qw38::cuda::q8_prompt_workspace_bytes(prompt_rows, columns));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_output, expected.size() * sizeof(expected[0]));
  }
  if (error != cudaSuccess) return fail_cuda("Q8 quality cudaMalloc", error);
  error = cudaMemcpy(device_weights, weights.data(), weights.size(),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_prompt, prompt.data(),
                       prompt.size() * sizeof(prompt[0]), cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("Q8 quality H2D", error);
  error = qw38::cuda::launch_q8_mmq_quality(
      device_weights, output_rows, columns, device_prompt, prompt_rows,
      device_workspace, device_output, nullptr);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("Q8 quality launch", error);

  std::vector<float> actual(expected.size());
  error = cudaMemcpy(actual.data(), device_output,
                     actual.size() * sizeof(actual[0]), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return fail_cuda("Q8 quality D2H", error);

  float max_abs = 0.0F;
  float max_rel = 0.0F;
  float rms = 0.0F;
  std::size_t bad = 0;
  std::size_t nonfinite = 0;
  const bool ok = ds4_q8_association_ok(actual, expected, columns, &max_abs,
                                        &max_rel, &rms, &bad, &nonfinite);
  std::printf("q8_quality_case=%s prompt_rows=%zu output_rows=%zu columns=%zu "
              "mma_max_abs=%.9g mma_max_rel=%.9g mma_rms=%.9g mma_bad=%zu "
              "mma_nonfinite=%zu gate=ds4_q8_association ref=cpu_dequant_gemm "
              "tile=%u occupancy=%d\n",
              name, prompt_rows, output_rows, columns, max_abs, max_rel, rms,
              bad, nonfinite, qw38::cuda::selected_q8_quality_mmq_prompt_tile(),
              qw38::cuda::q8_quality_mmq_occupancy(
                  qw38::cuda::selected_q8_quality_mmq_prompt_tile()));
  cudaFree(device_output);
  cudaFree(device_workspace);
  cudaFree(device_prompt);
  cudaFree(device_weights);
  return ok ? 0 : 1;
}

int run_q8_shared_y_identity() {
  constexpr std::size_t kPrompt = 8;
  constexpr std::size_t kOut = 1024;
  constexpr std::size_t kCols = 5120;
  std::vector<std::uint8_t> weights_a;
  std::vector<std::uint8_t> weights_b;
  fill_weights(qw38::cuda::QuantKind::kQ8_0, kOut, kCols, &weights_a);
  fill_weights(qw38::cuda::QuantKind::kQ8_0, kOut, kCols, &weights_b);
  for (std::size_t index = 0; index < weights_b.size(); ++index) {
    weights_b[index] = static_cast<std::uint8_t>(weights_b[index] ^ 0x5AU);
  }
  for (std::size_t offset = 0; offset < weights_b.size(); offset += 34) {
    write_u16(weights_b.data() + offset, 0x3400U);
    for (std::size_t lane = 0; lane < 32; ++lane) {
      weights_b[offset + 2 + lane] = static_cast<std::uint8_t>(
          static_cast<std::int8_t>(static_cast<int>((lane * 7 + offset) % 63) -
                                   31));
    }
  }
  std::vector<__nv_bfloat16> prompt(kPrompt * kCols);
  for (std::size_t prompt_row = 0; prompt_row < kPrompt; ++prompt_row) {
    std::vector<__nv_bfloat16> row_activation;
    std::vector<qw38::cuda::Q8Block> row_staged;
    make_activation(kCols, &row_activation, &row_staged, prompt_row);
    std::copy(row_activation.begin(), row_activation.end(),
              prompt.begin() + prompt_row * kCols);
  }

  std::uint8_t* device_a = nullptr;
  std::uint8_t* device_b = nullptr;
  __nv_bfloat16* device_prompt = nullptr;
  qw38::cuda::Q8Block* shared_y = nullptr;
  qw38::cuda::Q8Block* independent_y = nullptr;
  float* shared_out_a = nullptr;
  float* shared_out_b = nullptr;
  float* independent_out_a = nullptr;
  float* independent_out_b = nullptr;
  const std::size_t bytes = kPrompt * kOut * sizeof(float);
  cudaError_t error = cudaMalloc(&device_a, weights_a.size());
  if (error == cudaSuccess) error = cudaMalloc(&device_b, weights_b.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_prompt, prompt.size() * sizeof(prompt[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&shared_y,
                       qw38::cuda::q8_prompt_workspace_bytes(kPrompt, kCols));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(
        &independent_y, qw38::cuda::q8_prompt_workspace_bytes(kPrompt, kCols));
  }
  if (error == cudaSuccess) error = cudaMalloc(&shared_out_a, bytes);
  if (error == cudaSuccess) error = cudaMalloc(&shared_out_b, bytes);
  if (error == cudaSuccess) error = cudaMalloc(&independent_out_a, bytes);
  if (error == cudaSuccess) error = cudaMalloc(&independent_out_b, bytes);
  if (error != cudaSuccess) return fail_cuda("shared-Y cudaMalloc", error);
  error = cudaMemcpy(device_a, weights_a.data(), weights_a.size(),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_b, weights_b.data(), weights_b.size(),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_prompt, prompt.data(),
                       prompt.size() * sizeof(prompt[0]),
                       cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("shared-Y H2D", error);

  error = qw38::cuda::launch_quantize_mmq_q8_1(
      qw38::cuda::QuantKind::kQ8_0, device_prompt, kPrompt, kCols, shared_y,
      nullptr);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q8_mmq_quality_mma(
        device_a, kOut, kCols, shared_y, kPrompt, shared_out_a, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q8_mmq_quality_mma(
        device_b, kOut, kCols, shared_y, kPrompt, shared_out_b, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q8_mmq_quality(device_a, kOut, kCols,
                                              device_prompt, kPrompt,
                                              independent_y, independent_out_a,
                                              nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q8_mmq_quality(device_b, kOut, kCols,
                                              device_prompt, kPrompt,
                                              independent_y, independent_out_b,
                                              nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("shared-Y launch", error);

  std::vector<float> shared_a(kPrompt * kOut);
  std::vector<float> shared_b(kPrompt * kOut);
  std::vector<float> independent_a(kPrompt * kOut);
  std::vector<float> independent_b(kPrompt * kOut);
  error = cudaMemcpy(shared_a.data(), shared_out_a, bytes,
                     cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(shared_b.data(), shared_out_b, bytes,
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(independent_a.data(), independent_out_a, bytes,
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(independent_b.data(), independent_out_b, bytes,
                       cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("shared-Y D2H", error);

  const bool identical =
      shared_a == independent_a && shared_b == independent_b;
  std::printf("q8_shared_y_identity prompt_rows=%zu output_rows=%zu columns=%zu "
              "byte_identical=%s\n",
              kPrompt, kOut, kCols, identical ? "true" : "false");
  cudaFree(independent_out_b);
  cudaFree(independent_out_a);
  cudaFree(shared_out_b);
  cudaFree(shared_out_a);
  cudaFree(independent_y);
  cudaFree(shared_y);
  cudaFree(device_prompt);
  cudaFree(device_b);
  cudaFree(device_a);
  return identical ? 0 : 1;
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

int run_q8_quality_i_case(const char* name, std::size_t output_rows,
                          std::size_t columns, std::size_t prompt_rows,
                          unsigned int quality_i) {
  std::vector<std::uint8_t> weights;
  fill_q8_quality_weights(output_rows, columns, &weights);
  std::vector<__nv_bfloat16> prompt;
  prompt.resize(prompt_rows * columns);
  for (std::size_t prompt_row = 0; prompt_row < prompt_rows; ++prompt_row) {
    std::vector<__nv_bfloat16> row_activation;
    make_q8_quality_activation(columns, &row_activation, prompt_row);
    std::copy(row_activation.begin(), row_activation.end(),
              prompt.begin() + prompt_row * columns);
  }
  std::vector<float> expected;
  if (!reference_dequant_gemm(qw38::cuda::QuantKind::kQ8_0, weights, output_rows,
                              columns, prompt, prompt_rows, &expected)) {
    return 1;
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
    error = cudaMalloc(
        &device_workspace,
        qw38::cuda::q8_prompt_workspace_bytes(prompt_rows, columns));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_output, expected.size() * sizeof(expected[0]));
  }
  if (error != cudaSuccess) return fail_cuda("Q8 quality-i cudaMalloc", error);
  error = cudaMemcpy(device_weights, weights.data(), weights.size(),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_prompt, prompt.data(),
                       prompt.size() * sizeof(prompt[0]), cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("Q8 quality-i H2D", error);
  error = qw38::cuda::launch_quantize_mmq_q8_1(
      qw38::cuda::QuantKind::kQ8_0, device_prompt, prompt_rows, columns,
      device_workspace, nullptr);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q8_mmq_quality_mma_i(
        device_weights, output_rows, columns, device_workspace, prompt_rows,
        device_output, quality_i, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("Q8 quality-i launch", error);

  std::vector<float> actual(expected.size());
  error = cudaMemcpy(actual.data(), device_output,
                     actual.size() * sizeof(actual[0]), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return fail_cuda("Q8 quality-i D2H", error);

  float max_abs = 0.0F;
  float max_rel = 0.0F;
  float rms = 0.0F;
  std::size_t bad = 0;
  std::size_t nonfinite = 0;
  const bool ok = ds4_q8_association_ok(actual, expected, columns, &max_abs,
                                        &max_rel, &rms, &bad, &nonfinite);
  std::printf("q8_quality_i_case=%s prompt_rows=%zu output_rows=%zu columns=%zu "
              "quality_i=%u mma_max_abs=%.9g mma_max_rel=%.9g mma_rms=%.9g "
              "mma_bad=%zu mma_nonfinite=%zu gate=ds4_q8_association "
              "ref=cpu_dequant_gemm occupancy=%d\n",
              name, prompt_rows, output_rows, columns, quality_i, max_abs,
              max_rel, rms, bad, nonfinite,
              qw38::cuda::q8_quality_mmq_occupancy_i(128, quality_i));
  cudaFree(device_output);
  cudaFree(device_workspace);
  cudaFree(device_prompt);
  cudaFree(device_weights);
  return ok ? 0 : 1;
}

int run_mmv_tiled_j1_exact() {
  constexpr std::size_t kPrompt = 8;
  constexpr std::size_t kOut = 48;
  constexpr std::size_t kCols = 5120;
  std::vector<std::uint8_t> weights;
  fill_weights(qw38::cuda::QuantKind::kQ8_0, kOut, kCols, &weights);
  std::vector<__nv_bfloat16> prompt(kPrompt * kCols);
  for (std::size_t prompt_row = 0; prompt_row < kPrompt; ++prompt_row) {
    std::vector<__nv_bfloat16> row_activation;
    std::vector<qw38::cuda::Q8Block> staged;
    make_activation(kCols, &row_activation, &staged, prompt_row);
    std::copy(row_activation.begin(), row_activation.end(),
              prompt.begin() + prompt_row * kCols);
  }
  std::uint8_t* device_weights = nullptr;
  __nv_bfloat16* device_prompt = nullptr;
  float* device_variant = nullptr;
  float* device_reference = nullptr;
  const std::size_t bytes = kPrompt * kOut * sizeof(float);
  cudaError_t error = cudaMalloc(&device_weights, weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_prompt, prompt.size() * sizeof(prompt[0]));
  }
  if (error == cudaSuccess) error = cudaMalloc(&device_variant, bytes);
  if (error == cudaSuccess) error = cudaMalloc(&device_reference, bytes);
  if (error != cudaSuccess) return fail_cuda("MMV exact cudaMalloc", error);
  error = cudaMemcpy(device_weights, weights.data(), weights.size(),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_prompt, prompt.data(),
                       prompt.size() * sizeof(prompt[0]),
                       cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("MMV exact H2D", error);
  error = qw38::cuda::launch_q8_mmq_bf16_variant(
      device_weights, kOut, kCols, device_prompt, kPrompt, device_variant, 1,
      nullptr);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q8_mmq_bf16_reference(
        device_weights, kOut, kCols, device_prompt, kPrompt, device_reference,
        nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("MMV exact launch", error);
  std::vector<float> variant(kPrompt * kOut);
  std::vector<float> reference(kPrompt * kOut);
  error = cudaMemcpy(variant.data(), device_variant, bytes,
                     cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(reference.data(), device_reference, bytes,
                       cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("MMV exact D2H", error);
  const bool equal =
      std::memcmp(variant.data(), reference.data(), bytes) == 0;
  std::printf("mmv_tiled_j1_exact prompt_rows=%zu output_rows=%zu columns=%zu "
              "byte_identical=%s\n",
              kPrompt, kOut, kCols, equal ? "true" : "false");
  cudaFree(device_reference);
  cudaFree(device_variant);
  cudaFree(device_prompt);
  cudaFree(device_weights);
  return equal ? 0 : 1;
}

void ensure_opt023_evidence_dir() {
  mkdir("evidence", 0755);
  mkdir("evidence/optimization", 0755);
  mkdir("evidence/optimization/opt023-skinny-mixer", 0755);
}

int run_skinny_ab() {
  constexpr std::size_t kPrompt = 4096;
  constexpr std::size_t kOut = 48;
  constexpr std::size_t kCols = 5120;
  if (qw38::cuda::q8_quality_mmq_occupancy_i(128, 32) < 1 ||
      qw38::cuda::q8_quality_mmq_occupancy_i(128, 64) < 1 ||
      qw38::cuda::q8_quality_mmq_occupancy_i(128, 128) < 1) {
    std::fprintf(stderr, "skinny MMA occupancy < 1\n");
    return 1;
  }
  std::vector<std::uint8_t> weights;
  fill_q8_quality_weights(kOut, kCols, &weights);
  std::vector<__nv_bfloat16> prompt(kPrompt * kCols);
  for (std::size_t prompt_row = 0; prompt_row < kPrompt; ++prompt_row) {
    std::vector<__nv_bfloat16> row_activation;
    make_q8_quality_activation(kCols, &row_activation, prompt_row);
    std::copy(row_activation.begin(), row_activation.end(),
              prompt.begin() + prompt_row * kCols);
  }
  std::uint8_t* device_weights = nullptr;
  __nv_bfloat16* device_prompt = nullptr;
  qw38::cuda::Q8Block* device_y = nullptr;
  float* device_output = nullptr;
  const std::size_t out_bytes = kPrompt * kOut * sizeof(float);
  cudaError_t error = cudaMalloc(&device_weights, weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_prompt, prompt.size() * sizeof(prompt[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_y,
                       qw38::cuda::q8_prompt_workspace_bytes(kPrompt, kCols));
  }
  if (error == cudaSuccess) error = cudaMalloc(&device_output, out_bytes);
  if (error != cudaSuccess) return fail_cuda("skinny A/B cudaMalloc", error);
  error = cudaMemcpy(device_weights, weights.data(), weights.size(),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_prompt, prompt.data(),
                       prompt.size() * sizeof(prompt[0]),
                       cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("skinny A/B H2D", error);
  error = qw38::cuda::launch_quantize_mmq_q8_1(
      qw38::cuda::QuantKind::kQ8_0, device_prompt, kPrompt, kCols, device_y,
      nullptr);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("skinny A/B quantize Y", error);

  struct Candidate {
    const char* id;
    unsigned int quality_i;
    bool mmv;
  };
  const Candidate candidates[] = {
      {"mma_i128_j128", 128, false},
      {"mma_i32_j128", 32, false},
      {"mma_i64_j128", 64, false},
      {"mmv_tiled_j1", 0, true},
  };

  ensure_opt023_evidence_dir();
  FILE* raw = std::fopen(
      "evidence/optimization/opt023-skinny-mixer/skinny-ab-raw.txt", "w");
  if (raw == nullptr) {
    std::fprintf(stderr, "skinny A/B fopen failed\n");
    return 1;
  }

  float means[4]{};
  int occupancies[4]{};
  std::size_t nonfinites[4]{};
  bool eligible[4]{};
  std::vector<float> host(kPrompt * kOut);

  for (int c = 0; c < 4; ++c) {
    const Candidate& cand = candidates[c];
    occupancies[c] =
        cand.mmv ? 1 : qw38::cuda::q8_quality_mmq_occupancy_i(128, cand.quality_i);
    eligible[c] = occupancies[c] >= 1;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    error = cudaEventCreate(&start);
    if (error == cudaSuccess) error = cudaEventCreate(&stop);
    float total = 0.0F;
    for (int sample = 0; sample < 3 && error == cudaSuccess; ++sample) {
      error = cudaEventRecord(start);
      if (error == cudaSuccess) {
        if (cand.mmv) {
          error = qw38::cuda::launch_q8_mmq_bf16_variant(
              device_weights, kOut, kCols, device_prompt, kPrompt,
              device_output, 1, nullptr);
        } else {
          error = qw38::cuda::launch_q8_mmq_quality_mma_i(
              device_weights, kOut, kCols, device_y, kPrompt, device_output,
              cand.quality_i, nullptr);
        }
      }
      if (error == cudaSuccess) error = cudaEventRecord(stop);
      if (error == cudaSuccess) error = cudaEventSynchronize(stop);
      float milliseconds = 0.0F;
      if (error == cudaSuccess) {
        error = cudaEventElapsedTime(&milliseconds, start, stop);
      }
      if (error == cudaSuccess) {
        total += milliseconds;
        std::fprintf(raw,
                     "skinny_ab id=%s sample=%d ms=%.9g occupancy=%d\n",
                     cand.id, sample, milliseconds, occupancies[c]);
        std::printf("skinny_ab id=%s sample=%d ms=%.9g occupancy=%d\n", cand.id,
                    sample, milliseconds, occupancies[c]);
      }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("skinny A/B time", error);
    }
    means[c] = total / 3.0F;
    error = cudaMemcpy(host.data(), device_output, out_bytes,
                       cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("skinny A/B D2H", error);
    }
    nonfinites[c] = 0;
    for (float value : host) {
      if (!std::isfinite(value)) ++nonfinites[c];
    }
    eligible[c] = eligible[c] && nonfinites[c] == 0 && error == cudaSuccess;
    std::fprintf(raw,
                 "skinny_ab_mean id=%s mean_ms=%.9g occupancy=%d nonfinite=%zu "
                 "eligible=%s\n",
                 cand.id, means[c], occupancies[c], nonfinites[c],
                 eligible[c] ? "true" : "false");
    std::printf("skinny_ab_mean id=%s mean_ms=%.9g occupancy=%d nonfinite=%zu "
                "eligible=%s\n",
                cand.id, means[c], occupancies[c], nonfinites[c],
                eligible[c] ? "true" : "false");
  }

  const bool baseline_ok = eligible[0];
  int winner = 0;
  if (baseline_ok) {
    for (int c = 1; c < 4; ++c) {
      if (eligible[c] && means[c] < means[winner]) winner = c;
    }
    if (winner != 0 && !(means[winner] < means[0])) winner = 0;
    if (winner != 0 && !(eligible[winner] && means[winner] < means[0])) {
      winner = 0;
    }
  }
  const bool ab_win = baseline_ok && winner != 0 && means[winner] < means[0];
  std::fprintf(raw,
               "skinny_ab_winner id=%s mean_ms=%.9g baseline_id=mma_i128_j128 "
               "baseline_ms=%.9g win=%s\n",
               candidates[winner].id, means[winner], means[0],
               ab_win ? "true" : "false");
  std::printf("skinny_ab_winner id=%s mean_ms=%.9g baseline_id=mma_i128_j128 "
              "baseline_ms=%.9g win=%s selected_path=%s\n",
              candidates[winner].id, means[winner], means[0],
              ab_win ? "true" : "false",
              qw38::cuda::selected_skinny_mixer_path());
  std::fclose(raw);
  cudaFree(device_output);
  cudaFree(device_y);
  cudaFree(device_prompt);
  cudaFree(device_weights);
  return baseline_ok ? 0 : 1;
}

__global__ void test_swiglu_bf16(const float* gate, const float* up,
                                 std::size_t count, __nv_bfloat16* output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    const float g = gate[index];
    output[index] = __float2bfloat16_rn((g / (1.0F + expf(-g))) * up[index]);
  }
}

cudaError_t launch_test_swiglu_bf16(const float* gate, const float* up,
                                    std::size_t count, __nv_bfloat16* output) {
  const unsigned int blocks =
      static_cast<unsigned int>((count + 255) / 256);
  test_swiglu_bf16<<<blocks, 256>>>(gate, up, count, output);
  return cudaPeekAtLastError();
}

int run_ffn_shared_y_identity() {
  constexpr std::size_t kPrompt = 64;
  constexpr std::size_t kHidden = 5120;
  constexpr std::size_t kFfn = 17408;
  std::vector<std::uint8_t> gate_w;
  std::vector<std::uint8_t> up_w;
  fill_weights(qw38::cuda::QuantKind::kQ4K, kFfn, kHidden, &gate_w);
  fill_weights(qw38::cuda::QuantKind::kQ4K, kFfn, kHidden, &up_w);
  std::vector<__nv_bfloat16> prompt(kPrompt * kHidden);
  for (std::size_t row = 0; row < kPrompt; ++row) {
    for (std::size_t col = 0; col < kHidden; ++col) {
      prompt[row * kHidden + col] = __float2bfloat16_rn(unit_normal(
          static_cast<std::uint32_t>(row * kHidden + col), 0xA5A5A5U));
    }
  }
  const std::size_t y_bytes =
      qw38::cuda::q8_prompt_workspace_bytes(kPrompt, kHidden);
  const std::size_t out_bytes = kPrompt * kFfn * sizeof(float);
  std::uint8_t* device_gate = nullptr;
  std::uint8_t* device_up = nullptr;
  __nv_bfloat16* device_prompt = nullptr;
  qw38::cuda::Q8Block* device_y_a = nullptr;
  qw38::cuda::Q8Block* device_y_b = nullptr;
  float* device_gate_a = nullptr;
  float* device_gate_b = nullptr;
  float* device_up_a = nullptr;
  float* device_up_b = nullptr;
  cudaError_t error = cudaMalloc(&device_gate, gate_w.size());
  if (error == cudaSuccess) error = cudaMalloc(&device_up, up_w.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_prompt, prompt.size() * sizeof(prompt[0]));
  }
  if (error == cudaSuccess) error = cudaMalloc(&device_y_a, y_bytes);
  if (error == cudaSuccess) error = cudaMalloc(&device_y_b, y_bytes);
  if (error == cudaSuccess) error = cudaMalloc(&device_gate_a, out_bytes);
  if (error == cudaSuccess) error = cudaMalloc(&device_gate_b, out_bytes);
  if (error == cudaSuccess) error = cudaMalloc(&device_up_a, out_bytes);
  if (error == cudaSuccess) error = cudaMalloc(&device_up_b, out_bytes);
  if (error != cudaSuccess) return fail_cuda("FFN shared-Y cudaMalloc", error);
  error = cudaMemcpy(device_gate, gate_w.data(), gate_w.size(),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_up, up_w.data(), up_w.size(),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_prompt, prompt.data(),
                       prompt.size() * sizeof(prompt[0]),
                       cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("FFN shared-Y H2D", error);
  error = qw38::cuda::launch_quant_mmq_mma(
      qw38::cuda::QuantKind::kQ4K, device_gate, kFfn, kHidden, device_prompt,
      kPrompt, device_y_a, device_gate_a, nullptr);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmq_mma(
        qw38::cuda::QuantKind::kQ4K, device_up, kFfn, kHidden, device_prompt,
        kPrompt, device_y_a, device_up_a, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_mmq_q8_1(
        qw38::cuda::QuantKind::kQ4K, device_prompt, kPrompt, kHidden, device_y_b,
        nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmq_mma_y(
        qw38::cuda::QuantKind::kQ4K, device_gate, kFfn, kHidden, device_y_b,
        kPrompt, device_gate_b, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmq_mma_y(
        qw38::cuda::QuantKind::kQ4K, device_up, kFfn, kHidden, device_y_b,
        kPrompt, device_up_b, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("FFN shared-Y launch", error);
  std::vector<float> gate_a(kPrompt * kFfn);
  std::vector<float> gate_b(kPrompt * kFfn);
  std::vector<float> up_a(kPrompt * kFfn);
  std::vector<float> up_b(kPrompt * kFfn);
  error = cudaMemcpy(gate_a.data(), device_gate_a, out_bytes,
                     cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(gate_b.data(), device_gate_b, out_bytes,
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(up_a.data(), device_up_a, out_bytes,
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(up_b.data(), device_up_b, out_bytes,
                       cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("FFN shared-Y D2H", error);
  const bool gate_eq =
      std::memcmp(gate_a.data(), gate_b.data(), out_bytes) == 0;
  const bool up_eq = std::memcmp(up_a.data(), up_b.data(), out_bytes) == 0;
  std::printf("ffn_shared_y_identity prompt_rows=%zu output_rows=%zu columns=%zu "
              "gate_byte_identical=%s up_byte_identical=%s\n",
              kPrompt, kFfn, kHidden, gate_eq ? "true" : "false",
              up_eq ? "true" : "false");
  cudaFree(device_up_b);
  cudaFree(device_up_a);
  cudaFree(device_gate_b);
  cudaFree(device_gate_a);
  cudaFree(device_y_b);
  cudaFree(device_y_a);
  cudaFree(device_prompt);
  cudaFree(device_up);
  cudaFree(device_gate);
  return gate_eq && up_eq ? 0 : 1;
}

int run_ffn_swiglu_q8_identity() {
  constexpr std::size_t kPrompt = 64;
  constexpr std::size_t kFfn = 17408;
  constexpr std::size_t kHidden = 5120;
  std::vector<float> gate(kPrompt * kFfn);
  std::vector<float> up(kPrompt * kFfn);
  for (std::size_t index = 0; index < gate.size(); ++index) {
    gate[index] = unit_normal(static_cast<std::uint32_t>(index), 0x111111U);
    up[index] = unit_normal(static_cast<std::uint32_t>(index), 0x222222U);
  }
  std::vector<std::uint8_t> down_w;
  fill_weights(qw38::cuda::QuantKind::kQ4K, kHidden, kFfn, &down_w);
  const std::size_t y_bytes =
      qw38::cuda::q8_prompt_workspace_bytes(kPrompt, kFfn);
  const std::size_t mid_bytes = kPrompt * kFfn * sizeof(__nv_bfloat16);
  const std::size_t out_bytes = kPrompt * kHidden * sizeof(float);
  float* device_gate = nullptr;
  float* device_up = nullptr;
  __nv_bfloat16* device_mid = nullptr;
  qw38::cuda::Q8Block* device_y_ref = nullptr;
  qw38::cuda::Q8Block* device_y_fused = nullptr;
  std::uint8_t* device_down = nullptr;
  float* device_out_ref = nullptr;
  float* device_out_fused = nullptr;
  cudaError_t error = cudaMalloc(&device_gate, gate.size() * sizeof(float));
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_up, up.size() * sizeof(float));
  }
  if (error == cudaSuccess) error = cudaMalloc(&device_mid, mid_bytes);
  if (error == cudaSuccess) error = cudaMalloc(&device_y_ref, y_bytes);
  if (error == cudaSuccess) error = cudaMalloc(&device_y_fused, y_bytes);
  if (error == cudaSuccess) error = cudaMalloc(&device_down, down_w.size());
  if (error == cudaSuccess) error = cudaMalloc(&device_out_ref, out_bytes);
  if (error == cudaSuccess) error = cudaMalloc(&device_out_fused, out_bytes);
  if (error != cudaSuccess) return fail_cuda("FFN SwiGLU-Q8 cudaMalloc", error);
  error = cudaMemcpy(device_gate, gate.data(), gate.size() * sizeof(float),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_up, up.data(), up.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_down, down_w.data(), down_w.size(),
                       cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("FFN SwiGLU-Q8 H2D", error);
  error = launch_test_swiglu_bf16(device_gate, device_up, kPrompt * kFfn,
                                  device_mid);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_mmq_q8_1(
        qw38::cuda::QuantKind::kQ4K, device_mid, kPrompt, kFfn, device_y_ref,
        nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_swiglu_quantize_mmq_q8_1(
        device_gate, device_up, kPrompt, kFfn, device_y_fused, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmq_mma_y(
        qw38::cuda::QuantKind::kQ4K, device_down, kHidden, kFfn, device_y_ref,
        kPrompt, device_out_ref, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmq_mma_y(
        qw38::cuda::QuantKind::kQ4K, device_down, kHidden, kFfn, device_y_fused,
        kPrompt, device_out_fused, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("FFN SwiGLU-Q8 launch", error);
  std::vector<std::uint8_t> y_ref(y_bytes);
  std::vector<std::uint8_t> y_fused(y_bytes);
  std::vector<float> out_ref(kPrompt * kHidden);
  std::vector<float> out_fused(kPrompt * kHidden);
  error = cudaMemcpy(y_ref.data(), device_y_ref, y_bytes, cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(y_fused.data(), device_y_fused, y_bytes,
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(out_ref.data(), device_out_ref, out_bytes,
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(out_fused.data(), device_out_fused, out_bytes,
                       cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("FFN SwiGLU-Q8 D2H", error);
  const bool y_eq = std::memcmp(y_ref.data(), y_fused.data(), y_bytes) == 0;
  const bool out_eq =
      std::memcmp(out_ref.data(), out_fused.data(), out_bytes) == 0;
  std::size_t nonfinite = 0;
  for (float value : out_fused) {
    if (!std::isfinite(value)) ++nonfinite;
  }
  std::printf("ffn_swiglu_q8_identity prompt_rows=%zu columns=%zu "
              "y_byte_identical=%s out_byte_identical=%s nonfinite=%zu\n",
              kPrompt, kFfn, y_eq ? "true" : "false",
              out_eq ? "true" : "false", nonfinite);
  cudaFree(device_out_fused);
  cudaFree(device_out_ref);
  cudaFree(device_down);
  cudaFree(device_y_fused);
  cudaFree(device_y_ref);
  cudaFree(device_mid);
  cudaFree(device_up);
  cudaFree(device_gate);
  return y_eq && out_eq && nonfinite == 0 ? 0 : 1;
}

cudaError_t launch_ffn_candidate(
    const char* id, const std::uint8_t* gate_w, const std::uint8_t* up_w,
    const std::uint8_t* down_w, const __nv_bfloat16* prompt,
    qw38::cuda::Q8Block* y, float* gate_out, float* up_out, float* down_out,
    __nv_bfloat16* mid, std::size_t prompt_rows) {
  constexpr std::size_t kHidden = 5120;
  constexpr std::size_t kFfn = 17408;
  const bool share = std::strcmp(id, "shared_y") == 0 ||
                     std::strcmp(id, "shared_y_swiglu_q8") == 0;
  const bool swiglu = std::strcmp(id, "swiglu_q8") == 0 ||
                      std::strcmp(id, "shared_y_swiglu_q8") == 0;
  cudaError_t error = cudaSuccess;
  if (share) {
    error = qw38::cuda::launch_quantize_mmq_q8_1(
        qw38::cuda::QuantKind::kQ4K, prompt, prompt_rows, kHidden, y, nullptr);
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_quant_mmq_mma_y(
          qw38::cuda::QuantKind::kQ4K, gate_w, kFfn, kHidden, y, prompt_rows,
          gate_out, nullptr);
    }
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_quant_mmq_mma_y(
          qw38::cuda::QuantKind::kQ4K, up_w, kFfn, kHidden, y, prompt_rows,
          up_out, nullptr);
    }
  } else {
    error = qw38::cuda::launch_quant_mmq_mma(
        qw38::cuda::QuantKind::kQ4K, gate_w, kFfn, kHidden, prompt, prompt_rows,
        y, gate_out, nullptr);
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_quant_mmq_mma(
          qw38::cuda::QuantKind::kQ4K, up_w, kFfn, kHidden, prompt, prompt_rows,
          y, up_out, nullptr);
    }
  }
  if (error != cudaSuccess) return error;
  if (swiglu) {
    error = qw38::cuda::launch_swiglu_quantize_mmq_q8_1(
        gate_out, up_out, prompt_rows, kFfn, y, nullptr);
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_quant_mmq_mma_y(
          qw38::cuda::QuantKind::kQ4K, down_w, kHidden, kFfn, y, prompt_rows,
          down_out, nullptr);
    }
    return error;
  }
  error = launch_test_swiglu_bf16(gate_out, up_out, prompt_rows * kFfn, mid);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmq_mma(
        qw38::cuda::QuantKind::kQ4K, down_w, kHidden, kFfn, mid, prompt_rows, y,
        down_out, nullptr);
  }
  return error;
}

void ensure_opt028_evidence_dir() {
  mkdir("evidence", 0755);
  mkdir("evidence/optimization", 0755);
  mkdir("evidence/optimization/opt028-mmq-streamk", 0755);
}

int run_mmq_stream_k_helpers() {
  if (std::strcmp(qw38::cuda::selected_mmq_stream_k_path(), "off") != 0 &&
      std::strcmp(qw38::cuda::selected_mmq_stream_k_path(), "stream_k") != 0 &&
      std::strcmp(qw38::cuda::selected_mmq_stream_k_path(), "stream_k_nsm") !=
          0) {
    std::fprintf(stderr, "selected_mmq_stream_k_path is not a legal enumerator\n");
    return 1;
  }
  if (std::strcmp(qw38::cuda::selected_ffn_path(), "shared_y_swiglu_q8") != 0) {
    std::fprintf(stderr, "kSelectedFfnPath must stay shared_y_swiglu_q8\n");
    return 1;
  }
  const int nsm = qw38::cuda::mmq_stream_k_nsm();
  if (nsm < 1) {
    std::fprintf(stderr, "mmq_stream_k_nsm=%d\n", nsm);
    return 1;
  }
  if (qw38::cuda::mmq_stream_k_nblocks(nsm, 4352, "stream_k") != 4352 ||
      qw38::cuda::mmq_stream_k_nblocks(nsm, 1280, "stream_k") != 1280 ||
      qw38::cuda::mmq_stream_k_nblocks(170, 136, "stream_k") != 170 ||
      qw38::cuda::mmq_stream_k_nblocks(nsm, 4352, "stream_k_nsm") != nsm ||
      qw38::cuda::mmq_stream_k_nblocks(nsm, 4352, "off") != 0 ||
      qw38::cuda::mmq_stream_k_fixup_needed(4352, 4352) ||
      !qw38::cuda::mmq_stream_k_fixup_needed(4352, 170) ||
      qw38::cuda::mmq_stream_k_fixup_floats(170, 128, 128) !=
          static_cast<std::size_t>(170) * 128U * 128U ||
      qw38::cuda::mmq_stream_k_fixup_floats(4352, 128, 128) !=
          static_cast<std::size_t>(4352) * 128U * 128U ||
      qw38::cuda::mmq_stream_k_occupancy(qw38::cuda::QuantKind::kQ4K) < 1 ||
      qw38::cuda::mmq_stream_k_occupancy(qw38::cuda::QuantKind::kQ6K) < 1 ||
      qw38::cuda::mmq_stream_k_fixup_occupancy() < 1) {
    std::fprintf(stderr, "MMQ stream-K helpers failed nsm=%d occ_q4=%d occ_q6=%d "
                         "occ_fixup=%d\n",
                 nsm,
                 qw38::cuda::mmq_stream_k_occupancy(qw38::cuda::QuantKind::kQ4K),
                 qw38::cuda::mmq_stream_k_occupancy(qw38::cuda::QuantKind::kQ6K),
                 qw38::cuda::mmq_stream_k_fixup_occupancy());
    return 1;
  }
  std::printf("mmq_stream_k_helpers nsm=%d path=%s uses=%s occ_q4=%d "
              "occ_fixup=%d\n",
              nsm, qw38::cuda::selected_mmq_stream_k_path(),
              qw38::cuda::mmq_uses_stream_k() ? "true" : "false",
              qw38::cuda::mmq_stream_k_occupancy(qw38::cuda::QuantKind::kQ4K),
              qw38::cuda::mmq_stream_k_fixup_occupancy());
  return 0;
}

int run_mma_stream_k_case(qw38::cuda::QuantKind kind, const char* name,
                          std::size_t output_rows, std::size_t columns,
                          std::size_t prompt_rows, const char* path) {
  std::vector<std::uint8_t> weights;
  fill_weights(kind, output_rows, columns, &weights);
  std::vector<__nv_bfloat16> prompt;
  prompt.resize(prompt_rows * columns);
  for (std::size_t prompt_row = 0; prompt_row < prompt_rows; ++prompt_row) {
    std::vector<__nv_bfloat16> row_activation;
    std::vector<qw38::cuda::Q8Block> row_staged;
    make_activation(columns, &row_activation, &row_staged, prompt_row);
    std::copy(row_activation.begin(), row_activation.end(),
              prompt.begin() + prompt_row * columns);
  }
  std::vector<float> expected;
  if (!reference_dequant_gemm(kind, weights, output_rows, columns, prompt,
                              prompt_rows, &expected)) {
    return 1;
  }
  const int nty = static_cast<int>((output_rows + 127) / 128);
  const int ntx = static_cast<int>((prompt_rows + 127) / 128);
  const int ntiles = ntx * nty;
  const int nsm = qw38::cuda::mmq_stream_k_nsm();
  const int nblocks = qw38::cuda::mmq_stream_k_nblocks(nsm, ntiles, path);
  const bool fixup = qw38::cuda::mmq_stream_k_fixup_needed(ntiles, nblocks);
  const std::size_t fixup_floats =
      fixup ? qw38::cuda::mmq_stream_k_fixup_floats(nblocks, 128, 128) : 0;
  std::uint8_t* device_weights = nullptr;
  __nv_bfloat16* device_prompt = nullptr;
  qw38::cuda::Q8Block* device_y = nullptr;
  float* device_out = nullptr;
  float* device_fixup = nullptr;
  cudaError_t error = cudaMalloc(&device_weights, weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_prompt, prompt.size() * sizeof(prompt[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_y, qw38::cuda::q8_prompt_workspace_bytes(
                                      prompt_rows, columns));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_out, expected.size() * sizeof(float));
  }
  if (error == cudaSuccess && fixup_floats > 0) {
    error = cudaMalloc(&device_fixup, fixup_floats * sizeof(float));
  }
  if (error != cudaSuccess) return fail_cuda("stream-K MMA cudaMalloc", error);
  error = cudaMemcpy(device_weights, weights.data(), weights.size(),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_prompt, prompt.data(),
                       prompt.size() * sizeof(prompt[0]), cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_mmq_q8_1(
        kind, device_prompt, prompt_rows, columns, device_y, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmq_mma_y_path(
        kind, device_weights, output_rows, columns, device_y, prompt_rows,
        device_out, nullptr, path, device_fixup, fixup_floats);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("stream-K MMA execution", error);
  std::vector<float> actual(expected.size());
  error = cudaMemcpy(actual.data(), device_out, actual.size() * sizeof(float),
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return fail_cuda("stream-K MMA D2H", error);
  float max_abs = 0.0F;
  float max_rel = 0.0F;
  float rms = 0.0F;
  std::size_t bad = 0;
  std::size_t nonfinite = 0;
  const bool ok = ds4_q4k_association_ok(actual, expected, columns, &max_abs,
                                         &max_rel, &rms, &bad, &nonfinite);
  std::printf("mma_stream_k_case=%s path=%s prompt_rows=%zu output_rows=%zu "
              "columns=%zu nblocks=%d fixup=%s max_abs=%.9g max_rel=%.9g "
              "rms=%.9g bad=%zu nonfinite=%zu\n",
              name, path, prompt_rows, output_rows, columns, nblocks,
              fixup ? "true" : "false", max_abs, max_rel, rms, bad, nonfinite);
  cudaFree(device_fixup);
  cudaFree(device_out);
  cudaFree(device_y);
  cudaFree(device_prompt);
  cudaFree(device_weights);
  return ok ? 0 : 1;
}

cudaError_t launch_ffn_stream_k_candidate(
    const char* path, const std::uint8_t* gate_w, const std::uint8_t* up_w,
    const std::uint8_t* down_w, const __nv_bfloat16* prompt,
    qw38::cuda::Q8Block* y, float* gate_out, float* up_out, float* down_out,
    float* tmp_fixup, std::size_t tmp_fixup_floats, std::size_t prompt_rows) {
  constexpr std::size_t kHidden = 5120;
  constexpr std::size_t kFfn = 17408;
  cudaError_t error = qw38::cuda::launch_quantize_mmq_q8_1(
      qw38::cuda::QuantKind::kQ4K, prompt, prompt_rows, kHidden, y, nullptr);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmq_mma_y_path(
        qw38::cuda::QuantKind::kQ4K, gate_w, kFfn, kHidden, y, prompt_rows,
        gate_out, nullptr, path, tmp_fixup, tmp_fixup_floats);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmq_mma_y_path(
        qw38::cuda::QuantKind::kQ4K, up_w, kFfn, kHidden, y, prompt_rows,
        up_out, nullptr, path, tmp_fixup, tmp_fixup_floats);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_swiglu_quantize_mmq_q8_1(
        gate_out, up_out, prompt_rows, kFfn, y, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmq_mma_y_path(
        qw38::cuda::QuantKind::kQ4K, down_w, kHidden, kFfn, y, prompt_rows,
        down_out, nullptr, path, tmp_fixup, tmp_fixup_floats);
  }
  return error;
}

int run_mmq_stream_k_ab() {
  if (qw38::cuda::mmq_stream_k_occupancy(qw38::cuda::QuantKind::kQ4K) < 1 ||
      qw38::cuda::mmq_stream_k_fixup_occupancy() < 1) {
    std::fprintf(stderr, "MMQ stream-K occupancy < 1\n");
    return 1;
  }
  constexpr std::size_t kPrompt = 4096;
  constexpr std::size_t kHidden = 5120;
  constexpr std::size_t kFfn = 17408;
  std::vector<std::uint8_t> gate_w;
  std::vector<std::uint8_t> up_w;
  std::vector<std::uint8_t> down_w;
  fill_weights(qw38::cuda::QuantKind::kQ4K, kFfn, kHidden, &gate_w);
  fill_weights(qw38::cuda::QuantKind::kQ4K, kFfn, kHidden, &up_w);
  fill_weights(qw38::cuda::QuantKind::kQ4K, kHidden, kFfn, &down_w);
  std::vector<__nv_bfloat16> prompt(kPrompt * kHidden);
  for (std::size_t row = 0; row < kPrompt; ++row) {
    for (std::size_t col = 0; col < kHidden; ++col) {
      prompt[row * kHidden + col] = __float2bfloat16_rn(
          0.02F * unit_normal(
              static_cast<std::uint32_t>(row * kHidden + col), 0xA5A5A5U));
    }
  }
  const int nsm = qw38::cuda::mmq_stream_k_nsm();
  const std::size_t fixup_floats =
      qw38::cuda::mmq_stream_k_fixup_floats(nsm, 128, 128);
  std::uint8_t* device_gate = nullptr;
  std::uint8_t* device_up = nullptr;
  std::uint8_t* device_down = nullptr;
  __nv_bfloat16* device_prompt = nullptr;
  qw38::cuda::Q8Block* device_y = nullptr;
  float* device_gate_out = nullptr;
  float* device_up_out = nullptr;
  float* device_down_out = nullptr;
  float* device_fixup = nullptr;
  cudaError_t error = cudaMalloc(&device_gate, gate_w.size());
  if (error == cudaSuccess) error = cudaMalloc(&device_up, up_w.size());
  if (error == cudaSuccess) error = cudaMalloc(&device_down, down_w.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_prompt, prompt.size() * sizeof(prompt[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_y,
                       qw38::cuda::q8_prompt_workspace_bytes(kPrompt, kFfn));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_gate_out, kPrompt * kFfn * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_up_out, kPrompt * kFfn * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_down_out, kPrompt * kHidden * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_fixup, fixup_floats * sizeof(float));
  }
  if (error != cudaSuccess) return fail_cuda("MMQ stream-K A/B cudaMalloc", error);
  error = cudaMemcpy(device_gate, gate_w.data(), gate_w.size(),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_up, up_w.data(), up_w.size(),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_down, down_w.data(), down_w.size(),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_prompt, prompt.data(),
                       prompt.size() * sizeof(prompt[0]),
                       cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("MMQ stream-K A/B H2D", error);

  const char* ids[] = {"off", "stream_k", "stream_k_nsm"};
  ensure_opt028_evidence_dir();
  FILE* raw = std::fopen(
      "evidence/optimization/opt028-mmq-streamk/mmq-ab-raw.txt", "w");
  if (raw == nullptr) {
    std::fprintf(stderr, "MMQ stream-K A/B fopen failed\n");
    return 1;
  }
  float means[3]{};
  bool eligible[3]{};
  std::size_t nonfinites[3]{};
  std::vector<float> host(kPrompt * kHidden);
  for (int c = 0; c < 3; ++c) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    error = cudaEventCreate(&start);
    if (error == cudaSuccess) error = cudaEventCreate(&stop);
    float total = 0.0F;
    for (int sample = 0; sample < 3 && error == cudaSuccess; ++sample) {
      error = cudaEventRecord(start);
      if (error == cudaSuccess) {
        error = launch_ffn_stream_k_candidate(
            ids[c], device_gate, device_up, device_down, device_prompt,
            device_y, device_gate_out, device_up_out, device_down_out,
            device_fixup, fixup_floats, kPrompt);
      }
      if (error == cudaSuccess) error = cudaEventRecord(stop);
      if (error == cudaSuccess) error = cudaEventSynchronize(stop);
      float milliseconds = 0.0F;
      if (error == cudaSuccess) {
        error = cudaEventElapsedTime(&milliseconds, start, stop);
      }
      if (error == cudaSuccess) {
        total += milliseconds;
        std::fprintf(raw, "mmq_ab id=%s sample=%d ms=%.9g occupancy=1\n",
                     ids[c], sample, milliseconds);
        std::printf("mmq_ab id=%s sample=%d ms=%.9g occupancy=1\n", ids[c],
                    sample, milliseconds);
      }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("MMQ stream-K A/B time", error);
    }
    means[c] = total / 3.0F;
    error = cudaMemcpy(host.data(), device_down_out,
                       host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("MMQ stream-K A/B D2H", error);
    }
    nonfinites[c] = 0;
    for (float value : host) {
      if (!std::isfinite(value)) ++nonfinites[c];
    }
    const int ntiles_gate = static_cast<int>((kFfn + 127) / 128) *
                            static_cast<int>((kPrompt + 127) / 128);
    const int nblocks =
        qw38::cuda::mmq_stream_k_nblocks(nsm, ntiles_gate, ids[c]);
    eligible[c] = nonfinites[c] == 0 &&
                  (c == 0 || (nblocks > 0 &&
                              qw38::cuda::mmq_stream_k_occupancy(
                                  qw38::cuda::QuantKind::kQ4K) >= 1));
    std::fprintf(raw,
                 "mmq_ab_mean id=%s mean_ms=%.9g occupancy=1 nonfinite=%zu "
                 "nblocks=%d eligible=%s\n",
                 ids[c], means[c], nonfinites[c], nblocks,
                 eligible[c] ? "true" : "false");
    std::printf("mmq_ab_mean id=%s mean_ms=%.9g occupancy=1 nonfinite=%zu "
                "nblocks=%d eligible=%s\n",
                ids[c], means[c], nonfinites[c], nblocks,
                eligible[c] ? "true" : "false");
  }
  int winner = 0;
  if (eligible[0]) {
    for (int c = 1; c < 3; ++c) {
      if (eligible[c] && means[c] < means[winner]) winner = c;
    }
    if (winner != 0 && !(means[winner] < means[0])) winner = 0;
  }
  const bool ab_win = eligible[0] && winner != 0 && means[winner] < means[0];
  std::fprintf(raw,
               "mmq_ab_winner id=%s mean_ms=%.9g baseline_id=off "
               "baseline_ms=%.9g win=%s\n",
               ids[winner], means[winner], means[0],
               ab_win ? "true" : "false");
  std::printf("mmq_ab_winner id=%s mean_ms=%.9g baseline_id=off "
              "baseline_ms=%.9g win=%s selected_path=%s\n",
              ids[winner], means[winner], means[0],
              ab_win ? "true" : "false",
              qw38::cuda::selected_mmq_stream_k_path());
  std::fclose(raw);
  cudaFree(device_fixup);
  cudaFree(device_down_out);
  cudaFree(device_up_out);
  cudaFree(device_gate_out);
  cudaFree(device_y);
  cudaFree(device_prompt);
  cudaFree(device_down);
  cudaFree(device_up);
  cudaFree(device_gate);
  return eligible[0] ? 0 : 1;
}

int run_mmq_stream_k_suite() {
  if (run_mmq_stream_k_helpers() != 0) return 1;
  if (run_mma_stream_k_case(qw38::cuda::QuantKind::kQ4K,
                            "q4_k_sk_64x17408x5120", 17408, 5120, 64,
                            "stream_k") != 0 ||
      run_mma_stream_k_case(qw38::cuda::QuantKind::kQ4K,
                            "q4_k_sknsm_64x17408x5120", 17408, 5120, 64,
                            "stream_k_nsm") != 0 ||
      run_mma_stream_k_case(qw38::cuda::QuantKind::kQ4K,
                            "q4_k_sk_2048x17408x5120", 17408, 5120, 2048,
                            "stream_k") != 0 ||
      run_mma_stream_k_case(qw38::cuda::QuantKind::kQ6K,
                            "q6_k_sk_64x5120x6144", 5120, 6144, 64,
                            "stream_k") != 0 ||
      run_mma_stream_k_case(qw38::cuda::QuantKind::kQ6K,
                            "q6_k_sknsm_64x5120x6144", 5120, 6144, 64,
                            "stream_k_nsm") != 0 ||
      run_mma_stream_k_case(qw38::cuda::QuantKind::kQ6K,
                            "q6_k_sk_2048x5120x6144", 5120, 6144, 2048,
                            "stream_k") != 0 ||
      run_mma_stream_k_case(qw38::cuda::QuantKind::kQ4K,
                            "q4_k_sk_4096x5120x17408", 5120, 17408, 4096,
                            "stream_k") != 0 ||
      run_mma_stream_k_case(qw38::cuda::QuantKind::kQ4K,
                            "q4_k_sknsm_4096x5120x17408", 5120, 17408, 4096,
                            "stream_k_nsm") != 0) {
    return 1;
  }
  return run_mmq_stream_k_ab();
}

void ensure_opt025_evidence_dir() {
  mkdir("evidence", 0755);
  mkdir("evidence/optimization", 0755);
  mkdir("evidence/optimization/opt025-ffn-shared-y", 0755);
}

int run_ffn_ab() {
  if (qw38::cuda::mma_mmq_occupancy(qw38::cuda::QuantKind::kQ4K, 128) < 1) {
    std::fprintf(stderr, "FFN MMA occupancy < 1\n");
    return 1;
  }
  constexpr std::size_t kPrompt = 4096;
  constexpr std::size_t kHidden = 5120;
  constexpr std::size_t kFfn = 17408;
  std::vector<std::uint8_t> gate_w;
  std::vector<std::uint8_t> up_w;
  std::vector<std::uint8_t> down_w;
  fill_weights(qw38::cuda::QuantKind::kQ4K, kFfn, kHidden, &gate_w);
  fill_weights(qw38::cuda::QuantKind::kQ4K, kFfn, kHidden, &up_w);
  fill_weights(qw38::cuda::QuantKind::kQ4K, kHidden, kFfn, &down_w);
  std::vector<__nv_bfloat16> prompt(kPrompt * kHidden);
  for (std::size_t row = 0; row < kPrompt; ++row) {
    for (std::size_t col = 0; col < kHidden; ++col) {
      prompt[row * kHidden + col] = __float2bfloat16_rn(
          0.02F * unit_normal(
              static_cast<std::uint32_t>(row * kHidden + col), 0xA5A5A5U));
    }
  }
  std::uint8_t* device_gate = nullptr;
  std::uint8_t* device_up = nullptr;
  std::uint8_t* device_down = nullptr;
  __nv_bfloat16* device_prompt = nullptr;
  __nv_bfloat16* device_mid = nullptr;
  qw38::cuda::Q8Block* device_y = nullptr;
  float* device_gate_out = nullptr;
  float* device_up_out = nullptr;
  float* device_down_out = nullptr;
  cudaError_t error = cudaMalloc(&device_gate, gate_w.size());
  if (error == cudaSuccess) error = cudaMalloc(&device_up, up_w.size());
  if (error == cudaSuccess) error = cudaMalloc(&device_down, down_w.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_prompt, prompt.size() * sizeof(prompt[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_mid, kPrompt * kFfn * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_y,
                       qw38::cuda::q8_prompt_workspace_bytes(kPrompt, kFfn));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_gate_out, kPrompt * kFfn * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_up_out, kPrompt * kFfn * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_down_out, kPrompt * kHidden * sizeof(float));
  }
  if (error != cudaSuccess) return fail_cuda("FFN A/B cudaMalloc", error);
  error = cudaMemcpy(device_gate, gate_w.data(), gate_w.size(),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_up, up_w.data(), up_w.size(),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_down, down_w.data(), down_w.size(),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_prompt, prompt.data(),
                       prompt.size() * sizeof(prompt[0]),
                       cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("FFN A/B H2D", error);

  const char* ids[] = {"baseline", "shared_y", "swiglu_q8",
                       "shared_y_swiglu_q8"};
  ensure_opt025_evidence_dir();
  FILE* raw = std::fopen(
      "evidence/optimization/opt025-ffn-shared-y/ffn-ab-raw.txt", "w");
  if (raw == nullptr) {
    std::fprintf(stderr, "FFN A/B fopen failed\n");
    return 1;
  }
  float means[4]{};
  bool eligible[4]{};
  std::size_t nonfinites[4]{};
  std::vector<float> host(kPrompt * kHidden);
  for (int c = 0; c < 4; ++c) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    error = cudaEventCreate(&start);
    if (error == cudaSuccess) error = cudaEventCreate(&stop);
    float total = 0.0F;
    for (int sample = 0; sample < 3 && error == cudaSuccess; ++sample) {
      error = cudaEventRecord(start);
      if (error == cudaSuccess) {
        error = launch_ffn_candidate(
            ids[c], device_gate, device_up, device_down, device_prompt,
            device_y, device_gate_out, device_up_out, device_down_out,
            device_mid, kPrompt);
      }
      if (error == cudaSuccess) error = cudaEventRecord(stop);
      if (error == cudaSuccess) error = cudaEventSynchronize(stop);
      float milliseconds = 0.0F;
      if (error == cudaSuccess) {
        error = cudaEventElapsedTime(&milliseconds, start, stop);
      }
      if (error == cudaSuccess) {
        total += milliseconds;
        std::fprintf(raw, "ffn_ab id=%s sample=%d ms=%.9g occupancy=1\n",
                     ids[c], sample, milliseconds);
        std::printf("ffn_ab id=%s sample=%d ms=%.9g occupancy=1\n", ids[c],
                    sample, milliseconds);
      }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("FFN A/B time", error);
    }
    means[c] = total / 3.0F;
    error = cudaMemcpy(host.data(), device_down_out,
                       host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("FFN A/B D2H", error);
    }
    nonfinites[c] = 0;
    for (float value : host) {
      if (!std::isfinite(value)) ++nonfinites[c];
    }
    eligible[c] = nonfinites[c] == 0;
    std::fprintf(raw,
                 "ffn_ab_mean id=%s mean_ms=%.9g occupancy=1 nonfinite=%zu "
                 "eligible=%s\n",
                 ids[c], means[c], nonfinites[c],
                 eligible[c] ? "true" : "false");
    std::printf("ffn_ab_mean id=%s mean_ms=%.9g occupancy=1 nonfinite=%zu "
                "eligible=%s\n",
                ids[c], means[c], nonfinites[c],
                eligible[c] ? "true" : "false");
  }
  int winner = 0;
  if (eligible[0]) {
    for (int c = 1; c < 4; ++c) {
      if (eligible[c] && means[c] < means[winner]) winner = c;
    }
    if (winner != 0 && !(means[winner] < means[0])) winner = 0;
  }
  const bool ab_win = eligible[0] && winner != 0 && means[winner] < means[0];
  std::fprintf(raw,
               "ffn_ab_winner id=%s mean_ms=%.9g baseline_id=baseline "
               "baseline_ms=%.9g win=%s\n",
               ids[winner], means[winner], means[0],
               ab_win ? "true" : "false");
  std::printf("ffn_ab_winner id=%s mean_ms=%.9g baseline_id=baseline "
              "baseline_ms=%.9g win=%s selected_path=%s\n",
              ids[winner], means[winner], means[0],
              ab_win ? "true" : "false", qw38::cuda::selected_ffn_path());
  std::fclose(raw);
  cudaFree(device_down_out);
  cudaFree(device_up_out);
  cudaFree(device_gate_out);
  cudaFree(device_y);
  cudaFree(device_mid);
  cudaFree(device_prompt);
  cudaFree(device_down);
  cudaFree(device_up);
  cudaFree(device_gate);
  return eligible[0] ? 0 : 1;
}

int run_q8_d2r_case(const char* name, std::size_t output_rows,
                    std::size_t columns, std::size_t prompt_rows) {
  std::vector<std::uint8_t> weights;
  fill_q8_quality_weights(output_rows, columns, &weights);
  std::vector<__nv_bfloat16> prompt;
  prompt.resize(prompt_rows * columns);
  for (std::size_t prompt_row = 0; prompt_row < prompt_rows; ++prompt_row) {
    std::vector<__nv_bfloat16> row_activation;
    make_q8_quality_activation(columns, &row_activation, prompt_row);
    std::copy(row_activation.begin(), row_activation.end(),
              prompt.begin() + prompt_row * columns);
  }
  std::vector<float> expected;
  if (!reference_dequant_gemm(qw38::cuda::QuantKind::kQ8_0, weights, output_rows,
                              columns, prompt, prompt_rows, &expected)) {
    return 1;
  }
  std::uint8_t* device_weights = nullptr;
  std::uint8_t* device_soa = nullptr;
  __nv_bfloat16* device_prompt = nullptr;
  qw38::cuda::Q8Block* device_workspace = nullptr;
  float* device_output = nullptr;
  const std::size_t soa_bytes =
      qw38::cuda::q8_aligned_soa_bytes(output_rows, columns);
  cudaError_t error = cudaMalloc(&device_weights, weights.size());
  if (error == cudaSuccess) error = cudaMalloc(&device_soa, soa_bytes);
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_prompt, prompt.size() * sizeof(prompt[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(
        &device_workspace,
        qw38::cuda::q8_prompt_workspace_bytes(prompt_rows, columns));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_output, expected.size() * sizeof(expected[0]));
  }
  if (error != cudaSuccess) return fail_cuda("Q8 D2R cudaMalloc", error);
  error = cudaMemcpy(device_weights, weights.data(), weights.size(),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_prompt, prompt.data(),
                       prompt.size() * sizeof(prompt[0]), cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("Q8 D2R H2D", error);
  error = qw38::cuda::launch_repack_q8_0_aligned(
      device_weights, output_rows, columns, device_soa, nullptr);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_mmq_q8_1(
        qw38::cuda::QuantKind::kQ8_0, device_prompt, prompt_rows, columns,
        device_workspace, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q8_mmq_d2r(
        device_soa, output_rows, columns, device_workspace, prompt_rows,
        device_output, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("Q8 D2R launch", error);
  std::vector<float> actual(expected.size());
  error = cudaMemcpy(actual.data(), device_output,
                     actual.size() * sizeof(actual[0]), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return fail_cuda("Q8 D2R D2H", error);
  float max_abs = 0.0F;
  float max_rel = 0.0F;
  float rms = 0.0F;
  std::size_t bad = 0;
  std::size_t nonfinite = 0;
  const bool ok = ds4_q8_association_ok(actual, expected, columns, &max_abs,
                                        &max_rel, &rms, &bad, &nonfinite);
  std::printf("q8_d2r_case=%s prompt_rows=%zu output_rows=%zu columns=%zu "
              "mma_max_abs=%.9g mma_max_rel=%.9g mma_rms=%.9g mma_bad=%zu "
              "mma_nonfinite=%zu gate=ds4_q8_association ref=cpu_dequant_gemm "
              "occupancy=%d\n",
              name, prompt_rows, output_rows, columns, max_abs, max_rel, rms,
              bad, nonfinite, qw38::cuda::q8_d2r_occupancy(128));
  cudaFree(device_output);
  cudaFree(device_workspace);
  cudaFree(device_prompt);
  cudaFree(device_soa);
  cudaFree(device_weights);
  return ok ? 0 : 1;
}

void ensure_opt024_evidence_dir() {
  mkdir("evidence", 0755);
  mkdir("evidence/optimization", 0755);
  mkdir("evidence/optimization/opt024-mixer-q8-d2r", 0755);
}

int run_d2r_ab() {
  if (qw38::cuda::q8_d2r_occupancy(128) < 1 ||
      qw38::cuda::q8_quality_mmq_occupancy(128) < 1) {
    std::fprintf(stderr, "D2R occupancy < 1\n");
    return 1;
  }
  struct Shape {
    const char* id;
    std::size_t output_rows;
    std::size_t columns;
  };
  const Shape shapes[] = {
      {"packed_qkv", 10240, 5120},
      {"query_gate", 12288, 5120},
      {"value_gate", 6144, 5120},
      {"gdn_output", 5120, 6144},
      {"key", 1024, 5120},
  };
  constexpr std::size_t kPrompt = 4096;
  ensure_opt024_evidence_dir();
  FILE* raw = std::fopen(
      "evidence/optimization/opt024-mixer-q8-d2r/d2r-ab-raw.txt", "w");
  if (raw == nullptr) {
    std::fprintf(stderr, "D2R A/B fopen failed\n");
    return 1;
  }
  bool all_win = true;
  const char* first_loss = "none";
  for (const Shape& shape : shapes) {
    std::vector<std::uint8_t> weights;
    fill_q8_quality_weights(shape.output_rows, shape.columns, &weights);
    std::vector<__nv_bfloat16> prompt(kPrompt * shape.columns);
    for (std::size_t prompt_row = 0; prompt_row < kPrompt; ++prompt_row) {
      std::vector<__nv_bfloat16> row_activation;
      make_q8_quality_activation(shape.columns, &row_activation, prompt_row);
      std::copy(row_activation.begin(), row_activation.end(),
                prompt.begin() + prompt_row * shape.columns);
    }
    std::uint8_t* device_weights = nullptr;
    std::uint8_t* device_soa = nullptr;
    __nv_bfloat16* device_prompt = nullptr;
    qw38::cuda::Q8Block* device_y = nullptr;
    float* device_output = nullptr;
    const std::size_t soa_bytes =
        qw38::cuda::q8_aligned_soa_bytes(shape.output_rows, shape.columns);
    const std::size_t out_bytes = kPrompt * shape.output_rows * sizeof(float);
    cudaError_t error = cudaMalloc(&device_weights, weights.size());
    if (error == cudaSuccess) error = cudaMalloc(&device_soa, soa_bytes);
    if (error == cudaSuccess) {
      error = cudaMalloc(&device_prompt, prompt.size() * sizeof(prompt[0]));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(
          &device_y,
          qw38::cuda::q8_prompt_workspace_bytes(kPrompt, shape.columns));
    }
    if (error == cudaSuccess) error = cudaMalloc(&device_output, out_bytes);
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("D2R A/B cudaMalloc", error);
    }
    error = cudaMemcpy(device_weights, weights.data(), weights.size(),
                       cudaMemcpyHostToDevice);
    if (error == cudaSuccess) {
      error = cudaMemcpy(device_prompt, prompt.data(),
                         prompt.size() * sizeof(prompt[0]),
                         cudaMemcpyHostToDevice);
    }
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("D2R A/B H2D", error);
    }
    error = qw38::cuda::launch_quantize_mmq_q8_1(
        qw38::cuda::QuantKind::kQ8_0, device_prompt, kPrompt, shape.columns,
        device_y, nullptr);
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_repack_q8_0_aligned(
          device_weights, shape.output_rows, shape.columns, device_soa,
          nullptr);
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("D2R A/B prepare", error);
    }
    float means[2]{};
    std::size_t nonfinites[2]{};
    bool eligible[2]{};
    const char* ids[2] = {"quality_mma", "d2r_soa"};
    std::vector<float> host(kPrompt * shape.output_rows);
    for (int c = 0; c < 2; ++c) {
      cudaEvent_t start = nullptr;
      cudaEvent_t stop = nullptr;
      error = cudaEventCreate(&start);
      if (error == cudaSuccess) error = cudaEventCreate(&stop);
      float total = 0.0F;
      for (int sample = 0; sample < 3 && error == cudaSuccess; ++sample) {
        error = cudaEventRecord(start);
        if (error == cudaSuccess) {
          if (c == 0) {
            error = qw38::cuda::launch_q8_mmq_quality_mma(
                device_weights, shape.output_rows, shape.columns, device_y,
                kPrompt, device_output, nullptr);
          } else {
            error = qw38::cuda::launch_q8_mmq_d2r(
                device_soa, shape.output_rows, shape.columns, device_y,
                kPrompt, device_output, nullptr);
          }
        }
        if (error == cudaSuccess) error = cudaEventRecord(stop);
        if (error == cudaSuccess) error = cudaEventSynchronize(stop);
        float milliseconds = 0.0F;
        if (error == cudaSuccess) {
          error = cudaEventElapsedTime(&milliseconds, start, stop);
        }
        if (error == cudaSuccess) {
          total += milliseconds;
          std::fprintf(raw,
                       "d2r_ab shape=%s id=%s sample=%d ms=%.9g occupancy=%d\n",
                       shape.id, ids[c], sample, milliseconds,
                       c == 0 ? qw38::cuda::q8_quality_mmq_occupancy(128)
                              : qw38::cuda::q8_d2r_occupancy(128));
          std::printf("d2r_ab shape=%s id=%s sample=%d ms=%.9g\n", shape.id,
                      ids[c], sample, milliseconds);
        }
      }
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      if (error != cudaSuccess) {
        std::fclose(raw);
        return fail_cuda("D2R A/B time", error);
      }
      means[c] = total / 3.0F;
      error = cudaMemcpy(host.data(), device_output, out_bytes,
                         cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) {
        std::fclose(raw);
        return fail_cuda("D2R A/B D2H", error);
      }
      nonfinites[c] = 0;
      for (float value : host) {
        if (!std::isfinite(value)) ++nonfinites[c];
      }
      eligible[c] = nonfinites[c] == 0;
      std::fprintf(raw,
                   "d2r_ab_mean shape=%s id=%s mean_ms=%.9g nonfinite=%zu "
                   "eligible=%s\n",
                   shape.id, ids[c], means[c], nonfinites[c],
                   eligible[c] ? "true" : "false");
      std::printf("d2r_ab_mean shape=%s id=%s mean_ms=%.9g nonfinite=%zu "
                  "eligible=%s\n",
                  shape.id, ids[c], means[c], nonfinites[c],
                  eligible[c] ? "true" : "false");
    }
    const bool win =
        eligible[0] && eligible[1] && means[1] < means[0];
    std::fprintf(raw,
                 "d2r_ab_shape_winner shape=%s winner=%s quality_ms=%.9g "
                 "d2r_ms=%.9g win=%s\n",
                 shape.id, win ? "d2r_soa" : "quality_mma", means[0], means[1],
                 win ? "true" : "false");
    std::printf("d2r_ab_shape_winner shape=%s winner=%s quality_ms=%.9g "
                "d2r_ms=%.9g win=%s\n",
                shape.id, win ? "d2r_soa" : "quality_mma", means[0], means[1],
                win ? "true" : "false");
    if (!win) {
      all_win = false;
      if (std::strcmp(first_loss, "none") == 0) first_loss = shape.id;
    }
    cudaFree(device_output);
    cudaFree(device_y);
    cudaFree(device_prompt);
    cudaFree(device_soa);
    cudaFree(device_weights);
  }
  std::fprintf(raw,
               "d2r_ab_winner id=%s win=%s first_loss=%s selected_path=%s\n",
               all_win ? "d2r_soa" : "quality_mma", all_win ? "true" : "false",
               first_loss, qw38::cuda::selected_large_mixer_q8_path());
  std::printf("d2r_ab_winner id=%s win=%s first_loss=%s selected_path=%s\n",
              all_win ? "d2r_soa" : "quality_mma", all_win ? "true" : "false",
              first_loss, qw38::cuda::selected_large_mixer_q8_path());
  std::fclose(raw);
  return 0;
}

int run_q8_quality_suite() {
  if (qw38::cuda::selected_q8_quality_mmq_prompt_tile() != 128U ||
      qw38::cuda::q8_quality_mmq_occupancy(128) < 1 ||
      qw38::cuda::q8_quality_mmq_occupancy_i(128, 32) < 1 ||
      qw38::cuda::q8_quality_mmq_occupancy_i(128, 64) < 1 ||
      !qw38::cuda::skinny_mixer_q8_output_rows(48) ||
      qw38::cuda::skinny_mixer_q8_output_rows(128) ||
      qw38::cuda::skinny_mixer_q8_output_rows(1024) ||
      qw38::cuda::skinny_mixer_q8_output_rows(0) ||
      qw38::cuda::launch_quantize_mmq_q8_1(
          qw38::cuda::QuantKind::kQ8_0, nullptr, 8, 255, nullptr, nullptr) !=
          cudaErrorInvalidValue ||
      qw38::cuda::launch_q8_mmq_quality_mma(nullptr, 17, 256, nullptr, 8,
                                            nullptr, nullptr) !=
          cudaErrorInvalidValue ||
      qw38::cuda::launch_q8_mmq_quality_mma_i(nullptr, 17, 256, nullptr, 8,
                                              nullptr, 32, nullptr) !=
          cudaErrorInvalidValue ||
      !qw38::cuda::large_mixer_q8_output_rows(128) ||
      !qw38::cuda::large_mixer_q8_output_rows(1024) ||
      qw38::cuda::large_mixer_q8_output_rows(48) ||
      qw38::cuda::large_mixer_q8_output_rows(0) ||
      qw38::cuda::q8_d2r_occupancy(128) < 1 ||
      qw38::cuda::q8_aligned_soa_bytes(0, 256) != 0 ||
      qw38::cuda::q8_aligned_soa_bytes(128, 255) != 0 ||
      qw38::cuda::launch_repack_q8_0_aligned(nullptr, 128, 256, nullptr,
                                             nullptr) !=
          cudaErrorInvalidValue ||
      qw38::cuda::launch_q8_mmq_d2r(nullptr, 128, 256, nullptr, 8, nullptr,
                                    nullptr) != cudaErrorInvalidValue) {
    std::fprintf(stderr, "Q8 quality launcher rejects failed\n");
    return 1;
  }
  if (run_q8_quality_case("q8_0_quality_128x256x256", 256, 256, 128) != 0 ||
      run_q8_quality_case("q8_0_quality_8x256x256", 256, 256, 8) != 0 ||
      run_q8_quality_case("q8_0_quality_64x1024x512", 1024, 512, 64) != 0 ||
      run_q8_quality_case("q8_0_quality_8x48x5120", 48, 5120, 8) != 0 ||
      run_q8_quality_case("q8_0_quality_64x48x5120", 48, 5120, 64) != 0 ||
      run_q8_quality_case("q8_0_quality_64x1024x5120", 1024, 5120, 64) != 0 ||
      run_q8_quality_case("q8_0_quality_8x5120x6144", 5120, 6144, 8) != 0 ||
      run_q8_quality_i_case("q8_0_quality_i32_8x48x5120", 48, 5120, 8, 32) !=
          0 ||
      run_q8_quality_i_case("q8_0_quality_i64_8x48x5120", 48, 5120, 8, 64) !=
          0 ||
      run_q8_quality_i_case("q8_0_quality_i32_64x48x5120", 48, 5120, 64, 32) !=
          0 ||
      run_q8_quality_i_case("q8_0_quality_i64_64x48x5120", 48, 5120, 64, 64) !=
          0 ||
      run_mmv_tiled_j1_exact() != 0 || run_q8_shared_y_identity() != 0 ||
      run_skinny_ab() != 0 ||
      qw38::cuda::launch_quant_mmq_mma_y(
          qw38::cuda::QuantKind::kQ4K, nullptr, 17, 256, nullptr, 8, nullptr,
          nullptr) != cudaErrorInvalidValue ||
      qw38::cuda::launch_swiglu_quantize_mmq_q8_1(nullptr, nullptr, 8, 256,
                                                 nullptr, nullptr) !=
          cudaErrorInvalidValue ||
      (std::strcmp(qw38::cuda::selected_ffn_path(), "baseline") != 0 &&
       std::strcmp(qw38::cuda::selected_ffn_path(), "shared_y") != 0 &&
       std::strcmp(qw38::cuda::selected_ffn_path(), "swiglu_q8") != 0 &&
       std::strcmp(qw38::cuda::selected_ffn_path(), "shared_y_swiglu_q8") !=
           0) ||
      run_ffn_shared_y_identity() != 0 || run_ffn_swiglu_q8_identity() != 0 ||
      run_ffn_ab() != 0 ||
      run_q8_d2r_case("q8_0_d2r_8x256x256", 256, 256, 8) != 0 ||
      run_q8_d2r_case("q8_0_d2r_64x1024x5120", 1024, 5120, 64) != 0 ||
      run_q8_d2r_case("q8_0_d2r_8x5120x6144", 5120, 6144, 8) != 0 ||
      run_d2r_ab() != 0) {
    return 1;
  }
  return 0;
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
  if (std::strcmp(qw38::cuda::selected_mmv_load_path(), "elementwise") != 0 &&
      std::strcmp(qw38::cuda::selected_mmv_load_path(), "packed") != 0) {
    std::fprintf(stderr, "mmv load path pin is not legal\n");
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
  if (qw38::cuda::launch_quant_mmv_path(
          qw38::cuda::QuantKind::kQ4K, nullptr, 1, 256, nullptr, nullptr,
          nullptr, "packed", nullptr) != cudaErrorInvalidValue ||
      qw38::cuda::launch_quant_mmv_path(
          qw38::cuda::QuantKind::kQ4K, nullptr, 1, 256, nullptr, nullptr,
          nullptr, "dp4a", nullptr) != cudaErrorInvalidValue) {
    std::fprintf(stderr, "invalid packed-path launch was not rejected\n");
    return 1;
  }
  if (run_case(qw38::cuda::QuantKind::kQ4K, "q4_k_17x256", 17, 256) != 0 ||
      run_case(qw38::cuda::QuantKind::kQ4K, "q4_k_257x512", 257, 512) != 0 ||
      run_case(qw38::cuda::QuantKind::kQ6K, "q6_k_17x256", 17, 256) != 0 ||
      run_case(qw38::cuda::QuantKind::kQ6K, "q6_k_257x512", 257, 512) != 0 ||
      run_case(qw38::cuda::QuantKind::kQ8_0, "q8_0_17x256", 17, 256) != 0 ||
      run_case(qw38::cuda::QuantKind::kQ8_0, "q8_0_257x512", 257, 512) != 0 ||
      run_packed_byte_equal(qw38::cuda::QuantKind::kQ4K, "q4_k_17x256", 17,
                            256) != 0 ||
      run_packed_byte_equal(qw38::cuda::QuantKind::kQ4K, "q4_k_257x512", 257,
                            512) != 0 ||
      run_packed_byte_equal(qw38::cuda::QuantKind::kQ6K, "q6_k_17x256", 17,
                            256) != 0 ||
      run_packed_byte_equal(qw38::cuda::QuantKind::kQ6K, "q6_k_257x512", 257,
                            512) != 0) {
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

  if (run_q8_quality_suite() != 0) return 1;
  if (run_mmq_stream_k_suite() != 0) return 1;

  std::printf("status=passed\n");
  return 0;
}
