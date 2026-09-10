#include "quant.h"
#include "sha256.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kQ4KBytes = 144;
constexpr std::size_t kQ6KBytes = 210;
constexpr std::size_t kQ80Bytes = 34;
constexpr std::size_t kValuesPerBlock = 256;
constexpr std::size_t kQ80Values = 32;
constexpr char kPrefix[] = "QW38_OPT044_PRODUCTION_NUMERICS_RESULT=";

std::uint16_t float_to_bf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t rounding_bias = ((bits >> 16U) & 1U) + 0x7FFFU;
  return static_cast<std::uint16_t>((bits + rounding_bias) >> 16U);
}

float bf16_to_float(std::uint16_t bits) {
  std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

std::uint16_t float_to_half(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  std::int32_t exponent = static_cast<std::int32_t>((bits >> 23U) & 0xFFU) - 127;
  std::uint32_t mantissa = bits & 0x7FFFFFU;
  if (((bits >> 23U) & 0xFFU) == 0xFFU) {
    return static_cast<std::uint16_t>(sign | 0x7C00U |
                                      (mantissa ? 0x200U : 0U));
  }
  if (exponent > 15) return static_cast<std::uint16_t>(sign | 0x7C00U);
  if (exponent > -15) {
    const std::uint32_t rounded =
        mantissa + 0x1000U + ((mantissa >> 13U) & 1U);
    if (rounded & 0x800000U) {
      ++exponent;
      mantissa = 0;
    } else {
      mantissa = rounded;
    }
    if (exponent > 15) return static_cast<std::uint16_t>(sign | 0x7C00U);
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exponent + 15) << 10U) |
        (mantissa >> 13U));
  }
  if (exponent < -25) return static_cast<std::uint16_t>(sign);
  mantissa |= 0x800000U;
  const int shift = -14 - exponent;
  const std::uint32_t rounded =
      (mantissa + (1U << (shift + 12U)) + ((mantissa >> (shift + 13U)) & 1U)) >>
      (shift + 13U);
  return static_cast<std::uint16_t>(sign | rounded);
}

float half_to_float(std::uint16_t half) {
  const std::uint32_t sign =
      static_cast<std::uint32_t>(half & 0x8000U) << 16U;
  std::uint32_t exponent = (half >> 10U) & 0x1FU;
  std::uint32_t fraction = half & 0x03FFU;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (fraction == 0) {
      bits = sign;
    } else {
      std::uint32_t shifts = 0;
      while ((fraction & 0x0400U) == 0) {
        fraction <<= 1U;
        ++shifts;
      }
      fraction &= 0x03FFU;
      bits = sign | ((113U - shifts) << 23U) | (fraction << 13U);
    }
  } else if (exponent == 0x1FU) {
    bits = sign | 0x7F800000U | (fraction << 13U);
  } else {
    exponent += 112U;
    bits = sign | (exponent << 23U) | (fraction << 13U);
  }
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

float unit_normal(std::uint32_t index, std::uint32_t seed) {
  std::uint32_t x = index * 747796405U + seed;
  x ^= x >> 16U;
  x *= 0x7feb352dU;
  x ^= x >> 15U;
  x *= 0x846ca68bU;
  x ^= x >> 16U;
  const float u1 = (static_cast<float>(x & 0x00FFFFFFU) + 1.0F) / 16777217.0F;
  x = x * 1664525U + 1013904223U;
  const float u2 = static_cast<float>(x & 0x00FFFFFFU) / 16777216.0F;
  return std::sqrt(-2.0F * std::log(u1)) *
         std::cos(6.283185307179586F * u2);
}

void write_u16(std::uint8_t* bytes, std::uint16_t value) {
  bytes[0] = static_cast<std::uint8_t>(value & 0xFFU);
  bytes[1] = static_cast<std::uint8_t>(value >> 8U);
}

void fill_q4k_weights(std::size_t rows, std::size_t columns,
                      std::vector<std::uint8_t>* weights) {
  weights->assign(rows * (columns / kValuesPerBlock) * kQ4KBytes, 0);
  for (std::size_t index = 0; index < weights->size(); ++index) {
    (*weights)[index] = static_cast<std::uint8_t>((index * 73 + 19) & 0xFFU);
  }
  for (std::size_t offset = 0; offset < weights->size(); offset += kQ4KBytes) {
    write_u16(weights->data() + offset, 0x2400U);
    write_u16(weights->data() + offset + 2, 0x1C00U);
  }
}

void fill_q6k_weights(std::size_t rows, std::size_t columns,
                      std::vector<std::uint8_t>* weights) {
  weights->assign(rows * (columns / kValuesPerBlock) * kQ6KBytes, 0);
  for (std::size_t index = 0; index < weights->size(); ++index) {
    (*weights)[index] = static_cast<std::uint8_t>((index * 73 + 19) & 0xFFU);
  }
}

void fill_q80_weights(std::size_t rows, std::size_t columns,
                      std::vector<std::uint8_t>* weights) {
  weights->assign(rows * (columns / kQ80Values) * kQ80Bytes, 0);
  for (std::size_t index = 0; index < weights->size(); ++index) {
    (*weights)[index] = static_cast<std::uint8_t>((index * 73 + 19) & 0xFFU);
  }
  for (std::size_t offset = 0; offset < weights->size(); offset += kQ80Bytes) {
    write_u16(weights->data() + offset, 0x3C00U);
  }
}

void fill_activation_unit(std::size_t columns,
                          std::vector<std::uint16_t>* activation) {
  activation->resize(columns);
  for (std::size_t column = 0; column < columns; ++column) {
    (*activation)[column] = float_to_bf16(
        unit_normal(static_cast<std::uint32_t>(column), 0xA11CE5u));
  }
}

void fill_activation_pattern(const char* pattern, std::size_t columns,
                             std::vector<std::uint16_t>* activation) {
  activation->resize(columns);
  for (std::size_t column = 0; column < columns; ++column) {
    float value = 0.0F;
    if (std::strcmp(pattern, "zero") == 0) {
      value = 0.0F;
    } else if (std::strcmp(pattern, "alternating") == 0) {
      value = (column % 2U == 0) ? 1.0F : -1.0F;
    } else if (std::strcmp(pattern, "cancellation") == 0) {
      value = (column % 2U == 0) ? 1024.0F : -1024.0F;
    } else if (std::strcmp(pattern, "extremes") == 0) {
      value = (column % 32U == 0) ? 64.0F : 1.0e-6F;
    } else if (std::strcmp(pattern, "small") == 0) {
      value = 1.0e-4F * ((column % 2U == 0) ? 1.0F : -1.0F);
    } else {
      value = unit_normal(static_cast<std::uint32_t>(column), 0xA11CE5u);
    }
    (*activation)[column] = float_to_bf16(value);
  }
}

bool checksum(const void* data, std::size_t bytes, std::string* digest) {
  return qw38::internal::sha256_bytes(
             static_cast<const unsigned char*>(data), bytes, digest)
      .is_ok();
}

struct Q8Block {
  float scale;
  std::int8_t values[32];
};

void stage_quartz_q8(const std::vector<std::uint16_t>& activation,
                     std::vector<Q8Block>* q8) {
  const std::size_t blocks = activation.size() / 32U;
  q8->assign(blocks, Q8Block{});
  for (std::size_t block = 0; block < blocks; ++block) {
    float maximum = 0.0F;
    for (std::size_t lane = 0; lane < 32U; ++lane) {
      maximum = std::max(
          maximum, std::fabs(bf16_to_float(activation[block * 32U + lane])));
    }
    const float scale = maximum / 127.0F;
    (*q8)[block].scale = scale;
    const float inv = scale > 0.0F ? 1.0F / scale : 0.0F;
    for (std::size_t lane = 0; lane < 32U; ++lane) {
      const float value = bf16_to_float(activation[block * 32U + lane]) * inv;
      int quantized = static_cast<int>(std::round(value));
      quantized = std::max(-127, std::min(127, quantized));
      (*q8)[block].values[lane] = static_cast<std::int8_t>(quantized);
    }
  }
}

void dequant_quartz_q8(const std::vector<Q8Block>& q8,
                       std::vector<double>* values) {
  values->assign(q8.size() * 32U, 0.0);
  for (std::size_t block = 0; block < q8.size(); ++block) {
    for (std::size_t lane = 0; lane < 32U; ++lane) {
      (*values)[block * 32U + lane] =
          static_cast<double>(q8[block].scale) *
          static_cast<double>(q8[block].values[lane]);
    }
  }
}

struct LlamaQ81 {
  float scale;
  float sum_term;
  std::int8_t values[32];
};

void stage_llama_q8_1(const std::vector<std::uint16_t>& activation,
                      std::vector<LlamaQ81>* q8) {
  const std::size_t blocks = activation.size() / 32U;
  q8->assign(blocks, LlamaQ81{});
  for (std::size_t block = 0; block < blocks; ++block) {
    float amax = 0.0F;
    for (std::size_t lane = 0; lane < 32U; ++lane) {
      amax = std::max(amax,
                      std::fabs(bf16_to_float(activation[block * 32U + lane])));
    }
    const float d = amax / 127.0F;
    const float id = d > 0.0F ? 1.0F / d : 0.0F;
    const float stored_d = half_to_float(float_to_half(d));
    int sum = 0;
    for (std::size_t lane = 0; lane < 32U; ++lane) {
      const float value = bf16_to_float(activation[block * 32U + lane]) * id;
      int quantized = static_cast<int>(std::round(value));
      quantized = std::max(-127, std::min(127, quantized));
      (*q8)[block].values[lane] = static_cast<std::int8_t>(quantized);
      sum += quantized;
    }
    (*q8)[block].scale = stored_d;
    (*q8)[block].sum_term = half_to_float(float_to_half(static_cast<float>(sum) * d));
  }
}

void dequant_llama_q8_1(const std::vector<LlamaQ81>& q8,
                        std::vector<double>* values) {
  values->assign(q8.size() * 32U, 0.0);
  for (std::size_t block = 0; block < q8.size(); ++block) {
    for (std::size_t lane = 0; lane < 32U; ++lane) {
      (*values)[block * 32U + lane] =
          static_cast<double>(q8[block].scale) *
          static_cast<double>(q8[block].values[lane]);
    }
  }
}

using DecodeFn = qw38::Status (*)(const std::uint8_t*, std::size_t, float*,
                                  std::size_t) noexcept;

bool decode_row(DecodeFn decode, const std::uint8_t* packed,
                std::size_t block_bytes, std::size_t block_values,
                std::vector<float>* decoded) {
  decoded->assign(block_values, 0.0F);
  return decode(packed, block_bytes, decoded->data(), decoded->size()).is_ok();
}

struct Metrics {
  double max_abs = 0.0;
  double rms = 0.0;
  double cosine = 0.0;
  double one_minus_cosine = 0.0;
  double relative_max = 0.0;
  int first_fail = -1;
  int nonfinite = 0;
  double candidate_norm = 0.0;
  double reference_norm = 0.0;
};

Metrics compare(const std::vector<double>& candidate,
                const std::vector<double>& reference, double abs_ceiling) {
  Metrics metrics;
  if (candidate.size() != reference.size() || candidate.empty()) {
    metrics.nonfinite = 1;
    metrics.first_fail = 0;
    return metrics;
  }
  double squared = 0.0;
  double dot = 0.0;
  double cand_sq = 0.0;
  double ref_sq = 0.0;
  for (std::size_t index = 0; index < candidate.size(); ++index) {
    const double left = candidate[index];
    const double right = reference[index];
    if (!std::isfinite(left) || !std::isfinite(right)) {
      ++metrics.nonfinite;
      if (metrics.first_fail < 0) {
        metrics.first_fail = static_cast<int>(index);
      }
      continue;
    }
    const double abs_err = std::fabs(left - right);
    if (abs_err > metrics.max_abs) metrics.max_abs = abs_err;
    squared += abs_err * abs_err;
    const double denom = std::max(std::fabs(right), 1.0e-12);
    metrics.relative_max = std::max(metrics.relative_max, abs_err / denom);
    if (metrics.first_fail < 0 && abs_err > abs_ceiling) {
      metrics.first_fail = static_cast<int>(index);
    }
    dot += left * right;
    cand_sq += left * left;
    ref_sq += right * right;
  }
  metrics.rms = std::sqrt(squared / static_cast<double>(candidate.size()));
  metrics.candidate_norm = std::sqrt(cand_sq);
  metrics.reference_norm = std::sqrt(ref_sq);
  const double denom = metrics.candidate_norm * metrics.reference_norm;
  if (denom == 0.0) {
    metrics.cosine = 0.0;
    metrics.one_minus_cosine = 1.0;
  } else {
    metrics.cosine = dot / denom;
    metrics.one_minus_cosine = 1.0 - metrics.cosine;
  }
  return metrics;
}

void gemm_fp64(const std::vector<std::uint8_t>& weights, std::size_t rows,
               std::size_t columns, std::size_t block_bytes,
               std::size_t block_values, DecodeFn decode,
               const std::vector<double>& activation,
               std::vector<double>* output) {
  output->assign(rows, 0.0);
  std::vector<float> decoded;
  for (std::size_t row = 0; row < rows; ++row) {
    double sum = 0.0;
    for (std::size_t block = 0; block < columns / block_values; ++block) {
      const std::uint8_t* packed =
          weights.data() +
          (row * (columns / block_values) + block) * block_bytes;
      if (!decode_row(decode, packed, block_bytes, block_values, &decoded)) {
        (*output)[row] = std::numeric_limits<double>::quiet_NaN();
        break;
      }
      for (std::size_t within = 0; within < block_values; ++within) {
        sum += static_cast<double>(decoded[within]) *
               activation[block * block_values + within];
      }
    }
    (*output)[row] = sum;
  }
}

void gemm_fp32(const std::vector<std::uint8_t>& weights, std::size_t rows,
               std::size_t columns, std::size_t block_bytes,
               std::size_t block_values, DecodeFn decode,
               const std::vector<Q8Block>& q8, std::vector<double>* output) {
  output->assign(rows, 0.0);
  std::vector<float> decoded;
  for (std::size_t row = 0; row < rows; ++row) {
    float sum = 0.0F;
    for (std::size_t block = 0; block < columns / block_values; ++block) {
      const std::uint8_t* packed =
          weights.data() +
          (row * (columns / block_values) + block) * block_bytes;
      if (!decode_row(decode, packed, block_bytes, block_values, &decoded)) {
        (*output)[row] = std::numeric_limits<double>::quiet_NaN();
        break;
      }
      for (std::size_t within = 0; within < block_values; ++within) {
        const std::size_t column = block * block_values + within;
        const auto& staged = q8[column / 32U];
        sum += decoded[within] *
               (staged.scale * static_cast<float>(staged.values[column % 32U]));
      }
    }
    (*output)[row] = static_cast<double>(sum);
  }
}

void print_metrics(const char* name, const Metrics& metrics) {
  std::printf(
      "\"%s\":{\"max_abs\":%.9g,\"rms\":%.9g,\"cosine\":%.9g,"
      "\"one_minus_cosine\":%.9g,\"relative_max\":%.9g,\"first_fail\":%d,"
      "\"nonfinite\":%d,\"candidate_norm\":%.9g,\"reference_norm\":%.9g}",
      name, metrics.max_abs, metrics.rms, metrics.cosine,
      metrics.one_minus_cosine, metrics.relative_max, metrics.first_fail,
      metrics.nonfinite, metrics.candidate_norm, metrics.reference_norm);
}

bool run_case(const char* id, const char* family, const char* activation_source,
              std::size_t rows, std::size_t columns, std::size_t block_bytes,
              std::size_t block_values, DecodeFn decode,
              const std::vector<std::uint8_t>& weights,
              const std::vector<std::uint16_t>& activation, bool first) {
  std::string weights_sha;
  std::string activation_sha;
  if (!checksum(weights.data(), weights.size(), &weights_sha) ||
      !checksum(activation.data(), activation.size() * sizeof(std::uint16_t),
                &activation_sha)) {
    std::fprintf(stderr, "checksum failed for %s\n", id);
    return false;
  }
  std::vector<double> bf16_f64(columns);
  for (std::size_t column = 0; column < columns; ++column) {
    bf16_f64[column] = static_cast<double>(bf16_to_float(activation[column]));
  }
  std::vector<Q8Block> quartz_q8;
  std::vector<LlamaQ81> llama_q8;
  stage_quartz_q8(activation, &quartz_q8);
  stage_llama_q8_1(activation, &llama_q8);
  std::vector<double> quartz_staged;
  std::vector<double> llama_staged;
  dequant_quartz_q8(quartz_q8, &quartz_staged);
  dequant_llama_q8_1(llama_q8, &llama_staged);

  std::vector<double> fp64_orig;
  std::vector<double> fp64_quartz_staged;
  std::vector<double> fp64_llama_staged;
  std::vector<double> fp32_host;
  gemm_fp64(weights, rows, columns, block_bytes, block_values, decode, bf16_f64,
            &fp64_orig);
  gemm_fp64(weights, rows, columns, block_bytes, block_values, decode,
            quartz_staged, &fp64_quartz_staged);
  gemm_fp64(weights, rows, columns, block_bytes, block_values, decode,
            llama_staged, &fp64_llama_staged);
  gemm_fp32(weights, rows, columns, block_bytes, block_values, decode, quartz_q8,
            &fp32_host);

  const Metrics host_vs_orig = compare(fp32_host, fp64_orig, 3.0e-4);
  const Metrics host_vs_quartz_staged =
      compare(fp32_host, fp64_quartz_staged, 3.0e-4);
  const Metrics llama_vs_orig = compare(fp64_llama_staged, fp64_orig, 3.0e-4);
  const Metrics llama_vs_llama_staged =
      compare(fp64_llama_staged, fp64_llama_staged, 3.0e-4);
  const Metrics quartz_quant_vs_orig =
      compare(fp64_quartz_staged, fp64_orig, 3.0e-4);
  int llama_nonfinite = 0;
  for (const auto& block : llama_q8) {
    if (!std::isfinite(block.scale) || !std::isfinite(block.sum_term)) {
      ++llama_nonfinite;
    }
  }

  if (!first) std::printf(",");
  std::printf(
      "{\"id\":\"%s\",\"family\":\"%s\",\"rows\":%zu,\"columns\":%zu,"
      "\"activation_source\":\"%s\",\"checksums\":{\"weights\":\"%s\","
      "\"activation\":\"%s\",\"weights_bytes\":%zu,\"activation_bytes\":%zu},"
      "\"llama_q8_1_nonfinite\":%d,\"comparisons\":{",
      id, family, rows, columns, activation_source, weights_sha.c_str(),
      activation_sha.c_str(), weights.size(),
      activation.size() * sizeof(std::uint16_t), llama_nonfinite);
  print_metrics("fp32_host_vs_fp64_original_bf16", host_vs_orig);
  std::printf(",");
  print_metrics("fp32_host_vs_fp64_quartz_staged", host_vs_quartz_staged);
  std::printf(",");
  print_metrics("llama_q8_1_vs_fp64_original_bf16", llama_vs_orig);
  std::printf(",");
  print_metrics("llama_q8_1_vs_fp64_llama_staged", llama_vs_llama_staged);
  std::printf(",");
  print_metrics("quartz_q8_quant_vs_fp64_original_bf16", quartz_quant_vs_orig);
  std::printf("}}");
  return llama_nonfinite == 0 && host_vs_orig.nonfinite == 0 &&
         llama_vs_orig.nonfinite == 0;
}

}  // namespace

int main() {
  std::printf("%s{\"schema_version\":1,\"task\":\"OPT-044\","
              "\"role\":\"independent_host_fp64_llama_q8_1\","
              "\"production_numerics_path\":\"strict\","
              "\"optimized_admitted\":false,\"cases\":[",
              kPrefix);

  bool first = true;
  struct Q4Case {
    const char* id;
    const char* pattern;
    std::size_t rows;
    std::size_t columns;
  };
  const Q4Case q4_cases[] = {
      {"q4_k_17x256", "unit_normal", 17, 256},
      {"q4_k_257x512", "unit_normal", 257, 512},
      {"q4k_gate_up", "unit_normal", 17408, 5120},
      {"q4k_down", "unit_normal", 5120, 17408},
      {"q4_k_17x256_zero", "zero", 17, 256},
      {"q4_k_17x256_alternating", "alternating", 17, 256},
      {"q4_k_17x256_cancellation", "cancellation", 17, 256},
      {"q4_k_17x256_extremes", "extremes", 17, 256},
      {"q4_k_17x256_small", "small", 17, 256},
  };
  for (const auto& spec : q4_cases) {
    std::vector<std::uint8_t> weights;
    std::vector<std::uint16_t> activation;
    fill_q4k_weights(spec.rows, spec.columns, &weights);
    fill_activation_pattern(spec.pattern, spec.columns, &activation);
    if (std::strcmp(spec.pattern, "unit_normal") == 0) {
      fill_activation_unit(spec.columns, &activation);
    }
    if (!run_case(spec.id, "q4_k", spec.pattern, spec.rows, spec.columns,
                  kQ4KBytes, kValuesPerBlock, qw38::internal::decode_q4_k,
                  weights, activation, first)) {
      std::fprintf(stderr, "case %s produced nonfinites\n", spec.id);
    }
    first = false;
  }

  struct SmallCase {
    const char* id;
    const char* family;
    std::size_t rows;
    std::size_t columns;
    std::size_t block_bytes;
    std::size_t block_values;
    DecodeFn decode;
    void (*fill)(std::size_t, std::size_t, std::vector<std::uint8_t>*);
  };
  const SmallCase small[] = {
      {"q6_k_17x256", "q6_k", 17, 256, kQ6KBytes, kValuesPerBlock,
       qw38::internal::decode_q6_k, fill_q6k_weights},
      {"q6_k_257x512", "q6_k", 257, 512, kQ6KBytes, kValuesPerBlock,
       qw38::internal::decode_q6_k, fill_q6k_weights},
      {"q8_0_17x256", "q8_0", 17, 256, kQ80Bytes, kQ80Values,
       qw38::internal::decode_q8_0, fill_q80_weights},
      {"q8_0_257x512", "q8_0", 257, 512, kQ80Bytes, kQ80Values,
       qw38::internal::decode_q8_0, fill_q80_weights},
  };
  for (const auto& spec : small) {
    std::vector<std::uint8_t> weights;
    std::vector<std::uint16_t> activation;
    spec.fill(spec.rows, spec.columns, &weights);
    fill_activation_unit(spec.columns, &activation);
    if (!run_case(spec.id, spec.family, "unit_normal", spec.rows, spec.columns,
                  spec.block_bytes, spec.block_values, spec.decode, weights,
                  activation, first)) {
      std::fprintf(stderr, "case %s produced nonfinites\n", spec.id);
    }
    first = false;
  }

  std::printf("]}\n");
  return 0;
}
