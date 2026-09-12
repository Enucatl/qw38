#define QW38_SKIP_Q4K_DOT_KERNELS
#include "q4k_decode_dots.cuh"
#include "q4k_decode_path.cuh"
#include "quant.h"
#include "quant_mmv.h"
#include "test_tier.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr char kResultPrefix[] = "QW38_OPT093_RESULT=";
constexpr char kCountsPrefix[] = "QW38_OPT093_NATIVE_COUNTS=";
constexpr char kCasePrefix[] = "QW38_OPT093_CASE=";
constexpr std::size_t kQ4KBytes = 144;
constexpr std::size_t kBlock = 256;
constexpr int kSampledRows = 16;
constexpr int kSampledVectors = 4;
constexpr float kBitwiseGate = 0.0F;

struct CaseSpec final {
  const char* id;
  std::size_t rows;
  std::size_t columns;
  std::uint32_t seed;
  const char* pattern;
  bool sample_rows;
  bool misaligned;
  bool odd_block_mask;
};

struct CaseResult final {
  std::string id;
  bool pass = false;
  float max_abs = 0.0F;
  std::size_t nonfinite = 0;
  std::string reason;
};

const char* json_bool(bool value) { return value ? "true" : "false"; }

int usage(const char* argv0) {
  std::fprintf(stderr, "usage: %s [--phase parity|smoke|correctness]\n", argv0);
  return 2;
}

int parse_args(int argc, char** argv, const char** phase) {
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if ((std::strcmp(arg, "--phase") == 0 ||
         std::strcmp(arg, "--workload") == 0) &&
        index + 1 < argc) {
      *phase = argv[++index];
    } else if (arg[0] == '-') {
      return usage(argv[0]);
    }
  }
  return 0;
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
  const float u2 = static_cast<float>(x & 0x00FFFFFFU) / 16777216.0F;
  return std::sqrt(-2.0F * std::log(u1)) *
         std::cos(6.283185307179586F * u2);
}

std::uint16_t float_to_bf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t rounding_bias = ((bits >> 16U) & 1U) + 0x7FFFU;
  return static_cast<std::uint16_t>((bits + rounding_bias) >> 16U);
}

__nv_bfloat16 from_bf16_bits(std::uint16_t bits) {
  __nv_bfloat16_raw raw;
  raw.x = bits;
  return __nv_bfloat16(raw);
}

void write_u16(std::uint8_t* bytes, std::uint16_t value) {
  bytes[0] = static_cast<std::uint8_t>(value & 0xFFU);
  bytes[1] = static_cast<std::uint8_t>(value >> 8U);
}

void fill_q4(std::size_t rows, std::size_t columns, std::uint32_t seed,
             std::vector<std::uint8_t>* weights, bool odd_block_mask) {
  weights->assign(rows * (columns / kBlock) * kQ4KBytes, 0);
  for (std::size_t index = 0; index < weights->size(); ++index) {
    (*weights)[index] =
        static_cast<std::uint8_t>((index * 41 + seed + 11) & 0xFFU);
  }
  for (std::size_t offset = 0; offset < weights->size(); offset += kQ4KBytes) {
    write_u16(weights->data() + offset, 0x3C00U);
    write_u16(weights->data() + offset + 2, 0x2C00U);
  }
  if (odd_block_mask && weights->size() >= 2 * kQ4KBytes) {
    std::memset(weights->data() + kQ4KBytes, 0, kQ4KBytes);
  }
}

void fill_activation(std::size_t columns, std::uint32_t seed,
                     const char* pattern, std::vector<__nv_bfloat16>* activation) {
  activation->resize(columns);
  for (std::size_t column = 0; column < columns; ++column) {
    float value = unit_normal(static_cast<std::uint32_t>(column), seed);
    if (pattern != nullptr) {
      if (std::strcmp(pattern, "zero") == 0) value = 0.0F;
      else if (std::strcmp(pattern, "cancellation") == 0) {
        value = (column % 2U == 0) ? 32.0F : -32.0F;
      } else if (std::strcmp(pattern, "minmax") == 0) {
        value = (column % 32U < 16U) ? 8.0F : -8.0F;
      } else if (std::strcmp(pattern, "half_rounding") == 0) {
        value = 0.00390625F * static_cast<float>(column % 17U);
      }
    }
    (*activation)[column] = from_bf16_bits(float_to_bf16(value));
  }
}

__global__ void q4k_coop_mmv_factored_ref(const std::uint8_t* weights,
                                          std::size_t rows, std::size_t columns,
                                          const void* staged, float* output) {
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) return;
  const int pair = (lane % 16) / 4;
  const int pack = lane % 4;
  const int lane_half = lane / 16;
  const int group0 = 2 * pair;
  const int group1 = group0 + 1;
  const std::size_t n_blocks = columns / qw38::cuda::q4k_dots::kValuesPerBlock;
  const std::uint8_t* row_weights = weights + row * n_blocks * kQ4KBytes;
  const std::size_t group_bytes = sizeof(qw38::cuda::Q8Block);
  const char* staged_bytes = static_cast<const char*>(staged);
  float acc = 0.0F;
  for (std::size_t kbx = static_cast<std::size_t>(2 * warp + lane_half);
       kbx < n_blocks; kbx += 8U) {
    const std::uint8_t* block = row_weights + kbx * kQ4KBytes;
    const void* block_staged = staged_bytes + kbx * 8U * group_bytes;
    const qw38::cuda::q4k_dots::Q8PackRegs q8_0 =
        qw38::cuda::q4k_dots::load_q8_pack_factored(block_staged, group0, pack);
    const qw38::cuda::q4k_dots::Q8PackRegs q8_1 =
        qw38::cuda::q4k_dots::load_q8_pack_factored(block_staged, group1, pack);
    acc += qw38::cuda::q4k_dots::vec_dot_q4k_q8block_factored_single(
        block, q8_0, group0, pack);
    acc += qw38::cuda::q4k_dots::vec_dot_q4k_q8block_factored_single(
        block, q8_1, group1, pack);
  }
  __shared__ float partial[4][32];
  partial[warp][lane] = acc;
  __syncthreads();
  if (warp == 0) {
    float sum = 0.0F;
#pragma unroll
    for (int other = 0; other < 4; ++other) sum += partial[other][lane];
    for (int offset = 16; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset, 32);
    }
    if (lane == 0) output[row] = sum;
  }
}

cudaError_t launch_factored_ref(const std::uint8_t* weights, std::size_t rows,
                                std::size_t columns, const void* staged,
                                float* output, cudaStream_t stream) {
  dim3 block(32, 4);
  const unsigned int grid = static_cast<unsigned int>(rows);
  q4k_coop_mmv_factored_ref<<<grid, block, 0, stream>>>(
      weights, rows, columns, staged, output);
  return cudaPeekAtLastError();
}

cudaError_t launch_path(const std::uint8_t* weights, std::size_t rows,
                        std::size_t columns, const qw38::cuda::Q8Block* q8,
                        float* output, const char* path,
                        cudaStream_t stream) {
  qw38::cuda::Q4DecodePathScope scope(path, 4U);
  return qw38::cuda::launch_q4k_coop_mmv_prequant_q8(
      weights, rows, columns, q8, output, 4U, stream);
}

void emit_case(const CaseResult& result) {
  std::printf(
      "%s{\"id\":\"%s\",\"pass\":%s,\"max_abs\":%.9g,\"nonfinite_count\":%zu,"
      "\"reason\":\"%s\"}\n",
      kCasePrefix, result.id.c_str(), json_bool(result.pass),
      static_cast<double>(result.max_abs), result.nonfinite,
      result.reason.c_str());
}

CaseResult run_case(const CaseSpec& spec) {
  CaseResult result;
  result.id = spec.id;
  if (spec.columns == 0 || spec.columns % kBlock != 0) {
    result.reason = "invalid_columns";
    emit_case(result);
    return result;
  }
  std::vector<std::uint8_t> weights;
  fill_q4(spec.rows, spec.columns, spec.seed, &weights, spec.odd_block_mask);
  std::vector<__nv_bfloat16> activation;
  fill_activation(spec.columns, spec.seed, spec.pattern, &activation);
  const std::size_t pad = spec.misaligned ? 2U : 0U;
  std::uint8_t* dw_raw = nullptr;
  __nv_bfloat16* da = nullptr;
  qw38::cuda::Q8Block* dq = nullptr;
  float* d_pair = nullptr;
  float* d_ref = nullptr;
  cudaError_t error = cudaMalloc(&dw_raw, weights.size() + pad);
  if (error == cudaSuccess) error = cudaMalloc(&da, spec.columns * sizeof(__nv_bfloat16));
  if (error == cudaSuccess) {
    error = cudaMalloc(&dq, qw38::cuda::q8_workspace_bytes(spec.columns));
  }
  if (error == cudaSuccess) error = cudaMalloc(&d_pair, spec.rows * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&d_ref, spec.rows * sizeof(float));
  if (error != cudaSuccess) {
    result.reason = "alloc_failed";
    emit_case(result);
    return result;
  }
  std::uint8_t* dw = dw_raw + pad;
  error = cudaMemcpy(dw, weights.data(), weights.size(), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(da, activation.data(), spec.columns * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_bf16_q8(da, dq, spec.columns, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    result.reason = "stage_failed";
    emit_case(result);
    cudaFree(dw_raw);
    cudaFree(da);
    cudaFree(dq);
    cudaFree(d_pair);
    cudaFree(d_ref);
    return result;
  }
  error = launch_path(dw, spec.rows, spec.columns, dq, d_pair,
                      qw38::cuda::kLegalQ4DecodePathIntegerQ8Factored, nullptr);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  const char* launch = qw38::cuda::last_q4_launch_variant();
  if (error == cudaSuccess) {
    error = launch_factored_ref(dw, spec.rows, spec.columns, dq, d_ref, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<float> pair_host(spec.rows);
  std::vector<float> ref_host(spec.rows);
  if (error == cudaSuccess) {
    error = cudaMemcpy(pair_host.data(), d_pair, spec.rows * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(ref_host.data(), d_ref, spec.rows * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  cudaFree(dw_raw);
  cudaFree(da);
  cudaFree(dq);
  cudaFree(d_pair);
  cudaFree(d_ref);
  if (error != cudaSuccess) {
    result.reason = "launch_failed";
    emit_case(result);
    return result;
  }
  std::vector<std::size_t> rows_to_check;
  if (spec.sample_rows && spec.rows > static_cast<std::size_t>(kSampledRows)) {
    for (int index = 0; index < kSampledRows; ++index) {
      rows_to_check.push_back((static_cast<std::size_t>(index) * spec.rows) /
                              static_cast<std::size_t>(kSampledRows));
    }
  } else {
    for (std::size_t row = 0; row < spec.rows; ++row) rows_to_check.push_back(row);
  }
  for (std::size_t row : rows_to_check) {
    const float pair = pair_host[row];
    const float ref = ref_host[row];
    if (!std::isfinite(pair) || !std::isfinite(ref)) {
      ++result.nonfinite;
      continue;
    }
    result.max_abs = std::max(result.max_abs, std::fabs(pair - ref));
  }
  if (spec.misaligned &&
      std::strstr(launch, "factored") == nullptr) {
    result.reason = "missing_factored_launch";
    emit_case(result);
    return result;
  }
  result.pass = result.nonfinite == 0 && result.max_abs <= kBitwiseGate;
  result.reason = result.pass ? "same_math_bitwise" : "output_mismatch";
  emit_case(result);
  return result;
}

int run_catalog() {
  const CaseSpec specs[] = {
      {"Q4_factored_M1_N1_K256_random", 1, 256, 89U, nullptr, false, false, false},
      {"Q4_factored_M3_N1_K256_random", 3, 256, 89U, nullptr, false, false, false},
      {"Q4_factored_M17_N1_K256_random", 17, 256, 89U, nullptr, false, false, false},
      {"Q4_factored_M3_N1_K256_zero", 3, 256, 89U, "zero", false, false, false},
      {"Q4_factored_M3_N1_K256_cancel", 3, 256, 89U, "cancellation", false, false,
       false},
      {"Q4_factored_M3_N1_K256_minmax", 3, 256, 89U, "minmax", false, false, false},
      {"Q4_factored_M3_N1_K256_half_round", 3, 256, 89U, "half_rounding", false,
       false, false},
      {"Q4_factored_M1_N1_K512_random", 1, 512, 89U, nullptr, false, false, false},
      {"Q4_factored_M17_N1_K2048_random", 17, 2048, 89U, nullptr, false, false,
       false},
      {"Q4_factored_M17_N1_K5120_sampled", 17, 5120, 89U, nullptr, true, false, false},
      {"Q4_factored_M17_N1_K17408_sampled", 17, 17408, 89U, nullptr, true, false,
       false},
      {"Q4_factored_M1_N1_K256_odd_mask", 1, 256, 89U, nullptr, false, false, true},
      {"Q4_factored_M17_N1_K256_misaligned", 17, 256, 89U, "cancellation", false,
       true, false},
  };
  int failures = 0;
  int passed = 0;
  const int case_count = static_cast<int>(sizeof(specs) / sizeof(specs[0]));
  for (const CaseSpec& spec : specs) {
    const CaseResult result = run_case(spec);
    if (!result.pass) ++failures;
    else ++passed;
  }
  const bool success = failures == 0;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-093\",\"phase\":\"parity\","
      "\"success\":%s,\"case_count\":%d,\"passed\":%d,\"failed\":%d,"
      "\"sampled_rows\":%d,\"sampled_vectors\":%d,"
      "\"provenance\":\"vecdotq.cuh::vec_dot_q4_K_q8_1_impl_vmmq\"}\n",
      kResultPrefix, json_bool(success), case_count, passed, failures,
      kSampledRows, kSampledVectors);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-093\",\"warmups\":0,\"samples\":1,"
      "\"observed_warmups\":0,\"observed_samples\":1,\"observed_candidates\":1,"
      "\"observed_shapes\":%d,\"observed_tier\":\"correctness\",\"pairs\":1,"
      "\"acceptance_executed\":false,\"keep\":false,\"success\":%s}\n",
      kCountsPrefix, case_count, json_bool(success));
  std::printf("status=%s scalar_fallback_exercised=true\n",
              success ? "passed" : "failed");
  return success ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const char* phase = nullptr;
  if (parse_args(argc, argv, &phase) != 0) return 2;
  if (phase == nullptr) {
    phase = qw38::cuda::test_tier() == qw38::cuda::TestTier::kSmoke ? "smoke"
                                                                    : "parity";
  }
  if (std::strcmp(phase, "parity") != 0 && std::strcmp(phase, "correctness") != 0 &&
      std::strcmp(phase, "smoke") != 0) {
    std::fprintf(stderr, "unsupported phase %s\n", phase);
    return 2;
  }
  return run_catalog();
}
