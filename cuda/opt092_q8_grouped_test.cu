#include "q8_decode_path.cuh"
#include "quant.h"
#include "quant_mmv.h"
#include "test_tier.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr char kResultPrefix[] = "QW38_OPT092_RESULT=";
constexpr char kCountsPrefix[] = "QW38_OPT092_NATIVE_COUNTS=";
constexpr char kCasePrefix[] = "QW38_OPT092_CASE=";
constexpr std::size_t kQ80Bytes = 34;
constexpr std::size_t kQ80Values = 32;
constexpr int kSampledRows = 16;

struct CaseSpec final {
  const char* id;
  std::size_t rows[4];
  int row_count;
  std::size_t columns;
  std::uint32_t seed;
  bool sample_rows;
};

struct CaseResult final {
  std::string id;
  bool pass = false;
  bool staged_bytes_match = true;
  float max_abs = 0.0F;
  std::size_t nonfinite = 0;
  std::string reason;
};

struct TensorBuffers final {
  std::vector<std::uint8_t> host_w;
  std::uint8_t* d_w = nullptr;
  std::size_t rows = 0;
  std::size_t row_offset = 0;
};

const char* json_bool(bool value) { return value ? "true" : "false"; }

int usage(const char* argv0) {
  std::fprintf(stderr, "usage: %s [--phase parity|smoke|correctness]\n", argv0);
  return 2;
}

const char* default_phase(qw38::cuda::TestTier tier) {
  if (tier == qw38::cuda::TestTier::kSmoke) return "smoke";
  if (tier == qw38::cuda::TestTier::kScreen) return "correctness";
  if (tier == qw38::cuda::TestTier::kAcceptance) return "parity";
  return "correctness";
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

int fail_cuda(const char* op, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", op, cudaGetErrorString(error));
  return 1;
}

void write_u16(std::uint8_t* bytes, std::uint16_t value) {
  bytes[0] = static_cast<std::uint8_t>(value & 0xFFU);
  bytes[1] = static_cast<std::uint8_t>(value >> 8U);
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

void fill_weights(std::size_t rows, std::size_t columns,
                  std::uint32_t seed, std::vector<std::uint8_t>* weights) {
  weights->assign(rows * (columns / kQ80Values) * kQ80Bytes, 0);
  for (std::size_t index = 0; index < weights->size(); ++index) {
    (*weights)[index] =
        static_cast<std::uint8_t>((index * 73 + seed + 19) & 0xFFU);
  }
  for (std::size_t offset = 0; offset < weights->size(); offset += kQ80Bytes) {
    write_u16(weights->data() + offset, 0x3C00U);
  }
}

void fill_activation(std::size_t columns, std::uint32_t seed,
                     std::vector<__nv_bfloat16>* activation) {
  activation->resize(columns);
  for (std::size_t column = 0; column < columns; ++column) {
    const float value = unit_normal(static_cast<std::uint32_t>(column), seed);
    (*activation)[column] = from_bf16_bits(float_to_bf16(value));
  }
}

std::size_t q81_bytes(std::size_t columns) {
  return (columns / kQ80Values) * 36U;
}

std::vector<std::size_t> active_rows(std::size_t rows, bool sample) {
  if (rows == 0) return {};
  if (!sample || rows <= static_cast<std::size_t>(kSampledRows)) {
    std::vector<std::size_t> all(rows);
    for (std::size_t row = 0; row < rows; ++row) all[row] = row;
    return all;
  }
  std::vector<std::size_t> picked;
  for (int i = 0; i < kSampledRows; ++i) {
    picked.push_back((static_cast<std::size_t>(i) * rows) /
                     static_cast<std::size_t>(kSampledRows));
  }
  return picked;
}

void emit_case(const CaseResult& result) {
  std::printf(
      "%s{\"id\":\"%s\",\"pass\":%s,\"gpu_pass\":%s,"
      "\"staged_bytes_match\":%s,\"max_abs\":%.9g,\"nonfinite_count\":%zu,"
      "\"reason\":\"%s\"}\n",
      kCasePrefix, result.id.c_str(), json_bool(result.pass),
      json_bool(result.pass), json_bool(result.staged_bytes_match),
      static_cast<double>(result.max_abs), result.nonfinite, result.reason.c_str());
}

CaseResult run_case(const CaseSpec& spec) {
  CaseResult result;
  result.id = spec.id;
  if (spec.columns == 0 || spec.columns % kQ80Values != 0) {
    result.reason = "invalid_columns";
    emit_case(result);
    return result;
  }
  std::vector<TensorBuffers> tensors;
  std::size_t total_rows = 0;
  for (int index = 0; index < spec.row_count; ++index) {
    const std::size_t rows = spec.rows[index];
    TensorBuffers tensor;
    tensor.rows = rows;
    tensor.row_offset = total_rows;
    fill_weights(rows, spec.columns, spec.seed + static_cast<std::uint32_t>(index),
                 &tensor.host_w);
    cudaError_t error = cudaMalloc(
        &tensor.d_w, tensor.host_w.size() * sizeof(std::uint8_t));
    if (error != cudaSuccess) {
      result.reason = "alloc_weights_failed";
      emit_case(result);
      return result;
    }
    error = cudaMemcpy(tensor.d_w, tensor.host_w.data(), tensor.host_w.size(),
                       cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
      result.reason = "upload_weights_failed";
      emit_case(result);
      return result;
    }
    total_rows += rows;
    tensors.push_back(tensor);
  }
  if (total_rows == 0) {
    result.pass = true;
    result.reason = "empty_ok";
    emit_case(result);
    for (const TensorBuffers& tensor : tensors) cudaFree(tensor.d_w);
    return result;
  }
  std::vector<__nv_bfloat16> activation;
  fill_activation(spec.columns, spec.seed, &activation);
  __nv_bfloat16* d_a = nullptr;
  void* d_s = nullptr;
  float* d_sep = nullptr;
  float* d_grp = nullptr;
  cudaError_t error = cudaMalloc(&d_a, spec.columns * sizeof(__nv_bfloat16));
  if (error == cudaSuccess) error = cudaMalloc(&d_s, q81_bytes(spec.columns));
  if (error == cudaSuccess) {
    error = cudaMalloc(&d_sep, (total_rows + 8) * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&d_grp, (total_rows + 8) * sizeof(float));
  }
  if (error != cudaSuccess) {
    result.reason = "alloc_failed";
    emit_case(result);
    for (const TensorBuffers& tensor : tensors) cudaFree(tensor.d_w);
    return result;
  }
  error = cudaMemcpy(d_a, activation.data(), spec.columns * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_bf16_q8_1(d_a, d_s, spec.columns, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    result.reason = "staging_failed";
    emit_case(result);
    cudaFree(d_a);
    cudaFree(d_s);
    cudaFree(d_sep);
    cudaFree(d_grp);
    for (const TensorBuffers& tensor : tensors) cudaFree(tensor.d_w);
    return result;
  }
  qw38::cuda::Q8GroupedProjDesc host[4]{};
  for (int index = 0; index < spec.row_count; ++index) {
    host[index].weights = tensors[index].d_w;
    host[index].rows = tensors[index].rows;
    host[index].output = d_grp + tensors[index].row_offset;
  }
  for (int index = spec.row_count; index < 4; ++index) {
    host[index] = {};
  }
  for (const TensorBuffers& tensor : tensors) {
    error = qw38::cuda::launch_q8_coop_mmv_prequant(
        tensor.d_w, tensor.rows, spec.columns, d_s,
        d_sep + tensor.row_offset, 1U, 4U, nullptr);
    if (error != cudaSuccess) break;
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    result.reason = "separate_failed";
    emit_case(result);
    cudaFree(d_a);
    cudaFree(d_s);
    cudaFree(d_sep);
    cudaFree(d_grp);
    for (const TensorBuffers& tensor : tensors) cudaFree(tensor.d_w);
    return result;
  }
  qw38::cuda::Q8GroupedProjDesc* d_desc = nullptr;
  error = cudaMalloc(&d_desc, sizeof(qw38::cuda::Q8GroupedProjDesc) * 4);
  if (error == cudaSuccess) {
    error = qw38::cuda::upload_q8_grouped_descriptors(d_desc, host, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q8_coop_mmv_grouped_r1_w4(
        d_desc, host, spec.columns, d_s, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  cudaFree(d_desc);
  if (error != cudaSuccess) {
    result.reason = "grouped_failed";
    emit_case(result);
    cudaFree(d_a);
    cudaFree(d_s);
    cudaFree(d_sep);
    cudaFree(d_grp);
    for (const TensorBuffers& tensor : tensors) cudaFree(tensor.d_w);
    return result;
  }
  std::vector<float> sep_host(total_rows + 8, 0.0F);
  std::vector<float> grp_host(total_rows + 8, 0.0F);
  error = cudaMemcpy(sep_host.data(), d_sep, sep_host.size() * sizeof(float),
                     cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(grp_host.data(), d_grp, grp_host.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) {
    result.reason = "readback_failed";
    emit_case(result);
    cudaFree(d_a);
    cudaFree(d_s);
    cudaFree(d_sep);
    cudaFree(d_grp);
    for (const TensorBuffers& tensor : tensors) cudaFree(tensor.d_w);
    return result;
  }
  for (const TensorBuffers& tensor : tensors) {
    for (std::size_t row : active_rows(tensor.rows, spec.sample_rows)) {
      const std::size_t index = tensor.row_offset + row;
      const float sep = sep_host[index];
      const float grp = grp_host[index];
      if (!std::isfinite(sep) || !std::isfinite(grp)) {
        ++result.nonfinite;
        continue;
      }
      result.max_abs = std::max(result.max_abs, std::fabs(sep - grp));
    }
  }
  result.pass = result.nonfinite == 0 && result.max_abs == 0.0F;
  result.reason = result.pass ? "bitwise_match" : "output_mismatch";
  emit_case(result);
  cudaFree(d_a);
  cudaFree(d_s);
  cudaFree(d_sep);
  cudaFree(d_grp);
  for (const TensorBuffers& tensor : tensors) cudaFree(tensor.d_w);
  return result;
}

int run_catalog() {
  const CaseSpec specs[] = {
      {"Q8_grouped_M1_N1_K256_staged_exact", {1, 0, 0, 0}, 1, 256, 0x55U, false},
      {"Q8_grouped_M3_N1_K256_staged_exact", {3, 0, 0, 0}, 1, 256, 0x56U, false},
      {"Q8_grouped_M17_N1_K256_staged_exact", {17, 0, 0, 0}, 1, 256, 0x57U, false},
      {"Q8_grouped_M0_N1_K256_staged_exact", {0, 0, 0, 0}, 1, 256, 0x58U, false},
      {"Q8_grouped_M1_N1_K5120_staged_exact", {1, 0, 0, 0}, 1, 5120, 0x59U, false},
      {"Q8_grouped_M3_N1_K5120_staged_exact", {3, 0, 0, 0}, 1, 5120, 0x5AU, false},
      {"Q8_grouped_M17_N1_K5120_staged_exact", {17, 0, 0, 0}, 1, 5120, 0x5BU, false},
      {"Q8_grouped_M0_N1_K5120_staged_exact", {0, 0, 0, 0}, 1, 5120, 0x5CU, false},
      {"Q8_gdn_input_group_prod", {10240, 6144, 48, 48}, 4, 5120, 0x60U, true},
      {"Q8_attn_input_group_prod", {12288, 1024, 1024, 0}, 3, 5120, 0x61U, true},
      {"Q8_gdn_output_separate_prod", {5120, 0, 0, 0}, 1, 6144, 0x62U, true},
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
      "%s{\"schema_version\":1,\"task\":\"OPT-092\",\"phase\":\"parity\","
      "\"success\":%s,\"case_count\":%d,\"passed\":%d,\"failed\":%d,"
      "\"separate_input_launches\":%u,\"grouped_input_launches\":%u,"
      "\"launch_reduction\":%u}\n",
      kResultPrefix, json_bool(success), case_count, passed, failures,
      qw38::cuda::kQ8GroupedSeparateInputLaunches,
      qw38::cuda::kQ8GroupedCandidateInputLaunches,
      qw38::cuda::kQ8GroupedLaunchReduction);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-092\",\"warmups\":0,\"samples\":1,"
      "\"observed_warmups\":0,\"observed_samples\":1,\"observed_candidates\":1,"
      "\"observed_shapes\":%d,\"observed_tier\":\"correctness\",\"pairs\":1,"
      "\"acceptance_executed\":false,\"keep\":false,\"success\":%s}\n",
      kCountsPrefix, case_count, json_bool(success));
  std::printf("status=%s\n", success ? "passed" : "failed");
  return success ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const char* phase = nullptr;
  if (parse_args(argc, argv, &phase) != 0) return 2;
  if (phase == nullptr) {
    phase = default_phase(qw38::cuda::test_tier());
  }
  if (std::strcmp(phase, "parity") != 0 && std::strcmp(phase, "correctness") != 0 &&
      std::strcmp(phase, "smoke") != 0) {
    std::fprintf(stderr, "unsupported phase %s\n", phase);
    return 2;
  }
  return run_catalog();
}
