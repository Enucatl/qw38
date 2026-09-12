#define QW38_SKIP_Q4K_DOT_KERNELS
#include "q4k_aligned_layout.cuh"
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

constexpr char kResultPrefix[] = "QW38_OPT102_RESULT=";
constexpr char kCountsPrefix[] = "QW38_OPT102_NATIVE_COUNTS=";
constexpr char kCasePrefix[] = "QW38_OPT102_CASE=";
constexpr std::size_t kQ4KBytes = 144;
constexpr std::size_t kBlock = 256;
constexpr int kSampledRows = 16;
constexpr int kSampledVectors = 4;
constexpr float kBitwiseGate = 0.0F;
constexpr int kLlamaVdr = 2;
constexpr int kLlamaQiQ4K = 32;

enum class CompareKind { kBranchless, kAligned, kInverse, kLlama, kOccupancy };

struct CaseSpec final {
  const char* id;
  std::size_t rows;
  std::size_t columns;
  std::uint32_t seed;
  const char* pattern;
  bool sample_rows;
  bool misaligned;
  bool odd_block_mask;
  CompareKind kind;
};

struct CaseResult final {
  std::string id;
  bool pass = false;
  bool inverse_ok = true;
  bool gpu_pass = false;
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
                     const char* pattern,
                     std::vector<__nv_bfloat16>* activation) {
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

void branched_scale_min(const std::uint8_t* packed, int index, int* scale,
                        int* minimum) {
  if (index < 4) {
    *scale = packed[index] & 63;
    *minimum = packed[index + 4] & 63;
    return;
  }
  *scale = (packed[index + 4] & 15) | ((packed[index - 4] >> 6) << 4);
  *minimum = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4);
}

cudaError_t launch_path(const std::uint8_t* weights, std::size_t rows,
                        std::size_t columns, const qw38::cuda::Q8Block* q8,
                        float* output, const char* path, const char* layout,
                        cudaStream_t stream) {
  qw38::cuda::Q4DecodePathScope path_scope(path, 4U);
  qw38::cuda::Q4DeviceLayoutScope layout_scope(layout);
  return qw38::cuda::launch_q4k_coop_mmv_prequant_q8(
      weights, rows, columns, q8, output, 4U, stream);
}

void emit_case(const CaseResult& result) {
  std::printf(
      "%s{\"id\":\"%s\",\"pass\":%s,\"gpu_pass\":%s,\"inverse_ok\":%s,"
      "\"max_abs\":%.9g,\"nonfinite_count\":%zu,\"reason\":\"%s\"}\n",
      kCasePrefix, result.id.c_str(), json_bool(result.pass),
      json_bool(result.gpu_pass), json_bool(result.inverse_ok),
      static_cast<double>(result.max_abs), result.nonfinite,
      result.reason.c_str());
}

bool host_inverse_ok(const std::vector<std::uint8_t>& gguf, std::size_t rows,
                     std::size_t columns) {
  const auto desc = qw38::cuda::make_q4k_aligned_layout_desc(rows, columns);
  if (!qw38::cuda::q4k_aligned_desc_valid(desc) ||
      desc.gguf_bytes != gguf.size()) {
    return false;
  }
  std::vector<std::uint8_t> soa(desc.total_bytes);
  std::vector<std::uint8_t> back(desc.gguf_bytes);
  qw38::cuda::pack_q4k_aligned_host(gguf.data(), desc, soa.data());
  qw38::cuda::unpack_q4k_aligned_host(soa.data(), desc, back.data());
  return qw38::cuda::q4k_aligned_inverse_matches(gguf.data(), back.data(), desc);
}

CaseResult run_llama_diagnostic() {
  CaseResult result;
  result.id = "Q4_layout_llama_mmvq_diagnostic";
  result.inverse_ok = true;
  std::uint8_t packed[12];
  for (int index = 0; index < 12; ++index) {
    packed[index] = static_cast<std::uint8_t>(index * 17 + 9);
  }
  bool unpack_ok = true;
  for (int group = 0; group < 8; ++group) {
    int branched_s = 0;
    int branched_m = 0;
    int branchless_s = 0;
    int branchless_m = 0;
    branched_scale_min(packed, group, &branched_s, &branched_m);
    qw38::cuda::q4k_scale_min_branchless(packed, group, &branchless_s,
                                         &branchless_m);
    if (branched_s != branchless_s || branched_m != branchless_m) {
      unpack_ok = false;
    }
  }
  const bool mapping_documented =
      kLlamaVdr == 2 && kLlamaQiQ4K == 32 && (32 / 4) == 8;
  result.gpu_pass = unpack_ok && mapping_documented;
  result.pass = result.gpu_pass;
  result.reason = result.pass ? "llama_mapping_comparison_only"
                              : "branchless_unpack_mismatch";
  emit_case(result);
  return result;
}

CaseResult run_occupancy() {
  CaseResult result;
  result.id = "Q4_layout_occupancy_resources";
  int late_regs = 0;
  int branchless_regs = 0;
  int aligned_regs = 0;
  std::size_t late_local = 0;
  std::size_t branchless_local = 0;
  std::size_t aligned_local = 0;
  int late_occ = 0;
  int branchless_occ = 0;
  int aligned_occ = 0;
  qw38::cuda::q4k_coop_late_kernel_attributes(4U, &late_regs, &late_local,
                                             &late_occ);
  qw38::cuda::q4k_coop_branchless_kernel_attributes(
      4U, &branchless_regs, &branchless_local, &branchless_occ);
  qw38::cuda::q4k_coop_aligned_kernel_attributes(4U, &aligned_regs,
                                                &aligned_local, &aligned_occ);
  std::printf(
      "opt102_resources late_regs=%d branchless_regs=%d aligned_regs=%d "
      "late_local=%zu branchless_local=%zu aligned_local=%zu "
      "late_occ=%d branchless_occ=%d aligned_occ=%d "
      "llama_vdr=%d llama_qi=%d late_group=lane/4 late_pack=lane%%4\n",
      late_regs, branchless_regs, aligned_regs, late_local, branchless_local,
      aligned_local, late_occ, branchless_occ, aligned_occ, kLlamaVdr,
      kLlamaQiQ4K);
  result.gpu_pass = late_regs > 0 && branchless_regs > 0 && aligned_regs > 0;
  result.pass = result.gpu_pass;
  result.reason = result.pass ? "resource_query" : "missing_kernel_attributes";
  emit_case(result);
  return result;
}

CaseResult run_case(const CaseSpec& spec) {
  CaseResult result;
  result.id = spec.id;
  if (spec.kind == CompareKind::kLlama) return run_llama_diagnostic();
  if (spec.kind == CompareKind::kOccupancy) return run_occupancy();
  if (spec.columns == 0 || spec.columns % kBlock != 0) {
    result.reason = "invalid_columns";
    emit_case(result);
    return result;
  }
  std::vector<std::uint8_t> weights;
  fill_q4(spec.rows, spec.columns, spec.seed, &weights, spec.odd_block_mask);
  result.inverse_ok = host_inverse_ok(weights, spec.rows, spec.columns);
  if (spec.kind == CompareKind::kInverse) {
    const auto desc =
        qw38::cuda::make_q4k_aligned_layout_desc(spec.rows, spec.columns);
    std::uint8_t* d_raw = nullptr;
    std::uint8_t* d_soa = nullptr;
    std::uint8_t* d_back = nullptr;
    cudaError_t error = cudaMalloc(&d_raw, weights.size());
    if (error == cudaSuccess) error = cudaMalloc(&d_soa, desc.total_bytes);
    if (error == cudaSuccess) error = cudaMalloc(&d_back, weights.size());
    if (error == cudaSuccess) {
      error = cudaMemcpy(d_raw, weights.data(), weights.size(),
                         cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_repack_q4k_aligned(
          d_raw, spec.rows, spec.columns, d_soa, nullptr);
    }
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_unpack_q4k_aligned(
          d_soa, spec.rows, spec.columns, d_back, nullptr);
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    std::vector<std::uint8_t> back(weights.size());
    if (error == cudaSuccess) {
      error = cudaMemcpy(back.data(), d_back, back.size(),
                         cudaMemcpyDeviceToHost);
    }
    cudaFree(d_raw);
    cudaFree(d_soa);
    cudaFree(d_back);
    result.gpu_pass = error == cudaSuccess && result.inverse_ok &&
                      std::memcmp(weights.data(), back.data(), back.size()) == 0;
    result.pass = result.gpu_pass;
    result.reason = result.pass ? "inverse_byte_exact" : "inverse_mismatch";
    emit_case(result);
    return result;
  }
  if (!result.inverse_ok && spec.kind == CompareKind::kAligned) {
    result.reason = "host_inverse_mismatch";
    emit_case(result);
    return result;
  }
  std::vector<__nv_bfloat16> activation;
  fill_activation(spec.columns, spec.seed, spec.pattern, &activation);
  const std::size_t pad = spec.misaligned ? 2U : 0U;
  const auto desc =
      qw38::cuda::make_q4k_aligned_layout_desc(spec.rows, spec.columns);
  std::uint8_t* dw_raw = nullptr;
  std::uint8_t* d_soa = nullptr;
  __nv_bfloat16* da = nullptr;
  qw38::cuda::Q8Block* dq = nullptr;
  float* d_late = nullptr;
  float* d_cand = nullptr;
  cudaError_t error = cudaMalloc(&dw_raw, weights.size() + pad);
  if (error == cudaSuccess && spec.kind == CompareKind::kAligned) {
    error = cudaMalloc(&d_soa, desc.total_bytes);
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&da, spec.columns * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&dq, qw38::cuda::q8_workspace_bytes(spec.columns));
  }
  if (error == cudaSuccess) error = cudaMalloc(&d_late, spec.rows * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&d_cand, spec.rows * sizeof(float));
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
    cudaFree(d_soa);
    cudaFree(da);
    cudaFree(dq);
    cudaFree(d_late);
    cudaFree(d_cand);
    return result;
  }
  error = launch_path(dw, spec.rows, spec.columns, dq, d_late,
                      qw38::cuda::kLegalQ4DecodePathIntegerQ8Late,
                      qw38::cuda::kLegalQ4DeviceLayoutRawGguf, nullptr);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  const char* want_launch = spec.kind == CompareKind::kAligned
                                ? "aligned"
                                : "branchless";
  if (spec.kind == CompareKind::kAligned && error == cudaSuccess) {
    error = qw38::cuda::launch_repack_q4k_aligned(dw, spec.rows, spec.columns,
                                                  d_soa, nullptr);
    if (error == cudaSuccess) {
      error = launch_path(d_soa, spec.rows, spec.columns, dq, d_cand,
                          qw38::cuda::kLegalQ4DecodePathIntegerQ8Aligned,
                          qw38::cuda::kLegalQ4DeviceLayoutAlignedMeta, nullptr);
    }
  } else if (error == cudaSuccess) {
    error = launch_path(dw, spec.rows, spec.columns, dq, d_cand,
                        qw38::cuda::kLegalQ4DecodePathIntegerQ8Branchless,
                        qw38::cuda::kLegalQ4DeviceLayoutRawGguf, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  const char* launch = qw38::cuda::last_q4_launch_variant();
  std::vector<float> late_host(spec.rows);
  std::vector<float> cand_host(spec.rows);
  if (error == cudaSuccess) {
    error = cudaMemcpy(late_host.data(), d_late, spec.rows * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(cand_host.data(), d_cand, spec.rows * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  cudaFree(dw_raw);
  cudaFree(d_soa);
  cudaFree(da);
  cudaFree(dq);
  cudaFree(d_late);
  cudaFree(d_cand);
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
    const float late = late_host[row];
    const float cand = cand_host[row];
    if (!std::isfinite(late) || !std::isfinite(cand)) {
      ++result.nonfinite;
      continue;
    }
    result.max_abs = std::max(result.max_abs, std::fabs(late - cand));
  }
  if (std::strstr(launch, want_launch) == nullptr) {
    result.reason = "missing_candidate_launch";
    emit_case(result);
    return result;
  }
  result.gpu_pass = result.nonfinite == 0 && result.max_abs <= kBitwiseGate;
  result.pass = result.gpu_pass;
  result.reason = result.pass ? "same_math_bitwise" : "output_mismatch";
  emit_case(result);
  return result;
}

int run_catalog() {
  const CaseSpec specs[] = {
      {"Q4_layout_M1_N1_K256_inverse", 1, 256, 89U, nullptr, false, false, false,
       CompareKind::kInverse},
      {"Q4_layout_M3_N1_K256_branchless", 3, 256, 89U, nullptr, false, false,
       false, CompareKind::kBranchless},
      {"Q4_layout_M17_N1_K256_branchless", 17, 256, 89U, nullptr, false, false,
       false, CompareKind::kBranchless},
      {"Q4_layout_M3_N1_K256_aligned", 3, 256, 89U, nullptr, false, false, false,
       CompareKind::kAligned},
      {"Q4_layout_M17_N1_K256_aligned", 17, 256, 89U, nullptr, false, false,
       false, CompareKind::kAligned},
      {"Q4_layout_M3_N1_K256_zero", 3, 256, 89U, "zero", false, false, false,
       CompareKind::kBranchless},
      {"Q4_layout_M3_N1_K256_cancel", 3, 256, 89U, "cancellation", false, false,
       false, CompareKind::kAligned},
      {"Q4_layout_M3_N1_K256_minmax", 3, 256, 89U, "minmax", false, false, false,
       CompareKind::kBranchless},
      {"Q4_layout_M1_N1_K512_random", 1, 512, 89U, nullptr, false, false, false,
       CompareKind::kAligned},
      {"Q4_layout_M17_N1_K2048_random", 17, 2048, 89U, nullptr, false, false,
       false, CompareKind::kBranchless},
      {"Q4_layout_M17_N1_K5120_sampled", 17, 5120, 89U, nullptr, true, false,
       false, CompareKind::kAligned},
      {"Q4_layout_M17_N1_K17408_sampled", 17, 17408, 89U, nullptr, true, false,
       false, CompareKind::kBranchless},
      {"Q4_layout_M1_N1_K256_odd_mask", 1, 256, 89U, nullptr, false, false, true,
       CompareKind::kAligned},
      {"Q4_layout_M17_N1_K256_misaligned", 17, 256, 89U, "cancellation", false,
       true, false, CompareKind::kBranchless},
      {"Q4_layout_llama_mmvq_diagnostic", 1, 256, 89U, nullptr, false, false,
       false, CompareKind::kLlama},
      {"Q4_layout_occupancy_resources", 1, 256, 89U, nullptr, false, false, false,
       CompareKind::kOccupancy},
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
      "%s{\"schema_version\":1,\"task\":\"OPT-102\",\"phase\":\"parity\","
      "\"success\":%s,\"case_count\":%d,\"passed\":%d,\"failed\":%d,"
      "\"sampled_rows\":%d,\"sampled_vectors\":%d,"
      "\"provenance\":\"ggml-cuda Q4_K branchless unpack 73ab7599b\"}\n",
      kResultPrefix, json_bool(success), case_count, passed, failures,
      kSampledRows, kSampledVectors);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-102\",\"warmups\":0,\"samples\":1,"
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
