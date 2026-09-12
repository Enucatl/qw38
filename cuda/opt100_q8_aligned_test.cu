#include "q8_aligned_layout.cuh"
#include "q8_decode_path.cuh"
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
#include <cuda_runtime.h>

namespace {

constexpr char kResultPrefix[] = "QW38_OPT100_RESULT=";
constexpr char kCountsPrefix[] = "QW38_OPT100_NATIVE_COUNTS=";
constexpr char kCasePrefix[] = "QW38_OPT100_CASE=";
constexpr std::size_t kQ80Bytes = 34;
constexpr std::size_t kQ80Values = 32;

struct CaseSpec final {
  const char* id;
  std::size_t rows;
  std::size_t columns;
  std::uint32_t seed;
  bool grouped;
  std::size_t group_rows[4];
  int group_count;
};

struct CaseResult final {
  std::string id;
  bool pass = false;
  bool inverse_ok = false;
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

const char* default_phase(qw38::cuda::TestTier tier) {
  if (tier == qw38::cuda::TestTier::kSmoke) return "smoke";
  if (tier == qw38::cuda::TestTier::kScreen) return "correctness";
  return "parity";
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

void fill_weights(std::size_t rows, std::size_t columns, std::uint32_t seed,
                  std::vector<std::uint8_t>* weights) {
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
  const auto desc = qw38::cuda::make_q8_aligned_layout_desc(rows, columns);
  if (!qw38::cuda::q8_aligned_desc_valid(desc) ||
      desc.gguf_bytes != gguf.size()) {
    return false;
  }
  std::vector<std::uint8_t> soa(desc.total_bytes);
  std::vector<std::uint8_t> back(desc.gguf_bytes);
  qw38::cuda::pack_q8_aligned_host(gguf.data(), desc, soa.data());
  qw38::cuda::unpack_q8_aligned_host(soa.data(), desc, back.data());
  return qw38::cuda::q8_aligned_inverse_matches(gguf.data(), back.data(), desc);
}

CaseResult run_case(const CaseSpec& spec) {
  CaseResult result;
  result.id = spec.id;
  if (spec.columns == 0 || spec.columns % kQ80Values != 0) {
    result.reason = "invalid_columns";
    emit_case(result);
    return result;
  }
  std::vector<std::uint8_t> host_w;
  fill_weights(spec.rows, spec.columns, spec.seed, &host_w);
  result.inverse_ok = host_inverse_ok(host_w, spec.rows, spec.columns);
  if (!result.inverse_ok) {
    result.reason = "host_inverse_mismatch";
    emit_case(result);
    return result;
  }

  std::vector<__nv_bfloat16> activation;
  fill_activation(spec.columns, spec.seed, &activation);
  std::uint8_t* d_raw = nullptr;
  std::uint8_t* d_soa = nullptr;
  std::uint8_t* d_back = nullptr;
  __nv_bfloat16* d_a = nullptr;
  void* d_s = nullptr;
  float* d_raw_out = nullptr;
  float* d_soa_out = nullptr;
  const auto desc =
      qw38::cuda::make_q8_aligned_layout_desc(spec.rows, spec.columns);
  cudaError_t error = cudaMalloc(&d_raw, host_w.size());
  if (error == cudaSuccess) error = cudaMalloc(&d_soa, desc.total_bytes);
  if (error == cudaSuccess) error = cudaMalloc(&d_back, host_w.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&d_a, spec.columns * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&d_s, (spec.columns / kQ80Values) * 36U);
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&d_raw_out, spec.rows * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&d_soa_out, spec.rows * sizeof(float));
  }
  if (error != cudaSuccess) {
    result.reason = "alloc_failed";
    emit_case(result);
    return result;
  }
  error = cudaMemcpy(d_raw, host_w.data(), host_w.size(), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_repack_q8_0_aligned(
        d_raw, spec.rows, spec.columns, d_soa, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_unpack_q8_0_aligned(
        d_soa, spec.rows, spec.columns, d_back, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<std::uint8_t> back(host_w.size());
  if (error == cudaSuccess) {
    error = cudaMemcpy(back.data(), d_back, back.size(), cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess || back != host_w) {
    result.reason = error != cudaSuccess ? "gpu_inverse_failed"
                                         : "gpu_inverse_mismatch";
    emit_case(result);
    cudaFree(d_raw);
    cudaFree(d_soa);
    cudaFree(d_back);
    cudaFree(d_a);
    cudaFree(d_s);
    cudaFree(d_raw_out);
    cudaFree(d_soa_out);
    return result;
  }

  error = cudaMemcpy(d_a, activation.data(),
                     spec.columns * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_bf16_q8_1(d_a, d_s, spec.columns,
                                                  nullptr);
  }
  if (error == cudaSuccess) {
    qw38::cuda::Q8DeviceLayoutScope raw(qw38::cuda::kLegalQ8DeviceLayoutRawGguf);
    error = qw38::cuda::launch_q8_coop_mmv_prequant(
        d_raw, spec.rows, spec.columns, d_s, d_raw_out, 1U, 4U, nullptr);
  }
  if (error == cudaSuccess) {
    qw38::cuda::Q8DeviceLayoutScope soa(qw38::cuda::kLegalQ8DeviceLayoutAlignedSoa);
    error = qw38::cuda::launch_q8_coop_mmv_prequant(
        d_soa, spec.rows, spec.columns, d_s, d_soa_out, 1U, 4U, nullptr);
  }
  if (error == cudaSuccess && spec.grouped) {
    qw38::cuda::Q8GroupedProjDesc host[4]{};
    host[0].weights = d_soa;
    host[0].rows = spec.rows;
    host[0].output = d_soa_out;
    qw38::cuda::Q8GroupedProjDesc* d_desc = nullptr;
    error = cudaMalloc(&d_desc, sizeof(host));
    if (error == cudaSuccess) {
      qw38::cuda::Q8DeviceLayoutScope soa(
          qw38::cuda::kLegalQ8DeviceLayoutAlignedSoa);
      error = qw38::cuda::upload_q8_grouped_descriptors(d_desc, host, nullptr);
      if (error == cudaSuccess) {
        error = qw38::cuda::launch_q8_coop_mmv_grouped_r1_w4(
            d_desc, host, spec.columns, d_s, nullptr);
      }
    }
    cudaFree(d_desc);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<float> raw_out(spec.rows);
  std::vector<float> soa_out(spec.rows);
  if (error == cudaSuccess) {
    error = cudaMemcpy(raw_out.data(), d_raw_out, spec.rows * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(soa_out.data(), d_soa_out, spec.rows * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  cudaFree(d_raw);
  cudaFree(d_soa);
  cudaFree(d_back);
  cudaFree(d_a);
  cudaFree(d_s);
  cudaFree(d_raw_out);
  cudaFree(d_soa_out);
  if (error != cudaSuccess) {
    result.reason = "kernel_failed";
    emit_case(result);
    return result;
  }
  for (std::size_t row = 0; row < spec.rows; ++row) {
    if (!std::isfinite(raw_out[row]) || !std::isfinite(soa_out[row])) {
      ++result.nonfinite;
    }
    result.max_abs =
        std::max(result.max_abs, std::fabs(raw_out[row] - soa_out[row]));
  }
  result.gpu_pass = result.nonfinite == 0 && result.max_abs == 0.0F;
  result.pass = result.inverse_ok && result.gpu_pass;
  result.reason = result.pass ? "bitwise_match" : "output_mismatch";
  emit_case(result);
  return result;
}

int run_catalog(const char* phase) {
  const bool smoke = std::strcmp(phase, "smoke") == 0;
  const CaseSpec specs[] = {
      {"Q8_aligned_M1_N1_K256_inverse_mmv", 1, 256, 0x11U, false, {}, 0},
      {"Q8_aligned_M3_N1_K256_pad_inverse_mmv", 3, 256, 0x12U, false, {}, 0},
      {"Q8_aligned_M17_N1_K256_inverse_mmv", 17, 256, 0x13U, false, {}, 0},
      {"Q8_aligned_M1_N1_K5120_inverse_mmv", 1, 5120, 0x14U, false, {}, 0},
      {"Q8_aligned_M3_N1_K5120_inverse_mmv", 3, 5120, 0x15U, false, {}, 0},
      {"Q8_aligned_M17_N1_K5120_inverse_mmv", 17, 5120, 0x16U, false, {}, 0},
      {"Q8_aligned_M48_N1_K5120_skinny_alpha", 48, 5120, 0x17U, false, {}, 0},
      {"Q8_aligned_M1024_N1_K5120_sampled", 1024, 5120, 0x18U, false, {}, 0},
      {"Q8_aligned_gdn_output_M5120_K6144", 5120, 6144, 0x19U, false, {}, 0},
      {"Q8_aligned_grouped_gdn_input", 10240, 5120, 0x1aU, true, {10240}, 1},
      {"Q8_aligned_grouped_attn_input", 12288, 5120, 0x1bU, true, {12288}, 1},
      {"Q8_aligned_grouped_empty_ok", 0, 256, 0x1cU, false, {}, 0},
  };
  int ran = 0;
  int passed = 0;
  const int limit = smoke ? 4 : 12;
  for (int index = 0; index < limit; ++index) {
    if (specs[index].rows == 0) {
      CaseResult empty;
      empty.id = specs[index].id;
      empty.inverse_ok = true;
      empty.gpu_pass = true;
      empty.pass = true;
      empty.reason = "empty_ok";
      emit_case(empty);
      ++ran;
      ++passed;
      continue;
    }
    const CaseResult result = run_case(specs[index]);
    ++ran;
    if (result.pass) ++passed;
  }
  int registers = 0;
  std::size_t local_bytes = 0;
  int occupancy = 0;
  {
    qw38::cuda::Q8DeviceLayoutScope soa(qw38::cuda::kLegalQ8DeviceLayoutAlignedSoa);
    qw38::cuda::q8_coop_kernel_attributes(1U, 4U, &registers, &local_bytes,
                                          &occupancy);
  }
  const bool ok = passed == ran;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-100\",\"phase\":\"%s\","
      "\"cases\":%d,\"passed\":%d,\"kernel_parity_pass\":%s,"
      "\"aligned_r1_w4_registers\":%d,\"aligned_r1_w4_local_bytes\":%zu,"
      "\"aligned_r1_w4_occupancy\":%d,\"d2r_not_selected\":%s}\n",
      kCountsPrefix, phase, ran, passed, json_bool(ok), registers, local_bytes,
      occupancy,
      json_bool(std::strcmp(qw38::cuda::selected_large_mixer_q8_path(),
                            "quality_mma") == 0));
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-100\",\"phase\":\"%s\","
      "\"keep\":false,\"claims_throughput\":false,\"kernel_parity_pass\":%s,"
      "\"cases\":%d,\"passed\":%d}\n",
      kResultPrefix, phase, json_bool(ok), ran, passed);
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const char* phase = default_phase(qw38::cuda::test_tier());
  if (parse_args(argc, argv, &phase) != 0) return 2;
  return run_catalog(phase);
}
