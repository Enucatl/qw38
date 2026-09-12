#include "ffn_decode_path.cuh"
#include "opt110_llama_q4_adapter.cuh"
#include "q4k_decode_path.cuh"
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
#include <cuda_runtime.h>

namespace {

constexpr char kResultPrefix[] = "QW38_OPT110_LLAMA_Q4_ADAPTER_RESULT=";
constexpr char kCountsPrefix[] = "QW38_OPT110_NATIVE_COUNTS=";
constexpr char kCasePrefix[] = "QW38_OPT110_CASE=";
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr std::size_t kQ4KBytes = 144;
constexpr std::size_t kBlock = 256;
constexpr float kSmallAbs = 3.0e-4F;
constexpr float kProductionAbs = 0.0111372F;
constexpr float kFusedAbs = 0.02F;

struct Options final {
  const char* workload = nullptr;
  int warmups = 0;
  int samples = 0;
};

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--workload smoke|correctness|screen|acceptance|"
               "parity|primitive] [--warmups N] [--samples N]\n",
               argv0);
  return 2;
}

int parse_args(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--workload") == 0 && index + 1 < argc) {
      options->workload = argv[++index];
    } else if (std::strcmp(arg, "--warmups") == 0 && index + 1 < argc) {
      options->warmups = std::atoi(argv[++index]);
    } else if (std::strcmp(arg, "--samples") == 0 && index + 1 < argc) {
      options->samples = std::atoi(argv[++index]);
    } else if (arg[0] == '-') {
      return usage(argv[0]);
    }
  }
  return 0;
}

const char* default_workload(qw38::cuda::TestTier tier) {
  if (tier == qw38::cuda::TestTier::kSmoke) return "smoke";
  if (tier == qw38::cuda::TestTier::kScreen) return "screen";
  if (tier == qw38::cuda::TestTier::kAcceptance) return "acceptance";
  return "correctness";
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

void fill_q4(std::size_t rows, std::size_t columns,
             std::vector<std::uint8_t>* weights) {
  weights->assign(rows * (columns / kBlock) * kQ4KBytes, 0);
  for (std::size_t index = 0; index < weights->size(); ++index) {
    (*weights)[index] = static_cast<std::uint8_t>((index * 37 + 5) & 0x3FU);
  }
  for (std::size_t offset = 0; offset < weights->size(); offset += kQ4KBytes) {
    write_u16(weights->data() + offset, 0x3C00U);
    write_u16(weights->data() + offset + 2, 0x2C00U);
  }
}

void fill_activation(const char* pattern, std::size_t columns,
                     std::uint32_t seed, std::vector<std::uint16_t>* act) {
  act->resize(columns);
  for (std::size_t column = 0; column < columns; ++column) {
    float value = unit_normal(static_cast<std::uint32_t>(column), seed);
    if (std::strcmp(pattern, "zero") == 0) {
      value = 0.0F;
    } else if (std::strcmp(pattern, "pos") == 0) {
      value = 4.0F;
    } else if (std::strcmp(pattern, "neg") == 0) {
      value = -4.0F;
    } else if (std::strcmp(pattern, "nan") == 0 && column == 0) {
      value = std::nanf("");
    }
    (*act)[column] = float_to_bf16(value);
  }
}

void gemm_fp64(const std::vector<std::uint8_t>& weights, std::size_t rows,
               std::size_t columns, const std::vector<double>& activation,
               std::vector<double>* output) {
  output->assign(rows, 0.0);
  std::vector<float> decoded(kBlock);
  for (std::size_t row = 0; row < rows; ++row) {
    double sum = 0.0;
    for (std::size_t block = 0; block < columns / kBlock; ++block) {
      const std::uint8_t* packed =
          weights.data() + (row * (columns / kBlock) + block) * kQ4KBytes;
      if (!qw38::internal::decode_q4_k(packed, kQ4KBytes, decoded.data(),
                                       decoded.size())
               .is_ok()) {
        (*output)[row] = std::nan("");
        break;
      }
      for (std::size_t within = 0; within < kBlock; ++within) {
        sum += static_cast<double>(decoded[within]) *
               activation[block * kBlock + within];
      }
    }
    (*output)[row] = sum;
  }
}

double max_abs(const std::vector<float>& got, const std::vector<double>& ref,
               int* nonfinite) {
  double worst = 0.0;
  const std::size_t n = std::min(got.size(), ref.size());
  for (std::size_t index = 0; index < n; ++index) {
    const double left = static_cast<double>(got[index]);
    const double right = ref[index];
    if (!std::isfinite(left) || !std::isfinite(right)) {
      ++(*nonfinite);
      continue;
    }
    worst = std::max(worst, std::fabs(left - right));
  }
  return worst;
}

struct Timed final {
  float staging_ms = 0.0F;
  float dot_ms = 0.0F;
  float total_ms = 0.0F;
};

Timed time_pair(cudaEvent_t start, cudaEvent_t mid, cudaEvent_t stop) {
  Timed timed;
  cudaEventElapsedTime(&timed.staging_ms, start, mid);
  cudaEventElapsedTime(&timed.dot_ms, mid, stop);
  timed.total_ms = timed.staging_ms + timed.dot_ms;
  return timed;
}

int emit_case(const char* id, bool pass, double abs_err, int nonfinite,
              const char* reason) {
  std::printf(
      "%s{\"id\":\"%s\",\"pass\":%s,\"max_abs\":%.9g,\"nonfinite_count\":%d,"
      "\"reason\":\"%s\"}\n",
      kCasePrefix, id, pass ? "true" : "false", abs_err, nonfinite, reason);
  return pass ? 0 : 1;
}

int run_type_isolation() {
  const std::size_t columns = 256;
  std::vector<std::uint16_t> act;
  fill_activation("random", columns, 89U, &act);
  std::vector<__nv_bfloat16> host(columns);
  for (std::size_t index = 0; index < columns; ++index) {
    host[index] = from_bf16_bits(act[index]);
  }
  __nv_bfloat16* da = nullptr;
  qw38::cuda::Q8Block* dq8 = nullptr;
  qw38::cuda::opt110::BlockQ81* dq81 = nullptr;
  cudaError_t error = cudaMalloc(&da, columns * sizeof(__nv_bfloat16));
  if (error == cudaSuccess) {
    error = cudaMalloc(&dq8, qw38::cuda::q8_workspace_bytes(columns));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&dq81, qw38::cuda::opt110::q8_1_bytes(columns));
  }
  if (error != cudaSuccess) return fail_cuda("type malloc", error);
  error = cudaMemcpy(da, host.data(), columns * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_bf16_q8(da, dq8, columns, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::opt110::launch_quantize_bf16_block_q8_1(da, dq81,
                                                                columns, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  qw38::cuda::Q8Block host_q8{};
  qw38::cuda::opt110::BlockQ81 host_q81{};
  if (error == cudaSuccess) {
    error = cudaMemcpy(&host_q8, dq8, sizeof(host_q8), cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error =
        cudaMemcpy(&host_q81, dq81, sizeof(host_q81), cudaMemcpyDeviceToHost);
  }
  cudaFree(da);
  cudaFree(dq8);
  cudaFree(dq81);
  if (error != cudaSuccess) return fail_cuda("type copy", error);
  std::uint8_t q8_bytes[36];
  std::uint8_t q81_bytes[36];
  std::memcpy(q8_bytes, &host_q8, 36);
  std::memcpy(q81_bytes, &host_q81, 36);
  const bool same_size = sizeof(qw38::cuda::Q8Block) ==
                         sizeof(qw38::cuda::opt110::BlockQ81);
  const bool bytes_differ = std::memcmp(q8_bytes, q81_bytes, 36) != 0;
  const bool scale_is_float = std::isfinite(host_q8.scale);
  const float q81_d = __half2float(__low2half(host_q81.ds));
  const bool q81_half_scale = std::isfinite(q81_d);
  const bool pass = same_size && bytes_differ && scale_is_float && q81_half_scale;
  std::printf(
      "type_isolation same_size=%s bytes_differ=%s q8block_scale_f32=%.9g "
      "block_q8_1_d_f16=%.9g q8block_hex=%02x%02x%02x%02x "
      "q81_hex=%02x%02x%02x%02x reinterpret_forbidden=true\n",
      same_size ? "true" : "false", bytes_differ ? "true" : "false",
      host_q8.scale, q81_d, q8_bytes[0], q8_bytes[1], q8_bytes[2], q8_bytes[3],
      q81_bytes[0], q81_bytes[1], q81_bytes[2], q81_bytes[3]);
  return emit_case("typed_activation_producers", pass, 0.0, 0,
                   pass ? "q8block_vs_block_q8_1" : "type_mix");
}

int run_mmv_parity(const char* id, std::size_t rows, std::size_t columns,
                   const char* pattern) {
  std::vector<std::uint8_t> weights;
  std::vector<std::uint16_t> act;
  fill_q4(rows, columns, &weights);
  fill_activation(pattern, columns, 89U, &act);
  std::vector<__nv_bfloat16> host_act(columns);
  for (std::size_t index = 0; index < columns; ++index) {
    host_act[index] = from_bf16_bits(act[index]);
  }
  std::uint8_t* dw = nullptr;
  __nv_bfloat16* da = nullptr;
  qw38::cuda::Q8Block* dq8 = nullptr;
  qw38::cuda::opt110::BlockQ81* dq81 = nullptr;
  float* dout_c = nullptr;
  float* dout_k = nullptr;
  cudaError_t error = cudaMalloc(&dw, weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&da, columns * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&dq8, qw38::cuda::q8_workspace_bytes(columns));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&dq81, qw38::cuda::opt110::q8_1_bytes(columns));
  }
  if (error == cudaSuccess) error = cudaMalloc(&dout_c, rows * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&dout_k, rows * sizeof(float));
  if (error != cudaSuccess) return fail_cuda("parity malloc", error);
  error = cudaMemcpy(dw, weights.data(), weights.size(), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(da, host_act.data(), columns * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_bf16_q8(da, dq8, columns, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::opt110::launch_quantize_bf16_block_q8_1(da, dq81,
                                                                columns, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<qw38::cuda::Q8Block> host_q8(columns / 32U);
  std::vector<qw38::cuda::opt110::BlockQ81> host_q81(columns / 32U);
  if (error == cudaSuccess) {
    error = cudaMemcpy(host_q8.data(), dq8,
                       host_q8.size() * sizeof(qw38::cuda::Q8Block),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(host_q81.data(), dq81,
                       host_q81.size() * sizeof(qw38::cuda::opt110::BlockQ81),
                       cudaMemcpyDeviceToHost);
  }
  qw38::cuda::Q4DecodePathScope scope(qw38::cuda::kLegalQ4DecodePathIntegerQ8Late,
                                      4U);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q4k_coop_mmv_prequant_q8(
        dw, rows, columns, dq8, dout_c, 4U, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::opt110::launch_mmvq_q4k_q8_1(
        reinterpret_cast<const qw38::cuda::opt110::BlockQ4K*>(dw), rows,
        columns, dq81, dout_k, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<float> got_c(rows);
  std::vector<float> got_k(rows);
  if (error == cudaSuccess) {
    error = cudaMemcpy(got_c.data(), dout_c, rows * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(got_k.data(), dout_k, rows * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  cudaFree(dw);
  cudaFree(da);
  cudaFree(dq8);
  cudaFree(dq81);
  cudaFree(dout_c);
  cudaFree(dout_k);
  if (error != cudaSuccess) return fail_cuda("parity launch", error);

  std::vector<double> ctrl_act(columns);
  std::vector<double> cand_act(columns);
  for (std::size_t group = 0; group < columns / 32U; ++group) {
    for (int lane = 0; lane < 32; ++lane) {
      ctrl_act[group * 32U + static_cast<std::size_t>(lane)] =
          static_cast<double>(host_q8[group].scale *
                              static_cast<float>(host_q8[group].values[lane]));
      const float d = __half2float(__low2half(host_q81[group].ds));
      cand_act[group * 32U + static_cast<std::size_t>(lane)] =
          static_cast<double>(d * static_cast<float>(host_q81[group].qs[lane]));
    }
  }
  std::vector<double> fp64_c;
  std::vector<double> fp64_k;
  gemm_fp64(weights, rows, columns, ctrl_act, &fp64_c);
  gemm_fp64(weights, rows, columns, cand_act, &fp64_k);
  int nonfinite = 0;
  const double abs_c = max_abs(got_c, fp64_c, &nonfinite);
  const double abs_k = max_abs(got_k, fp64_k, &nonfinite);
  const float control_gate =
      (rows <= 33 && columns <= 512) ? kSmallAbs : kProductionAbs;
  const float cand_gate = kProductionAbs;
  const bool nan_ok =
      std::strcmp(pattern, "nan") != 0 || nonfinite > 0 || abs_c > 0.0;
  const bool pass = (std::strcmp(pattern, "nan") == 0)
                        ? nan_ok
                        : (nonfinite == 0 &&
                           abs_c <= static_cast<double>(control_gate) &&
                           abs_k <= static_cast<double>(cand_gate));
  const char* launch = qw38::cuda::last_q4_launch_variant();
  const auto info = qw38::cuda::opt110::last_launch();
  std::printf(
      "mmv id=%s rows=%zu cols=%zu control_abs=%.9g candidate_abs=%.9g "
      "control_launch=%s candidate_launch=%s nwarps=%d vdr=%d occupancy=%d "
      "regs=%d local=%zu staging_q8block=%zu staging_q81=%zu\n",
      id, rows, columns, abs_c, abs_k, launch, info.launch_name, info.nwarps,
      info.vdr, info.occupancy, info.registers, info.local_bytes,
      qw38::cuda::q8_workspace_bytes(columns),
      qw38::cuda::opt110::q8_1_bytes(columns));
  return emit_case(id, pass, std::max(abs_c, abs_k), nonfinite,
                   pass ? "typed_fp64" : "parity_fail");
}

int run_fused_parity(const char* id, std::size_t rows, std::size_t columns) {
  std::vector<std::uint8_t> gate;
  std::vector<std::uint8_t> up;
  std::vector<std::uint16_t> act;
  fill_q4(rows, columns, &gate);
  fill_q4(rows, columns, &up);
  fill_activation("random", columns, 101U, &act);
  std::vector<__nv_bfloat16> host_act(columns);
  for (std::size_t index = 0; index < columns; ++index) {
    host_act[index] = from_bf16_bits(act[index]);
  }
  std::uint8_t* dg = nullptr;
  std::uint8_t* du = nullptr;
  __nv_bfloat16* da = nullptr;
  qw38::cuda::Q8Block* dq8 = nullptr;
  qw38::cuda::opt110::BlockQ81* dq81 = nullptr;
  __nv_bfloat16* dout_c = nullptr;
  __nv_bfloat16* dout_k = nullptr;
  cudaError_t error = cudaMalloc(&dg, gate.size());
  if (error == cudaSuccess) error = cudaMalloc(&du, up.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&da, columns * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&dq8, qw38::cuda::q8_workspace_bytes(columns));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&dq81, qw38::cuda::opt110::q8_1_bytes(columns));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&dout_c, rows * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&dout_k, rows * sizeof(__nv_bfloat16));
  }
  if (error != cudaSuccess) return fail_cuda("fused malloc", error);
  error = cudaMemcpy(dg, gate.data(), gate.size(), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(du, up.data(), up.size(), cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(da, host_act.data(), columns * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_bf16_q8(da, dq8, columns, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::opt110::launch_quantize_bf16_block_q8_1(da, dq81,
                                                                columns, nullptr);
  }
  qw38::cuda::Q4DecodePathScope scope(qw38::cuda::kLegalQ4DecodePathIntegerQ8Late,
                                      4U);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q4k_coop_gate_up_swiglu_prequant_q8(
        dg, du, rows, columns, dq8, dout_c, 4U, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::opt110::launch_mmvq_q4k_q8_1_swiglu(
        reinterpret_cast<const qw38::cuda::opt110::BlockQ4K*>(dg),
        reinterpret_cast<const qw38::cuda::opt110::BlockQ4K*>(du), rows,
        columns, dq81, dout_k, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<__nv_bfloat16> got_c(rows);
  std::vector<__nv_bfloat16> got_k(rows);
  if (error == cudaSuccess) {
    error = cudaMemcpy(got_c.data(), dout_c, rows * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(got_k.data(), dout_k, rows * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost);
  }
  cudaFree(dg);
  cudaFree(du);
  cudaFree(da);
  cudaFree(dq8);
  cudaFree(dq81);
  cudaFree(dout_c);
  cudaFree(dout_k);
  if (error != cudaSuccess) return fail_cuda("fused launch", error);
  double worst = 0.0;
  int nonfinite = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    const float left = __bfloat162float(got_c[row]);
    const float right = __bfloat162float(got_k[row]);
    if (!std::isfinite(left) || !std::isfinite(right)) {
      ++nonfinite;
      continue;
    }
    worst = std::max(worst, static_cast<double>(std::fabs(left - right)));
  }
  const bool pass = nonfinite == 0 && worst <= static_cast<double>(kFusedAbs);
  return emit_case(id, pass, worst, nonfinite,
                   pass ? "same_input_glu" : "fused_mismatch");
}

int run_graph_eager(std::size_t rows, std::size_t columns) {
  std::vector<std::uint8_t> weights;
  std::vector<std::uint16_t> act;
  fill_q4(rows, columns, &weights);
  fill_activation("random", columns, 7U, &act);
  std::vector<__nv_bfloat16> host_act(columns);
  for (std::size_t index = 0; index < columns; ++index) {
    host_act[index] = from_bf16_bits(act[index]);
  }
  std::uint8_t* dw = nullptr;
  __nv_bfloat16* da = nullptr;
  qw38::cuda::opt110::BlockQ81* dq81 = nullptr;
  float* dout_e = nullptr;
  float* dout_g = nullptr;
  cudaError_t error = cudaMalloc(&dw, weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&da, columns * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&dq81, qw38::cuda::opt110::q8_1_bytes(columns));
  }
  if (error == cudaSuccess) error = cudaMalloc(&dout_e, rows * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&dout_g, rows * sizeof(float));
  if (error != cudaSuccess) return fail_cuda("graph malloc", error);
  error = cudaMemcpy(dw, weights.data(), weights.size(), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(da, host_act.data(), columns * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::opt110::launch_quantize_bf16_block_q8_1(da, dq81,
                                                                columns, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::opt110::launch_mmvq_q4k_q8_1(
        reinterpret_cast<const qw38::cuda::opt110::BlockQ4K*>(dw), rows,
        columns, dq81, dout_e, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  cudaStream_t stream = nullptr;
  if (error == cudaSuccess) error = cudaStreamCreate(&stream);
  if (error == cudaSuccess) {
    error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::opt110::launch_quantize_bf16_block_q8_1(da, dq81,
                                                                columns, stream);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::opt110::launch_mmvq_q4k_q8_1(
        reinterpret_cast<const qw38::cuda::opt110::BlockQ4K*>(dw), rows,
        columns, dq81, dout_g, stream);
  }
  if (error == cudaSuccess) error = cudaStreamEndCapture(stream, &graph);
  if (error == cudaSuccess) {
    error = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
  }
  if (error == cudaSuccess) error = cudaGraphLaunch(exec, stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  std::vector<float> eager(rows);
  std::vector<float> graphed(rows);
  if (error == cudaSuccess) {
    error = cudaMemcpy(eager.data(), dout_e, rows * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(graphed.data(), dout_g, rows * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (exec != nullptr) cudaGraphExecDestroy(exec);
  if (graph != nullptr) cudaGraphDestroy(graph);
  if (stream != nullptr) cudaStreamDestroy(stream);
  cudaFree(dw);
  cudaFree(da);
  cudaFree(dq81);
  cudaFree(dout_e);
  cudaFree(dout_g);
  if (error != cudaSuccess) return fail_cuda("graph", error);
  bool same = eager.size() == graphed.size();
  int nonfinite = 0;
  for (std::size_t index = 0; index < eager.size(); ++index) {
    if (!std::isfinite(eager[index]) || !std::isfinite(graphed[index])) {
      ++nonfinite;
      same = false;
    } else if (eager[index] != graphed[index]) {
      same = false;
    }
  }
  return emit_case("graph_eager", same && nonfinite == 0, 0.0, nonfinite,
                   same ? "bitwise" : "graph_mismatch");
}

struct PrimitiveRow final {
  const char* role;
  std::size_t rows;
  std::size_t columns;
  bool fused;
};

int run_primitive(int warmups, int samples) {
  const PrimitiveRow rows[] = {
      {"down", 5120, 17408, false},
      {"gate_up", 17408, 5120, true},
  };
  int failed = 0;
  for (const auto& spec : rows) {
    std::vector<std::uint8_t> w0;
    std::vector<std::uint8_t> w1;
    std::vector<std::uint16_t> act;
    fill_q4(spec.rows, spec.columns, &w0);
    if (spec.fused) fill_q4(spec.rows, spec.columns, &w1);
    fill_activation("random", spec.columns, 42U, &act);
    std::vector<__nv_bfloat16> host_act(spec.columns);
    for (std::size_t index = 0; index < spec.columns; ++index) {
      host_act[index] = from_bf16_bits(act[index]);
    }
    std::uint8_t* dw0 = nullptr;
    std::uint8_t* dw1 = nullptr;
    __nv_bfloat16* da = nullptr;
    qw38::cuda::Q8Block* dq8 = nullptr;
    qw38::cuda::opt110::BlockQ81* dq81 = nullptr;
    float* dout_f = nullptr;
    __nv_bfloat16* dout_b0 = nullptr;
    __nv_bfloat16* dout_b1 = nullptr;
    cudaError_t error = cudaMalloc(&dw0, w0.size());
    if (spec.fused && error == cudaSuccess) error = cudaMalloc(&dw1, w1.size());
    if (error == cudaSuccess) {
      error = cudaMalloc(&da, spec.columns * sizeof(__nv_bfloat16));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(&dq8, qw38::cuda::q8_workspace_bytes(spec.columns));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(&dq81, qw38::cuda::opt110::q8_1_bytes(spec.columns));
    }
    if (!spec.fused && error == cudaSuccess) {
      error = cudaMalloc(&dout_f, spec.rows * sizeof(float));
    }
    if (spec.fused && error == cudaSuccess) {
      error = cudaMalloc(&dout_b0, spec.rows * sizeof(__nv_bfloat16));
    }
    if (spec.fused && error == cudaSuccess) {
      error = cudaMalloc(&dout_b1, spec.rows * sizeof(__nv_bfloat16));
    }
    if (error != cudaSuccess) return fail_cuda("primitive malloc", error);
    error = cudaMemcpy(dw0, w0.data(), w0.size(), cudaMemcpyHostToDevice);
    if (spec.fused && error == cudaSuccess) {
      error = cudaMemcpy(dw1, w1.data(), w1.size(), cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(da, host_act.data(),
                         spec.columns * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice);
    }
    if (error != cudaSuccess) return fail_cuda("primitive h2d", error);
    cudaEvent_t start = nullptr;
    cudaEvent_t mid = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&mid);
    cudaEventCreate(&stop);
    std::vector<float> control_total;
    std::vector<float> candidate_total;
    std::vector<float> control_stage;
    std::vector<float> candidate_stage;
    std::vector<float> control_dot;
    std::vector<float> candidate_dot;
    auto once = [&](bool candidate) -> Timed {
      qw38::cuda::Q4DecodePathScope scope(
          qw38::cuda::kLegalQ4DecodePathIntegerQ8Late, 4U);
      cudaEventRecord(start, nullptr);
      if (candidate) {
        qw38::cuda::opt110::launch_quantize_bf16_block_q8_1(da, dq81,
                                                            spec.columns,
                                                            nullptr);
      } else {
        qw38::cuda::launch_quantize_bf16_q8(da, dq8, spec.columns, nullptr);
      }
      cudaEventRecord(mid, nullptr);
      if (candidate) {
        if (spec.fused) {
          qw38::cuda::opt110::launch_mmvq_production_swiglu<17408, 5120>(
              reinterpret_cast<const qw38::cuda::opt110::BlockQ4K*>(dw0),
              reinterpret_cast<const qw38::cuda::opt110::BlockQ4K*>(dw1),
              dq81, dout_b1, nullptr);
        } else {
          qw38::cuda::opt110::launch_mmvq_production<5120, 17408>(
              reinterpret_cast<const qw38::cuda::opt110::BlockQ4K*>(dw0),
              dq81, dout_f, nullptr);
        }
      } else if (spec.fused) {
        qw38::cuda::launch_q4k_coop_gate_up_swiglu_prequant_q8(
            dw0, dw1, spec.rows, spec.columns, dq8, dout_b0, 4U, nullptr);
      } else {
        qw38::cuda::launch_q4k_coop_mmv_prequant_q8(
            dw0, spec.rows, spec.columns, dq8, dout_f, 4U, nullptr);
      }
      cudaEventRecord(stop, nullptr);
      cudaEventSynchronize(stop);
      return time_pair(start, mid, stop);
    };
    for (int warm = 0; warm < warmups; ++warm) {
      once(false);
      once(true);
    }
    for (int sample = 0; sample < samples; ++sample) {
      const bool ab = sample % 2 == 0;
      const Timed first = once(!ab);
      const Timed second = once(ab);
      const Timed control = ab ? first : second;
      const Timed candidate = ab ? second : first;
      control_total.push_back(control.total_ms);
      candidate_total.push_back(candidate.total_ms);
      control_stage.push_back(control.staging_ms);
      candidate_stage.push_back(candidate.staging_ms);
      control_dot.push_back(control.dot_ms);
      candidate_dot.push_back(candidate.dot_ms);
      std::printf(
          "%s{\"role\":\"%s\",\"path\":\"%s\",\"ms\":%.9g,\"staging_ms\":%.9g,"
          "\"dot_ms\":%.9g,\"sample\":%d}\n",
          kCasePrefix, spec.role, "integer_q8_late", control.total_ms,
          control.staging_ms, control.dot_ms, sample);
      std::printf(
          "%s{\"role\":\"%s\",\"path\":\"%s\",\"ms\":%.9g,\"staging_ms\":%.9g,"
          "\"dot_ms\":%.9g,\"sample\":%d}\n",
          kCasePrefix, spec.role, "llama_q4k_mmvq", candidate.total_ms,
          candidate.staging_ms, candidate.dot_ms, sample);
    }
    auto mean = [](const std::vector<float>& values) {
      double sum = 0.0;
      for (float item : values) sum += static_cast<double>(item);
      return values.empty() ? 0.0 : sum / static_cast<double>(values.size());
    };
    const double c_mean = mean(control_total);
    const double k_mean = mean(candidate_total);
    const double saving = c_mean - k_mean;
    std::printf(
        "primitive_role=%s rows=%zu cols=%zu fused=%s control_ms=%.9g "
        "candidate_ms=%.9g saving_ms=%.9g control_stage=%.9g "
        "candidate_stage=%.9g control_dot=%.9g candidate_dot=%.9g "
        "faster=%s launch=%s nwarps=%d occupancy=%d regs=%d local=%zu "
        "staging_q8block=%zu staging_q81=%zu\n",
        spec.role, spec.rows, spec.columns, spec.fused ? "true" : "false",
        c_mean, k_mean, saving, mean(control_stage), mean(candidate_stage),
        mean(control_dot), mean(candidate_dot),
        saving > 0.0 ? "true" : "false",
        qw38::cuda::opt110::last_launch().launch_name,
        qw38::cuda::opt110::last_launch().nwarps,
        qw38::cuda::opt110::last_launch().occupancy,
        qw38::cuda::opt110::last_launch().registers,
        qw38::cuda::opt110::last_launch().local_bytes,
        qw38::cuda::q8_workspace_bytes(spec.columns),
        qw38::cuda::opt110::q8_1_bytes(spec.columns));
    cudaEventDestroy(start);
    cudaEventDestroy(mid);
    cudaEventDestroy(stop);
    cudaFree(dw0);
    cudaFree(dw1);
    cudaFree(da);
    cudaFree(dq8);
    cudaFree(dq81);
    cudaFree(dout_f);
    cudaFree(dout_b0);
    cudaFree(dout_b1);
  }
  return failed;
}

int run_attrs_and_sass() {
  int regs = 0;
  std::size_t local = 0;
  int occ = 0;
  qw38::cuda::opt110::mmvq_kernel_attributes(&regs, &local, &occ);
  int glu_regs = 0;
  std::size_t glu_local = 0;
  int glu_occ = 0;
  qw38::cuda::opt110::mmvq_swiglu_kernel_attributes(&glu_regs, &glu_local,
                                                    &glu_occ);
  int q_regs = 0;
  std::size_t q_local = 0;
  int q_occ = 0;
  qw38::cuda::opt110::quantize_kernel_attributes(&q_regs, &q_local, &q_occ);
  int late_occ = qw38::cuda::q4k_coop_late_occupancy(4U);
  std::printf(
      "attrs mmvq_regs=%d mmvq_local=%zu mmvq_occ=%d glu_regs=%d "
      "glu_local=%zu glu_occ=%d quant_regs=%d quant_local=%zu quant_occ=%d "
      "late_occ=%d nwarps=%d vdr=%d launch_geometry=32x4\n",
      regs, local, occ, glu_regs, glu_local, glu_occ, q_regs, q_local, q_occ,
      late_occ, qw38::cuda::opt110::kNwarps, qw38::cuda::opt110::kVdr);
  FILE* pipe = popen(
      "cuobjdump -sass build/qw38-cuda-opt110-llama-q4-adapter-test "
      "2>/dev/null",
      "r");
  int dp4a = 0;
  int shfl = 0;
  int ldg = 0;
  if (pipe == nullptr) {
    std::printf("sass_status=unavailable reason=popen_failed\n");
    return 0;
  }
  char line[512];
  while (fgets(line, sizeof(line), pipe) != nullptr) {
    if (std::strstr(line, "DP4A") != nullptr ||
        std::strstr(line, "IDP.4A") != nullptr) {
      ++dp4a;
    }
    if (std::strstr(line, "SHFL") != nullptr) ++shfl;
    if (std::strstr(line, "LDG") != nullptr) ++ldg;
  }
  const int status = pclose(pipe);
  std::printf("sass_status=%s dp4a_or_idp=%d shfl=%d ldg=%d\n",
              status == 0 ? "ok" : "unavailable", dp4a, shfl, ldg);
  return 0;
}

int run_parity() {
  int rc = 0;
  rc |= run_type_isolation();
  rc |= run_mmv_parity("Q4_mmv_M1_N1_K256", 1, 256, "random");
  rc |= run_mmv_parity("Q4_mmv_M3_N1_K256", 3, 256, "random");
  rc |= run_mmv_parity("Q4_mmv_M17_N1_K256_tail", 17, 256, "random");
  rc |= run_mmv_parity("Q4_mmv_zero", 1, 256, "zero");
  rc |= run_mmv_parity("Q4_mmv_pos", 1, 256, "pos");
  rc |= run_mmv_parity("Q4_mmv_neg", 1, 256, "neg");
  rc |= run_mmv_parity("Q4_mmv_nan", 1, 256, "nan");
  rc |= run_fused_parity("Q4_fused_M3_K256", 3, 256);
  rc |= run_graph_eager(3, 256);
  rc |= run_mmv_parity("Q4_mmv_down_sampled", 16, 17408, "random");
  rc |= run_mmv_parity("Q4_mmv_gate_sampled", 16, 5120, "random");
  rc |= run_attrs_and_sass();
  return rc;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (parse_args(argc, argv, &options) != 0) return 2;
  const char* workload = options.workload;
  if (workload == nullptr) {
    workload = default_workload(qw38::cuda::test_tier());
  }
  int warmups = options.warmups;
  int samples = options.samples;
  if (std::strcmp(workload, "acceptance") == 0 ||
      std::strcmp(workload, "primitive") == 0) {
    if (warmups == 0) warmups = 3;
    if (samples == 0) samples = 10;
  } else if (std::strcmp(workload, "screen") == 0) {
    if (warmups == 0) warmups = 1;
    if (samples == 0) samples = 3;
  }
  int rc = 0;
  if (std::strcmp(workload, "smoke") == 0) {
    rc = run_mmv_parity("smoke", 1, 256, "random");
  } else if (std::strcmp(workload, "primitive") == 0 ||
             std::strcmp(workload, "screen") == 0 ||
             std::strcmp(workload, "acceptance") == 0) {
    rc = run_parity();
    if (rc == 0) rc = run_primitive(warmups, samples);
  } else {
    rc = run_parity();
  }
  const bool pass = rc == 0;
  std::printf(
      "%s{\"task\":\"OPT-110\",\"workload\":\"%s\",\"pass\":%s,"
      "\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\","
      "\"control\":\"integer_q8_late\",\"candidate\":\"llama_q4k_mmvq\","
      "\"nwarps\":4,\"vdr\":2}\n",
      kResultPrefix, workload, pass ? "true" : "false", kLlamaRev, kGgufSha);
  std::printf("%s{\"cases\":1,\"candidates\":2,\"warmups\":%d,\"samples\":%d}\n",
              kCountsPrefix, warmups, samples);
  return rc;
}
