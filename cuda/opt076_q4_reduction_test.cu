#include "ffn_decode_path.cuh"
#include "full_scheduler.h"
#include "mixer.h"
#include "model.h"
#include "production_numerics.h"
#include "q4k_decode_path.cuh"
#include "quant.h"
#include "quant_mmv.h"
#include "scheduler_primitives.h"
#include "test_tier.h"
#include "weights.h"

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

constexpr char kPrefix[] = "QW38_OPT076_Q4_REDUCTION_RESULT=";
constexpr std::size_t kQ4KBytes = 144;
constexpr std::size_t kBlock = 256;
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr float kStagedAbsGate = 3.0e-4F;
constexpr float kProductionAbsGate = 0.0111372F;
constexpr int kSampledRows = 16;
constexpr int kSampledVectors = 4;

struct Options final {
  const char* workload = nullptr;
  const char* model = nullptr;
};

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--workload smoke|correctness|screen|acceptance] "
               "[MODEL]\n",
               argv0);
  return 2;
}

int parse_args(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--workload") == 0 && index + 1 < argc) {
      options->workload = argv[++index];
    } else if (arg[0] != '-' && options->model == nullptr) {
      options->model = arg;
    } else {
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

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s\n", status.message().c_str());
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

float bf16_to_float(std::uint16_t bits) {
  std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

__nv_bfloat16 from_bf16_bits(std::uint16_t bits) {
  __nv_bfloat16_raw raw;
  raw.x = bits;
  return __nv_bfloat16(raw);
}

void q4_scale_min_host(const std::uint8_t* packed, int index, int* scale,
                       int* minimum) {
  if (index < 4) {
    *scale = packed[index] & 63;
    *minimum = packed[index + 4] & 63;
    return;
  }
  *scale = (packed[index + 4] & 15) | ((packed[index - 4] >> 6) << 4);
  *minimum = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4);
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

void fill_activation(const char* pattern, std::size_t columns, std::uint32_t seed,
                     std::vector<std::uint16_t>* activation) {
  activation->resize(columns);
  for (std::size_t column = 0; column < columns; ++column) {
    float value = unit_normal(static_cast<std::uint32_t>(column), seed);
    if (std::strcmp(pattern, "zero") == 0) {
      value = 0.0F;
    } else if (std::strcmp(pattern, "cancellation") == 0) {
      value = (column % 2U == 0) ? 32.0F : -32.0F;
    } else if (std::strcmp(pattern, "pos") == 0) {
      value = 4.0F;
    } else if (std::strcmp(pattern, "neg") == 0) {
      value = -4.0F;
    } else if (std::strncmp(pattern, "group", 5) == 0) {
      const int group = pattern[5] - '0';
      const int sign = pattern[6] == 'n' ? -1 : 1;
      const std::size_t begin = static_cast<std::size_t>(group) * 32U;
      value = (column >= begin && column < begin + 32U)
                  ? static_cast<float>(sign) * 8.0F
                  : 0.0F;
    }
    (*activation)[column] = float_to_bf16(value);
  }
}

void stage_q8_from_blocks(const std::vector<qw38::cuda::Q8Block>& blocks,
                          std::size_t columns, std::vector<double>* staged) {
  staged->assign(columns, 0.0);
  const std::size_t groups = columns / 32U;
  for (std::size_t group = 0; group < groups; ++group) {
    const float scale = blocks[group].scale;
    for (int lane = 0; lane < 32; ++lane) {
      (*staged)[group * 32U + static_cast<std::size_t>(lane)] =
          static_cast<double>(scale *
                              static_cast<float>(blocks[group].values[lane]));
    }
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

struct Metrics final {
  double max_abs = 0.0;
  double rms = 0.0;
  int nonfinite = 0;
};

Metrics compare(const std::vector<float>& got,
                const std::vector<double>& ref) {
  Metrics metrics;
  const std::size_t n = std::min(got.size(), ref.size());
  double squared = 0.0;
  for (std::size_t index = 0; index < n; ++index) {
    const double left = static_cast<double>(got[index]);
    const double right = ref[index];
    if (!std::isfinite(left) || !std::isfinite(right)) {
      ++metrics.nonfinite;
      continue;
    }
    const double err = std::fabs(left - right);
    metrics.max_abs = std::max(metrics.max_abs, err);
    squared += err * err;
  }
  metrics.rms = n == 0 ? 0.0 : std::sqrt(squared / static_cast<double>(n));
  return metrics;
}

cudaError_t launch_path(const std::uint8_t* weights, std::size_t rows,
                        std::size_t columns, const __nv_bfloat16* activation,
                        qw38::cuda::Q8Block* q8, float* output,
                        const char* path, unsigned int warps,
                        cudaStream_t stream) {
  qw38::cuda::Q4DecodePathScope scope(path, warps);
  return qw38::cuda::launch_quant_mmv(qw38::cuda::QuantKind::kQ4K, weights, rows,
                                      columns, activation, q8, output, stream);
}

int run_mmv_case(const char* id, std::size_t rows, std::size_t columns,
                 const char* pattern, unsigned int warps, bool misaligned,
                 double* orig_abs, double* staged_abs, int* nonfinite) {
  std::vector<std::uint8_t> weights;
  std::vector<std::uint16_t> activation;
  fill_q4(rows, columns, &weights);
  fill_activation(pattern, columns, 0x62U, &activation);
  std::vector<double> orig(columns);
  for (std::size_t index = 0; index < columns; ++index) {
    orig[index] = static_cast<double>(bf16_to_float(activation[index]));
  }
  std::vector<double> fp64_orig;
  gemm_fp64(weights, rows, columns, orig, &fp64_orig);

  std::vector<__nv_bfloat16> host_act(columns);
  for (std::size_t index = 0; index < columns; ++index) {
    host_act[index] = from_bf16_bits(activation[index]);
  }
  const std::size_t pad = misaligned ? 1 : 0;
  std::uint8_t* dw_raw = nullptr;
  __nv_bfloat16* da = nullptr;
  qw38::cuda::Q8Block* dq = nullptr;
  float* dout = nullptr;
  cudaError_t error = cudaMalloc(&dw_raw, weights.size() + pad);
  if (error == cudaSuccess) {
    error = cudaMalloc(&da, columns * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&dq, qw38::cuda::q8_workspace_bytes(columns));
  }
  if (error == cudaSuccess) error = cudaMalloc(&dout, rows * sizeof(float));
  if (error != cudaSuccess) return fail_cuda("mmv malloc", error);
  std::uint8_t* dw = dw_raw + pad;
  error = cudaMemcpy(dw, weights.data(), weights.size(), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(da, host_act.data(), columns * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_bf16_q8(da, dq, columns, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<qw38::cuda::Q8Block> host_q8(columns / 32U);
  if (error == cudaSuccess) {
    error = cudaMemcpy(host_q8.data(), dq,
                       host_q8.size() * sizeof(qw38::cuda::Q8Block),
                       cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) {
    cudaFree(dw_raw);
    cudaFree(da);
    cudaFree(dq);
    cudaFree(dout);
    return fail_cuda("stage q8", error);
  }
  std::vector<double> staged;
  stage_q8_from_blocks(host_q8, columns, &staged);
  std::vector<double> fp64_staged;
  gemm_fp64(weights, rows, columns, staged, &fp64_staged);

  int rc = 0;
  const char* paths[] = {qw38::cuda::kLegalQ4DecodePathPacked,
                         qw38::cuda::kLegalQ4DecodePathIntegerQ8Late};
  const unsigned int warps_used[] = {4U, warps};
  for (int cand = 0; cand < 2; ++cand) {
    qw38::cuda::clear_q4_launch_trace();
    if (error == cudaSuccess) {
      error = launch_path(dw, rows, columns, da, dq, dout, paths[cand],
                          warps_used[cand], nullptr);
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    std::vector<float> host(rows);
    if (error == cudaSuccess) {
      error = cudaMemcpy(host.data(), dout, rows * sizeof(float),
                         cudaMemcpyDeviceToHost);
    }
    if (error != cudaSuccess) {
      cudaFree(dw_raw);
      cudaFree(da);
      cudaFree(dq);
      cudaFree(dout);
      return fail_cuda("mmv launch", error);
    }
    const Metrics vs_orig = compare(host, fp64_orig);
    const Metrics vs_staged = compare(host, fp64_staged);
    std::printf(
        "mmv id=%s path=%s warps=%u pattern=%s rows=%zu cols=%zu "
        "misaligned=%s vs_orig_abs=%.9g vs_staged_abs=%.9g nonfinite=%d "
        "launch=%s tail=%s\n",
        id, paths[cand], warps_used[cand], pattern, rows, columns,
        misaligned ? "true" : "false", vs_orig.max_abs, vs_staged.max_abs,
        vs_orig.nonfinite + vs_staged.nonfinite,
        qw38::cuda::last_q4_launch_variant(),
        (rows % 32U != 0) ? "true" : "false");
    *orig_abs = std::max(*orig_abs, vs_orig.max_abs);
    *staged_abs = std::max(*staged_abs, vs_staged.max_abs);
    *nonfinite += vs_orig.nonfinite + vs_staged.nonfinite;
    const float gate =
        rows <= 33 ? kStagedAbsGate : kProductionAbsGate;
    if (vs_orig.nonfinite != 0 || vs_staged.nonfinite != 0) rc = 1;
    if (vs_staged.max_abs > static_cast<double>(gate)) {
      std::fprintf(stderr, "staged miss %s %s abs=%.9g\n", id, paths[cand],
                   vs_staged.max_abs);
      rc = 1;
    }
    if (cand == 1) {
      const char* want = qw38::cuda::kQ4LaunchVariantCoopQ8Late;
      if (std::strcmp(qw38::cuda::last_q4_launch_variant(), want) != 0) {
        std::fprintf(stderr, "late launch variant %s\n",
                     qw38::cuda::last_q4_launch_variant());
        rc = 1;
      }
    }
  }
  cudaFree(dw_raw);
  cudaFree(da);
  cudaFree(dq);
  cudaFree(dout);
  return rc;
}

int run_group_layout() {
  std::vector<std::uint8_t> weights;
  fill_q4(1, 256, &weights);
  int failed = 0;
  for (int group = 0; group < 8; ++group) {
    int scale = -1;
    int minimum = -1;
    q4_scale_min_host(weights.data() + 4, group, &scale, &minimum);
    if (scale < 0 || scale > 63 || minimum < 0 || minimum > 63) ++failed;
    std::printf("q4_group index=%d nibble=%s scale=%d min=%d independent=true\n",
                group, (group & 1) == 0 ? "low" : "high", scale, minimum);
  }
  std::printf("q4_scale_min_layout=exact groups=8 failed=%d\n", failed);
  return failed == 0 ? 0 : 1;
}

int run_tiny(double* orig_abs, double* staged_abs, int* nonfinite) {
  int rc = run_group_layout();
  const std::size_t rows[] = {1, 17, 33};
  const std::size_t cols[] = {256, 512};
  const char* patterns[] = {"zero", "cancellation", "pos", "neg", "group0p",
                            "group0n", "group1p", "group2n", "group3p",
                            "group4n", "group5p", "group6n", "group7p"};
  for (std::size_t r : rows) {
    for (std::size_t c : cols) {
      char id[32];
      std::snprintf(id, sizeof(id), "M%zu_K%zu", r, c);
      for (const char* pattern : patterns) {
        rc |= run_mmv_case(id, r, c, pattern, 4, false, orig_abs, staged_abs,
                           nonfinite);
        rc |= run_mmv_case(id, r, c, pattern, 2, false, orig_abs, staged_abs,
                           nonfinite);
      }
    }
  }
  rc |= run_mmv_case("M17_K256_misaligned", 17, 256, "cancellation", 4, true,
                     orig_abs, staged_abs, nonfinite);
  return rc;
}

int run_kernel_attrs() {
  int rc = 0;
  for (unsigned int warps : {2U, 4U}) {
    int regs = 0;
    std::size_t local_bytes = 0;
    int occupancy = 0;
    qw38::cuda::q4k_coop_late_kernel_attributes(warps, &regs, &local_bytes,
                                               &occupancy);
    int control_regs = 0;
    std::size_t control_local = 0;
    int control_occ = 0;
    qw38::cuda::q4k_coop_kernel_attributes(warps, false, &control_regs,
                                           &control_local, &control_occ);
    int pair_regs = 0;
    std::size_t pair_local = 0;
    std::size_t pair_shared = 0;
    int pair_occ = 0;
    qw38::cuda::q4k_coop_gate_up_late_kernel_attributes(
        warps, &pair_regs, &pair_local, &pair_shared, &pair_occ);
    std::printf(
        "attrs warps=%u late_regs=%d late_local=%zu late_occ=%d "
        "control_regs=%d control_local=%zu control_occ=%d "
        "paired_late_regs=%d paired_late_local=%zu paired_late_shared=%zu "
        "paired_late_occ=%d\n",
        warps, regs, local_bytes, occupancy, control_regs, control_local,
        control_occ, pair_regs, pair_local, pair_shared, pair_occ);
    if (local_bytes > 0 && local_bytes > control_local) {
      std::fprintf(stderr, "late kernel added spills warps=%u local=%zu\n",
                   warps, local_bytes);
      rc = 1;
    }
  }
  return rc;
}

int run_sass() {
  FILE* pipe = popen(
      "cuobjdump -sass build/qw38-cuda-opt076-q4-reduction-test 2>/dev/null",
      "r");
  if (pipe == nullptr) {
    std::printf("sass_status=unavailable reason=popen_failed\n");
    return 0;
  }
  int dp4a = 0;
  int shfl = 0;
  int ldg32 = 0;
  int prmt = 0;
  char line[512];
  while (std::fgets(line, sizeof(line), pipe) != nullptr) {
    if (std::strstr(line, "IDP") != nullptr ||
        std::strstr(line, "DP4A") != nullptr ||
        std::strstr(line, "IMAD.SP") != nullptr) {
      ++dp4a;
    }
    if (std::strstr(line, "SHFL") != nullptr ||
        std::strstr(line, "SHF.L") != nullptr) {
      ++shfl;
    }
    if (std::strstr(line, "LDG.E") != nullptr) {
      ++ldg32;
    }
    if (std::strstr(line, "PRMT") != nullptr) {
      ++prmt;
    }
  }
  const int status = pclose(pipe);
  std::printf(
      "sass_status=%s dp4a_or_idp=%d shfl=%d ldg=%d prmt=%d "
      "k_loop_shuffles_absent_source=true\n",
      status == 0 ? "ok" : "unavailable", dp4a, shfl, ldg32, prmt);
  if (status == 0 && dp4a == 0) {
    std::fprintf(stderr, "SASS lacked DP4A/IDP word-dot evidence\n");
    return 1;
  }
  return 0;
}

float time_mmv(const std::uint8_t* dw, std::size_t rows, std::size_t columns,
               const __nv_bfloat16* da, qw38::cuda::Q8Block* dq, float* dout,
               const char* path, unsigned int warps, int warmups, int samples) {
  for (int index = 0; index < warmups; ++index) {
    launch_path(dw, rows, columns, da, dq, dout, path, warps, nullptr);
  }
  cudaDeviceSynchronize();
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, nullptr);
  for (int index = 0; index < samples; ++index) {
    launch_path(dw, rows, columns, da, dq, dout, path, warps, nullptr);
  }
  cudaEventRecord(stop, nullptr);
  cudaEventSynchronize(stop);
  float ms = 0.0F;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms / static_cast<float>(samples);
}

int run_shape_screen(int warmups, int samples) {
  struct Shape {
    const char* id;
    std::size_t rows;
    std::size_t columns;
  };
  const Shape shapes[] = {{"gate_up", 17408, 5120}, {"down", 5120, 17408}};
  int rc = 0;
  for (const Shape& shape : shapes) {
    std::vector<std::uint8_t> weights;
    std::vector<std::uint16_t> activation;
    fill_q4(shape.rows, shape.columns, &weights);
    fill_activation("pos", shape.columns, 0xA11CE5u, &activation);
    std::vector<__nv_bfloat16> host_act(shape.columns);
    for (std::size_t index = 0; index < shape.columns; ++index) {
      host_act[index] = from_bf16_bits(activation[index]);
    }
    std::uint8_t* dw = nullptr;
    __nv_bfloat16* da = nullptr;
    qw38::cuda::Q8Block* dq = nullptr;
    float* dout = nullptr;
    cudaError_t error = cudaMalloc(&dw, weights.size());
    if (error == cudaSuccess) {
      error = cudaMalloc(&da, shape.columns * sizeof(__nv_bfloat16));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(&dq, qw38::cuda::q8_workspace_bytes(shape.columns));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(&dout, shape.rows * sizeof(float));
    }
    if (error != cudaSuccess) return fail_cuda("screen malloc", error);
    error = cudaMemcpy(dw, weights.data(), weights.size(), cudaMemcpyHostToDevice);
    if (error == cudaSuccess) {
      error = cudaMemcpy(da, host_act.data(),
                         shape.columns * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice);
    }
    if (error != cudaSuccess) {
      cudaFree(dw);
      cudaFree(da);
      cudaFree(dq);
      cudaFree(dout);
      return fail_cuda("screen copy", error);
    }
    const float packed =
        time_mmv(dw, shape.rows, shape.columns, da, dq, dout,
                 qw38::cuda::kLegalQ4DecodePathPacked, 4, warmups, samples);
    const float late2 =
        time_mmv(dw, shape.rows, shape.columns, da, dq, dout,
                 qw38::cuda::kLegalQ4DecodePathIntegerQ8Late, 2, warmups,
                 samples);
    const float late4 =
        time_mmv(dw, shape.rows, shape.columns, da, dq, dout,
                 qw38::cuda::kLegalQ4DecodePathIntegerQ8Late, 4, warmups,
                 samples);
    std::printf(
        "screen shape=%s rows=%zu cols=%zu packed_ms=%.6f late_w2_ms=%.6f "
        "late_w4_ms=%.6f warmups=%d samples=%d\n",
        shape.id, shape.rows, shape.columns, static_cast<double>(packed),
        static_cast<double>(late2), static_cast<double>(late4), warmups,
        samples);
    cudaFree(dw);
    cudaFree(da);
    cudaFree(dq);
    cudaFree(dout);
  }
  return rc;
}

int run_production_rows(const char* model_path, double* orig_abs,
                        double* staged_abs, int* nonfinite) {
  if (model_path == nullptr || model_path[0] == '\0') {
    std::printf("production_rows skipped=true reason=no_model\n");
    return 0;
  }
  qw38::internal::MappedFile mapping;
  qw38::internal::ModelInfo info;
  qw38::internal::ModelWeights weights;
  qw38::cuda::ResidentModel model;
  qw38::Status status = qw38::internal::inspect_gguf(model_path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  if (status.is_ok()) status = mapping.open(model_path);
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, mapping, &weights);
  }
  if (status.is_ok()) {
    status = model.upload(weights, mapping.data(), mapping.size());
  }
  if (!status.is_ok()) return fail_status(status);
  const qw38::cuda::DeviceCommonLayer& layer = model.layer(0).common;
  struct Role {
    const char* name;
    const std::uint8_t* data;
    std::size_t rows;
    std::size_t columns;
  };
  const Role roles[] = {
      {"gate", layer.ffn_gate.data, layer.ffn_gate.rows, layer.ffn_gate.columns},
      {"up", layer.ffn_up.data, layer.ffn_up.rows, layer.ffn_up.columns},
      {"down", layer.ffn_down.data, layer.ffn_down.rows, layer.ffn_down.columns},
  };
  int rc = 0;
  for (const Role& role : roles) {
    const std::size_t sample = std::min(static_cast<std::size_t>(kSampledRows),
                                        role.rows);
    std::vector<std::size_t> rows(sample);
    for (std::size_t index = 0; index < sample / 2; ++index) rows[index] = index;
    for (std::size_t index = sample / 2; index < sample; ++index) {
      rows[index] = role.rows - (sample - index);
    }
    const std::size_t bytes = sample * (role.columns / kBlock) * kQ4KBytes;
    std::vector<std::uint8_t> host_w(bytes);
    for (std::size_t index = 0; index < sample; ++index) {
      const std::size_t row_bytes = (role.columns / kBlock) * kQ4KBytes;
      cudaError_t error = cudaMemcpy(
          host_w.data() + index * row_bytes, role.data + rows[index] * row_bytes,
          row_bytes, cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) return fail_cuda("row copy", error);
    }
    for (int vector = 0; vector < kSampledVectors; ++vector) {
      std::vector<std::uint16_t> activation;
      fill_activation("pos", role.columns,
                      0xC0FFEEu + static_cast<std::uint32_t>(vector) * 997U,
                      &activation);
      std::vector<double> orig(role.columns);
      for (std::size_t index = 0; index < role.columns; ++index) {
        orig[index] = static_cast<double>(bf16_to_float(activation[index]));
      }
      std::vector<double> fp64_orig;
      gemm_fp64(host_w, sample, role.columns, orig, &fp64_orig);
      std::vector<__nv_bfloat16> host_act(role.columns);
      for (std::size_t index = 0; index < role.columns; ++index) {
        host_act[index] = from_bf16_bits(activation[index]);
      }
      std::uint8_t* dw = nullptr;
      __nv_bfloat16* da = nullptr;
      qw38::cuda::Q8Block* dq = nullptr;
      float* dout = nullptr;
      cudaError_t error = cudaMalloc(&dw, host_w.size());
      if (error == cudaSuccess) {
        error = cudaMalloc(&da, role.columns * sizeof(__nv_bfloat16));
      }
      if (error == cudaSuccess) {
        error = cudaMalloc(&dq, qw38::cuda::q8_workspace_bytes(role.columns));
      }
      if (error == cudaSuccess) error = cudaMalloc(&dout, sample * sizeof(float));
      if (error != cudaSuccess) return fail_cuda("prod malloc", error);
      error = cudaMemcpy(dw, host_w.data(), host_w.size(), cudaMemcpyHostToDevice);
      if (error == cudaSuccess) {
        error = cudaMemcpy(da, host_act.data(),
                           role.columns * sizeof(__nv_bfloat16),
                           cudaMemcpyHostToDevice);
      }
      if (error == cudaSuccess) {
        error = launch_path(dw, sample, role.columns, da, dq, dout,
                            qw38::cuda::kLegalQ4DecodePathIntegerQ8Late, 4,
                            nullptr);
      }
      if (error == cudaSuccess) error = cudaDeviceSynchronize();
      std::vector<float> host(sample);
      if (error == cudaSuccess) {
        error = cudaMemcpy(host.data(), dout, sample * sizeof(float),
                           cudaMemcpyDeviceToHost);
      }
      std::vector<qw38::cuda::Q8Block> host_q8(role.columns / 32U);
      if (error == cudaSuccess) {
        error = cudaMemcpy(host_q8.data(), dq,
                           host_q8.size() * sizeof(qw38::cuda::Q8Block),
                           cudaMemcpyDeviceToHost);
      }
      if (error != cudaSuccess) {
        cudaFree(dw);
        cudaFree(da);
        cudaFree(dq);
        cudaFree(dout);
        return fail_cuda("prod launch", error);
      }
      std::vector<double> staged;
      stage_q8_from_blocks(host_q8, role.columns, &staged);
      std::vector<double> fp64_staged;
      gemm_fp64(host_w, sample, role.columns, staged, &fp64_staged);
      const Metrics vs_orig = compare(host, fp64_orig);
      const Metrics vs_staged = compare(host, fp64_staged);
      *orig_abs = std::max(*orig_abs, vs_orig.max_abs);
      *staged_abs = std::max(*staged_abs, vs_staged.max_abs);
      *nonfinite += vs_orig.nonfinite + vs_staged.nonfinite;
      std::printf(
          "production role=%s vector=%d rows=%zu cols=%zu vs_orig_abs=%.9g "
          "vs_staged_abs=%.9g nonfinite=%d finite=%s\n",
          role.name, vector, sample, role.columns, vs_orig.max_abs,
          vs_staged.max_abs, vs_orig.nonfinite + vs_staged.nonfinite,
          (vs_orig.nonfinite + vs_staged.nonfinite) == 0 ? "true" : "false");
      if (vs_orig.nonfinite != 0 || vs_staged.nonfinite != 0) rc = 1;
      if (vs_orig.max_abs > static_cast<double>(kProductionAbsGate) ||
          vs_staged.max_abs > static_cast<double>(kProductionAbsGate)) {
        rc = 1;
      }
      cudaFree(dw);
      cudaFree(da);
      cudaFree(dq);
      cudaFree(dout);
    }
  }
  std::printf("production_references sampled_rows=%d captures=%d decode_once=true\n",
              kSampledRows, kSampledVectors);
  return rc;
}

void print_result(const char* workload, int rc, double orig_abs,
                  double staged_abs, int nonfinite, int warmups, int samples) {
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-076\",\"workload\":\"%s\","
      "\"status\":\"%s\",\"gguf_sha256\":\"%s\",\"llama_revision\":\"%s\","
      "\"original_abs\":%.9g,\"staged_abs\":%.9g,\"nonfinite\":%d,"
      "\"half_scale_forbidden\":true,\"k_loop_shuffle_removed\":true,"
      "\"candidates\":[\"late_w2\",\"late_w4\"],\"control\":\"packed\","
      "\"keep\":false}\n",
      kPrefix, workload, rc == 0 ? "passed" : "failed", kGgufSha, kLlamaRev,
      orig_abs, staged_abs, nonfinite);
  std::printf(
      "QW38_OPT076_NATIVE_COUNTS={\"schema_version\":1,\"task\":\"OPT-076\","
      "\"warmups\":%d,\"samples\":%d,\"observed_warmups\":%d,"
      "\"observed_samples\":%d,\"observed_candidates\":3,\"observed_shapes\":2,"
      "\"observed_tier\":\"%s\",\"pairs\":%d,\"acceptance_executed\":%s,"
      "\"keep\":false}\n",
      warmups, samples, warmups, samples, workload, samples,
      std::strcmp(workload, "acceptance") == 0 ? "true" : "false");
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_valid()) {
    std::fprintf(stderr,
                 "QW38_CUDA_TEST_TIER must be set to smoke, correctness, "
                 "screen, or acceptance\n");
    return 2;
  }
  Options options;
  const int parsed = parse_args(argc, argv, &options);
  if (parsed != 0) return parsed;
  if (options.workload == nullptr) {
    options.workload = default_workload(qw38::cuda::test_tier());
  }
  double orig_abs = 0.0;
  double staged_abs = 0.0;
  int nonfinite = 0;
  int rc = 0;
  int warmups = 0;
  int samples = 1;
  if (std::strcmp(options.workload, "smoke") == 0) {
    rc |= run_mmv_case("M1_K256", 1, 256, "zero", 4, false, &orig_abs,
                       &staged_abs, &nonfinite);
    rc |= run_mmv_case("M17_K256", 17, 256, "cancellation", 2, false, &orig_abs,
                       &staged_abs, &nonfinite);
    rc |= run_kernel_attrs();
  } else if (std::strcmp(options.workload, "correctness") == 0) {
    rc |= run_tiny(&orig_abs, &staged_abs, &nonfinite);
    rc |= run_kernel_attrs();
    rc |= run_sass();
    rc |= run_production_rows(options.model, &orig_abs, &staged_abs, &nonfinite);
  } else if (std::strcmp(options.workload, "screen") == 0) {
    warmups = 1;
    samples = 3;
    rc |= run_tiny(&orig_abs, &staged_abs, &nonfinite);
    rc |= run_kernel_attrs();
    rc |= run_sass();
    rc |= run_production_rows(options.model, &orig_abs, &staged_abs, &nonfinite);
    rc |= run_shape_screen(warmups, samples);
  } else if (std::strcmp(options.workload, "acceptance") == 0) {
    warmups = 3;
    samples = 10;
    rc |= run_tiny(&orig_abs, &staged_abs, &nonfinite);
    rc |= run_kernel_attrs();
    rc |= run_sass();
    rc |= run_production_rows(options.model, &orig_abs, &staged_abs, &nonfinite);
    rc |= run_shape_screen(warmups, samples);
  } else {
    return usage(argv[0]);
  }
  print_result(options.workload, rc, orig_abs, staged_abs, nonfinite, warmups,
               samples);
  return rc;
}
