#include "ffn_decode_path.cuh"
#include "full_scheduler.h"
#include "mixer.h"
#include "model.h"
#include "optimization_component_replay.h"
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
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr char kPrefix[] = "QW38_OPT062_Q4_ADMISSION_RESULT=";
constexpr std::size_t kQ4KBytes = 144;
constexpr std::size_t kBlock = 256;
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr int kMaxProductionFp64 = 128;
constexpr float kStagedAbsGate = 3.0e-4F;

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

void fill_activation(const char* pattern, std::size_t columns,
                     std::vector<std::uint16_t>* activation) {
  activation->resize(columns);
  for (std::size_t column = 0; column < columns; ++column) {
    float value = unit_normal(static_cast<std::uint32_t>(column), 0x62U);
    if (std::strcmp(pattern, "zero") == 0) {
      value = 0.0F;
    } else if (std::strcmp(pattern, "alternating") == 0) {
      value = (column % 2U == 0) ? 32.0F : -32.0F;
    } else if (std::strcmp(pattern, "pos_group") == 0) {
      value = 4.0F;
    } else if (std::strcmp(pattern, "neg_group") == 0) {
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
      // Match the GPU Q8 reconstruction (float mul, then promote). Promoting
      // scale to double first disagrees with lossless ±4 activations by ~4e-4
      // on the decode-Q4 · staged product even when the kernel is exact.
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

int run_group_layout() {
  std::vector<std::uint8_t> weights;
  fill_q4(1, 256, &weights);
  int failed = 0;
  for (int group = 0; group < 8; ++group) {
    int scale = -1;
    int minimum = -1;
    q4_scale_min_host(weights.data() + 4, group, &scale, &minimum);
    if (scale < 0 || scale > 63 || minimum < 0 || minimum > 63) ++failed;
    const int high = group & 1;
    std::printf(
        "q4_group index=%d nibble=%s scale=%d min=%d\n", group,
        high == 0 ? "low" : "high", scale, minimum);
  }
  std::printf("q4_scale_min_layout=exact groups=8 failed=%d\n", failed);
  return failed == 0 ? 0 : 1;
}

cudaError_t launch_integer_or_packed(const std::uint8_t* weights,
                                     std::size_t rows, std::size_t columns,
                                     const __nv_bfloat16* activation,
                                     qw38::cuda::Q8Block* q8, float* output,
                                     bool integer, cudaStream_t stream) {
  if (integer) {
    qw38::cuda::Q4DecodePathScope scope(qw38::cuda::kLegalQ4DecodePathIntegerQ8,
                                        4);
    return qw38::cuda::launch_quant_mmv(qw38::cuda::QuantKind::kQ4K, weights,
                                        rows, columns, activation, q8, output,
                                        stream);
  }
  qw38::cuda::Q4DecodePathScope scope(qw38::cuda::kLegalQ4DecodePathPacked, 4);
  return qw38::cuda::launch_quant_mmv(qw38::cuda::QuantKind::kQ4K, weights, rows,
                                      columns, activation, q8, output, stream);
}

int run_mmv_shape(const char* id, std::size_t rows, std::size_t columns,
                  const char* pattern, bool full_output, int* fp64_dots,
                  bool production) {
  std::vector<std::uint8_t> weights;
  std::vector<std::uint16_t> activation;
  fill_q4(rows, columns, &weights);
  fill_activation(pattern, columns, &activation);
  std::vector<double> orig(columns);
  for (std::size_t index = 0; index < columns; ++index) {
    orig[index] = static_cast<double>(bf16_to_float(activation[index]));
  }
  std::vector<double> fp64_orig;
  gemm_fp64(weights, rows, columns, orig, &fp64_orig);
  if (production) *fp64_dots += static_cast<int>(rows);

  std::vector<__nv_bfloat16> host_act(columns);
  for (std::size_t index = 0; index < columns; ++index) {
    host_act[index] = from_bf16_bits(activation[index]);
  }
  std::uint8_t* dw = nullptr;
  __nv_bfloat16* da = nullptr;
  qw38::cuda::Q8Block* dq = nullptr;
  float* dout = nullptr;
  cudaError_t error = cudaMalloc(&dw, weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&da, columns * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&dq, qw38::cuda::q8_workspace_bytes(columns));
  }
  if (error == cudaSuccess) error = cudaMalloc(&dout, rows * sizeof(float));
  if (error != cudaSuccess) return fail_cuda("mmv malloc", error);
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
    error = cudaMemcpy(host_q8.data(), dq, host_q8.size() * sizeof(qw38::cuda::Q8Block),
                       cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) {
    cudaFree(dw);
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
  std::vector<float> packed_host;
  for (int cand = 0; cand < 2; ++cand) {
    const bool integer = cand == 1;
    qw38::cuda::clear_q4_launch_trace();
    if (error == cudaSuccess) {
      error = launch_integer_or_packed(dw, rows, columns, da, dq, dout, integer,
                                       nullptr);
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    std::vector<float> host(rows);
    if (error == cudaSuccess) {
      error = cudaMemcpy(host.data(), dout, rows * sizeof(float),
                         cudaMemcpyDeviceToHost);
    }
    if (error != cudaSuccess) {
      cudaFree(dw);
      cudaFree(da);
      cudaFree(dq);
      cudaFree(dout);
      return fail_cuda("mmv launch", error);
    }
    const Metrics vs_orig = compare(host, fp64_orig);
    const Metrics vs_staged = compare(host, fp64_staged);
    const char* path = integer ? "integer_q8" : "packed";
    std::printf(
        "mmv id=%s path=%s pattern=%s rows=%zu cols=%zu vs_orig_abs=%.9g "
        "vs_staged_abs=%.9g nonfinite=%d launch=%s full=%s\n",
        id, path, pattern, rows, columns, vs_orig.max_abs, vs_staged.max_abs,
        vs_orig.nonfinite + vs_staged.nonfinite,
        qw38::cuda::last_q4_launch_variant(), full_output ? "true" : "false");
    if (vs_orig.nonfinite != 0 || vs_staged.nonfinite != 0) rc = 1;
    if (vs_staged.max_abs > static_cast<double>(kStagedAbsGate)) {
      std::fprintf(stderr, "staged FP64 miss %s %s abs=%.9g\n", id, path,
                   vs_staged.max_abs);
      rc = 1;
    }
    if (!integer) {
      packed_host = host;
    } else {
      double vs_packed = 0.0;
      for (std::size_t index = 0; index < host.size(); ++index) {
        vs_packed = std::max(
            vs_packed, static_cast<double>(std::fabs(host[index] - packed_host[index])));
      }
      std::printf("mmv id=%s integer_vs_packed_abs=%.9g\n", id, vs_packed);
      if (vs_packed > static_cast<double>(kStagedAbsGate)) {
        std::fprintf(stderr, "integer diverged from packed %s abs=%.9g\n", id,
                     vs_packed);
        rc = 1;
      }
      if (std::strcmp(qw38::cuda::last_q4_launch_variant(),
                      qw38::cuda::kQ4LaunchVariantCoopQ8) != 0) {
        std::fprintf(stderr, "integer path did not record coop complete launch\n");
        rc = 1;
      }
    }
  }
  cudaFree(dw);
  cudaFree(da);
  cudaFree(dq);
  cudaFree(dout);
  return rc;
}

int run_typed_prequant_guard() {
  constexpr std::size_t kRows = 17;
  constexpr std::size_t kCols = 256;
  std::vector<std::uint8_t> weights;
  std::vector<std::uint16_t> activation;
  fill_q4(kRows, kCols, &weights);
  fill_activation("pos_group", kCols, &activation);
  std::vector<__nv_bfloat16> host_act(kCols);
  for (std::size_t index = 0; index < kCols; ++index) {
    host_act[index] = from_bf16_bits(activation[index]);
  }
  std::uint8_t* dw = nullptr;
  __nv_bfloat16* da = nullptr;
  qw38::cuda::Q8Block* dq = nullptr;
  float* dout = nullptr;
  cudaError_t error = cudaMalloc(&dw, weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&da, kCols * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&dq, qw38::cuda::q8_workspace_bytes(kCols));
  }
  if (error == cudaSuccess) error = cudaMalloc(&dout, kRows * sizeof(float));
  if (error != cudaSuccess) return fail_cuda("prequant malloc", error);
  error = cudaMemcpy(dw, weights.data(), weights.size(), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(da, host_act.data(), kCols * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_bf16_q8(da, dq, kCols, nullptr);
  }
  int rc = 0;
  {
    qw38::cuda::Q4DecodePathScope scope(qw38::cuda::kLegalQ4DecodePathIntegerQ81,
                                        4);
    qw38::cuda::clear_q4_launch_trace();
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_quant_mmv_prequant(
          qw38::cuda::QuantKind::kQ4K, dw, kRows, kCols, dq, dout, nullptr);
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
      rc = fail_cuda("q8_1 prequant", error);
    } else if (std::strcmp(qw38::cuda::last_q4_launch_variant(),
                           qw38::cuda::kQ4LaunchVariantPackedPrequant) != 0) {
      std::fprintf(stderr,
                   "Q8Block* prequant reinterpreted as Q8_1 variant=%s\n",
                   qw38::cuda::last_q4_launch_variant());
      rc = 1;
    } else {
      std::printf("typed_prequant integer_q8_1_override=packed_q8block\n");
    }
  }
  {
    qw38::cuda::Q4DecodePathScope scope(qw38::cuda::kLegalQ4DecodePathIntegerQ8,
                                        4);
    qw38::cuda::clear_q4_launch_trace();
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_quant_mmv_prequant(
          qw38::cuda::QuantKind::kQ4K, dw, kRows, kCols, dq, dout, nullptr);
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
      rc = fail_cuda("q8 prequant", error);
    } else if (std::strcmp(qw38::cuda::last_q4_launch_variant(),
                           qw38::cuda::kQ4LaunchVariantCoopQ8Prequant) != 0) {
      std::fprintf(stderr, "integer_q8 prequant variant=%s\n",
                   qw38::cuda::last_q4_launch_variant());
      rc = 1;
    } else {
      std::printf("typed_prequant integer_q8=q4k_coop_mmv_prequant_q8\n");
    }
  }
  cudaFree(dw);
  cudaFree(da);
  cudaFree(dq);
  cudaFree(dout);
  return rc;
}

int run_buffer_invalidation() {
  constexpr std::size_t kRows = 17;
  constexpr std::size_t kCols = 256;
  std::vector<std::uint8_t> weights;
  std::vector<std::uint16_t> act_a;
  std::vector<std::uint16_t> act_b;
  fill_q4(kRows, kCols, &weights);
  fill_activation("pos_group", kCols, &act_a);
  fill_activation("neg_group", kCols, &act_b);
  std::vector<__nv_bfloat16> host_a(kCols);
  std::vector<__nv_bfloat16> host_b(kCols);
  for (std::size_t index = 0; index < kCols; ++index) {
    host_a[index] = from_bf16_bits(act_a[index]);
    host_b[index] = from_bf16_bits(act_b[index]);
  }
  std::uint8_t* dw = nullptr;
  __nv_bfloat16* da = nullptr;
  __nv_bfloat16* db = nullptr;
  qw38::cuda::Q8Block* dq = nullptr;
  float* dout_fresh = nullptr;
  float* dout_stale = nullptr;
  cudaError_t error = cudaMalloc(&dw, weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&da, kCols * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&db, kCols * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&dq, qw38::cuda::q8_workspace_bytes(kCols));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&dout_fresh, kRows * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&dout_stale, kRows * sizeof(float));
  }
  if (error != cudaSuccess) return fail_cuda("invalidate malloc", error);
  error = cudaMemcpy(dw, weights.data(), weights.size(), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(da, host_a.data(), kCols * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(db, host_b.data(), kCols * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice);
  }
  qw38::cuda::Q4DecodePathScope scope(qw38::cuda::kLegalQ4DecodePathIntegerQ8, 4);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_bf16_q8(da, dq, kCols, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q4k_coop_mmv_prequant_q8(dw, kRows, kCols, dq,
                                                        dout_stale, 4, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_bf16_q8(db, dq, kCols, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q4k_coop_mmv_prequant_q8(dw, kRows, kCols, dq,
                                                        dout_fresh, 4, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<float> stale(kRows);
  std::vector<float> fresh(kRows);
  if (error == cudaSuccess) {
    error = cudaMemcpy(stale.data(), dout_stale, kRows * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(fresh.data(), dout_fresh, kRows * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  cudaFree(dw);
  cudaFree(da);
  cudaFree(db);
  cudaFree(dq);
  cudaFree(dout_fresh);
  cudaFree(dout_stale);
  if (error != cudaSuccess) return fail_cuda("invalidate", error);
  double max_abs = 0.0;
  for (std::size_t index = 0; index < kRows; ++index) {
    max_abs = std::max(max_abs, static_cast<double>(
                                    std::fabs(stale[index] - fresh[index])));
  }
  std::printf("reused_buffer_invalidation delta_abs=%.9g restage_required=true\n",
              max_abs);
  if (max_abs < 1.0e-4) {
    std::fprintf(stderr, "restaging Q8Block did not change MMV outputs\n");
    return 1;
  }
  return 0;
}

int run_graph_eager_small() {
  constexpr std::size_t kRows = 17;
  constexpr std::size_t kCols = 256;
  std::vector<std::uint8_t> weights;
  std::vector<std::uint16_t> activation;
  fill_q4(kRows, kCols, &weights);
  fill_activation("alternating", kCols, &activation);
  std::vector<__nv_bfloat16> host_act(kCols);
  for (std::size_t index = 0; index < kCols; ++index) {
    host_act[index] = from_bf16_bits(activation[index]);
  }
  std::uint8_t* dw = nullptr;
  __nv_bfloat16* da = nullptr;
  qw38::cuda::Q8Block* dq = nullptr;
  float* deager = nullptr;
  float* dgraph = nullptr;
  cudaError_t error = cudaMalloc(&dw, weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&da, kCols * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&dq, qw38::cuda::q8_workspace_bytes(kCols));
  }
  if (error == cudaSuccess) error = cudaMalloc(&deager, kRows * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&dgraph, kRows * sizeof(float));
  if (error != cudaSuccess) return fail_cuda("graph malloc", error);
  error = cudaMemcpy(dw, weights.data(), weights.size(), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(da, host_act.data(), kCols * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice);
  }
  qw38::cuda::Q4DecodePathScope scope(qw38::cuda::kLegalQ4DecodePathIntegerQ8, 4);
  qw38::cuda::clear_q4_launch_trace();
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_bf16_q8(da, dq, kCols, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q4k_coop_mmv_prequant_q8(dw, kRows, kCols, dq,
                                                        deager, 4, nullptr);
  }
  const char* eager_variant = qw38::cuda::last_q4_launch_variant();
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  if (error == cudaSuccess) error = cudaStreamCreate(&stream);
  if (error == cudaSuccess) {
    error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  }
  qw38::cuda::clear_q4_launch_trace();
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_bf16_q8(da, dq, kCols, stream);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q4k_coop_mmv_prequant_q8(dw, kRows, kCols, dq,
                                                        dgraph, 4, stream);
  }
  const char* graph_variant = qw38::cuda::last_q4_launch_variant();
  if (error == cudaSuccess) error = cudaStreamEndCapture(stream, &graph);
  if (error == cudaSuccess) {
    error = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
  }
  if (error == cudaSuccess) error = cudaGraphLaunch(exec, stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  std::vector<float> eager(kRows);
  std::vector<float> graphed(kRows);
  if (error == cudaSuccess) {
    error = cudaMemcpy(eager.data(), deager, kRows * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(graphed.data(), dgraph, kRows * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (exec != nullptr) cudaGraphExecDestroy(exec);
  if (graph != nullptr) cudaGraphDestroy(graph);
  if (stream != nullptr) cudaStreamDestroy(stream);
  cudaFree(dw);
  cudaFree(da);
  cudaFree(dq);
  cudaFree(deager);
  cudaFree(dgraph);
  if (error != cudaSuccess) return fail_cuda("graph/eager", error);
  int mismatch = 0;
  for (std::size_t index = 0; index < kRows; ++index) {
    if (eager[index] != graphed[index]) ++mismatch;
  }
  std::printf(
      "graph_eager_dispatch eager=%s graph=%s mismatch=%d equal=%s\n",
      eager_variant, graph_variant, mismatch, mismatch == 0 ? "true" : "false");
  if (mismatch != 0) return 1;
  if (std::strcmp(eager_variant, qw38::cuda::kQ4LaunchVariantCoopQ8Prequant) !=
          0 ||
      std::strcmp(graph_variant, eager_variant) != 0) {
    return 1;
  }
  return 0;
}

int run_group_dots() {
  int rc = 0;
  for (int group = 0; group < 8; ++group) {
    char pos[] = "group0p";
    char neg[] = "group0n";
    pos[5] = static_cast<char>('0' + group);
    neg[5] = static_cast<char>('0' + group);
    int dots = 0;
    rc |= run_mmv_shape("q4_k_17x256", 17, 256, pos, true, &dots, false);
    rc |= run_mmv_shape("q4_k_17x256", 17, 256, neg, true, &dots, false);
  }
  return rc;
}

int run_smoke() {
  int dots = 0;
  int rc = 0;
  rc |= run_group_layout();
  for (const char* pattern :
       {"zero", "alternating", "pos_group", "neg_group"}) {
    rc |= run_mmv_shape("q4_k_17x256", 17, 256, pattern, true, &dots, false);
  }
  rc |= run_typed_prequant_guard();
  rc |= run_buffer_invalidation();
  rc |= run_graph_eager_small();
  const bool half = qw38::cuda::production_numerics_v2_admitted(
      "q4_k", 0, qw38::cuda::kStagingQuartzQ81SumQ, "integer_dp4a_q8_1");
  std::printf("half_scale_skipped=%s reason=opt059_v2_unadmitted\n",
              half ? "false" : "true");
  if (half) rc = 1;
  std::printf("fp64_dots=%d production_cap=%d\n", dots, kMaxProductionFp64);
  return rc;
}

int run_correctness() {
  int dots = 0;
  int rc = run_smoke();
  for (const char* pattern :
       {"zero", "alternating", "pos_group", "neg_group"}) {
    rc |= run_mmv_shape("q4_k_19x512", 19, 512, pattern, true, &dots, false);
  }
  rc |= run_group_dots();
  for (const char* pattern :
       {"zero", "alternating", "pos_group", "neg_group"}) {
    rc |= run_mmv_shape("q4_k_16x5120", 16, 5120, pattern, false, &dots, true);
    rc |= run_mmv_shape("q4_k_16x17408", 16, 17408, pattern, false, &dots, true);
  }
  if (dots > kMaxProductionFp64) {
    std::fprintf(stderr, "production fp64_dots=%d exceeds %d\n", dots,
                 kMaxProductionFp64);
    rc = 1;
  }
  std::printf("correctness_production_fp64_dots=%d\n", dots);
  return rc;
}

int load_model(const char* path, qw38::internal::MappedFile* mapping,
               qw38::internal::ModelInfo* info,
               qw38::internal::ModelWeights* weights,
               qw38::cuda::ResidentModel* model) {
  qw38::Status status = qw38::internal::inspect_gguf(path, info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(info);
  if (status.is_ok()) status = mapping->open(path);
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(*info, *mapping, weights);
  }
  if (status.is_ok()) {
    status = model->upload(*weights, mapping->data(), mapping->size());
  }
  if (!status.is_ok()) return fail_status(status);
  return 0;
}

int compare_ffn_outputs(const std::vector<float>& left,
                        const std::vector<float>& right, const char* label) {
  double max_abs = 0.0;
  int nonfinite = 0;
  const std::size_t n = std::min(left.size(), right.size());
  for (std::size_t index = 0; index < n; ++index) {
    if (!std::isfinite(left[index]) || !std::isfinite(right[index])) {
      ++nonfinite;
      continue;
    }
    max_abs = std::max(
        max_abs, static_cast<double>(std::fabs(left[index] - right[index])));
  }
  std::printf("complete_ffn %s max_abs=%.9g nonfinite=%d\n", label, max_abs,
              nonfinite);
  return nonfinite == 0 ? 0 : 1;
}

int run_complete_ffn(const char* model_path, bool acceptance) {
  if (model_path == nullptr || model_path[0] == '\0') {
    std::fprintf(stderr, "screen/acceptance require the pinned GGUF\n");
    return 1;
  }
  qw38::internal::MappedFile mapping;
  qw38::internal::ModelInfo info;
  qw38::internal::ModelWeights weights;
  qw38::cuda::ResidentModel model;
  const int loaded = load_model(model_path, &mapping, &info, &weights, &model);
  if (loaded != 0) return loaded;

  qw38::cuda::SchedulerWorkspace workspace;
  qw38::Status status = workspace.create(64);
  if (!status.is_ok()) return fail_status(status);
  std::vector<float> residual(qw38::internal::kResidualWidth);
  for (std::size_t index = 0; index < residual.size(); ++index) {
    residual[index] = unit_normal(static_cast<std::uint32_t>(index), 0xF1U);
  }
  cudaError_t error =
      cudaMemcpy(workspace.residual_a_, residual.data(),
                 residual.size() * sizeof(float), cudaMemcpyHostToDevice);
  const qw38::cuda::DeviceCommonLayer& layer0 = model.layer(0).common;
  const float* next_norm = model.layer(1).common.input_norm;

  auto run_path = [&](const char* q4_path,
                      std::vector<float>* residual_out,
                      std::vector<float>* down_out,
                      std::vector<float>* activated_out,
                      bool capture_graph) -> int {
    qw38::cuda::Q4DecodePathScope scope(q4_path, 4);
    qw38::cuda::clear_q4_launch_trace();
    qw38::cuda::clear_ffn_decode_dispatch();
    error = cudaMemcpy(workspace.residual_a_, residual.data(),
                       residual.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    if (error == cudaSuccess) error = cudaStreamCreate(&stream);
    if (capture_graph && error == cudaSuccess) {
      error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    }
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_decode_ffn(
          layer0, workspace.residual_a_, &workspace, workspace.residual_b_,
          next_norm, stream);
    }
    const qw38::cuda::FfnDecodeDispatchRecord rec =
        qw38::cuda::last_ffn_decode_dispatch();
    if (capture_graph && error == cudaSuccess) {
      error = cudaStreamEndCapture(stream, &graph);
      if (error == cudaSuccess) {
        error = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
      }
      if (error == cudaSuccess) error = cudaGraphLaunch(exec, stream);
    }
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
    residual_out->assign(qw38::internal::kResidualWidth, 0.0F);
    down_out->assign(qw38::internal::kResidualWidth, 0.0F);
    activated_out->assign(64, 0.0F);
    if (error == cudaSuccess) {
      error = cudaMemcpy(residual_out->data(), workspace.residual_b_,
                         residual_out->size() * sizeof(float),
                         cudaMemcpyDeviceToHost);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(down_out->data(), workspace.mixer_output_,
                         down_out->size() * sizeof(float),
                         cudaMemcpyDeviceToHost);
    }
    std::vector<__nv_bfloat16> act(64);
    if (error == cudaSuccess) {
      error = cudaMemcpy(act.data(), workspace.ffn_activated_,
                         act.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost);
    }
    for (std::size_t index = 0; index < act.size(); ++index) {
      (*activated_out)[index] = __bfloat162float(act[index]);
    }
    std::printf(
        "ffn_dispatch path=%s graph=%s gate=%s up=%s down=%s "
        "gate_up_stages=%d down_stages=%d q4=%s ffn=%s scratch_q8=%zu\n",
        q4_path, rec.captured_in_graph ? "true" : "false", rec.gate_variant,
        rec.up_variant, rec.down_variant, rec.gate_up_stage_count,
        rec.down_stage_count, rec.q4_path, rec.ffn_path,
        qw38::cuda::q8_workspace_bytes(qw38::internal::kFfnWidth));
    if (q4_path == qw38::cuda::kLegalQ4DecodePathIntegerQ8) {
      if (std::strcmp(rec.gate_variant,
                      qw38::cuda::kQ4LaunchVariantCoopQ8Prequant) != 0 ||
          std::strcmp(rec.up_variant,
                      qw38::cuda::kQ4LaunchVariantCoopQ8Prequant) != 0 ||
          std::strcmp(rec.down_variant, qw38::cuda::kQ4LaunchVariantCoopQ8) !=
              0 ||
          rec.gate_up_stage_count != 1 || rec.down_stage_count != 1) {
        std::fprintf(stderr, "integer Q4 did not reach all three FFN legs\n");
        if (exec != nullptr) cudaGraphExecDestroy(exec);
        if (graph != nullptr) cudaGraphDestroy(graph);
        if (stream != nullptr) cudaStreamDestroy(stream);
        return 1;
      }
    }
    if (exec != nullptr) cudaGraphExecDestroy(exec);
    if (graph != nullptr) cudaGraphDestroy(graph);
    if (stream != nullptr) cudaStreamDestroy(stream);
    return error == cudaSuccess ? 0 : fail_cuda("complete ffn", error);
  };

  std::vector<float> packed_res;
  std::vector<float> packed_down;
  std::vector<float> packed_act;
  std::vector<float> int_res;
  std::vector<float> int_down;
  std::vector<float> int_act;
  int rc = run_path(qw38::cuda::kLegalQ4DecodePathPacked, &packed_res,
                    &packed_down, &packed_act, false);
  rc |= run_path(qw38::cuda::kLegalQ4DecodePathIntegerQ8, &int_res, &int_down,
                 &int_act, false);
  std::vector<float> graph_res;
  std::vector<float> graph_down;
  std::vector<float> graph_act;
  rc |= run_path(qw38::cuda::kLegalQ4DecodePathIntegerQ8, &graph_res,
                 &graph_down, &graph_act, true);
  rc |= compare_ffn_outputs(packed_res, int_res, "residual");
  rc |= compare_ffn_outputs(packed_down, int_down, "down");
  rc |= compare_ffn_outputs(packed_act, int_act, "activated_prefix");
  rc |= compare_ffn_outputs(int_res, graph_res, "graph_eager_residual");

  const int warmups = acceptance ? 3 : 1;
  const int samples = acceptance ? 10 : 3;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error != cudaSuccess) return fail_cuda("events", error);
  float packed_ms = 0.0F;
  float integer_ms = 0.0F;
  for (int cand = 0; cand < 2; ++cand) {
    const char* path = cand == 0 ? qw38::cuda::kLegalQ4DecodePathPacked
                                 : qw38::cuda::kLegalQ4DecodePathIntegerQ8;
    qw38::cuda::Q4DecodePathScope scope(path, 4);
    float acc = 0.0F;
    int measured = 0;
    for (int sample = 0; sample < warmups + samples; ++sample) {
      error = cudaMemcpy(workspace.residual_a_, residual.data(),
                         residual.size() * sizeof(float),
                         cudaMemcpyHostToDevice);
      if (error == cudaSuccess) error = cudaDeviceSynchronize();
      if (error == cudaSuccess) error = cudaEventRecord(start);
      for (std::size_t step = 0; step < 64 && error == cudaSuccess; ++step) {
        const std::size_t layer_index = step;
        const qw38::cuda::DeviceLayer& layer = model.layer(layer_index);
        const float* next =
            layer_index + 1 < model.layer_count()
                ? model.layer(layer_index + 1).common.input_norm
                : nullptr;
        error = qw38::cuda::launch_decode_ffn(
            layer.common, workspace.residual_a_, &workspace,
            workspace.residual_b_, next, nullptr);
        std::swap(workspace.residual_a_, workspace.residual_b_);
      }
      if (error == cudaSuccess) error = cudaEventRecord(stop);
      if (error == cudaSuccess) error = cudaEventSynchronize(stop);
      float ms = 0.0F;
      if (error == cudaSuccess) error = cudaEventElapsedTime(&ms, start, stop);
      if (error != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return fail_cuda("screen time", error);
      }
      if (sample >= warmups) {
        acc += ms;
        ++measured;
      }
    }
    const float mean = measured == 0 ? 0.0F : acc / static_cast<float>(measured);
    if (cand == 0) packed_ms = mean;
    else integer_ms = mean;
    std::printf("complete_ffn_time path=%s mean_ms=%.9g samples=%d rotating=true\n",
                path, static_cast<double>(mean), measured);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  const bool win = integer_ms < packed_ms && integer_ms > 0.0F;
  std::printf(
      "screen_winner=%s packed_ms=%.9g integer_ms=%.9g selected_q4=%s "
      "effective_q4=%s\n",
      win ? "integer_q8" : "packed_paired_staged",
      static_cast<double>(packed_ms), static_cast<double>(integer_ms),
      qw38::cuda::selected_q4_decode_path(),
      qw38::cuda::effective_q4_decode_path());
  if (std::strcmp(qw38::cuda::selected_q4_decode_path(),
                  qw38::cuda::kLegalQ4DecodePathPacked) != 0) {
    std::fprintf(stderr, "production pin drifted from packed\n");
    rc = 1;
  }
  return rc;
}

void emit_payload(const char* workload, const char* status, int fp64_dots) {
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-062\",\"workload\":\"%s\","
      "\"tier\":\"%s\",\"status\":\"%s\",\"selected_q4_decode_path\":\"%s\","
      "\"selected_ffn_decode_path\":\"%s\",\"half_scale_skipped\":true,"
      "\"fp64_dots\":%d,\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\","
      "\"opt046_installed\":false,\"claims_throughput\":false}\n",
      kPrefix, workload, qw38::cuda::test_tier_name(), status,
      qw38::cuda::selected_q4_decode_path(),
      qw38::cuda::selected_ffn_decode_path(), fp64_dots, kLlamaRev, kGgufSha);
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
  if (parse_args(argc, argv, &options) != 0) return 2;
  const char* workload = options.workload;
  if (workload == nullptr) workload = default_workload(qw38::cuda::test_tier());

  std::printf("opt046_installed=false selected_q4=%s warps=%u ffn=%s\n",
              qw38::cuda::selected_q4_decode_path(),
              qw38::cuda::selected_q4_decode_warps_per_row(),
              qw38::cuda::selected_ffn_decode_path());
  std::printf(
      "q4_k_integer_q8 testing_admitted=%s v2_admitted=%s\n",
      qw38::cuda::production_numerics_testing_admitted(
          "q4_k", 0, qw38::cuda::kStagingQ8Fp32, "integer_dp4a_q8")
          ? "true"
          : "false",
      qw38::cuda::production_numerics_v2_admitted(
          "q4_k", 0, qw38::cuda::kStagingQ8Fp32, "integer_dp4a_q8")
          ? "true"
          : "false");

  int rc = 0;
  if (std::strcmp(workload, "smoke") == 0) {
    rc = run_smoke();
  } else if (std::strcmp(workload, "correctness") == 0) {
    rc = run_correctness();
  } else if (std::strcmp(workload, "screen") == 0 ||
             std::strcmp(workload, "acceptance") == 0) {
    rc = run_correctness();
    if (rc == 0) {
      rc = run_complete_ffn(options.model,
                            std::strcmp(workload, "acceptance") == 0);
    }
  } else {
    return usage(argv[0]);
  }
  emit_payload(workload, rc == 0 ? "passed" : "failed", 0);
  std::printf("status=%s\n", rc == 0 ? "passed" : "failed");
  return rc;
}
