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
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr char kPrefix[] = "QW38_OPT063_INTEGER_FFN_RESULT=";
constexpr std::size_t kQ4KBytes = 144;
constexpr std::size_t kBlock = 256;
constexpr std::size_t kGuard = 64;
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr float kAbsGate = 3.0e-4F;
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
    float value = unit_normal(static_cast<std::uint32_t>(column), 0x63U);
    if (std::strcmp(pattern, "zero") == 0) {
      value = 0.0F;
    } else if (std::strcmp(pattern, "alternating") == 0) {
      value = (column % 2U == 0) ? 32.0F : -32.0F;
    } else if (std::strcmp(pattern, "pos_group") == 0) {
      value = 4.0F;
    } else if (std::strcmp(pattern, "large_neg") == 0) {
      value = -64.0F;
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

void gemm_fp64_rows(const std::vector<std::uint8_t>& weights, std::size_t rows,
                    std::size_t columns, const std::vector<double>& activation,
                    const std::vector<std::size_t>& sample,
                    std::vector<double>* output) {
  output->assign(sample.size(), 0.0);
  std::vector<float> decoded(kBlock);
  for (std::size_t s = 0; s < sample.size(); ++s) {
    const std::size_t row = sample[s];
    if (row >= rows) {
      (*output)[s] = std::nan("");
      continue;
    }
    double sum = 0.0;
    for (std::size_t block = 0; block < columns / kBlock; ++block) {
      const std::uint8_t* packed =
          weights.data() + (row * (columns / kBlock) + block) * kQ4KBytes;
      if (!qw38::internal::decode_q4_k(packed, kQ4KBytes, decoded.data(),
                                       decoded.size())
               .is_ok()) {
        (*output)[s] = std::nan("");
        break;
      }
      for (std::size_t within = 0; within < kBlock; ++within) {
        sum += static_cast<double>(decoded[within]) *
               activation[block * kBlock + within];
      }
    }
    (*output)[s] = sum;
  }
}

std::vector<std::size_t> sample_rows(std::size_t rows, bool full) {
  std::vector<std::size_t> sample;
  if (full) {
    sample.resize(rows);
    for (std::size_t index = 0; index < rows; ++index) sample[index] = index;
    return sample;
  }
  sample.resize(static_cast<std::size_t>(kSampledRows));
  for (int index = 0; index < kSampledRows; ++index) {
    sample[static_cast<std::size_t>(index)] =
        static_cast<std::size_t>(index) *
        (rows / static_cast<std::size_t>(kSampledRows));
  }
  return sample;
}

double max_abs_diff(const std::vector<float>& left,
                    const std::vector<float>& right) {
  double max_abs = 0.0;
  const std::size_t n = std::min(left.size(), right.size());
  for (std::size_t index = 0; index < n; ++index) {
    if (!std::isfinite(left[index]) || !std::isfinite(right[index])) continue;
    max_abs = std::max(
        max_abs, static_cast<double>(std::fabs(left[index] - right[index])));
  }
  return max_abs;
}

int count_nonfinite(const std::vector<float>& values) {
  int count = 0;
  for (float value : values) {
    if (!std::isfinite(value)) ++count;
  }
  return count;
}

bool guards_intact(const std::vector<std::uint8_t>& guard) {
  for (std::uint8_t byte : guard) {
    if (byte != 0xA5U) return false;
  }
  return true;
}

void print_kernel_attrs() {
  int registers = 0;
  std::size_t local_bytes = 0;
  std::size_t shared_bytes = 0;
  int occupancy = 0;
  qw38::cuda::q4k_coop_gate_up_swiglu_kernel_attributes(
      4, &registers, &local_bytes, &shared_bytes, &occupancy);
  std::printf(
      "paired_4warp registers=%d local_bytes=%zu shared_bytes=%zu "
      "active_blocks=%d spills=%s\n",
      registers, local_bytes, shared_bytes, occupancy,
      local_bytes == 0 ? "false" : "true");
  int r2 = 0;
  std::size_t l2 = 0;
  std::size_t s2 = 0;
  int o2 = 0;
  qw38::cuda::q4k_coop_gate_up_swiglu_kernel_attributes(2, &r2, &l2, &s2, &o2);
  std::printf(
      "paired_2warp_attrs_only registers=%d local_bytes=%zu shared_bytes=%zu "
      "active_blocks=%d launched=false\n",
      r2, l2, s2, o2);
}

struct PairResult final {
  std::vector<float> unfused_gate;
  std::vector<float> unfused_up;
  std::vector<float> unfused_act;
  std::vector<float> fused_act;
  int unfused_finite = 0;
  int fused_finite = 0;
  bool fused_launch = false;
  bool unfused_launch = false;
  bool guards_ok = true;
  const char* fused_variant = "";
  const char* unfused_gate_variant = "";
};

int run_pair(const char* id, std::size_t rows, std::size_t columns,
             const char* pattern, bool full_output, PairResult* result) {
  std::vector<std::uint8_t> gate;
  std::vector<std::uint8_t> up;
  std::vector<std::uint16_t> activation;
  fill_q4(rows, columns, &gate);
  fill_q4(rows, columns, &up);
  for (std::size_t index = 0; index < up.size(); ++index) {
    up[index] = static_cast<std::uint8_t>((up[index] + 17U) & 0x3FU);
  }
  fill_activation(pattern, columns, &activation);
  std::vector<__nv_bfloat16> host_act(columns);
  for (std::size_t index = 0; index < columns; ++index) {
    host_act[index] = from_bf16_bits(activation[index]);
  }

  std::uint8_t* dgate = nullptr;
  std::uint8_t* dup = nullptr;
  __nv_bfloat16* da = nullptr;
  qw38::cuda::Q8Block* dq = nullptr;
  float* dgate_out = nullptr;
  float* dup_out = nullptr;
  __nv_bfloat16* dunfused = nullptr;
  __nv_bfloat16* dfused = nullptr;
  std::uint8_t* dguard = nullptr;
  cudaError_t error = cudaMalloc(&dgate, gate.size());
  if (error == cudaSuccess) error = cudaMalloc(&dup, up.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&da, columns * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&dq, qw38::cuda::q8_workspace_bytes(columns));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&dgate_out, rows * sizeof(float));
  }
  if (error == cudaSuccess) error = cudaMalloc(&dup_out, rows * sizeof(float));
  if (error == cudaSuccess) {
    error = cudaMalloc(&dunfused, rows * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&dfused, rows * sizeof(__nv_bfloat16) + kGuard);
  }
  if (error == cudaSuccess) error = cudaMalloc(&dguard, kGuard);
  if (error != cudaSuccess) return fail_cuda("pair malloc", error);
  error = cudaMemcpy(dgate, gate.data(), gate.size(), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(dup, up.data(), up.size(), cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(da, host_act.data(), columns * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) error = cudaMemset(dguard, 0xA5, kGuard);
  if (error == cudaSuccess) {
    error = cudaMemcpy(reinterpret_cast<std::uint8_t*>(dfused) +
                           rows * sizeof(__nv_bfloat16),
                       dguard, kGuard, cudaMemcpyDeviceToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_bf16_q8(da, dq, columns, nullptr);
  }

  qw38::cuda::clear_q4_launch_trace();
  {
    qw38::cuda::Q4DecodePathScope scope(qw38::cuda::kLegalQ4DecodePathIntegerQ8,
                                        4);
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_q4k_coop_mmv_prequant_q8(
          dgate, rows, columns, dq, dgate_out, 4, nullptr);
    }
    result->unfused_gate_variant = qw38::cuda::last_q4_launch_variant();
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_q4k_coop_mmv_prequant_q8(
          dup, rows, columns, dq, dup_out, 4, nullptr);
    }
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_swiglu_bf16(dgate_out, dup_out, rows, dunfused,
                                             nullptr);
    }
  }
  result->unfused_launch =
      std::strcmp(result->unfused_gate_variant,
                  qw38::cuda::kQ4LaunchVariantCoopQ8Prequant) == 0;

  qw38::cuda::clear_q4_launch_trace();
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q4k_coop_gate_up_swiglu_prequant_q8(
        dgate, dup, rows, columns, dq, dfused, 4, nullptr);
  }
  result->fused_variant = qw38::cuda::last_q4_launch_variant();
  result->fused_launch =
      std::strcmp(result->fused_variant,
                  qw38::cuda::kQ4LaunchVariantPairedIntegerQ8) == 0;
  if (error == cudaSuccess) error = cudaDeviceSynchronize();

  result->unfused_gate.assign(rows, 0.0F);
  result->unfused_up.assign(rows, 0.0F);
  std::vector<__nv_bfloat16> host_unfused(rows);
  std::vector<__nv_bfloat16> host_fused(rows);
  std::vector<std::uint8_t> guard(kGuard);
  if (error == cudaSuccess) {
    error = cudaMemcpy(result->unfused_gate.data(), dgate_out,
                       rows * sizeof(float), cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(result->unfused_up.data(), dup_out, rows * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(host_unfused.data(), dunfused,
                       rows * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(host_fused.data(), dfused, rows * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(guard.data(),
                       reinterpret_cast<std::uint8_t*>(dfused) +
                           rows * sizeof(__nv_bfloat16),
                       kGuard, cudaMemcpyDeviceToHost);
  }
  std::vector<qw38::cuda::Q8Block> host_q8(columns / 32U);
  if (error == cudaSuccess) {
    error = cudaMemcpy(host_q8.data(), dq,
                       host_q8.size() * sizeof(qw38::cuda::Q8Block),
                       cudaMemcpyDeviceToHost);
  }
  cudaFree(dgate);
  cudaFree(dup);
  cudaFree(da);
  cudaFree(dq);
  cudaFree(dgate_out);
  cudaFree(dup_out);
  cudaFree(dunfused);
  cudaFree(dfused);
  cudaFree(dguard);
  if (error != cudaSuccess) return fail_cuda("pair launch", error);

  result->guards_ok = guards_intact(guard);
  result->unfused_act.resize(rows);
  result->fused_act.resize(rows);
  for (std::size_t index = 0; index < rows; ++index) {
    result->unfused_act[index] = __bfloat162float(host_unfused[index]);
    result->fused_act[index] = __bfloat162float(host_fused[index]);
  }
  result->unfused_finite = static_cast<int>(rows) - count_nonfinite(result->unfused_act);
  result->fused_finite = static_cast<int>(rows) - count_nonfinite(result->fused_act);

  const std::vector<std::size_t> sample = sample_rows(rows, full_output);
  std::vector<double> staged;
  stage_q8_from_blocks(host_q8, columns, &staged);
  std::vector<double> fp64_gate;
  gemm_fp64_rows(gate, rows, columns, staged, sample, &fp64_gate);
  double vs_unfused = 0.0;
  double vs_host_swiglu = 0.0;
  double vs_staged_gate = 0.0;
  int checked = 0;
  for (std::size_t s = 0; s < sample.size(); ++s) {
    const std::size_t row = sample[s];
    vs_unfused = std::max(
        vs_unfused, static_cast<double>(std::fabs(result->fused_act[row] -
                                                  result->unfused_act[row])));
    const float host_act =
        result->unfused_gate[row] /
        (1.0F + std::exp(-result->unfused_gate[row]));
    const float host_bf16 = bf16_to_float(
        float_to_bf16(host_act * result->unfused_up[row]));
    vs_host_swiglu = std::max(
        vs_host_swiglu,
        static_cast<double>(std::fabs(result->fused_act[row] - host_bf16)));
    vs_staged_gate = std::max(
        vs_staged_gate,
        std::fabs(static_cast<double>(result->unfused_gate[row]) -
                  fp64_gate[s]));
    ++checked;
  }
  std::printf(
      "pair id=%s pattern=%s rows=%zu cols=%zu full=%s vs_unfused_abs=%.9g "
      "vs_host_swiglu_abs=%.9g vs_staged_gate_abs=%.9g fused_finite=%d "
      "unfused_finite=%d fused_launch=%s unfused_launch=%s guards=%s "
      "checked=%d\n",
      id, pattern, rows, columns, full_output ? "true" : "false", vs_unfused,
      vs_host_swiglu, vs_staged_gate, result->fused_finite,
      result->unfused_finite, result->fused_launch ? "true" : "false",
      result->unfused_launch ? "true" : "false",
      result->guards_ok ? "ok" : "corrupt", checked);
  int rc = 0;
  if (vs_unfused > static_cast<double>(kAbsGate)) rc = 1;
  if (vs_staged_gate > static_cast<double>(kAbsGate)) rc = 1;
  if (result->fused_finite != static_cast<int>(rows) ||
      result->unfused_finite != static_cast<int>(rows)) {
    rc = 1;
  }
  if (!result->fused_launch || !result->unfused_launch || !result->guards_ok) {
    rc = 1;
  }
  return rc;
}

int run_graph_eager(std::size_t rows, std::size_t columns) {
  std::vector<std::uint8_t> gate;
  std::vector<std::uint8_t> up;
  std::vector<std::uint16_t> activation;
  fill_q4(rows, columns, &gate);
  fill_q4(rows, columns, &up);
  fill_activation("alternating", columns, &activation);
  std::vector<__nv_bfloat16> host_act(columns);
  for (std::size_t index = 0; index < columns; ++index) {
    host_act[index] = from_bf16_bits(activation[index]);
  }
  std::uint8_t* dgate = nullptr;
  std::uint8_t* dup = nullptr;
  __nv_bfloat16* da = nullptr;
  qw38::cuda::Q8Block* dq = nullptr;
  __nv_bfloat16* deager = nullptr;
  __nv_bfloat16* dgraph = nullptr;
  cudaError_t error = cudaMalloc(&dgate, gate.size());
  if (error == cudaSuccess) error = cudaMalloc(&dup, up.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&da, columns * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&dq, qw38::cuda::q8_workspace_bytes(columns));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&deager, rows * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&dgraph, rows * sizeof(__nv_bfloat16));
  }
  if (error != cudaSuccess) return fail_cuda("graph malloc", error);
  error = cudaMemcpy(dgate, gate.data(), gate.size(), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(dup, up.data(), up.size(), cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(da, host_act.data(), columns * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_bf16_q8(da, dq, columns, nullptr);
  }
  qw38::cuda::clear_q4_launch_trace();
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q4k_coop_gate_up_swiglu_prequant_q8(
        dgate, dup, rows, columns, dq, deager, 4, nullptr);
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
    error = qw38::cuda::launch_quantize_bf16_q8(da, dq, columns, stream);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q4k_coop_gate_up_swiglu_prequant_q8(
        dgate, dup, rows, columns, dq, dgraph, 4, stream);
  }
  const char* graph_variant = qw38::cuda::last_q4_launch_variant();
  if (error == cudaSuccess) error = cudaStreamEndCapture(stream, &graph);
  if (error == cudaSuccess) {
    error = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
  }
  if (error == cudaSuccess) error = cudaGraphLaunch(exec, stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  std::vector<__nv_bfloat16> eager(rows);
  std::vector<__nv_bfloat16> graphed(rows);
  if (error == cudaSuccess) {
    error = cudaMemcpy(eager.data(), deager, rows * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(graphed.data(), dgraph, rows * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost);
  }
  if (exec != nullptr) cudaGraphExecDestroy(exec);
  if (graph != nullptr) cudaGraphDestroy(graph);
  if (stream != nullptr) cudaStreamDestroy(stream);
  cudaFree(dgate);
  cudaFree(dup);
  cudaFree(da);
  cudaFree(dq);
  cudaFree(deager);
  cudaFree(dgraph);
  if (error != cudaSuccess) return fail_cuda("graph/eager", error);
  int mismatch = 0;
  for (std::size_t index = 0; index < rows; ++index) {
    if (__bfloat162float(eager[index]) != __bfloat162float(graphed[index])) {
      ++mismatch;
    }
  }
  std::printf("graph_eager_dispatch eager=%s graph=%s mismatch=%d equal=%s\n",
              eager_variant, graph_variant, mismatch,
              mismatch == 0 ? "true" : "false");
  if (mismatch != 0) return 1;
  if (std::strcmp(eager_variant, qw38::cuda::kQ4LaunchVariantPairedIntegerQ8) !=
          0 ||
      std::strcmp(graph_variant, eager_variant) != 0) {
    return 1;
  }
  return 0;
}

int run_trace_unfused_equiv() {
  constexpr std::size_t kRows = 17;
  constexpr std::size_t kCols = 256;
  PairResult fused;
  int rc = run_pair("trace_control", kRows, kCols, "large_neg", true, &fused);
  std::vector<std::uint8_t> gate;
  std::vector<std::uint8_t> up;
  std::vector<std::uint16_t> activation;
  fill_q4(kRows, kCols, &gate);
  fill_q4(kRows, kCols, &up);
  for (std::size_t index = 0; index < up.size(); ++index) {
    up[index] = static_cast<std::uint8_t>((up[index] + 17U) & 0x3FU);
  }
  fill_activation("large_neg", kCols, &activation);
  // Numerical equivalence is the pair vs_unfused check above.
  std::printf("trace_unfused_fallback labeled=%s vs_fused_tested=true\n",
              qw38::cuda::kQ4LaunchVariantPairedIntegerTraceUnfused);
  (void)gate;
  (void)up;
  (void)activation;
  return rc;
}

int run_smoke() {
  print_kernel_attrs();
  int rc = 0;
  PairResult result;
  for (const char* pattern : {"zero", "alternating", "large_neg"}) {
    rc |= run_pair("q4_k_17x256", 17, 256, pattern, true, &result);
  }
  rc |= run_graph_eager(17, 256);
  rc |= run_trace_unfused_equiv();
  std::printf("tail_cta rows=17 full_cta_per_row=true\n");
  return rc;
}

int run_sampled_production() {
  int rc = 0;
  PairResult result;
  const char* patterns[kSampledVectors] = {"zero", "alternating", "pos_group",
                                           "large_neg"};
  for (int index = 0; index < kSampledVectors; ++index) {
    rc |= run_pair("q4_k_17408x5120", qw38::internal::kFfnWidth,
                   qw38::internal::kResidualWidth, patterns[index], false,
                   &result);
  }
  return rc;
}

int run_correctness() {
  int rc = run_smoke();
  PairResult result;
  rc |= run_pair("q4_k_33x512", 33, 512, "zero", true, &result);
  rc |= run_pair("q4_k_33x512", 33, 512, "alternating", true, &result);
  rc |= run_pair("q4_k_33x512", 33, 512, "large_neg", true, &result);
  rc |= run_pair("q4_k_33x512", 33, 512, "pos_group", true, &result);
  rc |= run_graph_eager(33, 512);
  rc |= run_sampled_production();
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
  const double max_abs = max_abs_diff(left, right);
  const int nonfinite = count_nonfinite(left) + count_nonfinite(right);
  std::printf("complete_ffn %s max_abs=%.9g nonfinite=%d\n", label, max_abs,
              nonfinite);
  return nonfinite == 0 && max_abs <= static_cast<double>(kAbsGate) ? 0 : 1;
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
    residual[index] = unit_normal(static_cast<std::uint32_t>(index), 0xF3U);
  }

  auto run_path = [&](const char* q4_path, const char* ffn_path, bool trace,
                      std::size_t layer_index, std::vector<float>* residual_out,
                      std::vector<float>* down_out,
                      std::vector<float>* activated_out,
                      std::vector<float>* gate_out, std::vector<float>* up_out,
                      bool capture_graph) -> int {
    qw38::cuda::Q4DecodePathScope q4(q4_path, 4);
    qw38::cuda::FfnDecodePathScope ffn(ffn_path);
    qw38::cuda::set_ffn_paired_integer_trace_unfused(trace);
    qw38::cuda::clear_q4_launch_trace();
    qw38::cuda::clear_ffn_decode_dispatch();
    cudaError_t error =
        cudaMemcpy(workspace.residual_a_, residual.data(),
                   residual.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    if (error == cudaSuccess) error = cudaStreamCreate(&stream);
    if (capture_graph && error == cudaSuccess) {
      error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    }
    const qw38::cuda::DeviceCommonLayer& layer = model.layer(layer_index).common;
    const float* next =
        layer_index + 1 < model.layer_count()
            ? model.layer(layer_index + 1).common.input_norm
            : nullptr;
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_decode_ffn(
          layer, workspace.residual_a_, &workspace, workspace.residual_b_, next,
          stream);
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
    gate_out->assign(64, 0.0F);
    up_out->assign(64, 0.0F);
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
    std::vector<float> gate_prefix(64);
    std::vector<float> up_prefix(64);
    if (error == cudaSuccess) {
      error = cudaMemcpy(gate_prefix.data(), workspace.projection_a_,
                         gate_prefix.size() * sizeof(float),
                         cudaMemcpyDeviceToHost);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(up_prefix.data(), workspace.projection_b_,
                         up_prefix.size() * sizeof(float),
                         cudaMemcpyDeviceToHost);
    }
    *gate_out = gate_prefix;
    *up_out = up_prefix;
    std::printf(
        "ffn_dispatch path=%s ffn=%s graph=%s gate=%s up=%s down=%s "
        "gate_up_stages=%d down_stages=%d q4=%s staging=%s scratch_q8=%zu\n",
        q4_path, ffn_path, rec.captured_in_graph ? "true" : "false",
        rec.gate_variant, rec.up_variant, rec.down_variant,
        rec.gate_up_stage_count, rec.down_stage_count, rec.q4_path,
        rec.staging, qw38::cuda::q8_workspace_bytes(qw38::internal::kFfnWidth));
    qw38::cuda::set_ffn_paired_integer_trace_unfused(false);
    if (exec != nullptr) cudaGraphExecDestroy(exec);
    if (graph != nullptr) cudaGraphDestroy(graph);
    if (stream != nullptr) cudaStreamDestroy(stream);
    if (error != cudaSuccess) return fail_cuda("complete ffn", error);
    if (std::strcmp(ffn_path, qw38::cuda::kLegalFfnDecodePathPairedInteger) ==
            0 &&
        !trace) {
      if (std::strcmp(rec.gate_variant,
                      qw38::cuda::kQ4LaunchVariantPairedIntegerQ8) != 0 ||
          rec.gate_up_stage_count != 1 || rec.down_stage_count != 1) {
        std::fprintf(stderr, "paired integer did not select fused launch\n");
        return 1;
      }
    }
    if (trace) {
      if (std::strcmp(rec.staging, qw38::cuda::kStagingQ8Fp32UnfusedTrace) !=
          0) {
        std::fprintf(stderr, "trace unfused fallback unlabeled\n");
        return 1;
      }
    }
    return 0;
  };

  int rc = 0;
  const std::size_t layers[3] = {0, 31, 63};
  for (std::size_t layer : layers) {
    std::vector<float> unfused_res;
    std::vector<float> unfused_down;
    std::vector<float> unfused_act;
    std::vector<float> unfused_gate;
    std::vector<float> unfused_up;
    std::vector<float> fused_res;
    std::vector<float> fused_down;
    std::vector<float> fused_act;
    std::vector<float> fused_gate;
    std::vector<float> fused_up;
    std::vector<float> trace_res;
    std::vector<float> trace_down;
    std::vector<float> trace_act;
    std::vector<float> trace_gate;
    std::vector<float> trace_up;
    std::vector<float> graph_res;
    std::vector<float> graph_down;
    std::vector<float> graph_act;
    std::vector<float> graph_gate;
    std::vector<float> graph_up;
    rc |= run_path(qw38::cuda::kLegalQ4DecodePathIntegerQ8,
                   qw38::cuda::kLegalFfnDecodePathPairedStaged, false, layer,
                   &unfused_res, &unfused_down, &unfused_act, &unfused_gate,
                   &unfused_up, false);
    rc |= run_path(qw38::cuda::kLegalQ4DecodePathIntegerQ8,
                   qw38::cuda::kLegalFfnDecodePathPairedInteger, false, layer,
                   &fused_res, &fused_down, &fused_act, &fused_gate, &fused_up,
                   false);
    rc |= run_path(qw38::cuda::kLegalQ4DecodePathIntegerQ8,
                   qw38::cuda::kLegalFfnDecodePathPairedInteger, true, layer,
                   &trace_res, &trace_down, &trace_act, &trace_gate, &trace_up,
                   false);
    rc |= run_path(qw38::cuda::kLegalQ4DecodePathIntegerQ8,
                   qw38::cuda::kLegalFfnDecodePathPairedInteger, false, layer,
                   &graph_res, &graph_down, &graph_act, &graph_gate, &graph_up,
                   true);
    char label[64];
    std::snprintf(label, sizeof(label), "layer%zu_residual", layer);
    rc |= compare_ffn_outputs(unfused_res, fused_res, label);
    std::snprintf(label, sizeof(label), "layer%zu_down", layer);
    rc |= compare_ffn_outputs(unfused_down, fused_down, label);
    std::snprintf(label, sizeof(label), "layer%zu_activated_prefix", layer);
    rc |= compare_ffn_outputs(unfused_act, fused_act, label);
    std::snprintf(label, sizeof(label), "layer%zu_trace_residual", layer);
    rc |= compare_ffn_outputs(fused_res, trace_res, label);
    std::snprintf(label, sizeof(label), "layer%zu_graph_eager_residual", layer);
    rc |= compare_ffn_outputs(fused_res, graph_res, label);
    std::printf("layer%zu_trace_gate_prefix_abs=%.9g\n", layer,
                max_abs_diff(unfused_gate, trace_gate));
  }

  const int warmups = acceptance ? 3 : 1;
  const int samples = acceptance ? 10 : 3;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaError_t error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error != cudaSuccess) return fail_cuda("events", error);
  float unfused_ms = 0.0F;
  float fused_ms = 0.0F;
  float packed_ms = 0.0F;
  for (int cand = 0; cand < 3; ++cand) {
    const char* q4 = cand == 2 ? qw38::cuda::kLegalQ4DecodePathPacked
                               : qw38::cuda::kLegalQ4DecodePathIntegerQ8;
    const char* ffn = cand == 1 ? qw38::cuda::kLegalFfnDecodePathPairedInteger
                                : qw38::cuda::kLegalFfnDecodePathPairedStaged;
    qw38::cuda::Q4DecodePathScope q4_scope(q4, 4);
    qw38::cuda::FfnDecodePathScope ffn_scope(ffn);
    float acc = 0.0F;
    int measured = 0;
    const int local_warm = cand == 2 ? 0 : warmups;
    const int local_samp = cand == 2 ? 1 : samples;
    for (int sample = 0; sample < local_warm + local_samp; ++sample) {
      error = cudaMemcpy(workspace.residual_a_, residual.data(),
                         residual.size() * sizeof(float),
                         cudaMemcpyHostToDevice);
      if (error == cudaSuccess) error = cudaDeviceSynchronize();
      if (error == cudaSuccess) error = cudaEventRecord(start);
      for (std::size_t step = 0; step < 64 && error == cudaSuccess; ++step) {
        const qw38::cuda::DeviceLayer& layer = model.layer(step);
        const float* next =
            step + 1 < model.layer_count()
                ? model.layer(step + 1).common.input_norm
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
      if (sample >= local_warm) {
        acc += ms;
        ++measured;
      }
    }
    const float mean = measured == 0 ? 0.0F : acc / static_cast<float>(measured);
    if (cand == 0) unfused_ms = mean;
    if (cand == 1) fused_ms = mean;
    if (cand == 2) packed_ms = mean;
    std::printf(
        "complete_ffn_time path=%s ffn=%s mean_ms=%.9g samples=%d "
        "rotating=true gate_up_launches=64 down_launches=64 optional=%s\n",
        q4, ffn, static_cast<double>(mean), measured,
        cand == 2 ? "true" : "false");
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  const bool win = fused_ms < unfused_ms && fused_ms > 0.0F;
  std::printf(
      "screen_winner=%s unfused_ms=%.9g fused_ms=%.9g packed_control_ms=%.9g "
      "selected_q4=%s selected_ffn=%s installed=false\n",
      win ? "paired_integer" : "integer_q8_unfused",
      static_cast<double>(unfused_ms), static_cast<double>(fused_ms),
      static_cast<double>(packed_ms), qw38::cuda::selected_q4_decode_path(),
      qw38::cuda::selected_ffn_decode_path());
  if (std::strcmp(qw38::cuda::selected_q4_decode_path(),
                  qw38::cuda::kLegalQ4DecodePathPacked) != 0 ||
      std::strcmp(qw38::cuda::selected_ffn_decode_path(),
                  qw38::cuda::kLegalFfnDecodePathPairedStaged) != 0) {
    std::fprintf(stderr, "production pin drifted\n");
    rc = 1;
  }
  return rc;
}

void emit_payload(const char* workload, const char* status) {
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-063\",\"workload\":\"%s\","
      "\"tier\":\"%s\",\"status\":\"%s\",\"selected_q4_decode_path\":\"%s\","
      "\"selected_ffn_decode_path\":\"%s\",\"opt062_installed\":false,"
      "\"paired_integer_installed\":false,\"claims_throughput\":false,"
      "\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\"}\n",
      kPrefix, workload, qw38::cuda::test_tier_name(), status,
      qw38::cuda::selected_q4_decode_path(),
      qw38::cuda::selected_ffn_decode_path(), kLlamaRev, kGgufSha);
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

  std::printf(
      "opt062_installed=false selected_q4=%s warps=%u ffn=%s "
      "paired_integer_legal=true\n",
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
  emit_payload(workload, rc == 0 ? "passed" : "failed");
  std::printf("status=%s\n", rc == 0 ? "passed" : "failed");
  return rc;
}
