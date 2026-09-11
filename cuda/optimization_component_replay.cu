#include "optimization_component_replay.h"

#include "engine_attribution.h"
#include "model.h"
#include "quant.h"
#include "quant_mmv.h"
#include "q8_decode_path.cuh"
#include "rms_norm.cuh"
#include "scheduler.h"
#include "scheduler_primitives.h"
#include "sha256.h"
#include "test_tier.h"
#include "weights.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr std::size_t kQ4KBytes = 144;
constexpr std::size_t kQ80Bytes = 34;
constexpr std::size_t kBlock = 256;
constexpr std::size_t kQ80 = 32;
constexpr std::size_t kTinyRows = 17;
constexpr std::size_t kTinyCols = 256;
constexpr std::uint8_t kGuardFill = 0xA5U;
constexpr char kCaptureCache[] =
    "evidence/optimization/opt061-component-replay/capture-bundle.json";
constexpr char kHardwareJson[] =
    "evidence/optimization/opt061-component-replay/hardware.json";
constexpr char kProvenanceJson[] =
    "evidence/optimization/opt061-component-replay/provenance.json";
constexpr char kOpt071CaptureRoot[] =
    "evidence/optimization/opt071-attribution-repair/captures";
constexpr std::size_t kLayerFloats =
    qw38::cuda::kOpt061LayerCount * qw38::internal::kResidualWidth;

struct Options final {
  const char* model = nullptr;
  const char* workload = nullptr;
  const char* cache_mode = nullptr;
  const char* capture_key = nullptr;
  int warmups = -1;
  int samples = -1;
  bool ncu = false;
  bool corrupt_guard = false;
};

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--workload smoke|correctness|decode-ffn|"
               "decode-mixer|prompt-ffn|acceptance|hardware] [MODEL] "
               "[--cache-mode hot|rotating] [--capture-key KEY] "
               "[--warmups N] [--samples N] [--ncu] [--corrupt-guard]\n",
               argv0);
  return 2;
}

int parse_args(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--workload") == 0 && index + 1 < argc) {
      options->workload = argv[++index];
    } else if (std::strcmp(arg, "--cache-mode") == 0 && index + 1 < argc) {
      options->cache_mode = argv[++index];
    } else if (std::strcmp(arg, "--capture-key") == 0 && index + 1 < argc) {
      options->capture_key = argv[++index];
    } else if (std::strcmp(arg, "--warmups") == 0 && index + 1 < argc) {
      options->warmups = std::atoi(argv[++index]);
    } else if (std::strcmp(arg, "--samples") == 0 && index + 1 < argc) {
      options->samples = std::atoi(argv[++index]);
    } else if (std::strcmp(arg, "--ncu") == 0) {
      options->ncu = true;
    } else if (std::strcmp(arg, "--corrupt-guard") == 0) {
      options->corrupt_guard = true;
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
  if (tier == qw38::cuda::TestTier::kCorrectness) return "correctness";
  if (tier == qw38::cuda::TestTier::kAcceptance) return "acceptance";
  return "decode-ffn";
}

int fail_cuda(const char* operation, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
  return 1;
}

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

const char* json_bool(bool value) { return value ? "true" : "false"; }

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
  const float u1 = (static_cast<float>(x & 0x00FFFFFFU) + 1.0F) / 16777217.0F;
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

void fill_q8(std::size_t rows, std::size_t columns,
             std::vector<std::uint8_t>* weights) {
  weights->assign(rows * (columns / kQ80) * kQ80Bytes, 0);
  for (std::size_t index = 0; index < weights->size(); ++index) {
    (*weights)[index] = static_cast<std::uint8_t>((index * 11 + 7) & 0x7FU);
  }
  for (std::size_t offset = 0; offset < weights->size(); offset += kQ80Bytes) {
    write_u16(weights->data() + offset, 0x3C00U);
  }
}

void fill_activation_bf16(const char* pattern, std::size_t columns,
                          std::vector<__nv_bfloat16>* activation) {
  activation->resize(columns);
  for (std::size_t column = 0; column < columns; ++column) {
    float value = unit_normal(static_cast<std::uint32_t>(column), 0x59U);
    if (std::strcmp(pattern, "zero") == 0) {
      value = 0.0F;
    } else if (std::strcmp(pattern, "alternating") == 0) {
      value = (column % 2U == 0) ? 32.0F : -32.0F;
    } else if (std::strcmp(pattern, "finite_b") == 0) {
      value = unit_normal(static_cast<std::uint32_t>(column), 0xA11CE5u);
    }
    (*activation)[column] = from_bf16_bits(float_to_bf16(value));
  }
}

void wrap_guards(std::vector<std::uint8_t>* packed,
                 std::vector<std::uint8_t>* guarded) {
  guarded->assign(qw38::cuda::kOpt061GuardBytes + packed->size() +
                      qw38::cuda::kOpt061GuardBytes,
                  kGuardFill);
  std::memcpy(guarded->data() + qw38::cuda::kOpt061GuardBytes, packed->data(),
              packed->size());
}

bool guards_ok(const std::vector<std::uint8_t>& guarded, std::size_t inner) {
  const std::size_t prefix = qw38::cuda::kOpt061GuardBytes;
  for (std::size_t index = 0; index < prefix; ++index) {
    if (guarded[index] != kGuardFill) return false;
    if (guarded[prefix + inner + index] != kGuardFill) return false;
  }
  return true;
}

bool decode_row(qw38::cuda::QuantKind kind, const std::uint8_t* packed,
                std::size_t block_bytes, std::vector<float>* decoded) {
  const std::size_t values =
      kind == qw38::cuda::QuantKind::kQ8_0 ? kQ80 : kBlock;
  decoded->assign(values, 0.0F);
  if (kind == qw38::cuda::QuantKind::kQ4K) {
    return qw38::internal::decode_q4_k(packed, block_bytes, decoded->data(),
                                       decoded->size())
        .is_ok();
  }
  return qw38::internal::decode_q8_0(packed, block_bytes, decoded->data(),
                                     decoded->size())
      .is_ok();
}

void sampled_fp64(qw38::cuda::QuantKind kind,
                  const std::vector<std::uint8_t>& weights, std::size_t rows,
                  std::size_t columns, const std::vector<float>& activation,
                  const std::vector<std::size_t>& sample_rows,
                  std::vector<double>* output) {
  const std::size_t values =
      kind == qw38::cuda::QuantKind::kQ8_0 ? kQ80 : kBlock;
  const std::size_t block_bytes =
      kind == qw38::cuda::QuantKind::kQ4K ? kQ4KBytes : kQ80Bytes;
  output->assign(sample_rows.size(), 0.0);
  std::vector<float> decoded;
  for (std::size_t s = 0; s < sample_rows.size(); ++s) {
    const std::size_t row = sample_rows[s];
    if (row >= rows) {
      (*output)[s] = std::nan("");
      continue;
    }
    double sum = 0.0;
    for (std::size_t block = 0; block < columns / values; ++block) {
      const std::uint8_t* packed =
          weights.data() + (row * (columns / values) + block) * block_bytes;
      if (!decode_row(kind, packed, block_bytes, &decoded)) {
        (*output)[s] = std::nan("");
        break;
      }
      for (std::size_t index = 0; index < decoded.size(); ++index) {
        sum += static_cast<double>(decoded[index]) *
               static_cast<double>(activation[block * values + index]);
      }
    }
    (*output)[s] = sum;
  }
}

__global__ void streaming_checksum(const std::uint8_t* data, std::size_t bytes,
                                   unsigned long long* out) {
  unsigned long long local = 0;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x * 16U;
  std::size_t cursor =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 16U;
  while (cursor + 8U <= bytes) {
    local += *reinterpret_cast<const unsigned long long*>(data + cursor);
    cursor += stride;
  }
  atomicAdd(out, local);
}

cudaError_t event_ms(cudaEvent_t start, cudaEvent_t stop, float* ms) {
  cudaError_t error = cudaEventSynchronize(stop);
  if (error == cudaSuccess) error = cudaEventElapsedTime(ms, start, stop);
  return error;
}

struct HardwareInfo final {
  int sm = 0;
  std::size_t l2_bytes = 0;
  std::size_t shared_per_block = 0;
  std::size_t shared_per_sm = 0;
  int regs_per_block = 0;
  char name[256]{};
  float power_limit_w = 0.0F;
  float power_draw_w = 0.0F;
  int sm_clock = 0;
  int mem_clock = 0;
  int temp_c = 0;
  char throttle[160]{};
  bool ncu_available = false;
  bool nsys_available = false;
};

void query_smi(HardwareInfo* info) {
  FILE* pipe = popen(
      "nvidia-smi --query-gpu=power.limit,power.draw,clocks.current.sm,"
      "clocks.current.memory,temperature.gpu,clocks_throttle_reasons.active "
      "--format=csv,noheader,nounits",
      "r");
  if (pipe == nullptr) {
    std::snprintf(info->throttle, sizeof(info->throttle), "nvidia-smi_unavailable");
    return;
  }
  char line[256]{};
  if (std::fgets(line, sizeof(line), pipe) != nullptr) {
    std::sscanf(line, "%f, %f, %d, %d, %d, %159[^\n]", &info->power_limit_w,
                &info->power_draw_w, &info->sm_clock, &info->mem_clock,
                &info->temp_c, info->throttle);
  }
  pclose(pipe);
}

int query_hardware(HardwareInfo* info) {
  cudaDeviceProp prop{};
  cudaError_t error = cudaGetDeviceProperties(&prop, 0);
  if (error != cudaSuccess) return fail_cuda("cudaGetDeviceProperties", error);
  info->sm = prop.multiProcessorCount;
  info->l2_bytes = static_cast<std::size_t>(prop.l2CacheSize);
  info->shared_per_block = prop.sharedMemPerBlock;
  info->shared_per_sm = prop.sharedMemPerMultiprocessor;
  info->regs_per_block = prop.regsPerBlock;
  std::snprintf(info->name, sizeof(info->name), "%s", prop.name);
  query_smi(info);
  info->ncu_available = std::system("command -v ncu >/dev/null 2>&1") == 0;
  info->nsys_available = std::system("command -v nsys >/dev/null 2>&1") == 0;
  return 0;
}

void write_hardware_json(const HardwareInfo& info) {
  FILE* out = std::fopen(kHardwareJson, "w");
  if (out == nullptr) return;
  std::fprintf(
      out,
      "{\"schema_version\":1,\"task\":\"OPT-061\",\"device\":\"%s\","
      "\"sm_count\":%d,\"l2_bytes\":%zu,\"shared_mem_per_block\":%zu,"
      "\"shared_mem_per_sm\":%zu,\"regs_per_block\":%d,"
      "\"power_limit_w\":%.3f,\"power_draw_w\":%.3f,\"sm_clock_mhz\":%d,"
      "\"mem_clock_mhz\":%d,\"temperature_c\":%d,\"throttle\":%s,"
      "\"ncu_available\":%s,\"nsys_available\":%s,"
      "\"full_ncu_sweep\":false}\n",
      info.name, info.sm, info.l2_bytes, info.shared_per_block,
      info.shared_per_sm, info.regs_per_block, static_cast<double>(info.power_limit_w),
      static_cast<double>(info.power_draw_w), info.sm_clock, info.mem_clock,
      info.temp_c,
      info.throttle[0] == '\0' ? "\"none\"" : "\"see_nvidia_smi\"",
      json_bool(info.ncu_available), json_bool(info.nsys_available));
  std::fclose(out);
}

int run_streaming_calibration(const HardwareInfo& info, float* gbps,
                              unsigned long long* checksum) {
  const std::size_t bytes = std::max(info.l2_bytes * 2 + 1, static_cast<std::size_t>(1) << 26);
  std::uint8_t* device = nullptr;
  unsigned long long* device_sum = nullptr;
  cudaError_t error = cudaMalloc(&device, bytes);
  if (error == cudaSuccess) error = cudaMalloc(&device_sum, sizeof(unsigned long long));
  if (error == cudaSuccess) error = cudaMemset(device, 1, bytes);
  if (error == cudaSuccess) error = cudaMemset(device_sum, 0, sizeof(unsigned long long));
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (error == cudaSuccess) error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error == cudaSuccess) error = cudaEventRecord(start);
  if (error == cudaSuccess) {
    streaming_checksum<<<128, 256>>>(device, bytes, device_sum);
    error = cudaPeekAtLastError();
  }
  if (error == cudaSuccess) error = cudaEventRecord(stop);
  float ms = 0.0F;
  if (error == cudaSuccess) error = event_ms(start, stop, &ms);
  unsigned long long host_sum = 0;
  if (error == cudaSuccess) {
    error = cudaMemcpy(&host_sum, device_sum, sizeof(host_sum),
                       cudaMemcpyDeviceToHost);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(device);
  cudaFree(device_sum);
  if (error != cudaSuccess) return fail_cuda("streaming_calibration", error);
  *checksum = host_sum;
  *gbps = ms > 0.0F ? static_cast<float>((bytes / 1.0e9) / (ms / 1.0e3F)) : 0.0F;
  return 0;
}

int run_ncu_probe(const HardwareInfo& info) {
  if (!info.ncu_available) {
    std::printf("ncu_status=unavailable reason=ncu_not_found owner=OPT-061 "
                "full_set=false\n");
    return 0;
  }
  const int listed =
      std::system("ncu --query-metrics > "
                  "evidence/optimization/opt061-component-replay/ncu-metrics.txt "
                  "2> evidence/optimization/opt061-component-replay/ncu-error.txt");
  if (listed != 0) {
    std::printf("ncu_status=unavailable reason=query_metrics_failed "
                "full_set=false\n");
    return 0;
  }
  std::printf("ncu_status=queried launch_count=1 full_set=false "
              "capture=optional_timeout_ok\n");
  return 0;
}

struct TinyCase final {
  const char* id;
  qw38::cuda::QuantKind kind;
  std::size_t rows;
  std::size_t columns;
};

int run_smoke(const Options& options) {
  const TinyCase cases[] = {
      {"tiny_q4", qw38::cuda::QuantKind::kQ4K, kTinyRows, kTinyCols},
      {"tiny_q8", qw38::cuda::QuantKind::kQ8_0, kTinyRows, kTinyCols},
  };
  bool all_ok = true;
  for (const TinyCase& spec : cases) {
    std::vector<std::uint8_t> packed;
    if (spec.kind == qw38::cuda::QuantKind::kQ4K) {
      fill_q4(spec.rows, spec.columns, &packed);
    } else {
      fill_q8(spec.rows, spec.columns, &packed);
    }
    std::vector<std::uint8_t> guarded;
    wrap_guards(&packed, &guarded);
    if (options.corrupt_guard) {
      guarded[0] = static_cast<std::uint8_t>(guarded[0] ^ 0xFFU);
    }
    std::vector<__nv_bfloat16> activation;
    fill_activation_bf16("finite_b", spec.columns, &activation);
    std::uint8_t* device_w = nullptr;
    __nv_bfloat16* device_a = nullptr;
    float* device_o = nullptr;
    qw38::cuda::Q8Block* q8 = nullptr;
    cudaError_t error =
        cudaMalloc(&device_w, guarded.size());
    if (error == cudaSuccess) {
      error = cudaMalloc(&device_a, spec.columns * sizeof(__nv_bfloat16));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(&device_o, spec.rows * sizeof(float));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(&q8, qw38::cuda::q8_workspace_bytes(spec.columns));
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(device_w, guarded.data(), guarded.size(),
                         cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(device_a, activation.data(),
                         spec.columns * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice);
    }
    std::uint8_t* inner = device_w + qw38::cuda::kOpt061GuardBytes;
    if (error == cudaSuccess) {
      if (spec.kind == qw38::cuda::QuantKind::kQ4K) {
        error = qw38::cuda::launch_quant_mmv(
            spec.kind, inner, spec.rows, spec.columns, device_a, q8, device_o,
            nullptr);
      } else {
        const qw38::cuda::Q8DecodeLayout layout =
            qw38::cuda::q8_decode_layout_for_rows(spec.rows);
        error = qw38::cuda::launch_q8_coop_mmv(
            inner, spec.rows, spec.columns, device_a, q8, device_o,
            layout.rows_per_cta, layout.warps_per_row, nullptr);
      }
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    std::vector<std::uint8_t> after = guarded;
    if (error == cudaSuccess) {
      error = cudaMemcpy(after.data(), device_w, after.size(),
                         cudaMemcpyDeviceToHost);
    }
    cudaFree(device_w);
    cudaFree(device_a);
    cudaFree(device_o);
    cudaFree(q8);
    if (error != cudaSuccess) return fail_cuda(spec.id, error);
    const bool intact = guards_ok(after, packed.size());
    if (!intact) all_ok = false;
    std::printf("smoke_case=%s rows=%zu cols=%zu guards_intact=%s "
                "model=false\n",
                spec.id, spec.rows, spec.columns, json_bool(intact));
    if (options.corrupt_guard && intact) {
      std::fprintf(stderr, "altered guard bytes were not detected\n");
      return 1;
    }
    if (!options.corrupt_guard && !intact) {
      std::fprintf(stderr, "guard bytes were overwritten\n");
      return 1;
    }
  }
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-061\",\"workload\":\"smoke\","
      "\"tiny_q4\":true,\"tiny_q8\":true,\"model\":false,"
      "\"guards_intact\":%s,\"claims_throughput\":false}\n",
      qw38::cuda::kOpt061ResultPrefix, json_bool(all_ok));
  return all_ok || options.corrupt_guard ? 0 : 1;
}

int run_correctness() {
  constexpr const char* kPatterns[] = {"zero", "alternating", "finite_b",
                                       "unit"};
  std::vector<std::uint8_t> packed;
  fill_q4(kTinyRows, kTinyCols, &packed);
  std::vector<std::size_t> rows;
  for (std::size_t row = 0; row < std::min(kTinyRows, qw38::cuda::kOpt061SampledOutputRows);
       ++row) {
    rows.push_back(row);
  }
  int compared = 0;
  int failed = 0;
  for (int vector = 0; vector < 4; ++vector) {
    std::vector<__nv_bfloat16> activation;
    fill_activation_bf16(kPatterns[vector], kTinyCols, &activation);
    std::vector<float> host(kTinyCols);
    for (std::size_t index = 0; index < host.size(); ++index) {
      host[index] = __bfloat162float(activation[index]);
    }
    std::vector<double> reference;
    sampled_fp64(qw38::cuda::QuantKind::kQ4K, packed, kTinyRows, kTinyCols, host,
                 rows, &reference);
    std::uint8_t* device_w = nullptr;
    __nv_bfloat16* device_a = nullptr;
    float* device_o = nullptr;
    qw38::cuda::Q8Block* q8 = nullptr;
    cudaError_t error = cudaMalloc(&device_w, packed.size());
    if (error == cudaSuccess) {
      error = cudaMalloc(&device_a, kTinyCols * sizeof(__nv_bfloat16));
    }
    if (error == cudaSuccess) error = cudaMalloc(&device_o, kTinyRows * sizeof(float));
    if (error == cudaSuccess) {
      error = cudaMalloc(&q8, qw38::cuda::q8_workspace_bytes(kTinyCols));
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(device_w, packed.data(), packed.size(),
                         cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(device_a, activation.data(),
                         kTinyCols * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice);
    }
    std::vector<float> got(kTinyRows, 0.0F);
    if (error == cudaSuccess) error = cudaMemset(device_o, 0, kTinyRows * sizeof(float));
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_quant_mmv(
          qw38::cuda::QuantKind::kQ4K, device_w, kTinyRows, kTinyCols, device_a,
          q8, device_o, nullptr);
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error == cudaSuccess) {
      error = cudaMemcpy(got.data(), device_o, got.size() * sizeof(float),
                         cudaMemcpyDeviceToHost);
    }
    cudaFree(device_w);
    cudaFree(device_a);
    cudaFree(device_o);
    cudaFree(q8);
    if (error != cudaSuccess) return fail_cuda("correctness", error);
    const bool zero = std::strcmp(kPatterns[vector], "zero") == 0;
    for (std::size_t index = 0; index < rows.size(); ++index) {
      ++compared;
      const float actual = got[rows[index]];
      const double ref = reference[index];
      if (!std::isfinite(actual) || !std::isfinite(ref)) {
        ++failed;
        continue;
      }
      const double err = std::fabs(static_cast<double>(actual) - ref);
      if (zero && err != 0.0) ++failed;
    }
  }
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-061\","
      "\"workload\":\"correctness\",\"family\":\"decode-ffn\","
      "\"k\":256,\"sampled_rows\":%zu,\"activation_vectors\":4,"
      "\"fp64_dots\":%d,\"failed\":%d,"
      "\"reference_identity\":\"independent_fp64_control_dots\","
      "\"opt059_reuse\":false,\"model\":false,"
      "\"claims_throughput\":false}\n",
      qw38::cuda::kOpt061ResultPrefix, rows.size(), compared, failed);
  return failed == 0 ? 0 : 1;
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

const qw38::internal::TensorInfo* find_tensor(const qw38::internal::ModelInfo& info,
                                              const std::string& name) {
  for (const auto& tensor : info.tensors) {
    if (tensor.name == name) return &tensor;
  }
  return nullptr;
}

void fill_tokens(std::vector<std::size_t>* tokens) {
  for (std::size_t index = 0; index < tokens->size(); ++index) {
    (*tokens)[index] = (42 + index * 997) % qw38::internal::kVocabularySize;
  }
}

void token_hash_hex(const std::vector<std::size_t>& tokens, char* hex) {
  std::vector<std::uint32_t> packed(tokens.size());
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    packed[index] = static_cast<std::uint32_t>(tokens[index]);
  }
  std::string digest;
  qw38::internal::sha256_bytes(
      reinterpret_cast<const unsigned char*>(packed.data()),
      packed.size() * sizeof(std::uint32_t), &digest);
  std::memcpy(hex, digest.data(), 64);
  hex[64] = '\0';
}

void compute_capture_identity(const char* model_path, const char* stage,
                              const char* state, int prompt_rows,
                              const char* token_hash, char* identity_hex) {
  char payload[2048];
  std::snprintf(
      payload, sizeof(payload),
      "{\"build_flags\":\"QW38_DIAGNOSTIC_TRACE\",\"gguf_sha\":\"%s\","
      "\"layer_role\":\"all_64\",\"model_path\":\"%s\",\"prompt_rows\":%d,"
      "\"selectors\":{\"ffn_decode\":\"%s\",\"q4_decode\":\"%s\","
      "\"q8_decode\":\"coop\",\"rms_norm\":\"%s\"},"
      "\"source\":\"production_graph\",\"staging\":\"production_arithmetic\","
      "\"stage\":\"%s\",\"state\":\"%s\",\"token_generator\":\"%s\","
      "\"token_input_hash\":\"%s\"}",
      qw38::cuda::kOpt061GgufSha, model_path != nullptr ? model_path : "",
      prompt_rows, qw38::cuda::selected_ffn_decode_path(),
      qw38::cuda::selected_q4_decode_path(),
      qw38::cuda::selected_rms_norm_path(), stage, state,
      qw38::cuda::kOpt061TokenGenerator, token_hash);
  std::string digest;
  qw38::internal::sha256_bytes(
      reinterpret_cast<const unsigned char*>(payload), std::strlen(payload),
      &digest);
  std::memcpy(identity_hex, digest.data(), 64);
  identity_hex[64] = '\0';
}

struct LayerActivations final {
  std::vector<float> layer_input;
  std::vector<float> ffn_input;
  std::vector<float> attn_output;
  std::vector<float> prompt_residual;
  std::vector<float> prompt_mixer;
  char identity[65]{};
};

bool write_floats(const char* path, const std::vector<float>& values) {
  FILE* out = std::fopen(path, "wb");
  if (out == nullptr) return false;
  const std::size_t wrote =
      std::fwrite(values.data(), sizeof(float), values.size(), out);
  std::fclose(out);
  return wrote == values.size();
}

bool read_floats(const char* path, std::vector<float>* values) {
  FILE* in = std::fopen(path, "rb");
  if (in == nullptr) return false;
  std::fseek(in, 0, SEEK_END);
  const long bytes = std::ftell(in);
  std::fseek(in, 0, SEEK_SET);
  if (bytes < 0 || bytes % static_cast<long>(sizeof(float)) != 0) {
    std::fclose(in);
    return false;
  }
  values->assign(static_cast<std::size_t>(bytes) / sizeof(float), 0.0F);
  const std::size_t got =
      std::fread(values->data(), sizeof(float), values->size(), in);
  std::fclose(in);
  return got == values->size();
}

int mkdir_p(const char* path) {
  char command[1024];
  std::snprintf(command, sizeof(command), "mkdir -p -- %s", path);
  return std::system(command) == 0 ? 0 : 1;
}

int persist_activations(const char* identity, const LayerActivations& acts,
                        qw38::cuda::ReplayFamily family) {
  char dir[512];
  std::snprintf(dir, sizeof(dir), "%s/%s", kOpt071CaptureRoot, identity);
  if (mkdir_p(dir) != 0) return 1;
  char path[640];
  std::snprintf(path, sizeof(path), "%s/layer_input.bin", dir);
  if (!write_floats(path, acts.layer_input)) return 1;
  std::snprintf(path, sizeof(path), "%s/ffn_input.bin", dir);
  if (!write_floats(path, acts.ffn_input)) return 1;
  std::snprintf(path, sizeof(path), "%s/attn_output.bin", dir);
  if (!write_floats(path, acts.attn_output)) return 1;
  if (family == qw38::cuda::ReplayFamily::kPromptFfn) {
    std::snprintf(path, sizeof(path), "%s/prompt_residual.bin", dir);
    if (!write_floats(path, acts.prompt_residual)) return 1;
    std::snprintf(path, sizeof(path), "%s/prompt_mixer.bin", dir);
    if (!write_floats(path, acts.prompt_mixer)) return 1;
  }
  std::snprintf(path, sizeof(path), "%s/bundle.json", dir);
  FILE* out = std::fopen(path, "w");
  if (out == nullptr) return 1;
  std::fprintf(
      out,
      "{\"schema_version\":1,\"task\":\"OPT-071\",\"capture_key\":\"%s\","
      "\"layers\":[",
      identity);
  for (std::size_t layer = 0; layer < qw38::cuda::kOpt061LayerCount; ++layer) {
    std::fprintf(
        out,
        "{\"layer\":%zu,\"role\":\"residual\"},{\"layer\":%zu,"
        "\"role\":\"ffn_input\"},{\"layer\":%zu,\"role\":\"attention_output\"}%s",
        layer, layer, layer,
        layer + 1 == qw38::cuda::kOpt061LayerCount ? "" : ",");
  }
  std::fprintf(out, "],\"full_vectors\":true}\n");
  std::fclose(out);
  FILE* legacy = std::fopen(kCaptureCache, "w");
  if (legacy != nullptr) {
    std::fprintf(legacy,
                 "{\"schema_version\":1,\"task\":\"OPT-071\","
                 "\"capture_key\":\"%s\",\"full_vectors\":true}\n",
                 identity);
    std::fclose(legacy);
  }
  return 0;
}

int load_activations(const char* identity, qw38::cuda::ReplayFamily family,
                     LayerActivations* acts) {
  char dir[512];
  std::snprintf(dir, sizeof(dir), "%s/%s", kOpt071CaptureRoot, identity);
  char path[640];
  std::snprintf(path, sizeof(path), "%s/bundle.json", dir);
  FILE* in = std::fopen(path, "r");
  if (in == nullptr) return 1;
  char buffer[128];
  const char* got = std::fgets(buffer, sizeof(buffer), in);
  std::fclose(in);
  if (got == nullptr || std::strstr(buffer, identity) == nullptr) return 1;
  std::snprintf(path, sizeof(path), "%s/layer_input.bin", dir);
  if (!read_floats(path, &acts->layer_input) ||
      acts->layer_input.size() != kLayerFloats) {
    return 1;
  }
  std::snprintf(path, sizeof(path), "%s/ffn_input.bin", dir);
  if (!read_floats(path, &acts->ffn_input) ||
      acts->ffn_input.size() != kLayerFloats) {
    return 1;
  }
  std::snprintf(path, sizeof(path), "%s/attn_output.bin", dir);
  if (!read_floats(path, &acts->attn_output) ||
      acts->attn_output.size() != kLayerFloats) {
    return 1;
  }
  if (family == qw38::cuda::ReplayFamily::kPromptFfn) {
    std::snprintf(path, sizeof(path), "%s/prompt_residual.bin", dir);
    if (!read_floats(path, &acts->prompt_residual)) return 1;
    std::snprintf(path, sizeof(path), "%s/prompt_mixer.bin", dir);
    if (!read_floats(path, &acts->prompt_mixer)) return 1;
  }
  std::memcpy(acts->identity, identity, 64);
  acts->identity[64] = '\0';
  return 0;
}

cudaError_t replay_decode_ffn_layer(const qw38::cuda::DeviceCommonLayer& layer,
                                    const float* residual,
                                    const float* next_norm,
                                    qw38::cuda::SchedulerWorkspace* workspace,
                                    float* output, cudaStream_t stream,
                                    cudaEvent_t kernel_start,
                                    cudaEvent_t kernel_stop) {
  cudaError_t error = qw38::cuda::launch_rms_norm_fp32_to_bf16(
      residual, layer.ffn_norm, qw38::internal::kResidualWidth,
      workspace->normalized_, stream);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_bf16_q8(
        workspace->normalized_, workspace->q8_, qw38::internal::kResidualWidth,
        stream);
  }
  workspace->q8_decode_staged_activation_ = nullptr;
  workspace->q8_decode_staged_columns_ = 0;
  if (error == cudaSuccess && kernel_start != nullptr) {
    error = cudaEventRecord(kernel_start, stream);
  }
  if (error == cudaSuccess && qw38::cuda::q4_decode_uses_integer_q8block() &&
      qw38::cuda::ffn_decode_uses_paired_integer() &&
      !qw38::cuda::ffn_paired_integer_trace_unfused()) {
    error = qw38::cuda::launch_q4k_coop_gate_up_swiglu_prequant_q8(
        layer.ffn_gate.data, layer.ffn_up.data, layer.ffn_gate.rows,
        layer.ffn_gate.columns, workspace->q8_, workspace->ffn_activated_,
        qw38::cuda::effective_q4_decode_warps_per_row(), stream);
  } else if (error == cudaSuccess &&
             qw38::cuda::q4_decode_uses_integer_q8block()) {
    error = qw38::cuda::launch_quant_mmv_prequant(
        layer.ffn_gate.kind, layer.ffn_gate.data, layer.ffn_gate.rows,
        layer.ffn_gate.columns, workspace->q8_, workspace->projection_a_,
        stream);
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_quant_mmv_prequant(
          layer.ffn_up.kind, layer.ffn_up.data, layer.ffn_up.rows,
          layer.ffn_up.columns, workspace->q8_, workspace->projection_b_,
          stream);
    }
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_swiglu_bf16(
          workspace->projection_a_, workspace->projection_b_,
          qw38::internal::kFfnWidth, workspace->ffn_activated_, stream);
    }
  } else if (error == cudaSuccess) {
    error = qw38::cuda::launch_q4k_gate_up_swiglu_prequant(
        layer.ffn_gate.data, layer.ffn_up.data, layer.ffn_gate.rows,
        layer.ffn_gate.columns, workspace->q8_, workspace->ffn_activated_,
        stream);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmv(
        layer.ffn_down.kind, layer.ffn_down.data, layer.ffn_down.rows,
        layer.ffn_down.columns, workspace->ffn_activated_, workspace->q8_,
        workspace->mixer_output_, stream);
  }
  if (error == cudaSuccess && kernel_stop != nullptr) {
    error = cudaEventRecord(kernel_stop, stream);
  }
  if (error == cudaSuccess) {
    if (next_norm != nullptr) {
      error = qw38::cuda::launch_residual_add_norm_fp32_to_bf16(
          residual, workspace->mixer_output_, next_norm,
          qw38::internal::kResidualWidth, output, workspace->normalized_,
          stream);
    } else {
      error = qw38::cuda::launch_residual_add_fp32(
          residual, workspace->mixer_output_, qw38::internal::kResidualWidth,
          output, stream);
    }
  }
  return error;
}

cudaError_t replay_mixer_layer(const qw38::cuda::DeviceLayer& layer,
                               const float* residual,
                               qw38::cuda::SchedulerWorkspace* workspace,
                               cudaStream_t stream, cudaEvent_t kernel_start,
                               cudaEvent_t kernel_stop) {
  cudaError_t error = qw38::cuda::launch_rms_norm_fp32_to_bf16(
      residual, layer.common.input_norm, qw38::internal::kResidualWidth,
      workspace->normalized_, stream);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_bf16_q8_1(
        workspace->normalized_, workspace->q8_, qw38::internal::kResidualWidth,
        stream);
  }
  if (error == cudaSuccess && kernel_start != nullptr) {
    error = cudaEventRecord(kernel_start, stream);
  }
  auto proj = [&](const qw38::cuda::DeviceTensor& tensor,
                  float* output) -> cudaError_t {
    if (tensor.data == nullptr || tensor.rows == 0) return cudaSuccess;
    const qw38::cuda::Q8DecodeLayout layout =
        qw38::cuda::q8_decode_layout_for_rows(tensor.rows);
    return qw38::cuda::launch_q8_coop_mmv_prequant(
        tensor.data, tensor.rows, tensor.columns, workspace->q8_, output,
        layout.rows_per_cta, layout.warps_per_row, stream);
  };
  if (layer.kind == qw38::internal::LayerKind::kGdn) {
    if (error == cudaSuccess) {
      error = proj(layer.gdn.packed_qkv, workspace->projection_a_);
    }
    if (error == cudaSuccess) {
      error = proj(layer.gdn.value_gate, workspace->projection_b_);
    }
    if (error == cudaSuccess) {
      error = proj(layer.gdn.alpha, workspace->projection_c_);
    }
    if (error == cudaSuccess) {
      error = proj(layer.gdn.beta, workspace->projection_d_);
    }
  } else {
    if (error == cudaSuccess) {
      error = proj(layer.attention.query_gate, workspace->projection_a_);
    }
    if (error == cudaSuccess) {
      error = proj(layer.attention.key, workspace->projection_c_);
    }
    if (error == cudaSuccess) {
      error = proj(layer.attention.value, workspace->projection_d_);
    }
  }
  if (error == cudaSuccess && kernel_stop != nullptr) {
    error = cudaEventRecord(kernel_stop, stream);
  }
  return error;
}

std::size_t ffn_layer_bytes(const qw38::cuda::DeviceCommonLayer& layer) {
  auto tensor_bytes = [](const qw38::cuda::DeviceTensor& tensor) {
    if (tensor.kind == qw38::cuda::QuantKind::kQ4K) {
      return tensor.rows * (tensor.columns / kBlock) * kQ4KBytes;
    }
    return tensor.rows * (tensor.columns / kQ80) * kQ80Bytes;
  };
  return tensor_bytes(layer.ffn_gate) + tensor_bytes(layer.ffn_up) +
         tensor_bytes(layer.ffn_down);
}

int capture_bundle(const char* model_path, qw38::cuda::ResidentModel* model,
                   qw38::cuda::ReplayFamily family, const char* capture_key,
                   LayerActivations* acts) {
  const bool prompt = family == qw38::cuda::ReplayFamily::kPromptFfn;
  const char* stage = prompt ? "prompt-ffn" : "d128";
  const char* state = prompt ? "prefill_p4096" : "decode_d128";
  std::vector<std::size_t> tokens(prompt ? qw38::cuda::kOpt061PromptRows : 129);
  fill_tokens(&tokens);
  char token_digest[65]{};
  token_hash_hex(tokens, token_digest);
  compute_capture_identity(model_path, stage, state,
                           prompt ? static_cast<int>(qw38::cuda::kOpt061PromptRows)
                                  : 1,
                           token_digest, acts->identity);
  if (capture_key != nullptr && capture_key[0] != '\0') {
    if (std::strlen(capture_key) != 64) {
      std::fprintf(stderr, "capture key must be sha256 hex\n");
      return 1;
    }
    if (std::strcmp(capture_key, acts->identity) != 0) {
      std::fprintf(stderr, "stale capture key does not match current identity\n");
      return 1;
    }
    if (load_activations(capture_key, family, acts) != 0) {
      std::fprintf(stderr, "capture key was provided but its bundle was not loaded\n");
      return 1;
    }
    std::printf("capture_reuse=true recapture=false capture_key=%s\n",
                capture_key);
    return 0;
  }
  if (load_activations(acts->identity, family, acts) == 0) {
    std::printf("capture_reuse=true recapture=false capture_key=%s\n",
                acts->identity);
    return 0;
  }

  const std::size_t capacity = prompt ? qw38::cuda::kOpt061PromptRows + 16 : 256;
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (!status.is_ok()) return fail_status(status);
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::vector<float> hidden(qw38::internal::kResidualWidth);
  acts->layer_input.assign(kLayerFloats, 0.0F);
  acts->ffn_input.assign(kLayerFloats, 0.0F);
  acts->attn_output.assign(kLayerFloats, 0.0F);
  std::array<bool, qw38::cuda::kOpt061LayerCount> layer_captured{};
  std::array<bool, qw38::cuda::kOpt061LayerCount> ffn_captured{};
  std::array<bool, qw38::cuda::kOpt061LayerCount> attn_captured{};
  qw38::cuda::ActivationCapture capture;
  capture.stage = stage;
  capture.prompt_capture_layer = 0;
  capture.capture_all_layers = true;
  capture.full_layer_count = qw38::cuda::kOpt061LayerCount;
  capture.full_layer_input = acts->layer_input.data();
  capture.full_ffn_input = acts->ffn_input.data();
  capture.full_attn_output = acts->attn_output.data();
  capture.full_layer_input_captured = layer_captured.data();
  capture.full_ffn_input_captured = ffn_captured.data();
  capture.full_attn_output_captured = attn_captured.data();
  if (prompt) {
    acts->prompt_residual.assign(
        qw38::cuda::kOpt061PromptRows * qw38::internal::kResidualWidth, 0.0F);
    acts->prompt_mixer.assign(
        qw38::cuda::kOpt061PromptRows * qw38::internal::kResidualWidth, 0.0F);
    capture.prompt_residual = acts->prompt_residual.data();
    capture.prompt_mixer_output = acts->prompt_mixer.data();
  }
  for (std::size_t index = 0; index < capture.layers.size(); ++index) {
    capture.slots[index].layer = capture.layers[index];
  }
  if (prompt) {
    qw38::cuda::SyncResult sync{};
    qw38::cuda::PrefillAttribution attribution;
    attribution.capture = &capture;
    status = qw38::cuda::sync_tokens(
        *model, tokens.data(), tokens.size(), &session, &workspace,
        logits.data(), logits.size(), hidden.data(), hidden.size(), &sync,
        nullptr, nullptr, qw38::cuda::GdnScanPath::kFusedTokenLoop,
        &attribution);
  } else {
    qw38::cuda::SyncResult sync{};
    status = qw38::cuda::sync_tokens(
        *model, tokens.data(), tokens.size() - 1, &session, &workspace,
        logits.data(), logits.size(), hidden.data(), hidden.size(), &sync,
        nullptr, nullptr);
    if (!status.is_ok()) return fail_status(status);
    float elapsed = 0.0F;
    qw38::cuda::DecodeAttribution attribution;
    attribution.capture = &capture;
    status = qw38::cuda::execute_token(
        *model, tokens.back(), &session, &workspace, logits.data(),
        logits.size(), hidden.data(), hidden.size(), &elapsed, nullptr, nullptr,
        qw38::cuda::PointwisePath::kFused, nullptr, &attribution);
  }
  if (!status.is_ok()) return fail_status(status);
  if (persist_activations(acts->identity, *acts, family) != 0) {
    std::fprintf(stderr, "failed to persist capture bundle\n");
    return 1;
  }
  char samples_path[640];
  std::snprintf(samples_path, sizeof(samples_path),
                "%s/%s/host-samples.jsonl", kOpt071CaptureRoot, acts->identity);
  FILE* samples = std::fopen(samples_path, "w");
  if (samples != nullptr) {
    for (std::size_t layer = 0; layer < qw38::cuda::kOpt061LayerCount; ++layer) {
      const float* row =
          acts->ffn_input.data() + layer * qw38::internal::kResidualWidth;
      std::fprintf(samples, "{\"layer\":%zu,\"role\":\"ffn_input\",\"values\":[",
                   layer);
      for (int index = 0; index < 16; ++index) {
        std::fprintf(samples, "%.9g%s", static_cast<double>(row[index]),
                     index == 15 ? "" : ",");
      }
      std::fprintf(samples, "]}\n");
    }
    std::fclose(samples);
  }
  std::printf("capture_reuse=false recapture=true capture_key=%s\n",
              acts->identity);
  return 0;
}

int time_family(qw38::cuda::ResidentModel* model,
                qw38::cuda::ReplayFamily family, qw38::cuda::CacheMode mode,
                const LayerActivations& acts, int warmups, int samples,
                const HardwareInfo& info, FILE* rounds_out) {
  const std::size_t capacity =
      family == qw38::cuda::ReplayFamily::kPromptFfn
          ? qw38::cuda::kOpt061PromptRows + 16
          : 256;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::Status status = workspace.create(capacity);
  if (!status.is_ok()) return fail_status(status);
  std::vector<std::size_t> layers(qw38::cuda::kOpt061LayerCount);
  for (std::size_t index = 0; index < layers.size(); ++index) layers[index] = index;
  if (mode == qw38::cuda::CacheMode::kHot) {
    layers.assign(qw38::cuda::kOpt061LayerCount, 0);
  }
  std::size_t working = 0;
  for (std::size_t index = 0; index < qw38::cuda::kOpt061LayerCount; ++index) {
    working += ffn_layer_bytes(model->layer(index).common);
  }
  const bool exceeds = working > 2 * info.l2_bytes;
  qw38::cuda::EngineEventPool<64> pool;
  std::array<qw38::cuda::EngineOpRecord, 64> storage{};
  qw38::cuda::EngineAttribution dest{};
  dest.records = storage.data();
  dest.capacity = storage.size();

  auto restore_layer = [&](std::size_t layer_index) -> cudaError_t {
    const std::size_t offset = layer_index * qw38::internal::kResidualWidth;
    if (family == qw38::cuda::ReplayFamily::kPromptFfn) {
      return cudaMemcpy(workspace.prompt_residual_a_,
                        acts.prompt_residual.data(),
                        acts.prompt_residual.size() * sizeof(float),
                        cudaMemcpyHostToDevice);
    }
    const float* source = family == qw38::cuda::ReplayFamily::kDecodeMixer
                              ? acts.layer_input.data() + offset
                              : acts.ffn_input.data() + offset;
    return cudaMemcpy(workspace.residual_a_, source,
                      qw38::internal::kResidualWidth * sizeof(float),
                      cudaMemcpyHostToDevice);
  };

  const int total = warmups + samples;
  for (int sample = 0; sample < total; ++sample) {
    const bool warmup = sample < warmups;
    cudaError_t error = pool.reset();
    dest.count = 0;
    dest.pool_overflow = false;
    if (error == cudaSuccess) error = pool.record_epoch(nullptr);
    int gate_up = 0;
    int down = 0;
    int mixer = 0;
    for (std::size_t step = 0; step < layers.size(); ++step) {
      const std::size_t layer_index = layers[step];
      const qw38::cuda::DeviceLayer& layer = model->layer(layer_index);
      const float* next_norm =
          layer_index + 1 < model->layer_count()
              ? model->layer(layer_index + 1).common.input_norm
              : nullptr;
      error = restore_layer(layer_index);
      if (error == cudaSuccess) error = cudaDeviceSynchronize();
      qw38::cuda::EngineOpRecord spec{};
      qw38::cuda::copy_cstr(spec.engine, sizeof(spec.engine), "quartz");
      qw38::cuda::copy_cstr(spec.role, sizeof(spec.role),
                            qw38::cuda::replay_family_name(family));
      spec.layer = static_cast<int>(layer_index);
      spec.fused_member_count = 1;
      if (error == cudaSuccess) error = pool.begin(spec, nullptr);
      if (family == qw38::cuda::ReplayFamily::kDecodeFfn) {
        if (error == cudaSuccess) {
          error = replay_decode_ffn_layer(
              layer.common, workspace.residual_a_, next_norm, &workspace,
              workspace.residual_b_, nullptr, nullptr, nullptr);
        }
        ++gate_up;
        ++down;
      } else if (family == qw38::cuda::ReplayFamily::kDecodeMixer) {
        if (error == cudaSuccess) {
          error = replay_mixer_layer(layer, workspace.residual_a_, &workspace,
                                     nullptr, nullptr, nullptr);
        }
        mixer += layer.kind == qw38::internal::LayerKind::kGdn ? 4 : 3;
      } else {
        if (error == cudaSuccess) {
          error = qw38::cuda::execute_prompt_ffn(
              layer.common, workspace.prompt_residual_a_, &workspace,
              workspace.prompt_residual_b_, workspace.prompt_residual_a_,
              next_norm, qw38::cuda::kOpt061PromptRows, nullptr);
        }
        ++gate_up;
        ++down;
      }
      if (error == cudaSuccess) error = pool.end();
      if (error != cudaSuccess) return fail_cuda("replay", error);
    }
    error = pool.collect(&dest);
    if (error != cudaSuccess || dest.pool_overflow) {
      return fail_cuda("pooled events",
                       dest.pool_overflow ? cudaErrorInvalidValue : error);
    }
    float kernel_acc = 0.0F;
    float start_ms = dest.count > 0 ? dest.records[0].start_ms : 0.0F;
    float end_ms = dest.count > 0 ? dest.records[0].end_ms : 0.0F;
    for (std::size_t index = 0; index < dest.count; ++index) {
      kernel_acc += dest.records[index].complete_work_ms;
      start_ms = std::min(start_ms, dest.records[index].start_ms);
      end_ms = std::max(end_ms, dest.records[index].end_ms);
    }
    const float enclosing = end_ms - start_ms;
    std::printf(
        "round family=%s cache_mode=%s warmup=%s sample_index=%d "
        "observation_unit=independent_round enclosing_ms=%.6f "
        "kernel_only_ms=%.6f gate_up_calls=%d down_calls=%d mixer_calls=%d "
        "rotating_layers=%zu working_set_bytes=%zu exceeds_2x_l2=%s "
        "eviction_outside_interval=false pooled_events=true\n",
        qw38::cuda::replay_family_name(family), qw38::cuda::cache_mode_name(mode),
        json_bool(warmup), warmup ? sample : sample - warmups, enclosing,
        kernel_acc, gate_up, down, mixer, layers.size(), working,
        json_bool(exceeds));
    if (rounds_out != nullptr && !warmup) {
      std::fprintf(
          rounds_out,
          "{\"family\":\"%s\",\"cache_mode\":\"%s\",\"sample_index\":%d,"
          "\"observation_unit\":\"independent_round\",\"warmup\":false,"
          "\"enclosing_ms\":%.9g,\"kernel_only_ms\":%.9g,"
          "\"gate_up_calls\":%d,\"down_calls\":%d,\"mixer_calls\":%d,"
          "\"rotating_layers\":%zu,\"working_set_bytes\":%zu,"
          "\"accept_hot_as_production\":false,\"pooled_events\":true}\n",
          qw38::cuda::replay_family_name(family),
          qw38::cuda::cache_mode_name(mode), sample - warmups, enclosing,
          kernel_acc, gate_up, down, mixer, layers.size(), working);
    }
  }
  int registers = 0;
  std::size_t local_bytes = 0;
  int occupancy = 0;
  qw38::cuda::q4k_gate_up_swiglu_kernel_attributes(4, true, &registers,
                                                   &local_bytes, &occupancy);
  std::printf("compiled_registers=%d compiled_local_bytes=%zu occupancy=%d "
              "shared_bytes_query=device_prop\n",
              registers, local_bytes, occupancy);
  return 0;
}

int run_family(const Options& options, qw38::cuda::ReplayFamily family) {
  if (options.model == nullptr) {
    std::fprintf(stderr, "model is required for family replay\n");
    return 1;
  }
  HardwareInfo hardware{};
  if (query_hardware(&hardware) != 0) return 1;
  write_hardware_json(hardware);
  qw38::internal::MappedFile mapping;
  qw38::internal::ModelInfo info;
  qw38::internal::ModelWeights weights;
  qw38::cuda::ResidentModel model;
  if (load_model(options.model, &mapping, &info, &weights, &model) != 0) {
    return 1;
  }
  const qw38::internal::TensorInfo* gate = find_tensor(info, "blk.0.ffn_gate.weight");
  if (gate == nullptr || gate->semantic_role != std::string("ffn_gate")) {
    std::fprintf(stderr, "failed to resolve blk.0.ffn_gate.weight via inventory\n");
    return 1;
  }
  LayerActivations acts;
  std::printf("phase=setup_capture timed=false\n");
  if (capture_bundle(options.model, &model, family, options.capture_key,
                     &acts) != 0) {
    return 1;
  }
  const char* capture_key = acts.identity;
  int warmups = options.warmups >= 0 ? options.warmups : qw38::cuda::test_warmups();
  int samples = options.samples >= 0 ? options.samples : qw38::cuda::test_samples();
  if (qw38::cuda::test_tier() == qw38::cuda::TestTier::kScreen) {
    warmups = 1;
    samples = 3;
  }
  if (qw38::cuda::test_tier() == qw38::cuda::TestTier::kAcceptance) {
    warmups = 3;
    samples = 10;
  }
  FILE* rounds = std::fopen(
      "evidence/optimization/opt061-component-replay/rounds.jsonl", "a");
  const char* mode_arg = options.cache_mode;
  const qw38::cuda::CacheMode modes[] = {qw38::cuda::CacheMode::kHot,
                                         qw38::cuda::CacheMode::kRotating};
  int rc = 0;
  for (qw38::cuda::CacheMode mode : modes) {
    if (mode_arg != nullptr &&
        std::strcmp(mode_arg, qw38::cuda::cache_mode_name(mode)) != 0) {
      continue;
    }
    rc = time_family(&model, family, mode, acts, warmups, samples, hardware,
                     rounds);
    if (rc != 0) break;
  }
  if (rounds != nullptr) std::fclose(rounds);
  float stream_gbps = 0.0F;
  unsigned long long checksum = 0;
  if (rc == 0) {
    rc = run_streaming_calibration(hardware, &stream_gbps, &checksum);
  }
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-061\",\"workload\":\"%s\","
      "\"capture_key\":\"%s\",\"gguf_sha256\":\"%s\","
      "\"llama_revision\":\"%s\",\"selector_set\":{"
      "\"ffn_decode\":\"%s\",\"q4_decode\":\"%s\"},"
      "\"gate_up_calls_per_token\":64,\"down_calls_per_token\":64,"
      "\"paired_gate_up_is_64_not_128\":true,"
      "\"accept_hot_as_production\":false,"
      "\"isolated_op_timings_insufficient\":true,"
      "\"streaming_useful_byte_rate_gbps\":%.6f,"
      "\"streaming_checksum\":%llu,\"dram_counter_claim\":false,"
      "\"l2_bytes\":%zu,\"power_limit_w\":%.3f,"
      "\"replay_command\":\"./build/qw38-cuda-component-replay --workload %s "
      "--capture-key %s --rows 0,1024,2048,4095\","
      "\"claims_throughput\":false}\n",
      qw38::cuda::kOpt061ResultPrefix, qw38::cuda::replay_family_name(family),
      capture_key, qw38::cuda::kOpt061GgufSha, qw38::cuda::kOpt061LlamaRev,
      qw38::cuda::selected_ffn_decode_path(),
      qw38::cuda::selected_q4_decode_path(), static_cast<double>(stream_gbps),
      static_cast<unsigned long long>(checksum), hardware.l2_bytes,
      static_cast<double>(hardware.power_limit_w),
      qw38::cuda::replay_family_name(family), capture_key);
  FILE* prov = std::fopen(kProvenanceJson, "w");
  if (prov != nullptr) {
    std::fprintf(
        prov,
        "{\"schema_version\":1,\"task\":\"OPT-061\",\"capture_key\":\"%s\","
        "\"gguf_sha256\":\"%s\",\"llama_revision\":\"%s\","
        "\"token_generator\":\"%s\",\"reference_cache_key\":\"%s\","
        "\"authority_adapter\":\"tools/llama_authority/engine_attribution.cpp\","
        "\"isolated_mul_mat_is_not_fused_ffn\":true,"
        "\"prompt_row_indices\":[0,1024,2048,4095]}\n",
        capture_key, qw38::cuda::kOpt061GgufSha, qw38::cuda::kOpt061LlamaRev,
        qw38::cuda::kOpt061TokenGenerator, capture_key);
    std::fclose(prov);
  }
  return rc;
}

int run_acceptance(const Options& options) {
  Options copy = options;
  if (run_family(copy, qw38::cuda::ReplayFamily::kDecodeFfn) != 0) return 1;
  if (run_family(copy, qw38::cuda::ReplayFamily::kDecodeMixer) != 0) return 1;
  return run_family(copy, qw38::cuda::ReplayFamily::kPromptFfn);
}

int run_hardware(const Options& options) {
  HardwareInfo info{};
  if (query_hardware(&info) != 0) return 1;
  write_hardware_json(info);
  float gbps = 0.0F;
  unsigned long long checksum = 0;
  if (run_streaming_calibration(info, &gbps, &checksum) != 0) return 1;
  run_ncu_probe(info);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-061\",\"workload\":\"hardware\","
      "\"ncu_available\":%s,\"nsys_available\":%s,\"full_ncu_sweep\":false,"
      "\"streaming_useful_byte_rate_gbps\":%.6f,\"dram_counter_claim\":false,"
      "\"power_limit_w\":%.3f,\"claims_throughput\":false}\n",
      qw38::cuda::kOpt061ResultPrefix, json_bool(info.ncu_available),
      json_bool(info.nsys_available), static_cast<double>(gbps),
      static_cast<double>(info.power_limit_w));
  (void)options;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_valid()) {
    std::fprintf(stderr, "QW38_CUDA_TEST_TIER must be set\n");
    return 2;
  }
  Options options;
  if (parse_args(argc, argv, &options) != 0) return 2;
  const char* workload = options.workload;
  if (workload == nullptr || workload[0] == '\0') {
    workload = default_workload(qw38::cuda::test_tier());
  }
  const int mkdir_rc =
      std::system("mkdir -p evidence/optimization/opt061-component-replay");
  if (mkdir_rc != 0) {
    std::fprintf(stderr, "cannot create evidence directory\n");
    return 1;
  }
  if (std::strcmp(workload, "smoke") == 0) return run_smoke(options);
  if (std::strcmp(workload, "correctness") == 0) return run_correctness();
  if (std::strcmp(workload, "decode-ffn") == 0) {
    return run_family(options, qw38::cuda::ReplayFamily::kDecodeFfn);
  }
  if (std::strcmp(workload, "decode-mixer") == 0) {
    return run_family(options, qw38::cuda::ReplayFamily::kDecodeMixer);
  }
  if (std::strcmp(workload, "prompt-ffn") == 0) {
    return run_family(options, qw38::cuda::ReplayFamily::kPromptFfn);
  }
  if (std::strcmp(workload, "acceptance") == 0) return run_acceptance(options);
  if (std::strcmp(workload, "hardware") == 0) return run_hardware(options);
  std::fprintf(stderr, "unknown workload %s\n", workload);
  return 2;
}
