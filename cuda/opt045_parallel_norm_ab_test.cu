#include "full_scheduler.h"
#include "model.h"
#include "rms_norm.cuh"
#include "scheduler.h"
#include "scheduler_primitives.h"
#include "weights.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

namespace {

constexpr char kPrefix[] = "QW38_OPT045_PARALLEL_NORM_AB_RESULT=";
constexpr int kWarmups = 3;
constexpr int kMeasured = 30;
constexpr std::size_t kWidth = 5120;
constexpr std::size_t kHeadWidth = 128;
constexpr std::size_t kPromptRows = 4096;
constexpr std::size_t kGdnHeads = 48;
constexpr float kEps = 1.0e-6F;

struct Candidate {
  const char* id;
  const char* path;
  int residual_threads;
  int gdn_threads;
};

constexpr Candidate kCandidates[] = {
    {"serial", "serial", 256, 256},
    {"parallel_fma_t128", "parallel_fma", 128, 128},
    {"parallel_fma_t256", "parallel_fma", 256, 32},
    {"parallel_rsqrt_t128", "parallel_rsqrt", 128, 128},
    {"parallel_rsqrt_t256", "parallel_rsqrt", 256, 32},
};
constexpr int kCandidateCount =
    static_cast<int>(sizeof(kCandidates) / sizeof(kCandidates[0]));

// OPT-043 capture-d128 layer-0 FFN preprojection prefix (realistic magnitudes).
constexpr float kOpt043Prefix[16] = {
    0.0349121094F, -0.0693359375F, -0.248046875F, 0.12890625F,
    -0.30859375F,  -0.113769531F,  0.0581054688F, -0.170898438F,
    -0.172851562F, -0.064453125F,  -0.173828125F, 0.26953125F,
    -0.245117188F, -0.0164794922F, -0.0693359375F, 0.0766601562F};

int fail_cuda(const char* operation, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
  return 1;
}

const char* json_bool(bool value) { return value ? "true" : "false"; }

void print_array(FILE* file, const char* key, const std::vector<float>& values) {
  std::fprintf(file, "\"%s\":[", key);
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) std::fprintf(file, ",");
    std::fprintf(file, "%.9g", static_cast<double>(values[index]));
  }
  std::fprintf(file, "]");
}

float bf16_float(__nv_bfloat16 value) { return __bfloat162float(value); }

struct Envelope {
  float max_abs = 0.0F;
  float rms = 0.0F;
  float one_minus_cosine = 0.0F;
  std::size_t nonfinite = 0;
  bool residual_fp32_exact = true;
};

Envelope compare_bf16(const std::vector<__nv_bfloat16>& actual,
                      const std::vector<__nv_bfloat16>& expected) {
  Envelope metrics;
  double squared = 0.0;
  double dot = 0.0;
  double norm_a = 0.0;
  double norm_e = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float value = bf16_float(actual[index]);
    const float reference = bf16_float(expected[index]);
    if (!std::isfinite(value)) ++metrics.nonfinite;
    const float error = std::fabs(value - reference);
    metrics.max_abs = std::max(metrics.max_abs, error);
    squared += static_cast<double>(error) * error;
    dot += static_cast<double>(value) * reference;
    norm_a += static_cast<double>(value) * value;
    norm_e += static_cast<double>(reference) * reference;
  }
  metrics.rms = static_cast<float>(
      std::sqrt(squared / static_cast<double>(actual.size())));
  if (norm_a == 0.0 && norm_e == 0.0) {
    metrics.one_minus_cosine = 0.0F;
  } else if (norm_a == 0.0 || norm_e == 0.0) {
    metrics.one_minus_cosine = 1.0F;
  } else {
    const double cosine = dot / (std::sqrt(norm_a) * std::sqrt(norm_e));
    metrics.one_minus_cosine = static_cast<float>(
        1.0 - std::min(1.0, std::max(-1.0, cosine)));
  }
  return metrics;
}

void host_fp64_norm(const std::vector<float>& input,
                    const std::vector<float>& scale,
                    std::vector<__nv_bfloat16>* output) {
  double sum = 0.0;
  for (float value : input) sum += static_cast<double>(value) * value;
  const double inverse =
      1.0 / std::sqrt(sum / static_cast<double>(input.size()) + kEps);
  output->resize(input.size());
  for (std::size_t index = 0; index < input.size(); ++index) {
    const double value =
        static_cast<double>(input[index]) * inverse *
        static_cast<double>(scale[index]);
    (*output)[index] = __float2bfloat16_rn(static_cast<float>(value));
  }
}

void host_llama_rsqrt(const std::vector<float>& input,
                      const std::vector<float>& scale, int threads,
                      std::vector<__nv_bfloat16>* output) {
  std::vector<float> partial(static_cast<std::size_t>(threads), 0.0F);
  for (int tid = 0; tid < threads; ++tid) {
    float sum = 0.0F;
    for (std::size_t index = static_cast<std::size_t>(tid); index < input.size();
         index += static_cast<std::size_t>(threads)) {
      sum = std::fmaf(input[index], input[index], sum);
    }
    partial[static_cast<std::size_t>(tid)] = sum;
  }
  float total = 0.0F;
  for (float value : partial) total += value;
  const float inverse =
      1.0F / std::sqrt(total / static_cast<float>(input.size()) + kEps);
  output->resize(input.size());
  for (std::size_t index = 0; index < input.size(); ++index) {
    (*output)[index] = __float2bfloat16_rn(
        (input[index] * inverse) * scale[index]);
  }
}

void fill_unit(std::vector<float>* values, float scale) {
  for (std::size_t index = 0; index < values->size(); ++index) {
    (*values)[index] =
        std::sin(static_cast<float>(index) * 0.013F) * scale;
  }
}

void fill_opt043(std::vector<float>* values) {
  for (std::size_t index = 0; index < values->size(); ++index) {
    (*values)[index] = kOpt043Prefix[index % 16];
    if (index >= 16) {
      (*values)[index] +=
          std::cos(static_cast<float>(index) * 0.007F) * 0.05F;
    }
  }
}

struct Timed {
  std::vector<float> warmup;
  std::vector<float> samples;
  float mean = 0.0F;
  bool launch_ok = false;
};

template <typename Launch>
Timed time_kernel(const Candidate& candidate, Launch launch) {
  Timed timed;
  qw38::cuda::RmsNormPathScope scope(candidate.path, candidate.residual_threads,
                                     candidate.gdn_threads);
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaError_t error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  for (int sample = -kWarmups; error == cudaSuccess && sample < kMeasured;
       ++sample) {
    error = cudaEventRecord(start);
    if (error == cudaSuccess) error = launch();
    if (error == cudaSuccess) error = cudaEventRecord(stop);
    if (error == cudaSuccess) error = cudaEventSynchronize(stop);
    float elapsed = 0.0F;
    if (error == cudaSuccess) error = cudaEventElapsedTime(&elapsed, start, stop);
    if (error != cudaSuccess) break;
    if (sample < 0) timed.warmup.push_back(elapsed);
    else {
      timed.samples.push_back(elapsed);
      timed.mean += elapsed;
    }
  }
  if (stop != nullptr) cudaEventDestroy(stop);
  if (start != nullptr) cudaEventDestroy(start);
  timed.launch_ok = error == cudaSuccess &&
                    timed.samples.size() == static_cast<std::size_t>(kMeasured);
  if (timed.launch_ok) timed.mean /= static_cast<float>(kMeasured);
  return timed;
}

void emit_timed(FILE* file, const Timed& timed) {
  std::fprintf(file, "\"mean_ms\":%.9g,\"launch_ok\":%s,", timed.mean,
               json_bool(timed.launch_ok));
  print_array(file, "warmup_ms", timed.warmup);
  std::fprintf(file, ",");
  print_array(file, "samples", timed.samples);
}

}  // namespace

int main(int argc, char** argv) {
  const char* raw_path =
      argc > 1 ? argv[1]
               : "evidence/optimization/opt045-parallel-norm/parallel-norm-ab-raw.txt";
  const char* model_path = argc > 2 ? argv[2] : nullptr;
  FILE* raw = std::fopen(raw_path, "w");
  if (raw == nullptr) {
    std::fprintf(stderr, "failed to open %s\n", raw_path);
    return 1;
  }

  cudaDeviceProp prop{};
  cudaError_t error = cudaGetDeviceProperties(&prop, 0);
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("cudaGetDeviceProperties", error);
  }
  std::time_t now = std::time(nullptr);
  char utc[32]{};
  std::tm tm{};
  gmtime_r(&now, &tm);
  std::strftime(utc, sizeof(utc), "%Y-%m-%dT%H:%M:%SZ", &tm);

  std::vector<float> input(kWidth);
  std::vector<float> scale(kWidth);
  std::vector<float> correction(kWidth);
  fill_unit(&input, 0.75F);
  for (std::size_t index = 0; index < kWidth; ++index) {
    scale[index] = 0.875F + static_cast<float>(index % 11) * 0.015625F;
    correction[index] =
        static_cast<float>(static_cast<int>(index % 17) - 8) * 0.0078125F;
  }
  std::vector<float> gdn_recurrent(kGdnHeads * kHeadWidth);
  std::vector<float> gdn_gate(kGdnHeads * kHeadWidth);
  std::vector<float> gdn_norm(kHeadWidth);
  fill_unit(&gdn_recurrent, 0.5F);
  fill_unit(&gdn_gate, 0.25F);
  for (std::size_t index = 0; index < kHeadWidth; ++index) {
    gdn_norm[index] = 0.9F + static_cast<float>(index % 5) * 0.02F;
  }

  float* device_input = nullptr;
  float* device_scale = nullptr;
  float* device_correction = nullptr;
  float* device_residual = nullptr;
  __nv_bfloat16* device_norm = nullptr;
  __nv_bfloat16* device_serial = nullptr;
  float* device_prompt_in = nullptr;
  float* device_prompt_corr = nullptr;
  float* device_prompt_out = nullptr;
  __nv_bfloat16* device_prompt_norm = nullptr;
  float* device_gdn_rec = nullptr;
  float* device_gdn_gate = nullptr;
  float* device_gdn_norm = nullptr;
  __nv_bfloat16* device_gdn_out = nullptr;
  __nv_bfloat16* device_bf16_in = nullptr;
  __nv_bfloat16* device_bf16_out = nullptr;
#define QW38_ALLOC(pointer, count)                                            \
  if (error == cudaSuccess)                                                   \
  error = cudaMalloc(&(pointer), (count) * sizeof(*(pointer)))
  error = cudaMalloc(&device_input, kWidth * sizeof(float));
  QW38_ALLOC(device_scale, kWidth);
  QW38_ALLOC(device_correction, kWidth);
  QW38_ALLOC(device_residual, kWidth);
  QW38_ALLOC(device_norm, kWidth);
  QW38_ALLOC(device_serial, kWidth);
  QW38_ALLOC(device_prompt_in, kPromptRows * kWidth);
  QW38_ALLOC(device_prompt_corr, kPromptRows * kWidth);
  QW38_ALLOC(device_prompt_out, kPromptRows * kWidth);
  QW38_ALLOC(device_prompt_norm, kPromptRows * kWidth);
  QW38_ALLOC(device_gdn_rec, kGdnHeads * kHeadWidth);
  QW38_ALLOC(device_gdn_gate, kGdnHeads * kHeadWidth);
  QW38_ALLOC(device_gdn_norm, kHeadWidth);
  QW38_ALLOC(device_gdn_out, kGdnHeads * kHeadWidth);
  QW38_ALLOC(device_bf16_in, kWidth);
  QW38_ALLOC(device_bf16_out, kWidth);
#undef QW38_ALLOC
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("ab cudaMalloc", error);
  }

  std::vector<float> prompt_in(kPromptRows * kWidth);
  std::vector<float> prompt_corr(kPromptRows * kWidth);
  for (std::size_t index = 0; index < prompt_in.size(); ++index) {
    prompt_in[index] = std::sin(static_cast<float>(index) * 0.0013F) * 0.5F;
    prompt_corr[index] =
        static_cast<float>(static_cast<int>(index % 19) - 9) * 0.01F;
  }
#define QW38_COPY(pointer, source)                                            \
  if (error == cudaSuccess)                                                   \
  error = cudaMemcpy((pointer), (source).data(),                              \
                     (source).size() * sizeof((source)[0]),                   \
                     cudaMemcpyHostToDevice)
  QW38_COPY(device_input, input);
  QW38_COPY(device_scale, scale);
  QW38_COPY(device_correction, correction);
  QW38_COPY(device_prompt_in, prompt_in);
  QW38_COPY(device_prompt_corr, prompt_corr);
  QW38_COPY(device_gdn_rec, gdn_recurrent);
  QW38_COPY(device_gdn_gate, gdn_gate);
  QW38_COPY(device_gdn_norm, gdn_norm);
#undef QW38_COPY
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_fp32_to_bf16(device_input, kWidth, device_bf16_in,
                                            nullptr);
  }
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("ab H2D", error);
  }

  {
    qw38::cuda::RmsNormPathScope serial("serial", 256, 256);
    error = qw38::cuda::launch_rms_norm_fp32_to_bf16(
        device_input, device_scale, kWidth, device_serial, nullptr);
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
  }
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("serial reference", error);
  }
  std::vector<__nv_bfloat16> serial_host(kWidth);
  error = cudaMemcpy(serial_host.data(), device_serial,
                     kWidth * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("serial D2H", error);
  }

  std::vector<__nv_bfloat16> fp64_ref;
  std::vector<__nv_bfloat16> llama_ref;
  host_fp64_norm(input, scale, &fp64_ref);
  host_llama_rsqrt(input, scale, 256, &llama_ref);
  const Envelope strict_vs_fp64 = compare_bf16(serial_host, fp64_ref);
  const Envelope llama_vs_fp64 = compare_bf16(llama_ref, fp64_ref);
  const float budget_abs = std::max(
      strict_vs_fp64.max_abs, 1.05F * llama_vs_fp64.max_abs + 1.0e-6F);
  const float budget_rms = std::max(
      strict_vs_fp64.rms, 1.05F * llama_vs_fp64.rms + 1.0e-6F);
  const float budget_cos = std::max(
      std::max(strict_vs_fp64.one_minus_cosine, 1.0e-7F),
      1.05F * llama_vs_fp64.one_minus_cosine + 1.0e-6F);

  struct CandResult {
    Timed decode_rms;
    Timed prompt_norm;
    Timed decode_residual;
    Timed gdn;
    Timed complete_ffn;
    Envelope vs_serial;
    Envelope vs_fp64;
    Envelope residual_fp32;
    Envelope gdn_vs_serial;
    Envelope zeros;
    Envelope extremes;
    Envelope opt043;
    Envelope primitives;
    int occupancy = 0;
    int registers = 0;
    std::size_t local_bytes = 0;
    bool eligible = false;
  };
  std::vector<CandResult> results(static_cast<std::size_t>(kCandidateCount));

  std::vector<__nv_bfloat16> serial_gdn(kGdnHeads * kHeadWidth);
  {
    qw38::cuda::RmsNormPathScope serial("serial", 256, 256);
    error = qw38::cuda::launch_gdn_gated_output(
        device_gdn_rec, device_gdn_gate, device_gdn_norm, 16, 3, kHeadWidth,
        device_gdn_out, nullptr);
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error == cudaSuccess) {
      error = cudaMemcpy(serial_gdn.data(), device_gdn_out,
                         serial_gdn.size() * sizeof(__nv_bfloat16),
                         cudaMemcpyDeviceToHost);
    }
  }
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("serial gdn", error);
  }

  for (int index = 0; index < kCandidateCount; ++index) {
    const Candidate& candidate = kCandidates[index];
    CandResult& result = results[static_cast<std::size_t>(index)];
    cudaFuncAttributes attrs{};
    if (qw38::cuda::rms_norm_kernel_attributes(
            candidate.path, candidate.residual_threads, &attrs) ==
        cudaSuccess) {
      result.registers = attrs.numRegs;
      result.local_bytes = attrs.localSizeBytes;
    }
    result.occupancy =
        qw38::cuda::rms_norm_parallel_occupancy(candidate.residual_threads);

    result.decode_rms = time_kernel(candidate, [&]() {
      return qw38::cuda::launch_rms_norm_fp32_to_bf16(
          device_input, device_scale, kWidth, device_norm, nullptr);
    });
    result.prompt_norm = time_kernel(candidate, [&]() {
      return qw38::cuda::launch_residual_add_norm_rows_fp32_to_bf16(
          device_prompt_in, device_prompt_corr, device_scale, kWidth,
          kPromptRows, device_prompt_out, device_prompt_norm, nullptr);
    });
    result.decode_residual = time_kernel(candidate, [&]() {
      return qw38::cuda::launch_residual_add_norm_fp32_to_bf16(
          device_input, device_correction, device_scale, kWidth,
          device_residual, device_norm, nullptr);
    });
    result.gdn = time_kernel(candidate, [&]() {
      return qw38::cuda::launch_gdn_gated_output(
          device_gdn_rec, device_gdn_gate, device_gdn_norm, 16, 3, kHeadWidth,
          device_gdn_out, nullptr);
    });

    {
      qw38::cuda::RmsNormPathScope scope(candidate.path,
                                         candidate.residual_threads,
                                         candidate.gdn_threads);
      error = qw38::cuda::launch_rms_norm_fp32_to_bf16(
          device_input, device_scale, kWidth, device_norm, nullptr);
      std::vector<__nv_bfloat16> actual(kWidth);
      if (error == cudaSuccess) {
        error = cudaMemcpy(actual.data(), device_norm,
                           kWidth * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost);
      }
      if (error == cudaSuccess) {
        result.vs_serial = compare_bf16(actual, serial_host);
        result.vs_fp64 = compare_bf16(actual, fp64_ref);
      }

      error = qw38::cuda::launch_residual_add_norm_fp32_to_bf16(
          device_input, device_correction, device_scale, kWidth,
          device_residual, device_norm, nullptr);
      std::vector<float> residual_actual(kWidth);
      std::vector<float> residual_expected(kWidth);
      if (error == cudaSuccess) {
        error = cudaMemcpy(residual_actual.data(), device_residual,
                           kWidth * sizeof(float), cudaMemcpyDeviceToHost);
      }
      {
        qw38::cuda::RmsNormPathScope serial("serial", 256, 256);
        cudaError_t serial_error = qw38::cuda::launch_residual_add_norm_fp32_to_bf16(
            device_input, device_correction, device_scale, kWidth,
            device_prompt_out, device_serial, nullptr);
        if (serial_error == cudaSuccess) {
          serial_error = cudaMemcpy(residual_expected.data(), device_prompt_out,
                                    kWidth * sizeof(float),
                                    cudaMemcpyDeviceToHost);
        }
        if (error == cudaSuccess) error = serial_error;
      }
      result.residual_fp32.residual_fp32_exact =
          error == cudaSuccess &&
          std::memcmp(residual_actual.data(), residual_expected.data(),
                      kWidth * sizeof(float)) == 0;
      result.residual_fp32.nonfinite = 0;

      error = qw38::cuda::launch_gdn_gated_output(
          device_gdn_rec, device_gdn_gate, device_gdn_norm, 16, 3, kHeadWidth,
          device_gdn_out, nullptr);
      std::vector<__nv_bfloat16> gdn_actual(kGdnHeads * kHeadWidth);
      if (error == cudaSuccess) {
        error = cudaMemcpy(gdn_actual.data(), device_gdn_out,
                           gdn_actual.size() * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost);
      }
      if (error == cudaSuccess) {
        result.gdn_vs_serial = compare_bf16(gdn_actual, serial_gdn);
      }

      std::vector<float> zeros(kWidth, 0.0F);
      std::vector<__nv_bfloat16> zero_ref;
      host_fp64_norm(zeros, scale, &zero_ref);
      error = cudaMemcpy(device_input, zeros.data(), kWidth * sizeof(float),
                         cudaMemcpyHostToDevice);
      if (error == cudaSuccess) {
        error = qw38::cuda::launch_rms_norm_fp32_to_bf16(
            device_input, device_scale, kWidth, device_norm, nullptr);
      }
      std::vector<__nv_bfloat16> zero_actual(kWidth);
      if (error == cudaSuccess) {
        error = cudaMemcpy(zero_actual.data(), device_norm,
                           kWidth * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost);
      }
      if (error == cudaSuccess) result.zeros = compare_bf16(zero_actual, zero_ref);

      std::vector<float> extremes(kWidth);
      for (std::size_t i = 0; i < kWidth; ++i) {
        const int kind = static_cast<int>(i % 4);
        extremes[i] = kind == 0   ? 1.0e-20F
                      : kind == 1 ? 1.0e4F
                      : kind == 2 ? -1.0e4F
                                  : 32.0F;
      }
      std::vector<__nv_bfloat16> extreme_ref;
      host_fp64_norm(extremes, scale, &extreme_ref);
      error = cudaMemcpy(device_input, extremes.data(), kWidth * sizeof(float),
                         cudaMemcpyHostToDevice);
      if (error == cudaSuccess) {
        error = qw38::cuda::launch_rms_norm_fp32_to_bf16(
            device_input, device_scale, kWidth, device_norm, nullptr);
      }
      std::vector<__nv_bfloat16> extreme_actual(kWidth);
      if (error == cudaSuccess) {
        error = cudaMemcpy(extreme_actual.data(), device_norm,
                           kWidth * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost);
      }
      if (error == cudaSuccess) {
        result.extremes = compare_bf16(extreme_actual, extreme_ref);
      }

      std::vector<float> opt043(kWidth);
      fill_opt043(&opt043);
      std::vector<__nv_bfloat16> opt043_ref;
      host_fp64_norm(opt043, scale, &opt043_ref);
      error = cudaMemcpy(device_input, opt043.data(), kWidth * sizeof(float),
                         cudaMemcpyHostToDevice);
      if (error == cudaSuccess) {
        error = qw38::cuda::launch_rms_norm_fp32_to_bf16(
            device_input, device_scale, kWidth, device_norm, nullptr);
      }
      std::vector<__nv_bfloat16> opt043_actual(kWidth);
      if (error == cudaSuccess) {
        error = cudaMemcpy(opt043_actual.data(), device_norm,
                           kWidth * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost);
      }
      if (error == cudaSuccess) {
        result.opt043 = compare_bf16(opt043_actual, opt043_ref);
      }

      error = cudaMemcpy(device_input, input.data(), kWidth * sizeof(float),
                         cudaMemcpyHostToDevice);
      error = qw38::cuda::launch_rms_norm_bf16(
          device_bf16_in, device_scale, kWidth, device_bf16_out, nullptr);
      std::vector<__nv_bfloat16> prim_actual(kWidth);
      std::vector<__nv_bfloat16> prim_serial(kWidth);
      if (error == cudaSuccess) {
        error = cudaMemcpy(prim_actual.data(), device_bf16_out,
                           kWidth * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost);
      }
      {
        qw38::cuda::RmsNormPathScope serial("serial", 256, 256);
        cudaError_t serial_error = qw38::cuda::launch_rms_norm_bf16(
            device_bf16_in, device_scale, kWidth, device_serial, nullptr);
        if (serial_error == cudaSuccess) {
          serial_error = cudaMemcpy(prim_serial.data(), device_serial,
                                    kWidth * sizeof(__nv_bfloat16),
                                    cudaMemcpyDeviceToHost);
        }
        if (error == cudaSuccess) error = serial_error;
      }
      if (error == cudaSuccess) {
        result.primitives = compare_bf16(prim_actual, prim_serial);
      }
    }

    result.eligible = false;
  }

  const CandResult& strict = results[0];
  for (int index = 0; index < kCandidateCount; ++index) {
    CandResult& result = results[static_cast<std::size_t>(index)];
    const float unit_abs =
        std::max(strict.vs_fp64.max_abs, 1.05F * llama_vs_fp64.max_abs + 1.0e-6F);
    const float unit_rms =
        std::max(strict.vs_fp64.rms, 1.05F * llama_vs_fp64.rms + 1.0e-6F);
    const float unit_cos = std::max(
        std::max(strict.vs_fp64.one_minus_cosine, 1.0e-7F),
        1.05F * llama_vs_fp64.one_minus_cosine + 1.0e-6F);
    const float opt043_abs = std::max(strict.opt043.max_abs, 1.0e-6F);
    const float opt043_rms = std::max(strict.opt043.rms, 1.0e-6F);
    const float zero_abs = std::max(strict.zeros.max_abs, 1.0e-6F);
    result.eligible =
        result.decode_rms.launch_ok && result.prompt_norm.launch_ok &&
        result.decode_residual.launch_ok && result.gdn.launch_ok &&
        result.vs_fp64.nonfinite == 0 && result.zeros.nonfinite == 0 &&
        result.extremes.nonfinite == 0 && result.opt043.nonfinite == 0 &&
        result.gdn_vs_serial.nonfinite == 0 &&
        result.residual_fp32.residual_fp32_exact &&
        result.vs_fp64.max_abs <= unit_abs && result.vs_fp64.rms <= unit_rms &&
        result.vs_fp64.one_minus_cosine <= unit_cos &&
        result.opt043.max_abs <= opt043_abs && result.opt043.rms <= opt043_rms &&
        result.zeros.max_abs <= zero_abs && result.primitives.nonfinite == 0;
  }

  int winner = 0;
  bool win = false;
  for (int index = 1; index < kCandidateCount; ++index) {
    const CandResult& cand = results[static_cast<std::size_t>(index)];
    const CandResult& serial = results[0];
    if (!cand.eligible || !serial.eligible) continue;
    if (!(cand.prompt_norm.mean < serial.prompt_norm.mean)) continue;
    if (!win) {
      winner = index;
      win = true;
      continue;
    }
    const Candidate& current = kCandidates[winner];
    const Candidate& next = kCandidates[index];
    const float current_mean =
        results[static_cast<std::size_t>(winner)].prompt_norm.mean;
    if (std::strcmp(next.path, "parallel_rsqrt") == 0 &&
        std::strcmp(current.path, "parallel_fma") == 0 &&
        next.residual_threads == current.residual_threads) {
      if (cand.prompt_norm.mean < 0.98F * current_mean) winner = index;
      continue;
    }
    if (cand.prompt_norm.mean < current_mean) winner = index;
  }
  if (winner == 0) win = false;

  bool complete_ffn_ran = false;
  if (model_path != nullptr) {
    qw38::internal::ModelInfo info;
    qw38::Status status = qw38::internal::inspect_gguf(model_path, &info);
    if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
    qw38::internal::MappedFile mapping;
    if (status.is_ok()) status = mapping.open(model_path);
    qw38::internal::ModelWeights weights;
    if (status.is_ok()) {
      status = qw38::internal::bind_model_weights(info, mapping, &weights);
    }
    qw38::cuda::ResidentModel model;
    if (status.is_ok()) {
      status = model.upload(weights, mapping.data(), mapping.size());
    }
    qw38::cuda::SchedulerWorkspace workspace;
    if (status.is_ok()) status = workspace.create(kPromptRows);
    if (status.is_ok()) {
      complete_ffn_ran = true;
      std::vector<float> hidden(kPromptRows * kWidth, 0.0F);
      fill_unit(&hidden, 0.4F);
      error = cudaMemcpy(workspace.prompt_residual_a_, hidden.data(),
                         hidden.size() * sizeof(float), cudaMemcpyHostToDevice);
      if (error == cudaSuccess) {
        error = cudaMemset(workspace.prompt_mixer_output_, 0,
                           hidden.size() * sizeof(float));
      }
      if (error == cudaSuccess) {
        for (int index = 0; index < kCandidateCount; ++index) {
          const Candidate& candidate = kCandidates[index];
          results[static_cast<std::size_t>(index)].complete_ffn = time_kernel(
              candidate, [&]() {
                return qw38::cuda::execute_prompt_ffn(
                    model.layer(0).common, workspace.prompt_residual_a_,
                    &workspace, workspace.prompt_residual_b_,
                    workspace.prompt_residual_a_,
                    model.layer(1).common.input_norm, kPromptRows, nullptr);
              });
        }
      }
    }
  }

  std::fprintf(stdout, "%s", kPrefix);
  std::fprintf(stdout,
               "{\"schema_version\":1,\"task\":\"OPT-045\","
               "\"measurement_utc\":\"%s\",\"device\":\"%s\","
               "\"compute_capability\":\"%d.%d\",\"complete_ffn_ran\":%s,"
               "\"winner\":\"%s\",\"win\":%s,\"selected_rms_norm_path\":\"%s\","
               "\"selected_rms_norm_threads\":%d,\"selected_gdn_norm_threads\":%d,",
               utc, prop.name, prop.major, prop.minor,
               json_bool(complete_ffn_ran), kCandidates[winner].id,
               json_bool(win),
               win ? kCandidates[winner].path : "serial",
               win ? kCandidates[winner].residual_threads : 256,
               win ? kCandidates[winner].gdn_threads : 256);
  std::fprintf(stdout,
               "\"production_numerics\":{\"formula\":\"max(strict_reference_"
               "ceiling, 1.05 * measured_llama_error + 1e-6)\","
               "\"strict_vs_fp64\":{\"max_abs\":%.9g,\"rms\":%.9g,"
               "\"one_minus_cosine\":%.9g,\"nonfinite\":%zu},"
               "\"llama_vs_fp64\":{\"max_abs\":%.9g,\"rms\":%.9g,"
               "\"one_minus_cosine\":%.9g,\"nonfinite\":%zu},"
               "\"budget\":{\"max_abs\":%.9g,\"rms\":%.9g,"
               "\"one_minus_cosine\":%.9g}},",
               strict_vs_fp64.max_abs, strict_vs_fp64.rms,
               strict_vs_fp64.one_minus_cosine, strict_vs_fp64.nonfinite,
               llama_vs_fp64.max_abs, llama_vs_fp64.rms,
               llama_vs_fp64.one_minus_cosine, llama_vs_fp64.nonfinite,
               budget_abs, budget_rms, budget_cos);
  std::fprintf(stdout, "\"candidates\":{");
  for (int index = 0; index < kCandidateCount; ++index) {
    const Candidate& candidate = kCandidates[index];
    const CandResult& result = results[static_cast<std::size_t>(index)];
    if (index != 0) std::fprintf(stdout, ",");
    std::fprintf(stdout,
                 "\"%s\":{\"id\":\"%s\",\"path\":\"%s\",\"residual_threads\":%d,"
                 "\"gdn_threads\":%d,\"occupancy\":%d,\"registers\":%d,"
                 "\"local_bytes\":%zu,\"eligible\":%s,\"removed_serial_"
                 "dependency\":%s,",
                 candidate.id, candidate.id, candidate.path,
                 candidate.residual_threads, candidate.gdn_threads,
                 result.occupancy, result.registers, result.local_bytes,
                 json_bool(result.eligible),
                 json_bool(std::strcmp(candidate.path, "serial") != 0));
    std::fprintf(stdout, "\"decode_rms\":{");
    emit_timed(stdout, result.decode_rms);
    std::fprintf(stdout, "},\"prompt_norm\":{");
    emit_timed(stdout, result.prompt_norm);
    std::fprintf(stdout, "},\"decode_residual\":{");
    emit_timed(stdout, result.decode_residual);
    std::fprintf(stdout, "},\"gdn\":{");
    emit_timed(stdout, result.gdn);
    std::fprintf(stdout, "},\"complete_ffn\":{");
    emit_timed(stdout, result.complete_ffn);
    std::fprintf(stdout,
                 "},\"vs_serial\":{\"max_abs\":%.9g,\"rms\":%.9g,"
                 "\"one_minus_cosine\":%.9g,\"nonfinite\":%zu},"
                 "\"vs_fp64\":{\"max_abs\":%.9g,\"rms\":%.9g,"
                 "\"one_minus_cosine\":%.9g,\"nonfinite\":%zu},"
                 "\"residual_fp32_exact\":%s,"
                 "\"gdn_vs_serial\":{\"max_abs\":%.9g,\"rms\":%.9g,"
                 "\"nonfinite\":%zu},"
                 "\"zeros\":{\"max_abs\":%.9g,\"rms\":%.9g,\"nonfinite\":%zu},"
                 "\"extremes\":{\"max_abs\":%.9g,\"rms\":%.9g,\"nonfinite\":%zu},"
                 "\"opt043\":{\"max_abs\":%.9g,\"rms\":%.9g,\"nonfinite\":%zu},"
                 "\"primitives\":{\"max_abs\":%.9g,\"rms\":%.9g,"
                 "\"nonfinite\":%zu}}",
                 result.vs_serial.max_abs, result.vs_serial.rms,
                 result.vs_serial.one_minus_cosine, result.vs_serial.nonfinite,
                 result.vs_fp64.max_abs, result.vs_fp64.rms,
                 result.vs_fp64.one_minus_cosine, result.vs_fp64.nonfinite,
                 json_bool(result.residual_fp32.residual_fp32_exact),
                 result.gdn_vs_serial.max_abs, result.gdn_vs_serial.rms,
                 result.gdn_vs_serial.nonfinite, result.zeros.max_abs,
                 result.zeros.rms, result.zeros.nonfinite,
                 result.extremes.max_abs, result.extremes.rms,
                 result.extremes.nonfinite, result.opt043.max_abs,
                 result.opt043.rms, result.opt043.nonfinite,
                 result.primitives.max_abs, result.primitives.rms,
                 result.primitives.nonfinite);
  }
  std::fprintf(stdout, "}}\n");
  std::fprintf(stdout, "status=passed\n");
  std::fprintf(raw, "OPT-045 A/B utc=%s winner=%s win=%s\n", utc,
               kCandidates[winner].id, json_bool(win));
  std::fclose(raw);
  return 0;
}
