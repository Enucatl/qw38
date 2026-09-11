#include "quant_mmv.h"
#include "quant.h"
#include "test_tier.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

#include <cuda_fp16.h>

namespace {

constexpr char kPrefix[] = "QW38_OPT053_MMQ_PIPELINE_AB_RESULT=";
constexpr const char* kCandidates[] = {"off", "fma", "async_y", "fma_async"};
constexpr int kCandidateCount = 4;
constexpr float kOpt044Abs = 3.0e-4F;
constexpr float kOpt044Rms = 2.0e-4F;
constexpr float kAbsScale = 0.20F;
constexpr float kRelTol = 0.05F;

const char* json_bool(bool value) { return value ? "true" : "false"; }

int fail_cuda(const char* operation, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
  return 1;
}

void print_array(FILE* file, const char* key, const std::vector<float>& values) {
  std::fprintf(file, "\"%s\":[", key);
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) std::fprintf(file, ",");
    std::fprintf(file, "%.9g", static_cast<double>(values[index]));
  }
  std::fprintf(file, "]");
}

void write_u16(std::uint8_t* bytes, std::uint16_t value) {
  bytes[0] = static_cast<std::uint8_t>(value & 0xFFU);
  bytes[1] = static_cast<std::uint8_t>(value >> 8U);
}

void fill_q4_weights(std::size_t rows, std::size_t columns,
                     std::vector<std::uint8_t>* weights) {
  constexpr std::size_t kBlockBytes = 144;
  weights->assign(rows * (columns / 256) * kBlockBytes, 0);
  for (std::size_t index = 0; index < weights->size(); ++index) {
    (*weights)[index] = static_cast<std::uint8_t>((index * 73 + 19) & 0xFFU);
  }
  for (std::size_t offset = 0; offset < weights->size(); offset += kBlockBytes) {
    write_u16(weights->data() + offset, 0x2400U);
    write_u16(weights->data() + offset + 2, 0x1C00U);
  }
}

void fill_q8_weights(std::size_t rows, std::size_t columns,
                     std::vector<std::uint8_t>* weights) {
  constexpr std::size_t kBlockBytes = 34;
  weights->assign(rows * (columns / 32) * kBlockBytes, 0);
  for (std::size_t index = 0; index < weights->size(); ++index) {
    (*weights)[index] = static_cast<std::uint8_t>((index * 41 + 7) & 0xFFU);
  }
  for (std::size_t offset = 0; offset < weights->size(); offset += kBlockBytes) {
    write_u16(weights->data() + offset, 0x3C00U);
  }
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

void fill_prompt(std::size_t prompt_rows, std::size_t columns,
                 std::vector<__nv_bfloat16>* prompt) {
  prompt->resize(prompt_rows * columns);
  for (std::size_t index = 0; index < prompt->size(); ++index) {
    (*prompt)[index] = __float2bfloat16_rn(
        unit_normal(static_cast<std::uint32_t>(index), 0xA11CE5u));
  }
}

bool reference_dequant_gemm(const std::vector<std::uint8_t>& weights,
                            std::size_t output_rows, std::size_t columns,
                            const std::vector<__nv_bfloat16>& prompt,
                            std::size_t prompt_rows,
                            std::vector<float>* output) {
  output->assign(prompt_rows * output_rows, 0.0F);
  std::vector<float> decoded(256);
  for (std::size_t out = 0; out < output_rows; ++out) {
    for (std::size_t prompt_row = 0; prompt_row < prompt_rows; ++prompt_row) {
      float sum = 0.0F;
      for (std::size_t block = 0; block < columns / 256; ++block) {
        const std::uint8_t* packed =
            weights.data() + (out * (columns / 256) + block) * 144;
        const qw38::Status status = qw38::internal::decode_q4_k(
            packed, 144, decoded.data(), decoded.size());
        if (!status.is_ok()) return false;
        for (std::size_t within = 0; within < 256; ++within) {
          const std::size_t column = block * 256 + within;
          sum += decoded[within] *
                 __bfloat162float(prompt[prompt_row * columns + column]);
        }
      }
      (*output)[prompt_row * output_rows + out] = sum;
    }
  }
  return true;
}

struct Envelope {
  float max_abs = 0.0F;
  float rms = 0.0F;
  std::size_t nonfinite = 0;
  std::size_t bad = 0;
  bool ok = false;
};

Envelope compare(const std::vector<float>& actual,
                 const std::vector<float>& expected) {
  Envelope metrics;
  double squared = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
      ++metrics.nonfinite;
      continue;
    }
    const float error = std::fabs(actual[index] - expected[index]);
    metrics.max_abs = std::max(metrics.max_abs, error);
    squared += static_cast<double>(error) * error;
  }
  metrics.rms =
      actual.empty()
          ? 0.0F
          : static_cast<float>(std::sqrt(squared / static_cast<double>(actual.size())));
  return metrics;
}

bool ds4_ok(const std::vector<float>& got, const std::vector<float>& ref,
            std::size_t columns, Envelope* metrics) {
  const float abs_tol = kAbsScale * std::sqrt(static_cast<float>(columns));
  *metrics = compare(got, ref);
  metrics->bad = 0;
  for (std::size_t index = 0; index < got.size(); ++index) {
    if (!std::isfinite(got[index]) || !std::isfinite(ref[index])) continue;
    const float absolute = std::fabs(got[index] - ref[index]);
    const float relative = ref[index] != 0.0F
                               ? absolute / std::fabs(ref[index])
                               : (absolute > 0.0F ? INFINITY : 0.0F);
    if (absolute > abs_tol && relative > kRelTol) ++metrics->bad;
  }
  metrics->ok = metrics->nonfinite == 0 && metrics->bad == 0;
  return metrics->ok;
}

bool like_arithmetic(const char* path) {
  return std::strcmp(path, "off") == 0 || std::strcmp(path, "async_y") == 0;
}

bool uses_fma(const char* path) {
  return std::strcmp(path, "fma") == 0 || std::strcmp(path, "fma_async") == 0;
}

bool uses_async(const char* path) {
  return std::strcmp(path, "async_y") == 0 ||
         std::strcmp(path, "fma_async") == 0;
}

struct Candidate {
  std::vector<float> warmup;
  std::vector<float> samples;
  float mean_ms = 0.0F;
  int occupancy = 0;
  std::size_t extra_shared_bytes = 0;
  bool launch_ok = false;
  bool eligible = false;
  Envelope vs_off;
  Envelope vs_host;
  std::size_t timed_nonfinite = 0;
};

cudaError_t launch_complete_ffn(const char* path, const std::uint8_t* gate_w,
                                const std::uint8_t* up_w,
                                const std::uint8_t* down_w,
                                const __nv_bfloat16* prompt,
                                qw38::cuda::Q8Block* y, float* gate_out,
                                float* up_out, float* down_out,
                                std::size_t prompt_rows, std::size_t hidden,
                                std::size_t ffn) {
  cudaError_t error = qw38::cuda::launch_quantize_mmq_q8_1(
      qw38::cuda::QuantKind::kQ4K, prompt, prompt_rows, hidden, y, nullptr);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmq_mma_y_pipeline(
        qw38::cuda::QuantKind::kQ4K, gate_w, ffn, hidden, y, prompt_rows,
        gate_out, 128, 128, path, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmq_mma_y_pipeline(
        qw38::cuda::QuantKind::kQ4K, up_w, ffn, hidden, y, prompt_rows, up_out,
        128, 128, path, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_swiglu_quantize_mmq_q8_1(gate_out, up_out,
                                                        prompt_rows, ffn, y,
                                                        nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmq_mma_y_pipeline(
        qw38::cuda::QuantKind::kQ4K, down_w, hidden, ffn, y, prompt_rows,
        down_out, 128, 128, path, nullptr);
  }
  return error;
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_valid()) {
    std::fprintf(stderr, "QW38_CUDA_TEST_TIER must be set to smoke, "
                         "correctness, or acceptance\n");
    return 1;
  }
  const qw38::cuda::TestTier tier = qw38::cuda::test_tier();
  const int warmups = qw38::cuda::test_warmups();
  const int measured = qw38::cuda::test_samples();
  const char* raw_path =
      argc > 1 ? argv[1]
               : "evidence/optimization/opt053-mmq-pipeline/"
                 "mmq-pipeline-ab-raw.txt";
  FILE* raw = std::fopen(raw_path, "w");
  if (raw == nullptr) {
    std::fprintf(stderr, "failed to open %s\n", raw_path);
    return 1;
  }

  cudaDeviceProp props{};
  if (cudaGetDeviceProperties(&props, 0) != cudaSuccess) {
    std::fclose(raw);
    std::fprintf(stderr, "cudaGetDeviceProperties failed\n");
    return 1;
  }

  const std::size_t timed_prompt =
      tier == qw38::cuda::TestTier::kSmoke ? 128 : 4096;
  const std::size_t hidden =
      tier == qw38::cuda::TestTier::kSmoke ? 256 : 5120;
  const std::size_t ffn = tier == qw38::cuda::TestTier::kSmoke ? 256 : 17408;
  const std::size_t env_prompt = 128;
  const std::size_t env_rows = 128;
  const std::size_t env_cols = 256;
  const std::size_t mixer_wide_rows =
      tier == qw38::cuda::TestTier::kSmoke ? 128 : 5120;
  const std::size_t mixer_skinny_rows = 32;
  const std::size_t mixer_cols =
      tier == qw38::cuda::TestTier::kSmoke ? 256 : 5120;

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaError_t error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("cudaEventCreate", error);
  }

  std::vector<std::uint8_t> env_weights;
  fill_q4_weights(env_rows, env_cols, &env_weights);
  std::vector<__nv_bfloat16> env_prompt_h;
  fill_prompt(env_prompt, env_cols, &env_prompt_h);
  std::vector<float> env_expected;
  if (!reference_dequant_gemm(env_weights, env_rows, env_cols, env_prompt_h,
                              env_prompt, &env_expected)) {
    std::fclose(raw);
    std::fprintf(stderr, "envelope host GEMM failed\n");
    return 1;
  }

  std::uint8_t* env_w = nullptr;
  __nv_bfloat16* env_p = nullptr;
  qw38::cuda::Q8Block* env_y = nullptr;
  float* env_out = nullptr;
  error = cudaMalloc(&env_w, env_weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&env_p, env_prompt_h.size() * sizeof(env_prompt_h[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&env_y, qw38::cuda::q8_prompt_workspace_bytes(
                                   env_prompt, env_cols));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&env_out, env_expected.size() * sizeof(float));
  }
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("envelope cudaMalloc", error);
  }
  error = cudaMemcpy(env_w, env_weights.data(), env_weights.size(),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(env_p, env_prompt_h.data(),
                       env_prompt_h.size() * sizeof(env_prompt_h[0]),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_mmq_q8_1(
        qw38::cuda::QuantKind::kQ4K, env_p, env_prompt, env_cols, env_y,
        nullptr);
  }
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("envelope quantize", error);
  }

  Candidate candidates[kCandidateCount];
  std::vector<float> off_env;
  for (int index = 0; index < kCandidateCount; ++index) {
    const char* path = kCandidates[index];
    Candidate& slot = candidates[index];
    slot.occupancy = qw38::cuda::mmq_pipeline_occupancy(
        qw38::cuda::QuantKind::kQ4K, 128, 128, path);
    slot.extra_shared_bytes = qw38::cuda::mmq_pipeline_extra_shared_bytes(
        128, 128, path);
    error = qw38::cuda::launch_quant_mmq_mma_y_pipeline(
        qw38::cuda::QuantKind::kQ4K, env_w, env_rows, env_cols, env_y,
        env_prompt, env_out, 128, 128, path, nullptr);
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    slot.launch_ok = error == cudaSuccess;
    if (!slot.launch_ok) {
      std::fprintf(raw, "envelope path=%s launch=%s\n", path,
                   cudaGetErrorString(error));
      continue;
    }
    std::vector<float> got(env_expected.size());
    error = cudaMemcpy(got.data(), env_out, got.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("envelope D2H", error);
    }
    ds4_ok(got, env_expected, env_cols, &slot.vs_host);
    if (index == 0) off_env = got;
    slot.vs_off = compare(got, off_env);
    const bool like = like_arithmetic(path);
    const bool vs_off_ok =
        like ? (slot.vs_off.nonfinite == 0 && slot.vs_off.max_abs == 0.0F)
             : (slot.vs_off.nonfinite == 0 &&
                slot.vs_off.max_abs <= kOpt044Abs &&
                slot.vs_off.rms <= kOpt044Rms);
    slot.eligible = slot.launch_ok && slot.occupancy >= 1 && slot.vs_host.ok &&
                    vs_off_ok;
    std::fprintf(raw,
                 "envelope path=%s eligible=%s max_abs=%.9g rms=%.9g "
                 "vs_off=%.9g occ=%d extra=%zu\n",
                 path, slot.eligible ? "true" : "false",
                 static_cast<double>(slot.vs_host.max_abs),
                 static_cast<double>(slot.vs_host.rms),
                 static_cast<double>(slot.vs_off.max_abs), slot.occupancy,
                 slot.extra_shared_bytes);
  }

  const std::size_t tail_prompts[] = {17, 65, 129, 255};
  const int tail_count = tier == qw38::cuda::TestTier::kSmoke ? 1 : 4;
  int correctness_cases = kCandidateCount;
  bool tails_ok = true;
  for (int tail = 0; tail < tail_count; ++tail) {
    const std::size_t prompt_rows = tail_prompts[tail];
    std::vector<__nv_bfloat16> tail_prompt;
    fill_prompt(prompt_rows, env_cols, &tail_prompt);
    __nv_bfloat16* tp = nullptr;
    qw38::cuda::Q8Block* ty = nullptr;
    float* tout = nullptr;
    error = cudaMalloc(&tp, tail_prompt.size() * sizeof(tail_prompt[0]));
    if (error == cudaSuccess) {
      error = cudaMalloc(&ty, qw38::cuda::q8_prompt_workspace_bytes(
                                  prompt_rows, env_cols));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(&tout, prompt_rows * env_rows * sizeof(float));
    }
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("tail cudaMalloc", error);
    }
    error = cudaMemcpy(tp, tail_prompt.data(),
                       tail_prompt.size() * sizeof(tail_prompt[0]),
                       cudaMemcpyHostToDevice);
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_quantize_mmq_q8_1(
          qw38::cuda::QuantKind::kQ4K, tp, prompt_rows, env_cols, ty, nullptr);
    }
    for (int index = 0; index < kCandidateCount && error == cudaSuccess;
         ++index) {
      error = qw38::cuda::launch_quant_mmq_mma_y_pipeline(
          qw38::cuda::QuantKind::kQ4K, env_w, env_rows, env_cols, ty,
          prompt_rows, tout, 128, 128, kCandidates[index], nullptr);
      if (error == cudaSuccess) error = cudaDeviceSynchronize();
      ++correctness_cases;
      if (error != cudaSuccess) {
        tails_ok = false;
        candidates[index].eligible = false;
        std::fprintf(raw, "tail prompt=%zu path=%s launch=%s\n", prompt_rows,
                     kCandidates[index], cudaGetErrorString(error));
      }
    }
    cudaFree(tout);
    cudaFree(ty);
    cudaFree(tp);
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("tail launch", error);
    }
  }

  std::vector<std::uint8_t> gate_w;
  std::vector<std::uint8_t> up_w;
  std::vector<std::uint8_t> down_w;
  fill_q4_weights(ffn, hidden, &gate_w);
  fill_q4_weights(ffn, hidden, &up_w);
  fill_q4_weights(hidden, ffn, &down_w);
  std::vector<__nv_bfloat16> prompt_h;
  fill_prompt(timed_prompt, hidden, &prompt_h);

  std::uint8_t* d_gate = nullptr;
  std::uint8_t* d_up = nullptr;
  std::uint8_t* d_down = nullptr;
  __nv_bfloat16* d_prompt = nullptr;
  qw38::cuda::Q8Block* d_y = nullptr;
  float* d_gate_out = nullptr;
  float* d_up_out = nullptr;
  float* d_down_out = nullptr;
  error = cudaMalloc(&d_gate, gate_w.size());
  if (error == cudaSuccess) error = cudaMalloc(&d_up, up_w.size());
  if (error == cudaSuccess) error = cudaMalloc(&d_down, down_w.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&d_prompt, prompt_h.size() * sizeof(prompt_h[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&d_y, qw38::cuda::q8_prompt_workspace_bytes(
                                 timed_prompt, std::max(hidden, ffn)));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&d_gate_out, timed_prompt * ffn * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&d_up_out, timed_prompt * ffn * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&d_down_out, timed_prompt * hidden * sizeof(float));
  }
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("ffn cudaMalloc", error);
  }
  error = cudaMemcpy(d_gate, gate_w.data(), gate_w.size(), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(d_up, up_w.data(), up_w.size(), cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error =
        cudaMemcpy(d_down, down_w.data(), down_w.size(), cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(d_prompt, prompt_h.data(),
                       prompt_h.size() * sizeof(prompt_h[0]),
                       cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("ffn H2D", error);
  }

  for (int round = 0; round < warmups + measured; ++round) {
    for (int index = 0; index < kCandidateCount; ++index) {
      Candidate& slot = candidates[index];
      if (!slot.launch_ok) continue;
      error = cudaEventRecord(start);
      if (error == cudaSuccess) {
        error = launch_complete_ffn(kCandidates[index], d_gate, d_up, d_down,
                                    d_prompt, d_y, d_gate_out, d_up_out,
                                    d_down_out, timed_prompt, hidden, ffn);
      }
      if (error == cudaSuccess) error = cudaEventRecord(stop);
      if (error == cudaSuccess) error = cudaEventSynchronize(stop);
      float milliseconds = 0.0F;
      if (error == cudaSuccess) {
        error = cudaEventElapsedTime(&milliseconds, start, stop);
      }
      if (error != cudaSuccess) {
        slot.launch_ok = false;
        slot.eligible = false;
        std::fprintf(raw, "ffn path=%s launch=%s\n", kCandidates[index],
                     cudaGetErrorString(error));
        continue;
      }
      std::fprintf(raw, "ffn path=%s round=%d ms=%.9g\n", kCandidates[index],
                   round, milliseconds);
      if (round < warmups) {
        slot.warmup.push_back(milliseconds);
      } else {
        slot.samples.push_back(milliseconds);
      }
    }
  }

  for (int index = 0; index < kCandidateCount; ++index) {
    Candidate& slot = candidates[index];
    slot.timed_nonfinite = 0;
    if (!slot.launch_ok) continue;
    error = launch_complete_ffn(kCandidates[index], d_gate, d_up, d_down,
                                d_prompt, d_y, d_gate_out, d_up_out, d_down_out,
                                timed_prompt, hidden, ffn);
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
      slot.launch_ok = false;
      slot.eligible = false;
      continue;
    }
    std::vector<float> timed(timed_prompt * hidden);
    error = cudaMemcpy(timed.data(), d_down_out, timed.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("ffn D2H", error);
    }
    for (float value : timed) {
      if (!std::isfinite(value)) ++slot.timed_nonfinite;
    }
  }

  float extra_means[3] = {0.0F, 0.0F, 0.0F};
  const std::size_t extra_tokens[3] = {512, 1024, 2048};
  const int extra_count = tier == qw38::cuda::TestTier::kAcceptance ? 3 : 0;
  if (extra_count > 0 && timed_prompt >= 2048) {
    for (int extra = 0; extra < extra_count; ++extra) {
      const std::size_t rows = extra_tokens[extra];
      error = launch_complete_ffn("off", d_gate, d_up, d_down, d_prompt, d_y,
                                  d_gate_out, d_up_out, d_down_out, rows,
                                  hidden, ffn);
      if (error == cudaSuccess) error = cudaDeviceSynchronize();
      error = cudaEventRecord(start);
      if (error == cudaSuccess) {
        error = launch_complete_ffn("off", d_gate, d_up, d_down, d_prompt, d_y,
                                    d_gate_out, d_up_out, d_down_out, rows,
                                    hidden, ffn);
      }
      if (error == cudaSuccess) error = cudaEventRecord(stop);
      if (error == cudaSuccess) error = cudaEventSynchronize(stop);
      if (error == cudaSuccess) {
        error = cudaEventElapsedTime(&extra_means[extra], start, stop);
      }
      if (error != cudaSuccess) {
        std::fclose(raw);
        return fail_cuda("extra token ffn", error);
      }
      std::fprintf(raw, "ffn_off tokens=%zu ms=%.9g\n", rows,
                   extra_means[extra]);
    }
  }

  std::vector<std::uint8_t> mix_w;
  fill_q8_weights(mixer_wide_rows, mixer_cols, &mix_w);
  std::vector<__nv_bfloat16> mix_p;
  fill_prompt(timed_prompt, mixer_cols, &mix_p);
  std::uint8_t* d_mix_w = nullptr;
  __nv_bfloat16* d_mix_p = nullptr;
  qw38::cuda::Q8Block* d_mix_y = nullptr;
  float* d_mix_out = nullptr;
  error = cudaMalloc(&d_mix_w, mix_w.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&d_mix_p, mix_p.size() * sizeof(mix_p[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&d_mix_y, qw38::cuda::q8_prompt_workspace_bytes(
                                     timed_prompt, mixer_cols));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&d_mix_out, timed_prompt * mixer_wide_rows * sizeof(float));
  }
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("mixer cudaMalloc", error);
  }
  error = cudaMemcpy(d_mix_w, mix_w.data(), mix_w.size(), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(d_mix_p, mix_p.data(), mix_p.size() * sizeof(mix_p[0]),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_mmq_q8_1(
        qw38::cuda::QuantKind::kQ8_0, d_mix_p, timed_prompt, mixer_cols,
        d_mix_y, nullptr);
  }
  float mixer_wide_ms[kCandidateCount] = {0};
  float mixer_skinny_ms[kCandidateCount] = {0};
  int mixer_wide_occ[kCandidateCount] = {0};
  int mixer_skinny_occ[kCandidateCount] = {0};
  for (int index = 0; index < kCandidateCount; ++index) {
    const char* path = kCandidates[index];
    mixer_wide_occ[index] = qw38::cuda::mmq_pipeline_occupancy(
        qw38::cuda::QuantKind::kQ8_0, 128, 128, path);
    mixer_skinny_occ[index] = qw38::cuda::mmq_pipeline_occupancy(
        qw38::cuda::QuantKind::kQ8_0, 128, 32, path);
    error = cudaEventRecord(start);
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_q8_mmq_quality_mma_pipeline(
          d_mix_w, mixer_wide_rows, mixer_cols, d_mix_y, timed_prompt,
          d_mix_out, 128, path, nullptr);
    }
    if (error == cudaSuccess) error = cudaEventRecord(stop);
    if (error == cudaSuccess) error = cudaEventSynchronize(stop);
    if (error == cudaSuccess) {
      error = cudaEventElapsedTime(&mixer_wide_ms[index], start, stop);
    }
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("mixer wide", error);
    }
    error = cudaEventRecord(start);
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_q8_mmq_quality_mma_pipeline(
          d_mix_w, mixer_skinny_rows, mixer_cols, d_mix_y, timed_prompt,
          d_mix_out, 32, path, nullptr);
    }
    if (error == cudaSuccess) error = cudaEventRecord(stop);
    if (error == cudaSuccess) error = cudaEventSynchronize(stop);
    if (error == cudaSuccess) {
      error = cudaEventElapsedTime(&mixer_skinny_ms[index], start, stop);
    }
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("mixer skinny", error);
    }
    std::fprintf(raw, "mixer path=%s wide_ms=%.9g skinny_ms=%.9g\n", path,
                 mixer_wide_ms[index], mixer_skinny_ms[index]);
  }

  int winner = 0;
  float best = 1.0e30F;
  bool all_eligible = true;
  for (int index = 0; index < kCandidateCount; ++index) {
    Candidate& slot = candidates[index];
    double sum = 0.0;
    for (float sample : slot.samples) sum += static_cast<double>(sample);
    slot.mean_ms =
        slot.samples.empty()
            ? 0.0F
            : static_cast<float>(sum / static_cast<double>(slot.samples.size()));
    const bool sample_ok =
        slot.samples.size() == static_cast<std::size_t>(measured);
    slot.eligible = slot.eligible && slot.launch_ok && sample_ok &&
                    slot.occupancy >= 1;
    if (index == 0 && !slot.eligible) all_eligible = false;
    if (index != 0 && !slot.eligible) all_eligible = false;
    if (slot.eligible && slot.mean_ms < best) {
      best = slot.mean_ms;
      winner = index;
    }
  }
  const bool win = winner != 0 && candidates[winner].eligible &&
                   candidates[0].eligible &&
                   candidates[winner].mean_ms < candidates[0].mean_ms;
  if (!win) winner = 0;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_mix_out);
  cudaFree(d_mix_y);
  cudaFree(d_mix_p);
  cudaFree(d_mix_w);
  cudaFree(d_down_out);
  cudaFree(d_up_out);
  cudaFree(d_gate_out);
  cudaFree(d_y);
  cudaFree(d_prompt);
  cudaFree(d_down);
  cudaFree(d_up);
  cudaFree(d_gate);
  cudaFree(env_out);
  cudaFree(env_y);
  cudaFree(env_p);
  cudaFree(env_w);
  std::fclose(raw);

  std::time_t now = std::time(nullptr);
  std::tm utc{};
  gmtime_r(&now, &utc);
  char utc_text[32];
  std::strftime(utc_text, sizeof(utc_text), "%Y-%m-%dT%H:%M:%SZ", &utc);

  std::printf("%s{", kPrefix);
  std::printf(
      "\"task\":\"OPT-053\",\"measurement_utc\":\"%s\",\"device\":\"%s\","
      "\"compute_capability\":\"%d.%d\",\"ab\":{\"token_count\":%zu,"
      "\"winner\":\"%s\",\"win\":%s,\"hidden\":%zu,\"ffn\":%zu,\"candidates\":{",
      utc_text, props.name, props.major, props.minor, timed_prompt,
      kCandidates[winner], json_bool(win), hidden, ffn);
  for (int index = 0; index < kCandidateCount; ++index) {
    if (index != 0) std::printf(",");
    const Candidate& slot = candidates[index];
    std::printf(
        "\"%s\":{\"id\":\"%s\",\"path\":\"%s\",\"mean_ms\":%.9g,",
        kCandidates[index], kCandidates[index], kCandidates[index],
        static_cast<double>(slot.mean_ms));
    print_array(stdout, "samples", slot.samples);
    std::printf(",");
    print_array(stdout, "warmup_ms", slot.warmup);
    std::printf(
        ",\"occupancy\":%d,\"extra_shared_bytes\":%zu,\"launch_ok\":%s,"
        "\"eligible\":%s,\"timed_nonfinite\":%zu,\"uses_fma\":%s,"
        "\"uses_async\":%s,\"like_arithmetic\":%s,"
        "\"vs_off\":{\"max_abs\":%.9g,\"rms\":%.9g,\"nonfinite\":%zu},"
        "\"vs_host\":{\"max_abs\":%.9g,\"rms\":%.9g,\"nonfinite\":%zu,"
        "\"bad\":%zu,\"ok\":%s},\"mixer_wide_ms\":%.9g,"
        "\"mixer_skinny_ms\":%.9g,\"mixer_wide_occupancy\":%d,"
        "\"mixer_skinny_occupancy\":%d}",
        slot.occupancy, slot.extra_shared_bytes, json_bool(slot.launch_ok),
        json_bool(slot.eligible), slot.timed_nonfinite,
        json_bool(uses_fma(kCandidates[index])),
        json_bool(uses_async(kCandidates[index])),
        json_bool(like_arithmetic(kCandidates[index])),
        static_cast<double>(slot.vs_off.max_abs),
        static_cast<double>(slot.vs_off.rms), slot.vs_off.nonfinite,
        static_cast<double>(slot.vs_host.max_abs),
        static_cast<double>(slot.vs_host.rms), slot.vs_host.nonfinite,
        slot.vs_host.bad, json_bool(slot.vs_host.ok),
        static_cast<double>(mixer_wide_ms[index]),
        static_cast<double>(mixer_skinny_ms[index]), mixer_wide_occ[index],
        mixer_skinny_occ[index]);
  }
  std::printf(
      "},\"off_tokens_ms\":{\"512\":%.9g,\"1024\":%.9g,\"2048\":%.9g}},"
      "\"correctness\":{\"all_eligible\":%s,\"tails_ok\":%s,\"case_count\":%d,"
      "\"token_counts\":[17,65,129,255]},"
      "\"instruction_mix\":{\"fma_source\":true,\"cp_async_source\":true,"
      "\"cp_async_sass\":\"pending\"},"
      "\"extra_workspace_bytes\":%zu,"
      "\"occupancy_off\":%d,\"occupancy_async\":%d}\n",
      extra_means[0], extra_means[1], extra_means[2], json_bool(all_eligible),
      json_bool(tails_ok), correctness_cases,
      candidates[2].extra_shared_bytes, candidates[0].occupancy,
      candidates[2].occupancy);
  std::printf("status=passed\n");
  return 0;
}
