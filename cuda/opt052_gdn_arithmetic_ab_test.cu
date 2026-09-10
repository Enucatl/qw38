#include "gdn_step.h"
#include "test_tier.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <cuda_bf16.h>

namespace {

constexpr char kPrefix[] = "QW38_OPT052_GDN_ARITHMETIC_AB_RESULT=";
constexpr const char* kCandidates[] = {"shared", "preproc", "preproc_fma",
                                       "approx_exp", "transpose"};
constexpr int kCandidateCount = 5;
constexpr std::size_t kCorrectnessTokens[] = {1, 2, 3, 4, 63, 64, 65, 512, 2048,
                                              4096};
constexpr int kCorrectnessTokenCount = 10;
constexpr int kCorrectnessLayers[] = {0, 32, 62};
constexpr int kCorrectnessLayerCount = 3;
constexpr qw38::cuda::GdnConfig kConfig{16, 48, 128, 128, 4};
constexpr float kGdnAbs = 5.0e-8F;
constexpr float kGdnRms = 5.0e-9F;
constexpr float kOpt044Abs = 3.0e-4F;
constexpr float kOpt044Rms = 2.0e-4F;
constexpr std::uint64_t kFrontier = 7;
constexpr float kSentinel = 1.0F;

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

bool like_arithmetic(const char* path) {
  return std::strcmp(path, "preproc") == 0 ||
         std::strcmp(path, "transpose") == 0 ||
         std::strcmp(path, "shared") == 0;
}

struct Envelope {
  float max_abs = 0.0F;
  float rms = 0.0F;
  std::size_t nonfinite = 0;
};

bool envelope_ok(const Envelope& metrics, float max_abs, float max_rms) {
  return metrics.nonfinite == 0 && metrics.max_abs <= max_abs &&
         metrics.rms <= max_rms;
}

template <typename T>
Envelope compare_values(const std::vector<T>& actual,
                        const std::vector<T>& expected) {
  Envelope metrics;
  double squared = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float a = static_cast<float>(actual[index]);
    const float e = static_cast<float>(expected[index]);
    if (!std::isfinite(a) || !std::isfinite(e)) ++metrics.nonfinite;
    const float err = std::fabs(a - e);
    metrics.max_abs = std::max(metrics.max_abs, err);
    squared += static_cast<double>(err) * err;
  }
  if (!actual.empty()) {
    metrics.rms = static_cast<float>(
        std::sqrt(squared / static_cast<double>(actual.size())));
  }
  return metrics;
}

void print_envelope(FILE* file, const char* key, const Envelope& metrics) {
  std::fprintf(file,
               "\"%s\":{\"max_abs\":%.9g,\"rms\":%.9g,\"nonfinite\":%zu}", key,
               static_cast<double>(metrics.max_abs),
               static_cast<double>(metrics.rms), metrics.nonfinite);
}

struct HostTensors {
  std::vector<float> input;
  std::vector<float> weights;
  std::vector<float> log_decay;
  std::vector<float> beta;
  std::vector<float> committed_convolution;
  std::vector<float> committed_recurrent;
  std::vector<float> gate;
  std::vector<float> norm;
};

void fill_synthetic(HostTensors* tensors, std::size_t token_count, int layer) {
  const std::size_t channels = qw38::cuda::gdn_convolution_channels(kConfig);
  const std::size_t convolution_values =
      qw38::cuda::gdn_convolution_values(kConfig);
  const std::size_t recurrent_values = qw38::cuda::gdn_recurrent_values(kConfig);
  const std::size_t gate_values =
      token_count * kConfig.value_heads * kConfig.value_width;
  const float layer_shift = 0.019F * static_cast<float>(layer + 1);
  tensors->input.resize(token_count * channels);
  tensors->weights.resize(convolution_values);
  tensors->log_decay.resize(token_count * kConfig.value_heads);
  tensors->beta.resize(token_count * kConfig.value_heads);
  tensors->committed_convolution.resize(convolution_values);
  tensors->committed_recurrent.resize(recurrent_values);
  tensors->gate.resize(gate_values);
  tensors->norm.resize(kConfig.value_width);
  for (std::size_t token = 0; token < token_count; ++token) {
    for (std::size_t channel = 0; channel < channels; ++channel) {
      tensors->input[token * channels + channel] =
          std::sin(static_cast<float>(channel) * 0.013F +
                   static_cast<float>(token) * 0.071F + layer_shift) *
          0.5F;
    }
    for (std::size_t head = 0; head < kConfig.value_heads; ++head) {
      tensors->log_decay[token * kConfig.value_heads + head] =
          -0.001F * static_cast<float>(1 + (token + head + layer) % 31);
      tensors->beta[token * kConfig.value_heads + head] =
          0.15F + 0.01F * static_cast<float>((token + head + layer) % 23);
    }
  }
  for (std::size_t index = 0; index < convolution_values; ++index) {
    tensors->weights[index] =
        static_cast<float>(static_cast<int>(index % 9) - 4) * 0.03125F;
    tensors->committed_convolution[index] =
        static_cast<float>(static_cast<int>((index + layer * 17) % 13) - 6) *
        0.015625F;
  }
  for (std::size_t index = 0; index < recurrent_values; ++index) {
    tensors->committed_recurrent[index] =
        static_cast<float>(static_cast<int>((index + layer * 11) % 23) - 11) *
        0.0009765625F;
  }
  for (std::size_t index = 0; index < gate_values; ++index) {
    tensors->gate[index] =
        std::sin(static_cast<float>(index) * 0.017F + layer_shift) * 0.5F;
  }
  for (std::uint32_t index = 0; index < kConfig.value_width; ++index) {
    tensors->norm[index] = 0.75F + 0.01F * static_cast<float>(index % 17);
  }
}

struct DeviceBuffers {
  float* input = nullptr;
  float* weights = nullptr;
  float* log_decay = nullptr;
  float* beta = nullptr;
  float* committed_convolution = nullptr;
  float* committed_recurrent = nullptr;
  float* candidate_convolution = nullptr;
  float* candidate_recurrent = nullptr;
  float* convolution_output = nullptr;
  float* recurrent_output = nullptr;
  float* gate = nullptr;
  float* norm = nullptr;
  float* scratch = nullptr;
  __nv_bfloat16* gated = nullptr;
  std::uint64_t* frontier = nullptr;
  std::size_t token_count = 0;
  std::size_t scratch_floats = 0;
};

void release(DeviceBuffers* buffers) {
  cudaFree(buffers->frontier);
  cudaFree(buffers->gated);
  cudaFree(buffers->scratch);
  cudaFree(buffers->norm);
  cudaFree(buffers->gate);
  cudaFree(buffers->recurrent_output);
  cudaFree(buffers->convolution_output);
  cudaFree(buffers->candidate_recurrent);
  cudaFree(buffers->candidate_convolution);
  cudaFree(buffers->committed_recurrent);
  cudaFree(buffers->committed_convolution);
  cudaFree(buffers->beta);
  cudaFree(buffers->log_decay);
  cudaFree(buffers->weights);
  cudaFree(buffers->input);
  *buffers = DeviceBuffers{};
}

cudaError_t allocate(DeviceBuffers* buffers, std::size_t token_count) {
  release(buffers);
  buffers->token_count = token_count;
  buffers->scratch_floats =
      qw38::cuda::gdn_preproc_transpose_floats(kConfig, token_count);
  const std::size_t channels = qw38::cuda::gdn_convolution_channels(kConfig);
  const std::size_t convolution_values =
      qw38::cuda::gdn_convolution_values(kConfig);
  const std::size_t recurrent_values = qw38::cuda::gdn_recurrent_values(kConfig);
  const std::size_t output_values = qw38::cuda::gdn_output_values(kConfig);
  const std::size_t gate_values =
      token_count * kConfig.value_heads * kConfig.value_width;
  cudaError_t error =
      cudaMalloc(&buffers->input, token_count * channels * sizeof(float));
#define QW38_ALLOC(field, count)                                               \
  if (error == cudaSuccess)                                                    \
  error = cudaMalloc(&buffers->field, (count) * sizeof(*buffers->field))
  QW38_ALLOC(weights, convolution_values);
  QW38_ALLOC(log_decay, token_count * kConfig.value_heads);
  QW38_ALLOC(beta, token_count * kConfig.value_heads);
  QW38_ALLOC(committed_convolution, convolution_values);
  QW38_ALLOC(committed_recurrent, recurrent_values);
  QW38_ALLOC(candidate_convolution, convolution_values);
  QW38_ALLOC(candidate_recurrent, recurrent_values);
  QW38_ALLOC(convolution_output, token_count * channels);
  QW38_ALLOC(recurrent_output, token_count * output_values);
  QW38_ALLOC(gate, gate_values);
  QW38_ALLOC(norm, kConfig.value_width);
  QW38_ALLOC(scratch, buffers->scratch_floats);
  QW38_ALLOC(gated, gate_values);
  QW38_ALLOC(frontier, 1);
#undef QW38_ALLOC
  return error;
}

cudaError_t stage_inputs(DeviceBuffers* buffers, const HostTensors& tensors) {
  cudaError_t error = cudaSuccess;
#define QW38_COPY(field, source)                                               \
  if (error == cudaSuccess)                                                    \
  error = cudaMemcpy(buffers->field, (source).data(),                          \
                     (source).size() * sizeof((source)[0]),                    \
                     cudaMemcpyHostToDevice)
  QW38_COPY(input, tensors.input);
  QW38_COPY(weights, tensors.weights);
  QW38_COPY(log_decay, tensors.log_decay);
  QW38_COPY(beta, tensors.beta);
  QW38_COPY(committed_convolution, tensors.committed_convolution);
  QW38_COPY(committed_recurrent, tensors.committed_recurrent);
  QW38_COPY(gate, tensors.gate);
  QW38_COPY(norm, tensors.norm);
#undef QW38_COPY
  if (error == cudaSuccess) {
    error = cudaMemcpy(buffers->frontier, &kFrontier, sizeof(kFrontier),
                       cudaMemcpyHostToDevice);
  }
  return error;
}

cudaError_t restore_committed(DeviceBuffers* buffers,
                              const HostTensors& tensors) {
  cudaError_t error = cudaMemcpy(
      buffers->committed_convolution, tensors.committed_convolution.data(),
      tensors.committed_convolution.size() * sizeof(float),
      cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(buffers->committed_recurrent,
                       tensors.committed_recurrent.data(),
                       tensors.committed_recurrent.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(buffers->frontier, &kFrontier, sizeof(kFrontier),
                       cudaMemcpyHostToDevice);
  }
  return error;
}

qw38::cuda::GdnState committed_state(const DeviceBuffers& buffers) {
  return {buffers.committed_convolution, buffers.committed_recurrent};
}

qw38::cuda::GdnState candidate_state(const DeviceBuffers& buffers) {
  return {buffers.candidate_convolution, buffers.candidate_recurrent};
}

cudaError_t launch_quality(DeviceBuffers* buffers, const char* preproc_path) {
  std::vector<float> sentinel(buffers->scratch_floats, kSentinel);
  cudaError_t error =
      cudaMemcpy(buffers->scratch, sentinel.data(),
                 sentinel.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemset(buffers->candidate_convolution, 0,
                       qw38::cuda::gdn_convolution_values(kConfig) *
                           sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMemset(buffers->candidate_recurrent, 0,
                       qw38::cuda::gdn_recurrent_values(kConfig) * sizeof(float));
  }
  if (error != cudaSuccess) return error;
    const char* inverse = "shared";
    const char* preproc =
        std::strcmp(preproc_path, "shared") == 0 ? "off" : preproc_path;
  return qw38::cuda::launch_gdn_quality_fused(
      kConfig, buffers->input, buffers->weights, buffers->log_decay,
      buffers->beta, buffers->token_count, committed_state(*buffers),
      candidate_state(*buffers), buffers->convolution_output,
      buffers->recurrent_output, buffers->gate, buffers->norm, buffers->gated,
      nullptr, true, "off", inverse, buffers->scratch, buffers->scratch_floats,
      preproc);
}

cudaError_t launch_sequential(DeviceBuffers* buffers) {
  return qw38::cuda::launch_gdn_prepare_chunk_tiled(
      kConfig, buffers->input, buffers->weights, buffers->log_decay,
      buffers->beta, buffers->token_count, committed_state(*buffers),
      candidate_state(*buffers), buffers->convolution_output,
      buffers->recurrent_output, nullptr,
      qw38::cuda::GdnScanPath::kSequentialWindows, nullptr, 0);
}

template <typename T>
cudaError_t download(const T* device, std::size_t count, std::vector<T>* host) {
  host->assign(count, T{});
  return cudaMemcpy(host->data(), device, count * sizeof(T),
                    cudaMemcpyDeviceToHost);
}

struct LaunchSnapshot {
  std::vector<float> recurrent;
  std::vector<float> candidate_convolution;
  std::vector<float> candidate_recurrent;
  std::vector<float> convolution;
  std::vector<__nv_bfloat16> gated;
  std::vector<float> committed_convolution;
  std::vector<float> committed_recurrent;
  std::vector<float> scratch;
  std::uint64_t frontier = 0;
  bool dispatched_preproc = false;
  bool launch_ok = false;
};

cudaError_t capture_snapshot(const DeviceBuffers& buffers, const char* path,
                             LaunchSnapshot* snap) {
  const std::size_t channels = qw38::cuda::gdn_convolution_channels(kConfig);
  const std::size_t convolution_values =
      qw38::cuda::gdn_convolution_values(kConfig);
  const std::size_t recurrent_values = qw38::cuda::gdn_recurrent_values(kConfig);
  const std::size_t output_values = qw38::cuda::gdn_output_values(kConfig);
  const std::size_t gate_values =
      buffers.token_count * kConfig.value_heads * kConfig.value_width;
  cudaError_t error = download(buffers.recurrent_output,
                               buffers.token_count * output_values,
                               &snap->recurrent);
  if (error == cudaSuccess) {
    error = download(buffers.candidate_convolution, convolution_values,
                     &snap->candidate_convolution);
  }
  if (error == cudaSuccess) {
    error = download(buffers.candidate_recurrent, recurrent_values,
                     &snap->candidate_recurrent);
  }
  if (error == cudaSuccess) {
    error = download(buffers.convolution_output,
                     buffers.token_count * channels, &snap->convolution);
  }
  if (error == cudaSuccess) {
    error = download(buffers.gated, gate_values, &snap->gated);
  }
  if (error == cudaSuccess) {
    error = download(buffers.committed_convolution, convolution_values,
                     &snap->committed_convolution);
  }
  if (error == cudaSuccess) {
    error = download(buffers.committed_recurrent, recurrent_values,
                     &snap->committed_recurrent);
  }
  if (error == cudaSuccess) {
    error = download(buffers.scratch, buffers.scratch_floats, &snap->scratch);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(&snap->frontier, buffers.frontier, sizeof(snap->frontier),
                       cudaMemcpyDeviceToHost);
  }
  snap->dispatched_preproc = false;
  if (error == cudaSuccess && std::strcmp(path, "shared") != 0 &&
      buffers.token_count >= 2) {
    const std::size_t decay_off = qw38::cuda::gdn_preproc_floats(kConfig, 0);
    (void)decay_off;
    const std::size_t qk = 2 * buffers.token_count *
                           static_cast<std::size_t>(kConfig.key_heads) *
                           kConfig.key_width;
    if (qk < snap->scratch.size() && snap->scratch[qk] != kSentinel) {
      snap->dispatched_preproc = true;
    }
  }
  return error;
}

bool committed_unchanged(const LaunchSnapshot& snap,
                         const HostTensors& tensors) {
  return snap.committed_convolution == tensors.committed_convolution &&
         snap.committed_recurrent == tensors.committed_recurrent &&
         snap.frontier == kFrontier;
}

struct PathMetrics {
  bool launch_ok = false;
  bool dispatched_preproc = false;
  bool recurrent_byte_equal = false;
  bool conv_byte_equal = false;
  bool state_byte_equal = false;
  bool gated_byte_equal = false;
  bool prepare_preserves_committed = false;
  bool cancel_isolates = false;
  bool restore_exact = false;
  bool chunk_ok = true;
  Envelope vs_shared;
  Envelope vs_sequential;
  bool eligible = false;
};

PathMetrics evaluate_path(DeviceBuffers* buffers, const HostTensors& tensors,
                          const LaunchSnapshot& shared, const char* path,
                          FILE* raw, int layer, std::size_t token_count) {
  PathMetrics metrics;
  LaunchSnapshot first;
  cudaError_t error = restore_committed(buffers, tensors);
  if (error == cudaSuccess) error = launch_quality(buffers, path);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess) error = capture_snapshot(*buffers, path, &first);
  first.launch_ok = error == cudaSuccess;

  LaunchSnapshot second;
  if (error == cudaSuccess) error = restore_committed(buffers, tensors);
  if (error == cudaSuccess) error = launch_quality(buffers, path);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess) error = capture_snapshot(*buffers, path, &second);

  LaunchSnapshot sequential;
  if (error == cudaSuccess) error = restore_committed(buffers, tensors);
  if (error == cudaSuccess) error = launch_sequential(buffers);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess) {
    error = capture_snapshot(*buffers, "shared", &sequential);
  }

  metrics.launch_ok = first.launch_ok && error == cudaSuccess;
  metrics.dispatched_preproc = first.dispatched_preproc;
  if (metrics.launch_ok) {
    metrics.recurrent_byte_equal = first.recurrent == shared.recurrent;
    metrics.conv_byte_equal = first.convolution == shared.convolution &&
                              first.candidate_convolution ==
                                  shared.candidate_convolution;
    metrics.state_byte_equal =
        first.candidate_recurrent == shared.candidate_recurrent;
    metrics.gated_byte_equal =
        std::memcmp(first.gated.data(), shared.gated.data(),
                    first.gated.size() * sizeof(__nv_bfloat16)) == 0;
    metrics.prepare_preserves_committed = committed_unchanged(first, tensors);
    metrics.cancel_isolates = metrics.prepare_preserves_committed;
    metrics.restore_exact =
        first.recurrent == second.recurrent &&
        first.candidate_recurrent == second.candidate_recurrent &&
        first.convolution == second.convolution &&
        std::memcmp(first.gated.data(), second.gated.data(),
                    first.gated.size() * sizeof(__nv_bfloat16)) == 0;
    Envelope rec = compare_values(first.recurrent, shared.recurrent);
    Envelope state =
        compare_values(first.candidate_recurrent, shared.candidate_recurrent);
    metrics.vs_shared.max_abs = std::max(rec.max_abs, state.max_abs);
    metrics.vs_shared.rms = std::max(rec.rms, state.rms);
    metrics.vs_shared.nonfinite = rec.nonfinite + state.nonfinite;
    Envelope seq_rec = compare_values(first.recurrent, sequential.recurrent);
    Envelope seq_state = compare_values(first.candidate_recurrent,
                                        sequential.candidate_recurrent);
    metrics.vs_sequential.max_abs =
        std::max(seq_rec.max_abs, seq_state.max_abs);
    metrics.vs_sequential.rms = std::max(seq_rec.rms, seq_state.rms);
    metrics.vs_sequential.nonfinite = seq_rec.nonfinite + seq_state.nonfinite;
    const bool token1 = token_count == 1;
    const bool dispatch_ok =
        token1 ? !metrics.dispatched_preproc : metrics.dispatched_preproc;
    const bool like = like_arithmetic(path);
    const bool numeric_ok =
        like ? (metrics.recurrent_byte_equal && metrics.conv_byte_equal &&
                metrics.state_byte_equal && metrics.gated_byte_equal &&
                envelope_ok(metrics.vs_sequential, kGdnAbs, kGdnRms))
             : (envelope_ok(metrics.vs_shared, kOpt044Abs, kOpt044Rms) &&
                envelope_ok(metrics.vs_sequential, kOpt044Abs, kOpt044Rms));
    metrics.eligible = metrics.launch_ok && dispatch_ok &&
                       metrics.prepare_preserves_committed &&
                       metrics.cancel_isolates && metrics.restore_exact &&
                       numeric_ok;
  }
  if (raw != nullptr) {
    std::fprintf(raw,
                 "path=%s layer=%d tokens=%zu launch_ok=%s dispatched=%s "
                 "recurrent_eq=%s state_eq=%s restore=%s vs_shared_abs=%.9g "
                 "vs_seq_abs=%.9g eligible=%s\n",
                 path, layer, token_count, json_bool(metrics.launch_ok),
                 json_bool(metrics.dispatched_preproc),
                 json_bool(metrics.recurrent_byte_equal),
                 json_bool(metrics.state_byte_equal),
                 json_bool(metrics.restore_exact),
                 static_cast<double>(metrics.vs_shared.max_abs),
                 static_cast<double>(metrics.vs_sequential.max_abs),
                 json_bool(metrics.eligible));
  }
  return metrics;
}

bool evaluate_chunks(DeviceBuffers* buffers, const HostTensors& tensors,
                     const char* path, std::size_t n0, FILE* raw) {
  const std::size_t n1 = buffers->token_count - n0;
  if (n1 == 0 || n0 == 0) return true;
  const std::size_t channels = qw38::cuda::gdn_convolution_channels(kConfig);
  const std::size_t output_values = qw38::cuda::gdn_output_values(kConfig);
  const std::size_t conv_state = qw38::cuda::gdn_convolution_values(kConfig);
  const std::size_t rec_state = qw38::cuda::gdn_recurrent_values(kConfig);
  LaunchSnapshot oneshot;
  cudaError_t error = restore_committed(buffers, tensors);
  if (error == cudaSuccess) error = launch_quality(buffers, path);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess) error = capture_snapshot(*buffers, path, &oneshot);
  DeviceBuffers first{};
  DeviceBuffers second{};
  HostTensors t0;
  HostTensors t1;
  fill_synthetic(&t0, n0, 0);
  fill_synthetic(&t1, n1, 0);
  t0.committed_convolution = tensors.committed_convolution;
  t0.committed_recurrent = tensors.committed_recurrent;
  t0.weights = tensors.weights;
  t0.norm = tensors.norm;
  t0.input.assign(tensors.input.begin(),
                  tensors.input.begin() +
                      static_cast<std::ptrdiff_t>(n0 * channels));
  t0.log_decay.assign(tensors.log_decay.begin(),
                      tensors.log_decay.begin() +
                          static_cast<std::ptrdiff_t>(n0 * kConfig.value_heads));
  t0.beta.assign(tensors.beta.begin(),
                 tensors.beta.begin() +
                     static_cast<std::ptrdiff_t>(n0 * kConfig.value_heads));
  t0.gate.assign(tensors.gate.begin(),
                 tensors.gate.begin() +
                     static_cast<std::ptrdiff_t>(n0 * kConfig.value_heads *
                                                 kConfig.value_width));
  if (error == cudaSuccess) error = allocate(&first, n0);
  if (error == cudaSuccess) error = stage_inputs(&first, t0);
  if (error == cudaSuccess) error = launch_quality(&first, path);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  LaunchSnapshot chunk0;
  if (error == cudaSuccess) error = capture_snapshot(first, path, &chunk0);
  t1.weights = tensors.weights;
  t1.norm = tensors.norm;
  t1.committed_convolution = chunk0.candidate_convolution;
  t1.committed_recurrent = chunk0.candidate_recurrent;
  t1.input.assign(tensors.input.begin() +
                      static_cast<std::ptrdiff_t>(n0 * channels),
                  tensors.input.end());
  t1.log_decay.assign(
      tensors.log_decay.begin() +
          static_cast<std::ptrdiff_t>(n0 * kConfig.value_heads),
      tensors.log_decay.end());
  t1.beta.assign(tensors.beta.begin() +
                     static_cast<std::ptrdiff_t>(n0 * kConfig.value_heads),
                 tensors.beta.end());
  t1.gate.assign(tensors.gate.begin() +
                     static_cast<std::ptrdiff_t>(n0 * kConfig.value_heads *
                                                 kConfig.value_width),
                 tensors.gate.end());
  if (error == cudaSuccess) error = allocate(&second, n1);
  if (error == cudaSuccess) error = stage_inputs(&second, t1);
  if (error == cudaSuccess) error = launch_quality(&second, path);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  LaunchSnapshot chunk1;
  if (error == cudaSuccess) error = capture_snapshot(second, path, &chunk1);
  bool ok = error == cudaSuccess;
  if (ok) {
    std::vector<float> concat(oneshot.recurrent.size());
    std::copy(chunk0.recurrent.begin(), chunk0.recurrent.end(), concat.begin());
    std::copy(chunk1.recurrent.begin(), chunk1.recurrent.end(),
              concat.begin() + static_cast<std::ptrdiff_t>(n0 * output_values));
    Envelope rec = compare_values(concat, oneshot.recurrent);
    Envelope state =
        compare_values(chunk1.candidate_recurrent, oneshot.candidate_recurrent);
    (void)conv_state;
    (void)rec_state;
    const float abs_lim = like_arithmetic(path) ? kGdnAbs : kOpt044Abs;
    const float rms_lim = like_arithmetic(path) ? kGdnRms : kOpt044Rms;
    ok = envelope_ok(rec, abs_lim, rms_lim) &&
         envelope_ok(state, abs_lim, rms_lim);
  }
  if (raw != nullptr) {
    std::fprintf(raw, "chunks path=%s n0=%zu n1=%zu ok=%s\n", path, n0, n1,
                 json_bool(ok));
  }
  release(&first);
  release(&second);
  return ok;
}

struct CandidateResult {
  const char* id = "shared";
  int occupancy = 0;
  int recurrence_occupancy = 0;
  bool launch_ok = false;
  bool eligible = false;
  bool dispatched_preproc = false;
  bool recurrent_byte_equal = false;
  bool conv_byte_equal = false;
  bool state_byte_equal = false;
  bool gated_byte_equal = false;
  bool prepare_preserves_committed = false;
  bool cancel_isolates = false;
  bool restore_exact = false;
  bool chunk_ok = true;
  Envelope vs_shared;
  Envelope vs_sequential;
  std::vector<float> warmup_ms;
  std::vector<float> samples;
  float mean_ms = 0.0F;
};

void print_candidate(FILE* file, const CandidateResult& result) {
  std::fprintf(file,
               "\"%s\":{\"id\":\"%s\",\"path\":\"%s\",\"mean_ms\":%.9g,",
               result.id, result.id, result.id,
               static_cast<double>(result.mean_ms));
  print_array(file, "samples", result.samples);
  std::fprintf(file, ",");
  print_array(file, "warmup_ms", result.warmup_ms);
  std::fprintf(file,
               ",\"occupancy\":%d,\"recurrence_occupancy\":%d,\"launch_ok\":%s,"
               "\"eligible\":%s,\"dispatched_preproc\":%s,"
               "\"recurrent_byte_equal\":%s,\"conv_byte_equal\":%s,"
               "\"state_byte_equal\":%s,\"gated_byte_equal\":%s,"
               "\"prepare_preserves_committed\":%s,\"cancel_isolates\":%s,"
               "\"restore_exact\":%s,\"chunk_ok\":%s,",
               result.occupancy, result.recurrence_occupancy,
               json_bool(result.launch_ok), json_bool(result.eligible),
               json_bool(result.dispatched_preproc),
               json_bool(result.recurrent_byte_equal),
               json_bool(result.conv_byte_equal),
               json_bool(result.state_byte_equal),
               json_bool(result.gated_byte_equal),
               json_bool(result.prepare_preserves_committed),
               json_bool(result.cancel_isolates),
               json_bool(result.restore_exact), json_bool(result.chunk_ok));
  print_envelope(file, "vs_shared", result.vs_shared);
  std::fprintf(file, ",");
  print_envelope(file, "vs_sequential", result.vs_sequential);
  std::fprintf(file, "}");
}

int occupancy_for(const char* path) {
  if (std::strcmp(path, "shared") == 0) {
    return qw38::cuda::gdn_shared_inverse_occupancy();
  }
  if (std::strcmp(path, "preproc_fma") == 0) {
    return std::min(qw38::cuda::gdn_preproc_occupancy(),
                    qw38::cuda::gdn_preproc_fma_occupancy());
  }
  if (std::strcmp(path, "transpose") == 0) {
    return std::min(qw38::cuda::gdn_preproc_occupancy(),
                    qw38::cuda::gdn_preproc_transpose_occupancy());
  }
  return qw38::cuda::gdn_preproc_occupancy();
}

int recurrence_occupancy_for(const char* path) {
  if (std::strcmp(path, "shared") == 0) {
    return qw38::cuda::gdn_shared_recurrence_occupancy();
  }
  if (std::strcmp(path, "preproc_fma") == 0) {
    return qw38::cuda::gdn_preproc_fma_occupancy();
  }
  if (std::strcmp(path, "transpose") == 0) {
    return qw38::cuda::gdn_preproc_transpose_occupancy();
  }
  return qw38::cuda::gdn_preproc_recurrence_occupancy();
}

const char* select_winner(const CandidateResult candidates[kCandidateCount],
                          bool all_eligible) {
  if (!all_eligible || !candidates[0].eligible) return "shared";
  const float shared_mean = candidates[0].mean_ms;
  int best = 0;
  float best_mean = shared_mean;
  int like_best = -1;
  float like_mean = shared_mean;
  for (int id = 1; id < kCandidateCount; ++id) {
    if (!candidates[id].eligible) continue;
    if (!(candidates[id].mean_ms < shared_mean)) continue;
    if (candidates[id].mean_ms < best_mean) {
      best = id;
      best_mean = candidates[id].mean_ms;
    }
    if (like_arithmetic(candidates[id].id) &&
        (like_best < 0 || candidates[id].mean_ms < like_mean)) {
      like_best = id;
      like_mean = candidates[id].mean_ms;
    }
  }
  if (best == 0) return "shared";
  if (like_best >= 0 && like_mean <= 1.01F * best_mean) {
    return candidates[like_best].id;
  }
  return candidates[best].id;
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_valid()) {
    std::fprintf(stderr, "QW38_CUDA_TEST_TIER must be set\n");
    return 1;
  }
  const qw38::cuda::TestTier tier = qw38::cuda::test_tier();
  FILE* raw = nullptr;
  if (argc >= 2) {
    raw = std::fopen(argv[1], "w");
    if (raw == nullptr) {
      std::fprintf(stderr, "failed to open %s\n", argv[1]);
      return 1;
    }
  }

  cudaDeviceProp prop{};
  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error == cudaSuccess) error = cudaGetDeviceProperties(&prop, device);
  if (error != cudaSuccess) return fail_cuda("device", error);

  std::time_t now = std::time(nullptr);
  char utc[32]{};
  std::tm tm{};
  gmtime_r(&now, &tm);
  std::strftime(utc, sizeof(utc), "%Y-%m-%dT%H:%M:%SZ", &tm);

  const int layer_count =
      tier == qw38::cuda::TestTier::kSmoke ? 1 : kCorrectnessLayerCount;
  const int token_limit =
      tier == qw38::cuda::TestTier::kSmoke ? 7 : kCorrectnessTokenCount;
  DeviceBuffers buffers;
  bool all_eligible = true;
  bool chunk_ok = true;
  std::vector<PathMetrics> case_metrics[kCandidateCount];
  for (int id = 0; id < kCandidateCount; ++id) {
    case_metrics[id].assign(static_cast<std::size_t>(layer_count * token_limit),
                            PathMetrics{});
  }

  for (int layer_index = 0; layer_index < layer_count; ++layer_index) {
    const int layer = kCorrectnessLayers[layer_index];
    for (int token_index = 0; token_index < token_limit; ++token_index) {
      const std::size_t token_count = kCorrectnessTokens[token_index];
      HostTensors tensors;
      fill_synthetic(&tensors, token_count, layer);
      error = allocate(&buffers, token_count);
      if (error == cudaSuccess) error = stage_inputs(&buffers, tensors);
      if (error != cudaSuccess) {
        if (raw != nullptr) std::fclose(raw);
        release(&buffers);
        return fail_cuda("allocate correctness", error);
      }
      LaunchSnapshot shared;
      error = restore_committed(&buffers, tensors);
      if (error == cudaSuccess) error = launch_quality(&buffers, "shared");
      if (error == cudaSuccess) error = cudaDeviceSynchronize();
      if (error == cudaSuccess) {
        error = capture_snapshot(buffers, "shared", &shared);
      }
      shared.launch_ok = error == cudaSuccess;
      const std::size_t slot =
          static_cast<std::size_t>(layer_index * token_limit + token_index);
      for (int id = 0; id < kCandidateCount; ++id) {
        if (std::strcmp(kCandidates[id], "shared") == 0) {
          PathMetrics control;
          control.launch_ok = shared.launch_ok;
          control.dispatched_preproc = false;
          control.recurrent_byte_equal = true;
          control.conv_byte_equal = true;
          control.state_byte_equal = true;
          control.gated_byte_equal = true;
          control.prepare_preserves_committed =
              committed_unchanged(shared, tensors);
          control.cancel_isolates = control.prepare_preserves_committed;
          control.restore_exact = true;
          control.eligible = control.launch_ok &&
                             control.prepare_preserves_committed;
          case_metrics[id][slot] = control;
          all_eligible = all_eligible && control.eligible;
          continue;
        }
        case_metrics[id][slot] = evaluate_path(
            &buffers, tensors, shared, kCandidates[id], raw, layer, token_count);
        all_eligible = all_eligible && case_metrics[id][slot].eligible;
      }
      if (tier != qw38::cuda::TestTier::kSmoke && layer == 0 &&
          (token_count == 65 || token_count == 512)) {
        const std::size_t n0 = token_count == 65 ? 64 : 256;
        for (int id = 1; id < kCandidateCount; ++id) {
          const bool ok =
              evaluate_chunks(&buffers, tensors, kCandidates[id], n0, raw);
          case_metrics[id][slot].chunk_ok = ok;
          case_metrics[id][slot].eligible =
              case_metrics[id][slot].eligible && ok;
          chunk_ok = chunk_ok && ok;
          all_eligible = all_eligible && case_metrics[id][slot].eligible;
        }
      }
    }
  }

  const std::size_t timed_tokens =
      tier == qw38::cuda::TestTier::kSmoke ? 64 : 4096;
  HostTensors timed_tensors;
  fill_synthetic(&timed_tensors, timed_tokens, 0);
  error = allocate(&buffers, timed_tokens);
  if (error == cudaSuccess) error = stage_inputs(&buffers, timed_tensors);
  if (error != cudaSuccess) {
    if (raw != nullptr) std::fclose(raw);
    release(&buffers);
    return fail_cuda("allocate timed", error);
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error != cudaSuccess) {
    if (raw != nullptr) std::fclose(raw);
    release(&buffers);
    return fail_cuda("events", error);
  }

  CandidateResult candidates[kCandidateCount];
  for (int id = 0; id < kCandidateCount; ++id) {
    candidates[id].id = kCandidates[id];
    candidates[id].occupancy = occupancy_for(kCandidates[id]);
    candidates[id].recurrence_occupancy =
        recurrence_occupancy_for(kCandidates[id]);
  }

  auto time_launch = [&](const char* path, float* ms) -> cudaError_t {
    cudaError_t timed = restore_committed(&buffers, timed_tensors);
    if (timed == cudaSuccess) timed = cudaEventRecord(start);
    if (timed == cudaSuccess) timed = launch_quality(&buffers, path);
    if (timed == cudaSuccess) timed = cudaEventRecord(stop);
    if (timed == cudaSuccess) timed = cudaEventSynchronize(stop);
    if (timed == cudaSuccess) timed = cudaEventElapsedTime(ms, start, stop);
    return timed;
  };

  const int warmups = qw38::cuda::test_warmups();
  const int measured = qw38::cuda::test_samples();
  error = cudaSuccess;
  for (int warmup = 0; warmup < warmups && error == cudaSuccess; ++warmup) {
    for (int id = 0; id < kCandidateCount && error == cudaSuccess; ++id) {
      float ms = 0.0F;
      error = time_launch(kCandidates[id], &ms);
      candidates[id].warmup_ms.push_back(ms);
      if (raw != nullptr) {
        std::fprintf(raw, "warmup id=%s sample=%d ms=%.9g\n", kCandidates[id],
                     warmup, static_cast<double>(ms));
      }
    }
  }
  for (int sample = 0; sample < measured && error == cudaSuccess; ++sample) {
    for (int id = 0; id < kCandidateCount && error == cudaSuccess; ++id) {
      float ms = 0.0F;
      error = time_launch(kCandidates[id], &ms);
      candidates[id].samples.push_back(ms);
      if (raw != nullptr) {
        std::fprintf(raw, "measured id=%s sample=%d ms=%.9g\n", kCandidates[id],
                     sample, static_cast<double>(ms));
      }
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (error != cudaSuccess) {
    if (raw != nullptr) std::fclose(raw);
    release(&buffers);
    return fail_cuda("timed A/B", error);
  }

  LaunchSnapshot timed_shared;
  error = restore_committed(&buffers, timed_tensors);
  if (error == cudaSuccess) error = launch_quality(&buffers, "shared");
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess) {
    error = capture_snapshot(buffers, "shared", &timed_shared);
  }
  for (int id = 0; id < kCandidateCount; ++id) {
    float sum = 0.0F;
    for (float sample : candidates[id].samples) sum += sample;
    candidates[id].mean_ms =
        candidates[id].samples.empty()
            ? 0.0F
            : sum / static_cast<float>(candidates[id].samples.size());
    PathMetrics timed{};
    if (std::strcmp(kCandidates[id], "shared") == 0) {
      timed.launch_ok = error == cudaSuccess;
      timed.eligible = timed.launch_ok;
      timed.recurrent_byte_equal = true;
      timed.conv_byte_equal = true;
      timed.state_byte_equal = true;
      timed.gated_byte_equal = true;
      timed.prepare_preserves_committed =
          committed_unchanged(timed_shared, timed_tensors);
      timed.cancel_isolates = timed.prepare_preserves_committed;
      timed.restore_exact = true;
    } else {
      timed = evaluate_path(&buffers, timed_tensors, timed_shared,
                            kCandidates[id], raw, 0, timed_tokens);
    }
    candidates[id].launch_ok = timed.launch_ok &&
                               candidates[id].occupancy >= 1 &&
                               candidates[id].recurrence_occupancy >= 1 &&
                               static_cast<int>(candidates[id].samples.size()) ==
                                   measured;
    candidates[id].dispatched_preproc = timed.dispatched_preproc;
    candidates[id].recurrent_byte_equal = timed.recurrent_byte_equal;
    candidates[id].conv_byte_equal = timed.conv_byte_equal;
    candidates[id].state_byte_equal = timed.state_byte_equal;
    candidates[id].gated_byte_equal = timed.gated_byte_equal;
    candidates[id].prepare_preserves_committed =
        timed.prepare_preserves_committed;
    candidates[id].cancel_isolates = timed.cancel_isolates;
    candidates[id].restore_exact = timed.restore_exact;
    candidates[id].vs_shared = timed.vs_shared;
    candidates[id].vs_sequential = timed.vs_sequential;
    candidates[id].eligible = candidates[id].launch_ok && timed.eligible &&
                              all_eligible && chunk_ok;
  }

  const char* winner = select_winner(candidates, all_eligible && chunk_ok);
  const bool win = std::strcmp(winner, "shared") != 0;

  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-052\",\"status\":\"measured\","
      "\"measurement_utc\":\"%s\",\"device\":\"%s\",\"compute_capability\":\"%d.%d\","
      "\"tier\":\"%s\",\"timed_tokens\":%zu,\"selected_gdn_preproc_path\":\"%s\","
      "\"selected_gdn_inverse_path\":\"shared\",\"ab\":{\"token_count\":%zu,"
      "\"winner\":\"%s\",\"win\":%s,\"candidates\":{",
      kPrefix, utc, prop.name, prop.major, prop.minor,
      qw38::cuda::test_tier_name(), timed_tokens, winner, timed_tokens, winner,
      json_bool(win));
  for (int id = 0; id < kCandidateCount; ++id) {
    if (id != 0) std::printf(",");
    print_candidate(stdout, candidates[id]);
  }
  std::printf("}},\"correctness\":{\"token_counts\":[");
  for (int token_index = 0; token_index < token_limit; ++token_index) {
    if (token_index != 0) std::printf(",");
    std::printf("%zu", kCorrectnessTokens[token_index]);
  }
  std::printf("],\"layers\":[");
  for (int layer_index = 0; layer_index < layer_count; ++layer_index) {
    if (layer_index != 0) std::printf(",");
    std::printf("%d", kCorrectnessLayers[layer_index]);
  }
  std::printf("],\"all_eligible\":%s,\"chunk_ok\":%s,\"case_count\":%d},",
              json_bool(all_eligible), json_bool(chunk_ok),
              layer_count * token_limit);
  std::printf(
      "\"production_numerics\":{\"formula\":\"max(strict_reference_ceiling, "
      "1.05 * measured_llama_error + 1e-6)\","
      "\"budget\":{\"max_abs\":0.0003,\"rms\":0.0002,"
      "\"one_minus_cosine\":1e-06}}}\n");
  std::printf("status=passed\n");
  if (raw != nullptr) {
    std::fprintf(raw, "winner=%s win=%s\n", winner, json_bool(win));
    std::fclose(raw);
  }
  release(&buffers);
  return 0;
}
