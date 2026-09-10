#include "gdn_step.h"

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

constexpr char kPrefix[] = "QW38_GDN_SHARED_INVERSE_AB_RESULT=";
constexpr const char* kCandidates[] = {"repeated", "shared"};
constexpr int kCandidateCount = 2;
constexpr int kWarmups = 3;
constexpr int kMeasured = 30;
constexpr std::size_t kCorrectnessTokens[] = {1, 2, 3, 4, 63, 64, 65, 512, 2048,
                                              4096};
constexpr int kCorrectnessTokenCount = 10;
constexpr int kCorrectnessLayers[] = {0, 1, 62};
constexpr int kCorrectnessLayerCount = 3;
constexpr qw38::cuda::GdnConfig kConfig{16, 48, 128, 128, 4};
constexpr float kMaxAbs = 5.0e-8F;
constexpr float kMaxRms = 5.0e-9F;
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

struct Envelope {
  float max_abs = 0.0F;
  float rms = 0.0F;
  std::size_t nonfinite = 0;
};

bool envelope_ok(const Envelope& metrics) {
  return metrics.nonfinite == 0 && metrics.max_abs <= kMaxAbs &&
         metrics.rms <= kMaxRms;
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
    const float error = std::fabs(a - e);
    metrics.max_abs = std::max(metrics.max_abs, error);
    squared += static_cast<double>(error) * error;
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
  float* inverse_scratch = nullptr;
  float* inverse_capture = nullptr;
  __nv_bfloat16* gated = nullptr;
  std::uint64_t* frontier = nullptr;
  std::size_t token_count = 0;
  std::size_t inverse_floats = 0;
};

void release(DeviceBuffers* buffers) {
  cudaFree(buffers->frontier);
  cudaFree(buffers->gated);
  cudaFree(buffers->inverse_capture);
  cudaFree(buffers->inverse_scratch);
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
  buffers->inverse_floats =
      qw38::cuda::gdn_shared_inverse_floats(token_count, kConfig.key_heads);
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
  QW38_ALLOC(inverse_scratch, buffers->inverse_floats);
  QW38_ALLOC(inverse_capture, buffers->inverse_floats);
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

cudaError_t launch_quality(DeviceBuffers* buffers, const char* inverse_path) {
  std::vector<float> sentinel(buffers->inverse_floats, kSentinel);
  cudaError_t error =
      cudaMemcpy(buffers->inverse_scratch, sentinel.data(),
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
  return qw38::cuda::launch_gdn_quality_fused(
      kConfig, buffers->input, buffers->weights, buffers->log_decay,
      buffers->beta, buffers->token_count, committed_state(*buffers),
      candidate_state(*buffers), buffers->convolution_output,
      buffers->recurrent_output, buffers->gate, buffers->norm, buffers->gated,
      nullptr, true, "off", inverse_path, buffers->inverse_scratch,
      buffers->inverse_floats);
}

cudaError_t launch_sequential(DeviceBuffers* buffers) {
  return qw38::cuda::launch_gdn_prepare_chunk_tiled(
      kConfig, buffers->input, buffers->weights, buffers->log_decay,
      buffers->beta, buffers->token_count, committed_state(*buffers),
      candidate_state(*buffers), buffers->convolution_output,
      buffers->recurrent_output, nullptr, qw38::cuda::GdnScanPath::kSequentialWindows,
      nullptr, 0);
}

template <typename T>
cudaError_t download(const T* device, std::size_t count, std::vector<T>* host) {
  host->assign(count, T{});
  return cudaMemcpy(host->data(), device, count * sizeof(T),
                    cudaMemcpyDeviceToHost);
}

struct LaunchSnapshot {
  std::vector<float> inverses;
  std::vector<float> recurrent;
  std::vector<float> candidate_convolution;
  std::vector<float> candidate_recurrent;
  std::vector<float> convolution;
  std::vector<__nv_bfloat16> gated;
  std::vector<float> committed_convolution;
  std::vector<float> committed_recurrent;
  std::uint64_t frontier = 0;
  bool dispatched_shared = false;
  bool launch_ok = false;
};

cudaError_t capture_snapshot(const DeviceBuffers& buffers,
                             const char* inverse_path, LaunchSnapshot* snap) {
  const std::size_t channels = qw38::cuda::gdn_convolution_channels(kConfig);
  const std::size_t convolution_values =
      qw38::cuda::gdn_convolution_values(kConfig);
  const std::size_t recurrent_values = qw38::cuda::gdn_recurrent_values(kConfig);
  const std::size_t output_values = qw38::cuda::gdn_output_values(kConfig);
  const std::size_t gate_values =
      buffers.token_count * kConfig.value_heads * kConfig.value_width;
  cudaError_t error = download(buffers.inverse_scratch, buffers.inverse_floats,
                               &snap->inverses);
  if (error == cudaSuccess) {
    error = download(buffers.recurrent_output,
                     buffers.token_count * output_values, &snap->recurrent);
  }
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
    error = cudaMemcpy(&snap->frontier, buffers.frontier, sizeof(snap->frontier),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess && std::strcmp(inverse_path, "repeated") == 0) {
    error = qw38::cuda::launch_gdn_shared_inverses(
        kConfig, buffers.convolution_output, buffers.token_count,
        buffers.inverse_capture, nullptr);
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error == cudaSuccess) {
      error = download(buffers.inverse_capture, buffers.inverse_floats,
                       &snap->inverses);
    }
  }
  snap->dispatched_shared = false;
  if (error == cudaSuccess && std::strcmp(inverse_path, "shared") == 0 &&
      buffers.token_count >= 2) {
    for (float value : snap->inverses) {
      if (value != kSentinel) {
        snap->dispatched_shared = true;
        break;
      }
    }
  }
  return error;
}

bool committed_unchanged(const LaunchSnapshot& snap, const HostTensors& tensors) {
  return snap.committed_convolution == tensors.committed_convolution &&
         snap.committed_recurrent == tensors.committed_recurrent &&
         snap.frontier == kFrontier;
}

struct CaseMetrics {
  bool launch_ok = false;
  bool dispatched_shared = false;
  bool inverses_byte_equal = false;
  bool recurrent_byte_equal = false;
  bool conv_byte_equal = false;
  bool state_byte_equal = false;
  bool gated_byte_equal = false;
  bool prepare_preserves_committed = false;
  bool cancel_isolates = false;
  Envelope vs_sequential;
  bool eligible = false;
};

CaseMetrics evaluate_case(DeviceBuffers* buffers, const HostTensors& tensors,
                          FILE* raw, int layer, std::size_t token_count) {
  CaseMetrics metrics;
  LaunchSnapshot repeated;
  LaunchSnapshot shared;
  cudaError_t error = restore_committed(buffers, tensors);
  if (error == cudaSuccess) error = launch_quality(buffers, "repeated");
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess) error = capture_snapshot(*buffers, "repeated", &repeated);
  repeated.launch_ok = error == cudaSuccess;

  if (error == cudaSuccess) error = restore_committed(buffers, tensors);
  if (error == cudaSuccess) error = launch_quality(buffers, "shared");
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess) error = capture_snapshot(*buffers, "shared", &shared);
  shared.launch_ok = error == cudaSuccess;

  LaunchSnapshot sequential;
  if (error == cudaSuccess) error = restore_committed(buffers, tensors);
  if (error == cudaSuccess) error = launch_sequential(buffers);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess) {
    error = capture_snapshot(*buffers, "repeated", &sequential);
  }

  metrics.launch_ok = repeated.launch_ok && shared.launch_ok && error == cudaSuccess;
  metrics.dispatched_shared = shared.dispatched_shared;
  if (metrics.launch_ok) {
    metrics.inverses_byte_equal =
        token_count == 1 ? true : repeated.inverses == shared.inverses;
    metrics.recurrent_byte_equal = repeated.recurrent == shared.recurrent;
    metrics.conv_byte_equal = repeated.convolution == shared.convolution &&
                              repeated.candidate_convolution ==
                                  shared.candidate_convolution;
    metrics.state_byte_equal =
        repeated.candidate_recurrent == shared.candidate_recurrent;
    metrics.gated_byte_equal =
        std::memcmp(repeated.gated.data(), shared.gated.data(),
                    repeated.gated.size() * sizeof(__nv_bfloat16)) == 0;
    metrics.prepare_preserves_committed =
        committed_unchanged(repeated, tensors) &&
        committed_unchanged(shared, tensors);
    metrics.cancel_isolates = metrics.prepare_preserves_committed;
    Envelope rec = compare_values(shared.recurrent, sequential.recurrent);
    Envelope cconv = compare_values(shared.candidate_convolution,
                                    sequential.candidate_convolution);
    Envelope crec = compare_values(shared.candidate_recurrent,
                                   sequential.candidate_recurrent);
    metrics.vs_sequential.max_abs = std::max(
        rec.max_abs, std::max(cconv.max_abs, crec.max_abs));
    metrics.vs_sequential.nonfinite =
        rec.nonfinite + cconv.nonfinite + crec.nonfinite;
    const double count = static_cast<double>(shared.recurrent.size() +
                                             shared.candidate_convolution.size() +
                                             shared.candidate_recurrent.size());
    const double rec_ss = static_cast<double>(rec.rms) * rec.rms *
                          static_cast<double>(shared.recurrent.size());
    const double cconv_ss =
        static_cast<double>(cconv.rms) * cconv.rms *
        static_cast<double>(shared.candidate_convolution.size());
    const double crec_ss = static_cast<double>(crec.rms) * crec.rms *
                           static_cast<double>(shared.candidate_recurrent.size());
    metrics.vs_sequential.rms =
        count > 0.0 ? static_cast<float>(std::sqrt((rec_ss + cconv_ss + crec_ss) /
                                                   count))
                    : 0.0F;
    const bool token1 = token_count == 1;
    const bool dispatch_ok =
        token1 ? !metrics.dispatched_shared : metrics.dispatched_shared;
    metrics.eligible =
        metrics.launch_ok && dispatch_ok && metrics.inverses_byte_equal &&
        metrics.recurrent_byte_equal && metrics.conv_byte_equal &&
        metrics.state_byte_equal && metrics.gated_byte_equal &&
        metrics.prepare_preserves_committed && metrics.cancel_isolates &&
        envelope_ok(metrics.vs_sequential) &&
        qw38::cuda::gdn_shared_inverse_occupancy() >= 1 &&
        qw38::cuda::gdn_shared_recurrence_occupancy() >= 1;
  }
  if (raw != nullptr) {
    std::fprintf(raw,
                 "correctness layer=%d tokens=%zu launch_ok=%s dispatched_shared=%s "
                 "inverses_eq=%s recurrent_eq=%s conv_eq=%s state_eq=%s gated_eq=%s "
                 "committed=%s seq_max_abs=%.9g seq_rms=%.9g seq_nonfinite=%zu "
                 "eligible=%s\n",
                 layer, token_count, json_bool(metrics.launch_ok),
                 json_bool(metrics.dispatched_shared),
                 json_bool(metrics.inverses_byte_equal),
                 json_bool(metrics.recurrent_byte_equal),
                 json_bool(metrics.conv_byte_equal),
                 json_bool(metrics.state_byte_equal),
                 json_bool(metrics.gated_byte_equal),
                 json_bool(metrics.prepare_preserves_committed),
                 static_cast<double>(metrics.vs_sequential.max_abs),
                 static_cast<double>(metrics.vs_sequential.rms),
                 metrics.vs_sequential.nonfinite, json_bool(metrics.eligible));
  }
  return metrics;
}

void print_case(FILE* file, const CaseMetrics& metrics) {
  std::fprintf(file,
               "\"launch_ok\":%s,\"dispatched_shared\":%s,"
               "\"inverses_byte_equal\":%s,\"recurrent_byte_equal\":%s,"
               "\"conv_byte_equal\":%s,\"state_byte_equal\":%s,"
               "\"gated_byte_equal\":%s,\"prepare_preserves_committed\":%s,"
               "\"cancel_isolates\":%s,\"eligible\":%s,",
               json_bool(metrics.launch_ok), json_bool(metrics.dispatched_shared),
               json_bool(metrics.inverses_byte_equal),
               json_bool(metrics.recurrent_byte_equal),
               json_bool(metrics.conv_byte_equal),
               json_bool(metrics.state_byte_equal),
               json_bool(metrics.gated_byte_equal),
               json_bool(metrics.prepare_preserves_committed),
               json_bool(metrics.cancel_isolates), json_bool(metrics.eligible));
  print_envelope(file, "vs_sequential", metrics.vs_sequential);
}

struct CandidateResult {
  const char* id = "repeated";
  int occupancy = 0;
  int recurrence_occupancy = 0;
  bool launch_ok = false;
  bool eligible = false;
  bool inverses_byte_equal = false;
  bool recurrent_byte_equal = false;
  bool conv_byte_equal = false;
  bool state_byte_equal = false;
  bool gated_byte_equal = false;
  bool prepare_preserves_committed = false;
  bool cancel_isolates = false;
  bool dispatched_shared = false;
  Envelope vs_sequential;
  std::vector<float> warmup_ms;
  std::vector<float> samples;
  float mean_ms = 0.0F;
};

void print_candidate(FILE* file, const CandidateResult& result) {
  std::fprintf(file, "\"%s\":{\"id\":\"%s\",\"mean_ms\":%.9g,", result.id,
               result.id, static_cast<double>(result.mean_ms));
  print_array(file, "samples", result.samples);
  std::fprintf(file, ",");
  print_array(file, "warmup_ms", result.warmup_ms);
  std::fprintf(file,
               ",\"occupancy\":%d,\"recurrence_occupancy\":%d,\"launch_ok\":%s,"
               "\"eligible\":%s,\"dispatched_shared\":%s,"
               "\"inverses_byte_equal\":%s,\"recurrent_byte_equal\":%s,"
               "\"conv_byte_equal\":%s,\"state_byte_equal\":%s,"
               "\"gated_byte_equal\":%s,\"prepare_preserves_committed\":%s,"
               "\"cancel_isolates\":%s,",
               result.occupancy, result.recurrence_occupancy,
               json_bool(result.launch_ok), json_bool(result.eligible),
               json_bool(result.dispatched_shared),
               json_bool(result.inverses_byte_equal),
               json_bool(result.recurrent_byte_equal),
               json_bool(result.conv_byte_equal),
               json_bool(result.state_byte_equal),
               json_bool(result.gated_byte_equal),
               json_bool(result.prepare_preserves_committed),
               json_bool(result.cancel_isolates));
  print_envelope(file, "vs_sequential", result.vs_sequential);
  std::fprintf(file, "}");
}

}  // namespace

int main(int argc, char** argv) {
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

  DeviceBuffers buffers;
  bool all_eligible = true;
  std::vector<CaseMetrics> cases(static_cast<std::size_t>(kCorrectnessLayerCount *
                                                          kCorrectnessTokenCount));
  for (int layer_index = 0; layer_index < kCorrectnessLayerCount; ++layer_index) {
    const int layer = kCorrectnessLayers[layer_index];
    for (int token_index = 0; token_index < kCorrectnessTokenCount;
         ++token_index) {
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
      const std::size_t slot =
          static_cast<std::size_t>(layer_index * kCorrectnessTokenCount +
                                   token_index);
      cases[slot] = evaluate_case(&buffers, tensors, raw, layer, token_count);
      all_eligible = all_eligible && cases[slot].eligible;
    }
  }

  constexpr std::size_t kTimedTokens = 4096;
  HostTensors timed_tensors;
  fill_synthetic(&timed_tensors, kTimedTokens, 0);
  error = allocate(&buffers, kTimedTokens);
  if (error == cudaSuccess) error = stage_inputs(&buffers, timed_tensors);
  if (error != cudaSuccess) {
    if (raw != nullptr) std::fclose(raw);
    release(&buffers);
    return fail_cuda("allocate timed", error);
  }
  CaseMetrics timed_correctness =
      evaluate_case(&buffers, timed_tensors, raw, 0, kTimedTokens);
  all_eligible = all_eligible && timed_correctness.eligible;

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error != cudaSuccess) {
    if (raw != nullptr) std::fclose(raw);
    release(&buffers);
    return fail_cuda("events", error);
  }

  CandidateResult candidates[2];
  candidates[0].id = "repeated";
  candidates[1].id = "shared";
  candidates[0].occupancy = qw38::cuda::gdn_fuse_occupancy("off");
  candidates[0].recurrence_occupancy = candidates[0].occupancy;
  candidates[1].occupancy = qw38::cuda::gdn_shared_inverse_occupancy();
  candidates[1].recurrence_occupancy =
      qw38::cuda::gdn_shared_recurrence_occupancy();
  for (int id = 0; id < kCandidateCount; ++id) {
    candidates[id].inverses_byte_equal = timed_correctness.inverses_byte_equal;
    candidates[id].recurrent_byte_equal = timed_correctness.recurrent_byte_equal;
    candidates[id].conv_byte_equal = timed_correctness.conv_byte_equal;
    candidates[id].state_byte_equal = timed_correctness.state_byte_equal;
    candidates[id].gated_byte_equal = timed_correctness.gated_byte_equal;
    candidates[id].prepare_preserves_committed =
        timed_correctness.prepare_preserves_committed;
    candidates[id].cancel_isolates = timed_correctness.cancel_isolates;
    candidates[id].vs_sequential = timed_correctness.vs_sequential;
    candidates[id].dispatched_shared =
        id == 1 ? timed_correctness.dispatched_shared : false;
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

  error = cudaSuccess;
  for (int warmup = 0; warmup < kWarmups && error == cudaSuccess; ++warmup) {
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
  for (int sample = 0; sample < kMeasured && error == cudaSuccess; ++sample) {
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

  for (int id = 0; id < kCandidateCount; ++id) {
    float sum = 0.0F;
    for (float sample : candidates[id].samples) sum += sample;
    candidates[id].mean_ms =
        candidates[id].samples.empty()
            ? 0.0F
            : sum / static_cast<float>(candidates[id].samples.size());
    candidates[id].launch_ok = timed_correctness.launch_ok &&
                               candidates[id].occupancy >= 1 &&
                               candidates[id].recurrence_occupancy >= 1 &&
                               candidates[id].samples.size() == kMeasured;
    candidates[id].eligible =
        candidates[id].launch_ok && timed_correctness.eligible && all_eligible;
    if (id == 1) {
      candidates[id].eligible =
          candidates[id].eligible && timed_correctness.dispatched_shared;
    }
  }

  const bool shared_win =
      candidates[1].eligible && candidates[0].eligible &&
      candidates[1].mean_ms < candidates[0].mean_ms;
  const char* winner = shared_win ? "shared" : "repeated";

  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-040\",\"status\":\"measured\","
      "\"measurement_utc\":\"%s\",\"device\":\"%s\",\"compute_capability\":\"%d.%d\","
      "\"selected_gdn_inverse_path\":\"%s\",\"ab_p4096\":{\"token_count\":4096,"
      "\"winner\":\"%s\",\"win\":%s,\"candidates\":{",
      kPrefix, utc, prop.name, prop.major, prop.minor, winner, winner,
      json_bool(shared_win));
  print_candidate(stdout, candidates[0]);
  std::printf(",");
  print_candidate(stdout, candidates[1]);
  std::printf("}},\"correctness\":{\"token_counts\":[1,2,3,4,63,64,65,512,2048,4096],"
              "\"layers\":[0,1,62],\"all_eligible\":%s,\"cases\":{",
              json_bool(all_eligible));
  bool first = true;
  for (int layer_index = 0; layer_index < kCorrectnessLayerCount; ++layer_index) {
    for (int token_index = 0; token_index < kCorrectnessTokenCount;
         ++token_index) {
      if (!first) std::printf(",");
      first = false;
      const int layer = kCorrectnessLayers[layer_index];
      const std::size_t token_count = kCorrectnessTokens[token_index];
      const std::size_t slot =
          static_cast<std::size_t>(layer_index * kCorrectnessTokenCount +
                                   token_index);
      std::printf("\"%d:%zu\":{\"layer\":%d,\"token_count\":%zu,", layer,
                  token_count, layer, token_count);
      print_case(stdout, cases[slot]);
      std::printf("}");
    }
  }
  std::printf("}}}\n");
  std::printf("status=passed\n");
  if (raw != nullptr) {
    std::fprintf(raw, "winner=%s win=%s repeated_mean=%.9g shared_mean=%.9g\n",
                 winner, json_bool(shared_win),
                 static_cast<double>(candidates[0].mean_ms),
                 static_cast<double>(candidates[1].mean_ms));
    std::fclose(raw);
  }
  release(&buffers);
  return 0;
}
