#include "gdn_step.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "gdn.h"

namespace {

int fail_cuda(const char* operation, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
  return 1;
}

struct Metrics {
  float maximum_absolute = 0.0F;
  double squared = 0.0;
  std::size_t count = 0;
  std::size_t nonfinite = 0;
};

void add_metrics(const std::vector<float>& actual,
                 const std::vector<float>& expected, Metrics* metrics) {
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index])) ++metrics->nonfinite;
    const float error = std::fabs(actual[index] - expected[index]);
    metrics->maximum_absolute = std::max(metrics->maximum_absolute, error);
    metrics->squared += static_cast<double>(error) * error;
    ++metrics->count;
  }
}

bool host_reference(const qw38::cuda::GdnConfig& config,
                    const std::vector<float>& convolution_input,
                    const std::vector<float>& convolution_weights,
                    const std::vector<float>& log_decay,
                    const std::vector<float>& beta,
                    const std::vector<float>& committed_convolution,
                    const std::vector<float>& committed_recurrent,
                    std::vector<float>* candidate_convolution,
                    std::vector<float>* candidate_recurrent,
                    std::vector<float>* convolution_output,
                    std::vector<float>* recurrent_output) {
  *candidate_convolution = committed_convolution;
  *candidate_recurrent = committed_recurrent;
  convolution_output->resize(qw38::cuda::gdn_convolution_channels(config));
  recurrent_output->resize(qw38::cuda::gdn_output_values(config));
  qw38::Status status = qw38::internal::causal_depthwise_conv_step(
      convolution_output->size(), config.convolution_width,
      convolution_input.data(), convolution_input.size(),
      convolution_weights.data(), convolution_weights.size(),
      candidate_convolution->data(), candidate_convolution->size(),
      convolution_output->data(), convolution_output->size());
  if (!status.is_ok()) return false;
  const std::size_t query_count =
      static_cast<std::size_t>(config.key_heads) * config.key_width;
  const std::size_t value_count =
      static_cast<std::size_t>(config.value_heads) * config.value_width;
  const qw38::internal::GdnShape shape{config.key_heads, config.value_heads,
                                       config.key_width, config.value_width};
  status = qw38::internal::gdn_recurrent_step_precomputed(
      shape, convolution_output->data(), query_count,
      convolution_output->data() + query_count, query_count,
      convolution_output->data() + 2 * query_count, value_count,
      log_decay.data(), beta.data(), beta.size(), candidate_recurrent->data(),
      candidate_recurrent->size(), recurrent_output->data(),
      recurrent_output->size());
  return status.is_ok();
}

int run_case(const char* name, const qw38::cuda::GdnConfig& config) {
  const std::size_t channels =
      qw38::cuda::gdn_convolution_channels(config);
  const std::size_t convolution_values =
      qw38::cuda::gdn_convolution_values(config);
  const std::size_t recurrent_values =
      qw38::cuda::gdn_recurrent_values(config);
  const std::size_t output_values = qw38::cuda::gdn_output_values(config);
  std::vector<float> convolution_input(channels);
  std::vector<float> convolution_weights(convolution_values);
  std::vector<float> log_decay(config.value_heads);
  std::vector<float> beta(config.value_heads);
  std::vector<float> committed_convolution(convolution_values);
  std::vector<float> committed_recurrent(recurrent_values);
  for (std::size_t index = 0; index < channels; ++index) {
    convolution_input[index] =
        std::sin(static_cast<float>(index) * 0.013F) * 0.5F;
  }
  for (std::size_t index = 0; index < convolution_values; ++index) {
    convolution_weights[index] =
        static_cast<float>(static_cast<int>(index % 9) - 4) * 0.03125F;
    committed_convolution[index] =
        static_cast<float>(static_cast<int>(index % 13) - 6) * 0.015625F;
  }
  for (std::size_t head = 0; head < config.value_heads; ++head) {
    log_decay[head] = -0.0025F * static_cast<float>(head + 1);
    beta[head] = 0.2F + 0.01F * static_cast<float>(head % 17);
  }
  for (std::size_t index = 0; index < recurrent_values; ++index) {
    committed_recurrent[index] =
        static_cast<float>(static_cast<int>(index % 23) - 11) * 0.0009765625F;
  }
  std::vector<float> expected_convolution_state;
  std::vector<float> expected_recurrent_state;
  std::vector<float> expected_convolution_output;
  std::vector<float> expected_recurrent_output;
  if (!host_reference(config, convolution_input, convolution_weights, log_decay,
                      beta, committed_convolution, committed_recurrent,
                      &expected_convolution_state, &expected_recurrent_state,
                      &expected_convolution_output,
                      &expected_recurrent_output)) {
    std::fprintf(stderr, "%s: host reference failed\n", name);
    return 1;
  }

  float* device_input = nullptr;
  float* device_weights = nullptr;
  float* device_log_decay = nullptr;
  float* device_beta = nullptr;
  float* device_committed_convolution = nullptr;
  float* device_committed_recurrent = nullptr;
  float* device_candidate_convolution = nullptr;
  float* device_candidate_recurrent = nullptr;
  float* device_convolution_output = nullptr;
  float* device_recurrent_output = nullptr;
  std::uint64_t* device_frontier = nullptr;
  cudaError_t error = cudaMalloc(&device_input, channels * sizeof(float));
#define QW38_ALLOC(pointer, count)                                            \
  if (error == cudaSuccess) error = cudaMalloc(&(pointer), (count) * sizeof(*(pointer)))
  QW38_ALLOC(device_weights, convolution_values);
  QW38_ALLOC(device_log_decay, config.value_heads);
  QW38_ALLOC(device_beta, config.value_heads);
  QW38_ALLOC(device_committed_convolution, convolution_values);
  QW38_ALLOC(device_committed_recurrent, recurrent_values);
  QW38_ALLOC(device_candidate_convolution, convolution_values);
  QW38_ALLOC(device_candidate_recurrent, recurrent_values);
  QW38_ALLOC(device_convolution_output, channels);
  QW38_ALLOC(device_recurrent_output, output_values);
  QW38_ALLOC(device_frontier, 1);
#undef QW38_ALLOC
  if (error != cudaSuccess) return fail_cuda("GDN cudaMalloc", error);
#define QW38_COPY(pointer, source)                                            \
  if (error == cudaSuccess)                                                   \
  error = cudaMemcpy((pointer), (source).data(),                              \
                     (source).size() * sizeof((source)[0]), cudaMemcpyHostToDevice)
  QW38_COPY(device_input, convolution_input);
  QW38_COPY(device_weights, convolution_weights);
  QW38_COPY(device_log_decay, log_decay);
  QW38_COPY(device_beta, beta);
  QW38_COPY(device_committed_convolution, committed_convolution);
  QW38_COPY(device_committed_recurrent, committed_recurrent);
#undef QW38_COPY
  const std::uint64_t initial_frontier = 41;
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_frontier, &initial_frontier,
                       sizeof(initial_frontier), cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("GDN cudaMemcpy H2D", error);

  const qw38::cuda::GdnState committed{device_committed_convolution,
                                        device_committed_recurrent};
  const qw38::cuda::GdnState candidate{device_candidate_convolution,
                                        device_candidate_recurrent};
  const cudaError_t rejected = qw38::cuda::launch_gdn_prepare(
      config, device_input, device_weights, device_log_decay, device_beta,
      committed, committed, device_convolution_output, device_recurrent_output,
      nullptr);
  if (rejected != cudaErrorInvalidValue) {
    std::fprintf(stderr, "%s: aliased candidate was not rejected\n", name);
    return 1;
  }

  for (int warmup = 0; warmup < 3 && error == cudaSuccess; ++warmup) {
    error = qw38::cuda::launch_gdn_prepare(
        config, device_input, device_weights, device_log_decay, device_beta,
        committed, candidate, device_convolution_output,
        device_recurrent_output, nullptr);
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (error == cudaSuccess) error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error == cudaSuccess) error = cudaEventRecord(start);
  for (int sample = 0; sample < 30 && error == cudaSuccess; ++sample) {
    error = qw38::cuda::launch_gdn_prepare(
        config, device_input, device_weights, device_log_decay, device_beta,
        committed, candidate, device_convolution_output,
        device_recurrent_output, nullptr);
  }
  if (error == cudaSuccess) error = cudaEventRecord(stop);
  if (error == cudaSuccess) error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) return fail_cuda("GDN prepare", error);
  float milliseconds = 0.0F;
  error = cudaEventElapsedTime(&milliseconds, start, stop);
  if (error != cudaSuccess) return fail_cuda("GDN timing", error);

  std::vector<float> actual_committed_convolution(convolution_values);
  std::vector<float> actual_committed_recurrent(recurrent_values);
  std::vector<float> actual_candidate_convolution(convolution_values);
  std::vector<float> actual_candidate_recurrent(recurrent_values);
  std::vector<float> actual_convolution_output(channels);
  std::vector<float> actual_recurrent_output(output_values);
#define QW38_READ(destination, pointer)                                       \
  if (error == cudaSuccess)                                                   \
  error = cudaMemcpy((destination).data(), (pointer),                         \
                     (destination).size() * sizeof((destination)[0]),         \
                     cudaMemcpyDeviceToHost)
  QW38_READ(actual_committed_convolution, device_committed_convolution);
  QW38_READ(actual_committed_recurrent, device_committed_recurrent);
  QW38_READ(actual_candidate_convolution, device_candidate_convolution);
  QW38_READ(actual_candidate_recurrent, device_candidate_recurrent);
  QW38_READ(actual_convolution_output, device_convolution_output);
  QW38_READ(actual_recurrent_output, device_recurrent_output);
  std::uint64_t actual_frontier = 0;
  if (error == cudaSuccess) {
    error = cudaMemcpy(&actual_frontier, device_frontier,
                       sizeof(actual_frontier), cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("GDN cudaMemcpy D2H", error);
  const bool prepare_atomic =
      actual_committed_convolution == committed_convolution &&
      actual_committed_recurrent == committed_recurrent &&
      actual_frontier == initial_frontier;

  Metrics metrics;
  add_metrics(actual_candidate_convolution, expected_convolution_state,
              &metrics);
  add_metrics(actual_candidate_recurrent, expected_recurrent_state, &metrics);
  add_metrics(actual_convolution_output, expected_convolution_output, &metrics);
  add_metrics(actual_recurrent_output, expected_recurrent_output, &metrics);
  const float rms = static_cast<float>(
      std::sqrt(metrics.squared / static_cast<double>(metrics.count)));

  error = qw38::cuda::launch_gdn_commit(config, candidate, committed, 42,
                                         device_frontier, nullptr);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("GDN commit", error);
  QW38_READ(actual_committed_convolution, device_committed_convolution);
  QW38_READ(actual_committed_recurrent, device_committed_recurrent);
#undef QW38_READ
  if (error == cudaSuccess) {
    error = cudaMemcpy(&actual_frontier, device_frontier,
                       sizeof(actual_frontier), cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("GDN committed read", error);
  const bool commit_exact =
      actual_committed_convolution == actual_candidate_convolution &&
      actual_committed_recurrent == actual_candidate_recurrent &&
      actual_frontier == 42;

  std::printf("gdn_case=%s key_heads=%u value_heads=%u key_width=%u "
              "value_width=%u max_abs=%.9g rms=%.9g nonfinite=%zu "
              "prepare_atomic=%s commit_exact=%s mean_ms=%.9g\n",
              name, config.key_heads, config.value_heads, config.key_width,
              config.value_width, metrics.maximum_absolute, rms,
              metrics.nonfinite, prepare_atomic ? "true" : "false",
              commit_exact ? "true" : "false", milliseconds / 30.0F);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(device_frontier);
  cudaFree(device_recurrent_output);
  cudaFree(device_convolution_output);
  cudaFree(device_candidate_recurrent);
  cudaFree(device_candidate_convolution);
  cudaFree(device_committed_recurrent);
  cudaFree(device_committed_convolution);
  cudaFree(device_beta);
  cudaFree(device_log_decay);
  cudaFree(device_weights);
  cudaFree(device_input);
  return prepare_atomic && commit_exact && metrics.nonfinite == 0 &&
                 metrics.maximum_absolute <= 5.0e-8F && rms <= 5.0e-9F
             ? 0
             : 1;
}

}  // namespace

int main() {
  const qw38::cuda::GdnConfig invalid{2, 5, 8, 8, 4};
  if (qw38::cuda::gdn_recurrent_values(invalid) != 0) {
    std::fprintf(stderr, "invalid GDN shape was accepted\n");
    return 1;
  }
  const qw38::cuda::GdnConfig small{2, 6, 8, 8, 4};
  const qw38::cuda::GdnConfig production{16, 48, 128, 128, 4};
  if (run_case("small", small) != 0 || run_case("production", production) != 0) {
    return 1;
  }
  std::printf("status=passed\n");
  return 0;
}
