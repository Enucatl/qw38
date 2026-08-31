#include "gdn_step.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "gdn.h"

namespace {

int fail_cuda(const char* operation, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
  return 1;
}

bool host_chunk(const qw38::cuda::GdnConfig& config, std::size_t token_count,
                const std::vector<float>& input,
                const std::vector<float>& weights,
                const std::vector<float>& log_decay,
                const std::vector<float>& beta,
                const std::vector<float>& initial_convolution,
                const std::vector<float>& initial_recurrent,
                std::vector<float>* final_convolution,
                std::vector<float>* final_recurrent,
                std::vector<float>* convolution_output,
                std::vector<float>* recurrent_output) {
  const std::size_t channels =
      qw38::cuda::gdn_convolution_channels(config);
  const std::size_t output_values = qw38::cuda::gdn_output_values(config);
  const std::size_t query_count =
      static_cast<std::size_t>(config.key_heads) * config.key_width;
  const std::size_t value_count = output_values;
  *final_convolution = initial_convolution;
  *final_recurrent = initial_recurrent;
  convolution_output->resize(token_count * channels);
  recurrent_output->resize(token_count * output_values);
  const qw38::internal::GdnShape shape{config.key_heads, config.value_heads,
                                       config.key_width, config.value_width};
  for (std::size_t token = 0; token < token_count; ++token) {
    qw38::Status status = qw38::internal::causal_depthwise_conv_step(
        channels, config.convolution_width, input.data() + token * channels,
        channels, weights.data(), weights.size(), final_convolution->data(),
        final_convolution->size(),
        convolution_output->data() + token * channels, channels);
    if (!status.is_ok()) return false;
    const float* token_convolution =
        convolution_output->data() + token * channels;
    status = qw38::internal::gdn_recurrent_step_precomputed(
        shape, token_convolution, query_count,
        token_convolution + query_count, query_count,
        token_convolution + 2 * query_count, value_count,
        log_decay.data() + token * config.value_heads,
        beta.data() + token * config.value_heads, config.value_heads,
        final_recurrent->data(), final_recurrent->size(),
        recurrent_output->data() + token * output_values, output_values);
    if (!status.is_ok()) return false;
  }
  return true;
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
  std::uint64_t* frontier = nullptr;
};

void release(DeviceBuffers* buffers) {
  cudaFree(buffers->frontier);
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
}

int run_chunk(const char* name, const qw38::cuda::GdnConfig& config,
              std::size_t token_count) {
  const std::size_t channels =
      qw38::cuda::gdn_convolution_channels(config);
  const std::size_t convolution_values =
      qw38::cuda::gdn_convolution_values(config);
  const std::size_t recurrent_values =
      qw38::cuda::gdn_recurrent_values(config);
  const std::size_t output_values = qw38::cuda::gdn_output_values(config);
  std::vector<float> input(token_count * channels);
  std::vector<float> weights(convolution_values);
  std::vector<float> log_decay(token_count * config.value_heads);
  std::vector<float> beta(token_count * config.value_heads);
  std::vector<float> initial_convolution(convolution_values);
  std::vector<float> initial_recurrent(recurrent_values);
  for (std::size_t token = 0; token < token_count; ++token) {
    for (std::size_t channel = 0; channel < channels; ++channel) {
      input[token * channels + channel] =
          std::sin(static_cast<float>(channel) * 0.013F +
                   static_cast<float>(token) * 0.071F) *
          0.5F;
    }
    for (std::size_t head = 0; head < config.value_heads; ++head) {
      log_decay[token * config.value_heads + head] =
          -0.001F * static_cast<float>(1 + (token + head) % 31);
      beta[token * config.value_heads + head] =
          0.15F + 0.01F * static_cast<float>((token + head) % 23);
    }
  }
  for (std::size_t index = 0; index < convolution_values; ++index) {
    weights[index] =
        static_cast<float>(static_cast<int>(index % 9) - 4) * 0.03125F;
    initial_convolution[index] =
        static_cast<float>(static_cast<int>(index % 13) - 6) * 0.015625F;
  }
  for (std::size_t index = 0; index < recurrent_values; ++index) {
    initial_recurrent[index] =
        static_cast<float>(static_cast<int>(index % 23) - 11) * 0.0009765625F;
  }
  std::vector<float> expected_convolution;
  std::vector<float> expected_recurrent;
  std::vector<float> expected_convolution_output;
  std::vector<float> expected_recurrent_output;
  if (!host_chunk(config, token_count, input, weights, log_decay, beta,
                  initial_convolution, initial_recurrent,
                  &expected_convolution, &expected_recurrent,
                  &expected_convolution_output, &expected_recurrent_output)) {
    std::fprintf(stderr, "%s: scalar chunk failed\n", name);
    return 1;
  }

  DeviceBuffers device;
  cudaError_t error = cudaMalloc(&device.input, input.size() * sizeof(float));
#define QW38_ALLOC(field, count)                                              \
  if (error == cudaSuccess)                                                   \
  error = cudaMalloc(&device.field, (count) * sizeof(*device.field))
  QW38_ALLOC(weights, weights.size());
  QW38_ALLOC(log_decay, log_decay.size());
  QW38_ALLOC(beta, beta.size());
  QW38_ALLOC(committed_convolution, convolution_values);
  QW38_ALLOC(committed_recurrent, recurrent_values);
  QW38_ALLOC(candidate_convolution, convolution_values);
  QW38_ALLOC(candidate_recurrent, recurrent_values);
  QW38_ALLOC(convolution_output, expected_convolution_output.size());
  QW38_ALLOC(recurrent_output, expected_recurrent_output.size());
  QW38_ALLOC(frontier, 1);
#undef QW38_ALLOC
  if (error != cudaSuccess) return fail_cuda("chunk cudaMalloc", error);
#define QW38_COPY(field, source)                                              \
  if (error == cudaSuccess)                                                   \
  error = cudaMemcpy(device.field, (source).data(),                           \
                     (source).size() * sizeof((source)[0]),                   \
                     cudaMemcpyHostToDevice)
  QW38_COPY(input, input);
  QW38_COPY(weights, weights);
  QW38_COPY(log_decay, log_decay);
  QW38_COPY(beta, beta);
  QW38_COPY(committed_convolution, initial_convolution);
  QW38_COPY(committed_recurrent, initial_recurrent);
#undef QW38_COPY
  const std::uint64_t initial_frontier = 7;
  if (error == cudaSuccess) {
    error = cudaMemcpy(device.frontier, &initial_frontier,
                       sizeof(initial_frontier), cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("chunk cudaMemcpy H2D", error);
  const qw38::cuda::GdnState committed{device.committed_convolution,
                                        device.committed_recurrent};
  const qw38::cuda::GdnState candidate{device.candidate_convolution,
                                        device.candidate_recurrent};

  for (int warmup = 0; warmup < 3 && error == cudaSuccess; ++warmup) {
    error = qw38::cuda::launch_gdn_prepare_chunk(
        config, device.input, device.weights, device.log_decay, device.beta,
        token_count, committed, candidate, device.convolution_output,
        device.recurrent_output, nullptr);
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (error == cudaSuccess) error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error == cudaSuccess) error = cudaEventRecord(start);
  for (int sample = 0; sample < 30 && error == cudaSuccess; ++sample) {
    error = qw38::cuda::launch_gdn_prepare_chunk(
        config, device.input, device.weights, device.log_decay, device.beta,
        token_count, committed, candidate, device.convolution_output,
        device.recurrent_output, nullptr);
  }
  if (error == cudaSuccess) error = cudaEventRecord(stop);
  if (error == cudaSuccess) error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) return fail_cuda("chunk prepare", error);
  float milliseconds = 0.0F;
  error = cudaEventElapsedTime(&milliseconds, start, stop);
  if (error != cudaSuccess) return fail_cuda("chunk timing", error);

  std::vector<float> chunk_convolution(convolution_values);
  std::vector<float> chunk_recurrent(recurrent_values);
  std::vector<float> chunk_convolution_output(expected_convolution_output.size());
  std::vector<float> chunk_recurrent_output(expected_recurrent_output.size());
#define QW38_READ(destination, field)                                         \
  if (error == cudaSuccess)                                                   \
  error = cudaMemcpy((destination).data(), device.field,                      \
                     (destination).size() * sizeof((destination)[0]),         \
                     cudaMemcpyDeviceToHost)
  QW38_READ(chunk_convolution, candidate_convolution);
  QW38_READ(chunk_recurrent, candidate_recurrent);
  QW38_READ(chunk_convolution_output, convolution_output);
  QW38_READ(chunk_recurrent_output, recurrent_output);
  std::vector<float> unchanged_convolution(convolution_values);
  std::vector<float> unchanged_recurrent(recurrent_values);
  QW38_READ(unchanged_convolution, committed_convolution);
  QW38_READ(unchanged_recurrent, committed_recurrent);
#undef QW38_READ
  std::uint64_t frontier = 0;
  if (error == cudaSuccess) {
    error = cudaMemcpy(&frontier, device.frontier, sizeof(frontier),
                       cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("chunk read", error);
  const bool prepare_atomic = unchanged_convolution == initial_convolution &&
                              unchanged_recurrent == initial_recurrent &&
                              frontier == initial_frontier;

  error = cudaMemcpy(device.committed_convolution, initial_convolution.data(),
                     convolution_values * sizeof(float),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device.committed_recurrent, initial_recurrent.data(),
                       recurrent_values * sizeof(float),
                       cudaMemcpyHostToDevice);
  }
  for (std::size_t token = 0; token < token_count && error == cudaSuccess;
       ++token) {
    error = qw38::cuda::launch_gdn_prepare(
        config, device.input + token * channels, device.weights,
        device.log_decay + token * config.value_heads,
        device.beta + token * config.value_heads, committed, candidate,
        device.convolution_output + token * channels,
        device.recurrent_output + token * output_values, nullptr);
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_gdn_commit(
          config, candidate, committed, initial_frontier + token + 1,
          device.frontier, nullptr);
    }
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("tokenwise GDN", error);
  std::vector<float> token_convolution(convolution_values);
  std::vector<float> token_recurrent(recurrent_values);
  std::vector<float> token_convolution_output(expected_convolution_output.size());
  std::vector<float> token_recurrent_output(expected_recurrent_output.size());
#define QW38_READ(destination, field)                                         \
  if (error == cudaSuccess)                                                   \
  error = cudaMemcpy((destination).data(), device.field,                      \
                     (destination).size() * sizeof((destination)[0]),         \
                     cudaMemcpyDeviceToHost)
  QW38_READ(token_convolution, committed_convolution);
  QW38_READ(token_recurrent, committed_recurrent);
  QW38_READ(token_convolution_output, convolution_output);
  QW38_READ(token_recurrent_output, recurrent_output);
#undef QW38_READ
  if (error != cudaSuccess) return fail_cuda("tokenwise read", error);
  const bool tokenwise_equal =
      chunk_convolution == token_convolution &&
      chunk_recurrent == token_recurrent &&
      chunk_convolution_output == token_convolution_output &&
      chunk_recurrent_output == token_recurrent_output;

  float maximum_absolute = 0.0F;
  double squared = 0.0;
  std::size_t metric_count = 0;
  std::size_t nonfinite = 0;
  auto compare = [&](const std::vector<float>& actual,
                     const std::vector<float>& expected) {
    for (std::size_t index = 0; index < actual.size(); ++index) {
      if (!std::isfinite(actual[index])) ++nonfinite;
      const float difference = std::fabs(actual[index] - expected[index]);
      maximum_absolute = std::max(maximum_absolute, difference);
      squared += static_cast<double>(difference) * difference;
      ++metric_count;
    }
  };
  compare(chunk_convolution, expected_convolution);
  compare(chunk_recurrent, expected_recurrent);
  compare(chunk_convolution_output, expected_convolution_output);
  compare(chunk_recurrent_output, expected_recurrent_output);
  const float rms = static_cast<float>(
      std::sqrt(squared / static_cast<double>(metric_count)));
  std::printf("gdn_chunk=%s tokens=%zu windows=%zu max_abs=%.9g rms=%.9g "
              "nonfinite=%zu prepare_atomic=%s tokenwise_equal=%s "
              "mean_ms=%.9g\n",
              name, token_count, (token_count + 63) / 64, maximum_absolute, rms,
              nonfinite, prepare_atomic ? "true" : "false",
              tokenwise_equal ? "true" : "false", milliseconds / 30.0F);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  release(&device);
  return prepare_atomic && tokenwise_equal && nonfinite == 0 &&
                 maximum_absolute <= 5.0e-8F && rms <= 5.0e-9F
             ? 0
             : 1;
}

}  // namespace

int main() {
  const qw38::cuda::GdnConfig small{2, 6, 8, 8, 4};
  const qw38::cuda::GdnConfig production{16, 48, 128, 128, 4};
  if (run_chunk("small_3", small, 3) != 0 ||
      run_chunk("small_64", small, 64) != 0 ||
      run_chunk("small_65", small, 65) != 0 ||
      run_chunk("small_129", small, 129) != 0 ||
      run_chunk("production_65", production, 65) != 0) {
    return 1;
  }
  std::printf("status=passed\n");
  return 0;
}
