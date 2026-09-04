#include "gdn_step.h"

#include <algorithm>
#include <cmath>

namespace qw38::cuda {
namespace {

constexpr std::uint32_t kMaximumKeyHeads = 16;
constexpr std::uint32_t kMaximumValueHeads = 48;
constexpr std::uint32_t kMaximumWidth = 128;
constexpr std::uint32_t kMaximumConvolutionWidth = 4;
constexpr int kThreads = 128;
constexpr std::size_t kScanWindow = 64;
constexpr float kL2Epsilon = 1.0e-6F;

bool valid_config(const GdnConfig& config) noexcept {
  return config.key_heads > 0 && config.key_heads <= kMaximumKeyHeads &&
         config.value_heads > 0 && config.value_heads <= kMaximumValueHeads &&
         config.value_heads % config.key_heads == 0 && config.key_width > 0 &&
         config.key_width <= kMaximumWidth && config.value_width > 0 &&
         config.value_width <= kMaximumWidth &&
         config.convolution_width > 0 &&
         config.convolution_width <= kMaximumConvolutionWidth;
}

__global__ void prepare_convolution_window(
    const float* input, const float* weights, const float* source,
    float* candidate, float* output, std::size_t channels, std::uint32_t width,
    std::size_t token_count) {
  const std::size_t channel =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= channels) return;
  const std::size_t base = channel * width;
  float history[kMaximumConvolutionWidth];
  for (std::uint32_t index = 0; index < width; ++index) {
    history[index] = source[base + index];
  }
  for (std::size_t token = 0; token < token_count; ++token) {
    for (std::uint32_t index = 0; index + 1 < width; ++index) {
      history[index] = history[index + 1];
    }
    history[width - 1] = input[token * channels + channel];
    float convolution = 0.0F;
    for (std::uint32_t index = 0; index < width; ++index) {
      convolution = __fadd_rn(
          convolution, __fmul_rn(history[index], weights[base + index]));
    }
    output[token * channels + channel] =
        convolution / (1.0F + expf(-convolution));
  }
  for (std::uint32_t index = 0; index < width; ++index) {
    candidate[base + index] = history[index];
  }
}

__global__ void prepare_recurrence_window(
    GdnConfig config, const float* convolution_output,
    const float* log_decay, const float* beta, const float* source,
    float* candidate, float* output, std::size_t token_count,
    bool value_is_tiled) {
  const std::uint32_t value_head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (value_head >= config.value_heads || lane >= config.value_width) return;
  const std::uint32_t reuse = config.value_heads / config.key_heads;
  const std::uint32_t key_head = value_head / reuse;
  const std::size_t query_count =
      static_cast<std::size_t>(config.key_heads) * config.key_width;
  __shared__ float query_inverse;
  __shared__ float key_inverse;
  const std::size_t head_base =
      static_cast<std::size_t>(value_head) * config.key_width *
      config.value_width;
  const std::size_t channels = 2 * query_count +
                               static_cast<std::size_t>(config.value_heads) *
                                   config.value_width;
  for (std::size_t token = 0; token < token_count; ++token) {
    const float* token_convolution = convolution_output + token * channels;
    const float* query = token_convolution + key_head * config.key_width;
    const float* key = token_convolution + query_count +
                       key_head * config.key_width;
    const std::uint32_t replica = value_head % reuse;
    const std::uint32_t tiled_head = replica * config.key_heads + key_head;
    const std::uint32_t source_head = value_is_tiled ? tiled_head : value_head;
    const float* value = token_convolution + 2 * query_count +
                         source_head * config.value_width;
    if (lane == 0) {
      float query_squares = 0.0F;
      float key_squares = 0.0F;
      for (std::uint32_t index = 0; index < config.key_width; ++index) {
        query_squares = __fadd_rn(
            query_squares, __fmul_rn(query[index], query[index]));
        key_squares =
            __fadd_rn(key_squares, __fmul_rn(key[index], key[index]));
      }
      query_inverse =
          1.0F / sqrtf(query_squares + kL2Epsilon) /
          sqrtf(static_cast<float>(config.key_width));
      key_inverse = 1.0F / sqrtf(key_squares + kL2Epsilon);
    }
    __syncthreads();

    const float* current = token == 0 ? source : candidate;
    const float decay =
        expf(log_decay[token * config.value_heads + value_head]);
    float prediction = 0.0F;
    for (std::uint32_t key_lane = 0; key_lane < config.key_width; ++key_lane) {
      const std::size_t index =
          head_base + static_cast<std::size_t>(key_lane) * config.value_width +
          lane;
      const float decayed = __fmul_rn(current[index], decay);
      prediction = __fadd_rn(
          prediction,
          __fmul_rn(__fmul_rn(key[key_lane], key_inverse), decayed));
    }
    const float delta = __fmul_rn(
        value[lane] - prediction,
        beta[token * config.value_heads + value_head]);
    float result = 0.0F;
    for (std::uint32_t key_lane = 0; key_lane < config.key_width; ++key_lane) {
      const std::size_t index =
          head_base + static_cast<std::size_t>(key_lane) * config.value_width +
          lane;
      const float decayed = __fmul_rn(current[index], decay);
      const float updated = __fadd_rn(
          decayed,
          __fmul_rn(__fmul_rn(key[key_lane], key_inverse), delta));
      candidate[index] = updated;
      result = __fadd_rn(
          result,
          __fmul_rn(__fmul_rn(query[key_lane], query_inverse), updated));
    }
    output[(token * config.value_heads + value_head) * config.value_width +
           lane] = result;
    __syncthreads();
  }
}

__global__ void commit_state(const float* candidate_convolution,
                             float* committed_convolution,
                             std::size_t convolution_values,
                             const float* candidate_recurrent,
                             float* committed_recurrent,
                             std::size_t recurrent_values,
                             std::uint64_t new_frontier,
                             std::uint64_t* committed_frontier) {
  for (std::size_t index = threadIdx.x; index < convolution_values;
       index += blockDim.x) {
    committed_convolution[index] = candidate_convolution[index];
  }
  for (std::size_t index = threadIdx.x; index < recurrent_values;
       index += blockDim.x) {
    committed_recurrent[index] = candidate_recurrent[index];
  }
  __syncthreads();
  if (threadIdx.x == 0) *committed_frontier = new_frontier;
}

}  // namespace

namespace {

cudaError_t launch_gdn_prepare_chunk_layout(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, std::size_t token_count, const GdnState& committed,
    const GdnState& candidate, float* convolution_output,
    float* recurrent_output, cudaStream_t stream, bool value_is_tiled) noexcept;

}  // namespace

std::size_t gdn_convolution_channels(const GdnConfig& config) noexcept {
  if (!valid_config(config)) return 0;
  return 2 * static_cast<std::size_t>(config.key_heads) * config.key_width +
         static_cast<std::size_t>(config.value_heads) * config.value_width;
}

std::size_t gdn_convolution_values(const GdnConfig& config) noexcept {
  return gdn_convolution_channels(config) * config.convolution_width;
}

std::size_t gdn_recurrent_values(const GdnConfig& config) noexcept {
  if (!valid_config(config)) return 0;
  return static_cast<std::size_t>(config.value_heads) * config.key_width *
         config.value_width;
}

std::size_t gdn_output_values(const GdnConfig& config) noexcept {
  if (!valid_config(config)) return 0;
  return static_cast<std::size_t>(config.value_heads) * config.value_width;
}

cudaError_t launch_gdn_prepare(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, const GdnState& committed, const GdnState& candidate,
    float* convolution_output, float* recurrent_output,
    cudaStream_t stream) noexcept {
  return launch_gdn_prepare_chunk(
      config, convolution_input, convolution_weights, log_decay, beta, 1,
      committed, candidate, convolution_output, recurrent_output, stream);
}

cudaError_t launch_gdn_prepare_tiled(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, const GdnState& committed, const GdnState& candidate,
    float* convolution_output, float* recurrent_output,
    cudaStream_t stream) noexcept {
  return launch_gdn_prepare_chunk_layout(
      config, convolution_input, convolution_weights, log_decay, beta, 1,
      committed, candidate, convolution_output, recurrent_output, stream, true);
}

cudaError_t launch_gdn_prepare_chunk(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, std::size_t token_count, const GdnState& committed,
    const GdnState& candidate, float* convolution_output,
    float* recurrent_output, cudaStream_t stream) noexcept {
  return launch_gdn_prepare_chunk_layout(
      config, convolution_input, convolution_weights, log_decay, beta,
      token_count, committed, candidate, convolution_output, recurrent_output,
      stream, false);
}

cudaError_t launch_gdn_prepare_chunk_tiled(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, std::size_t token_count, const GdnState& committed,
    const GdnState& candidate, float* convolution_output,
    float* recurrent_output, cudaStream_t stream) noexcept {
  return launch_gdn_prepare_chunk_layout(
      config, convolution_input, convolution_weights, log_decay, beta,
      token_count, committed, candidate, convolution_output, recurrent_output,
      stream, true);
}

namespace {

cudaError_t launch_gdn_prepare_chunk_layout(
    const GdnConfig& config, const float* convolution_input,
    const float* convolution_weights, const float* log_decay,
    const float* beta, std::size_t token_count, const GdnState& committed,
    const GdnState& candidate, float* convolution_output,
    float* recurrent_output, cudaStream_t stream,
    bool value_is_tiled) noexcept {
  const std::size_t channels = gdn_convolution_channels(config);
  if (channels == 0 || token_count == 0 || convolution_input == nullptr ||
      convolution_weights == nullptr || log_decay == nullptr ||
      beta == nullptr || committed.convolution == nullptr ||
      committed.recurrent == nullptr || candidate.convolution == nullptr ||
      candidate.recurrent == nullptr || convolution_output == nullptr ||
      recurrent_output == nullptr || committed.convolution == candidate.convolution ||
      committed.recurrent == candidate.recurrent) {
    return cudaErrorInvalidValue;
  }
  const unsigned int blocks =
      static_cast<unsigned int>((channels + kThreads - 1) / kThreads);
  for (std::size_t start = 0; start < token_count; start += kScanWindow) {
    const std::size_t window =
        std::min(kScanWindow, token_count - start);
    const float* source_convolution =
        start == 0 ? committed.convolution : candidate.convolution;
    const float* source_recurrent =
        start == 0 ? committed.recurrent : candidate.recurrent;
    prepare_convolution_window<<<blocks, kThreads, 0, stream>>>(
        convolution_input + start * channels, convolution_weights,
        source_convolution, candidate.convolution,
        convolution_output + start * channels, channels,
        config.convolution_width, window);
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    prepare_recurrence_window<<<config.value_heads, kThreads, 0, stream>>>(
        config, convolution_output + start * channels,
        log_decay + start * config.value_heads,
        beta + start * config.value_heads, source_recurrent,
        candidate.recurrent,
        recurrent_output + start * gdn_output_values(config), window,
        value_is_tiled);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
  }
  return cudaSuccess;
}

}  // namespace

cudaError_t launch_gdn_commit(const GdnConfig& config,
                              const GdnState& candidate,
                              const GdnState& committed,
                              std::uint64_t new_frontier,
                              std::uint64_t* committed_frontier,
                              cudaStream_t stream) noexcept {
  const std::size_t convolution_values = gdn_convolution_values(config);
  const std::size_t recurrent_values = gdn_recurrent_values(config);
  if (convolution_values == 0 || recurrent_values == 0 ||
      candidate.convolution == nullptr || candidate.recurrent == nullptr ||
      committed.convolution == nullptr || committed.recurrent == nullptr ||
      committed_frontier == nullptr ||
      candidate.convolution == committed.convolution ||
      candidate.recurrent == committed.recurrent) {
    return cudaErrorInvalidValue;
  }
  commit_state<<<1, 256, 0, stream>>>(
      candidate.convolution, committed.convolution, convolution_values,
      candidate.recurrent, committed.recurrent, recurrent_values, new_frontier,
      committed_frontier);
  return cudaPeekAtLastError();
}

}  // namespace qw38::cuda
