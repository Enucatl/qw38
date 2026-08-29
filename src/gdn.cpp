#include "gdn.h"

#include <array>
#include <cmath>

namespace qw38::internal {
namespace {

constexpr std::size_t kMaximumKeyHeads = 16;
constexpr std::size_t kMaximumValueHeads = 48;
constexpr std::size_t kMaximumKeyWidth = 128;
constexpr std::size_t kMaximumValueWidth = 128;
constexpr float kL2Epsilon = 1.0e-6F;

float sigmoid(float value) noexcept {
  if (value >= 0.0F) {
    const float exponential = std::exp(-value);
    return 1.0F / (1.0F + exponential);
  }
  const float exponential = std::exp(value);
  return exponential / (1.0F + exponential);
}

float softplus(float value) noexcept {
  if (value > 20.0F) return value;
  if (value < -20.0F) return std::exp(value);
  return std::log1p(std::exp(value));
}

Status validate_shape(const GdnShape& shape) noexcept {
  if (shape.key_heads == 0 || shape.value_heads == 0 ||
      shape.key_width == 0 || shape.value_width == 0) {
    return {StatusCode::kInvalidArgument, "GDN dimensions must be nonzero"};
  }
  if (shape.key_heads > kMaximumKeyHeads ||
      shape.value_heads > kMaximumValueHeads ||
      shape.key_width > kMaximumKeyWidth ||
      shape.value_width > kMaximumValueWidth) {
    return {StatusCode::kInvalidArgument,
            "GDN dimensions exceed the admitted Qwen3.8 contract"};
  }
  if (shape.value_heads % shape.key_heads != 0) {
    return {StatusCode::kInvalidArgument,
            "GDN value heads must be an exact multiple of key heads"};
  }
  return Status::ok();
}

Status validate_gates(const float* a, const float* b, const float* decay,
                      const float* dt_bias, std::size_t count,
                      float* log_decay, float* beta) noexcept {
  if (a == nullptr || b == nullptr || decay == nullptr || dt_bias == nullptr ||
      log_decay == nullptr || beta == nullptr) {
    return {StatusCode::kInvalidArgument, "GDN gate pointers must not be null"};
  }
  if (count == 0 || count > kMaximumValueHeads) {
    return {StatusCode::kInvalidArgument,
            "GDN gate count exceeds the admitted Qwen3.8 contract"};
  }
  return Status::ok();
}

Status validate_recurrent_buffers(
    const GdnShape& shape, const float* query, std::size_t query_count,
    const float* key, std::size_t key_count, const float* value,
    std::size_t value_count, const float* log_decay, const float* beta,
    std::size_t gate_count, const float* state, std::size_t state_count,
    const float* output, std::size_t output_count) noexcept {
  Status status = validate_shape(shape);
  if (!status.is_ok()) return status;
  if (query == nullptr || key == nullptr || value == nullptr ||
      log_decay == nullptr || beta == nullptr || state == nullptr ||
      output == nullptr) {
    return {StatusCode::kInvalidArgument, "GDN pointers must not be null"};
  }
  const std::size_t expected_query = shape.key_heads * shape.key_width;
  const std::size_t expected_value = shape.value_heads * shape.value_width;
  const std::size_t expected_state =
      shape.value_heads * shape.key_width * shape.value_width;
  if (query_count != expected_query || key_count != expected_query ||
      value_count != expected_value || gate_count != shape.value_heads ||
      state_count != expected_state || output_count != expected_value) {
    return {StatusCode::kInvalidArgument,
            "GDN buffer counts do not match the declared shape"};
  }
  return Status::ok();
}

void normalize_heads(const float* input, std::size_t heads,
                     std::size_t width, bool scale_query,
                     float* output) noexcept {
  for (std::size_t head = 0; head < heads; ++head) {
    const std::size_t base = head * width;
    float sum_squares = 0.0F;
    for (std::size_t lane = 0; lane < width; ++lane) {
      const float value = input[base + lane];
      sum_squares += value * value;
    }
    float inverse = 1.0F / std::sqrt(sum_squares + kL2Epsilon);
    if (scale_query) inverse /= std::sqrt(static_cast<float>(width));
    for (std::size_t lane = 0; lane < width; ++lane) {
      output[base + lane] = input[base + lane] * inverse;
    }
  }
}

}  // namespace

Status gdn_gates_from_source(const float* a, const float* b,
                             const float* a_log, const float* dt_bias,
                             std::size_t count, float* log_decay,
                             float* beta) noexcept {
  Status status =
      validate_gates(a, b, a_log, dt_bias, count, log_decay, beta);
  if (!status.is_ok()) return status;
  for (std::size_t head = 0; head < count; ++head) {
    beta[head] = sigmoid(b[head]);
    log_decay[head] =
        -std::exp(a_log[head]) * softplus(a[head] + dt_bias[head]);
  }
  return Status::ok();
}

Status gdn_gates_from_gguf(const float* a, const float* b,
                           const float* folded_a, const float* dt_bias,
                           std::size_t count, float* log_decay,
                           float* beta) noexcept {
  Status status =
      validate_gates(a, b, folded_a, dt_bias, count, log_decay, beta);
  if (!status.is_ok()) return status;
  for (std::size_t head = 0; head < count; ++head) {
    if (!std::isfinite(folded_a[head]) || folded_a[head] >= 0.0F) {
      return {StatusCode::kInvalidArgument,
              "GGUF folded GDN A values must be finite and negative"};
    }
  }
  for (std::size_t head = 0; head < count; ++head) {
    beta[head] = sigmoid(b[head]);
    log_decay[head] =
        folded_a[head] * softplus(a[head] + dt_bias[head]);
  }
  return Status::ok();
}

Status gdn_recurrent_step_precomputed(
    const GdnShape& shape, const float* query, std::size_t query_count,
    const float* key, std::size_t key_count, const float* value,
    std::size_t value_count, const float* log_decay, const float* beta,
    std::size_t gate_count, float* state, std::size_t state_count,
    float* output, std::size_t output_count) noexcept {
  Status status = validate_recurrent_buffers(
      shape, query, query_count, key, key_count, value, value_count, log_decay,
      beta, gate_count, state, state_count, output, output_count);
  if (!status.is_ok()) return status;

  std::array<float, kMaximumKeyHeads * kMaximumKeyWidth> normalized_query{};
  std::array<float, kMaximumKeyHeads * kMaximumKeyWidth> normalized_key{};
  normalize_heads(query, shape.key_heads, shape.key_width, true,
                  normalized_query.data());
  normalize_heads(key, shape.key_heads, shape.key_width, false,
                  normalized_key.data());

  const std::size_t reuse = shape.value_heads / shape.key_heads;
  for (std::size_t value_head = 0; value_head < shape.value_heads;
       ++value_head) {
    const std::size_t key_head = value_head / reuse;
    const std::size_t key_base = key_head * shape.key_width;
    const std::size_t value_base = value_head * shape.value_width;
    const std::size_t state_base =
        value_head * shape.key_width * shape.value_width;
    const float decay = std::exp(log_decay[value_head]);
    for (std::size_t index = 0;
         index < shape.key_width * shape.value_width; ++index) {
      state[state_base + index] *= decay;
    }
    for (std::size_t value_lane = 0; value_lane < shape.value_width;
         ++value_lane) {
      float prediction = 0.0F;
      for (std::size_t key_lane = 0; key_lane < shape.key_width; ++key_lane) {
        prediction +=
            normalized_key[key_base + key_lane] *
            state[state_base + key_lane * shape.value_width + value_lane];
      }
      const float delta =
          (value[value_base + value_lane] - prediction) * beta[value_head];
      for (std::size_t key_lane = 0; key_lane < shape.key_width; ++key_lane) {
        state[state_base + key_lane * shape.value_width + value_lane] +=
            normalized_key[key_base + key_lane] * delta;
      }
    }
    for (std::size_t value_lane = 0; value_lane < shape.value_width;
         ++value_lane) {
      float result = 0.0F;
      for (std::size_t key_lane = 0; key_lane < shape.key_width; ++key_lane) {
        result +=
            normalized_query[key_base + key_lane] *
            state[state_base + key_lane * shape.value_width + value_lane];
      }
      output[value_base + value_lane] = result;
    }
  }
  return Status::ok();
}

Status gdn_recurrent_step(
    const GdnShape& shape, const float* query, std::size_t query_count,
    const float* key, std::size_t key_count, const float* value,
    std::size_t value_count, const float* a, const float* b,
    const float* a_log, const float* dt_bias, std::size_t gate_count,
    float* state, std::size_t state_count, float* output,
    std::size_t output_count, float* log_decay, float* beta) noexcept {
  Status status = validate_recurrent_buffers(
      shape, query, query_count, key, key_count, value, value_count, log_decay,
      beta, gate_count, state, state_count, output, output_count);
  if (!status.is_ok()) return status;
  status = gdn_gates_from_source(a, b, a_log, dt_bias, gate_count, log_decay,
                                 beta);
  if (!status.is_ok()) return status;
  return gdn_recurrent_step_precomputed(
      shape, query, query_count, key, key_count, value, value_count, log_decay,
      beta, gate_count, state, state_count, output, output_count);
}

Status causal_depthwise_conv_step(
    std::size_t channels, std::size_t kernel_width, const float* input,
    std::size_t input_count, const float* weights, std::size_t weight_count,
    float* state, std::size_t state_count, float* output,
    std::size_t output_count) noexcept {
  if (channels == 0 || kernel_width == 0) {
    return {StatusCode::kInvalidArgument,
            "convolution dimensions must be nonzero"};
  }
  if (input == nullptr || weights == nullptr || state == nullptr ||
      output == nullptr) {
    return {StatusCode::kInvalidArgument,
            "convolution pointers must not be null"};
  }
  const std::size_t state_values = channels * kernel_width;
  if (input_count != channels || output_count != channels ||
      weight_count != state_values || state_count != state_values) {
    return {StatusCode::kInvalidArgument,
            "convolution buffer counts do not match the declared shape"};
  }
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const std::size_t base = channel * kernel_width;
    for (std::size_t index = 0; index + 1 < kernel_width; ++index) {
      state[base + index] = state[base + index + 1];
    }
    state[base + kernel_width - 1] = input[channel];
    float convolution = 0.0F;
    for (std::size_t index = 0; index < kernel_width; ++index) {
      convolution += state[base + index] * weights[base + index];
    }
    output[channel] = convolution / (1.0F + std::exp(-convolution));
  }
  return Status::ok();
}

}  // namespace qw38::internal
