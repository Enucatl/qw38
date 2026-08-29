#include "attention.h"

#include <array>
#include <cmath>

namespace qw38::internal {
namespace {

constexpr std::size_t kMaximumQueryHeads = 24;
constexpr std::size_t kMaximumKvHeads = 4;
constexpr std::size_t kMaximumHeadWidth = 256;
constexpr float kRmsEpsilon = 1.0e-6F;
constexpr float kRopeTheta = 10000000.0F;

float sigmoid(float value) noexcept {
  if (value >= 0.0F) {
    const float exponential = std::exp(-value);
    return 1.0F / (1.0F + exponential);
  }
  const float exponential = std::exp(value);
  return exponential / (1.0F + exponential);
}

void rotate_partial(float* vector, std::size_t rotary_width,
                    std::size_t position) noexcept {
  const std::size_t half = rotary_width / 2;
  std::array<float, kMaximumHeadWidth> original{};
  for (std::size_t lane = 0; lane < rotary_width; ++lane) {
    original[lane] = vector[lane];
  }
  for (std::size_t lane = 0; lane < half; ++lane) {
    const float exponent =
        static_cast<float>(lane * 2) / static_cast<float>(rotary_width);
    const float angle = static_cast<float>(position) /
                        std::pow(kRopeTheta, exponent);
    const float cosine = std::cos(angle);
    const float sine = std::sin(angle);
    vector[lane] =
        original[lane] * cosine - original[half + lane] * sine;
    vector[half + lane] =
        original[half + lane] * cosine + original[lane] * sine;
  }
}

void matrix_vector(const float* weights, std::size_t rows,
                   std::size_t columns, const float* input,
                   float* output) noexcept {
  for (std::size_t row = 0; row < rows; ++row) {
    float total = 0.0F;
    for (std::size_t column = 0; column < columns; ++column) {
      total += weights[row * columns + column] * input[column];
    }
    output[row] = total;
  }
}

Status rms_norm_impl(const float* input, const float* weight,
                     std::size_t count, bool weight_is_offset,
                     float* output) noexcept {
  if (input == nullptr || weight == nullptr || output == nullptr) {
    return {StatusCode::kInvalidArgument, "RMSNorm pointers must not be null"};
  }
  if (count == 0) {
    return {StatusCode::kInvalidArgument, "RMSNorm width must be nonzero"};
  }
  float sum_squares = 0.0F;
  for (std::size_t index = 0; index < count; ++index) {
    sum_squares += input[index] * input[index];
  }
  const float inverse =
      1.0F / std::sqrt(sum_squares / static_cast<float>(count) + kRmsEpsilon);
  for (std::size_t index = 0; index < count; ++index) {
    const float scale =
        weight_is_offset ? 1.0F + weight[index] : weight[index];
    output[index] = input[index] * inverse * scale;
  }
  return Status::ok();
}

}  // namespace

Status rms_norm(const float* input, const float* weight, std::size_t count,
                float* output) noexcept {
  return rms_norm_impl(input, weight, count, true, output);
}

Status rms_norm_scale(const float* input, const float* scale,
                      std::size_t count, float* output) noexcept {
  return rms_norm_impl(input, scale, count, false, output);
}

namespace {

Status attention_decode_step_impl(
    const AttentionShape& shape, std::size_t position, const float* query,
    std::size_t query_count, const float* key, std::size_t key_count,
    const float* value, std::size_t value_count, const float* query_norm_weight,
    const float* key_norm_weight, const float* output_gate,
    std::size_t gate_count, float* key_cache, std::size_t key_cache_count,
    float* value_cache, std::size_t value_cache_count, float* score_workspace,
    std::size_t score_count, float* output, std::size_t output_count,
    bool norm_weight_is_offset) noexcept {
  if (shape.query_heads == 0 || shape.kv_heads == 0 ||
      shape.head_width == 0 || shape.capacity == 0 ||
      shape.query_heads > kMaximumQueryHeads ||
      shape.kv_heads > kMaximumKvHeads ||
      shape.head_width > kMaximumHeadWidth ||
      shape.query_heads % shape.kv_heads != 0 ||
      shape.rotary_width > shape.head_width || shape.rotary_width % 2 != 0) {
    return {StatusCode::kInvalidArgument, "invalid attention shape"};
  }
  if (position >= shape.capacity) {
    return {StatusCode::kResourceExhausted,
            "attention position exceeds KV capacity"};
  }
  if (query == nullptr || key == nullptr || value == nullptr ||
      query_norm_weight == nullptr || key_norm_weight == nullptr ||
      output_gate == nullptr || key_cache == nullptr || value_cache == nullptr ||
      score_workspace == nullptr || output == nullptr) {
    return {StatusCode::kInvalidArgument,
            "attention pointers must not be null"};
  }
  const std::size_t query_values = shape.query_heads * shape.head_width;
  const std::size_t kv_values = shape.kv_heads * shape.head_width;
  const std::size_t cache_values = shape.capacity * kv_values;
  if (query_count != query_values || key_count != kv_values ||
      value_count != kv_values || gate_count != query_values ||
      key_cache_count != cache_values || value_cache_count != cache_values ||
      score_count < position + 1 || output_count != query_values) {
    return {StatusCode::kInvalidArgument,
            "attention buffer counts do not match the declared shape"};
  }

  std::array<float, kMaximumQueryHeads * kMaximumHeadWidth> normalized_query{};
  std::array<float, kMaximumKvHeads * kMaximumHeadWidth> normalized_key{};
  for (std::size_t head = 0; head < shape.query_heads; ++head) {
    Status status = rms_norm_impl(
        query + head * shape.head_width, query_norm_weight, shape.head_width,
        norm_weight_is_offset,
        normalized_query.data() + head * shape.head_width);
    if (!status.is_ok()) return status;
    rotate_partial(normalized_query.data() + head * shape.head_width,
                   shape.rotary_width, position);
  }
  for (std::size_t head = 0; head < shape.kv_heads; ++head) {
    Status status = rms_norm_impl(
        key + head * shape.head_width, key_norm_weight, shape.head_width,
        norm_weight_is_offset,
        normalized_key.data() + head * shape.head_width);
    if (!status.is_ok()) return status;
    rotate_partial(normalized_key.data() + head * shape.head_width,
                   shape.rotary_width, position);
    const std::size_t cache_base =
        position * kv_values + head * shape.head_width;
    for (std::size_t lane = 0; lane < shape.head_width; ++lane) {
      key_cache[cache_base + lane] =
          normalized_key[head * shape.head_width + lane];
      value_cache[cache_base + lane] = value[head * shape.head_width + lane];
    }
  }

  const std::size_t group_size = shape.query_heads / shape.kv_heads;
  const float scaling = 1.0F / std::sqrt(static_cast<float>(shape.head_width));
  for (std::size_t query_head = 0; query_head < shape.query_heads;
       ++query_head) {
    const std::size_t kv_head = query_head / group_size;
    float maximum = -INFINITY;
    for (std::size_t context = 0; context <= position; ++context) {
      const std::size_t cache_base =
          context * kv_values + kv_head * shape.head_width;
      float score = 0.0F;
      for (std::size_t lane = 0; lane < shape.head_width; ++lane) {
        score += normalized_query[query_head * shape.head_width + lane] *
                 key_cache[cache_base + lane];
      }
      score_workspace[context] = score * scaling;
      if (score_workspace[context] > maximum) maximum = score_workspace[context];
    }
    float denominator = 0.0F;
    for (std::size_t context = 0; context <= position; ++context) {
      score_workspace[context] =
          std::exp(score_workspace[context] - maximum);
      denominator += score_workspace[context];
    }
    for (std::size_t lane = 0; lane < shape.head_width; ++lane) {
      float result = 0.0F;
      for (std::size_t context = 0; context <= position; ++context) {
        const std::size_t cache_base =
            context * kv_values + kv_head * shape.head_width;
        result += (score_workspace[context] / denominator) *
                  value_cache[cache_base + lane];
      }
      output[query_head * shape.head_width + lane] =
          result * sigmoid(output_gate[query_head * shape.head_width + lane]);
    }
  }
  return Status::ok();
}

}  // namespace

Status attention_decode_step(
    const AttentionShape& shape, std::size_t position, const float* query,
    std::size_t query_count, const float* key, std::size_t key_count,
    const float* value, std::size_t value_count, const float* query_norm_weight,
    const float* key_norm_weight, const float* output_gate,
    std::size_t gate_count, float* key_cache, std::size_t key_cache_count,
    float* value_cache, std::size_t value_cache_count, float* score_workspace,
    std::size_t score_count, float* output, std::size_t output_count) noexcept {
  return attention_decode_step_impl(
      shape, position, query, query_count, key, key_count, value, value_count,
      query_norm_weight, key_norm_weight, output_gate, gate_count, key_cache,
      key_cache_count, value_cache, value_cache_count, score_workspace,
      score_count, output, output_count, true);
}

Status attention_decode_step_scale(
    const AttentionShape& shape, std::size_t position, const float* query,
    std::size_t query_count, const float* key, std::size_t key_count,
    const float* value, std::size_t value_count, const float* query_norm_scale,
    const float* key_norm_scale, const float* output_gate,
    std::size_t gate_count, float* key_cache, std::size_t key_cache_count,
    float* value_cache, std::size_t value_cache_count, float* score_workspace,
    std::size_t score_count, float* output, std::size_t output_count) noexcept {
  return attention_decode_step_impl(
      shape, position, query, query_count, key, key_count, value, value_count,
      query_norm_scale, key_norm_scale, output_gate, gate_count, key_cache,
      key_cache_count, value_cache, value_cache_count, score_workspace,
      score_count, output, output_count, false);
}

Status swiglu_ffn(
    const float* input, std::size_t hidden_width,
    const float* gate_weights, const float* up_weights,
    std::size_t intermediate_width, const float* down_weights,
    float* gate_projection, float* up_projection, float* activated,
    float* output) noexcept {
  if (input == nullptr || gate_weights == nullptr || up_weights == nullptr ||
      down_weights == nullptr || gate_projection == nullptr ||
      up_projection == nullptr || activated == nullptr || output == nullptr) {
    return {StatusCode::kInvalidArgument, "FFN pointers must not be null"};
  }
  if (hidden_width == 0 || intermediate_width == 0) {
    return {StatusCode::kInvalidArgument, "FFN dimensions must be nonzero"};
  }
  matrix_vector(gate_weights, intermediate_width, hidden_width, input,
                gate_projection);
  matrix_vector(up_weights, intermediate_width, hidden_width, input,
                up_projection);
  for (std::size_t index = 0; index < intermediate_width; ++index) {
    const float gate = gate_projection[index];
    const float silu = gate / (1.0F + std::exp(-gate));
    activated[index] = silu * up_projection[index];
  }
  matrix_vector(down_weights, hidden_width, intermediate_width, activated,
                output);
  return Status::ok();
}

}  // namespace qw38::internal
