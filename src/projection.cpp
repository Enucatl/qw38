#include "projection.h"

#include <cstdint>
#include <limits>

namespace qw38::internal {
namespace {

bool multiply(std::size_t left, std::size_t right,
              std::size_t* result) noexcept {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

bool overlap(const float* left, std::size_t left_count, const float* right,
             std::size_t right_count) noexcept {
  const std::uintptr_t left_begin = reinterpret_cast<std::uintptr_t>(left);
  const std::uintptr_t right_begin = reinterpret_cast<std::uintptr_t>(right);
  const std::size_t maximum =
      std::numeric_limits<std::uintptr_t>::max() / sizeof(float);
  if (left_count > maximum || right_count > maximum) return true;
  const std::uintptr_t left_bytes = left_count * sizeof(float);
  const std::uintptr_t right_bytes = right_count * sizeof(float);
  if (left_begin > std::numeric_limits<std::uintptr_t>::max() - left_bytes ||
      right_begin > std::numeric_limits<std::uintptr_t>::max() - right_bytes) {
    return true;
  }
  return left_begin < right_begin + right_bytes &&
         right_begin < left_begin + left_bytes;
}

}  // namespace

Status split_gdn_qkv(const float* packed, std::size_t packed_count,
                     std::size_t key_heads, std::size_t key_width,
                     std::size_t value_heads, std::size_t value_width,
                     float* query, std::size_t query_count, float* key,
                     std::size_t key_count, float* value,
                     std::size_t value_count) noexcept {
  std::size_t key_values = 0;
  std::size_t value_values = 0;
  if (packed == nullptr || query == nullptr || key == nullptr ||
      value == nullptr || key_heads == 0 || key_width == 0 ||
      value_heads == 0 || value_width == 0 ||
      !multiply(key_heads, key_width, &key_values) ||
      !multiply(value_heads, value_width, &value_values) ||
      key_values > (std::numeric_limits<std::size_t>::max() - value_values) / 2 ||
      packed_count != key_values * 2 + value_values ||
      query_count != key_values || key_count != key_values ||
      value_count != value_values) {
    return {StatusCode::kInvalidArgument,
            "GDN packed QKV dimensions or output counts are invalid"};
  }
  if (overlap(packed, packed_count, query, query_count) ||
      overlap(packed, packed_count, key, key_count) ||
      overlap(packed, packed_count, value, value_count) ||
      overlap(query, query_count, key, key_count) ||
      overlap(query, query_count, value, value_count) ||
      overlap(key, key_count, value, value_count)) {
    return {StatusCode::kInvalidArgument,
            "GDN packed QKV input and outputs must not overlap"};
  }
  for (std::size_t index = 0; index < key_values; ++index) {
    query[index] = packed[index];
    key[index] = packed[key_values + index];
  }
  for (std::size_t index = 0; index < value_values; ++index) {
    value[index] = packed[key_values * 2 + index];
  }
  return Status::ok();
}

Status split_attention_query_gate(
    const float* packed, std::size_t packed_count, std::size_t query_heads,
    std::size_t head_width, float* query, std::size_t query_count, float* gate,
    std::size_t gate_count) noexcept {
  std::size_t output_values = 0;
  if (packed == nullptr || query == nullptr || gate == nullptr ||
      query_heads == 0 || head_width == 0 ||
      !multiply(query_heads, head_width, &output_values) ||
      output_values > std::numeric_limits<std::size_t>::max() / 2 ||
      packed_count != output_values * 2 || query_count != output_values ||
      gate_count != output_values) {
    return {StatusCode::kInvalidArgument,
            "attention packed query/gate dimensions or counts are invalid"};
  }
  if (overlap(packed, packed_count, query, query_count) ||
      overlap(packed, packed_count, gate, gate_count) ||
      overlap(query, query_count, gate, gate_count)) {
    return {StatusCode::kInvalidArgument,
            "attention packed input and outputs must not overlap"};
  }
  for (std::size_t head = 0; head < query_heads; ++head) {
    const std::size_t packed_base = head * head_width * 2;
    const std::size_t output_base = head * head_width;
    for (std::size_t lane = 0; lane < head_width; ++lane) {
      query[output_base + lane] = packed[packed_base + lane];
      gate[output_base + lane] = packed[packed_base + head_width + lane];
    }
  }
  return Status::ok();
}

}  // namespace qw38::internal
