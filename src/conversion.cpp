#include "conversion.h"

#include <limits>

namespace qw38::internal {
namespace {

Status validate(const float* input, std::size_t key_heads,
                std::size_t replicas_per_key, std::size_t head_width,
                float* output, std::size_t count) noexcept {
  if (input == nullptr || output == nullptr || key_heads == 0 ||
      replicas_per_key == 0 || head_width == 0 ||
      key_heads > std::numeric_limits<std::size_t>::max() / replicas_per_key ||
      key_heads * replicas_per_key >
          std::numeric_limits<std::size_t>::max() / head_width ||
      key_heads * replicas_per_key * head_width != count) {
    return {StatusCode::kInvalidArgument,
            "invalid GDN grouped/tiled permutation shape or buffers"};
  }
  if (input == output && replicas_per_key != 1) {
    return {StatusCode::kInvalidArgument,
            "GDN grouped/tiled permutation cannot run in place"};
  }
  return Status::ok();
}

}  // namespace

Status gdn_tiled_to_grouped(const float* tiled, std::size_t key_heads,
                            std::size_t replicas_per_key,
                            std::size_t head_width, float* grouped,
                            std::size_t count) noexcept {
  Status status = validate(tiled, key_heads, replicas_per_key, head_width,
                           grouped, count);
  if (!status.is_ok()) return status;
  for (std::size_t key = 0; key < key_heads; ++key) {
    for (std::size_t replica = 0; replica < replicas_per_key; ++replica) {
      for (std::size_t lane = 0; lane < head_width; ++lane) {
        const std::size_t tiled_index =
            (replica * key_heads + key) * head_width + lane;
        const std::size_t grouped_index =
            (key * replicas_per_key + replica) * head_width + lane;
        grouped[grouped_index] = tiled[tiled_index];
      }
    }
  }
  return Status::ok();
}

Status gdn_grouped_to_tiled(const float* grouped, std::size_t key_heads,
                            std::size_t replicas_per_key,
                            std::size_t head_width, float* tiled,
                            std::size_t count) noexcept {
  Status status = validate(grouped, key_heads, replicas_per_key, head_width,
                           tiled, count);
  if (!status.is_ok()) return status;
  for (std::size_t key = 0; key < key_heads; ++key) {
    for (std::size_t replica = 0; replica < replicas_per_key; ++replica) {
      for (std::size_t lane = 0; lane < head_width; ++lane) {
        const std::size_t grouped_index =
            (key * replicas_per_key + replica) * head_width + lane;
        const std::size_t tiled_index =
            (replica * key_heads + key) * head_width + lane;
        tiled[tiled_index] = grouped[grouped_index];
      }
    }
  }
  return Status::ok();
}

}  // namespace qw38::internal
