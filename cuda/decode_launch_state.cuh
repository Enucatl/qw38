#pragma once

#include <cstddef>
#include <cstdint>

namespace qw38::cuda {

// Session-owned device record for values that change every token but must not
// recapture decode graphs. Physical GDN ping-pong bases stay stable; kernels
// load committed_slot and frontier/position from this record.
struct DecodeLaunchState {
  std::uint32_t token = 0;
  std::uint32_t position = 0;
  std::uint32_t frontier = 0;
  std::uint32_t kv_bucket = 0;
  std::uint32_t gdn_committed_slot = 0;
  std::uint32_t generation = 0;
  float* gdn_conv[2] = {};
  float* gdn_rec[2] = {};
};

inline constexpr std::uint32_t kDecodeLaunchKvBucketRows = 2048;
inline constexpr int kDecodeGraphTopologyCount = 2;
inline constexpr char kGdnParityDesign[] = "stable_pingpong_committed_slot";

__host__ __device__ inline std::size_t decode_launch_position(
    const DecodeLaunchState* launch, std::size_t fallback) noexcept {
  return launch != nullptr ? static_cast<std::size_t>(launch->position)
                           : fallback;
}

__host__ __device__ inline const float* decode_launch_gdn_source(
    const DecodeLaunchState* launch, const float* fallback,
    std::size_t layer_offset, bool recurrent) noexcept {
  if (launch == nullptr) return fallback;
  float* const* bases = recurrent ? launch->gdn_rec : launch->gdn_conv;
  return bases[launch->gdn_committed_slot] + layer_offset;
}

__host__ __device__ inline float* decode_launch_gdn_candidate(
    const DecodeLaunchState* launch, float* fallback, std::size_t layer_offset,
    bool recurrent) noexcept {
  if (launch == nullptr) return fallback;
  float* const* bases = recurrent ? launch->gdn_rec : launch->gdn_conv;
  return bases[launch->gdn_committed_slot ^ 1u] + layer_offset;
}

}  // namespace qw38::cuda
