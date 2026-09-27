#pragma once

#include "cuda/error.hpp"
#include "cuda/decode_control.hpp"
#include "cuda/stream.hpp"

#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <expected>

namespace qw38::cuda {

inline constexpr std::uint32_t kAttnQueryHeads = 24;
inline constexpr std::uint32_t kAttnKvHeads = 4;
inline constexpr std::uint32_t kAttnGqaGroup = 6;
inline constexpr std::uint32_t kAttnHeadDim = 256;
inline constexpr std::uint32_t kAttnRotaryDim = 64;
inline constexpr std::uint32_t kAttnRopeFreqs = 32;
inline constexpr std::uint32_t kAttnLayers = 16;
inline constexpr int kAttnPrepThreads = 128;  // T-03: one block/head
inline constexpr int kAttnScanThreads = 128;  // T-03: one block/head/segment
inline constexpr int kAttnMergeThreads = 128;  // T-03: one block/query head
inline constexpr int kAttnSegmentKeys = 256;   // T-03
inline constexpr int kAttnSubtileKeys = 32;    // T-03
inline constexpr int kAttnPartialStride = 2 + static_cast<int>(kAttnHeadDim);
inline constexpr std::uint32_t kAttnMaxPartitions = 128;

// At most two CTA waves for four KV/query tiles, with >=256 keys per split.
// The capacity bound is independent of the active length and can be reused by
// a fixed launch bucket. Every partition writes neutral data when empty.
[[nodiscard]] constexpr std::uint32_t attention_partition_count(
    std::uint64_t length, unsigned sm_count) noexcept {
  auto n = (length / 256 + (length % 256 != 0));
  auto waves = (2ull * sm_count + kAttnKvHeads - 1) / kAttnKvHeads;
  return static_cast<std::uint32_t>(std::min<std::uint64_t>(
      n, std::min<std::uint64_t>(waves, kAttnMaxPartitions)));
}
inline constexpr float kAttnScale = 1.0f / 16.0f;
inline constexpr int kAttnPrepQueryBlocks = static_cast<int>(kAttnQueryHeads);
inline constexpr int kAttnPrepKvBlocks = static_cast<int>(kAttnKvHeads);
inline constexpr int kAttnPrepBlocks = kAttnPrepQueryBlocks + kAttnPrepKvBlocks;

[[nodiscard]] constexpr std::uint32_t kv_head_for_query(
    std::uint32_t query_head) noexcept {
  return query_head / kAttnGqaGroup;
}

// One block per query head plus one block per KV head. QK RMS in FP32, partial
// RoPE, one BF16 store for Q and rotated K. g is copied per query head. V is
// copied once into cache. Writes K/V at token without a prepared K/V buffer.
[[nodiscard]] std::expected<void, Error> launch_attention_prepare(
    std::uint16_t const* qg, std::uint16_t const* k_raw,
    std::uint16_t const* v_raw, std::uint16_t const* gamma_q,
    std::uint16_t const* gamma_k, float const* inv_freq, float eps,
    std::int32_t position, std::uint16_t* q_out, std::uint16_t* g_out,
    std::uint16_t* kv, std::uint32_t attn_layer, std::uint64_t capacity,
    std::uint64_t token, Stream const& stream, DecodeControl const* control = nullptr);

// Token-major chunk preparation. Only valid rows are written to cache.
[[nodiscard]] std::expected<void, Error> launch_attention_prepare_chunk(
    std::uint16_t const* qg, std::uint16_t const* k_raw,
    std::uint16_t const* v_raw, std::uint16_t const* gamma_q,
    std::uint16_t const* gamma_k, float const* inv_freq, float eps,
    std::uint64_t first_position, std::uint32_t valid_tokens,
    std::uint16_t* q_out, std::uint16_t* g_out, std::uint16_t* kv,
    std::uint32_t attn_layer, std::uint64_t capacity, Stream const& stream);

inline constexpr std::uint32_t kAttnPrefillQueryTile = 32;
inline constexpr std::uint32_t kAttnPrefillQueryTileScalar = 1;
inline constexpr std::uint32_t kAttnPrefillQueryTileControl = 4;
inline constexpr std::uint32_t kAttnPrefillKeyTile = 64;
inline constexpr std::uint32_t kAttnPrefillKeyTileControl = 32;

struct AttentionPrefillResources {
  int registers{};
  std::size_t shared_bytes{};
  std::size_t local_bytes{};
  int occupancy_blocks_per_sm{};
};

// One block per query head and 32 query rows. Register-resident BF16 Q,
// FP32 softmax/PV, two BF16 P components and separately pipelined BF16 K/V.
// Explicit query tiles 1 and 4 select the scalar diagnostic controls.
[[nodiscard]] std::expected<void, Error> launch_attention_prefill_scan(
    std::uint16_t const* q, std::uint16_t const* g,
    std::uint16_t const* kv, std::uint32_t attn_layer,
    std::uint64_t capacity, std::uint64_t first_position,
    std::uint32_t valid_tokens, std::uint16_t* y,
    Stream const& stream,
    std::uint32_t query_tile = kAttnPrefillQueryTile,
    std::uint8_t* fp8_codes = nullptr, float* fp8_scales = nullptr);

[[nodiscard]] std::expected<AttentionPrefillResources, Error>
attention_prefill_resources(std::uint32_t query_tile = kAttnPrefillQueryTile);

[[nodiscard]] std::expected<AttentionPrefillResources, Error> attention_decode_resources();

// One 128-thread block per six-query GQA tile and contiguous partition.
// n_segments is a bounded launch partition count (not keys/256); active
// length determines tile boundaries. Writes FP32 max/sum/numerator, including
// neutral empty partitions. Two BF16 P components feed FP32 MMA accumulation.
[[nodiscard]] std::expected<void, Error> launch_attention_scan(
    std::uint16_t const* q, std::uint16_t const* kv, std::uint32_t attn_layer,
    std::uint64_t capacity, std::uint64_t populated, float* partials,
    std::uint32_t n_segments, Stream const& stream, DecodeControl const* control = nullptr,
    std::uint32_t partition_limit = 0);

// One block per query head. Merges segment statistics in increasing index,
// normalizes, applies FP32 sigmoid(g), stores BF16 gated y. No score vector.
[[nodiscard]] std::expected<void, Error> launch_attention_merge(
    float const* partials, std::uint16_t const* g, std::uint32_t n_segments,
    std::uint16_t* y_out, Stream const& stream, DecodeControl const* control = nullptr,
    std::uint32_t partition_limit = 0);

}  // namespace qw38::cuda
