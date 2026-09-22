#pragma once

#include "cuda/error.hpp"
#include "cuda/stream.hpp"

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
    std::uint64_t token, Stream const& stream);

// One 128-thread block per query head per 256-key segment. FP32 online softmax
// over causal range [0, populated). Writes FP32 max, sum, and 256-value
// numerator. Empty/tail keys are masked. Staging is one 32-key K or V subtile.
[[nodiscard]] std::expected<void, Error> launch_attention_scan(
    std::uint16_t const* q, std::uint16_t const* kv, std::uint32_t attn_layer,
    std::uint64_t capacity, std::uint64_t populated, float* partials,
    std::uint32_t n_segments, Stream const& stream);

// One block per query head. Merges segment statistics in increasing index,
// normalizes, applies FP32 sigmoid(g), stores BF16 gated y. No score vector.
[[nodiscard]] std::expected<void, Error> launch_attention_merge(
    float const* partials, std::uint16_t const* g, std::uint32_t n_segments,
    std::uint16_t* y_out, Stream const& stream);

}  // namespace qw38::cuda
