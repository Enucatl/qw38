#pragma once

#include "runtime/error.hpp"

#include "format/constants.hpp"
#include "format/schema.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <span>

namespace qw38::runtime {

inline constexpr std::uint32_t kGdnLayers = 48;
inline constexpr std::uint32_t kGdnValueHeads = 48;
inline constexpr std::uint32_t kGdnKeyDim = 128;
inline constexpr std::uint32_t kGdnValueDim = 128;
inline constexpr std::uint32_t kConvLayers = 48;
inline constexpr std::uint32_t kConvTaps = 3;
inline constexpr std::uint32_t kConvChannels = 10240;
inline constexpr std::uint32_t kAttnLayers = 16;
inline constexpr std::uint32_t kKvHeads = 4;
inline constexpr std::uint32_t kHeadDim = 256;
inline constexpr std::uint32_t kKvComponents = 2;
inline constexpr std::uint32_t kHidden = 5120;
inline constexpr std::uint32_t kFfnWidth = 17408;
inline constexpr std::uint32_t kVocab = 248320;

// Language-only persistent state. Architecture V0 DERIVED totals.
inline constexpr std::uint64_t kGdnSBytes = 150994944;
inline constexpr std::uint64_t kConvHistoryBytes = 2949120;
inline constexpr std::uint64_t kFixedPersistentBytes = 153944064;
inline constexpr std::uint64_t kKvBytesPerToken = 65536;

// T-02 tuning: scratch/residual arena provision, not an architecture contract.
inline constexpr std::uint64_t kArenaTokenCapacity = 256;

inline constexpr std::uint64_t kResidualBytesPerToken = 20480;
inline constexpr std::uint64_t kNormalizedBytesPerToken = 10240;
inline constexpr std::uint64_t kGdnWorkspaceBytesPerToken = 107264;
inline constexpr std::uint64_t kAttentionWorkspaceBytesPerToken = 78016;
inline constexpr std::uint64_t kSwigluBytesPerToken = 34816;
inline constexpr std::uint64_t kLogitsBytesPerToken = 993280;

[[nodiscard]] std::expected<std::uint64_t, Error> kv_cache_bytes(
    std::uint64_t capacity);
[[nodiscard]] std::expected<std::uint64_t, Error> persistent_state_bytes(
    std::uint64_t capacity);
[[nodiscard]] std::expected<std::uint64_t, Error> residual_bytes(
    std::uint64_t token_capacity);

[[nodiscard]] std::expected<std::uint64_t, Error> gdn_s_byte_offset(
    std::uint32_t layer, std::uint32_t value_head, std::uint32_t value,
    std::uint32_t key);
[[nodiscard]] std::expected<std::uint64_t, Error> conv_history_byte_offset(
    std::uint32_t layer, std::uint32_t tap, std::uint32_t channel);
[[nodiscard]] std::expected<std::uint64_t, Error> kv_byte_offset(
    std::uint32_t layer, std::uint32_t component, std::uint32_t head,
    std::uint64_t token, std::uint32_t dim, std::uint64_t capacity);

[[nodiscard]] std::expected<void, Error> require_language_state(
    std::span<qw38::format::StateAllocation const> state);
[[nodiscard]] std::expected<void, Error> require_language_scratch(
    std::span<qw38::format::ScratchAllocation const> scratch);

[[nodiscard]] std::array<qw38::format::StateAllocation, 3>
language_persistent_schema();

}  // namespace qw38::runtime
