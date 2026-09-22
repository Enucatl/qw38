#pragma once

#include "runtime/error.hpp"

#include "format/constants.hpp"
#include "format/schema.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <span>

namespace qw38::runtime {

inline constexpr std::uint32_t kLanguageLayers = 64;
inline constexpr std::uint32_t kGdnLayers = 48;
inline constexpr std::uint32_t kGdnKeyHeads = 16;
inline constexpr std::uint32_t kGdnValueHeads = 48;
inline constexpr std::uint32_t kGdnKeyDim = 128;
inline constexpr std::uint32_t kGdnValueDim = 128;
inline constexpr std::uint32_t kGdnRepeat = 3;  // value_head / 3 → key head
inline constexpr std::uint32_t kConvLayers = 48;
inline constexpr std::uint32_t kConvTaps = 3;
inline constexpr std::uint32_t kConvKernel = 4;
inline constexpr std::uint32_t kConvChannels = 10240;
inline constexpr std::uint32_t kGdnZWidth = 6144;
inline constexpr std::uint32_t kAttnLayers = 16;
inline constexpr std::uint32_t kQueryHeads = 24;
inline constexpr std::uint32_t kKvHeads = 4;
inline constexpr std::uint32_t kGqaGroup = 6;  // 24 query heads / 4 KV heads
inline constexpr std::uint32_t kHeadDim = 256;
inline constexpr std::uint32_t kKvComponents = 2;
inline constexpr std::uint32_t kKvComponentK = 0;
inline constexpr std::uint32_t kKvComponentV = 1;
inline constexpr std::uint32_t kQgWidth = 12288;     // 2 * 24 * 256
inline constexpr std::uint32_t kAttnKvWidth = 1024;  // 4 * 256
inline constexpr std::uint32_t kAttnOutWidth = 6144; // 24 * 256 gated y / o_proj K
inline constexpr std::uint32_t kHidden = 5120;
inline constexpr std::uint32_t kFfnWidth = 17408;
inline constexpr std::uint32_t kVocab = 248320;

// Language-only persistent state. Architecture V0 DERIVED totals.
// Physical S is FP32 [layer, value_head, value, key] (S-01/S-02).
inline constexpr std::uint64_t kGdnSElemsPerLayer =
    static_cast<std::uint64_t>(kGdnValueHeads) * kGdnValueDim * kGdnKeyDim;
inline constexpr std::uint64_t kGdnSBytesPerLayer =
    kGdnSElemsPerLayer * qw38::format::kFp32Size;
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

// Typed aliases inside GdnWorkspace (M-01). One allocation; no extra buffers.
// q/k stay 16 heads (indexed by value_head/3); v aliases convolved[4096:].
inline constexpr std::uint64_t kGdnOffQkv = 0;             // BF16 [10240]
inline constexpr std::uint64_t kGdnBytesQkv = 20480;
inline constexpr std::uint64_t kGdnOffZ = 20480;           // BF16 [48,128]
inline constexpr std::uint64_t kGdnBytesZ = 12288;
inline constexpr std::uint64_t kGdnOffConvolved = 32768;   // BF16 [10240]
inline constexpr std::uint64_t kGdnBytesConvolved = 20480;
inline constexpr std::uint64_t kGdnOffQHat = 53248;        // FP32 [16,128]
inline constexpr std::uint64_t kGdnBytesQHat = 8192;
inline constexpr std::uint64_t kGdnOffKHat = 61440;        // FP32 [16,128]
inline constexpr std::uint64_t kGdnBytesKHat = 8192;
inline constexpr std::uint64_t kGdnOffA = 69632;           // FP32 [48]
inline constexpr std::uint64_t kGdnBytesGate = 192;
inline constexpr std::uint64_t kGdnOffB = 69824;
inline constexpr std::uint64_t kGdnOffAlpha = 70016;
inline constexpr std::uint64_t kGdnOffBeta = 70208;
inline constexpr std::uint64_t kGdnOffO = 70400;           // FP32 [48,128] (TASK-012)
inline constexpr std::uint64_t kGdnBytesO = 24576;
inline constexpr std::uint64_t kGdnOffU = 94976;           // BF16 [48,128] (TASK-013)
inline constexpr std::uint64_t kGdnBytesU = 12288;
inline constexpr std::uint32_t kGdnVOffset = 4096;         // convolved v slice start

static_assert(kGdnOffZ == kGdnOffQkv + kGdnBytesQkv);
static_assert(kGdnOffConvolved == kGdnOffZ + kGdnBytesZ);
static_assert(kGdnOffQHat == kGdnOffConvolved + kGdnBytesConvolved);
static_assert(kGdnOffKHat == kGdnOffQHat + kGdnBytesQHat);
static_assert(kGdnOffA == kGdnOffKHat + kGdnBytesKHat);
static_assert(kGdnOffB == kGdnOffA + kGdnBytesGate);
static_assert(kGdnOffAlpha == kGdnOffB + kGdnBytesGate);
static_assert(kGdnOffBeta == kGdnOffAlpha + kGdnBytesGate);
static_assert(kGdnOffO == kGdnOffBeta + kGdnBytesGate);
static_assert(kGdnOffU == kGdnOffO + kGdnBytesO);
static_assert(kGdnOffU + kGdnBytesU == kGdnWorkspaceBytesPerToken);
static_assert(kGdnValueHeads / kGdnRepeat == kGdnKeyHeads);
static_assert(kGdnVOffset * 2u + kGdnBytesZ == kGdnBytesConvolved);
static_assert(kGdnSBytesPerLayer * kGdnLayers == kGdnSBytes);
static_assert(kGdnBytesO ==
              static_cast<std::uint64_t>(kGdnValueHeads) * kGdnValueDim *
                  qw38::format::kFp32Size);
static_assert(kQueryHeads / kKvHeads == kGqaGroup);
static_assert(kQgWidth == 2u * kQueryHeads * kHeadDim);
static_assert(kAttnKvWidth == kKvHeads * kHeadDim);

// Typed aliases inside AttentionWorkspace (M-01). Projected q/g/k/v, prepared Q,
// and per-query-head g. K/V append directly to cache (no prepared K/V duplicate).
// Trailing partials are reserved for TASK-015 segment statistics.
inline constexpr std::uint64_t kAttnOffQg = 0;            // BF16 [24, 512]
inline constexpr std::uint64_t kAttnBytesQg = 24576;
inline constexpr std::uint64_t kAttnOffK = 24576;          // BF16 [4, 256]
inline constexpr std::uint64_t kAttnBytesK = 2048;
inline constexpr std::uint64_t kAttnOffV = 26624;          // BF16 [4, 256]
inline constexpr std::uint64_t kAttnBytesV = 2048;
inline constexpr std::uint64_t kAttnOffQ = 28672;          // BF16 [24, 256]
inline constexpr std::uint64_t kAttnBytesQ = 12288;
inline constexpr std::uint64_t kAttnOffG = 40960;          // BF16 [24, 256]
inline constexpr std::uint64_t kAttnBytesG = 12288;
inline constexpr std::uint64_t kAttnOffPartials = 53248;   // FP32 TASK-015
inline constexpr std::uint64_t kAttnBytesPartials = 24768;
// T-03 decode attention geometry. Tuning, not ABI.
inline constexpr std::uint32_t kAttnSegmentKeys = 256;
inline constexpr std::uint32_t kAttnSubtileKeys = 32;
inline constexpr std::uint32_t kAttnPartialStride = 2u + kHeadDim;  // max, sum, num[256]
inline constexpr float kAttnScale = 1.0f / 16.0f;  // 1/sqrt(256)

[[nodiscard]] constexpr std::uint64_t attn_segment_count(
    std::uint64_t populated) noexcept {
  if (populated == 0) {
    return 0;
  }
  return (populated + (kAttnSegmentKeys - 1u)) / kAttnSegmentKeys;
}

[[nodiscard]] constexpr std::uint64_t attn_partials_bytes(
    std::uint64_t n_segments) noexcept {
  return n_segments * static_cast<std::uint64_t>(kQueryHeads) *
         kAttnPartialStride * qw38::format::kFp32Size;
}

[[nodiscard]] constexpr std::uint64_t attn_workspace_bytes_for_segments(
    std::uint64_t n_segments) noexcept {
  std::uint64_t const need = kAttnOffPartials + attn_partials_bytes(n_segments);
  return need < kAttentionWorkspaceBytesPerToken ? kAttentionWorkspaceBytesPerToken
                                                 : need;
}

static_assert(kAttnOffK == kAttnOffQg + kAttnBytesQg);
static_assert(kAttnOffV == kAttnOffK + kAttnBytesK);
static_assert(kAttnOffQ == kAttnOffV + kAttnBytesV);
static_assert(kAttnOffG == kAttnOffQ + kAttnBytesQ);
static_assert(kAttnOffPartials == kAttnOffG + kAttnBytesG);
static_assert(kAttnOffPartials + kAttnBytesPartials ==
              kAttentionWorkspaceBytesPerToken);
static_assert(kAttnBytesQg ==
              static_cast<std::uint64_t>(kQgWidth) * qw38::format::kBf16Size);
static_assert(kAttnBytesQ == static_cast<std::uint64_t>(kQueryHeads) *
                                 kHeadDim * qw38::format::kBf16Size);
static_assert(kAttnBytesPartials ==
              static_cast<std::uint64_t>(kQueryHeads) * (2u + kHeadDim) *
                  qw38::format::kFp32Size);
static_assert(kAttnOutWidth == kQueryHeads * kHeadDim);
static_assert(kAttnPartialStride == 2u + kHeadDim);
static_assert(kAttnBytesPartials == attn_partials_bytes(1));
static_assert(kAttnSubtileKeys * 8u == kAttnSegmentKeys);
static_assert(kAttnSubtileKeys * kHeadDim * qw38::format::kBf16Size ==
              16384u);  // 16 KiB K or V staging bound

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
