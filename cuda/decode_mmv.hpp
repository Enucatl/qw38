#pragma once

#include "cuda/error.hpp"
#include "cuda/stream.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>

namespace qw38::cuda {

// T-01 decode projection geometry. Tuning, not ABI.
inline constexpr int kDecodeWarps = 8;
inline constexpr int kDecodeThreads = 256;
inline constexpr int kDecodeTileRows = 8;
inline constexpr int kDecodeTileK = 256;
inline constexpr int kDecodeMaxK = 17408;  // MLP down input width
// Largest logical N whose row-tile padding is representable by padded_n.
inline constexpr std::uint32_t kDecodeMaxN =
    std::numeric_limits<std::uint32_t>::max() -
    (static_cast<std::uint32_t>(kDecodeTileRows) - 1u);

// Wire IDs match format::PhysicalLayoutId / LogicalQuantizerId (A-02).
inline constexpr std::uint16_t kDecodeLayoutQ4G64V0 = 0x0201;
inline constexpr std::uint16_t kDecodeLayoutQ8G32V0 = 0x0202;
inline constexpr std::uint16_t kDecodeLayoutBf16DenseTileV0 = 0x0203;
inline constexpr std::uint16_t kDecodeLayoutQ4G64CandidateV1 = 0x020B;
inline constexpr std::uint16_t kDecodeLayoutQ8G32CandidateV1 = 0x020C;
inline constexpr std::uint16_t kDecodeLayoutQ4KCandidateV2 = 0x020D;
inline constexpr std::uint16_t kDecodeQuantizerNone = 0x0100;
inline constexpr std::uint16_t kDecodeQuantizerQ4G64V0 = 0x0101;
inline constexpr std::uint16_t kDecodeQuantizerQ8G32V0 = 0x0102;
inline constexpr std::uint16_t kDecodeQuantizerQ4G64CandidateV1 = 0x0103;
inline constexpr std::uint16_t kDecodeQuantizerQ8G32CandidateV1 = 0x0104;
inline constexpr std::uint16_t kDecodeQuantizerQ4KCandidateV2 = 0x0105;

enum class DecodeActivationPolicy : std::uint8_t { Bf16 = 1 };

enum class DecodeEpilogue : std::uint8_t {
  StoreBf16 = 1,
  StoreFp32 = 2,
  ResidualAddFp32 = 3,
  // Paired gate/up only: y = BF16(SiLU(gate) * up). Does not materialize gate/up.
  SwigluStoreBf16 = 4,
};

enum class DecodeMemorySpace : std::uint8_t {
  Device = 1,
  Host = 2,
};

enum class DecodeDtype : std::uint8_t {
  Q4 = 1,
  Q8 = 2,
  Bf16 = 3,
  Fp16 = 4,
  Fp32 = 5,
};

inline constexpr std::uint16_t kDecodeLayoutBf16VectorV0 = 0x0205;
inline constexpr std::uint16_t kDecodeLayoutFp32VectorV0 = 0x020A;

// Complete borrowed operand metadata at the CUDA launch boundary. Matrix
// shapes are [N,K]; vectors use logical/padded N with K=1.
struct DecodeOperandView {
  void* pointer{};
  DecodeMemorySpace space{DecodeMemorySpace::Device};
  DecodeDtype dtype{};
  std::uint16_t layout{};
  std::uint32_t logical_n{};
  std::uint32_t logical_k{};
  std::uint32_t padded_n{};
  std::uint32_t padded_k{};
  std::uint64_t bytes{};
  std::uint32_t alignment{};
  bool writable{};

  friend bool operator==(DecodeOperandView const&,
                         DecodeOperandView const&) = default;
};

[[nodiscard]] constexpr DecodeOperandView decode_matrix_view(
    void* pointer, DecodeDtype dtype, std::uint16_t layout,
    std::uint32_t logical_n, std::uint32_t logical_k,
    std::uint32_t padded_n, std::uint32_t padded_k, std::uint64_t bytes,
    std::uint32_t alignment, bool writable = false,
    DecodeMemorySpace space = DecodeMemorySpace::Device) noexcept {
  return {pointer, space, dtype, layout, logical_n, logical_k,
          padded_n, padded_k, bytes, alignment, writable};
}

[[nodiscard]] constexpr DecodeOperandView decode_vector_view(
    void* pointer, DecodeDtype dtype, std::uint16_t layout,
    std::uint32_t elements, std::uint64_t bytes, std::uint32_t alignment,
    bool writable,
    DecodeMemorySpace space = DecodeMemorySpace::Device) noexcept {
  return {pointer, space, dtype, layout, elements, 1, elements, 1,
          bytes, alignment, writable};
}

// Sequential CPU GEMV vs warp-tree FP32 reduction. Decoded BF16 operands match
// TASK-006; this budget covers accumulation-order differences only.
namespace decode_mmv_tol {
inline constexpr float kFp32Abs = 1.0e-3f;
inline constexpr float kFp32Rel = 1.0e-4f;
inline constexpr float kBf16StoreAbs = 8.0e-3f;
}  // namespace decode_mmv_tol

struct DecodeMmvDesc {
  std::uint16_t layout{};
  std::uint16_t quantizer{};
  DecodeActivationPolicy activation_policy{DecodeActivationPolicy::Bf16};
  std::uint32_t n{};
  std::uint32_t k{};
  std::uint32_t padded_n{};
  std::uint32_t padded_k{};
  DecodeOperandView codes{};
  // Q4_K uses this FP16-backed span for opaque 16-byte superblock metadata.
  DecodeOperandView scales{};
  DecodeOperandView input{};
  DecodeOperandView output{};
  // ResidualAddFp32 reads residual and writes output when output is supplied.
  // An empty output retains the legacy in-place residual update.
  DecodeOperandView residual{};
  DecodeEpilogue epilogue{DecodeEpilogue::StoreBf16};
};

struct DecodeMmvPairedDesc {
  DecodeMmvDesc a;
  DecodeMmvDesc b;
};

// One launch over two or three independent output-N ranges that share
// K/input/layout. This combines scheduling only; reductions stay per row.
struct DecodeMmvRangeDesc {
  std::span<DecodeMmvDesc const> ranges{};
};

[[nodiscard]] constexpr std::uint64_t decode_pad_n(std::uint32_t n) noexcept {
  if (n == 0) {
    return 0;
  }
  std::uint64_t const wide_n = n;
  std::uint64_t const tile_rows = kDecodeTileRows;
  std::uint64_t const rem = wide_n % tile_rows;
  return rem == 0 ? wide_n : wide_n + (tile_rows - rem);
}

[[nodiscard]] constexpr std::uint32_t decode_pad_k(std::uint32_t k) noexcept {
  if (k == 0) {
    return 0;
  }
  auto const rem = k % static_cast<std::uint32_t>(kDecodeTileK);
  return rem == 0 ? k : k + (static_cast<std::uint32_t>(kDecodeTileK) - rem);
}

[[nodiscard]] constexpr std::uint64_t decode_code_bytes(
    std::uint16_t layout, std::uint32_t padded_n, std::uint32_t padded_k) noexcept {
  std::uint64_t const nk =
      static_cast<std::uint64_t>(padded_n) * static_cast<std::uint64_t>(padded_k);
  if (layout == kDecodeLayoutQ4G64V0 ||
      layout == kDecodeLayoutQ4G64CandidateV1 ||
      layout == kDecodeLayoutQ4KCandidateV2) {
    return nk / 2u;
  }
  if (layout == kDecodeLayoutQ8G32V0 ||
      layout == kDecodeLayoutQ8G32CandidateV1) {
    return nk;
  }
  if (layout == kDecodeLayoutBf16DenseTileV0) {
    return nk * 2u;
  }
  return 0;
}

[[nodiscard]] constexpr std::uint64_t decode_scale_bytes(
    std::uint16_t layout, std::uint32_t padded_n, std::uint32_t padded_k) noexcept {
  std::uint64_t const rows = padded_n;
  if (layout == kDecodeLayoutQ4KCandidateV2) {
    return rows * (static_cast<std::uint64_t>(padded_k) / 256u) * 16u;
  }
  if (layout == kDecodeLayoutQ4G64V0 ||
      layout == kDecodeLayoutQ4G64CandidateV1) {
    return rows * (static_cast<std::uint64_t>(padded_k) / 64u) * 2u;
  }
  if (layout == kDecodeLayoutQ8G32V0 ||
      layout == kDecodeLayoutQ8G32CandidateV1) {
    return rows * (static_cast<std::uint64_t>(padded_k) / 32u) * 2u;
  }
  return 0;
}

[[nodiscard]] std::expected<void, Error> launch_decode_mmv(
    DecodeMmvDesc const& desc, Stream const& stream);

[[nodiscard]] std::expected<void, Error> launch_decode_mmv_paired(
    DecodeMmvPairedDesc const& desc, Stream const& stream);

// Grouped GDN a/b: two BF16 dense-tile matrices, FP32 outputs, one launch.
[[nodiscard]] std::expected<void, Error> launch_decode_ab_bf16(
    DecodeMmvPairedDesc const& desc, Stream const& stream);

// Two or three Q4 or BF16-control projections in one launch into separate slices.
[[nodiscard]] std::expected<void, Error> launch_decode_mmv_ranges(
    DecodeMmvRangeDesc const& desc, Stream const& stream);

}  // namespace qw38::cuda
