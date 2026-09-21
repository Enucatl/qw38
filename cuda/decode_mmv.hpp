#pragma once

#include "cuda/error.hpp"
#include "cuda/stream.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>

namespace qw38::cuda {

// T-01 decode projection geometry. Tuning, not ABI.
inline constexpr int kDecodeWarps = 8;
inline constexpr int kDecodeThreads = 256;
inline constexpr int kDecodeTileRows = 8;
inline constexpr int kDecodeTileK = 256;
inline constexpr int kDecodeMaxK = 17408;  // MLP down input width

// Wire IDs match format::PhysicalLayoutId / LogicalQuantizerId (A-02).
inline constexpr std::uint16_t kDecodeLayoutQ4G64V0 = 0x0201;
inline constexpr std::uint16_t kDecodeLayoutQ8G32V0 = 0x0202;
inline constexpr std::uint16_t kDecodeLayoutBf16DenseTileV0 = 0x0203;
inline constexpr std::uint16_t kDecodeQuantizerNone = 0x0100;
inline constexpr std::uint16_t kDecodeQuantizerQ4G64V0 = 0x0101;
inline constexpr std::uint16_t kDecodeQuantizerQ8G32V0 = 0x0102;

enum class DecodeEpilogue : std::uint8_t {
  StoreBf16 = 1,
  StoreFp32 = 2,
  ResidualAddFp32 = 3,
  // Paired gate/up only: y = BF16(SiLU(gate) * up). Does not materialize gate/up.
  SwigluStoreBf16 = 4,
};

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
  std::uint32_t n{};
  std::uint32_t k{};
  std::uint32_t padded_n{};
  std::uint32_t padded_k{};
  std::byte const* codes{};
  std::byte const* scales{};
  std::uint64_t codes_bytes{};
  std::uint64_t scales_bytes{};
  std::uint16_t const* input{};
  void* output{};
  float* residual{};
  DecodeEpilogue epilogue{DecodeEpilogue::StoreBf16};
};

struct DecodeMmvPairedDesc {
  DecodeMmvDesc a;
  std::byte const* codes_b{};
  std::byte const* scales_b{};
  std::uint64_t codes_b_bytes{};
  std::uint64_t scales_b_bytes{};
  void* output_b{};
  float* residual_b{};
};

[[nodiscard]] constexpr std::uint32_t decode_pad_n(std::uint32_t n) noexcept {
  if (n == 0) {
    return 0;
  }
  auto const rem = n % static_cast<std::uint32_t>(kDecodeTileRows);
  return rem == 0 ? n : n + (static_cast<std::uint32_t>(kDecodeTileRows) - rem);
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
  if (layout == kDecodeLayoutQ4G64V0) {
    return nk / 2u;
  }
  if (layout == kDecodeLayoutQ8G32V0) {
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
  if (layout == kDecodeLayoutQ4G64V0) {
    return rows * (static_cast<std::uint64_t>(padded_k) / 64u) * 2u;
  }
  if (layout == kDecodeLayoutQ8G32V0) {
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

}  // namespace qw38::cuda
