#pragma once

#include "format/pack.hpp"

namespace qw38::format {

// cuda_nvfp4_v1: row-major nibble codes, CUTLASS SM120 K-major scales.
// The scale span starts with a 256-byte header (FP32 factor, then zeros).
inline constexpr std::uint64_t kNvFp4ScaleHeader = 256;
[[nodiscard]] constexpr std::uint64_t nvfp4_scale_count(std::uint64_t n,
                                                       std::uint64_t k) {
  return ((n + 127) / 128) * 128 * (k / 16);
}
[[nodiscard]] constexpr std::uint64_t nvfp4_scale_index(std::uint64_t row,
    std::uint64_t block, std::uint64_t padded_k) {
  return (row / 128) * 128 * (padded_k / 16) + (block / 4) * 512 +
         (row % 32) * 16 + ((row % 128) / 32) * 4 + block % 4;
}
[[nodiscard]] float nvfp4_e2m1(std::uint8_t code) noexcept;
[[nodiscard]] float nvfp4_e4m3(std::uint8_t code) noexcept;
[[nodiscard]] float nvfp4_factor(std::span<std::byte const> scales) noexcept;
[[nodiscard]] std::expected<void, FormatError> validate_nvfp4(
    std::uint64_t n, std::uint64_t k, std::span<std::byte const> codes,
    std::span<std::byte const> scales);

}  // namespace qw38::format
