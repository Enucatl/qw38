#pragma once

#include "compiler/error.hpp"
#include "compiler/identity.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string_view>
#include <vector>

namespace qw38::compiler {

[[nodiscard]] bool bf16_is_finite(std::uint16_t bits) noexcept;

[[nodiscard]] std::expected<void, CompilerError> require_all_finite_bf16(
    std::span<std::byte const> bytes, std::string_view tensor_name);

// Logical W[N,K] row-major BF16 -> physical [N/8, K/256, 8, 256].
[[nodiscard]] std::expected<std::vector<std::byte>, CompilerError>
tile_nk_from_row_major(std::span<std::byte const> src, std::uint64_t n,
                       std::uint64_t k);

[[nodiscard]] std::expected<std::vector<std::byte>, CompilerError>
row_major_from_tile_nk(std::span<std::byte const> tiled, std::uint64_t n,
                       std::uint64_t k);

// Source [channel, 1, tap] -> stored [tap, channel].
[[nodiscard]] std::expected<std::vector<std::byte>, CompilerError>
conv_to_tap_major(std::span<std::byte const> src, std::uint64_t channels,
                  std::uint64_t taps);

[[nodiscard]] std::expected<std::vector<std::byte>, CompilerError>
conv_from_tap_major(std::span<std::byte const> tap_major,
                    std::uint64_t channels, std::uint64_t taps);

// ω_j = θ^{-2j/d_rot}, j=0..31, stored IEEE FP32 little-endian.
[[nodiscard]] std::array<std::byte, kRopeFreqs * 4> generate_rope_inv_freq();

struct TileExtents {
  std::uint64_t n{};
  std::uint64_t k{};
  std::uint64_t tiles_n{};
  std::uint64_t tiles_k{};
};

[[nodiscard]] std::expected<TileExtents, CompilerError> dense_tile_extents(
    std::uint64_t n, std::uint64_t k);

// Physical index of logical (row, col) inside the tiled payload, in BF16 elements.
[[nodiscard]] std::expected<std::uint64_t, CompilerError> tiled_element_index(
    std::uint64_t n, std::uint64_t k, std::uint64_t row,
    std::uint64_t col);

}  // namespace qw38::compiler
