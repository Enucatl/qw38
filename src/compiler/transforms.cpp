#include "compiler/transforms.hpp"

#include "format/constants.hpp"

#include <cmath>
#include <cstring>

namespace qw38::compiler {
namespace {

using qw38::format::kBf16Size;
using qw38::format::kDenseTileK;
using qw38::format::kDenseTileRows;

CompilerError size_error(std::string_view field, std::string_view detail) {
  return make_error(CompilerErrorCode::ShapeMismatch, field, detail);
}

std::expected<void, CompilerError> require_size(std::span<std::byte const> bytes,
                                                std::uint64_t elems,
                                                std::string_view field) {
  auto const need = elems * kBf16Size;
  if (bytes.size() != need) {
    return std::unexpected(size_error(field, "byte length does not match geometry"));
  }
  return {};
}

}  // namespace

bool bf16_is_finite(std::uint16_t bits) noexcept {
  return (bits & 0x7F80u) != 0x7F80u;
}

std::expected<void, CompilerError> require_all_finite_bf16(
    std::span<std::byte const> bytes, std::string_view tensor_name) {
  if (bytes.size() % kBf16Size != 0) {
    return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch,
                                      tensor_name, "BF16 payload is not even"));
  }
  auto const n = bytes.size() / kBf16Size;
  for (std::size_t i = 0; i < n; ++i) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, bytes.data() + i * kBf16Size, kBf16Size);
    if (!bf16_is_finite(bits)) {
      return std::unexpected(make_error(CompilerErrorCode::Nonfinite, tensor_name,
                                        "nonfinite BF16 value"));
    }
  }
  return {};
}

std::expected<TileExtents, CompilerError> dense_tile_extents(std::uint64_t n,
                                                             std::uint64_t k) {
  if (n == 0 || k == 0 || n % kDenseTileRows != 0 || k % kDenseTileK != 0) {
    return std::unexpected(size_error("dense.tile",
                                      "N must divide 8 and K must divide 256"));
  }
  return TileExtents{.n = n,
                     .k = k,
                     .tiles_n = n / kDenseTileRows,
                     .tiles_k = k / kDenseTileK};
}

std::uint64_t tiled_element_index([[maybe_unused]] std::uint64_t n, std::uint64_t k,
                                  std::uint64_t row, std::uint64_t col) noexcept {
  auto const tiles_k = k / kDenseTileK;
  auto const tile_n = row / kDenseTileRows;
  auto const row_in_tile = row % kDenseTileRows;
  auto const tile_k = col / kDenseTileK;
  auto const col_in_tile = col % kDenseTileK;
  return ((tile_n * tiles_k + tile_k) * kDenseTileRows + row_in_tile) *
             kDenseTileK +
         col_in_tile;
}

std::expected<std::vector<std::byte>, CompilerError> tile_nk_from_row_major(
    std::span<std::byte const> src, std::uint64_t n, std::uint64_t k) {
  auto extents = dense_tile_extents(n, k);
  if (!extents) {
    return std::unexpected(extents.error());
  }
  if (auto st = require_size(src, n * k, "dense.src"); !st) {
    return std::unexpected(st.error());
  }
  std::vector<std::byte> out(src.size());
  for (std::uint64_t row = 0; row < n; ++row) {
    for (std::uint64_t col = 0; col < k; ++col) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, src.data() + (row * k + col) * kBf16Size, kBf16Size);
      std::memcpy(out.data() + tiled_element_index(n, k, row, col) * kBf16Size,
                  &bits, kBf16Size);
    }
  }
  return out;
}

std::expected<std::vector<std::byte>, CompilerError> row_major_from_tile_nk(
    std::span<std::byte const> tiled, std::uint64_t n, std::uint64_t k) {
  auto extents = dense_tile_extents(n, k);
  if (!extents) {
    return std::unexpected(extents.error());
  }
  if (auto st = require_size(tiled, n * k, "dense.tiled"); !st) {
    return std::unexpected(st.error());
  }
  std::vector<std::byte> out(tiled.size());
  for (std::uint64_t row = 0; row < n; ++row) {
    for (std::uint64_t col = 0; col < k; ++col) {
      std::uint16_t bits = 0;
      std::memcpy(&bits,
                  tiled.data() + tiled_element_index(n, k, row, col) * kBf16Size,
                  kBf16Size);
      std::memcpy(out.data() + (row * k + col) * kBf16Size, &bits, kBf16Size);
    }
  }
  return out;
}

std::expected<std::vector<std::byte>, CompilerError> conv_to_tap_major(
    std::span<std::byte const> src, std::uint64_t channels,
    std::uint64_t taps) {
  if (channels == 0 || taps == 0) {
    return std::unexpected(size_error("conv", "channels and taps must be > 0"));
  }
  if (auto st = require_size(src, channels * taps, "conv.src"); !st) {
    return std::unexpected(st.error());
  }
  std::vector<std::byte> out(src.size());
  for (std::uint64_t c = 0; c < channels; ++c) {
    for (std::uint64_t t = 0; t < taps; ++t) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, src.data() + (c * taps + t) * kBf16Size, kBf16Size);
      std::memcpy(out.data() + (t * channels + c) * kBf16Size, &bits, kBf16Size);
    }
  }
  return out;
}

std::expected<std::vector<std::byte>, CompilerError> conv_from_tap_major(
    std::span<std::byte const> tap_major, std::uint64_t channels,
    std::uint64_t taps) {
  if (channels == 0 || taps == 0) {
    return std::unexpected(size_error("conv", "channels and taps must be > 0"));
  }
  if (auto st = require_size(tap_major, channels * taps, "conv.tap_major"); !st) {
    return std::unexpected(st.error());
  }
  std::vector<std::byte> out(tap_major.size());
  for (std::uint64_t c = 0; c < channels; ++c) {
    for (std::uint64_t t = 0; t < taps; ++t) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, tap_major.data() + (t * channels + c) * kBf16Size,
                  kBf16Size);
      std::memcpy(out.data() + (c * taps + t) * kBf16Size, &bits, kBf16Size);
    }
  }
  return out;
}

std::array<std::byte, kRopeFreqs * 4> generate_rope_inv_freq() {
  std::array<std::byte, kRopeFreqs * 4> out{};
  for (std::uint32_t j = 0; j < kRopeFreqs; ++j) {
    double const exponent = -2.0 * static_cast<double>(j) /
                            static_cast<double>(kRotaryDim);
    auto const value = static_cast<float>(std::pow(kRopeTheta, exponent));
    std::uint32_t bits = 0;
    static_assert(sizeof(float) == 4);
    std::memcpy(&bits, &value, sizeof(bits));
    out[j * 4 + 0] = static_cast<std::byte>(bits & 0xFFu);
    out[j * 4 + 1] = static_cast<std::byte>((bits >> 8) & 0xFFu);
    out[j * 4 + 2] = static_cast<std::byte>((bits >> 16) & 0xFFu);
    out[j * 4 + 3] = static_cast<std::byte>((bits >> 24) & 0xFFu);
  }
  return out;
}

}  // namespace qw38::compiler
