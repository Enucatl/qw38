#include "compiler/transforms.hpp"

#include "format/constants.hpp"

#include <cstring>
#include <limits>

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
  if (elems > std::numeric_limits<std::uint64_t>::max() / kBf16Size ||
      elems > std::numeric_limits<std::size_t>::max() / kBf16Size) {
    return std::unexpected(make_error(CompilerErrorCode::Unrepresentable, field,
                                      "BF16 byte size overflows"));
  }
  auto const need = elems * kBf16Size;
  if (bytes.size() != need) {
    return std::unexpected(size_error(field, "byte length does not match geometry"));
  }
  return {};
}

std::expected<std::uint64_t, CompilerError> element_count(
    std::uint64_t a, std::uint64_t b, std::string_view field) {
  if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b) {
    return std::unexpected(make_error(CompilerErrorCode::Unrepresentable, field,
                                      "element count overflows"));
  }
  return a * b;
}

std::uint64_t tiled_index_unchecked(std::uint64_t k, std::uint64_t row,
                                    std::uint64_t col) noexcept {
  auto const tiles_k = k / kDenseTileK;
  auto const tile_n = row / kDenseTileRows;
  auto const row_in_tile = row % kDenseTileRows;
  auto const tile_k = col / kDenseTileK;
  auto const col_in_tile = col % kDenseTileK;
  return ((tile_n * tiles_k + tile_k) * kDenseTileRows + row_in_tile) *
             kDenseTileK +
         col_in_tile;
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
  auto elements = element_count(n, k, "dense.geometry");
  if (!elements) return std::unexpected(elements.error());
  if (*elements > std::numeric_limits<std::size_t>::max() / kBf16Size) {
    return std::unexpected(make_error(CompilerErrorCode::Unrepresentable,
                                      "dense.geometry", "byte size overflows"));
  }
  return TileExtents{.n = n,
                     .k = k,
                     .tiles_n = n / kDenseTileRows,
                     .tiles_k = k / kDenseTileK};
}

std::expected<std::uint64_t, CompilerError> tiled_element_index(
    std::uint64_t n, std::uint64_t k, std::uint64_t row, std::uint64_t col) {
  if (auto extents = dense_tile_extents(n, k); !extents) {
    return std::unexpected(extents.error());
  }
  if (row >= n || col >= k) {
    return std::unexpected(size_error("dense.index", "coordinate is out of range"));
  }
  return tiled_index_unchecked(k, row, col);
}

std::expected<std::vector<std::byte>, CompilerError> tile_nk_from_row_major(
    std::span<std::byte const> src, std::uint64_t n, std::uint64_t k) {
  auto extents = dense_tile_extents(n, k);
  if (!extents) {
    return std::unexpected(extents.error());
  }
  auto elements = element_count(n, k, "dense.src");
  if (!elements) return std::unexpected(elements.error());
  if (auto st = require_size(src, *elements, "dense.src"); !st) {
    return std::unexpected(st.error());
  }
  std::vector<std::byte> out(src.size());
  for (std::uint64_t row = 0; row < n; ++row) {
    for (std::uint64_t col = 0; col < k; ++col) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, src.data() + (row * k + col) * kBf16Size, kBf16Size);
      std::memcpy(out.data() + tiled_index_unchecked(k, row, col) * kBf16Size,
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
  auto elements = element_count(n, k, "dense.tiled");
  if (!elements) return std::unexpected(elements.error());
  if (auto st = require_size(tiled, *elements, "dense.tiled"); !st) {
    return std::unexpected(st.error());
  }
  std::vector<std::byte> out(tiled.size());
  for (std::uint64_t row = 0; row < n; ++row) {
    for (std::uint64_t col = 0; col < k; ++col) {
      std::uint16_t bits = 0;
      std::memcpy(&bits,
                  tiled.data() + tiled_index_unchecked(k, row, col) * kBf16Size,
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
  auto elements = element_count(channels, taps, "conv.src");
  if (!elements) return std::unexpected(elements.error());
  if (auto st = require_size(src, *elements, "conv.src"); !st) {
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
  auto elements = element_count(channels, taps, "conv.tap_major");
  if (!elements) return std::unexpected(elements.error());
  if (auto st = require_size(tap_major, *elements, "conv.tap_major"); !st) {
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
  // Correctly rounded IEEE-754 FP32 values of 10,000,000^(-2j/64), derived
  // independently of the host libm. These bytes are part of the compiler's
  // revisioned artifact contract, not a platform-specific pow result.
  constexpr std::array<std::uint32_t, kRopeFreqs> kBits{
      0x3f800000u, 0x3f5e8d27u, 0x3f41791du, 0x3f2831b4u,
      0x3f1237d7u, 0x3efe3a16u, 0x3edd028bu, 0x3ec02211u,
      0x3ea7077au, 0x3e913494u, 0x3e7c7751u, 0x3e5b7aacu,
      0x3e3ecd65u, 0x3e25df51u, 0x3e10331eu, 0x3dfab7abu,
      0x3dd9f583u, 0x3dbd7b15u, 0x3da4b936u, 0x3d8f336fu,
      0x3d78fb1fu, 0x3d58730du, 0x3d3c2b1du, 0x3d239523u,
      0x3d0e3586u, 0x3cf741a7u, 0x3cd6f343u, 0x3cbadd79u,
      0x3ca27317u, 0x3c8d3960u, 0x3c758b3eu, 0x3c557622u};
  std::array<std::byte, kRopeFreqs * 4> out{};
  for (std::uint32_t j = 0; j < kRopeFreqs; ++j) {
    auto const bits = kBits[j];
    out[j * 4 + 0] = static_cast<std::byte>(bits & 0xFFu);
    out[j * 4 + 1] = static_cast<std::byte>((bits >> 8) & 0xFFu);
    out[j * 4 + 2] = static_cast<std::byte>((bits >> 16) & 0xFFu);
    out[j * 4 + 3] = static_cast<std::byte>((bits >> 24) & 0xFFu);
  }
  return out;
}

}  // namespace qw38::compiler
