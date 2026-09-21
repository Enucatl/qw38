#include "format/unpack.hpp"

#include "format/floatcvt.hpp"
#include "format/layout.hpp"

namespace qw38::format {
namespace {

FormatError unpack_err(FormatErrorCode code, std::string_view field,
                       std::string_view detail) {
  return make_error(code, 0, field, detail);
}

std::expected<std::int8_t, FormatError> decode_q4_nibble(std::uint8_t nibble) {
  nibble = static_cast<std::uint8_t>(nibble & 0x0Fu);
  if (nibble == 0x08u) {
    return std::unexpected(unpack_err(FormatErrorCode::InvalidSpan, "unpack.q4",
                                      "forbidden Q4 code -8"));
  }
  if ((nibble & 0x08u) != 0) {
    return static_cast<std::int8_t>(static_cast<int>(nibble) - 16);
  }
  return static_cast<std::int8_t>(nibble);
}

std::expected<std::int8_t, FormatError> decode_q8_byte(std::uint8_t b) {
  if (b == 0x80u) {
    return std::unexpected(unpack_err(FormatErrorCode::InvalidSpan, "unpack.q8",
                                      "forbidden Q8 code -128"));
  }
  return static_cast<std::int8_t>(b);
}

int qmax_of(LogicalQuantizerId id) noexcept {
  if (id == LogicalQuantizerId::Q4G64V0) {
    return 7;
  }
  if (id == LogicalQuantizerId::Q8G32V0) {
    return 127;
  }
  return 0;
}

}  // namespace

std::expected<LogicalWeightCodes, FormatError> unpack_cuda_v0(
    LogicalQuantizerId quantizer, PhysicalLayoutId layout, std::uint64_t n,
    std::uint64_t k, std::span<std::byte const> codes,
    std::span<std::byte const> scales) {
  if (!quantizer_layout_pair_ok(quantizer, layout) ||
      (layout != PhysicalLayoutId::CudaQ4G64V0 &&
       layout != PhysicalLayoutId::CudaQ8G32V0)) {
    return std::unexpected(unpack_err(
        FormatErrorCode::InvalidQuantizerLayoutPair, "unpack",
        "logical quantizer and physical layout disagree"));
  }
  std::uint32_t group = 0;
  std::uint32_t packed_row = 0;
  if (layout == PhysicalLayoutId::CudaQ4G64V0) {
    group = kQ4GroupSize;
    packed_row = kQ4PackedBytesPerTileRow;
  } else {
    group = kQ8GroupSize;
    packed_row = kQ8PackedBytesPerTileRow;
  }
  if (n == 0 || k == 0 || k % group != 0) {
    return std::unexpected(unpack_err(FormatErrorCode::InvalidShape, "unpack.shape",
                                      "logical K must divide the group size"));
  }
  auto const padded_n = dense_pad_n(n);
  auto const padded_k = dense_pad_k(k);
  auto const tiles_n = padded_n / kDenseTileRows;
  auto const tiles_k = padded_k / kDenseTileK;
  auto const groups_in_tile = kDenseTileK / group;
  auto const want_codes = tiles_n * tiles_k * kDenseTileRows * packed_row;
  auto const want_scales =
      tiles_n * tiles_k * kDenseTileRows * groups_in_tile * kFp16Size;
  if (codes.size() != want_codes) {
    return std::unexpected(unpack_err(FormatErrorCode::InvalidSpan, "unpack.codes",
                                      "packed code length does not match layout"));
  }
  if (scales.size() != want_scales) {
    return std::unexpected(unpack_err(FormatErrorCode::InvalidSpan, "unpack.scales",
                                      "packed scale length does not match layout"));
  }

  LogicalWeightCodes out;
  out.quantizer = quantizer;
  out.n = n;
  out.k = k;
  out.group_size = group;
  out.qmax = qmax_of(quantizer);
  out.codes.resize(static_cast<std::size_t>(n * k));
  auto const groups_row = k / group;
  out.scales.resize(static_cast<std::size_t>(n * groups_row));

  for (std::uint64_t row = 0; row < n; ++row) {
    auto const tn = row / kDenseTileRows;
    auto const r = static_cast<std::uint32_t>(row % kDenseTileRows);
    for (std::uint64_t g = 0; g < groups_row; ++g) {
      auto const col0 = g * group;
      auto const tk = col0 / kDenseTileK;
      auto const g_in_tile =
          static_cast<std::uint32_t>((col0 % kDenseTileK) / group);
      auto const scale_index =
          ((tn * tiles_k + tk) * kDenseTileRows + r) * groups_in_tile + g_in_tile;
      out.scales[static_cast<std::size_t>(row * groups_row + g)] =
          load_u16_le(scales.data() + scale_index * kFp16Size);

      for (std::uint32_t i = 0; i < group; ++i) {
        auto const col = col0 + i;
        auto const c_in_tile = static_cast<std::uint32_t>(col % kDenseTileK);
        auto const tile_row =
            (tn * tiles_k + tk) * kDenseTileRows + r;
        std::int8_t decoded = 0;
        if (layout == PhysicalLayoutId::CudaQ4G64V0) {
          auto const byte_i = c_in_tile / 2;
          auto const raw = static_cast<std::uint8_t>(
              codes[static_cast<std::size_t>(tile_row * packed_row + byte_i)]);
          auto const nib = (c_in_tile % 2u == 0) ? (raw & 0x0Fu) : (raw >> 4);
          auto got = decode_q4_nibble(nib);
          if (!got) {
            return std::unexpected(got.error());
          }
          decoded = *got;
        } else {
          auto const raw = static_cast<std::uint8_t>(
              codes[static_cast<std::size_t>(tile_row * packed_row + c_in_tile)]);
          auto got = decode_q8_byte(raw);
          if (!got) {
            return std::unexpected(got.error());
          }
          decoded = *got;
        }
        out.codes[static_cast<std::size_t>(row * k + col)] = decoded;
      }
    }
  }
  return out;
}

std::expected<std::vector<std::byte>, FormatError> unpack_bf16_dense_tile_v0(
    std::span<std::byte const> tiled, std::uint64_t n, std::uint64_t k) {
  if (n == 0 || k == 0 || n % kDenseTileRows != 0 || k % kDenseTileK != 0) {
    return std::unexpected(unpack_err(FormatErrorCode::InvalidShape, "unpack.bf16",
                                      "BF16 dense tile requires N%8==0 and K%256==0"));
  }
  auto const elems = n * k;
  if (elems / k != n || tiled.size() != elems * kBf16Size) {
    return std::unexpected(unpack_err(FormatErrorCode::InvalidSpan, "unpack.bf16",
                                      "tiled BF16 length must equal 2*N*K"));
  }
  std::vector<std::byte> out(static_cast<std::size_t>(elems * kBf16Size));
  auto const tiles_k = k / kDenseTileK;
  for (std::uint64_t row = 0; row < n; ++row) {
    auto const tn = row / kDenseTileRows;
    auto const r = row % kDenseTileRows;
    for (std::uint64_t tk = 0; tk < tiles_k; ++tk) {
      auto const src =
          ((tn * tiles_k + tk) * kDenseTileRows + r) * kBf16PackedBytesPerTileRow;
      auto const dst = (row * k + tk * kDenseTileK) * kBf16Size;
      for (std::uint32_t c = 0; c < kDenseTileK * kBf16Size; ++c) {
        out[static_cast<std::size_t>(dst + c)] =
            tiled[static_cast<std::size_t>(src + c)];
      }
    }
  }
  return out;
}

}  // namespace qw38::format
