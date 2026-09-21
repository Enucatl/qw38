#include "format/pack.hpp"

#include "format/floatcvt.hpp"
#include "format/layout.hpp"

#include <cstring>

namespace qw38::format {
namespace {

FormatError pack_err(FormatErrorCode code, std::string_view field,
                     std::string_view detail) {
  return make_error(code, 0, field, detail);
}

std::uint8_t q4_nibble(std::int8_t code, FormatError* err) {
  if (code < -7 || code > 7) {
    *err = pack_err(FormatErrorCode::InvalidSpan, "pack.q4",
                    "Q4 code outside [-7,7] or forbidden -8");
    return 0;
  }
  return static_cast<std::uint8_t>(code) & 0x0Fu;
}

std::uint8_t q8_byte(std::int8_t code, FormatError* err) {
  if (code == static_cast<std::int8_t>(-128)) {
    *err = pack_err(FormatErrorCode::InvalidSpan, "pack.q8",
                    "Q8 code outside [-127,127] or forbidden -128");
    return 0;
  }
  return static_cast<std::uint8_t>(code);
}

}  // namespace

bool quantizer_layout_pair_ok(LogicalQuantizerId quantizer,
                              PhysicalLayoutId layout) noexcept {
  if (quantizer == LogicalQuantizerId::Q4G64V0) {
    return layout == PhysicalLayoutId::CudaQ4G64V0;
  }
  if (quantizer == LogicalQuantizerId::Q8G32V0) {
    return layout == PhysicalLayoutId::CudaQ8G32V0;
  }
  if (quantizer == LogicalQuantizerId::None) {
    return layout == PhysicalLayoutId::CudaBf16DenseTileV0 ||
           layout == PhysicalLayoutId::CudaBf16RowMajorV0 ||
           layout == PhysicalLayoutId::CudaBf16VectorV0 ||
           layout == PhysicalLayoutId::CudaBf16TapMajorV0;
  }
  return false;
}

std::uint64_t dense_pad_n(std::uint64_t n) noexcept {
  if (n == 0) {
    return 0;
  }
  auto const rem = n % kDenseTileRows;
  return rem == 0 ? n : n + (kDenseTileRows - rem);
}

std::uint64_t dense_pad_k(std::uint64_t k) noexcept {
  if (k == 0) {
    return 0;
  }
  auto const rem = k % kDenseTileK;
  return rem == 0 ? k : k + (kDenseTileK - rem);
}

std::expected<PackedMatrix, FormatError> pack_cuda_v0(
    LogicalQuantizerId quantizer, PhysicalLayoutId layout,
    LogicalWeightCodes const& logical) {
  if (!quantizer_layout_pair_ok(quantizer, layout) ||
      logical.quantizer != quantizer) {
    return std::unexpected(pack_err(FormatErrorCode::InvalidQuantizerLayoutPair,
                                    "pack",
                                    "logical quantizer and physical layout disagree"));
  }
  if (logical.n == 0 || logical.k == 0 || logical.group_size == 0 ||
      logical.k % logical.group_size != 0) {
    return std::unexpected(pack_err(FormatErrorCode::InvalidShape, "pack.shape",
                                    "logical N,K,group are invalid"));
  }
  std::uint32_t packed_row = 0;
  std::uint32_t groups_in_tile = 0;
  if (layout == PhysicalLayoutId::CudaQ4G64V0) {
    if (logical.group_size != kQ4GroupSize) {
      return std::unexpected(pack_err(FormatErrorCode::InvalidMapping, "pack",
                                      "Q4G64 group size must be 64"));
    }
    packed_row = kQ4PackedBytesPerTileRow;
    groups_in_tile = kDenseTileK / kQ4GroupSize;
  } else {
    if (logical.group_size != kQ8GroupSize) {
      return std::unexpected(pack_err(FormatErrorCode::InvalidMapping, "pack",
                                      "Q8G32 group size must be 32"));
    }
    packed_row = kQ8PackedBytesPerTileRow;
    groups_in_tile = kDenseTileK / kQ8GroupSize;
  }

  auto const need = logical.n * logical.k;
  if (need / logical.k != logical.n ||
      logical.codes.size() != need) {
    return std::unexpected(pack_err(FormatErrorCode::InvalidSpan, "pack.codes",
                                    "code count must equal N*K"));
  }
  auto const groups_row = logical.k / logical.group_size;
  if (logical.scales.size() != logical.n * groups_row) {
    return std::unexpected(pack_err(FormatErrorCode::InvalidSpan, "pack.scales",
                                    "scale count must equal N*(K/group)"));
  }

  PackedMatrix out;
  out.layout = layout;
  out.quantizer = quantizer;
  out.logical_n = logical.n;
  out.logical_k = logical.k;
  out.padded_n = dense_pad_n(logical.n);
  out.padded_k = dense_pad_k(logical.k);
  auto const tiles_n = out.padded_n / kDenseTileRows;
  auto const tiles_k = out.padded_k / kDenseTileK;
  auto tile_code = checked_mul(tiles_n, tiles_k, 0, "pack.tiles");
  if (!tile_code) {
    return std::unexpected(tile_code.error());
  }
  auto rows = checked_mul(*tile_code, kDenseTileRows, 0, "pack.rows");
  if (!rows) {
    return std::unexpected(rows.error());
  }
  auto code_n = checked_mul(*rows, packed_row, 0, "pack.codes");
  if (!code_n) {
    return std::unexpected(code_n.error());
  }
  auto scale_n =
      checked_mul(*rows, groups_in_tile * kFp16Size, 0, "pack.scales");
  if (!scale_n) {
    return std::unexpected(scale_n.error());
  }
  out.codes.assign(static_cast<std::size_t>(*code_n), std::byte{0});
  out.scales.assign(static_cast<std::size_t>(*scale_n), std::byte{0});

  FormatError err{};
  bool failed = false;
  auto fail_once = [&](FormatError e) {
    if (!failed) {
      err = std::move(e);
      failed = true;
    }
  };

  for (std::uint64_t tn = 0; tn < tiles_n; ++tn) {
    for (std::uint64_t tk = 0; tk < tiles_k; ++tk) {
      for (std::uint32_t r = 0; r < kDenseTileRows; ++r) {
        auto const row = tn * kDenseTileRows + r;
        auto const tile_row_index =
            ((tn * tiles_k + tk) * kDenseTileRows + r);
        auto const code_base = tile_row_index * packed_row;
        auto const scale_base =
            tile_row_index * groups_in_tile * kFp16Size;
        for (std::uint32_t g = 0; g < groups_in_tile; ++g) {
          auto const col0 = tk * kDenseTileK + g * logical.group_size;
          if (row < logical.n && col0 < logical.k) {
            auto const lg = col0 / logical.group_size;
            auto const sb =
                logical.scales[static_cast<std::size_t>(row * groups_row + lg)];
            store_u16_le(out.scales.data() + scale_base + g * kFp16Size, sb);
          }
        }
        if (layout == PhysicalLayoutId::CudaQ4G64V0) {
          for (std::uint32_t c = 0; c < kDenseTileK; c += 2) {
            auto const col0 = tk * kDenseTileK + c;
            std::int8_t a = 0;
            std::int8_t b = 0;
            if (row < logical.n && col0 < logical.k) {
              a = logical.codes[static_cast<std::size_t>(row * logical.k + col0)];
            }
            if (row < logical.n && col0 + 1 < logical.k) {
              b = logical.codes[static_cast<std::size_t>(row * logical.k +
                                                         col0 + 1)];
            }
            auto const lo = q4_nibble(a, &err);
            auto const hi = q4_nibble(b, &err);
            if (a < -7 || a > 7 || b < -7 || b > 7) {
              fail_once(err);
            }
            out.codes[static_cast<std::size_t>(code_base + c / 2)] =
                static_cast<std::byte>(static_cast<std::uint8_t>(lo | (hi << 4)));
          }
        } else {
          for (std::uint32_t c = 0; c < kDenseTileK; ++c) {
            auto const col = tk * kDenseTileK + c;
            std::int8_t v = 0;
            if (row < logical.n && col < logical.k) {
              v = logical.codes[static_cast<std::size_t>(row * logical.k + col)];
            }
            auto const b = q8_byte(v, &err);
            if (v == static_cast<std::int8_t>(-128)) {
              fail_once(err);
            }
            out.codes[static_cast<std::size_t>(code_base + c)] =
                static_cast<std::byte>(b);
          }
        }
      }
    }
  }
  if (failed) {
    return std::unexpected(err);
  }
  return out;
}

std::expected<PackedMatrix, FormatError> pack_bf16_dense_tile_v0(
    std::span<std::byte const> row_major, std::uint64_t n, std::uint64_t k) {
  if (n == 0 || k == 0 || n % kDenseTileRows != 0 || k % kDenseTileK != 0) {
    return std::unexpected(pack_err(FormatErrorCode::InvalidShape, "pack.bf16",
                                    "BF16 dense tile requires N%8==0 and K%256==0"));
  }
  auto const elems = n * k;
  if (elems / k != n || row_major.size() != elems * kBf16Size) {
    return std::unexpected(pack_err(FormatErrorCode::InvalidSpan, "pack.bf16",
                                    "BF16 payload length must equal 2*N*K"));
  }
  PackedMatrix out;
  out.layout = PhysicalLayoutId::CudaBf16DenseTileV0;
  out.quantizer = LogicalQuantizerId::None;
  out.logical_n = n;
  out.logical_k = k;
  out.padded_n = n;
  out.padded_k = k;
  out.codes.resize(static_cast<std::size_t>(elems * kBf16Size));
  auto const tiles_n = n / kDenseTileRows;
  auto const tiles_k = k / kDenseTileK;
  for (std::uint64_t tn = 0; tn < tiles_n; ++tn) {
    for (std::uint64_t tk = 0; tk < tiles_k; ++tk) {
      for (std::uint32_t r = 0; r < kDenseTileRows; ++r) {
        auto const row = tn * kDenseTileRows + r;
        auto const src = (row * k + tk * kDenseTileK) * kBf16Size;
        auto const dst =
            ((tn * tiles_k + tk) * kDenseTileRows + r) * kBf16PackedBytesPerTileRow;
        std::memcpy(out.codes.data() + dst, row_major.data() + src,
                    kDenseTileK * kBf16Size);
      }
    }
  }
  return out;
}

}  // namespace qw38::format
