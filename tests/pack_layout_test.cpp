#include "format/floatcvt.hpp"
#include "format/logical.hpp"
#include "format/pack.hpp"
#include "format/unpack.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

using qw38::format::dense_pad_k;
using qw38::format::dense_pad_n;
using qw38::format::error_message;
using qw38::format::FormatErrorCode;
using qw38::format::kDenseTileK;
using qw38::format::kDenseTileRows;
using qw38::format::kQ4GroupSize;
using qw38::format::kQ4PackedBytesPerTileRow;
using qw38::format::kQ8GroupSize;
using qw38::format::kQ8PackedBytesPerTileRow;
using qw38::format::load_u16_le;
using qw38::format::LogicalQuantizerId;
using qw38::format::LogicalWeightCodes;
using qw38::format::pack_cuda_v0;
using qw38::format::PhysicalLayoutId;
using qw38::format::quantizer_layout_pair_ok;
using qw38::format::store_u16_le;
using qw38::format::unpack_bf16_dense_tile_v0;

namespace {

int g_failures = 0;

void fail(std::string_view what) {
  std::cerr << "FAIL: " << what << '\n';
  ++g_failures;
}

void expect(bool cond, std::string_view what) {
  if (!cond) {
    fail(what);
  }
}

LogicalWeightCodes make_logical(LogicalQuantizerId id, std::uint64_t n,
                                std::uint64_t k) {
  LogicalWeightCodes m;
  m.quantizer = id;
  m.n = n;
  m.k = k;
  bool const q4 = id == LogicalQuantizerId::Q4G64V0 ||
                  id == LogicalQuantizerId::Q4G64CandidateV1;
  m.group_size = q4 ? kQ4GroupSize : kQ8GroupSize;
  m.qmax = q4 ? 7 : 127;
  m.codes.assign(static_cast<std::size_t>(n * k), 0);
  m.scales.assign(static_cast<std::size_t>(n * (k / m.group_size)), 0);
  return m;
}

// Independent golden encoder copied from the architecture byte-order sentence,
// not from pack_cuda_v0.
std::vector<std::byte> golden_q4_codes(LogicalWeightCodes const& m) {
  auto const pn = dense_pad_n(m.n);
  auto const pk = dense_pad_k(m.k);
  auto const tiles_n = pn / kDenseTileRows;
  auto const tiles_k = pk / kDenseTileK;
  std::vector<std::byte> out(static_cast<std::size_t>(
      tiles_n * tiles_k * kDenseTileRows * kQ4PackedBytesPerTileRow));
  for (std::uint64_t tn = 0; tn < tiles_n; ++tn) {
    for (std::uint64_t tk = 0; tk < tiles_k; ++tk) {
      for (std::uint32_t r = 0; r < kDenseTileRows; ++r) {
        auto const row = tn * kDenseTileRows + r;
        auto const base =
            ((tn * tiles_k + tk) * kDenseTileRows + r) * kQ4PackedBytesPerTileRow;
        for (std::uint32_t c = 0; c < kDenseTileK; c += 2) {
          auto nibble = [&](std::uint32_t off) -> std::uint8_t {
            auto const col = tk * kDenseTileK + off;
            std::int8_t code = 0;
            if (row < m.n && col < m.k) {
              code = m.codes[static_cast<std::size_t>(row * m.k + col)];
            }
            return static_cast<std::uint8_t>(code) & 0x0Fu;
          };
          out[static_cast<std::size_t>(base + c / 2)] =
              static_cast<std::byte>(nibble(c) | (nibble(c + 1) << 4));
        }
      }
    }
  }
  return out;
}

std::vector<std::byte> golden_q4_scales(LogicalWeightCodes const& m) {
  auto const pn = dense_pad_n(m.n);
  auto const pk = dense_pad_k(m.k);
  auto const tiles_n = pn / kDenseTileRows;
  auto const tiles_k = pk / kDenseTileK;
  constexpr std::uint32_t groups_tile = kDenseTileK / kQ4GroupSize;
  std::vector<std::byte> out(static_cast<std::size_t>(
      tiles_n * tiles_k * kDenseTileRows * groups_tile * 2));
  auto const groups_row = m.k / m.group_size;
  std::size_t cursor = 0;
  for (std::uint64_t tn = 0; tn < tiles_n; ++tn) {
    for (std::uint64_t tk = 0; tk < tiles_k; ++tk) {
      for (std::uint32_t r = 0; r < kDenseTileRows; ++r) {
        auto const row = tn * kDenseTileRows + r;
        for (std::uint32_t g = 0; g < groups_tile; ++g) {
          auto const col0 = tk * kDenseTileK + g * kQ4GroupSize;
          std::uint16_t bits = 0;
          if (row < m.n && col0 < m.k) {
            auto const lg = col0 / m.group_size;
            bits = m.scales[static_cast<std::size_t>(row * groups_row + lg)];
          }
          store_u16_le(out.data() + cursor, bits);
          cursor += 2;
        }
      }
    }
  }
  return out;
}

// Independent Q8 encoder from the V0 layout contract. Keep this separate from
// the production packer so the full physical payload remains a byte-level
// authority for tile, row, scale, and padding order.
std::vector<std::byte> golden_q8_codes(LogicalWeightCodes const& m) {
  auto const pn = dense_pad_n(m.n);
  auto const pk = dense_pad_k(m.k);
  auto const tiles_n = pn / kDenseTileRows;
  auto const tiles_k = pk / kDenseTileK;
  std::vector<std::byte> out(static_cast<std::size_t>(
      tiles_n * tiles_k * kDenseTileRows * kQ8PackedBytesPerTileRow));
  for (std::uint64_t tn = 0; tn < tiles_n; ++tn) {
    for (std::uint64_t tk = 0; tk < tiles_k; ++tk) {
      for (std::uint32_t r = 0; r < kDenseTileRows; ++r) {
        auto const row = tn * kDenseTileRows + r;
        auto const base =
            ((tn * tiles_k + tk) * kDenseTileRows + r) * kQ8PackedBytesPerTileRow;
        for (std::uint32_t c = 0; c < kDenseTileK; ++c) {
          auto const col = tk * kDenseTileK + c;
          std::int8_t code = 0;
          if (row < m.n && col < m.k) {
            code = m.codes[static_cast<std::size_t>(row * m.k + col)];
          }
          out[static_cast<std::size_t>(base + c)] =
              static_cast<std::byte>(static_cast<std::uint8_t>(code));
        }
      }
    }
  }
  return out;
}

std::vector<std::byte> golden_q8_scales(LogicalWeightCodes const& m) {
  auto const pn = dense_pad_n(m.n);
  auto const pk = dense_pad_k(m.k);
  auto const tiles_n = pn / kDenseTileRows;
  auto const tiles_k = pk / kDenseTileK;
  constexpr std::uint32_t groups_tile = kDenseTileK / kQ8GroupSize;
  std::vector<std::byte> out(static_cast<std::size_t>(
      tiles_n * tiles_k * kDenseTileRows * groups_tile * 2));
  auto const groups_row = m.k / m.group_size;
  std::size_t cursor = 0;
  for (std::uint64_t tn = 0; tn < tiles_n; ++tn) {
    for (std::uint64_t tk = 0; tk < tiles_k; ++tk) {
      for (std::uint32_t r = 0; r < kDenseTileRows; ++r) {
        auto const row = tn * kDenseTileRows + r;
        for (std::uint32_t g = 0; g < groups_tile; ++g) {
          auto const col0 = tk * kDenseTileK + g * kQ8GroupSize;
          std::uint16_t bits = 0;
          if (row < m.n && col0 < m.k) {
            auto const logical_group = col0 / m.group_size;
            bits = m.scales[static_cast<std::size_t>(row * groups_row + logical_group)];
          }
          store_u16_le(out.data() + cursor, bits);
          cursor += 2;
        }
      }
    }
  }
  return out;
}

}  // namespace

int main() {
  // Exhaustively cover both signs of FP16 subnormals and the normal boundary.
  for (std::uint16_t h = 1; h <= 0x400; ++h) {
    auto const expected = std::ldexp(static_cast<float>(h), -24);
    expect(qw38::format::fp16_to_fp32(h) == expected,
           "FP16 positive subnormal has the correct exponent");
    expect(qw38::format::fp16_to_fp32(h | 0x8000u) == -expected,
           "FP16 negative subnormal has the correct exponent");
  }
  for (auto const n : {5u, 16u}) {
    constexpr auto quantizer = LogicalQuantizerId::Q4KCandidateV2;
    constexpr auto layout = PhysicalLayoutId::CudaQ4KCandidateV2;
    LogicalWeightCodes m;
    m.quantizer = quantizer;
    m.n = n;
    m.k = 512;
    m.group_size = 256;
    m.qmax = 15;
    m.codes.resize(n * 512);
    m.q4k_metadata.resize(n * 2 * 16);
    for (std::size_t i = 0; i < m.codes.size(); ++i) {
      m.codes[i] = static_cast<std::int8_t>((i + i / 256) % 16);
    }
    for (std::size_t b = 0; b < n * 2; ++b) {
      m.q4k_metadata[b * 16] = 1;  // smallest FP16 subnormal d
      m.q4k_metadata[b * 16 + 2] = 2;
      for (std::size_t i = 4; i < 16; ++i) {
        m.q4k_metadata[b * 16 + i] = static_cast<std::uint8_t>(b * 17 + i);
      }
    }
    auto packed = pack_cuda_v0(quantizer, layout, m);
    expect(packed.has_value(), "Q4_K pack");
    if (!packed) continue;
    expect(packed->codes == golden_q4_codes(m), "Q4_K independent unsigned code packing");
    expect(packed->scales.size() == dense_pad_n(n) * 2 * 16,
           "Q4_K requires 16 metadata bytes per tile row");
    for (std::size_t row = 0; row < n; ++row) {
      for (std::size_t block = 0; block < 2; ++block) {
        auto const physical = ((row / 8 * 2 + block) * 8 + row % 8) * 16;
        for (std::size_t i = 0; i < 16; ++i) {
          expect(packed->scales[physical + i] ==
                     static_cast<std::byte>(m.q4k_metadata[(row * 2 + block) * 16 + i]),
                 "Q4_K metadata independent tile coordinates");
        }
      }
    }
    auto unpacked = qw38::format::unpack_cuda_v0(
        quantizer, layout, n, 512, packed->codes, packed->scales);
    expect(unpacked && unpacked->codes == m.codes && unpacked->scales.empty() &&
               unpacked->q4k_metadata == m.q4k_metadata && unpacked->group_size == 256 &&
               unpacked->qmax == 15,
           "Q4_K independent unpack preserves codes and all metadata bits");
    auto invalid = packed->scales;
    invalid[1] = std::byte{0x7c};
    expect(!qw38::format::validate_cuda_v0(quantizer, layout, n, 512,
                                         packed->codes, invalid),
           "Q4_K infinite superblock scale rejected");
    invalid[1] = std::byte{0x80};
    expect(!qw38::format::validate_cuda_v0(quantizer, layout, n, 512,
                                         packed->codes, invalid),
           "Q4_K negative superblock scale rejected");
    if (n == 5) {
      invalid = packed->scales;
      invalid.back() = std::byte{1};
      expect(!qw38::format::validate_cuda_v0(quantizer, layout, n, 512,
                                           packed->codes, invalid),
             "Q4_K nonzero padding metadata rejected");
      auto codes = packed->codes;
      codes.back() = std::byte{1};
      expect(!qw38::format::validate_cuda_v0(quantizer, layout, n, 512,
                                           codes, packed->scales),
             "Q4_K nonzero padding code rejected");
    }
    for (auto const bad : {-1, 16}) {
      m.codes[0] = static_cast<std::int8_t>(bad);
      expect(!pack_cuda_v0(quantizer, layout, m), "Q4_K rejects codes outside 0..15");
    }
    expect(!quantizer_layout_pair_ok(quantizer, PhysicalLayoutId::CudaQ4G64CandidateV1),
           "Q4_K cannot reinterpret Q4G64 layout");
  }
  for (auto const& [quantizer, layout] : {
           std::pair{LogicalQuantizerId::Q4G64CandidateV1,
                     PhysicalLayoutId::CudaQ4G64CandidateV1},
           std::pair{LogicalQuantizerId::Q8G32CandidateV1,
                     PhysicalLayoutId::CudaQ8G32CandidateV1}}) {
    auto m = make_logical(quantizer, 5, 192);
    m.codes[4 * 192 + 191] = -7;
    m.scales[4 * (192 / m.group_size) + 191 / m.group_size] = 0x3C00;
    auto packed = pack_cuda_v0(quantizer, layout, m);
    expect(static_cast<bool>(packed), "candidate tail pack");
    if (!packed) continue;
    expect(packed->codes == (m.qmax == 7 ? golden_q4_codes(m)
                                        : golden_q8_codes(m)),
           "candidate independent golden codes including padding");
    expect(packed->scales == (m.qmax == 7 ? golden_q4_scales(m)
                                         : golden_q8_scales(m)),
           "candidate independent golden scales including padding");
    auto unpacked = qw38::format::unpack_cuda_v0(
        quantizer, layout, m.n, m.k, packed->codes, packed->scales);
    expect(unpacked && unpacked->codes == m.codes &&
               unpacked->scales == m.scales,
           "candidate tail independent unpack");
    auto bad_codes = packed->codes;
    bad_codes.back() = std::byte{1};
    expect(!qw38::format::validate_cuda_v0(
               quantizer, layout, m.n, m.k, bad_codes, packed->scales),
           "candidate nonzero padded code rejected");
    auto bad_scales = packed->scales;
    bad_scales.back() = std::byte{1};
    expect(!qw38::format::validate_cuda_v0(
               quantizer, layout, m.n, m.k, packed->codes, bad_scales),
           "candidate nonzero padded scale rejected");
    expect(!quantizer_layout_pair_ok(quantizer,
             m.qmax == 7 ? PhysicalLayoutId::CudaQ4G64V0
                         : PhysicalLayoutId::CudaQ8G32V0),
           "candidate ID cannot bind V0 physical layout");
  }
  expect(quantizer_layout_pair_ok(LogicalQuantizerId::Q4G64V0,
                                  PhysicalLayoutId::CudaQ4G64V0),
         "Q4 pair ok");
  expect(!quantizer_layout_pair_ok(LogicalQuantizerId::Q4G64V0,
                                   PhysicalLayoutId::CudaQ8G32V0),
         "Q4 with Q8 layout rejected");

  {
    auto m = make_logical(LogicalQuantizerId::Q4G64V0, 8, 256);
    auto packed = pack_cuda_v0(LogicalQuantizerId::Q4G64V0,
                               PhysicalLayoutId::CudaQ4G64V0, m);
    if (!packed) {
      fail(error_message(packed.error()));
    } else {
      expect(packed->codes.size() == 8 * 128, "zero Q4 code bytes");
      expect(packed->scales.size() == 8 * 4 * 2, "zero Q4 scale bytes");
      bool z = true;
      for (auto b : packed->codes) {
        z = z && (b == std::byte{0});
      }
      expect(z, "zero Q4 payload");
    }
  }

  {
    auto m = make_logical(LogicalQuantizerId::Q4G64V0, 8, 256);
    m.codes[0] = 1;
    m.codes[1] = 2;
    m.codes[2] = 3;
    m.codes[3] = 4;
    m.codes[4] = -1;
    m.codes[5] = 7;
    m.scales[0] = 0x3C00;
    auto packed = pack_cuda_v0(LogicalQuantizerId::Q4G64V0,
                               PhysicalLayoutId::CudaQ4G64V0, m);
    auto want_c = golden_q4_codes(m);
    auto want_s = golden_q4_scales(m);
    if (!packed) {
      fail(error_message(packed.error()));
    } else {
      expect(packed->codes[0] == std::byte{0x21}, "lower nibble is earlier K");
      expect(packed->codes[1] == std::byte{0x43}, "second pair 3 then 4");
      expect(packed->codes[2] == std::byte{0x7F}, "signed nibbles -1 then 7");
      expect(packed->codes == want_c, "Q4 golden codes");
      expect(packed->scales == want_s, "Q4 golden scales");
      expect(load_u16_le(packed->scales.data()) == 0x3C00, "scale little-endian 1.0");
    }
  }

  {
    auto m = make_logical(LogicalQuantizerId::Q4G64V0, 16, 512);
    m.codes[9 * 512 + 300] = 5;
    m.scales[9 * (512 / 64) + (300 / 64)] = 0x4200;
    auto packed = pack_cuda_v0(LogicalQuantizerId::Q4G64V0,
                               PhysicalLayoutId::CudaQ4G64V0, m);
    if (!packed) {
      fail(error_message(packed.error()));
    } else {
      // tn=1, tk=1, r=1, c=44 → tile_row 25, byte 22, low nibble
      expect(packed->codes.size() == 2 * 2 * 8 * 128, "16x512 Q4 size");
      auto const byte = static_cast<std::uint8_t>(packed->codes[3222]);
      expect((byte & 0x0F) == 5, "tile/group order places code at 3222");
      auto const scale_i = ((1 * 2 + 1) * 8 + 1) * 4 + (44 / 64);
      expect(load_u16_le(packed->scales.data() + scale_i * 2) == 0x4200,
             "scale order [N/8,K/256,row,group]");
      expect(packed->codes == golden_q4_codes(m), "16x512 golden codes");
      expect(packed->scales == golden_q4_scales(m), "16x512 golden scales");
    }
  }

  {
    auto m = make_logical(LogicalQuantizerId::Q4G64V0, 5, 192);
    m.codes[0] = 3;
    m.scales[0] = 0x4000;
    auto packed = pack_cuda_v0(LogicalQuantizerId::Q4G64V0,
                               PhysicalLayoutId::CudaQ4G64V0, m);
    if (!packed) {
      fail(error_message(packed.error()));
    } else {
      expect(packed->padded_n == 8 && packed->padded_k == 256, "pad to 8x256");
      expect(packed->codes.size() == 8 * 128, "padded Q4 payload");
      expect(packed->scales.size() == 8 * 4 * 2, "padded Q4 scales");
      expect(packed->codes == golden_q4_codes(m), "padding golden codes");
      expect(packed->scales == golden_q4_scales(m), "padding golden scales");
      expect(static_cast<std::uint8_t>(packed->codes[0]) == 3, "logical code kept");
      bool rest_zero = true;
      for (std::size_t i = 1; i < packed->codes.size(); ++i) {
        rest_zero = rest_zero && packed->codes[i] == std::byte{0};
      }
      expect(rest_zero, "padded coordinates are zero codes");
      expect(load_u16_le(packed->scales.data() + 2) == 0, "padded K group scale 0");
    }
  }

  {
    auto m = make_logical(LogicalQuantizerId::Q8G32V0, 8, 256);
    m.codes[0] = 1;
    m.codes[1] = -1;
    m.codes[2] = -127;
    m.scales[0] = 0x3C00;
    auto packed = pack_cuda_v0(LogicalQuantizerId::Q8G32V0,
                               PhysicalLayoutId::CudaQ8G32V0, m);
    if (!packed) {
      fail(error_message(packed.error()));
    } else {
      expect(packed->codes.size() == 8 * 256, "Q8 256 bytes/row");
      expect(packed->scales.size() == 8 * 8 * 2, "Q8 eight scales/row");
      expect(static_cast<std::uint8_t>(packed->codes[0]) == 1, "Q8 +1");
      expect(static_cast<std::uint8_t>(packed->codes[1]) == 0xFF, "Q8 two's complement -1");
      expect(static_cast<std::uint8_t>(packed->codes[2]) == 0x81, "Q8 two's complement -127");
      expect(packed->codes == golden_q8_codes(m), "Q8 complete golden codes");
      expect(packed->scales == golden_q8_scales(m), "Q8 complete golden scales");
    }
  }

  {
    // Covers all four physical tiles, Q8 scale order, and determinism at the
    // representative 16x512 matrix geometry.
    auto m = make_logical(LogicalQuantizerId::Q8G32V0, 16, 512);
    m.codes[9 * 512 + 300] = -127;
    m.codes[15 * 512 + 511] = 126;
    m.scales[9 * (512 / 32) + (300 / 32)] = 0x4200;
    m.scales[15 * (512 / 32) + (511 / 32)] = 0x3555;
    auto a = pack_cuda_v0(LogicalQuantizerId::Q8G32V0,
                          PhysicalLayoutId::CudaQ8G32V0, m);
    auto b = pack_cuda_v0(LogicalQuantizerId::Q8G32V0,
                          PhysicalLayoutId::CudaQ8G32V0, m);
    if (!a || !b) {
      fail("16x512 Q8 pack");
    } else {
      expect(a->codes.size() == 2 * 2 * 8 * 256, "16x512 Q8 complete code size");
      expect(a->scales.size() == 2 * 2 * 8 * 8 * 2,
             "16x512 Q8 complete scale size");
      expect(a->codes == golden_q8_codes(m), "16x512 Q8 golden codes");
      expect(a->scales == golden_q8_scales(m), "16x512 Q8 golden scales");
      expect(a->codes == b->codes && a->scales == b->scales,
             "16x512 Q8 deterministic bytes");
    }
  }

  {
    auto m = make_logical(LogicalQuantizerId::Q8G32V0, 5, 192);
    m.codes[4 * 192 + 191] = -127;
    m.scales[4 * (192 / 32) + 5] = 0x3C00;
    auto packed = pack_cuda_v0(LogicalQuantizerId::Q8G32V0,
                               PhysicalLayoutId::CudaQ8G32V0, m);
    if (!packed) {
      fail(error_message(packed.error()));
    } else {
      expect(packed->padded_n == 8 && packed->padded_k == 256,
             "Q8 padding to one tile");
      expect(packed->codes == golden_q8_codes(m), "padded Q8 golden codes");
      expect(packed->scales == golden_q8_scales(m), "padded Q8 golden scales");
      expect(static_cast<std::uint8_t>(packed->codes[4 * 256 + 191]) == 0x81,
             "Q8 padded logical edge code retained");
      expect(load_u16_le(packed->scales.data() + (4 * 8 + 5) * 2) == 0x3C00,
             "Q8 padded scale order");
    }
  }

  {
    auto m = make_logical(LogicalQuantizerId::Q4G64V0, 8, 256);
    m.codes[0] = -8;
    auto packed = pack_cuda_v0(LogicalQuantizerId::Q4G64V0,
                               PhysicalLayoutId::CudaQ4G64V0, m);
    expect(!packed, "packer rejects forbidden Q4 -8");
    auto m8 = make_logical(LogicalQuantizerId::Q8G32V0, 8, 256);
    m8.codes[0] = static_cast<std::int8_t>(-128);
    auto p8 = pack_cuda_v0(LogicalQuantizerId::Q8G32V0,
                           PhysicalLayoutId::CudaQ8G32V0, m8);
    expect(!p8, "packer rejects forbidden Q8 -128");
  }

  {
    auto m = make_logical(LogicalQuantizerId::Q4G64V0, 8, 256);
    m.codes[10] = 6;
    auto a = pack_cuda_v0(LogicalQuantizerId::Q4G64V0,
                          PhysicalLayoutId::CudaQ4G64V0, m);
    auto b = pack_cuda_v0(LogicalQuantizerId::Q4G64V0,
                          PhysicalLayoutId::CudaQ4G64V0, m);
    expect(static_cast<bool>(a) && static_cast<bool>(b) && a->codes == b->codes &&
               a->scales == b->scales,
           "deterministic packed bytes");
    auto mismatch = pack_cuda_v0(LogicalQuantizerId::Q4G64V0,
                                 PhysicalLayoutId::CudaQ8G32V0, m);
    expect(!mismatch, "mismatched strong IDs rejected");
  }

  {
    constexpr std::uint64_t huge_n = std::uint64_t{1} << 55;
    std::span<std::byte const> empty;
    auto packed = qw38::format::pack_bf16_dense_tile_v0(empty, huge_n, 256);
    expect(!packed && packed.error().code == FormatErrorCode::Overflow,
           "BF16 pack rejects byte-size overflow");
    auto unpacked = unpack_bf16_dense_tile_v0(empty, huge_n, 256);
    expect(!unpacked && unpacked.error().code == FormatErrorCode::Overflow,
           "BF16 unpack rejects byte-size overflow");
  }

  if (g_failures != 0) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }
  return 0;
}
