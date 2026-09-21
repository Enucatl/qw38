#include "format/floatcvt.hpp"
#include "format/logical.hpp"
#include "format/pack.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>
#include <vector>

using qw38::format::dense_pad_k;
using qw38::format::dense_pad_n;
using qw38::format::error_message;
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
  m.group_size = (id == LogicalQuantizerId::Q4G64V0) ? kQ4GroupSize : kQ8GroupSize;
  m.qmax = (id == LogicalQuantizerId::Q4G64V0) ? 7 : 127;
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

}  // namespace

int main() {
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
      expect(static_cast<std::uint8_t>(packed->codes[0]) == 1, "Q8 +1");
      expect(static_cast<std::uint8_t>(packed->codes[1]) == 0xFF, "Q8 two's complement -1");
      expect(static_cast<std::uint8_t>(packed->codes[2]) == 0x81, "Q8 two's complement -127");
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

  if (g_failures != 0) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }
  return 0;
}
