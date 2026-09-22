#include "compiler/compiler.hpp"
#include "format/floatcvt.hpp"
#include "format/pack.hpp"
#include "format/unpack.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>
#include <vector>

using qw38::compiler::dequantize_to_bf16;
using qw38::compiler::decode_bf16_payload;
using qw38::compiler::CompilerErrorCode;
using qw38::compiler::quantize_bf16;
using qw38::compiler::quantize_fp32;
using qw38::compiler::reference_gemv_bf16;
using qw38::compiler::reference_gemv_packed;
using qw38::format::bf16_to_fp32;
using qw38::format::fp32_to_bf16_rne;
using qw38::format::LogicalQuantizerId;
using qw38::format::pack_bf16_dense_tile_v0;
using qw38::format::pack_cuda_v0;
using qw38::format::PhysicalLayoutId;
using qw38::format::PackedMatrix;
using qw38::format::store_u16_le;
using qw38::format::unpack_bf16_dense_tile_v0;
using qw38::format::unpack_cuda_v0;

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

std::vector<std::byte> bf16_matrix(std::uint64_t n, std::uint64_t k,
                                   float scale) {
  std::vector<std::byte> out(static_cast<std::size_t>(n * k * 2));
  for (std::uint64_t i = 0; i < n * k; ++i) {
    float const v =
        scale * (static_cast<float>(static_cast<int>(i % 13) - 6) / 7.0f);
    store_u16_le(out.data() + i * 2, fp32_to_bf16_rne(v));
  }
  return out;
}

std::vector<std::uint16_t> bf16_vec(std::uint64_t k, float seed) {
  std::vector<std::uint16_t> out(static_cast<std::size_t>(k));
  for (std::uint64_t i = 0; i < k; ++i) {
    out[i] = fp32_to_bf16_rne(seed * (static_cast<float>(i % 5) - 2.0f));
  }
  return out;
}

void check_roundtrip(LogicalQuantizerId qid, PhysicalLayoutId layout,
                     std::uint64_t n, std::uint64_t k, float scale,
                     std::string_view tag) {
  auto src = bf16_matrix(n, k, scale);
  auto logical = quantize_bf16(qid, n, k, src);
  if (!logical) {
    fail(std::string(tag) + " quantize: " +
         qw38::compiler::error_message(logical.error()));
    return;
  }
  auto packed = pack_cuda_v0(qid, layout, *logical);
  if (!packed) {
    fail(std::string(tag) + " pack: " +
         qw38::format::error_message(packed.error()));
    return;
  }
  auto unpacked = unpack_cuda_v0(qid, layout, n, k, packed->codes, packed->scales);
  if (!unpacked) {
    fail(std::string(tag) + " unpack: " +
         qw38::format::error_message(unpacked.error()));
    return;
  }
  expect(unpacked->codes == logical->codes, std::string(tag) + " codes agree");
  expect(unpacked->scales == logical->scales, std::string(tag) + " scales agree");
  expect(unpacked->quantizer == qid && unpacked->n == n && unpacked->k == k,
         std::string(tag) + " geometry");

  auto dq = dequantize_to_bf16(*logical);
  auto x = bf16_vec(k, 0.25f);
  if (!dq) {
    fail(std::string(tag) + " dequant");
    return;
  }
  auto from_explicit = reference_gemv_bf16(*dq, x, n, k);
  auto from_packed = reference_gemv_packed(*packed, x);
  if (!from_explicit || !from_packed) {
    fail(std::string(tag) + " gemv");
    return;
  }
  expect(*from_explicit == *from_packed,
         std::string(tag) + " contraction matches explicit BF16 GEMV");
}

// Test-local physical decoder. It deliberately reads bytes using the V0 tile
// contract instead of calling unpack/dequantize or float conversion helpers.
float oracle_bf16(std::uint16_t bits) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

float oracle_round_bf16(float value) {
  auto const raw = std::bit_cast<std::uint32_t>(value);
  auto const lsb = (raw >> 16) & 1u;
  auto const rounded = static_cast<std::uint16_t>((raw + 0x7FFFu + lsb) >> 16);
  return oracle_bf16(rounded);
}

float oracle_fp16(std::uint16_t bits) {
  auto const sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
  auto const exponent = (bits >> 10) & 0x1Fu;
  auto const mantissa = bits & 0x03FFu;
  std::uint32_t out = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      out = sign;
    } else {
      auto m = static_cast<std::uint32_t>(mantissa);
      auto e = 112u;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        --e;
      }
      out = sign | (e << 23) | ((m & 0x3FFu) << 13);
    }
  } else if (exponent == 31) {
    out = sign | 0x7F800000u | (static_cast<std::uint32_t>(mantissa) << 13);
  } else {
    out = sign | (static_cast<std::uint32_t>(exponent + 112) << 23) |
          (static_cast<std::uint32_t>(mantissa) << 13);
  }
  return std::bit_cast<float>(out);
}

std::int8_t oracle_code(PackedMatrix const& packed, std::uint64_t row,
                        std::uint64_t col) {
  auto const tiles_k = packed.padded_k / 256;
  auto const row_tile = row / 8;
  auto const row_in_tile = row % 8;
  auto const col_tile = col / 256;
  auto const col_in_tile = col % 256;
  auto const tile_row = (row_tile * tiles_k + col_tile) * 8 + row_in_tile;
  if (packed.quantizer == LogicalQuantizerId::Q4G64V0) {
    auto const byte = static_cast<std::uint8_t>(
        packed.codes[static_cast<std::size_t>(tile_row * 128 + col_in_tile / 2)]);
    auto const nibble = static_cast<std::int8_t>(
        (col_in_tile & 1u) == 0 ? byte & 0x0Fu : byte >> 4);
    return nibble >= 8 ? static_cast<std::int8_t>(nibble - 16) : nibble;
  }
  return static_cast<std::int8_t>(packed.codes[static_cast<std::size_t>(
      tile_row * 256 + col_in_tile)]);
}

float oracle_scale(PackedMatrix const& packed, std::uint64_t row,
                   std::uint64_t col, std::uint32_t group_size) {
  auto const tiles_k = packed.padded_k / 256;
  auto const groups_tile = 256 / group_size;
  auto const scale = ((row / 8 * tiles_k + col / 256) * 8 + row % 8) *
                         groups_tile +
                     (col % 256) / group_size;
  auto const bits = static_cast<std::uint16_t>(
      static_cast<std::uint8_t>(packed.scales[scale * 2]) |
      (static_cast<std::uint16_t>(static_cast<std::uint8_t>(packed.scales[scale * 2 + 1]))
       << 8));
  return oracle_fp16(bits);
}

std::vector<float> oracle_packed_gemv(PackedMatrix const& packed,
                                      std::span<std::uint16_t const> x) {
  std::vector<float> out(static_cast<std::size_t>(packed.logical_n));
  for (std::uint64_t row = 0; row < packed.logical_n; ++row) {
    float acc = 0.0f;
    for (std::uint64_t col = 0; col < packed.logical_k; ++col) {
      float weight = 0.0f;
      if (packed.quantizer == LogicalQuantizerId::None) {
        auto const tiles_k = packed.padded_k / 256;
        auto const index = (((row / 8 * tiles_k + col / 256) * 8 + row % 8) *
                            256 + col % 256) * 2;
        auto const bits = static_cast<std::uint16_t>(
            static_cast<std::uint8_t>(packed.codes[index]) |
            (static_cast<std::uint16_t>(static_cast<std::uint8_t>(packed.codes[index + 1]))
             << 8));
        weight = oracle_bf16(bits);
      } else {
        auto const group = packed.quantizer == LogicalQuantizerId::Q4G64V0 ? 64u : 32u;
        // V0 rounds decoded operands to BF16 before FP32 accumulation.
        weight = oracle_round_bf16(static_cast<float>(oracle_code(packed, row, col)) *
                                   oracle_scale(packed, row, col, group));
      }
      acc += weight * oracle_bf16(x[static_cast<std::size_t>(col)]);
    }
    out[static_cast<std::size_t>(row)] = acc;
  }
  return out;
}

std::uint32_t bits(float value) {
  return std::bit_cast<std::uint32_t>(value);
}

void expect_oracle_contract(PackedMatrix const& packed,
                            std::span<std::uint16_t const> x,
                            std::span<std::uint32_t const> expected,
                            std::string_view tag) {
  auto const oracle = oracle_packed_gemv(packed, x);
  auto production = reference_gemv_packed(packed, x);
  expect(oracle.size() == expected.size(), std::string(tag) + " oracle size");
  expect(static_cast<bool>(production), std::string(tag) + " production GEMV");
  if (!production || oracle.size() != expected.size()) {
    return;
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    expect(bits(oracle[i]) == expected[i], std::string(tag) + " frozen oracle result");
    expect(bits((*production)[i]) == expected[i],
           std::string(tag) + " packed contraction result");
  }
}

}  // namespace

int main() {
  check_roundtrip(LogicalQuantizerId::Q4G64V0, PhysicalLayoutId::CudaQ4G64V0, 8,
                  256, 1.0f, "q4-8x256");
  check_roundtrip(LogicalQuantizerId::Q4G64V0, PhysicalLayoutId::CudaQ4G64V0, 16,
                  512, 0.35f, "q4-16x512");
  check_roundtrip(LogicalQuantizerId::Q4G64V0, PhysicalLayoutId::CudaQ4G64V0, 24,
                  256, 2.5f, "q4-24x256");
  check_roundtrip(LogicalQuantizerId::Q8G32V0, PhysicalLayoutId::CudaQ8G32V0, 8,
                  256, 4.0f, "q8-8x256");
  check_roundtrip(LogicalQuantizerId::Q8G32V0, PhysicalLayoutId::CudaQ8G32V0, 16,
                  512, 0.9f, "q8-16x512");

  {
    // Raw Q4 tile bytes exercise lower/upper nibbles, both signs, ±7 edges,
    // an asymmetric three-row matrix, and a scale distinct per row.
    PackedMatrix packed{.layout = PhysicalLayoutId::CudaQ4G64V0,
                        .quantizer = LogicalQuantizerId::Q4G64V0,
                        .logical_n = 3,
                        .logical_k = 64,
                        .padded_n = 8,
                        .padded_k = 256,
                        .codes = std::vector<std::byte>(8 * 128),
                        .scales = std::vector<std::byte>(8 * 4 * 2)};
    auto set_code = [&](std::uint64_t row, std::uint64_t col, std::int8_t code) {
      auto const index = row * 128 + col / 2;
      auto byte = static_cast<std::uint8_t>(packed.codes[index]);
      auto const nibble = static_cast<std::uint8_t>(code) & 0x0Fu;
      byte = (col & 1u) == 0 ? static_cast<std::uint8_t>((byte & 0xF0u) | nibble)
                              : static_cast<std::uint8_t>((byte & 0x0Fu) | (nibble << 4));
      packed.codes[index] = static_cast<std::byte>(byte);
    };
    set_code(0, 0, 7);   set_code(0, 1, -7); set_code(0, 2, 1);  set_code(0, 63, -1);
    set_code(1, 0, -3);  set_code(1, 1, 5);  set_code(1, 2, -7); set_code(1, 63, 7);
    set_code(2, 1, -1);  set_code(2, 2, 7);  set_code(2, 63, -7);
    store_u16_le(packed.scales.data(), 0x3800);       // 0.5
    store_u16_le(packed.scales.data() + 4 * 2, 0x3400);   // 0.25
    store_u16_le(packed.scales.data() + 8 * 2, 0x3A00);   // 0.75
    std::vector<std::uint16_t> x(64);
    for (std::uint64_t i = 0; i < x.size(); ++i) {
      x[i] = fp32_to_bf16_rne(static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25f);
    }
    constexpr std::array<std::uint32_t, 3> expected{
        0xbf200000u, 0xbf700000u, 0x40400000u};  // -0.625, -0.9375, 3
    expect_oracle_contract(packed, x, expected, "Q4 raw contraction oracle");
  }

  {
    PackedMatrix packed{.layout = PhysicalLayoutId::CudaQ8G32V0,
                        .quantizer = LogicalQuantizerId::Q8G32V0,
                        .logical_n = 2,
                        .logical_k = 32,
                        .padded_n = 8,
                        .padded_k = 256,
                        .codes = std::vector<std::byte>(8 * 256),
                        .scales = std::vector<std::byte>(8 * 8 * 2)};
    auto set_code = [&](std::uint64_t row, std::uint64_t col, std::int8_t code) {
      packed.codes[row * 256 + col] = static_cast<std::byte>(static_cast<std::uint8_t>(code));
    };
    set_code(0, 0, 127); set_code(0, 1, -127); set_code(0, 2, 1);    set_code(0, 31, -1);
    set_code(1, 0, -64); set_code(1, 1, 63);   set_code(1, 2, -127); set_code(1, 31, 127);
    store_u16_le(packed.scales.data(), 0x3400);       // 0.25
    store_u16_le(packed.scales.data() + 8 * 2, 0x3800);   // 0.5
    std::vector<std::uint16_t> x(32);
    for (std::uint64_t i = 0; i < x.size(); ++i) {
      x[i] = fp32_to_bf16_rne(static_cast<float>(static_cast<int>(i % 5) - 2) * 0.5f);
    }
    constexpr std::array<std::uint32_t, 2> expected{
        0xc17c0000u, 0xc1780000u};  // -15.75, -15.5
    expect_oracle_contract(packed, x, expected, "Q8 raw contraction oracle");
  }

  {
    PackedMatrix packed{.layout = PhysicalLayoutId::CudaBf16DenseTileV0,
                        .quantizer = LogicalQuantizerId::None,
                        .logical_n = 8,
                        .logical_k = 256,
                        .padded_n = 8,
                        .padded_k = 256,
                        .codes = std::vector<std::byte>(8 * 256 * 2),
                        .scales = {}};
    auto set_weight = [&](std::uint64_t row, std::uint64_t col, float value) {
      store_u16_le(packed.codes.data() + (row * 256 + col) * 2,
                   fp32_to_bf16_rne(value));
    };
    set_weight(0, 0, 1.5f); set_weight(0, 1, -2.0f); set_weight(0, 255, 0.5f);
    set_weight(1, 0, -1.0f); set_weight(1, 1, 0.75f); set_weight(1, 255, 2.0f);
    std::vector<std::uint16_t> x(256, fp32_to_bf16_rne(0.0f));
    x[0] = fp32_to_bf16_rne(-1.0f);
    x[1] = fp32_to_bf16_rne(-0.5f);
    x[255] = fp32_to_bf16_rne(-1.0f);
    constexpr std::array<std::uint32_t, 8> expected{
        0xbf800000u, 0xbfb00000u, 0, 0, 0, 0, 0, 0};  // -1, -1.375, then zero rows
    expect_oracle_contract(packed, x, expected, "BF16 raw contraction oracle");
  }

  {
    auto src = bf16_matrix(8, 256, 1.25f);
    auto packed = pack_bf16_dense_tile_v0(src, 8, 256);
    if (!packed) {
      fail(qw38::format::error_message(packed.error()));
    } else {
      auto back = unpack_bf16_dense_tile_v0(packed->codes, 8, 256);
      expect(static_cast<bool>(back) && *back == src, "BF16 tile passthrough");
      auto x = bf16_vec(256, 0.5f);
      std::vector<std::uint16_t> w(8 * 256);
      for (std::size_t i = 0; i < w.size(); ++i) {
        w[i] = qw38::format::load_u16_le(src.data() + i * 2);
      }
      auto a = reference_gemv_bf16(w, x, 8, 256);
      auto b = reference_gemv_packed(*packed, x);
      expect(static_cast<bool>(a) && static_cast<bool>(b) && *a == *b,
             "BF16 control contraction");
    }
  }

  {
    std::vector<float> w(8 * 256, 0.0f);
    w[0] = 3.0f;
    auto q = quantize_fp32(LogicalQuantizerId::Q4G64V0, 8, 256, w);
    if (!q) {
      fail(qw38::compiler::error_message(q.error()));
    } else {
      auto packed = pack_cuda_v0(LogicalQuantizerId::Q4G64V0,
                                 PhysicalLayoutId::CudaQ4G64V0, *q);
      if (!packed) {
        fail(qw38::format::error_message(packed.error()));
      } else {
        auto unpacked = unpack_cuda_v0(LogicalQuantizerId::Q4G64V0,
                                       PhysicalLayoutId::CudaQ8G32V0, 8, 256,
                                       packed->codes, packed->scales);
        expect(!unpacked, "unpack rejects mismatched layout id");
        packed->codes[0] = std::byte{0x08};
        auto bad = unpack_cuda_v0(LogicalQuantizerId::Q4G64V0,
                                  PhysicalLayoutId::CudaQ4G64V0, 8, 256,
                                  packed->codes, packed->scales);
        expect(!bad, "unpack rejects forbidden Q4 nibble -8");
      }
    }
  }

  {
    std::vector<std::byte> exact(4);
    auto decoded = decode_bf16_payload(
        PhysicalLayoutId::CudaBf16RowMajorV0, exact, 1, 2);
    expect(decoded && decoded->size() == 2,
           "BF16 decoder accepts exact declared payload");

    std::vector<std::byte> short_payload(2);
    auto short_result = decode_bf16_payload(
        PhysicalLayoutId::CudaBf16RowMajorV0, short_payload, 1, 2);
    expect(!short_result &&
               short_result.error().code == CompilerErrorCode::ShapeMismatch,
           "BF16 decoder rejects short payload");

    std::vector<std::byte> long_payload(6);
    auto long_result = decode_bf16_payload(
        PhysicalLayoutId::CudaBf16RowMajorV0, long_payload, 1, 2);
    expect(!long_result &&
               long_result.error().code == CompilerErrorCode::ShapeMismatch,
           "BF16 decoder rejects long payload");

    std::span<std::byte const> empty;
    auto overflow = decode_bf16_payload(
        PhysicalLayoutId::CudaBf16RowMajorV0, empty,
        std::uint64_t{1} << 63, 2);
    expect(!overflow && overflow.error().code == CompilerErrorCode::Format,
           "BF16 decoder rejects declared geometry overflow");
  }

  if (g_failures != 0) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }
  return 0;
}
