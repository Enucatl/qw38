#include "compiler/compiler.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>
#include <vector>

using qw38::compiler::bf16_is_finite;
using qw38::compiler::CompilerErrorCode;
using qw38::compiler::conv_from_tap_major;
using qw38::compiler::conv_to_tap_major;
using qw38::compiler::generate_rope_inv_freq;
using qw38::compiler::kRopeFreqs;
using qw38::compiler::require_all_finite_bf16;
using qw38::compiler::row_major_from_tile_nk;
using qw38::compiler::tile_nk_from_row_major;
using qw38::compiler::tiled_element_index;

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

std::vector<std::byte> pattern(std::uint64_t elems, std::uint16_t seed) {
  std::vector<std::byte> out(elems * 2);
  for (std::uint64_t i = 0; i < elems; ++i) {
    std::uint16_t bits = static_cast<std::uint16_t>(0x3C00 + ((seed + i) & 0x007F));
    std::memcpy(out.data() + i * 2, &bits, 2);
  }
  return out;
}

}  // namespace

int main() {
  expect(bf16_is_finite(0x3F80), "1.0 is finite");
  expect(!bf16_is_finite(0x7F80), "+inf is nonfinite");
  expect(!bf16_is_finite(0x7FC0), "nan is nonfinite");

  auto inf = pattern(4, 0);
  std::uint16_t nan = 0x7FC0;
  std::memcpy(inf.data() + 2, &nan, 2);
  auto nf = require_all_finite_bf16(inf, "t");
  expect(!nf && nf.error().code == CompilerErrorCode::Nonfinite,
         "nonfinite rejection");

  constexpr std::uint64_t n = 16;
  constexpr std::uint64_t k = 512;
  auto src = pattern(n * k, 1);
  auto tiled = tile_nk_from_row_major(src, n, k);
  if (!tiled) {
    fail(qw38::compiler::error_message(tiled.error()));
  } else {
    expect(tiled->size() == src.size(), "tile preserves numel");
    expect(tiled_element_index(n, k, 0, 0) == 0, "first element stays first");
    expect(tiled_element_index(n, k, 0, 256) == 8 * 256,
           "next K-tile starts after 8 rows of 256");
    expect(tiled_element_index(n, k, 8, 0) == 2 * 8 * 256,
           "next N-tile follows both K tiles");
    auto restored = row_major_from_tile_nk(*tiled, n, k);
    if (!restored) {
      fail(qw38::compiler::error_message(restored.error()));
    } else {
      expect(*restored == src, "exact BF16 tile inverse");
    }
  }

  auto bad = tile_nk_from_row_major(src, 7, k);
  expect(!bad && bad.error().code == CompilerErrorCode::ShapeMismatch,
         "tile mapping rejects unaligned N");

  constexpr std::uint64_t channels = 32;
  constexpr std::uint64_t taps = 4;
  auto conv = pattern(channels * taps, 9);
  auto tap = conv_to_tap_major(conv, channels, taps);
  if (!tap) {
    fail(qw38::compiler::error_message(tap.error()));
  } else {
    std::uint16_t src00 = 0;
    std::uint16_t dst00 = 0;
    std::memcpy(&src00, conv.data(), 2);
    std::memcpy(&dst00, tap->data(), 2);
    expect(src00 == dst00, "[c=0,t=0] stays at tap-major [0,0]");
    std::uint16_t src10 = 0;
    std::uint16_t dst01 = 0;
    std::memcpy(&src10, conv.data() + taps * 2, 2);
    std::memcpy(&dst01, tap->data() + 2, 2);
    expect(src10 == dst01, "[c=1,t=0] maps to [t=0,c=1]");
    auto back = conv_from_tap_major(*tap, channels, taps);
    expect(static_cast<bool>(back) && *back == conv, "exact conv inverse");
  }

  auto rope = generate_rope_inv_freq();
  expect(rope.size() == kRopeFreqs * 4, "32 FP32 frequencies");
  std::uint32_t first = 0;
  std::memcpy(&first, rope.data(), 4);
  expect(first == 0x3f800000u, "ω_0 is FP32 1.0");
  float prev = 1.0f;
  for (std::uint32_t j = 1; j < kRopeFreqs; ++j) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, rope.data() + j * 4, 4);
    float value = 0;
    std::memcpy(&value, &bits, 4);
    expect(value > 0.0f && value < prev, "RoPE frequencies decrease");
    prev = value;
  }
  auto rope2 = generate_rope_inv_freq();
  expect(rope == rope2, "RoPE generation is deterministic");

  if (g_failures != 0) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }
  return 0;
}
