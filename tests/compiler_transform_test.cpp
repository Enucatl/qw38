#include "compiler/compiler.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
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
    expect(tiled_element_index(n, k, 0, 0) &&
               *tiled_element_index(n, k, 0, 0) == 0,
           "first element stays first");
    expect(tiled_element_index(n, k, 0, 256) &&
               *tiled_element_index(n, k, 0, 256) == 8 * 256,
           "next K-tile starts after 8 rows of 256");
    expect(tiled_element_index(n, k, 8, 0) &&
               *tiled_element_index(n, k, 8, 0) == 2 * 8 * 256,
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
  auto empty = std::span<std::byte const>{};
  constexpr auto huge_k = std::uint64_t{1} << 61;
  for (auto transform : {tile_nk_from_row_major, row_major_from_tile_nk}) {
    auto overflow = transform(empty, 8, huge_k);
    expect(!overflow && overflow.error().code == CompilerErrorCode::Unrepresentable,
           "dense transform rejects element overflow before access");
    expect(!transform(empty, 0, 256), "dense transform rejects zero dimension");
    expect(!transform(std::span<std::byte const>{src}.first(src.size() - 2), n, k),
           "dense transform rejects short payload");
    auto long_bytes = src;
    long_bytes.resize(src.size() + 2);
    expect(!transform(long_bytes, n, k),
           "dense transform rejects long payload");
  }
  expect(!tiled_element_index(n, k, n, 0),
         "dense index rejects out-of-range row");

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
  for (auto transform : {conv_to_tap_major, conv_from_tap_major}) {
    auto overflow = transform(empty, std::numeric_limits<std::uint64_t>::max(), 2);
    expect(!overflow && overflow.error().code == CompilerErrorCode::Unrepresentable,
           "conv transform rejects element overflow before access");
    expect(!transform(empty, 0, 4), "conv transform rejects zero dimension");
    expect(!transform(std::span<std::byte const>{conv}.first(conv.size() - 2), channels, taps),
           "conv transform rejects short payload");
    auto long_bytes = conv;
    long_bytes.resize(conv.size() + 2);
    expect(!transform(long_bytes, channels, taps),
           "conv transform rejects long payload");
  }

  // These bits are independently rounded from ω_j = 10,000,000^(-2j/64),
  // rather than recomputing the implementation's pow expression. They belong
  // to the current identity-compiler revision; a numerical-convention change
  // must advance that revision and replace this complete payload deliberately.
  constexpr std::array<std::uint32_t, kRopeFreqs> kRopeBits{
      0x3f800000u, 0x3f1ab32bu, 0x3ebaf81au, 0x3e61f836u,
      0x3e088d77u, 0x3da50957u, 0x3d47763fu, 0x3cf11176u,
      0x3c91ad39u, 0x3c301052u, 0x3bd4ca14u, 0x3b80967du,
      0x3b1b690du, 0x3abbd3ecu, 0x3a6301e2u, 0x3a092e02u,
      0x39a5cb5fu, 0x394860c1u, 0x38f22ce3u, 0x3892587fu,
      0x3830df51u, 0x37d5c442u, 0x37812dacu, 0x371c1fc4u,
      0x36bcb0c1u, 0x36640cc6u, 0x3609cf4bu, 0x35a68e4cu,
      0x35494c56u, 0x34f3499cu, 0x3493048eu, 0x3431af44u};
  qw38::format::CompilerRevision const kRopeGoldenRevision{
      .ident = "qw38-bf16-identity", .major = 0, .minor = 1, .patch = 2};
  qw38::format::CompilerRevision const kCurrentRevision{
      .ident = qw38::compiler::kCompilerIdent,
      .major = qw38::compiler::kCompilerMajor,
      .minor = qw38::compiler::kCompilerMinor,
      .patch = qw38::compiler::kCompilerPatch};
  expect(kRopeGoldenRevision == kCurrentRevision,
         "RoPE golden payload is tied to compiler revision");
  auto revision_drift = kCurrentRevision;
  ++revision_drift.patch;
  expect(revision_drift != kRopeGoldenRevision,
         "RoPE golden rejects compiler revision drift");

  auto rope = generate_rope_inv_freq();
  expect(rope.size() == kRopeFreqs * 4, "32 FP32 frequencies");
  for (std::uint32_t j = 0; j < kRopeFreqs; ++j) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, rope.data() + j * 4, 4);
    expect(bits == kRopeBits[j], "RoPE FP32 golden word");
    float value = 0.0f;
    std::memcpy(&value, &bits, 4);
    auto const formula = static_cast<float>(
        std::pow(10000000.0, -static_cast<double>(j) / 32.0));
    expect(std::abs(value - formula) <=
               std::numeric_limits<float>::epsilon() * formula,
           "RoPE payload follows model theta and rotary dimension");
  }
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
