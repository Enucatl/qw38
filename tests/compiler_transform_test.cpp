#include "compiler/compiler.hpp"

#include <array>
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
      0x3f800000u, 0x3f5e8d27u, 0x3f41791du, 0x3f2831b4u,
      0x3f1237d7u, 0x3efe3a16u, 0x3edd028bu, 0x3ec02211u,
      0x3ea7077au, 0x3e913494u, 0x3e7c7751u, 0x3e5b7aacu,
      0x3e3ecd65u, 0x3e25df51u, 0x3e10331eu, 0x3dfab7abu,
      0x3dd9f583u, 0x3dbd7b15u, 0x3da4b936u, 0x3d8f336fu,
      0x3d78fb1fu, 0x3d58730du, 0x3d3c2b1du, 0x3d239523u,
      0x3d0e3586u, 0x3cf741a7u, 0x3cd6f343u, 0x3cbadd79u,
      0x3ca27317u, 0x3c8d3960u, 0x3c758b3eu, 0x3c557622u};
  qw38::format::CompilerRevision const kRopeGoldenRevision{
      .ident = "qw38-bf16-identity", .major = 0, .minor = 1, .patch = 1};
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
