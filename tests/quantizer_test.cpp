#include "compiler/quantization/quantizer.hpp"
#include "format/floatcvt.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

using qw38::compiler::CompilerErrorCode;
using qw38::compiler::quantize_bf16;
using qw38::compiler::quantize_fp32;
using qw38::compiler::quantize_group;
using qw38::format::ceil_pos_fp16;
using qw38::format::fp16_to_fp32;
using qw38::format::fp32_to_bf16_rne;
using qw38::format::kFp16MinNormal;
using qw38::format::LogicalQuantizerId;
using qw38::format::rne_to_int;

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

std::vector<float> zeros(std::size_t n) { return std::vector<float>(n, 0.0f); }

}  // namespace

int main() {
  expect(rne_to_int(1.5f) == 2, "RNE 1.5 -> 2");
  expect(rne_to_int(2.5f) == 2, "RNE 2.5 -> 2");
  expect(rne_to_int(-1.5f) == -2, "RNE -1.5 -> -2");
  expect(rne_to_int(-2.5f) == -2, "RNE -2.5 -> -2");
  expect(rne_to_int(0.5f) == 0, "RNE 0.5 -> 0");
  expect(rne_to_int(-0.5f) == 0, "RNE -0.5 -> 0");
  expect(rne_to_int(7.0f) == 7, "RNE exact 7");
  expect(rne_to_int(-7.0f) == -7, "RNE exact -7");

  {
    auto z = quantize_group(LogicalQuantizerId::Q4G64V0, zeros(64));
    expect(static_cast<bool>(z) && z->scale_bits == 0, "Q4 zero group scale");
    expect(z && z->codes.size() == 64, "Q4 zero group width");
    bool all0 = z && std::all_of(z->codes.begin(), z->codes.end(),
                                 [](std::int8_t c) { return c == 0; });
    expect(all0, "Q4 zero group codes");
  }
  {
    auto z = quantize_group(LogicalQuantizerId::Q8G32V0, zeros(32));
    expect(static_cast<bool>(z) && z->scale_bits == 0, "Q8 zero group scale");
    bool all0 = z && std::all_of(z->codes.begin(), z->codes.end(),
                                 [](std::int8_t c) { return c == 0; });
    expect(all0, "Q8 zero group codes");
  }

  {
    std::vector<float> w(64, 0.0f);
    w[0] = 7.0f;
    w[1] = 1.5f;
    w[2] = 2.5f;
    w[3] = -1.5f;
    w[4] = -2.5f;
    w[5] = 0.5f;
    w[6] = -0.5f;
    w[7] = -7.0f;
    auto q = quantize_group(LogicalQuantizerId::Q4G64V0, w);
    if (!q) {
      fail(qw38::compiler::error_message(q.error()));
    } else {
      expect(q->scale_bits == 0x3C00, "absmax 7 yields FP16 1.0");
      expect(q->codes[0] == 7 && q->codes[7] == -7, "Q4 extrema");
      expect(q->codes[1] == 2 && q->codes[2] == 2, "Q4 positive ties");
      expect(q->codes[3] == -2 && q->codes[4] == -2, "Q4 negative ties");
      expect(q->codes[5] == 0 && q->codes[6] == 0, "Q4 half-ties to even 0");
      bool forbidden = false;
      for (auto c : q->codes) {
        if (c < -7 || c > 7 || c == -8) {
          forbidden = true;
        }
      }
      expect(!forbidden, "Q4 never emits forbidden codes");
    }
  }

  {
    std::vector<float> w(32, 0.0f);
    w[0] = 127.0f;
    w[1] = 1.5f;
    w[2] = 2.5f;
    w[3] = -127.0f;
    auto q = quantize_group(LogicalQuantizerId::Q8G32V0, w);
    if (!q) {
      fail(qw38::compiler::error_message(q.error()));
    } else {
      expect(q->scale_bits == 0x3C00, "Q8 absmax 127 yields scale 1.0");
      expect(q->codes[0] == 127 && q->codes[3] == -127, "Q8 extrema");
      expect(q->codes[1] == 2 && q->codes[2] == 2, "Q8 ties");
      bool forbidden = false;
      for (auto c : q->codes) {
        if (c < -127 || c == static_cast<std::int8_t>(-128)) {
          forbidden = true;
        }
      }
      expect(!forbidden, "Q8 never emits -128");
    }
  }

  {
    std::uint16_t const s0 = 0x4000;  // 2.0
    std::uint16_t const s1 = 0x4001;  // next FP16
    float const a0 = fp16_to_fp32(s0);
    float const a1 = fp16_to_fp32(s1);
    float const just_above = std::nextafter(a0, a1);
    expect(ceil_pos_fp16(a0) == s0, "exact FP16 is kept");
    expect(ceil_pos_fp16(just_above) == s1, "scale rounds up to next FP16");
    std::vector<float> w(64, 0.0f);
    w[0] = just_above * 7.0f;
    auto q = quantize_group(LogicalQuantizerId::Q4G64V0, w);
    if (!q) {
      fail("scale round-up group");
    } else {
      expect(q->scale_bits == s1, "stored scale is ceiled FP16");
      float const stored = fp16_to_fp32(q->scale_bits);
      expect(stored >= just_above, "stored scale >= a/qmax");
    }
  }

  {
    std::vector<float> w(64, 0.0f);
    float const tiny = 3.0f * fp16_to_fp32(kFp16MinNormal) / 7.0f;
    w[0] = tiny * 3.0f;
    auto q = quantize_group(LogicalQuantizerId::Q4G64V0, w);
    if (!q) {
      fail("scale floor group");
    } else {
      expect(q->scale_bits == kFp16MinNormal, "sub-normal need floors to min normal");
      expect(q->codes[0] != 0 || tiny == 0.0f, "floor still yields a code");
    }
  }

  {
    std::vector<float> w(64, 0.0f);
    w[0] = std::numeric_limits<float>::quiet_NaN();
    auto q = quantize_group(LogicalQuantizerId::Q4G64V0, w);
    expect(!q && q.error().code == CompilerErrorCode::Nonfinite, "NaN rejected");
    w[0] = std::numeric_limits<float>::infinity();
    q = quantize_group(LogicalQuantizerId::Q4G64V0, w);
    expect(!q && q.error().code == CompilerErrorCode::Nonfinite, "+Inf rejected");
    w[0] = -std::numeric_limits<float>::infinity();
    q = quantize_group(LogicalQuantizerId::Q4G64V0, w);
    expect(!q && q.error().code == CompilerErrorCode::Nonfinite, "-Inf rejected");
  }

  {
    std::vector<float> w(64, 0.0f);
    w[0] = 1.0e10f;
    auto q = quantize_group(LogicalQuantizerId::Q4G64V0, w);
    expect(!q && q.error().code == CompilerErrorCode::Unrepresentable,
           "unrepresentable scale rejected");
  }

  {
    std::vector<float> w(64, 0.0f);
    w[0] = 7.0f * fp16_to_fp32(kFp16MinNormal);
    w[1] = 100.0f * fp16_to_fp32(kFp16MinNormal);
    auto q = quantize_group(LogicalQuantizerId::Q4G64V0, w);
    if (!q) {
      fail("saturation group");
    } else {
      expect(q->codes[1] == 7, "large-over-min-normal saturates to qmax");
      expect(q->codes[0] >= 0 && q->codes[0] <= 7, "in-range code stays in range");
    }
  }

  {
    std::vector<float> mat(8 * 64, 0.0f);
    mat[0] = 3.0f;
    mat[64] = -4.0f;
    auto q = quantize_fp32(LogicalQuantizerId::Q4G64V0, 8, 64, mat);
    if (!q) {
      fail(qw38::compiler::error_message(q.error()));
    } else {
      expect(q->n == 8 && q->k == 64 && q->group_size == 64, "logical geometry");
      expect(q->codes.size() == 8 * 64 && q->scales.size() == 8, "logical counts");
      expect(q->codes[0] != 0 && q->codes[64] != 0, "two rows quantized independently");
    }
  }

  {
    auto bad = quantize_group(LogicalQuantizerId::None, zeros(64));
    expect(!bad, "None quantizer rejected");
    auto wrong = quantize_group(LogicalQuantizerId::Q4G64V0, zeros(32));
    expect(!wrong && wrong.error().code == CompilerErrorCode::ShapeMismatch,
           "wrong group length rejected");
  }

  {
    constexpr std::uint64_t huge_n = std::uint64_t{1} << 57;
    std::span<std::byte const> empty;
    auto overflow =
        quantize_bf16(LogicalQuantizerId::Q4G64V0, huge_n, 64, empty);
    expect(!overflow && overflow.error().code == CompilerErrorCode::Format,
           "BF16 quantizer rejects byte-size overflow");
  }

  (void)fp32_to_bf16_rne;

  if (g_failures != 0) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }
  return 0;
}
