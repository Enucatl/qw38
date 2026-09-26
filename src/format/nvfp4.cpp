#include "format/nvfp4.hpp"
#include "format/floatcvt.hpp"
#include "format/layout.hpp"

#include <bit>
#include <cmath>

namespace qw38::format {
float nvfp4_e2m1(std::uint8_t c) noexcept {
  constexpr float values[]{0, .5f, 1, 1.5f, 2, 3, 4, 6};
  return (c & 8) ? -values[c & 7] : values[c & 7];
}
float nvfp4_e4m3(std::uint8_t c) noexcept {
  return (c >> 3) == 0 ? std::ldexp(float(c & 7), -9)
      : std::ldexp(1.f + float(c & 7) / 8.f, int(c >> 3) - 7);
}
float nvfp4_factor(std::span<std::byte const> scales) noexcept {
  if (scales.size() < 4) return 0;
  std::uint32_t bits = 0;
  for (unsigned i = 0; i < 4; ++i) bits |= std::uint32_t(scales[i]) << (8 * i);
  return std::bit_cast<float>(bits);
}
std::expected<void, FormatError> validate_nvfp4(std::uint64_t n, std::uint64_t k,
    std::span<std::byte const> codes, std::span<std::byte const> scales) {
  auto fail = [] { return std::unexpected(make_error(FormatErrorCode::InvalidSpan,
      0, "nvfp4", "invalid geometry, scale/factor, padding or zero-block codes")); };
  if (!n || !k || n > UINT32_MAX - 127u || k > 17408) return fail();
  auto pn = dense_pad_n(n), pk = dense_pad_k(k);
  auto count = checked_mul(pn, pk, 0, "nvfp4");
  if (!count || codes.size() != *count / 2 ||
      scales.size() != kNvFp4ScaleHeader + nvfp4_scale_count(pn, pk)) return fail();
  float factor = nvfp4_factor(scales);
  if (!std::isfinite(factor) || factor <= 0) return fail();
  for (unsigned i = 4; i < kNvFp4ScaleHeader; ++i)
    if (scales[i] != std::byte{}) return fail();
  for (std::uint64_t r = 0; r < ((pn + 127) / 128) * 128; ++r) {
    for (std::uint64_t b = 0; b < pk / 16; ++b) {
      auto s = std::uint8_t(scales[kNvFp4ScaleHeader + nvfp4_scale_index(r,b,pk)]);
      if (s >= 127 || ((r >= n || b * 16 >= k) && s != 0)) return fail();
      if (r >= pn) continue;
      bool overflow_risk = factor > std::numeric_limits<float>::max() / (nvfp4_e4m3(s) * 6.f);
      for (std::uint64_t i = 0; i < 16; ++i) {
        auto col = b * 16 + i, index = r * pk + col;
        auto c = (std::uint8_t(codes[index / 2]) >> (4 * (index & 1))) & 15;
        if ((r >= n || col >= k || s == 0) && c != 0) return fail();
        if (overflow_risk && !std::isfinite(factor * nvfp4_e4m3(s) * nvfp4_e2m1(c))) return fail();
      }
    }
  }
  return {};
}
}  // namespace qw38::format
