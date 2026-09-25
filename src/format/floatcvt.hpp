#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace qw38::format {

inline constexpr std::uint16_t kFp16Zero = 0x0000;
inline constexpr std::uint16_t kFp16MinNormal = 0x0400;  // 2^-14
inline constexpr std::uint16_t kFp16MaxFinite = 0x7BFF;  // 65504
inline constexpr std::uint16_t kFp16PosInf = 0x7C00;
inline constexpr std::uint16_t kBf16PosInf = 0x7F80;

[[nodiscard]] inline float fp16_to_fp32(std::uint16_t h) noexcept {
  std::uint32_t const sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  std::uint32_t const exp = (h >> 10) & 0x1Fu;
  std::uint32_t const man = h & 0x3FFu;
  std::uint32_t bits = 0;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      std::uint32_t m = man;
      std::uint32_t e = 127 - 14;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x3FFu;
      bits = sign | (e << 23) | (m << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (man << 13);
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
  }
  return std::bit_cast<float>(bits);
}

[[nodiscard]] inline float bf16_to_fp32(std::uint16_t h) noexcept {
  std::uint32_t const bits = static_cast<std::uint32_t>(h) << 16;
  return std::bit_cast<float>(bits);
}

[[nodiscard]] inline std::uint16_t load_u16_le(std::byte const* p) noexcept {
  return static_cast<std::uint16_t>(static_cast<std::uint8_t>(p[0])) |
         static_cast<std::uint16_t>(static_cast<std::uint8_t>(p[1]) << 8);
}

inline void store_u16_le(std::byte* p, std::uint16_t v) noexcept {
  p[0] = static_cast<std::byte>(v & 0xFFu);
  p[1] = static_cast<std::byte>((v >> 8) & 0xFFu);
}

[[nodiscard]] inline bool fp32_is_finite(float x) noexcept {
  return x == x && x <= std::numeric_limits<float>::max() &&
         x >= std::numeric_limits<float>::lowest();
}

// Round finite FP32 to integer with IEEE-754 ties-to-even. Values with
// magnitude >= 2^31 are saturated by the caller before use.
[[nodiscard]] inline int rne_to_int(float x) noexcept {
  double const d = static_cast<double>(x);
  double const toward_zero = std::trunc(d);
  auto const truncated = static_cast<long long>(toward_zero);
  double const afrac = std::fabs(d - toward_zero);
  if (afrac < 0.5) {
    return static_cast<int>(truncated);
  }
  if (afrac > 0.5) {
    return static_cast<int>(truncated + (d >= 0.0 ? 1 : -1));
  }
  if ((truncated & 1) == 0) {
    return static_cast<int>(truncated);
  }
  return static_cast<int>(truncated + (d >= 0.0 ? 1 : -1));
}

[[nodiscard]] inline std::uint16_t fp32_to_bf16_rne(float x) noexcept {
  std::uint32_t const bits = std::bit_cast<std::uint32_t>(x);
  std::uint32_t const exp = (bits >> 23) & 0xFFu;
  if (exp == 0xFFu) {
    std::uint16_t h = static_cast<std::uint16_t>(bits >> 16);
    if ((bits & 0x7FFFFFu) != 0) {
      h = static_cast<std::uint16_t>(h | 0x0040u);
    }
    return h;
  }
  std::uint32_t const lsb = (bits >> 16) & 1u;
  std::uint32_t const add = 0x7FFFu + lsb;
  return static_cast<std::uint16_t>((bits + add) >> 16);
}

// Smallest positive finite FP16 that is >= x, for x > 0 and finite.
// Returns 0xFFFF if x is unrepresentable as a finite FP16 (NaN/Inf/too large).
[[nodiscard]] inline std::uint16_t ceil_pos_fp16(float x) noexcept {
  if (!(x > 0.0f) || x != x) {
    return 0xFFFF;
  }
  float const max_finite = fp16_to_fp32(kFp16MaxFinite);
  if (x > max_finite) {
    return 0xFFFF;
  }
  std::uint32_t const bits = std::bit_cast<std::uint32_t>(x);
  int const exp32 = static_cast<int>((bits >> 23) & 0xFFu) - 127;
  std::uint32_t const man32 = bits & 0x7FFFFFu;
  if (exp32 < -14) {
    return kFp16MinNormal;
  }
  int const exp16 = exp32 + 15;
  if (exp16 <= 0) {
    return kFp16MinNormal;
  }
  if (exp16 >= 31) {
    return 0xFFFF;
  }
  std::uint16_t h = static_cast<std::uint16_t>(
      (static_cast<unsigned>(exp16) << 10) | (man32 >> 13));
  if ((man32 & 0x1FFFu) != 0) {
    if (h == kFp16MaxFinite) {
      return 0xFFFF;
    }
    ++h;
  }
  return h;
}

}  // namespace qw38::format
