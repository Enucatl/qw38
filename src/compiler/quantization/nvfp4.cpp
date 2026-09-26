#include "compiler/quantization/nvfp4.hpp"
#include "format/nvfp4.hpp"
#include "format/floatcvt.hpp"
#include "cutlass/float8.h"
#include "cutlass/float_subbyte.h"

#include <algorithm>
#include <bit>
#include <cmath>

namespace qw38::compiler {
std::expected<qw38::format::PackedMatrix, CompilerError> quantize_nvfp4(
    std::span<std::byte const> source, std::uint64_t n, std::uint64_t k) {
  using namespace qw38::format;
  if (!n || !k || n > 248320 || k > 17408 || source.size() != n * k * 2)
    return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch,
        "nvfp4", "invalid BF16 matrix shape or extent"));
  float peak = 0;
  for (std::uint64_t i = 0; i < n * k; ++i) {
    float v = bf16_to_fp32(load_u16_le(source.data() + i * 2));
    if (!std::isfinite(v)) return std::unexpected(make_error(
        CompilerErrorCode::Nonfinite, "nvfp4", "nonfinite BF16 weight"));
    peak = std::max(peak, std::abs(v));
  }
  // Power-of-two factor keeps factor application exact; block scales use RNE.
  float factor = peak == 0 ? 1.f : std::exp2(std::ceil(std::log2(double(peak) / 2688.)));
  // The smallest BF16 subnormal divided by 2688 underflows FP32.
  factor = std::max(factor, std::numeric_limits<float>::min());
  PackedMatrix p{.layout = PhysicalLayoutId::CudaNvFp4V1,
      .quantizer = LogicalQuantizerId::NvFp4V1, .logical_n = n, .logical_k = k,
      .padded_n = dense_pad_n(n), .padded_k = dense_pad_k(k), .codes = {}, .scales = {}};
  p.codes.resize(p.padded_n * p.padded_k / 2);
  p.scales.resize(kNvFp4ScaleHeader + nvfp4_scale_count(p.padded_n, p.padded_k));
  auto bits = std::bit_cast<std::uint32_t>(factor);
  for (unsigned i = 0; i < 4; ++i) p.scales[i] = std::byte(bits >> (8 * i));
  for (std::uint64_t r = 0; r < n; ++r) {
    for (std::uint64_t b = 0; b < (k + 15) / 16; ++b) {
      float values[16]{};
      float amax = 0;
      for (unsigned i = 0; i < 16 && b * 16 + i < k; ++i) {
        values[i] = bf16_to_fp32(load_u16_le(source.data() +
            (r * k + b * 16 + i) * 2)) / factor;
        amax = std::max(amax, std::abs(values[i]));
      }
      auto scale = cutlass::float_ue4m3_t(amax / 6.f);
      std::uint8_t sb = scale.raw();
      if (amax != 0 && sb == 0) sb = 1;
      float s = nvfp4_e4m3(sb);
      p.scales[kNvFp4ScaleHeader + nvfp4_scale_index(r,b,p.padded_k)] = std::byte(sb);
      for (unsigned i = 0; i < 16; i += 2) {
        auto encode = [&](float v) -> unsigned {
          if (v == 0 || amax == 0) return 0;
          return cutlass::float_e2m1_t(v / s).raw();
        };
        auto index = (r * p.padded_k + b * 16 + i) / 2;
        auto lo = encode(values[i]), hi = encode(values[i+1]);
        if (factor > 1 && (!std::isfinite(factor * s * nvfp4_e2m1(lo)) ||
                           !std::isfinite(factor * s * nvfp4_e2m1(hi))))
          return std::unexpected(make_error(CompilerErrorCode::Unrepresentable,
              "nvfp4", "reconstructed weight exceeds FP32 range"));
        p.codes[index] = std::byte(lo | (hi << 4));
      }
    }
  }
  return p;
}
}  // namespace qw38::compiler
