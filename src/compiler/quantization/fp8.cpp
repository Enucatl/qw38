#include "compiler/quantization/fp8.hpp"
#include "format/floatcvt.hpp"
#include "cutlass/float8.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace qw38::compiler {
std::expected<Fp8Matrix, CompilerError> quantize_fp8(
    std::span<std::byte const> source, std::uint32_t n, std::uint32_t k) {
  if (!n || !k || n > 17408 || k > 17408 || source.size() != std::uint64_t(n)*k*2)
    return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch,
        "fp8.v1", "invalid BF16 extent"));
  Fp8Matrix p{n,k,(n+127)/128*128,(k+127)/128*128,{}, {}};
  p.codes.resize(std::uint64_t(p.padded_n)*p.padded_k);
  p.scales.resize(std::uint64_t(p.padded_n/128)*(p.padded_k/128),1.f);
  for (unsigned bn=0; bn<p.padded_n/128; ++bn)
    for (unsigned bk=0; bk<p.padded_k/128; ++bk) {
      float peak=0;
      for (unsigned r=bn*128;r<std::min(n,(bn+1)*128);++r)
        for (unsigned c=bk*128;c<std::min(k,(bk+1)*128);++c) {
          float v=format::bf16_to_fp32(format::load_u16_le(source.data()+(std::uint64_t(r)*k+c)*2));
          if (!std::isfinite(v)) return std::unexpected(make_error(
              CompilerErrorCode::Nonfinite,"fp8.v1","nonfinite BF16 weight"));
          peak=std::max(peak,std::abs(v));
        }
      float scale=peak==0?1.f:std::max(peak/448.f,std::numeric_limits<float>::min());
      p.scales[bk*(p.padded_n/128)+bn]=scale;
      for (unsigned r=bn*128;r<std::min(n,(bn+1)*128);++r)
        for (unsigned c=bk*128;c<std::min(k,(bk+1)*128);++c) {
          float v=format::bf16_to_fp32(format::load_u16_le(source.data()+(std::uint64_t(r)*k+c)*2));
          p.codes[std::uint64_t(r)*p.padded_k+c]=cutlass::float_e4m3_t(v/scale).raw();
        }
    }
  return p;
}
}
