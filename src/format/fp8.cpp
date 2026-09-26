#include "format/fp8.hpp"
#include <bit>
#include <cmath>
namespace qw38::format {
std::expected<void,FormatError> validate_fp8(std::uint64_t n,std::uint64_t k,
    std::span<std::byte const> codes,std::span<std::byte const> scales) {
  auto fail=[] {return std::unexpected(make_error(FormatErrorCode::InvalidSpan,0,"fp8.v1","invalid geometry, codes, scales or padding"));};
  if(!n || !k || n>17408 || k>17408)return fail();
  auto pn=(n+127)/128*128,pk=(k+127)/128*128;
  if(codes.size()!=pn*pk || scales.size()!=(pn/128)*(pk/128)*4)return fail();
  for(std::size_t i=0;i<scales.size();i+=4) {
    std::uint32_t bits=0;for(unsigned b=0;b<4;++b)bits|=std::uint32_t(scales[i+b])<<(8*b);
    float s=std::bit_cast<float>(bits);
    if(!std::isfinite(s) || s<0x1p-126f)return fail();
  }
  for(std::uint64_t r=0;r<pn;++r)for(std::uint64_t c=0;c<pk;++c) {
    auto code=unsigned(codes[r*pk+c]);
    if((code&127)==127 || ((r>=n || c>=k) && code!=0))return fail();
  }
  return {};
}
}
