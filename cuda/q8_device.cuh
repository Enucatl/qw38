#pragma once
#include <cstdint>

namespace qw38::cuda {
// All lanes participate. One warp owns exactly one K32 activation group.
__device__ __forceinline__ void q8_pack_warp(float x, std::int8_t* codes,
    float* scales, std::int32_t* sums, unsigned group, int* failure) {
  unsigned const lane = threadIdx.x & 31;
  bool const finite = isfinite(x);
  float peak = finite ? fabsf(x) : 0.f;
  for (int off = 16; off; off >>= 1)
    peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, off));
  float const a = peak == 0.f ? 1.f : __fdiv_rn(peak, 127.f);
  bool const valid = __all_sync(0xffffffff, finite) && isfinite(a) && a > 0.f;
  int const z = valid ? max(-127, min(127, __float2int_rn(__fdiv_rn(x, a)))) : 0;
  codes[std::size_t(group) * 32 + lane] = static_cast<std::int8_t>(z);
  int sum = z;
  for (int off = 16; off; off >>= 1)
    sum += __shfl_xor_sync(0xffffffff, sum, off);
  if (lane == 0) {
    scales[group] = valid ? a : __int_as_float(0x7fc00000);
    sums[group] = sum;
    if (!valid) atomicExch(failure, 1);
  }
}
}  // namespace qw38::cuda
