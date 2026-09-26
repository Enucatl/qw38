#pragma once
#include "cutlass/float8.h"
#include "cutlass/float_subbyte.h"

namespace qw38::cuda {
__device__ __forceinline__ unsigned long long nvfp4_sf_index(
    unsigned row, unsigned block, unsigned pk) {
  return static_cast<unsigned long long>(row / 128) * 128 * (pk / 16) +
         (block / 4) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 + block % 4;
}
__device__ __forceinline__ float nvfp4_scale_value(unsigned c) {
  return c / 8 == 0 ? ldexpf(float(c & 7), -9)
      : ldexpf(1.f + float(c & 7) / 8.f, int(c / 8) - 7);
}
__device__ __forceinline__ float nvfp4_code_value(unsigned c) {
  constexpr float values[]{0, .5f, 1, 1.5f, 2, 3, 4, 6};
  return (c & 8) ? -values[c & 7] : values[c & 7];
}
// One thread owns a complete block, so nibble stores and scales cannot race.
__device__ __forceinline__ void nvfp4_pack_row(std::uint16_t const* input,
    std::uint8_t* codes, std::uint8_t* scales, unsigned row, unsigned k,
    unsigned pk, bool valid) {
  for (unsigned b = threadIdx.x; b < pk / 16; b += blockDim.x) {
    float v[16]{};
    float peak = 0;
    bool finite = true;
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) {
      if (valid && b * 16 + i < k) {
        v[i] = __uint_as_float(unsigned(input[b * 16 + i]) << 16);
        finite &= isfinite(v[i]);
        peak = fmaxf(peak, fabsf(v[i]));
      }
    }
    unsigned s = finite ? cutlass::float_ue4m3_t(peak / 6.f).raw() : 127;
    if (peak != 0 && s == 0) s = 1;
    scales[nvfp4_sf_index(row,b,pk)] = s;
    if (!valid) continue;
    float scale = nvfp4_scale_value(s);
#pragma unroll
    for (unsigned i = 0; i < 16; i += 2) {
      unsigned lo = v[i] == 0 || !finite ? 0 : cutlass::float_e2m1_t(v[i] / scale).raw();
      unsigned hi = v[i+1] == 0 || !finite ? 0 : cutlass::float_e2m1_t(v[i+1] / scale).raw();
      codes[(static_cast<unsigned long long>(row) * pk + b * 16 + i) / 2] = lo | (hi << 4);
    }
  }
}
}  // namespace qw38::cuda
