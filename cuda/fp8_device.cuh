#pragma once
#include <cuda_fp8.h>
#include <math_constants.h>
#include <cstdint>

namespace qw38::cuda {
// A warp owns one K128 block, with the same row-local recipe at every M.
__device__ inline void fp8_pack_row(std::uint16_t const* row, unsigned r,
    unsigned pm, unsigned k, std::uint8_t* codes, float* scales,
    unsigned first_block = 0, unsigned row_stride = 0) {
  if(!row_stride) row_stride=k;
  unsigned lane=threadIdx.x%32,warp=threadIdx.x/32;
  for(unsigned b=warp;b<k/128;b+=blockDim.x/32) {
    float v[4],peak=0;
    for(unsigned i=0;i<4;++i) {
      v[i]=row?__uint_as_float(unsigned(row[b*128+lane+i*32])<<16):0.f;
      peak=fmaxf(peak,isfinite(v[i])?fabsf(v[i]):CUDART_INF_F);
    }
    for(int off=16;off;off/=2) peak=fmaxf(peak,__shfl_xor_sync(0xffffffff,peak,off));
    float scale=isfinite(peak)?(peak==0?1.f:fmaxf(peak/448.f,0x1p-126f)):CUDART_NAN_F;
    if(!lane) scales[(b+first_block)*pm+r]=scale;
    for(unsigned i=0;i<4;++i)
      codes[std::uint64_t(r)*row_stride+(b+first_block)*128+lane+i*32]=__nv_fp8_e4m3(v[i]/scale).__x;
  }
}
}
