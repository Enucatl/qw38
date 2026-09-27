#include "cuda/q4k_q8.hpp"
#include "cuda/graph.hpp"
#include "cuda/activation.hpp"
#include "cuda/prefill.hpp"
#include "cuda/q8_device.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <array>
#include <cmath>
#include <limits>

namespace qw38::cuda {
namespace {
// Adapted from llama.cpp e6ab7c1a41054a888ada952eab4c886444c2f5ad:
// mmq-config-{blackwell,ampere}, mmq-load-tiles, mmq-vec-dot, mma and vecdotq.
// MIT, copyright (c) 2023-2026 The ggml authors; see third_party/llama.cpp-q4k.
// SM120 Q4_K uses Ampere integer MMA: J32 for small M, J128 for M>=128.
// Unlike upstream, group scale products and activation metadata remain FP32.
constexpr unsigned I = 128, J = 32, Stride = 68;
static_assert(kQ4KQ8DotBound < std::numeric_limits<int>::max());
static_assert(kQ8SumBound < std::numeric_limits<int>::max());

__device__ float bf16(std::uint16_t x) { return __uint_as_float(unsigned(x) << 16); }
__device__ std::uint64_t tile_row(PrefillWeight w, unsigned row, unsigned kb) {
  return (std::uint64_t(row / 8) * (w.padded_k / 256) + kb) * 8 + row % 8;
}
__device__ float2 affine(PrefillWeight w, unsigned row, unsigned kb, unsigned g) {
  if (row >= w.n) return {0,0};
  auto* meta = static_cast<std::uint8_t const*>(w.scales) + tile_row(w,row,kb)*16;
  auto* p = meta + 4;
  unsigned s = g < 4 ? p[g]&63 : (p[g+4]&15) | ((p[g-4]>>6)<<4);
  unsigned m = g < 4 ? p[g+4]&63 : (p[g+4]>>4) | ((p[g]>>6)<<4);
  auto* dm = reinterpret_cast<__half const*>(meta);
  return {__half2float(dm[0])*s, __half2float(dm[1])*m};
}
// Four adjacent logical nibbles become signed INT8 values in [0,15].
__device__ unsigned unpack4(unsigned v) {
  return (v&15) | ((v&0xf0)<<4) | ((v&0xf00)<<8) | ((v&0xf000)<<12);
}

__global__ void pack_kernel(std::uint16_t const* input, std::int8_t* codes,
    float* scales, int* sums, unsigned k, unsigned pk, int* failure) {
  unsigned c = blockIdx.x*256 + threadIdx.x;
  if (c >= pk) return; // pk is K256 padded, hence whole warps participate.
  float x = c < k ? bf16(input[std::size_t(blockIdx.y)*k+c]) : 0.f;
  q8_pack_warp(x,codes,scales,sums,(blockIdx.y*pk+c)/32,failure);
}

__global__ void swiglu_pack_kernel(float const* gate, float const* up,
    std::int8_t* codes, float* scales, int* sums, int* failure) {
  unsigned i = blockIdx.x*256 + threadIdx.x;
  float g = gate[i], u = up[i];
  float v = (g / (1.f + expf(-g))) * u;
  // A finite result must not mask a nonfinite gate/up producer.
  if (!isfinite(g) || !isfinite(u)) v = __int_as_float(0x7fc00000);
  v = __bfloat162float(__float2bfloat16_rn(v));
  q8_pack_warp(v,codes,scales,sums,i/32,failure);
}

__global__ void mmvq(PrefillWeight w, std::int8_t const* x, float const* a,
    int const* sums, float* out, float const* residual) {
  unsigned lane = threadIdx.x & 31;
  unsigned row = blockIdx.x*8 + threadIdx.x/32;
  if (row >= w.n) return;
  float acc = 0;
  // Each lane owns a complete K32 subgroup: eight DP4A instructions, with
  // accumulator reset per group. |dot|<=60960, |sum(z)|<=4064. No integer
  // accumulation crosses a group or includes scale/minimum multipliers.
  for (unsigned group = lane; group < w.padded_k/32; group += 32) {
    auto tr = tile_row(w,row,group/8);
    auto* q = reinterpret_cast<std::uint16_t const*>(
        static_cast<std::uint8_t const*>(w.codes)+tr*128+(group%8)*16);
    auto* z = reinterpret_cast<int const*>(x+group*32);
    int dot = 0;
    #pragma unroll
    for (unsigned t=0;t<8;++t) dot = __dp4a(int(unpack4(q[t])),z[t],dot);
    auto dm = affine(w,row,group/8,group%8);
    acc += a[group]*(dm.x*float(dot)-dm.y*float(sums[group]));
  }
  for (unsigned off=16;off;off>>=1) acc += __shfl_down_sync(0xffffffff,acc,off);
  if (lane==0) out[row]=acc+(residual?residual[row]:0.f);
}

__global__ void mmq(PrefillWeight w, std::int8_t const* x, float const* a,
    int const* sums, float* out, unsigned m, float const* residual) {
  __shared__ unsigned wx[I][Stride], ax[J][Stride];
  unsigned lane=threadIdx.x&31, warp=threadIdx.x/32;
  unsigned r0=warp*16+lane/4, c0=(lane%4)*2;
  float acc[4][4]{};
  for (unsigned kb=0;kb<w.padded_k/256;++kb) {
    for (unsigned idx=threadIdx.x;idx<I*32;idx+=256) {
      unsigned r=idx/32, c=idx%32, row=blockIdx.x*I+r;
      unsigned v=0;
      if (row<w.n) v=reinterpret_cast<unsigned const*>(w.codes)[tile_row(w,row,kb)*32+c];
      wx[r][2*c]=unpack4(v&0xffff);
      wx[r][2*c+1]=unpack4(v>>16);
    }
    for (unsigned idx=threadIdx.x;idx<J*64;idx+=256) {
      unsigned r=idx/64,c=idx%64,token=blockIdx.y*J+r;
      ax[r][c]=token<m ? reinterpret_cast<unsigned const*>(x)[std::size_t(token)*(w.padded_k/4)+kb*64+c] : 0;
    }
    __syncthreads();
    #pragma unroll
    for (unsigned g=0;g<8;++g) {
      unsigned k=g*8+lane%4;
      unsigned av[4]{wx[r0][k],wx[r0+8][k],wx[r0][k+4],wx[r0+8][k+4]};
      float2 dm[2]{affine(w,blockIdx.x*I+r0,kb,g),affine(w,blockIdx.x*I+r0+8,kb,g)};
      #pragma unroll
      for (unsigned j=0;j<4;++j) {
        unsigned bv0=ax[j*8+lane/4][k],bv1=ax[j*8+lane/4][k+4];
        int d[4]{}; // Exact K32 dot; never carries integer state into next group.
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d[0]),"+r"(d[1]),"+r"(d[2]),"+r"(d[3])
            : "r"(av[0]),"r"(av[1]),"r"(av[2]),"r"(av[3]),"r"(bv0),"r"(bv1));
        #pragma unroll
        for (unsigned l=0;l<4;++l) {
          unsigned token=blockIdx.y*J+j*8+c0+l%2;
          if (token<m) {
            auto index=std::size_t(token)*(w.padded_k/32)+kb*8+g;
            acc[j][l]+=a[index]*(dm[l/2].x*float(d[l])-dm[l/2].y*float(sums[index]));
          }
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (unsigned j=0;j<4;++j) {
    #pragma unroll
    for (unsigned l=0;l<4;++l) {
      unsigned row=blockIdx.x*I+r0+(l/2)*8,token=blockIdx.y*J+j*8+c0+l%2;
      if (row<w.n && token<m) {
        auto i=std::size_t(token)*w.n+row;
        out[i]=acc[j][l]+(residual?residual[i]:0.f);
      }
    }
  }
}

// FP32 metadata doubles upstream's half2 storage. Both row strides remain
// 4 mod 8 words for the MMA bank layout: W=(64+8+8+4), X=(32+4+4+4).
// 128 rows each consume 65536 shared bytes; only X's K128 half is replaced.
struct WeightTile { unsigned codes[64]; float scale[8], minimum[8]; unsigned pad[4]; };
struct InputTile { unsigned codes[32]; float scale[4]; int sum[4]; unsigned pad[4]; };
struct MmqShared { WeightTile w[128]; InputTile x[128]; };
static_assert(sizeof(MmqShared)==65536);

template<bool Full>
__global__ __launch_bounds__(256,1) void mmq_j128(PrefillWeight w,
    std::int8_t const* x, float const* a, int const* sums, float* out,
    unsigned m, float const* residual) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto& tile=*reinterpret_cast<MmqShared*>(storage);
  unsigned lane=threadIdx.x%32,warp=threadIdx.x/32;
  // A pair of warps owns 32 output rows and alternating 8-token fragments.
  unsigned r0=(warp/2)*32, j0=(warp%2)*8;
  float acc[8][2][4]{};
  for(unsigned kb=0;kb<w.padded_k/256;++kb) {
    for(unsigned idx=threadIdx.x;idx<128*32;idx+=256) {
      unsigned r=idx/32,c=idx%32,row=blockIdx.x*128+r,v=0;
      if(Full || row<w.n)
        v=reinterpret_cast<unsigned const*>(w.codes)[tile_row(w,row,kb)*32+c];
      tile.w[r].codes[2*c]=unpack4(v&0xffff);
      tile.w[r].codes[2*c+1]=unpack4(v>>16);
    }
    // One lane decodes each row/K32 metadata pair, shared by all fragments.
    for(unsigned idx=threadIdx.x;idx<128*8;idx+=256) {
      unsigned r=idx/8,g=idx%8;
      auto dm=affine(w,blockIdx.x*128+r,kb,g);
      tile.w[r].scale[g]=dm.x;tile.w[r].minimum[g]=dm.y;
    }
    #pragma unroll
    for(unsigned half=0;half<2;++half) {
      for(unsigned idx=threadIdx.x;idx<128*32;idx+=256) {
        unsigned t=idx/32,c=idx%32,token=blockIdx.y*128+t;
        tile.x[t].codes[c]=(Full || token<m) ?
            reinterpret_cast<unsigned const*>(x)[std::size_t(token)*(w.padded_k/4)+kb*64+half*32+c] : 0;
      }
      for(unsigned idx=threadIdx.x;idx<128*4;idx+=256) {
        unsigned t=idx/4,g=idx%4,token=blockIdx.y*128+t;
        auto index=std::size_t(token)*(w.padded_k/32)+kb*8+half*4+g;
        tile.x[t].scale[g]=(Full || token<m)?a[index]:0.f;
        tile.x[t].sum[g]=(Full || token<m)?sums[index]:0;
      }
      __syncthreads();
      unsigned av[2][4][4];
      float2 dm[2][2][4];
      // Preread both output fragments for this half, then reuse across J.
      #pragma unroll
      for(unsigned n=0;n<2;++n) {
        #pragma unroll
        for(unsigned g=0;g<4;++g) {
          unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(
              &tile.w[r0+n*16+lane%16].codes[half*32+g*8+(lane/16)*4]));
          asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
              : "=r"(av[n][g][0]),"=r"(av[n][g][1]),"=r"(av[n][g][2]),"=r"(av[n][g][3]) : "r"(address));
          #pragma unroll
          for(unsigned l=0;l<2;++l) {
            auto const& row=tile.w[r0+n*16+lane/4+l*8];
            dm[n][l][g]={row.scale[half*4+g],row.minimum[half*4+g]};
          }
        }
      }
      #pragma unroll
      for(unsigned j=0;j<8;++j) {
        #pragma unroll
        for(unsigned g=0;g<4;++g) {
          auto const& row=tile.x[j*16+j0+lane/4];
          unsigned bv0=row.codes[g*8+lane%4],bv1=row.codes[g*8+lane%4+4];
          float scale[2];int sum[2];
          #pragma unroll
          for(unsigned l=0;l<2;++l) {
            auto const& meta=tile.x[j*16+j0+(lane%4)*2+l];
            scale[l]=meta.scale[g];sum[l]=meta.sum[g];
          }
          #pragma unroll
          for(unsigned n=0;n<2;++n) {
            int d[4]{};
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+r"(d[0]),"+r"(d[1]),"+r"(d[2]),"+r"(d[3])
                : "r"(av[n][g][0]),"r"(av[n][g][1]),"r"(av[n][g][2]),"r"(av[n][g][3]),"r"(bv0),"r"(bv1));
            #pragma unroll
            for(unsigned l=0;l<4;++l)
              acc[j][n][l]+=scale[l%2]*(dm[n][l/2][g].x*float(d[l])-dm[n][l/2][g].y*float(sum[l%2]));
          }
        }
      }
      __syncthreads();
    }
  }
  #pragma unroll
  for(unsigned j=0;j<8;++j) {
    #pragma unroll
    for(unsigned n=0;n<2;++n) {
      #pragma unroll
      for(unsigned l=0;l<4;++l) {
        unsigned row=blockIdx.x*128+r0+n*16+lane/4+(l/2)*8;
        unsigned token=blockIdx.y*128+j*16+j0+(lane%4)*2+l%2;
        if(Full || (row<w.n && token<m)) {
          auto index=std::size_t(token)*w.n+row;
          out[index]=acc[j][n][l]+(residual?residual[index]:0.f);
        }
      }
    }
  }
}

struct Region { void const* p; std::uint64_t n; unsigned alignment; };
std::expected<void,Error> distinct(std::span<Region const> regions, Stream const& s) {
  if (s.empty()) return std::unexpected(make_error(ErrorCode::InvalidArgument,"q8.stream","closed stream"));
  for (unsigned i=0;i<regions.size();++i) {
    auto r=regions[i];
    if (auto st=validate_prefill_device_span(r.p,r.n,r.alignment,s.device());!st) return st;
    auto b=reinterpret_cast<std::uintptr_t>(r.p);
    if (r.n>std::numeric_limits<std::uintptr_t>::max()-b)
      return std::unexpected(make_error(ErrorCode::Overflow,"q8.span","address overflow"));
    for (unsigned j=0;j<i;++j) {
      auto c=reinterpret_cast<std::uintptr_t>(regions[j].p);
      if (b<c+regions[j].n && c<b+r.n)
        return std::unexpected(make_error(ErrorCode::InvalidArgument,"q8.alias","live operands overlap"));
    }
  }
  return {};
}
bool valid_input(Q8Input x) {
  return x.m && x.m<=256 && x.k && x.k<=17408 && x.padded_k==decode_pad_k(x.k) &&
      x.codes.size()==std::uint64_t(x.m)*x.padded_k &&
      x.scales.size()==x.codes.size()/32 && x.sums.size()==x.scales.size();
}
std::expected<void,Error> valid_weight(PrefillWeight w) {
  if (!w.n || w.n>17408 || !w.k || w.k>17408 || w.layout!=kDecodeLayoutQ4KCandidateV2 ||
      w.quantizer!=kDecodeQuantizerQ4KCandidateV2 || w.padded_n!=decode_pad_n(w.n) ||
      w.padded_k!=decode_pad_k(w.k) || w.codes_bytes!=decode_code_bytes(w.layout,w.padded_n,w.padded_k) ||
      w.scales_bytes!=decode_scale_bytes(w.layout,w.padded_n,w.padded_k))
    return std::unexpected(make_error(ErrorCode::InvalidArgument,"q8.weight","Q4_K geometry required"));
  return {};
}
std::expected<void,Error> project(PrefillWeight w,Q8Input x,float* out,float const* residual,Stream const& s) {
  if (x.m==1) mmvq<<<(w.n+7)/8,256,0,s.native()>>>(w,x.codes.data(),x.scales.data(),x.sums.data(),out,residual);
  else if(x.m>=128) {
    auto kernel=(w.n%128==0 && w.k==w.padded_k && x.m%128==0)?mmq_j128<true>:mmq_j128<false>;
    if(auto st=check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(MmqShared)),"q4k_q8.shared");!st)return st;
    kernel<<<dim3((w.n+127)/128,(x.m+127)/128),256,sizeof(MmqShared),s.native()>>>(w,x.codes.data(),x.scales.data(),x.sums.data(),out,x.m,residual);
  }
  else mmq<<<dim3((w.n+I-1)/I,(x.m+J-1)/J),256,0,s.native()>>>(w,x.codes.data(),x.scales.data(),x.sums.data(),out,x.m,residual);
  return check(cudaGetLastError(),"q4k_q8.project");
}
} // namespace

std::expected<void,Error> pack_q8(std::span<std::uint16_t const> input,Q8Input x,
    std::span<int> failure,Stream const& stream) {
  if (!valid_input(x) || input.size()!=std::uint64_t(x.m)*x.k || failure.size()!=1)
    return std::unexpected(make_error(ErrorCode::InvalidArgument,"q8.pack","invalid geometry"));
  std::array regions{Region{input.data(),input.size_bytes(),2},Region{x.codes.data(),x.codes.size_bytes(),4},
      Region{x.scales.data(),x.scales.size_bytes(),4},Region{x.sums.data(),x.sums.size_bytes(),4},Region{failure.data(),4,4}};
  if (auto st=distinct(regions,stream);!st) return st;
  auto guard=stream.activate();if(!guard)return std::unexpected(guard.error());
  pack_kernel<<<dim3(x.padded_k/256,x.m),256,0,stream.native()>>>(input.data(),x.codes.data(),x.scales.data(),x.sums.data(),x.k,x.padded_k,failure.data());
  return check(cudaGetLastError(),"q8.pack");
}

std::expected<void,Error> q4k_q8_project(PrefillWeight const& w,Q8Input x,
    std::span<float> out,Stream const& stream,std::span<float const> residual) {
  if (auto st=valid_weight(w);!st)return st;
  if (!valid_input(x) || x.k!=w.k || out.size()!=std::uint64_t(x.m)*w.n ||
      (!residual.empty() && residual.size()!=out.size()))
    return std::unexpected(make_error(ErrorCode::InvalidArgument,"q8.project","invalid geometry"));
  std::array regions{Region{w.codes,w.codes_bytes,4},Region{w.scales,w.scales_bytes,2},
      Region{x.codes.data(),x.codes.size_bytes(),4},Region{x.scales.data(),x.scales.size_bytes(),4},
      Region{x.sums.data(),x.sums.size_bytes(),4},Region{out.data(),out.size_bytes(),4},
      Region{residual.data(),residual.size_bytes(),4}};
  if(auto st=distinct(std::span(regions).first(residual.empty()?6:7),stream);!st)return st;
  auto guard=stream.activate();if(!guard)return std::unexpected(guard.error());
  return project(w,x,out.data(),residual.data(),stream);
}

std::expected<void,Error> q4k_q8_mlp(PrefillWeight const& gate,PrefillWeight const& up,
    PrefillWeight const& down,float const* residual,std::uint16_t const* gamma,
    float eps,float* output,unsigned m,std::span<std::byte> workspace,Stream const& stream,bool& submitted, int* pending_failure) {
  submitted=false;
  if (!pending_failure) {
    if (auto st = require_uncaptured(stream); !st) return st;
  }
  if (!m || m>256 || workspace.size()<q8_mlp_workspace_bytes(m) ||
      !std::isfinite(eps) || eps<=0 || gate.n!=17408 || gate.k!=5120 ||
      up.n!=gate.n || up.k!=gate.k || down.n!=5120 || down.k!=17408)
    return std::unexpected(make_error(ErrorCode::InvalidArgument,"q8.mlp","invalid geometry"));
  for (auto const* w:{&gate,&up,&down}) if(auto st=valid_weight(*w);!st)return st;
  std::array regions{Region{gate.codes,gate.codes_bytes,4},Region{gate.scales,gate.scales_bytes,2},
      Region{up.codes,up.codes_bytes,4},Region{up.scales,up.scales_bytes,2},
      Region{down.codes,down.codes_bytes,4},Region{down.scales,down.scales_bytes,2},
      Region{residual,std::uint64_t(m)*5120*4,4},Region{gamma,5120*2,2},
      Region{output,std::uint64_t(m)*5120*4,4},Region{workspace.data(),workspace.size(),4},
      Region{pending_failure,4,4}};
  if(auto st=distinct(std::span(regions).first(pending_failure ? 11 : 10),stream);!st)return st;
  auto guard=stream.activate();if(!guard)return std::unexpected(guard.error());
  auto* codes=reinterpret_cast<std::int8_t*>(workspace.data());
  auto* scales=reinterpret_cast<float*>(codes+std::size_t(m)*17408);
  auto* sums=reinterpret_cast<int*>(scales+std::size_t(m)*544);
  auto* g=reinterpret_cast<float*>(sums+std::size_t(m)*544);
  auto* u=g+std::size_t(m)*17408;
  auto* failure=pending_failure ? pending_failure : reinterpret_cast<int*>(u+std::size_t(m)*17408);
  auto fail=[&](Error e)->std::expected<void,Error> { if (!pending_failure) (void)stream.sync();return std::unexpected(std::move(e)); };
  submitted=true;
  if (!pending_failure)
    if(auto st=check(cudaMemsetAsync(failure,0,4,stream.native()),"q8.clear_failure");!st)return fail(st.error());
  if(auto st=launch_hidden_rms_q8(residual,gamma,eps,m,codes,scales,sums,failure,stream);!st)return fail(st.error());
  Q8Input x{{codes,std::size_t(m)*5120},{scales,std::size_t(m)*160},{sums,std::size_t(m)*160},m,5120,5120};
  if(auto st=project(gate,x,g,nullptr,stream);!st)return fail(st.error());
  if(auto st=project(up,x,u,nullptr,stream);!st)return fail(st.error());
  swiglu_pack_kernel<<<m*17408/256,256,0,stream.native()>>>(g,u,codes,scales,sums,failure);
  if(auto st=check(cudaGetLastError(),"q8.swiglu_pack");!st)return fail(st.error());
  x={{codes,std::size_t(m)*17408},{scales,std::size_t(m)*544},{sums,std::size_t(m)*544},m,17408,17408};
  if(auto st=project(down,x,output,residual,stream);!st)return fail(st.error());
  if (pending_failure) return {};
  int bad=0;
  if(auto st=check(cudaMemcpyAsync(&bad,failure,4,cudaMemcpyDeviceToHost,stream.native()),"q8.failure");!st)return fail(st.error());
  if(auto st=stream.sync();!st)return st;
  if(bad)return std::unexpected(make_error(ErrorCode::InvalidArgument,"q8.producer","nonfinite producer or invalid Q8 scale"));
  return {};
}
} // namespace qw38::cuda
