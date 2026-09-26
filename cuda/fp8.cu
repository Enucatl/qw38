#include "cuda/fp8.hpp"
#include "cuda/prefill.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include <cuda_bf16.h>
#include <math_constants.h>
#include <algorithm>
#include <array>
#include <cmath>

namespace qw38::cuda {
namespace {
using namespace cute;
using Tile=Shape<_128,_128,_128>;
using Cluster=Shape<_1,_1,_1>;
using Row=cutlass::layout::RowMajor;
using Col=cutlass::layout::ColumnMajor;
using E4=cutlass::float_e4m3_t;
using Scale=cutlass::detail::Sm120BlockwiseScaleConfig<1,128,128>;
using Epilogue=typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120,cutlass::arch::OpClassTensorOp,Tile,Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto,float,float,float,Row,4,float,Row,4,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using Mainloop=typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120,cutlass::arch::OpClassTensorOp,
    E4,cute::tuple<Row,decltype(Scale::deduce_layoutSFA())>,16,
    E4,cute::tuple<Col,decltype(Scale::deduce_layoutSFB())>,16,
    float,Tile,Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
    cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;
using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Mainloop,Epilogue,void>;
using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
std::expected<void,Error> status(cutlass::Status s) {
  if(s==cutlass::Status::kSuccess) return {};
  return std::unexpected(make_error(ErrorCode::Status,"fp8.gemm",cutlass::cutlassGetStatusString(s)));
}
std::unexpected<Error> invalid() {
  return std::unexpected(make_error(ErrorCode::InvalidArgument,"fp8.v1","invalid geometry, extent or alias"));
}
struct Region { void const* p; std::size_t bytes; unsigned alignment; bool write; };
std::expected<void,Error> validate(std::span<Region const> regions, Stream const& stream) {
  for(unsigned i=0;i<regions.size();++i) {
    auto const& a=regions[i];
    if(auto s=validate_prefill_device_span(a.p,a.bytes,a.alignment,stream.device());!s) return s;
    auto ab=reinterpret_cast<std::uintptr_t>(a.p);
    for(unsigned j=0;j<i;++j) {
      auto const& b=regions[j]; auto bb=reinterpret_cast<std::uintptr_t>(b.p);
      if((a.write || b.write) && ab<bb+b.bytes && bb<ab+a.bytes) return invalid();
    }
  }
  return {};
}
bool geometry(Fp8Weight w) {
  return w.n && w.n<=17408 && w.k && w.k<=17408 && !(w.n%128) && !(w.k%128) &&
      w.codes.size()==std::uint64_t(w.n)*w.k && w.scales.size()==std::uint64_t(w.n/128)*(w.k/128);
}
__global__ void store_kernel(float const* x,std::uint16_t* y,unsigned count) {
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count) y[i]=__bfloat16_as_ushort(__float2bfloat16_rn(x[i]));
}
__global__ void pack_kernel(std::uint16_t const* input,unsigned m,unsigned k,unsigned pm,unsigned pk,
    std::uint8_t* codes,float* scales) {
  unsigned r=blockIdx.x,b=blockIdx.y,c=b*128+threadIdx.x;
  float v=r<m && c<k?__uint_as_float(unsigned(input[std::uint64_t(r)*k+c])<<16):0.f;
  __shared__ float values[128];
  values[threadIdx.x]=isfinite(v)?fabsf(v):CUDART_INF_F;
  __syncthreads();
  for(unsigned off=64;off;off/=2) {
    if(threadIdx.x<off) values[threadIdx.x]=fmaxf(values[threadIdx.x],values[threadIdx.x+off]);
    __syncthreads();
  }
  // Nonfinite blocks carry NaN through contraction and existing finite-output
  // checks. Do not allow saturating conversion to silently hide an infinity.
  float scale=isfinite(values[0])?(values[0]==0?1.f:fmaxf(values[0]/448.f,0x1p-126f)):CUDART_NAN_F;
  if(!threadIdx.x) scales[b*pm+r]=scale;
  codes[std::uint64_t(r)*pk+c]=E4(v/scale).raw();
}
__global__ void gemv_kernel(std::uint8_t const* codes,float const* scales,
    std::uint16_t const* x,void* y,unsigned n,unsigned k,
    DecodeEpilogue epilogue = DecodeEpilogue::StoreFp32, float const* residual = nullptr) {
  unsigned row=blockIdx.x*4+threadIdx.y;
  float sum=0;
  if(row<n) for(unsigned c=threadIdx.x;c<k;c+=32) {
    E4 w; w.raw()=codes[std::uint64_t(row)*k+c];
    float weight=float(w)*scales[(c/128)*(n/128)+row/128];
    sum=fmaf(weight,__uint_as_float(unsigned(x[c])<<16),sum);
  }
  for(int off=16;off;off/=2) sum+=__shfl_down_sync(0xffffffff,sum,off);
  if(!threadIdx.x && row<n) {
    if(epilogue == DecodeEpilogue::StoreBf16)
      static_cast<std::uint16_t*>(y)[row]=__bfloat16_as_ushort(__float2bfloat16_rn(sum));
    else if(epilogue == DecodeEpilogue::ResidualAddFp32)
      static_cast<float*>(y)[row]=sum+residual[row];
    else static_cast<float*>(y)[row]=sum;
  }
}
}
std::expected<int,Error> prepare_fp8_gemm(Stream const& stream) {
  auto guard=stream.activate(); if(!guard) return std::unexpected(guard.error());
  int sm=0;
  if(auto s=check(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,stream.device()),"fp8.prepare");!s)
    return std::unexpected(s.error());
  if(auto s=check(cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,
      Kernel::SharedStorageSize),"fp8.prepare");!s) return std::unexpected(s.error());
  return sm;
}
std::expected<void,Error> fp8_store_bf16(std::span<float const> x,
    std::span<std::uint16_t> y,Stream const& stream) {
  if(x.empty() || x.size()>1024u*17408u || x.size()!=y.size() || stream.empty()) return invalid();
  std::array regions{Region{x.data(),x.size_bytes(),4,false},Region{y.data(),y.size_bytes(),2,true}};
  if(auto s=validate(regions,stream);!s) return s;
  auto guard=stream.activate(); if(!guard) return std::unexpected(guard.error());
  store_kernel<<<(x.size()+255)/256,256,0,stream.native()>>>(x.data(),y.data(),x.size());
  return check(cudaGetLastError(),"fp8.store");
}
std::expected<void,Error> pack_fp8(std::span<std::uint16_t const> input,unsigned m,unsigned k,
    std::span<std::uint8_t> codes,std::span<float> scales,Stream const& stream) {
  if(!m || m>1024 || !k || k>17408 || input.size()!=std::uint64_t(m)*k || stream.empty()) return invalid();
  unsigned pm=(m+127)/128*128,pk=(k+127)/128*128;
  if(codes.size()!=std::uint64_t(pm)*pk || scales.size()!=std::uint64_t(pm)*(pk/128)) return invalid();
  std::array regions{Region{input.data(),input.size_bytes(),2,false},
      Region{codes.data(),codes.size_bytes(),16,true},Region{scales.data(),scales.size_bytes(),16,true}};
  if(auto s=validate(regions,stream);!s) return s;
  auto guard=stream.activate(); if(!guard) return std::unexpected(guard.error());
  pack_kernel<<<dim3(pm,pk/128),128,0,stream.native()>>>(input.data(),m,k,pm,pk,codes.data(),scales.data());
  return check(cudaGetLastError(),"fp8.pack");
}
std::expected<void,Error> fp8_gemm(Fp8Weight w,Fp8Input x,std::span<float> y,
    std::span<std::byte> workspace,int sm_count,Stream const& stream) {
  if(!geometry(w) || !x.m || x.m>1024 || x.m%128 || x.k!=w.k ||
      x.codes.size()!=std::uint64_t(x.m)*x.k || x.scales.size()!=std::uint64_t(x.m)*(x.k/128) ||
      y.size()!=std::uint64_t(x.m)*w.n || workspace.empty() || sm_count<=0 || stream.empty()) return invalid();
  std::array regions{Region{w.codes.data(),w.codes.size_bytes(),16,false},Region{w.scales.data(),w.scales.size_bytes(),16,false},
      Region{x.codes.data(),x.codes.size_bytes(),16,false},Region{x.scales.data(),x.scales.size_bytes(),16,false},
      Region{y.data(),y.size_bytes(),16,true},Region{workspace.data(),workspace.size_bytes(),256,true}};
  if(auto s=validate(regions,stream);!s) return s;
  auto guard=stream.activate(); if(!guard) return std::unexpected(guard.error());
  auto shape=make_shape(int(x.m),int(w.n),int(w.k),1);
  auto sa=cutlass::make_cute_packed_stride(typename Kernel::StrideA{},{int(x.m),int(w.k),1});
  auto sb=cutlass::make_cute_packed_stride(typename Kernel::StrideB{},{int(w.n),int(w.k),1});
  auto sd=cutlass::make_cute_packed_stride(typename Kernel::StrideD{},{int(x.m),int(w.n),1});
  Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,shape,
      {reinterpret_cast<E4 const*>(x.codes.data()),sa,reinterpret_cast<E4 const*>(w.codes.data()),sb,
       x.scales.data(),Scale::tile_atom_to_shape_SFA(shape),w.scales.data(),Scale::tile_atom_to_shape_SFB(shape)},
      {{1.f,0.f},nullptr,sd,y.data(),sd}};
  args.hw_info.device_id=stream.device(); args.hw_info.sm_count=sm_count;
  if(auto s=status(Gemm::can_implement(args));!s) return s;
  if(Gemm::get_workspace_size(args)>workspace.size()) return invalid();
  if(auto s=status(Kernel::initialize_workspace(args,workspace.data(),stream.native()));!s) return s;
  auto params=Kernel::to_underlying_arguments(args,workspace.data());
  return status(Gemm::run(params,stream.native()));
}
std::expected<void,Error> fp8_decode(DecodeMmvDesc const& d,Stream const& stream) {
  auto guard=stream.activate(); if(!guard) return std::unexpected(guard.error());
  gemv_kernel<<<(d.n+3)/4,dim3(32,4),0,stream.native()>>>(
      static_cast<std::uint8_t const*>(d.codes.pointer),static_cast<float const*>(d.scales.pointer),
      static_cast<std::uint16_t const*>(d.input.pointer),d.output.pointer?d.output.pointer:d.residual.pointer,
      d.n,d.k,d.epilogue,static_cast<float const*>(d.residual.pointer));
  return check(cudaGetLastError(),"fp8.decode");
}
std::expected<void,Error> fp8_gemv(Fp8Weight w,std::span<std::uint16_t const> x,
    std::span<float> y,Stream const& stream) {
  if(!geometry(w) || x.size()!=w.k || y.size()!=w.n || stream.empty()) return invalid();
  std::array regions{Region{w.codes.data(),w.codes.size_bytes(),16,false},Region{w.scales.data(),w.scales.size_bytes(),16,false},
      Region{x.data(),x.size_bytes(),2,false},Region{y.data(),y.size_bytes(),4,true}};
  if(auto s=validate(regions,stream);!s) return s;
  auto guard=stream.activate(); if(!guard) return std::unexpected(guard.error());
  gemv_kernel<<<(w.n+3)/4,dim3(32,4),0,stream.native()>>>(w.codes.data(),w.scales.data(),x.data(),y.data(),w.n,w.k);
  return check(cudaGetLastError(),"fp8.gemv");
}
}
