#include "compiler/quantization/fp8.hpp"
#include "compiler/compile.hpp"
#include "format/fp8.hpp"
#include "format/reader.hpp"
#include "cuda/attention.hpp"
#include "cuda/prefill.hpp"
#include <unistd.h>
#include "cuda/fp8.hpp"
#include "cuda/activation.hpp"
#include "cuda/alloc.hpp"
#include "cuda/copy.hpp"
#include "cuda/upload.hpp"
#include "format/floatcvt.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace {
using namespace qw38;
void require(bool ok,char const* msg) { if(!ok) { std::cerr<<"FAIL: "<<msg<<'\n'; std::exit(1); } }
template<class T,class E> T take(std::expected<T,E> r) {
  if(!r) { std::cerr<<error_message(r.error())<<'\n';std::exit(1); } return std::move(*r);
}
void check(std::expected<void,qw38::cuda::Error> r) { if(!r) { std::cerr<<qw38::cuda::error_message(r.error())<<'\n';std::exit(1); } }
template<class T> auto bytes(std::vector<T> const& v) { return std::as_bytes(std::span(v)); }
template<class T> std::span<T> span(qw38::cuda::DeviceBuffer& b) { return {static_cast<T*>(b.data()),b.bytes()/sizeof(T)}; }
template<class T> std::vector<T> download(qw38::cuda::DeviceBuffer& b,qw38::cuda::Stream const& s) {
  std::vector<T> v(b.bytes()/sizeof(T));check(qw38::cuda::copy_d2h(v.data(),b.data(),b.bytes(),s));check(s.sync());return v;
}
float value(unsigned code) {
  unsigned mag=code&127,exp=mag>>3,frac=mag&7;
  float v=exp?std::ldexp(1.f+frac/8.f,int(exp)-7):frac/512.f;
  return code&128?-v:v;
}
unsigned nearest(float x) {
  unsigned best=0;float err=std::abs(x);
  for(unsigned c=1;c<127;++c) { float e=std::abs(std::abs(x)-value(c)); if(e<err || (e==err && !(c&1))) {err=e;best=c;} }
  return best|(std::signbit(x)?128u:0u);
}
std::vector<std::uint16_t> input(unsigned n,unsigned k) {
  std::vector<std::uint16_t> v(n*k);
  for(unsigned r=0;r<n;++r) for(unsigned c=0;c<k;++c)
    v[r*k+c]=format::fp32_to_bf16_rne((int((r*23+c*17)%97)-48)*0.03125f*(1+r/128+c/128));
  return v;
}
}
int main() {
  using namespace qw38;
  auto stream=take(qw38::cuda::Stream::create());int sm=take(qw38::cuda::prepare_fp8_gemm(stream));
  unsigned n=259,k=387,pn=384,pk=512;
  auto w=input(n,k); w[0]=format::fp32_to_bf16_rne(80.f);
  auto p=take(compiler::quantize_fp8(bytes(w),n,k));
  using SF=cutlass::detail::Sm120BlockwiseScaleConfig<1,128,128>;
  for(unsigned br=0;br<pn/128;++br)for(unsigned bc=0;bc<pk/128;++bc) {
    float peak=0;
    for(unsigned r=br*128;r<std::min(n,(br+1)*128);++r)
      for(unsigned c=bc*128;c<std::min(k,(bc+1)*128);++c)
        peak=std::max(peak,std::abs(format::bf16_to_fp32(w[r*k+c])));
    require(p.scales[bc*(pn/128)+br]==(peak==0?1.f:std::max(peak/448.f,std::numeric_limits<float>::min())),"independent weight absmax/448 blocks");
  }
  auto layout=SF::tile_atom_to_shape_SFB(cute::make_shape(256,int(pn),int(pk),1));
  for(unsigned r=0;r<pn;++r) for(unsigned c=0;c<pk;++c) {
    auto index=layout(cute::make_coord(r,c,0));
    require(index==(c/128)*(pn/128)+r/128,"CUTLASS weight scale indexing");
    if(r>=n || c>=k) require(p.codes[r*pk+c]==0,"zero code padding");
    else require(p.codes[r*pk+c]==nearest(format::bf16_to_fp32(w[r*k+c])/p.scales[index]),"independent E4M3 RNE codes");
  }
  auto dc=take(qw38::cuda::upload(bytes(p.codes),stream)),ds=take(qw38::cuda::upload(bytes(p.scales),stream));
  qw38::cuda::Fp8Weight weight{span<std::uint8_t>(dc),span<float>(ds),pn,pk};
  auto work=take(qw38::cuda::DeviceBuffer::allocate(4*1024*1024));
  std::vector<std::uint8_t> first_codes;
  std::vector<float> first_scales;
  for(unsigned m:{1u,256u}) {
    unsigned pm=(m+127)/128*128;
    auto x=input(m,k);auto dx=take(qw38::cuda::upload(bytes(x),stream));
    auto ac=take(qw38::cuda::DeviceBuffer::allocate(pm*pk)),as=take(qw38::cuda::DeviceBuffer::allocate(pm*(pk/128)*4));
    auto out=take(qw38::cuda::DeviceBuffer::allocate(pm*pn*4));
    check(qw38::cuda::pack_fp8(span<std::uint16_t>(dx),m,k,span<std::uint8_t>(ac),span<float>(as),stream));
    auto codes=download<std::uint8_t>(ac,stream);auto scales=download<float>(as,stream);
    if(m==1) {first_codes.assign(codes.begin(),codes.begin()+pk);for(unsigned b=0;b<pk/128;++b)first_scales.push_back(scales[b*pm]);}
    else for(unsigned c=0;c<pk;++c) {
      require(codes[c]==first_codes[c],"row codes independent of adjacent rows");
      require(scales[(c/128)*pm]==first_scales[c/128],"row scales independent of adjacent rows");
    }
    for(unsigned r=0;r<pm;++r) for(unsigned c=0;c<pk;++c) {
      if(r>=m || c>=k) require(codes[r*pk+c]==0,"activation zero tails");
      else require(codes[r*pk+c]==nearest(format::bf16_to_fp32(x[r*k+c])/scales[(c/128)*pm+r]),"activation RNE");
    }
    qw38::cuda::Fp8Input packed{span<std::uint8_t>(ac),span<float>(as),pm,pk};
    check(qw38::cuda::fp8_gemm(weight,packed,span<float>(out),span<std::byte>(work),sm,stream));
    auto y=download<float>(out,stream);
    for(unsigned r:{0u,m-1}) for(unsigned col:{0u,127u,128u,258u,383u}) {
      double ref=0,l1=0;
      for(unsigned c=0;c<pk;++c) {
        double a=value(codes[r*pk+c])*scales[(c/128)*pm+r];
        double b=value(p.codes[col*pk+c])*p.scales[(c/128)*(pn/128)+col/128];
        ref+=a*b;l1+=std::abs(a*b);
      }
      require(std::isfinite(y[r*pn+col]) && std::abs(y[r*pn+col]-ref)<1e-4+1e-5*l1,"independent native contraction");
    }
    check(qw38::cuda::fp8_gemm(weight,packed,span<float>(out),span<std::byte>(work),sm,stream));
    require(download<float>(out,stream)==y,"sibling consumer reuses identical pack");
    require(download<std::uint8_t>(ac,stream)==codes && download<float>(as,stream)==scales,"consumer leaves codes/scales unchanged");
    auto bad=weight;bad.codes=bad.codes.first(1);
    require(!qw38::cuda::fp8_gemm(bad,packed,span<float>(out),span<std::byte>(work),sm,stream),"truncated weights rejected");
    std::vector<std::uint16_t> xv(pk);std::copy_n(x.begin(),k,xv.begin());
    auto dv=take(qw38::cuda::upload(bytes(xv),stream));auto vo=take(qw38::cuda::DeviceBuffer::allocate(pn*4));
    check(qw38::cuda::fp8_gemv(weight,span<std::uint16_t>(dv),span<float>(vo),stream));
    auto vy=download<float>(vo,stream);
    for(unsigned r=0;r<pn;++r) {
      double ref=0,l1=0;for(unsigned c=0;c<pk;++c) {double z=value(p.codes[r*pk+c])*p.scales[(c/128)*(pn/128)+r/128]*format::bf16_to_fp32(xv[c]);ref+=z;l1+=std::abs(z);}
      require(std::abs(vy[r]-ref)<1e-4+1e-5*l1,"independent same-view BF16 GEMV");
    }
    if(m==1) {
      // Exercise the two-byte public input alignment, vector-load ends and
      // epilogues against the independently checked standalone result.
      std::vector<std::uint16_t> shifted(pk+2,0xdead);
      std::copy(xv.begin(),xv.end(),shifted.begin()+1);
      auto misaligned=take(qw38::cuda::upload(bytes(shifted),stream));
      std::vector<float> guarded(pn+2,-12345.f);
      auto guarded_out=take(qw38::cuda::upload(bytes(guarded),stream));
      auto target=span<float>(guarded_out).subspan(1,pn);
      auto operand=span<std::uint16_t>(misaligned).subspan(1,pk);
      check(qw38::cuda::fp8_gemv(weight,operand,target,stream));
      auto actual=download<float>(guarded_out,stream);
      require(std::equal(vy.begin(),vy.end(),actual.begin()+1) &&
          actual.front()==guarded.front() && actual.back()==guarded.back(),"misaligned BF16 input and FP32 output guards");
      qw38::cuda::DecodeMmvDesc d;
      d.layout=qw38::cuda::kDecodeLayoutFp8V1;d.quantizer=qw38::cuda::kDecodeQuantizerFp8V1;
      d.n=d.padded_n=pn;d.k=d.padded_k=pk;
      d.codes=qw38::cuda::decode_matrix_view(dc.data(),qw38::cuda::DecodeDtype::Fp8,d.layout,pn,pk,pn,pk,dc.bytes(),16);
      d.scales=qw38::cuda::decode_matrix_view(ds.data(),qw38::cuda::DecodeDtype::Fp32,d.layout,1,p.scales.size(),1,p.scales.size(),ds.bytes(),16);
      d.input=qw38::cuda::decode_vector_view(operand.data(),qw38::cuda::DecodeDtype::Bf16,qw38::cuda::kDecodeLayoutBf16VectorV0,pk,pk*2,2,false);
      d.output=qw38::cuda::decode_vector_view(target.data(),qw38::cuda::DecodeDtype::Fp32,qw38::cuda::kDecodeLayoutFp32VectorV0,pn,pn*4,4,true);
      d.epilogue=qw38::cuda::DecodeEpilogue::StoreFp32;
      check(qw38::cuda::launch_decode_mmv(d,stream));
      require(download<float>(guarded_out,stream)==actual,"production FP32 output exact");
      d.epilogue=qw38::cuda::DecodeEpilogue::ResidualAddFp32;
      d.residual=d.output;d.output={};
      check(qw38::cuda::launch_decode_mmv(d,stream));
      auto added=download<float>(guarded_out,stream);
      for(unsigned r=0;r<pn;++r)require(added[r+1]==vy[r]+vy[r],"in-place FP8 residual epilogue");
      require(added.front()==guarded.front() && added.back()==guarded.back(),"in-place residual guards");
      std::vector<std::uint16_t> bf16_guard(pn+2,0xdead);
      auto bf16_out=take(qw38::cuda::upload(bytes(bf16_guard),stream));
      d.residual={};d.epilogue=qw38::cuda::DecodeEpilogue::StoreBf16;
      d.output=qw38::cuda::decode_vector_view(span<std::uint16_t>(bf16_out).data()+1,
          qw38::cuda::DecodeDtype::Bf16,qw38::cuda::kDecodeLayoutBf16VectorV0,pn,pn*2,2,true);
      check(qw38::cuda::launch_decode_mmv(d,stream));
      auto rounded=download<std::uint16_t>(bf16_out,stream);
      for(unsigned r=0;r<pn;++r)require(rounded[r+1]==format::fp32_to_bf16_rne(vy[r]),"production BF16 RNE epilogue");
      require(rounded.front()==0xdead && rounded.back()==0xdead,"BF16 output guards");
      for(auto bad:{0x7f80u,0xff80u,0x7fc0u}) {
        shifted[1]=bad;check(qw38::cuda::copy_h2d(misaligned.data(),bytes(shifted),stream));
        check(qw38::cuda::fp8_gemv(weight,operand,target,stream));
        require(!std::isfinite(download<float>(guarded_out,stream)[1]),"GEMV preserves nonfinite input");
      }
      std::cout<<"FP8 vector ends, minimum alignment, guarded epilogues and nonfinite decode PASS\n";
    }
    x[0]=0x7f80;check(qw38::cuda::copy_h2d(dx.data(),bytes(x),stream));
    check(qw38::cuda::pack_fp8(span<std::uint16_t>(dx),m,k,span<std::uint8_t>(ac),span<float>(as),stream));
    check(qw38::cuda::fp8_gemm(weight,packed,span<float>(out),span<std::byte>(work),sm,stream));
    require(!std::isfinite(download<float>(out,stream)[0]),"nonfinite activation propagates");
    std::cout<<"FP8 M="<<m<<" tails, contraction, fan-out, row independence PASS\n";
  }
  std::fill(w.begin(),w.end(),0);auto zero=take(compiler::quantize_fp8(bytes(w),n,k));
  require(std::ranges::all_of(zero.codes,[](auto v){return v==0;}) && std::ranges::all_of(zero.scales,[](auto v){return v==1;}),"zero blocks");
  w[0]=1;auto tiny=take(compiler::quantize_fp8(bytes(w),n,k));require(tiny.scales[0]==std::numeric_limits<float>::min(),"underflow clamp");
  for(unsigned v:{0x7f80u,0xff80u,0x7fc0u}) {w[0]=v;require(!compiler::quantize_fp8(bytes(w),n,k),"nonfinite weights rejected");}
  // RMS owns the pack and companion; overwriting the same residual address
  // must produce a new pack after its old siblings complete.
  std::vector<float> h(256*5120);for(unsigned i=0;i<h.size();++i)h[i]=(int(i%97)-48)*0.03125f;
  auto dh=take(qw38::cuda::upload(bytes(h),stream));std::vector<std::uint16_t> gamma(5120,0);
  auto dg=take(qw38::cuda::upload(bytes(gamma),stream));auto bf=take(qw38::cuda::DeviceBuffer::allocate(h.size()*2));
  auto ac=take(qw38::cuda::DeviceBuffer::allocate(h.size())),as=take(qw38::cuda::DeviceBuffer::allocate(256*40*4));
  auto bc=take(qw38::cuda::DeviceBuffer::allocate(h.size())),bs=take(qw38::cuda::DeviceBuffer::allocate(256*40*4));
  check(qw38::cuda::launch_hidden_rms_fp8(span<float>(dh).data(),span<std::uint16_t>(dg).data(),1e-6f,256,span<std::uint16_t>(bf).data(),span<std::uint8_t>(ac).data(),span<float>(as).data(),stream));
  auto old=download<std::uint8_t>(ac,stream);
  check(qw38::cuda::pack_fp8(span<std::uint16_t>(bf),256,5120,span<std::uint8_t>(bc),span<float>(bs),stream));
  require(old==download<std::uint8_t>(bc,stream) && download<float>(as,stream)==download<float>(bs,stream),"fused RMS matches BF16 companion recipe");
  h[0]=100;check(qw38::cuda::copy_h2d(dh.data(),bytes(h),stream));
  check(qw38::cuda::launch_hidden_rms_fp8(span<float>(dh).data(),span<std::uint16_t>(dg).data(),1e-6f,256,span<std::uint16_t>(bf).data(),span<std::uint8_t>(ac).data(),span<float>(as).data(),stream));
  require(old!=download<std::uint8_t>(ac,stream),"changed residual never reuses stale pack");
  std::cout<<"FP8 compiler boundaries and fused producer scratch reuse PASS\n";

  // Exercise the wire format and compiler policy through the real writer/reader.
  std::vector<compiler::SyntheticTensor> tensors;
  for(auto e:compiler::expand_identity_table()) {
    bool qkv=e.family==compiler::TensorFamily::LinearAttnInProjQkv && e.layer_index==0;
    if(!qkv && e.family!=compiler::TensorFamily::Embed && e.family!=compiler::TensorFamily::LmHead)continue;
    e.shape={.rank=2,.dims={128,256}};
    auto raw=input(128,256);auto b=bytes(raw);tensors.push_back({e,{b.begin(),b.end()}});
  }
  auto path=std::filesystem::temp_directory_path()/("qw38-fp8-"+std::to_string(getpid())+".qw38");
  format::CompilerRevision rev{.ident=compiler::kFp8CompilerIdent,.major=0,.minor=1,.patch=2};
  auto compiled=take(compiler::compile_synthetic(path,{},{},{},rev,tensors,compiler::WeightFormatPolicy::Fp8MixerV1));
  auto artifact=take(format::Artifact::open(path));
  require(artifact.precision().id==format::PrecisionPolicyId::Fp8MixerV1,"FP8 policy identity roundtrip");
  auto schema=artifact.schema();schema.precision.id=format::PrecisionPolicyId::CandidateV2;
  require(!format::validate_schema(schema),"reject FP8 under old precision policy");
  std::filesystem::remove(path);
  require(bool(format::validate_fp8(n,k,bytes(p.codes),bytes(p.scales))),"FP8 wire validator");
  auto corrupt=p.codes;corrupt.back()=1;
  require(!format::validate_fp8(n,k,bytes(corrupt),bytes(p.scales)),"wire padding rejected");
  corrupt=p.codes;corrupt[0]=127;
  require(!format::validate_fp8(n,k,bytes(corrupt),bytes(p.scales)),"wire NaN rejected");
  auto corrupt_scales=p.scales;corrupt_scales[0]=0;
  require(!format::validate_fp8(n,k,bytes(p.codes),bytes(corrupt_scales)),"wire zero scale rejected");

  // Output producers must round to BF16 locally, then pack exactly the same
  // row/K128 recipe as a separately materialized reference, including padding.
  constexpr unsigned m=3,pm=128,width=6144;
  auto yc=take(qw38::cuda::DeviceBuffer::allocate(pm*width));
  auto ys=take(qw38::cuda::DeviceBuffer::allocate(pm*48*4));
  auto rc=take(qw38::cuda::DeviceBuffer::allocate(pm*width));
  auto rs=take(qw38::cuda::DeviceBuffer::allocate(pm*48*4));
  auto yb=take(qw38::cuda::DeviceBuffer::allocate(m*width*2));
  auto z=input(m,width);auto dz=take(qw38::cuda::upload(bytes(z),stream));
  std::vector<float> o(m*width);for(unsigned i=0;i<o.size();++i)o[i]=(int(i%61)-30)*.02f;
  auto od=take(qw38::cuda::upload(bytes(o),stream));std::vector<std::uint16_t> norm(128,0x3f80);
  auto nd=take(qw38::cuda::upload(bytes(norm),stream));
  check(qw38::cuda::launch_gdn_gated_rms(span<float>(od).data(),span<std::uint16_t>(dz).data(),span<std::uint16_t>(nd).data(),1e-6f,m*48,span<std::uint16_t>(yb).data(),stream));
  check(qw38::cuda::pack_fp8(span<std::uint16_t>(yb),m,width,span<std::uint8_t>(rc),span<float>(rs),stream));
  check(qw38::cuda::launch_gdn_gated_rms(span<float>(od).data(),span<std::uint16_t>(dz).data(),span<std::uint16_t>(nd).data(),1e-6f,m*48,span<std::uint16_t>(yb).data(),stream,span<std::uint8_t>(yc).data(),span<float>(ys).data()));
  require(download<std::uint8_t>(rc,stream)==download<std::uint8_t>(yc,stream) && download<float>(rs,stream)==download<float>(ys,stream),"fused GDN norm pack exact");
  auto dq=take(qw38::cuda::upload(bytes(z),stream));
  auto kv=input(16*2*4*8,256);auto kd=take(qw38::cuda::upload(bytes(kv),stream));
  check(qw38::cuda::launch_attention_prefill_scan(span<std::uint16_t>(dq).data(),span<std::uint16_t>(dz).data(),span<std::uint16_t>(kd).data(),0,8,1,m,span<std::uint16_t>(yb).data(),stream));
  check(qw38::cuda::pack_fp8(span<std::uint16_t>(yb),m,width,span<std::uint8_t>(rc),span<float>(rs),stream));
  check(qw38::cuda::launch_attention_prefill_scan(span<std::uint16_t>(dq).data(),span<std::uint16_t>(dz).data(),span<std::uint16_t>(kd).data(),0,8,1,m,span<std::uint16_t>(yb).data(),stream,qw38::cuda::kAttnPrefillQueryTile,span<std::uint8_t>(yc).data(),span<float>(ys).data()));
  require(download<std::uint8_t>(rc,stream)==download<std::uint8_t>(yc,stream) && download<float>(rs,stream)==download<float>(ys,stream),"fused attention gate pack exact");
  std::cout<<"FP8 wire policy and fused output producers PASS\n";
  // Production prefill dispatch consumes a prepacked operand and preserves it
  // through sibling launches; decode dispatch uses the same weight directly.
  auto engine=take(qw38::cuda::PrefillEngine::create(stream,256,512,
      qw38::cuda::PrefillDispatch::BoundedUnpackBf16Cublas,true));
  auto op=engine.fp8_operand(256,5120);
  auto wp=take(compiler::quantize_fp8(bytes(input(128,5120)),128,5120));
  auto wc=take(qw38::cuda::upload(bytes(wp.codes),stream)),ws=take(qw38::cuda::upload(bytes(wp.scales),stream));
  qw38::cuda::PrefillWeight weight_view{wc.data(),ws.data(),qw38::cuda::kDecodeLayoutFp8V1,
      qw38::cuda::kDecodeQuantizerFp8V1,128,5120,128,5120,wc.bytes(),ws.bytes()};
  auto projected=take(qw38::cuda::DeviceBuffer::allocate(256*128*4));
  auto reference=take(qw38::cuda::DeviceBuffer::allocate(256*128*4));
  check(qw38::cuda::launch_hidden_rms_fp8(span<float>(dh).data(),span<std::uint16_t>(dg).data(),1e-6f,256,
      span<std::uint16_t>(bf).data(),const_cast<std::uint8_t*>(op.codes.data()),const_cast<float*>(op.scales.data()),stream));
  auto allocation_count=qw38::cuda::malloc_count();
  qw38::cuda::PrefillProjection projection{.weight=weight_view,.input=span<std::uint16_t>(bf).data(),
      .output=projected.data(),.valid_tokens=256,.epilogue=qw38::cuda::PrefillEpilogue::StoreFp32,.packed=op};
  check(engine.project(projection));
  check(qw38::cuda::fp8_gemm({span<std::uint8_t>(wc),span<float>(ws),128,5120},op,span<float>(reference),span<std::byte>(work),sm,stream));
  auto result=download<float>(projected,stream);
  require(result==download<float>(reference,stream),"production FP8 dispatch matches verified contraction");
  check(engine.project(projection));require(result==download<float>(projected,stream),"production sibling preserves pack");
  projection.packed={};require(!engine.project(projection),"consumer refuses missing pack");
  projection.packed=op;projection.packed.k=6144;
  require(!engine.project(projection),"consumer refuses incompatible pack");
  projection.packed={};projection.valid_tokens=1;
  check(engine.project(projection));
  check(qw38::cuda::fp8_gemv({span<std::uint8_t>(wc),span<float>(ws),128,5120},span<std::uint16_t>(bf).first(5120),span<float>(reference).first(128),stream));
  auto one=download<float>(projected,stream),ref=download<float>(reference,stream);
  require(std::equal(one.begin(),one.begin()+128,ref.begin()),"production decode matches same-view reference");
  require(qw38::cuda::malloc_count()==allocation_count,"all consumer workspace preallocated");
  std::cout<<"production FP8 consumers and ownership PASS workspace_bytes="<<engine.workspace_bytes()<<'\n';
  if(auto filename=std::getenv("QW38_AUTHORITY_ARTIFACT")) {
    auto a=take(format::Artifact::open(filename));std::uint64_t resident=0,codes=0,scales=0,largest=0;
    for(auto const& t:a.tensors()) {
      bool alias=std::ranges::any_of(a.shared_bindings(),[&](auto const& b){return b.alias_tensor_id==t.tensor_id;});
      if(alias)continue;
      resident+=t.payload.length+t.scales.length;largest=std::max({largest,t.payload.length,t.scales.length});
      if(t.quantizer==format::LogicalQuantizerId::Fp8V1) {codes+=t.payload.length;scales+=t.scales.length;}
    }
    std::cout<<"artifact_model_bytes="<<resident<<" fp8_codes_bytes="<<codes<<" fp8_scales_bytes="<<scales
        <<" largest_direct_upload_bytes="<<largest<<" extra_device_load_copy_bytes=0\n";
  }


}
