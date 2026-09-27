#include "cuda/q4k_q8.hpp"
#include "cuda/prefill.hpp"
#include "cuda/buffer.hpp"
#include "cuda/copy.hpp"
#include "cuda/upload.hpp"
#include "format/pack.hpp"
#include "format/floatcvt.hpp"
#include "format/reader.hpp"
#include "compiler/compile.hpp"
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace {
using namespace qw38;
void require(bool ok,char const* why) { if(!ok) {std::cerr<<"FAIL: "<<why<<'\n';std::exit(1);} }
template<class T,class E> T take(std::expected<T,E> r) {
  if(!r){std::cerr<<error_message(r.error())<<'\n';std::exit(1);}return std::move(*r);
}
void check(std::expected<void,cuda::Error> r) { if(!r){std::cerr<<cuda::error_message(r.error())<<'\n';std::exit(1);} }
template<class T> std::span<T> span(cuda::DeviceBuffer& b){return {static_cast<T*>(b.data()),b.bytes()/sizeof(T)};}
template<class T> auto upload(std::vector<T> const& v,cuda::Stream const& s){return take(cuda::upload(std::as_bytes(std::span(v)),s));}
template<class T> auto download(cuda::DeviceBuffer& b,cuda::Stream const& s){
  std::vector<T> v(b.bytes()/sizeof(T));check(cuda::copy_d2h(v.data(),b.data(),b.bytes(),s));check(s.sync());return v;
}
}
int main() {
  using namespace qw38;
  auto stream=take(cuda::Stream::create());
  constexpr unsigned n=137,k=259,pk=512,groups=pk/32;
  format::LogicalWeightCodes logical{.quantizer=format::LogicalQuantizerId::Q4KCandidateV2,
    .n=n,.k=pk,.group_size=256,.qmax=15};
  logical.codes.resize(n*pk);logical.q4k_metadata.resize(n*2*16);
  // Independent logical scalars retained for the CPU equation; no packed
  // decoder, GPU helper, or decoded BF16 contraction defines this reference.
  std::vector<float> d(n*2),dm(n*2);
  std::vector<unsigned> sc(n*groups),mn(n*groups);
  for(unsigned r=0;r<n;++r) {
    for(unsigned c=0;c<k;++c)logical.codes[r*pk+c]=(r*7+c*11)%16;
    for(unsigned b=0;b<2;++b){
      auto idx=r*2+b;auto* meta=logical.q4k_metadata.data()+idx*16;
      std::uint16_t dh=r==0?1:std::array{0x1400,0x1800,0x1a00}[r%3];
      std::uint16_t mh=r==0?2:(b==0?0x1000:0x1400);
      d[idx]=format::fp16_to_fp32(dh);dm[idx]=format::fp16_to_fp32(mh);
      meta[0]=dh&255;meta[1]=dh>>8;meta[2]=mh&255;meta[3]=mh>>8;
      auto* p=meta+4;
      for(unsigned g=0;g<8;++g){
        unsigned s=(r*13+g*17+3)%64,m=(r*11+g*7+5)%64;
        sc[r*groups+b*8+g]=s;mn[r*groups+b*8+g]=m;
        if(g<4){p[g]|=s;p[g+4]|=m;}
        else{p[g+4]|=(s&15)|((m&15)<<4);p[g-4]|=(s>>4)<<6;p[g]|=(m>>4)<<6;}
      }
    }
  }
  auto packed=take(format::pack_cuda_v0(logical.quantizer,format::PhysicalLayoutId::CudaQ4KCandidateV2,logical));
  auto wc=upload(packed.codes,stream),ws=upload(packed.scales,stream);
  cuda::PrefillWeight w{wc.data(),ws.data(),cuda::kDecodeLayoutQ4KCandidateV2,cuda::kDecodeQuantizerQ4KCandidateV2,
      n,k,unsigned(packed.padded_n),pk,wc.bytes(),ws.bytes()};
  std::vector<float> small_tile;
  for(unsigned m:{1u,3u,33u,127u,128u,129u,256u,257u,511u,512u}) {
    std::vector<std::uint16_t> x(m*k);
    for(unsigned r=0;r<m;++r)for(unsigned c=0;c<k;++c){
      float value=(int((r*23+c*19)%67)-44)*.125f*(1+r%5);
      if(c>=32 && c<64)value=0;
      if(c<5)value=std::array{127.f,.5f,1.5f,-.5f,-1.5f}[c];
      if(r==2)value=0; // Whole zero token crosses both tile schedules.
      x[r*k+c]=format::fp32_to_bf16_rne(value);
    }
    auto dx=upload(x,stream);
    auto codes=take(cuda::DeviceBuffer::allocate(m*pk)),scales=take(cuda::DeviceBuffer::allocate(m*groups*4));
    auto sums=take(cuda::DeviceBuffer::allocate(m*groups*4)),flag=take(cuda::DeviceBuffer::allocate(4));
    auto out=take(cuda::DeviceBuffer::allocate(m*n*4));check(cuda::zero(flag,stream));
    cuda::Q8Input q{span<std::int8_t>(codes),span<float>(scales),span<int>(sums),m,k,pk};
    check(cuda::pack_q8(span<std::uint16_t>(dx),q,span<int>(flag),stream));
    auto z=download<std::int8_t>(codes,stream);auto a=download<float>(scales,stream);auto sum=download<int>(sums,stream);
    require(download<int>(flag,stream)[0]==0,"finite pack");
    std::vector<int> refz(m*pk),refs(m*groups);std::vector<float> refa(m*groups);
    for(unsigned r=0;r<m;++r)for(unsigned g=0;g<groups;++g){
      float peak=0;for(unsigned t=0;t<32 && g*32+t<k;++t)peak=std::max(peak,std::abs(format::bf16_to_fp32(x[r*k+g*32+t])));
      unsigned idx=r*groups+g;refa[idx]=peak==0?1:peak/127;
      for(unsigned t=0;t<32;++t){unsigned c=g*32+t;float v=c<k?format::bf16_to_fp32(x[r*k+c]):0;
        int code=std::clamp(int(std::nearbyint(v/refa[idx])),-127,127);refz[r*pk+c]=code;refs[idx]+=code;
        require(z[r*pk+c]==code,"exact independent codes, RNE and padding");}
      require(a[idx]==refa[idx] && sum[idx]==refs[idx],"exact independent scales/sums");
      require(std::abs(sum[idx])<=cuda::kQ8SumBound,"sum bound");
    }
    require(z[0]==127 && z[1]==0 && z[2]==2 && z[3]==0 && z[4]==-2,"halfway ties and saturation endpoint");
    require(sum[2]<0,"negative activation sum fixture");
    check(cuda::q4k_q8_project(w,q,span<float>(out),stream));auto y=download<float>(out,stream);
    if(m==1) {
      // The fast load is 16 bytes, while public spans permit four-byte
      // alignment. Shift each independently and protect both output ends.
      std::vector<std::byte> shifted_weight(wc.bytes()+8);
      std::copy(packed.codes.begin(),packed.codes.end(),shifted_weight.begin()+4);
      auto unaligned_weight=upload(shifted_weight,stream);
      std::vector<std::int8_t> shifted_codes(pk+8,99);
      std::copy(z.begin(),z.end(),shifted_codes.begin()+4);
      auto unaligned_codes=upload(shifted_codes,stream);
      std::vector<float> guards(n+2,-12345.f),residual(n);
      for(unsigned r=0;r<n;++r)residual[r]=(int(r%17)-8)*.125f;
      auto guarded=upload(guards,stream),res=upload(residual,stream);
      for(unsigned variant=0;variant<4;++variant) {
        auto shifted_w=w;auto shifted_q=q;
        if(variant&1)shifted_w.codes=static_cast<std::byte*>(unaligned_weight.data())+4;
        if(variant&2)shifted_q.codes=span<std::int8_t>(unaligned_codes).subspan(4,pk);
        check(cuda::q4k_q8_project(shifted_w,shifted_q,span<float>(guarded).subspan(1,n),stream,span<float>(res)));
        auto result=download<float>(guarded,stream);
        for(unsigned r=0;r<n;++r)require(result[r+1]==y[r]+residual[r],"aligned/unaligned Q4 loads and residual exactly agree");
        require(result.front()==guards.front() && result.back()==guards.back(),"Q4 M=1 row-tail output guards");
      }
    }
    if(m==127)small_tile=y;
    if(m>=128)require(std::equal(small_tile.begin(),small_tile.end(),y.begin()),
        "J128 preserves J32 K32 expression and accumulation order bitwise");
    for(unsigned t=0;t<m;++t)for(unsigned r=0;r<n;++r){
      double ref=0;
      for(unsigned g=0;g<groups;++g){int dot=0;
        for(unsigned j=0;j<32 && g*32+j<k;++j)dot+=logical.codes[r*pk+g*32+j]*refz[t*pk+g*32+j];
        require(std::abs(dot)<=cuda::kQ4KQ8DotBound,"raw dot bound");
        float s=d[r*2+g/8]*sc[r*groups+g],minimum=dm[r*2+g/8]*mn[r*groups+g];
        ref+=refa[t*groups+g]*(s*dot-minimum*refs[t*groups+g]);
      }
      if(!std::isfinite(y[t*n+r]) || std::abs(y[t*n+r]-ref)>.02+.002*std::abs(ref)) {
        std::cerr<<"m="<<m<<" token="<<t<<" row="<<r<<" actual="<<y[t*n+r]<<" ref="<<ref<<'\n';require(false,"affine contraction");}
    }
    check(cuda::q4k_q8_project(w,q,span<float>(out),stream));require(download<float>(out,stream)==y,"reused pack identical sibling output");
    require(download<std::int8_t>(codes,stream)==z && download<float>(scales,stream)==a && download<int>(sums,stream)==sum,"consumer preserves complete pack");
    auto bad=q;bad.codes=bad.codes.first(1);require(!cuda::q4k_q8_project(w,bad,span<float>(out),stream),"truncated input rejected");
    auto badw=w;badw.codes_bytes--;require(!cuda::q4k_q8_project(badw,q,span<float>(out),stream),"truncated weights rejected");
    for(auto value:{0x7f80u,0xff80u,0x7fc0u}){
      x[0]=value;check(cuda::copy_h2d(dx.data(),std::as_bytes(std::span(x)),stream));check(cuda::zero(flag,stream));
      check(cuda::pack_q8(span<std::uint16_t>(dx),q,span<int>(flag),stream));
      require(download<int>(flag,stream)[0]==1 && std::isnan(download<float>(scales,stream)[0]),"nonfinite pack rejected, not finite zero");
    }
    std::cout<<"Q4_K/Q8 M="<<m<<" pack, affine equation, tails, bounds, reuse, rejection PASS\n";
  }
  std::vector<compiler::SyntheticTensor> tensors;
  for(auto e:compiler::expand_identity_table()) {
    using F=compiler::TensorFamily;
    if(e.family!=F::Embed && e.family!=F::LmHead && !(e.layer_index==0 &&
        (e.family==F::MlpGateProj || e.family==F::MlpUpProj || e.family==F::MlpDownProj || e.family==F::LinearAttnInProjQkv)))continue;
    e.shape={.rank=2,.dims={128,256}};
    std::vector<std::uint16_t> values(128*256);
    for(unsigned i=0;i<values.size();++i)values[i]=format::fp32_to_bf16_rne((int(i%71)-35)*.03125f);
    auto bytes=std::as_bytes(std::span(values));tensors.push_back({e,{bytes.begin(),bytes.end()}});
  }
  auto path=std::filesystem::temp_directory_path()/("qw38-q8-"+std::to_string(getpid())+".qw38");
  auto converted=path.string()+".converted";
  format::CompilerRevision revision{.ident=compiler::kFp8CompilerIdent,.major=0,.minor=1,.patch=2};
  take(compiler::compile_synthetic(path,{},{},{},revision,tensors,compiler::WeightFormatPolicy::Fp8MixerV1));
  take(compiler::convert_fp8_q8_mlp(path,converted));
  auto old=take(format::Artifact::open(path));auto changed=take(format::Artifact::open(converted));
  require(old.precision().id==format::PrecisionPolicyId::Fp8MixerV1,"conversion preserves source policy");
  require(changed.precision().id==format::PrecisionPolicyId::Fp8MixerQ8MlpV1,"new policy roundtrip");
  revision.ident=compiler::kFp8Q8MlpCompilerIdent;
  take(compiler::compile_synthetic(path,{},{},{},revision,tensors,compiler::WeightFormatPolicy::Fp8MixerQ8MlpV1));
  auto fresh=take(format::Artifact::open(path));
  for(auto const& t:fresh.schema().tensors) {
    require(std::ranges::equal(take(fresh.payload(t.logical_name)),take(changed.payload(t.logical_name))),"new compiler reproduces converted weight codes");
    require(std::ranges::equal(take(fresh.scales(t.logical_name)),take(changed.scales(t.logical_name))),"new compiler reproduces converted scales");
  }
  auto schema=fresh.schema();
  for(auto& t:schema.tensors)if(t.logical_name.ends_with(".mlp.gate_proj.weight")) {
    t.quantizer=format::LogicalQuantizerId::Q4G64CandidateV1;
    break;
  }
  require(!format::validate_schema(schema),"mixed MLP policy rejected before upload");
  schema=fresh.schema();schema.precision.id=format::PrecisionPolicyId::CandidateV2;
  require(!format::validate_schema(schema),"FP8 mislabeled as old Q4_K policy rejected");
  std::filesystem::remove(path);std::filesystem::remove(converted);
  std::cout<<"Q8 MLP compiler, conversion equality and negative policy admission PASS\n";
}
