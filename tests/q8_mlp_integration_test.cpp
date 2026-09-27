#include "runtime/runtime.hpp"
#include "runtime/mlp.hpp"
#include "runtime/prefill.hpp"
#include "cuda/q4k_q8.hpp"
#include "cuda/copy.hpp"
#include "cuda/alloc.hpp"
#include "format/unpack.hpp"
#include "format/floatcvt.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace {
using namespace qw38;
void require(bool ok,char const* msg){if(!ok){std::cerr<<"FAIL: "<<msg<<'\n';std::exit(1);}}
template<class T,class E>T take(std::expected<T,E> r){if(!r){std::cerr<<error_message(r.error())<<'\n';std::exit(1);}return std::move(*r);}
template<class E>void check(std::expected<void,E> r){if(!r){std::cerr<<error_message(r.error())<<'\n';std::exit(1);}}
std::vector<float> contract(format::LogicalWeightCodes const& w,std::vector<float> const& input) {
  auto k=unsigned(w.k),n=unsigned(w.n);
  std::vector<int> z(k),sums(k/32);std::vector<float> scales(k/32);
  for(unsigned g=0;g<k/32;++g){
    float peak=0;float x[32];
    for(unsigned t=0;t<32;++t){x[t]=format::bf16_to_fp32(format::fp32_to_bf16_rne(input[g*32+t]));peak=std::max(peak,std::abs(x[t]));}
    scales[g]=peak==0?1:peak/127;
    for(unsigned t=0;t<32;++t){z[g*32+t]=std::clamp(int(std::nearbyint(x[t]/scales[g])),-127,127);sums[g]+=z[g*32+t];}
  }
  std::vector<float> result(n);
  for(unsigned r=0;r<n;++r){
    double sum=0;
    for(unsigned g=0;g<k/32;++g){
      auto* meta=w.q4k_metadata.data()+(std::size_t(r)*(k/256)+g/8)*16;auto* p=meta+4;unsigned j=g%8;
      unsigned s=j<4?p[j]&63:(p[j+4]&15)|((p[j-4]>>6)<<4);
      unsigned m=j<4?p[j+4]&63:(p[j+4]>>4)|((p[j]>>6)<<4);
      float d=format::fp16_to_fp32(meta[0]|unsigned(meta[1])<<8);
      float dm=format::fp16_to_fp32(meta[2]|unsigned(meta[3])<<8);
      int dot=0;for(unsigned t=0;t<32;++t)dot+=int(w.codes[std::size_t(r)*k+g*32+t])*z[g*32+t];
      sum+=scales[g]*((d*s)*dot-(dm*m)*sums[g]);
    }
    result[r]=float(sum);
  }
  return result;
}
std::vector<float> download(void const* p,unsigned count,cuda::Stream const& s){std::vector<float> v(count);check(cuda::copy_d2h(v.data(),p,count*4,s));check(s.sync());return v;}
void close(std::span<float const> a,std::span<float const> b){require(a.size()==b.size(),"equal extents");for(unsigned i=0;i<a.size();++i)if(!std::isfinite(a[i]) || std::abs(a[i]-b[i])>.02f+.002f*std::abs(b[i])){std::cerr<<i<<": "<<a[i]<<" vs "<<b[i]<<'\n';require(false,"independent complete MLP reference");}}
}
int main(){
  using namespace qw38;
  auto* filename=std::getenv("QW38_AUTHORITY_ARTIFACT");require(filename,"Q8 MLP artifact required");
  auto artifact=take(format::Artifact::open(filename));
  require(artifact.precision().id==format::PrecisionPolicyId::Fp8MixerQ8MlpV1,"policy 1030 required");
  auto rt=take(runtime::Runtime::create());auto model=take(rt.upload_diagnostic(artifact,runtime::DiagnosticWeights::Layer,0));
  auto session=take(rt.create_session(model,3));auto const& stream=rt.stream();
  auto decode=take(runtime::bind_mlp_plan(model,session,0,stream));
  auto prefill=take(runtime::bind_prefill_layer_projections(model,0,stream,&session));
  auto engine=take(cuda::PrefillEngine::create(stream,512));
  std::vector<float> h(512*5120);for(unsigned r=0;r<512;++r)for(unsigned c=0;c<5120;++c)h[r*5120+c]=(int((c*17)%97)-48)*.03125f;
  auto gamma=take(artifact.payload(runtime::mlp_norm_name(0)));
  double square=0;for(unsigned c=0;c<5120;++c)square+=double(h[c])*h[c];
  float inv=1.f/std::sqrt(float(square/5120)+1e-6f);
  std::vector<float> normalized(5120);for(unsigned c=0;c<5120;++c)normalized[c]=(1+format::bf16_to_fp32(format::load_u16_le(gamma.data()+2*c)))*h[c]*inv;
  auto weight=[&](std::string const& name,unsigned n,unsigned k){return take(format::unpack_cuda_v0(
      format::LogicalQuantizerId::Q4KCandidateV2,format::PhysicalLayoutId::CudaQ4KCandidateV2,n,k,
      take(artifact.payload(name)),take(artifact.scales(name))));};
  auto gate=contract(weight(runtime::mlp_gate_name(0),17408,5120),normalized);
  auto up=contract(weight(runtime::mlp_up_name(0),17408,5120),normalized);
  for(unsigned i=0;i<17408;++i)gate[i]=(gate[i]/(1+std::exp(-gate[i])))*up[i];
  auto ref=contract(weight(runtime::mlp_down_name(0),5120,17408),gate);
  for(unsigned i=0;i<5120;++i)ref[i]+=h[i];
  auto input=session.residual_h_mid().pointer,output=session.residual_h().pointer;
  check(cuda::copy_h2d(input,std::as_bytes(std::span(h)),stream));check(stream.sync());
  auto allocations=cuda::malloc_count();
  check(runtime::execute_decode_mlp(decode));close(download(output,5120,stream),ref);
  for(unsigned m:{3u,127u,128u,129u,255u,256u,257u,511u,512u,1u}){
    check(runtime::execute_prefill_mlp(prefill,engine,static_cast<float const*>(input),static_cast<float*>(output),m,0));
    auto values=download(output,m*5120,stream);
    for(unsigned r=0;r<m;++r)close(std::span(values).subspan(r*5120,5120),ref);
  }
  require(cuda::malloc_count()==allocations,"no hot allocation across chunks and phases");
  auto first=download(output,5120,stream);
  auto before=take(session.save());
  auto rejected=[&](std::expected<void,runtime::Error> result){
    require(!result && result.error().code==runtime::ErrorCode::InvalidArgument,"typed prelaunch rejection");
    require(!runtime::detail::SessionPlanAccess::execution_state(session)->is_poisoned(),"prelaunch rejection preserves session health");
    auto after=take(session.save());
    require(after.gdn_s==before.gdn_s && after.conv_history==before.conv_history && after.kv==before.kv &&
        after.conv_cursor==before.conv_cursor && after.gdn_position==before.gdn_position &&
        after.kv_populated==before.kv_populated && after.token_position==before.token_position,"rejected call preserves device state and host metadata");
    require(download(output,5120,stream)==first,"rejected call preserves output");
  };
  auto run_prefill=[&](float* target,unsigned m,float eps=1e-6f){return runtime::execute_prefill_mlp(
      prefill,engine,static_cast<float const*>(input),target,m,0,eps);};
  rejected(run_prefill(static_cast<float*>(output),0));
  rejected(run_prefill(static_cast<float*>(output),513));
  rejected(run_prefill(nullptr,1));
  rejected(run_prefill(static_cast<float*>(input),1));
  rejected(run_prefill(static_cast<float*>(output),1,0));
  auto invalid_decode=decode;invalid_decode.eps=0;
  rejected(runtime::execute_decode_mlp(invalid_decode));
  invalid_decode=decode;invalid_decode.gate.codes_bytes--;
  rejected(runtime::execute_decode_mlp(invalid_decode));
  check(runtime::execute_decode_mlp(decode));check(run_prefill(static_cast<float*>(output),1));
  close(download(output,5120,stream),ref);
  h[0]=100;check(cuda::copy_h2d(input,std::as_bytes(std::span(h)),stream));check(runtime::execute_decode_mlp(decode));
  require(download(output,5120,stream)!=first,"changed residual produces fresh gate/up and down packs");
  h[0]=std::numeric_limits<float>::infinity();check(cuda::copy_h2d(input,std::as_bytes(std::span(h)),stream));
  require(!runtime::execute_decode_mlp(decode),"nonfinite RMS typed failure");
  require(runtime::detail::SessionPlanAccess::execution_state(session)->is_poisoned(),"failure poisons session");
  require(!runtime::execute_decode_mlp(decode),"poison prevents continuation");
  check(session.reset());check(cuda::copy_h2d(input,std::as_bytes(std::span(h)),stream));
  require(!run_prefill(static_cast<float*>(output),512),"J128 nonfinite RMS typed failure");
  require(runtime::detail::SessionPlanAccess::execution_state(session)->is_poisoned(),"J128 failure poisons session");
  require(!run_prefill(static_cast<float*>(output),512),"J128 poison prevents continuation");
  check(session.reset());h[0]=-1;check(cuda::copy_h2d(input,std::as_bytes(std::span(h)),stream));
  check(runtime::execute_decode_mlp(decode));
  cuda::testing::fail_next_stream_sync();
  require(!runtime::execute_decode_mlp(decode),"deferred execution failure returned");
  require(runtime::detail::SessionPlanAccess::execution_state(session)->is_poisoned(),"submitted failure still poisons");
  check(session.reset());check(cuda::copy_h2d(input,std::as_bytes(std::span(h)),stream));check(runtime::execute_decode_mlp(decode));
  auto moved=std::move(session);check(runtime::execute_decode_mlp(decode));
  require(moved.q8_mlp_workspace().size()==cuda::q8_mlp_workspace_bytes(512),"bounded scratch follows Session move");
  std::cout<<"real Session M=1/3/127/128/129/255/256/257/511/512 complete RMS/Q8/SwiGLU/down/residual reference, reuse, poison/reset/move PASS\n";
}
