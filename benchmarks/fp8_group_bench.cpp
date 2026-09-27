#include "compiler/checkpoint.hpp"
#include "compiler/quantization/fp8.hpp"
#include "cuda/activation.hpp"
#include "cuda/alloc.hpp"
#include "cuda/copy.hpp"
#include "cuda/event.hpp"
#include "cuda/fp8.hpp"
#include "cuda/prefill.hpp"
#include "cuda/upload.hpp"
#include "format/floatcvt.hpp"
#include "format/reader.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

namespace {
using namespace qw38;
template<class T,class E> T take(std::expected<T,E> r) {
  if(!r) { std::cerr<<error_message(r.error())<<'\n';std::exit(1); } return std::move(*r);
}
void check(std::expected<void,cuda::Error> r) { if(!r) {std::cerr<<error_message(r.error())<<'\n';std::exit(1);} }
template<class T> auto bytes(std::vector<T> const& v) {return std::as_bytes(std::span(v));}
template<class T> std::span<T> span(cuda::DeviceBuffer& b) {return {static_cast<T*>(b.data()),b.bytes()/sizeof(T)};}
}
// Policy-1030 decode groups, using the resident artifact and production ranged
// launchers. Each process measures one first use including preparation/readout.
int decode_group(char const* path, std::string const& group, char const* saved_path) {
  using namespace qw38;
  bool gdn=group=="gdn-in" || group=="gdn-out";
  bool front=group=="gdn-in" || group=="attention-in";
  if(!gdn && group!="attention-in" && group!="attention-out")return 2;
  auto setup=std::chrono::steady_clock::now();
  auto artifact=take(format::Artifact::open(path));
  auto stream=take(cuda::Stream::create());
  unsigned k=front?5120:6144;
  std::string layer="model.language_model.layers."+std::string(gdn?"0.":"3.");
  std::vector<std::string> names;
  if(front && gdn)names={"linear_attn.in_proj_qkv.weight","linear_attn.in_proj_z.weight",
      "linear_attn.in_proj_a.weight","linear_attn.in_proj_b.weight"};
  else if(front)names={"self_attn.q_proj.weight","self_attn.k_proj.weight","self_attn.v_proj.weight"};
  else names={gdn?"linear_attn.out_proj.weight":"self_attn.o_proj.weight"};
  std::vector<float> residual(5120);
  std::vector<std::uint16_t> input(k);
  for(unsigned c=0;c<k;++c)input[c]=format::fp32_to_bf16_rne((int(c*17%97)-48)*.03125f);
  for(unsigned c=0;c<5120;++c)residual[c]=(int(c*19%89)-44)*.03125f;
  auto x=take(cuda::upload(bytes(input),stream)),h=take(cuda::upload(bytes(residual),stream));
  cuda::DeviceBuffer gamma;
  if(front)gamma=take(cuda::upload(take(artifact.payload(layer+"input_layernorm.weight")),stream));
  std::array<cuda::DeviceBuffer,4> codes,scales,output;
  std::array<std::vector<std::byte>,4> host;
  std::array<cuda::DecodeMmvDesc,4> desc;
  std::uint64_t resident=0;
  for(unsigned i=0;i<names.size();++i) {
    auto name=layer+names[i];auto const* t=artifact.find_tensor(name);if(!t)return 2;
    codes[i]=take(cuda::upload(take(artifact.payload(name)),stream));
    auto s=take(artifact.scales(name));if(!s.empty())scales[i]=take(cuda::upload(s,stream));
    auto& d=desc[i];d.n=t->shape.logical[0];d.k=t->shape.logical[1];
    d.padded_n=t->shape.padded[0];d.padded_k=t->shape.padded[1];
    d.layout=std::uint16_t(t->layout);d.quantizer=std::uint16_t(t->quantizer);
    bool fp8=d.layout==cuda::kDecodeLayoutFp8V1,bf16=front && (!gdn || i<2);
    if(d.k!=k || fp8!=(i<2 || !gdn))return 2;
    d.codes=cuda::decode_matrix_view(codes[i].data(),fp8?cuda::DecodeDtype::Fp8:cuda::DecodeDtype::Bf16,
        d.layout,d.n,d.k,d.padded_n,d.padded_k,codes[i].bytes(),16);
    if(fp8)d.scales=cuda::decode_matrix_view(scales[i].data(),cuda::DecodeDtype::Fp32,d.layout,
        1,scales[i].bytes()/4,1,scales[i].bytes()/4,scales[i].bytes(),16);
    d.input=cuda::decode_vector_view(x.data(),cuda::DecodeDtype::Bf16,cuda::kDecodeLayoutBf16VectorV0,k,k*2,2,false);
    output[i]=take(cuda::DeviceBuffer::allocate(d.n*(bf16?2:4)));host[i].resize(output[i].bytes());
    d.output=cuda::decode_vector_view(output[i].data(),bf16?cuda::DecodeDtype::Bf16:cuda::DecodeDtype::Fp32,
        bf16?cuda::kDecodeLayoutBf16VectorV0:cuda::kDecodeLayoutFp32VectorV0,d.n,output[i].bytes(),bf16?2:4,true);
    d.epilogue=bf16?cuda::DecodeEpilogue::StoreBf16:cuda::DecodeEpilogue::StoreFp32;
    if(!front) {
      d.epilogue=cuda::DecodeEpilogue::ResidualAddFp32;
      d.residual=cuda::decode_vector_view(h.data(),cuda::DecodeDtype::Fp32,cuda::kDecodeLayoutFp32VectorV0,5120,5120*4,4,false);
    }
    resident+=codes[i].bytes()+scales[i].bytes();
  }
  auto start=take(cuda::Event::create_timing()),end=take(cuda::Event::create_timing());
  check(stream.sync());
  double setup_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-setup).count();
  auto allocations=cuda::malloc_count();auto begin=std::chrono::steady_clock::now();
  check(start.record(stream));
  if(front) {
    check(cuda::launch_hidden_rms(span<float>(h).data(),span<std::uint16_t>(gamma).data(),1e-6f,1,span<std::uint16_t>(x).data(),stream));
    check(cuda::launch_decode_mmv_ranges({std::span(desc).first(gdn?2:3)},stream));
    if(gdn)check(cuda::launch_decode_ab_bf16({desc[2],desc[3]},stream));
  } else check(cuda::launch_decode_mmv(desc[0],stream));
  for(unsigned i=0;i<names.size();++i)check(cuda::copy_d2h(host[i],output[i].data(),stream));
  check(end.record(stream));check(end.sync());
  double host_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
  std::ofstream saved(saved_path,std::ios::binary);
  for(unsigned i=0;i<names.size();++i) {
    for(unsigned j=0;j<desc[i].n;++j) {
      float value;
      if(desc[i].epilogue==cuda::DecodeEpilogue::StoreBf16)value=format::bf16_to_fp32(format::load_u16_le(host[i].data()+j*2));
      else std::memcpy(&value,host[i].data()+j*4,4);
      if(!std::isfinite(value))return 1;
    }
    saved.write(reinterpret_cast<char const*>(host[i].data()),host[i].size());
  }
  std::cout<<std::setprecision(12)<<"group="<<group<<" calls_per_token="<<(gdn?48:16)
      <<" setup_ms="<<setup_ms<<" gpu_ms="<<take(cuda::elapsed_ms(start,end))<<" host_ms="<<host_ms
      <<" weight_bytes="<<resident<<" workspace_bytes="<<x.bytes()+h.bytes()+gamma.bytes()
      <<" hot_allocations="<<cuda::malloc_count()-allocations<<" first_use=1 warmups=0 repetitions=1 readout_included=1\n";
  return saved && cuda::malloc_count()==allocations?0:1;
}
// One real layer-0 GDN projection group; no warmup or repeated measurement.
// Separate processes load Q8 or FP8, so there is one resident view per weight.
int main(int argc,char** argv) {
  using namespace qw38;
  if(argc==4)return decode_group(argv[1],argv[2],argv[3]);
  if(argc!=7 || (std::string(argv[4])!="q8" && std::string(argv[4])!="fp8")) {
    std::cerr<<"checkpoint artifact prompt.u32le q8|fp8 1|256 output.bin\n";return 2;
  }
  unsigned m=std::string(argv[5])=="1"?1:std::string(argv[5])=="256"?256:0;
  if(!m)return 2;
  bool fp8=std::string(argv[4])!="q8",native=m>1;
    auto checkpoint=take(compiler::open_checkpoint(argv[1]));
  auto artifact=take(format::Artifact::open(argv[2]));auto stream=take(cuda::Stream::create());
  auto engine=take(cuda::PrefillEngine::create(stream,m,512,cuda::PrefillDispatch::BoundedUnpackBf16Cublas,fp8));
  auto source=[&](compiler::TensorFamily family,auto&& use) {
    for(auto const& t:checkpoint.classified.included) if(t.expected.family==family &&
        (family==compiler::TensorFamily::Embed || t.expected.layer_index==0)) {
      auto shard=take(compiler::MappedShard::open(checkpoint.root/t.source.shard));
      use(take(shard.tensor_bytes(t.source)),t);return;
    }
    std::cerr<<"missing source tensor\n";std::exit(1);
  };
  std::vector<std::uint32_t> tokens(m);std::ifstream prompt(argv[3],std::ios::binary);
  if(!prompt.read(reinterpret_cast<char*>(tokens.data()),m*4))return 2;
  std::vector<float> residual(m*5120);
  source(compiler::TensorFamily::Embed,[&](auto b,auto const&) {
    for(unsigned r=0;r<m;++r)for(unsigned c=0;c<5120;++c) {
      auto off=(std::uint64_t(tokens[r])*5120+c)*2;
      if(off+2>b.size())std::exit(2);
      residual[r*5120+c]=format::bf16_to_fp32(format::load_u16_le(b.data()+off));
    }
  });
  auto dh=take(cuda::upload(bytes(residual),stream));
  auto gamma=take(cuda::upload(take(artifact.payload("model.language_model.layers.0.input_layernorm.weight")),stream));
  auto bf=take(cuda::DeviceBuffer::allocate(m*5120*2));
  std::array<cuda::DeviceBuffer,4> codes,scales,output;
  std::array<cuda::PrefillWeight,4> q8;
  std::array<unsigned,4> ns{10240,6144,48,48};
  std::array<std::string,4> suffix{"qkv","z","a","b"};
  std::uint64_t resident=0;
  for(unsigned i=0;i<4;++i) {
    std::string name="model.language_model.layers.0.linear_attn.in_proj_"+suffix[i]+".weight";
    auto const* t=artifact.find_tensor(name);if(!t)return 2;
    if(fp8 && i<2) {
      source(i==0?compiler::TensorFamily::LinearAttnInProjQkv:compiler::TensorFamily::LinearAttnInProjZ,[&](auto b,auto const&) {
        auto packed=take(compiler::quantize_fp8(b,ns[i],5120));
        codes[i]=take(cuda::upload(bytes(packed.codes),stream));scales[i]=take(cuda::upload(bytes(packed.scales),stream));
        check(stream.sync());
        q8[i]={codes[i].data(),scales[i].data(),cuda::kDecodeLayoutFp8V1,cuda::kDecodeQuantizerFp8V1,
            ns[i],5120,ns[i],5120,codes[i].bytes(),scales[i].bytes()};
      });
    } else {
      codes[i]=take(cuda::upload(take(artifact.payload(name)),stream));
      auto s=take(artifact.scales(name));if(!s.empty())scales[i]=take(cuda::upload(s,stream));
      q8[i]={codes[i].data(),scales[i].data(),std::uint16_t(t->layout),std::uint16_t(t->quantizer),ns[i],5120,
          unsigned(t->shape.padded[0]),unsigned(t->shape.padded[1]),codes[i].bytes(),scales[i].bytes()};
    }
    resident+=codes[i].bytes()+scales[i].bytes();
    output[i]=take(cuda::DeviceBuffer::allocate(m*ns[i]*(i<2?2:4)));
  }
  auto packed=fp8 && native ? engine.fp8_operand(m,5120) : cuda::Fp8Input{};
  auto start=take(cuda::Event::create_timing()),end=take(cuda::Event::create_timing());
  check(stream.sync());auto begin=std::chrono::steady_clock::now();check(start.record(stream));
  if(fp8 && native)
    check(cuda::launch_hidden_rms_fp8(span<float>(dh).data(),span<std::uint16_t>(gamma).data(),1e-6f,m,
        span<std::uint16_t>(bf).data(),const_cast<std::uint8_t*>(packed.codes.data()),const_cast<float*>(packed.scales.data()),stream));
  else check(cuda::launch_hidden_rms(span<float>(dh).data(),span<std::uint16_t>(gamma).data(),1e-6f,m,span<std::uint16_t>(bf).data(),stream));
  std::array<cuda::DecodeMmvDesc,4> decode;
  for(unsigned i=0;i<4;++i) {
    if(m==1) {
      auto const& w=q8[i];auto& d=decode[i];
      d.layout=w.layout;d.quantizer=w.quantizer;d.n=w.n;d.k=w.k;d.padded_n=w.padded_n;d.padded_k=w.padded_k;
      d.codes=cuda::decode_matrix_view(const_cast<void*>(w.codes),i<2?(fp8?cuda::DecodeDtype::Fp8:cuda::DecodeDtype::Q8):cuda::DecodeDtype::Bf16,
          w.layout,w.n,w.k,w.padded_n,w.padded_k,w.codes_bytes,16);
      if(w.scales_bytes) {
        unsigned groups=fp8?unsigned(w.scales_bytes/4):w.padded_k/32;
        unsigned rows=fp8?1:w.padded_n;
        d.scales=cuda::decode_matrix_view(const_cast<void*>(w.scales),fp8?cuda::DecodeDtype::Fp32:cuda::DecodeDtype::Fp16,w.layout,
            rows,groups,rows,groups,w.scales_bytes,fp8?16u:2u);
      }
      d.input=cuda::decode_vector_view(bf.data(),cuda::DecodeDtype::Bf16,cuda::kDecodeLayoutBf16VectorV0,5120,5120*2,2,false);
      d.output=cuda::decode_vector_view(output[i].data(),i<2?cuda::DecodeDtype::Bf16:cuda::DecodeDtype::Fp32,
          i<2?cuda::kDecodeLayoutBf16VectorV0:cuda::kDecodeLayoutFp32VectorV0,w.n,output[i].bytes(),i<2?2:4,true);
      d.epilogue=i<2?cuda::DecodeEpilogue::StoreBf16:cuda::DecodeEpilogue::StoreFp32;
    } else check(engine.project({.weight=q8[i],.input=span<std::uint16_t>(bf).data(),.output=output[i].data(),.valid_tokens=m,
        .epilogue=i<2?cuda::PrefillEpilogue::StoreBf16:cuda::PrefillEpilogue::StoreFp32,
        .packed=fp8 && i<2?packed:cuda::Fp8Input{}}));
  }
  if(m==1) {
    check(cuda::launch_decode_mmv_ranges({std::span(decode).first(2)},stream));
    check(cuda::launch_decode_ab_bf16({decode[2],decode[3]},stream));
  }
  check(end.record(stream));check(end.sync());double host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
  std::cout<<"mode="<<argv[4]<<" M="<<m<<" gpu_ms="<<take(cuda::elapsed_ms(start,end))<<" host_ms="<<host
      <<" weight_bytes="<<resident<<" packed_bytes="<<(packed.codes.size_bytes()+packed.scales.size_bytes())
      <<" companion_bytes="<<bf.bytes()<<" engine_workspace="<<engine.workspace_bytes()
      <<" first_use=1 warmups=0 repetitions=1\n";
  std::ofstream saved(argv[6],std::ios::binary);
  for(unsigned i=0;i<4;++i) {
    std::vector<std::byte> v(output[i].bytes());check(cuda::copy_d2h(v,output[i].data(),stream));check(stream.sync());
    saved.write(reinterpret_cast<char const*>(v.data()),v.size());
  }
  return saved?0:1;
}
