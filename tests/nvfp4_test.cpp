#include "compiler/compile.hpp"
#include "compiler/quantization/nvfp4.hpp"
#include "cuda/activation.hpp"
#include "cuda/alloc.hpp"
#include "cuda/copy.hpp"
#include "cuda/nvfp4.hpp"
#include "cuda/prefill.hpp"
#include "cuda/upload.hpp"
#include "format/floatcvt.hpp"
#include "format/nvfp4.hpp"
#include "format/reader.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <unistd.h>

namespace {
using namespace qw38;
void require(bool ok, char const* message) {
  if (!ok) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}
template<class T, class E> T take(std::expected<T,E> result) {
  require(bool(result),"operation succeeded"); return std::move(*result);
}
void check(std::expected<void,qw38::cuda::Error> st) {
  if (!st) std::cerr << qw38::cuda::error_message(st.error()) << '\n';
  require(bool(st),"CUDA operation");
}
template<class T> auto bytes(std::vector<T> const& v) {
  return std::as_bytes(std::span<T const>(v));
}
template<class T> std::vector<T> download(void const* p, std::size_t n, qw38::cuda::Stream const& s) {
  std::vector<T> v(n); check(qw38::cuda::copy_d2h(v.data(),p,n*sizeof(T),s)); check(s.sync()); return v;
}
std::vector<std::uint16_t> input(unsigned m, unsigned k, int seed) {
  std::vector<std::uint16_t> x(std::size_t(m)*k);
  for (std::size_t i=0;i<x.size();++i)
    x[i]=format::fp32_to_bf16_rne(float(int((i*17+seed)%131)-65)/27.f);
  std::fill_n(x.begin(),std::min<std::size_t>(16,x.size()),0);
  return x;
}
// Independent exhaustive nearest-value encoder, including ties to even.
unsigned nearest(float v, bool scale) {
  float error=std::numeric_limits<float>::infinity(); unsigned best=0;
  for (unsigned c=0;c<(scale?127u:8u);++c) {
    float value=scale?format::nvfp4_e4m3(c):format::nvfp4_e2m1(c);
    float e=std::abs(std::abs(v)-value);
    if (e<error || (e==error && !(c&1))) {error=e; best=c;}
  }
  return best | (!scale && std::signbit(v) && v!=0 ? 8u:0u);
}
float decoded(format::PackedMatrix const& p, unsigned r, unsigned k) {
  auto i=std::uint64_t(r)*p.padded_k+k;
  unsigned c=(unsigned(p.codes[i/2])>>(4*(i&1)))&15;
  unsigned s=unsigned(p.scales[256+format::nvfp4_scale_index(r,k/16,p.padded_k)]);
  return format::nvfp4_factor(p.scales)*format::nvfp4_e4m3(s)*format::nvfp4_e2m1(c);
}
void host_checks() {
  auto x=input(5,129,3);
  auto p=take(compiler::quantize_nvfp4(bytes(x),5,129));
  require(bool(format::validate_nvfp4(5,129,p.codes,p.scales)),"host tails/zeros validate");
  using Sf=cutlass::detail::Sm1xxBlockScaledConfig<16>;
  auto layout=Sf::tile_atom_to_shape_SFB(cute::make_shape(3,136,512,1));
  for(unsigned r=0;r<256;++r) for(unsigned b=0;b<32;++b)
    require(std::uint64_t(layout(r,b*16,0))==format::nvfp4_scale_index(r,b,512),"native scale swizzle");
  auto corrupt=p.scales; corrupt[256]=std::byte{127};
  require(!format::validate_nvfp4(5,129,p.codes,corrupt),"NaN scale rejected");
  corrupt=p.scales; corrupt[3]=std::byte{0x7f};
  require(!format::validate_nvfp4(5,129,p.codes,corrupt),"nonfinite factor rejected");
  corrupt=p.scales; corrupt[4]=std::byte{1};
  require(!format::validate_nvfp4(5,129,p.codes,corrupt),"reserved header rejected");
  auto codes=p.codes; codes.back()=std::byte{1};
  require(!format::validate_nvfp4(5,129,codes,p.scales),"nonzero padding rejected");
  x[0]=0x7f80;
  require(!compiler::quantize_nvfp4(bytes(x),5,129),"nonfinite source rejected");
  require(!format::is_known(format::PhysicalLayoutId{0x02ff}),"unknown layout rejected");
  std::vector<compiler::SyntheticTensor> tensors;
  for(auto e:compiler::expand_identity_table()) {
    bool gate=e.family==compiler::TensorFamily::MlpGateProj && e.layer_index==0;
    if(!gate && e.family!=compiler::TensorFamily::Embed && e.family!=compiler::TensorFamily::LmHead) continue;
    e.shape={.rank=2,.dims={8,256}};
    auto source=input(8,256,19); auto span=bytes(source);
    tensors.push_back({e,{span.begin(),span.end()}});
  }
  auto path=std::filesystem::temp_directory_path()/ ("qw38-nvfp4-"+std::to_string(getpid())+".qw38");
  format::CompilerRevision rev{.ident=compiler::kNvFp4CompilerIdent,.major=0,.minor=1,.patch=2};
  auto result=compiler::compile_synthetic(path,{},{},{},rev,tensors,compiler::WeightFormatPolicy::NvFp4MlpV1);
  if(!result) std::cerr << compiler::error_message(result.error()) << '\n';
  require(bool(result),"synthetic compiler path");
  auto artifact=take(format::Artifact::open(path));
  require(artifact.precision().id==format::PrecisionPolicyId::NvFp4MlpV1,"artifact precision identity");
  auto schema=artifact.schema();
  for(auto& tensor:schema.tensors) if(tensor.quantizer==format::LogicalQuantizerId::NvFp4V1) {
    tensor.layout=format::PhysicalLayoutId{0x02ff}; break;
  }
  require(!format::validate_schema(schema),"reader schema rejects unknown physical version");
  std::filesystem::remove(path);
}
void packing(qw38::cuda::Stream const& stream) {
  constexpr unsigned m=3,k=5120;
  auto x=input(m,k,2);
  for(unsigned b=1;b<126;++b) {
    float midpoint=(format::nvfp4_e4m3(b)+format::nvfp4_e4m3(b+1))*3.f;
    std::fill_n(x.begin()+b*16,16,format::fp32_to_bf16_rne(midpoint));
  }
  constexpr float ties[]{.25f,.75f,1.25f,1.75f,2.5f,3.5f,5.f,6.f};
  for(unsigned j=0;j<16;++j) x[128*16+j]=format::fp32_to_bf16_rne((j<8?1.f:-1.f)*ties[j%8]);
  x[k+17]=format::fp32_to_bf16_rne(3000.f);
  auto dx=take(qw38::cuda::upload(bytes(x),stream));
  auto dc=take(qw38::cuda::DeviceBuffer::allocate(m*k/2));
  auto ds=take(qw38::cuda::DeviceBuffer::allocate(format::nvfp4_scale_count(m,k)));
  check(qw38::cuda::launch_pack_nvfp4(static_cast<std::uint16_t const*>(dx.data()),
      static_cast<std::uint8_t*>(dc.data()),static_cast<std::uint8_t*>(ds.data()),m,k,stream));
  auto c=download<std::uint8_t>(dc.data(),m*k/2,stream);
  auto s=download<std::uint8_t>(ds.data(),ds.bytes(),stream);
  for(unsigned r=0;r<m;++r) for(unsigned b=0;b<k/16;++b) {
    float peak=0; for(unsigned i=0;i<16;++i) peak=std::max(peak,std::abs(format::bf16_to_fp32(x[r*k+b*16+i])));
    unsigned sf=nearest(peak/6,true); if(peak && !sf) sf=1;
    require(s[format::nvfp4_scale_index(r,b,k)]==sf,"independent activation scale RNE/saturation");
    for(unsigned i=0;i<16;++i) {
      auto index=r*k+b*16+i;
      float v=format::bf16_to_fp32(x[index]);
      unsigned want=peak==0?0:nearest(v/format::nvfp4_e4m3(sf),false);
      require(((c[index/2]>>(4*(index&1)))&15u)==want,"independent activation code");
    }
  }
  check(qw38::cuda::launch_pack_nvfp4(static_cast<std::uint16_t const*>(dx.data()),
      static_cast<std::uint8_t*>(dc.data()),static_cast<std::uint8_t*>(ds.data()),1,k,stream));
  require(download<std::uint8_t>(dc.data(),k/2,stream)==std::vector<std::uint8_t>(c.begin(),c.begin()+k/2),"same-row chunk independence");
  auto padded=download<std::uint8_t>(ds.data(),ds.bytes(),stream);
  for(unsigned r=1;r<128;++r) for(unsigned b=0;b<k/16;++b)
    require(padded[format::nvfp4_scale_index(r,b,k)]==0,"scale padding");
  // Compare fused producer bytes to the existing RMS kernel plus independent pack.
  std::vector<float> h(m*k); for(unsigned i=0;i<h.size();++i) h[i]=float(int(i%71)-35)*.3f;
  auto gamma=input(1,k,7);
  auto dh=take(qw38::cuda::upload(bytes(h),stream)); auto dg=take(qw38::cuda::upload(bytes(gamma),stream));
  check(qw38::cuda::launch_hidden_rms(static_cast<float const*>(dh.data()),static_cast<std::uint16_t const*>(dg.data()),1e-6f,m,static_cast<std::uint16_t*>(dx.data()),stream));
  check(qw38::cuda::launch_pack_nvfp4(static_cast<std::uint16_t const*>(dx.data()),static_cast<std::uint8_t*>(dc.data()),static_cast<std::uint8_t*>(ds.data()),m,k,stream));
  c=download<std::uint8_t>(dc.data(),dc.bytes(),stream); s=download<std::uint8_t>(ds.data(),ds.bytes(),stream);
  check(qw38::cuda::launch_hidden_rms_nvfp4(static_cast<float const*>(dh.data()),static_cast<std::uint16_t const*>(dg.data()),1e-6f,m,static_cast<std::uint8_t*>(dc.data()),static_cast<std::uint8_t*>(ds.data()),stream));
  require(c==download<std::uint8_t>(dc.data(),dc.bytes(),stream) && s==download<std::uint8_t>(ds.data(),ds.bytes(),stream),"fused RMS preserves exact rounding");
  x[0]=0x7fc0; check(qw38::cuda::copy_h2d(dx.data(),bytes(x),stream));
  check(qw38::cuda::launch_pack_nvfp4(static_cast<std::uint16_t const*>(dx.data()),static_cast<std::uint8_t*>(dc.data()),static_cast<std::uint8_t*>(ds.data()),m,k,stream));
  require(download<std::uint8_t>(ds.data(),ds.bytes(),stream)[0]==127,"nonfinite input is explicit NaN scale");
}
void contractions(std::span<std::byte const> source,unsigned n,unsigned k,
                  qw38::cuda::PrefillEngine& engine,qw38::cuda::Stream const& stream,char const* label) {
  auto p=take(compiler::quantize_nvfp4(source,n,k));
  require(bool(format::validate_nvfp4(n,k,p.codes,p.scales)),"weight payload valid");
  // Check sampled weight codes/scales independently of the CUTLASS encoders.
  float factor=format::nvfp4_factor(p.scales);
  for(unsigned row: {0u,n/2,n-1}) for(unsigned b: {0u,k/32,k/16-1}) {
    float peak=0, v[16];
    for(unsigned j=0;j<16;++j) {
      v[j]=format::bf16_to_fp32(format::load_u16_le(source.data()+(std::uint64_t(row)*k+b*16+j)*2))/factor;
      peak=std::max(peak,std::abs(v[j]));
    }
    unsigned sf=nearest(peak/6,true); if(peak && !sf) sf=1;
    require(unsigned(p.scales[256+format::nvfp4_scale_index(row,b,p.padded_k)])==sf,"independent weight scale");
    for(unsigned j=0;j<16;++j) {
      auto index=std::uint64_t(row)*p.padded_k+b*16+j;
      unsigned code=(unsigned(p.codes[index/2])>>(4*(index&1)))&15;
      require(code==(peak?nearest(v[j]/format::nvfp4_e4m3(sf),false):0),"independent weight code");
    }
  }
  auto dc=take(qw38::cuda::upload(p.codes,stream)); auto ds=take(qw38::cuda::upload(p.scales,stream));
  qw38::cuda::PrefillWeight w{.codes=dc.data(),.scales=ds.data(),.layout=qw38::cuda::kDecodeLayoutNvFp4V1,
      .quantizer=qw38::cuda::kDecodeQuantizerNvFp4V1,.n=n,.k=k,.padded_n=unsigned(p.padded_n),
      .padded_k=unsigned(p.padded_k),.codes_bytes=p.codes.size(),.scales_bytes=p.scales.size()};
  float previous=0;
  for(unsigned m: {1u,2u,3u,129u}) {
    if(n>1024 && m==129) continue;
    auto x=input(m,k,13); auto dx=take(qw38::cuda::upload(bytes(x),stream));
    std::vector<float> init(std::size_t(m+1)*n,123.f); auto dy=take(qw38::cuda::upload(bytes(init),stream));
    auto before=qw38::cuda::malloc_count();
    check(engine.project({.weight=w,.input=static_cast<std::uint16_t const*>(dx.data()),
      .output=dy.data(),.valid_tokens=m,.first_position=19,.epilogue=qw38::cuda::PrefillEpilogue::StoreFp32}));
    require(before==qw38::cuda::malloc_count(),"no projection allocations");
    auto got=download<float>(dy.data(),init.size(),stream);
    float max_error=0;
    for(unsigned r: {0u,m-1}) for(unsigned row: {0u,1u,n/2,n-1}) {
      float expected=0;
      for(unsigned b=0;b<k/16;++b) {
        float peak=0; for(unsigned j=0;j<16;++j) peak=std::max(peak,std::abs(format::bf16_to_fp32(x[r*k+b*16+j])));
        unsigned sf=nearest(peak/6,true); if(peak && !sf) sf=1;
        for(unsigned j=0;j<16;++j) {
          unsigned col=b*16+j; float a=format::bf16_to_fp32(x[r*k+col]);
          if(m>=2) a=peak?format::nvfp4_e4m3(sf)*format::nvfp4_e2m1(nearest(a/format::nvfp4_e4m3(sf),false)):0;
          expected=std::fma(a,decoded(p,row,col),expected);
        }
      }
      float actual=got[r*n+row], error=std::abs(actual-expected); max_error=std::max(max_error,error);
      require(std::isfinite(actual) && error<=1e-3f+1e-4f*std::abs(expected),"independent FP32 contraction/orientation");
    }
    for(unsigned i=m*n;i<(m+1)*n;++i) require(got[i]==123.f,"logical output guard");
    std::cout<<label<<" M="<<m<<" max_abs="<<max_error<<" first_output="<<got[0];
    if(m==2) std::cout<<" W4A4_minus_GEMV="<<got[0]-previous;
    std::cout<<'\n'; previous=got[0];
    auto bad=w; bad.layout=0x02ff;
    require(!engine.project({.weight=bad,.input=static_cast<std::uint16_t const*>(dx.data()),.output=dy.data(),.valid_tokens=m}),"bad layout launch rejected");
  }
}
void paired(qw38::cuda::PrefillEngine& engine, qw38::cuda::Stream const& stream) {
  constexpr unsigned n=136,k=5120;
  auto a=input(n,k,4), b=input(n,k,29);
  auto pa=take(compiler::quantize_nvfp4(bytes(a),n,k));
  auto pb=take(compiler::quantize_nvfp4(bytes(b),n,k));
  auto ac=take(qw38::cuda::upload(pa.codes,stream)), as=take(qw38::cuda::upload(pa.scales,stream));
  auto bc=take(qw38::cuda::upload(pb.codes,stream)), bs=take(qw38::cuda::upload(pb.scales,stream));
  qw38::cuda::PrefillWeight wa{.codes=ac.data(),.scales=as.data(),.layout=qw38::cuda::kDecodeLayoutNvFp4V1,
      .quantizer=qw38::cuda::kDecodeQuantizerNvFp4V1,.n=n,.k=k,.padded_n=n,.padded_k=k,
      .codes_bytes=pa.codes.size(),.scales_bytes=pa.scales.size()};
  auto wb=wa; wb.codes=bc.data(); wb.scales=bs.data();
  for(unsigned m: {1u,2u,3u}) {
    auto x=input(m,k,13); auto dx=take(qw38::cuda::upload(bytes(x),stream));
    auto ga=take(qw38::cuda::DeviceBuffer::allocate(m*n*4));
    auto up=take(qw38::cuda::DeviceBuffer::allocate(m*n*4));
    auto y=take(qw38::cuda::DeviceBuffer::allocate(m*n*2));
    auto* input=static_cast<std::uint16_t const*>(dx.data());
    if (m == 1) {
      auto descriptor = [&](qw38::cuda::PrefillWeight const& w, void* output) {
        qw38::cuda::DecodeMmvDesc d;
        d.layout = w.layout; d.quantizer = w.quantizer;
        d.n = d.padded_n = n; d.k = d.padded_k = k;
        d.codes = qw38::cuda::decode_matrix_view(const_cast<void*>(w.codes),
            qw38::cuda::DecodeDtype::NvFp4, w.layout, n, k, n, k, w.codes_bytes, 16);
        auto units = static_cast<std::uint32_t>(w.scales_bytes / 2);
        d.scales = qw38::cuda::decode_matrix_view(const_cast<void*>(w.scales),
            qw38::cuda::DecodeDtype::Fp16, w.layout, 1, units, 1, units, w.scales_bytes, 16);
        d.input = qw38::cuda::decode_vector_view(dx.data(), qw38::cuda::DecodeDtype::Bf16,
            qw38::cuda::kDecodeLayoutBf16VectorV0, k, k * 2, 2, false);
        d.output = qw38::cuda::decode_vector_view(output, qw38::cuda::DecodeDtype::Bf16,
            qw38::cuda::kDecodeLayoutBf16VectorV0, n, n * 2, 2, true);
        return d;
      };
      std::array ranges{descriptor(wa, ga.data()), descriptor(wb, up.data())};
      // Both descriptors independently pass the public single-launch boundary.
      for (auto const& d : ranges) check(qw38::cuda::launch_decode_mmv(d, stream));
      check(stream.sync());
      std::vector<std::uint16_t> sentinel(n, 0x3f80);
      for (auto const& d : ranges)
        check(qw38::cuda::copy_h2d(d.output.pointer, bytes(sentinel), stream));
      auto rejected = qw38::cuda::launch_decode_mmv_ranges({ranges}, stream);
      require(!rejected && rejected.error().code == qw38::cuda::ErrorCode::InvalidArgument,
              "NVFP4 ranged launch rejects before BF16 fallback");
      check(stream.sync());
      for (auto const& d : ranges)
        require(download<std::uint16_t>(d.output.pointer, n, stream) == sentinel,
                "rejected ranged launch leaves output unchanged");
    }
    check(engine.project({.weight=wa,.input=input,.output=ga.data(),.valid_tokens=m,.epilogue=qw38::cuda::PrefillEpilogue::StoreFp32}));
    check(engine.project({.weight=wb,.input=input,.output=up.data(),.valid_tokens=m,.epilogue=qw38::cuda::PrefillEpilogue::StoreFp32}));
    auto before=qw38::cuda::malloc_count();
    check(engine.paired_swiglu(wa,wb,input,static_cast<std::uint16_t*>(y.data()),m,17));
    require(before==qw38::cuda::malloc_count(),"paired input/workspace reuse");
    auto g=download<float>(ga.data(),m*n,stream), u=download<float>(up.data(),m*n,stream);
    auto got=download<std::uint16_t>(y.data(),m*n,stream);
    for(unsigned i=0;i<m*n;++i) {
      float sigmoid=g[i]>=0 ? 1.f/(1.f+std::exp(-g[i])) : std::exp(g[i])/(1.f+std::exp(g[i]));
      float expected=format::bf16_to_fp32(format::fp32_to_bf16_rne((g[i]*sigmoid)*u[i]));
      float actual=format::bf16_to_fp32(got[i]);
      require(std::isfinite(actual) && std::abs(actual-expected)<=.008f*std::abs(expected)+.001f,"paired SwiGLU and BF16 rounding");
    }
  }
}
}
int main() {
  host_checks(); auto stream=take(qw38::cuda::Stream::create()); packing(stream);
  auto engine=take(qw38::cuda::PrefillEngine::create(stream,129,512));
  paired(engine,stream);
  auto x=input(136,5120,27); contractions(bytes(x),136,5120,engine,stream,"synthetic");
  if(auto root=std::getenv("QW38_AUTHORITY_CHECKPOINT")) {
    auto checkpoint=take(compiler::open_checkpoint(root)); unsigned checked=0;
    for(auto const& item:checkpoint.classified.included) {
      if((item.expected.layer_index!=0 && item.expected.layer_index!=63) ||
          (item.expected.family!=compiler::TensorFamily::MlpGateProj &&
           item.expected.family!=compiler::TensorFamily::MlpUpProj)) continue;
      auto shard=take(compiler::MappedShard::open(checkpoint.shards.at(item.source.shard).path));
      auto source=take(shard.tensor_bytes(item.source));
      contractions(source,17408,5120,engine,stream,item.expected.name.c_str()); ++checked;
    }
    require(checked==4,"early/late real gate/up coverage");
  }
  std::cout<<"NVFP4 checks PASS\n";
}
