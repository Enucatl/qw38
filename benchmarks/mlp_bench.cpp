#include "cuda/buffer.hpp"
#include "cuda/decode_mmv.hpp"
#include "cuda/error.hpp"
#include "cuda/event.hpp"
#include "cuda/stream.hpp"
#include "format/constants.hpp"
#include "runtime/mlp.hpp"
#include "runtime/prefill.hpp"
#include "runtime/runtime.hpp"
#include "cuda/copy.hpp"
#include "cuda/q4k_q8.hpp"
#include "cuda/alloc.hpp"
#include "cuda/activation.hpp"
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <fstream>
#include "runtime/view.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

using qw38::cuda::DeviceBuffer;
using qw38::cuda::Event;
using qw38::cuda::Stream;
using qw38::cuda::elapsed_ms;
using qw38::cuda::kDecodeThreads;
using qw38::cuda::kDecodeTileK;
using qw38::cuda::kDecodeWarps;
using qw38::cuda::zero;
using qw38::format::ArithmeticDtype;
using qw38::format::PhysicalLayoutId;
using qw38::format::StorageClass;
using qw38::runtime::MlpBindViews;
using qw38::runtime::MemorySpace;
using qw38::runtime::TensorView;
using qw38::runtime::bind_mlp_plan;
using qw38::runtime::execute_decode_mlp;
using qw38::runtime::kFfnWidth;
using qw38::runtime::kHidden;

namespace {

TensorView make_view(void* ptr, ArithmeticDtype dtype, PhysicalLayoutId layout,
                     StorageClass storage, bool writable, std::uint8_t rank,
                     std::uint64_t e0, std::uint64_t e1 = 0) {
  TensorView v{};
  v.pointer = ptr;
  v.dtype = dtype;
  v.layout = layout;
  v.storage = storage;
  v.space = MemorySpace::Device;
  (void)writable;  // Mutability is represented by TensorView/ConstTensorView.
  v.rank = rank;
  v.extent[0] = e0;
  v.extent[1] = e1;
  return v;
}

}  // namespace

// One complete real-shape operation per process, identical nonzero residuals
// and unchanged layer-0 Q4_K weights in the 1029/1030 arms. Includes first use.
int real_mlp(char const* artifact, char const* rows, char const* saved_path=nullptr) {
  using namespace qw38;
  unsigned m=0;std::string_view text(rows);
  auto parsed=std::from_chars(text.data(),text.data()+text.size(),m);
  if(parsed.ec!=std::errc{} || parsed.ptr!=text.end() || (m!=1 && m!=256))return 2;
  auto setup_started=std::chrono::steady_clock::now();
  auto rt=runtime::Runtime::create();if(!rt)return 1;
  auto source=format::Artifact::open(artifact);if(!source)return 1;
  auto model=rt->upload_diagnostic(*source,runtime::DiagnosticWeights::Layer,0);if(!model)return 1;
  auto session=rt->create_session(*model,1);if(!session)return 1;
  auto const& stream=rt->stream();
  auto decode=bind_mlp_plan(*model,*session,0,stream);
  auto prefill=runtime::bind_prefill_layer_projections(*model,0,stream,&*session);
  auto engine=cuda::PrefillEngine::create(stream,256);
  if(!decode || !prefill || !engine)return 1;
  std::vector<float> h(m*5120);
  for(unsigned t=0;t<m;++t)for(unsigned c=0;c<5120;++c)
    h[t*5120+c]=(int((t*23+c*17)%97)-48)*.03125f;
  if(!cuda::copy_h2d(session->residual_h_mid().pointer,std::as_bytes(std::span(h)),stream) || !stream.sync())return 1;
  auto begin=Event::create_timing(),end=Event::create_timing();if(!begin || !end)return 1;
  double setup_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-setup_started).count();
  auto count=cuda::malloc_count();
  std::vector<float> out(m*5120);
  if(!begin->record(stream))return 1;
  auto started=std::chrono::steady_clock::now();
  auto result=m==1?execute_decode_mlp(*decode):runtime::execute_prefill_mlp(*prefill,*engine,
      static_cast<float const*>(session->residual_h_mid().pointer),static_cast<float*>(session->residual_h().pointer),m,0);
  if(!result){std::cerr<<runtime::error_message(result.error())<<'\n';return 1;}
  if(!cuda::copy_d2h(out.data(),session->residual_h().pointer,out.size()*4,stream) ||
      !end->record(stream) || !stream.sync())return 1;
  double host_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
  auto gpu_ms=elapsed_ms(*begin,*end);if(!gpu_ms)return 1;
  auto allocations=cuda::malloc_count()-count;
  for(float v:out)if(!std::isfinite(v))return 1;
  if(saved_path) {
    std::ofstream saved(saved_path,std::ios::binary);
    saved.write(reinterpret_cast<char const*>(out.data()),out.size()*4);
    if(!saved)return 1;
  }
  std::cout<<std::setprecision(12)<<"policy="<<unsigned(model->schema().precision.id)<<" m="<<m
      <<" setup_ms="<<setup_ms
      <<" gpu_ms="<<*gpu_ms<<" host_ms="<<host_ms<<" q8_workspace_bytes="<<session->q8_mlp_workspace().size()
      <<" prefill_workspace_bytes="<<engine->workspace_bytes()<<" hot_allocations="<<allocations
      <<" runs=1 warmups=0 includes_library_first_use=true readout_included=true finite=true\n";
  return allocations?1:0;
}

// Isolate the complete producer, including its output/pack and completion.
int rms_bench(char const* mode, char const* rows, char const* saved_path) {
  using namespace qw38;
  unsigned m = 0;
  std::string_view text(rows), kind(mode);
  auto parsed = std::from_chars(text.data(), text.data() + text.size(), m);
  if (parsed.ec != std::errc{} || parsed.ptr != text.end() ||
      (m != 1 && m != 512) || (kind != "plain" && kind != "q8")) return 2;
  auto const setup_start = std::chrono::steady_clock::now();
  auto stream = Stream::create();
  if (!stream) return 1;
  std::vector<float> input(m * kHidden);
  std::vector<std::uint16_t> gamma(kHidden);
  for (unsigned i = 0; i < input.size(); ++i)
    input[i] = (int((i * 17 + i / kHidden * 23) % 97) - 48) * .03125f;
  for (unsigned i = 0; i < gamma.size(); ++i) gamma[i] = 0x3d00u + i % 127u;
  auto x = DeviceBuffer::allocate(input.size() * 4);
  auto g = DeviceBuffer::allocate(gamma.size() * 2);
  auto y = DeviceBuffer::allocate(input.size() * (kind == "plain" ? 2 : 1));
  auto scales = DeviceBuffer::allocate(input.size() / 32 * 4);
  auto sums = DeviceBuffer::allocate(input.size() / 32 * 4);
  auto failure = DeviceBuffer::allocate(4);
  if (!x || !g || !y || !scales || !sums || !failure ||
      !cuda::copy_h2d(x->data(), std::as_bytes(std::span(input)), *stream) ||
      !cuda::copy_h2d(g->data(), std::as_bytes(std::span(gamma)), *stream) ||
      !zero(*failure, *stream) || !stream->sync()) return 1;
  std::vector<std::byte> result(y->bytes() + (kind == "q8" ? scales->bytes() + sums->bytes() : 0));
  auto begin = Event::create_timing(), end = Event::create_timing();
  if (!begin || !end) return 1;
  double const setup_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - setup_start).count();
  auto const count = cuda::malloc_count();
  int failed = 0;
  if (!begin->record(*stream)) return 1;
  auto const start = std::chrono::steady_clock::now();
  auto status = kind == "plain"
      ? cuda::launch_hidden_rms(static_cast<float const*>(x->data()),
          static_cast<std::uint16_t const*>(g->data()), 1e-6f, m,
          static_cast<std::uint16_t*>(y->data()), *stream)
      : cuda::launch_hidden_rms_q8(static_cast<float const*>(x->data()),
          static_cast<std::uint16_t const*>(g->data()), 1e-6f, m,
          static_cast<std::int8_t*>(y->data()), static_cast<float*>(scales->data()),
          static_cast<std::int32_t*>(sums->data()), static_cast<int*>(failure->data()), *stream);
  if (!status || !cuda::copy_d2h(result.data(), y->data(), y->bytes(), *stream)) return 1;
  if (kind == "q8" &&
      (!cuda::copy_d2h(result.data() + y->bytes(), scales->data(), scales->bytes(), *stream) ||
       !cuda::copy_d2h(result.data() + y->bytes() + scales->bytes(), sums->data(), sums->bytes(), *stream) ||
       !cuda::copy_d2h(&failed, failure->data(), 4, *stream))) return 1;
  if (!end->record(*stream) || !stream->sync() || failed) return 1;
  double const host_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  auto gpu_ms = elapsed_ms(*begin, *end);
  if (!gpu_ms) return 1;
  std::ofstream saved(saved_path, std::ios::binary);
  saved.write(reinterpret_cast<char const*>(result.data()), result.size());
  if (!saved) return 1;
  auto const hot = cuda::malloc_count() - count;
  std::cout << std::setprecision(12) << "rms=" << kind << " m=" << m
            << " setup_ms=" << setup_ms << " gpu_ms=" << *gpu_ms
            << " host_ms=" << host_ms << " hot_allocations=" << hot
            << " runs=1 warmups=0 first_use=true readout_included=true\n";
  return hot ? 1 : 0;
}

int main(int argc,char** argv) {
  if (argc == 5 && std::string_view(argv[1]) == "--rms")
    return rms_bench(argv[2], argv[3], argv[4]);
  if(argc==3 || argc==4)return real_mlp(argv[1],argv[2],argc==4?argv[3]:nullptr);
  if(argc!=1)return 2;
  cudaDeviceProp prop{};
  if (auto st = qw38::cuda::check(cudaGetDeviceProperties(&prop, 0),
                                  "cudaGetDeviceProperties");
      !st) {
    std::cerr << qw38::cuda::error_message(st.error()) << '\n';
    return 1;
  }
  auto stream = Stream::create();
  if (!stream) {
    std::cerr << qw38::cuda::error_message(stream.error()) << '\n';
    return 1;
  }

  auto const gate_codes_n = qw38::cuda::decode_code_bytes(
      qw38::cuda::kDecodeLayoutQ4G64V0, kFfnWidth, kHidden);
  auto const gate_scale_n = qw38::cuda::decode_scale_bytes(
      qw38::cuda::kDecodeLayoutQ4G64V0, kFfnWidth, kHidden);
  auto const down_codes_n = qw38::cuda::decode_code_bytes(
      qw38::cuda::kDecodeLayoutQ4G64V0, kHidden, kFfnWidth);
  auto const down_scale_n = qw38::cuda::decode_scale_bytes(
      qw38::cuda::kDecodeLayoutQ4G64V0, kHidden, kFfnWidth);

  auto gate_c = DeviceBuffer::allocate(gate_codes_n);
  auto gate_s = DeviceBuffer::allocate(gate_scale_n);
  auto up_c = DeviceBuffer::allocate(gate_codes_n);
  auto up_s = DeviceBuffer::allocate(gate_scale_n);
  auto down_c = DeviceBuffer::allocate(down_codes_n);
  auto down_s = DeviceBuffer::allocate(down_scale_n);
  auto gamma = DeviceBuffer::allocate(kHidden * 2);
  auto h_mid = DeviceBuffer::allocate(kHidden * 4);
  auto next_h = DeviceBuffer::allocate(kHidden * 4);
  auto normalized = DeviceBuffer::allocate(kHidden * 2);
  auto swiglu = DeviceBuffer::allocate(kFfnWidth * 2);
  if (!gate_c || !gate_s || !up_c || !up_s || !down_c || !down_s || !gamma ||
      !h_mid || !next_h || !normalized || !swiglu) {
    std::cerr << "allocate failed\n";
    return 1;
  }
  if (!zero(*gate_c, *stream) || !zero(*gate_s, *stream) || !zero(*up_c, *stream) ||
      !zero(*up_s, *stream) || !zero(*down_c, *stream) || !zero(*down_s, *stream) ||
      !zero(*gamma, *stream) || !zero(*h_mid, *stream) ||
      !zero(*next_h, *stream) ||
      !zero(*normalized, *stream) || !zero(*swiglu, *stream)) {
    std::cerr << "zero failed\n";
    return 1;
  }

  MlpBindViews views;
  views.gate = make_view(gate_c->data(), ArithmeticDtype::Bf16,
                         PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped,
                         false, 2, kFfnWidth, kHidden);
  views.up = make_view(up_c->data(), ArithmeticDtype::Bf16,
                       PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped,
                       false, 2, kFfnWidth, kHidden);
  views.down = make_view(down_c->data(), ArithmeticDtype::Bf16,
                         PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped,
                         false, 2, kHidden, kFfnWidth);
  views.gate_scales = make_view(gate_s->data(), ArithmeticDtype::Fp16,
                                PhysicalLayoutId::CudaQ4G64V0,
                                StorageClass::Int4Grouped, false, 1,
                                gate_scale_n / 2);
  views.up_scales = make_view(up_s->data(), ArithmeticDtype::Fp16,
                              PhysicalLayoutId::CudaQ4G64V0,
                              StorageClass::Int4Grouped, false, 1,
                              gate_scale_n / 2);
  views.down_scales = make_view(down_s->data(), ArithmeticDtype::Fp16,
                                PhysicalLayoutId::CudaQ4G64V0,
                                StorageClass::Int4Grouped, false, 1,
                                down_scale_n / 2);
  views.gamma = make_view(gamma->data(), ArithmeticDtype::Bf16,
                          PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16,
                          false, 1, kHidden);
  views.h_mid = make_view(h_mid->data(), ArithmeticDtype::Fp32,
                          PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32,
                          true, 1, kHidden);
  views.next_h = make_view(next_h->data(), ArithmeticDtype::Fp32,
                           PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32,
                           true, 1, kHidden);
  views.normalized = make_view(normalized->data(), ArithmeticDtype::Bf16,
                               PhysicalLayoutId::CudaBf16RowMajorV0,
                               StorageClass::Bf16, true, 1, kHidden);
  views.swiglu = make_view(swiglu->data(), ArithmeticDtype::Bf16,
                           PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16,
                           true, 1, kFfnWidth);

  auto plan = bind_mlp_plan(views, *stream);
  if (!plan) {
    std::cerr << qw38::runtime::error_message(plan.error()) << '\n';
    return 1;
  }

  int const warmup = 2;
  int const launches = 8;
  for (int i = 0; i < warmup; ++i) {
    auto st = execute_decode_mlp(*plan);
    if (!st) {
      std::cerr << qw38::runtime::error_message(st.error()) << '\n';
      return 1;
    }
  }
  auto t0 = Event::create_timing();
  auto t1 = Event::create_timing();
  if (!t0 || !t1) {
    return 1;
  }
  auto r0 = t0->record(*stream);
  if (!r0) {
    return 1;
  }
  for (int i = 0; i < launches; ++i) {
    auto st = execute_decode_mlp(*plan);
    if (!st) {
      std::cerr << qw38::runtime::error_message(st.error()) << '\n';
      return 1;
    }
  }
  auto r1 = t1->record(*stream);
  if (!r1) {
    return 1;
  }
  auto s1 = t1->sync();
  if (!s1) {
    return 1;
  }
  auto ms = elapsed_ms(*t0, *t1);
  if (!ms) {
    std::cerr << qw38::cuda::error_message(ms.error()) << '\n';
    return 1;
  }
  float const avg = *ms / static_cast<float>(launches);
  std::uint64_t const weight_bytes =
      2 * (gate_codes_n + gate_scale_n) + down_codes_n + down_scale_n;
  std::cout << "qw38_bench_mlp\n";
  std::cout << "device=" << prop.name << " sm_" << prop.major << prop.minor << '\n';
  std::cout << "geometry warps/block=" << kDecodeWarps
            << " threads=" << kDecodeThreads << " k_tile=" << kDecodeTileK
            << " max_k=" << qw38::cuda::kDecodeMaxK << '\n';
  std::cout << "decode-mlp q4 hidden=" << kHidden << " ffn=" << kFfnWidth
            << " regions=3 (rms, paired-swiglu, down-residual)"
            << " weight_bytes=" << weight_bytes << " launches=" << launches
            << " ms=" << avg << '\n';
  return 0;
}
