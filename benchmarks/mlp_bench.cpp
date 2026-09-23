#include "cuda/buffer.hpp"
#include "cuda/decode_mmv.hpp"
#include "cuda/error.hpp"
#include "cuda/event.hpp"
#include "cuda/stream.hpp"
#include "format/constants.hpp"
#include "runtime/mlp.hpp"
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

int main() {
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
