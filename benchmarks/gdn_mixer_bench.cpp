#include "cuda/buffer.hpp"
#include "cuda/decode_mmv.hpp"
#include "cuda/error.hpp"
#include "cuda/event.hpp"
#include "cuda/stream.hpp"
#include "format/constants.hpp"
#include "runtime/gdn.hpp"
#include "runtime/view.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

using qw38::cuda::DeviceBuffer;
using qw38::cuda::Event;
using qw38::cuda::Stream;
using qw38::cuda::elapsed_ms;
using qw38::cuda::zero;
using qw38::format::ArithmeticDtype;
using qw38::format::PhysicalLayoutId;
using qw38::format::StorageClass;
using qw38::runtime::GdnBindViews;
using qw38::runtime::GdnRegionTimings;
using qw38::runtime::MemorySpace;
using qw38::runtime::TensorView;
using qw38::runtime::bind_gdn_plan;
using qw38::runtime::execute_decode_gdn;
using qw38::runtime::execute_decode_gdn_timed;
using qw38::runtime::kConvChannels;
using qw38::runtime::kConvKernel;
using qw38::runtime::kConvTaps;
using qw38::runtime::kGdnMixerRegionNames;
using qw38::runtime::kGdnMixerRegions;
using qw38::runtime::kGdnSElemsPerLayer;
using qw38::runtime::kGdnValueDim;
using qw38::runtime::kGdnValueHeads;
using qw38::runtime::kGdnWorkspaceBytesPerToken;
using qw38::runtime::kGdnZWidth;
using qw38::runtime::kHidden;

namespace {

TensorView make_view(void* ptr, ArithmeticDtype dtype, PhysicalLayoutId layout,
                     StorageClass storage, bool /*writable*/, std::uint8_t rank,
                     std::uint64_t e0, std::uint64_t e1 = 0) {
  TensorView v{};
  v.pointer = ptr;
  v.dtype = dtype;
  v.layout = layout;
  v.storage = storage;
  v.space = MemorySpace::Device;
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

  auto const qkv_codes = qw38::cuda::decode_code_bytes(
      qw38::cuda::kDecodeLayoutQ4G64V0, kConvChannels, kHidden);
  auto const qkv_scales = qw38::cuda::decode_scale_bytes(
      qw38::cuda::kDecodeLayoutQ4G64V0, kConvChannels, kHidden);
  auto const z_codes = qw38::cuda::decode_code_bytes(
      qw38::cuda::kDecodeLayoutQ4G64V0, kGdnZWidth, kHidden);
  auto const z_scales = qw38::cuda::decode_scale_bytes(
      qw38::cuda::kDecodeLayoutQ4G64V0, kGdnZWidth, kHidden);
  auto const out_codes = qw38::cuda::decode_code_bytes(
      qw38::cuda::kDecodeLayoutQ4G64V0, kHidden, kGdnZWidth);
  auto const out_scales = qw38::cuda::decode_scale_bytes(
      qw38::cuda::kDecodeLayoutQ4G64V0, kHidden, kGdnZWidth);
  auto const ab_codes = qw38::cuda::decode_code_bytes(
      qw38::cuda::kDecodeLayoutBf16DenseTileV0, kGdnValueHeads, kHidden);

  auto qkv_c = DeviceBuffer::allocate(qkv_codes);
  auto qkv_s = DeviceBuffer::allocate(qkv_scales);
  auto z_c = DeviceBuffer::allocate(z_codes);
  auto z_s = DeviceBuffer::allocate(z_scales);
  auto out_c = DeviceBuffer::allocate(out_codes);
  auto out_s = DeviceBuffer::allocate(out_scales);
  auto a = DeviceBuffer::allocate(ab_codes);
  auto b = DeviceBuffer::allocate(ab_codes);
  auto gamma = DeviceBuffer::allocate(kHidden * 2);
  auto gated = DeviceBuffer::allocate(kGdnValueDim * 2);
  auto taps = DeviceBuffer::allocate(static_cast<std::uint64_t>(kConvKernel) *
                                     kConvChannels * 2u);
  auto alog = DeviceBuffer::allocate(kGdnValueHeads * 2);
  auto dt = DeviceBuffer::allocate(kGdnValueHeads * 2);
  auto residual = DeviceBuffer::allocate(kHidden * 4);
  auto residual_out = DeviceBuffer::allocate(kHidden * 4);
  auto normalized = DeviceBuffer::allocate(kHidden * 2);
  auto workspace = DeviceBuffer::allocate(kGdnWorkspaceBytesPerToken);
  auto history = DeviceBuffer::allocate(static_cast<std::uint64_t>(kConvTaps) *
                                        kConvChannels * 2u);
  auto s = DeviceBuffer::allocate(kGdnSElemsPerLayer * 4);
  if (!qkv_c || !qkv_s || !z_c || !z_s || !out_c || !out_s || !a || !b || !gamma ||
      !gated || !taps || !alog || !dt || !residual || !residual_out ||
      !normalized || !workspace || !history || !s) {
    std::cerr << "allocate failed\n";
    return 1;
  }
  auto zero_all = [&](DeviceBuffer& buf) {
    return static_cast<bool>(zero(buf, *stream));
  };
  if (!zero_all(*qkv_c) || !zero_all(*qkv_s) || !zero_all(*z_c) || !zero_all(*z_s) ||
      !zero_all(*out_c) || !zero_all(*out_s) || !zero_all(*a) || !zero_all(*b) ||
      !zero_all(*gamma) || !zero_all(*gated) || !zero_all(*taps) ||
      !zero_all(*alog) || !zero_all(*dt) || !zero_all(*residual) ||
      !zero_all(*residual_out) || !zero_all(*normalized) || !zero_all(*workspace) ||
      !zero_all(*history) || !zero_all(*s)) {
    std::cerr << "zero failed\n";
    return 1;
  }

  std::uint32_t cursor = 0;
  std::uint64_t position = 0;
  GdnBindViews views;
  views.qkv = make_view(qkv_c->data(), ArithmeticDtype::Bf16,
                        PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped,
                        false, 2, kConvChannels, kHidden);
  views.z = make_view(z_c->data(), ArithmeticDtype::Bf16,
                      PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped, false,
                      2, kGdnZWidth, kHidden);
  views.out = make_view(out_c->data(), ArithmeticDtype::Bf16,
                        PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped,
                        false, 2, kHidden, kGdnZWidth);
  views.qkv_scales = make_view(qkv_s->data(), ArithmeticDtype::Fp16,
                               PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped,
                               false, 1, qkv_scales / 2);
  views.z_scales = make_view(z_s->data(), ArithmeticDtype::Fp16,
                             PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped,
                             false, 1, z_scales / 2);
  views.out_scales = make_view(out_s->data(), ArithmeticDtype::Fp16,
                               PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped,
                               false, 1, out_scales / 2);
  views.a = make_view(a->data(), ArithmeticDtype::Bf16,
                      PhysicalLayoutId::CudaBf16DenseTileV0, StorageClass::Bf16, false,
                      2, kGdnValueHeads, kHidden);
  views.b = make_view(b->data(), ArithmeticDtype::Bf16,
                      PhysicalLayoutId::CudaBf16DenseTileV0, StorageClass::Bf16, false,
                      2, kGdnValueHeads, kHidden);
  views.gamma = make_view(gamma->data(), ArithmeticDtype::Bf16,
                          PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16,
                          false, 1, kHidden);
  views.gated_gamma = make_view(gated->data(), ArithmeticDtype::Bf16,
                                PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16,
                                false, 1, kGdnValueDim);
  views.taps = make_view(taps->data(), ArithmeticDtype::Bf16,
                         PhysicalLayoutId::CudaBf16TapMajorV0, StorageClass::Bf16,
                         false, 2, kConvKernel, kConvChannels);
  views.a_log = make_view(alog->data(), ArithmeticDtype::Bf16,
                          PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16,
                          false, 1, kGdnValueHeads);
  views.dt_bias = make_view(dt->data(), ArithmeticDtype::Bf16,
                            PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16,
                            false, 1, kGdnValueHeads);
  views.residual = make_view(residual->data(), ArithmeticDtype::Fp32,
                             PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32,
                             true, 1, kHidden);
  views.residual_out =
      make_view(residual_out->data(), ArithmeticDtype::Fp32,
                PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true, 1,
                kHidden);
  views.normalized = make_view(normalized->data(), ArithmeticDtype::Bf16,
                               PhysicalLayoutId::CudaBf16RowMajorV0,
                               StorageClass::Bf16, true, 1, kHidden);
  views.workspace = make_view(workspace->data(), ArithmeticDtype::Fp32,
                              PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32,
                              true, 1, kGdnWorkspaceBytesPerToken / 4);
  views.history = make_view(history->data(), ArithmeticDtype::Bf16,
                            PhysicalLayoutId::CudaBf16ConvHistoryV0,
                            StorageClass::Bf16, true, 2, kConvTaps,
                            kConvChannels);
  views.s = make_view(s->data(), ArithmeticDtype::Fp32,
                      PhysicalLayoutId::CudaFp32GdnSHvKV0, StorageClass::Fp32, true,
                      1, kGdnSElemsPerLayer);
  auto cursor_slot = qw38::runtime::ConvCursorSlot::bind(&cursor);
  if (!cursor_slot) {
    std::cerr << qw38::runtime::error_message(cursor_slot.error()) << '\n';
    return 1;
  }
  views.cursor = *cursor_slot;
  auto position_slot = qw38::runtime::GdnPositionSlot::bind(&position);
  if (!position_slot) {
    std::cerr << qw38::runtime::error_message(position_slot.error()) << '\n';
    return 1;
  }
  views.position = *position_slot;
  views.language_layer = 0;

  auto plan = bind_gdn_plan(views, *stream);
  if (!plan) {
    std::cerr << qw38::runtime::error_message(plan.error()) << '\n';
    return 1;
  }

  int const warmup = 2;
  int const iterations = 8;
  int const kernel_launches_per_iteration = 9;
  int const device_copies_per_iteration = 1;
  for (int i = 0; i < warmup; ++i) {
    cursor = 0;
    auto st = execute_decode_gdn(*plan, position);
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
  cursor = 0;
  auto r0 = t0->record(*stream);
  if (!r0) {
    return 1;
  }
  for (int i = 0; i < iterations; ++i) {
    auto st = execute_decode_gdn(*plan, position);
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
  float const avg = *ms / static_cast<float>(iterations);

  cursor = 0;
  GdnRegionTimings regions{};
  GdnRegionTimings acc{};
  for (int i = 0; i < iterations; ++i) {
    auto st = execute_decode_gdn_timed(*plan, position, regions);
    if (!st) {
      std::cerr << qw38::runtime::error_message(st.error()) << '\n';
      return 1;
    }
    for (int r = 0; r < kGdnMixerRegions; ++r) {
      acc.ms[r] += regions.ms[r];
    }
  }

  std::uint64_t const weight_bytes =
      qkv_codes + qkv_scales + z_codes + z_scales + out_codes + out_scales +
      2 * ab_codes + static_cast<std::uint64_t>(kConvKernel) * kConvChannels * 2u +
      kGdnValueHeads * 4u + kHidden * 2u + kGdnValueDim * 2u;

  std::cout << "qw38_bench_gdn_mixer\n";
  std::cout << "device=" << prop.name << " sm_" << prop.major << prop.minor << '\n';
  std::cout << "decode-gdn q4 hidden=" << kHidden << " qkv=" << kConvChannels
            << " z=" << kGdnZWidth << " regions=" << kGdnMixerRegions
            << " (rms, qkvz, ab, conv, prep, recur, gated, out-residual)"
            << " weight_bytes=" << weight_bytes << " iterations=" << iterations
            << " kernel_launches="
            << iterations * kernel_launches_per_iteration
            << " device_copies="
            << iterations * device_copies_per_iteration
            << " kernels_per_iteration=" << kernel_launches_per_iteration
            << " copies_per_iteration=" << device_copies_per_iteration
            << " ms=" << avg << '\n';
  std::cout << "region_ms";
  for (int r = 0; r < kGdnMixerRegions; ++r) {
    std::cout << " " << kGdnMixerRegionNames[r] << "="
              << (acc.ms[r] / static_cast<float>(iterations));
  }
  std::cout << '\n';
  return 0;
}
