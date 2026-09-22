#pragma once

#include "mlp_support.hpp"
#include "runtime/gdn.hpp"

#include "cuda/gdn.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace qw38::gdn::test {

using qw38::mlp::test::DeviceBuffer;
using qw38::mlp::test::Stream;
using qw38::mlp::test::all_finite;
using qw38::mlp::test::as_bytes;
using qw38::mlp::test::bf16;
using qw38::mlp::test::bf16_vec;
using qw38::mlp::test::download_vec;
using qw38::mlp::test::expect;
using qw38::mlp::test::expect_bf16_close;
using qw38::mlp::test::expect_fp32_close;
using qw38::mlp::test::f32;
using qw38::mlp::test::fail;
using qw38::mlp::test::fill_logical_pattern;
using qw38::mlp::test::g_failures;
using qw38::mlp::test::make_logical;
using qw38::mlp::test::make_view;
using qw38::mlp::test::pack_q4_from_logical;
using qw38::mlp::test::residual_vec;
using qw38::mlp::test::scale_view;
using qw38::mlp::test::upload_vec;
using qw38::mlp::test::vec_view;
using qw38::mlp::test::weight_view;
using qw38::format::ArithmeticDtype;
using qw38::format::LogicalQuantizerId;
using qw38::format::PhysicalLayoutId;
using qw38::format::StorageClass;
using qw38::format::fp32_to_bf16_rne;
using qw38::reference::kConvHistoryTaps;
using qw38::reference::kConvKernel;
using qw38::reference::kDefaultRmsEps;
using qw38::reference::kGdnHeadDim;
using qw38::reference::kGdnKeyHeads;
using qw38::reference::kGdnValueHeads;
using qw38::reference::kGdnZWidth;
using qw38::reference::kHidden;
using qw38::reference::kQkvWidth;
using qw38::reference::tol::kGdnConvBf16Abs;
using qw38::reference::tol::kGdnGateFp32Abs;
using qw38::reference::tol::kGdnGateFp32Rel;
using qw38::reference::tol::kGdnQkFp32Abs;
using qw38::reference::tol::kGdnQkFp32Rel;
using qw38::reference::tol::kGdnRecurMultiOAbs;
using qw38::reference::tol::kGdnRecurMultiORel;
using qw38::reference::tol::kGdnRecurMultiSAbs;
using qw38::reference::tol::kGdnRecurMultiSRel;
using qw38::reference::tol::kGdnRecurOAbs;
using qw38::reference::tol::kGdnRecurORel;
using qw38::reference::tol::kGdnRecurSAbs;
using qw38::reference::tol::kGdnRecurSRel;
using qw38::runtime::GdnFrontBindViews;
using qw38::runtime::MemorySpace;
using qw38::runtime::TensorView;
using qw38::runtime::kGdnSElemsPerLayer;

inline TensorView gdn_state_view(void* ptr) {
  TensorView v = make_view(ptr, ArithmeticDtype::Fp32,
                           PhysicalLayoutId::CudaFp32GdnSHvKV0,
                           StorageClass::Fp32, true, 1, kGdnSElemsPerLayer);
  v.rank = 4;
  v.extent = {qw38::runtime::kGdnLayers, kGdnValueHeads, kGdnHeadDim,
              kGdnHeadDim};
  return v;
}

inline std::vector<std::uint16_t> filled_h(std::uint32_t n, float v) {
  return std::vector<std::uint16_t>(n, fp32_to_bf16_rne(v));
}

inline std::vector<std::uint16_t> zeros_h(std::uint32_t n) {
  return filled_h(n, 0.0f);
}

inline std::vector<std::uint16_t> pattern_h(std::uint32_t n, float seed) {
  std::vector<std::uint16_t> out(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    out[i] = fp32_to_bf16_rne(
        seed * (static_cast<float>(static_cast<int>(i % 9) - 4) / 4.0f));
  }
  return out;
}

inline std::vector<float> zeros_f(std::uint32_t n) {
  return std::vector<float>(n, 0.0f);
}

inline std::vector<float> filled_f(std::uint32_t n, float v) {
  return std::vector<float>(n, v);
}

inline std::vector<float> pattern_f(std::uint32_t n, float seed) {
  std::vector<float> out(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    out[i] = seed * (static_cast<float>(static_cast<int>(i % 11) - 5) / 5.0f);
  }
  return out;
}

inline std::size_t gdn_s_index(std::uint32_t vh, std::uint32_t value,
                               std::uint32_t key) {
  return (static_cast<std::size_t>(vh) * kGdnHeadDim + value) * kGdnHeadDim + key;
}

inline std::vector<std::uint16_t> v_from_convolved(
    std::span<std::uint16_t const> convolved) {
  std::size_t const n = static_cast<std::size_t>(kGdnValueHeads) * kGdnHeadDim;
  return std::vector<std::uint16_t>(convolved.begin() + 4096,
                                    convolved.begin() + 4096 + static_cast<std::ptrdiff_t>(n));
}

inline std::vector<std::byte> bf16_bytes(std::span<std::uint16_t const> src) {
  return as_bytes(src);
}

inline bool pack_bf16_tile(std::span<std::uint16_t const> row_major, std::uint32_t n,
                           std::uint32_t k, qw38::format::PackedMatrix& out,
                           std::string_view tag) {
  auto packed = qw38::format::pack_bf16_dense_tile_v0(bf16_bytes(row_major), n, k);
  if (!packed) {
    fail(std::string(tag) + " bf16 pack: " +
         qw38::format::error_message(packed.error()));
    return false;
  }
  out = std::move(*packed);
  return true;
}

struct HostGdnMixer {
  qw38::format::PackedMatrix qkv;
  qw38::format::PackedMatrix z;
  qw38::format::PackedMatrix a;
  qw38::format::PackedMatrix b;
  qw38::format::PackedMatrix out;
  std::vector<std::uint16_t> w_qkv;
  std::vector<std::uint16_t> w_z;
  std::vector<std::uint16_t> w_a;
  std::vector<std::uint16_t> w_b;
  std::vector<std::uint16_t> w_out;
  std::vector<std::uint16_t> gamma;
  std::vector<std::uint16_t> gated_gamma;
  std::vector<std::uint16_t> taps;
  std::vector<std::uint16_t> a_log;
  std::vector<std::uint16_t> dt_bias;
  std::vector<float> residual;
};

struct DeviceGdnMixer {
  DeviceBuffer qkv_codes;
  DeviceBuffer qkv_scales;
  DeviceBuffer z_codes;
  DeviceBuffer z_scales;
  DeviceBuffer a;
  DeviceBuffer b;
  DeviceBuffer out_codes;
  DeviceBuffer out_scales;
  DeviceBuffer gamma;
  DeviceBuffer gated_gamma;
  DeviceBuffer taps;
  DeviceBuffer a_log;
  DeviceBuffer dt_bias;
  DeviceBuffer residual;
  DeviceBuffer residual_out;
  DeviceBuffer normalized;
  DeviceBuffer workspace;
  DeviceBuffer history;
  DeviceBuffer s;
  std::uint32_t cursor{0};
  std::uint64_t position{0};
  bool has_scales{true};
  PhysicalLayoutId layout{PhysicalLayoutId::CudaQ4G64V0};
  StorageClass storage{StorageClass::Int4Grouped};
};

inline bool upload_packed_named(DeviceBuffer& codes, DeviceBuffer& scales,
                                qw38::format::PackedMatrix const& packed,
                                Stream const& stream, std::string_view tag) {
  return qw38::mlp::test::upload_packed(codes, scales, packed, stream, tag);
}

inline bool upload_host_mixer(HostGdnMixer const& host, DeviceGdnMixer& dev,
                              Stream const& stream, std::uint32_t s_layers = 1) {
  if (!upload_packed_named(dev.qkv_codes, dev.qkv_scales, host.qkv, stream, "qkv") ||
      !upload_packed_named(dev.z_codes, dev.z_scales, host.z, stream, "z") ||
      !upload_packed_named(dev.out_codes, dev.out_scales, host.out, stream, "out")) {
    return false;
  }
  auto a = upload_vec(host.a.codes, stream);
  auto b = upload_vec(host.b.codes, stream);
  auto g = upload_vec(host.gamma, stream);
  auto gg = upload_vec(host.gated_gamma, stream);
  auto t = upload_vec(host.taps, stream);
  auto al = upload_vec(host.a_log, stream);
  auto dt = upload_vec(host.dt_bias, stream);
  auto r = upload_vec(host.residual, stream);
  auto ro = DeviceBuffer::allocate(kHidden * 4);
  auto n = DeviceBuffer::allocate(kHidden * 2);
  auto ws = DeviceBuffer::allocate(qw38::runtime::kGdnWorkspaceBytesPerToken);
  auto hist = DeviceBuffer::allocate(static_cast<std::uint64_t>(kConvHistoryTaps) *
                                     kQkvWidth * 2u);
  auto s = DeviceBuffer::allocate(static_cast<std::uint64_t>(s_layers) *
                                  kGdnSElemsPerLayer * 4u);
  if (!a || !b || !g || !gg || !t || !al || !dt || !r || !ro || !n || !ws ||
      !hist || !s) {
    fail("mixer upload");
    return false;
  }
  if (!qw38::cuda::zero(*ro, stream) || !qw38::cuda::zero(*n, stream) ||
      !qw38::cuda::zero(*ws, stream) || !qw38::cuda::zero(*hist, stream) ||
      !qw38::cuda::zero(*s, stream)) {
    fail("mixer zero");
    return false;
  }
  dev.a = std::move(*a);
  dev.b = std::move(*b);
  dev.gamma = std::move(*g);
  dev.gated_gamma = std::move(*gg);
  dev.taps = std::move(*t);
  dev.a_log = std::move(*al);
  dev.dt_bias = std::move(*dt);
  dev.residual = std::move(*r);
  dev.residual_out = std::move(*ro);
  dev.normalized = std::move(*n);
  dev.workspace = std::move(*ws);
  dev.history = std::move(*hist);
  dev.s = std::move(*s);
  dev.cursor = 0;
  dev.position = 0;
  dev.has_scales = !host.qkv.scales.empty();
  dev.layout = host.qkv.layout;
  dev.storage = (host.qkv.layout == PhysicalLayoutId::CudaQ4G64V0)
                    ? StorageClass::Int4Grouped
                    : StorageClass::Bf16;
  return true;
}

inline qw38::runtime::GdnBindViews mixer_views(DeviceGdnMixer& dev) {
  qw38::runtime::GdnBindViews v;
  v.qkv = weight_view(dev.qkv_codes, dev.layout, dev.storage, kQkvWidth, kHidden);
  v.z = weight_view(dev.z_codes, dev.layout, dev.storage, kGdnZWidth, kHidden);
  v.out = weight_view(dev.out_codes, dev.layout, dev.storage, kHidden, kGdnZWidth);
  if (dev.has_scales) {
    v.qkv_scales = scale_view(dev.qkv_scales, dev.layout, dev.storage,
                              dev.qkv_scales.bytes() / 2);
    v.z_scales = scale_view(dev.z_scales, dev.layout, dev.storage,
                            dev.z_scales.bytes() / 2);
    v.out_scales = scale_view(dev.out_scales, dev.layout, dev.storage,
                              dev.out_scales.bytes() / 2);
  }
  v.a = weight_view(dev.a, PhysicalLayoutId::CudaBf16DenseTileV0, StorageClass::Bf16,
                    kGdnValueHeads, kHidden);
  v.b = weight_view(dev.b, PhysicalLayoutId::CudaBf16DenseTileV0, StorageClass::Bf16,
                    kGdnValueHeads, kHidden);
  v.gamma = vec_view(dev.gamma, ArithmeticDtype::Bf16,
                     PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16, false,
                     kHidden);
  v.gated_gamma = vec_view(dev.gated_gamma, ArithmeticDtype::Bf16,
                           PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16,
                           false, kGdnHeadDim);
  v.taps = make_view(dev.taps.data(), ArithmeticDtype::Bf16,
                     PhysicalLayoutId::CudaBf16TapMajorV0, StorageClass::Bf16, false,
                     2, kConvKernel, kQkvWidth);
  v.a_log = vec_view(dev.a_log, ArithmeticDtype::Bf16,
                     PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16, false,
                     kGdnValueHeads);
  v.dt_bias = vec_view(dev.dt_bias, ArithmeticDtype::Bf16,
                       PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16, false,
                       kGdnValueHeads);
  v.residual = vec_view(dev.residual, ArithmeticDtype::Fp32,
                        PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true,
                        kHidden);
  v.residual_out =
      vec_view(dev.residual_out, ArithmeticDtype::Fp32,
               PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true, kHidden);
  v.normalized =
      vec_view(dev.normalized, ArithmeticDtype::Bf16,
               PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, true,
               kHidden);
  v.workspace = make_view(dev.workspace.data(), ArithmeticDtype::Fp32,
                          PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true,
                          1, qw38::runtime::kGdnWorkspaceBytesPerToken / 4);
  v.history = make_view(dev.history.data(), ArithmeticDtype::Bf16,
                        PhysicalLayoutId::CudaBf16ConvHistoryV0, StorageClass::Bf16,
                        true, 2, kConvHistoryTaps, kQkvWidth);
  v.s = gdn_state_view(dev.s.data());
  auto cursor = qw38::runtime::ConvCursorSlot::bind(&dev.cursor);
  if (cursor) {
    v.cursor = *cursor;
  }
  auto position = qw38::runtime::GdnPositionSlot::bind(&dev.position);
  if (position) {
    v.position = *position;
  }
  v.language_layer = 0;
  return v;
}

}  // namespace qw38::gdn::test
