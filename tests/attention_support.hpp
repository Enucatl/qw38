#pragma once

#include "mlp_support.hpp"
#include "runtime/attention.hpp"

#include "cuda/attention.hpp"
#include "cuda/buffer.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace qw38::attn::test {

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
using qw38::reference::kAttnKvWidth;
using qw38::reference::kAttnOutWidth;
using qw38::reference::kDefaultRmsEps;
using qw38::reference::kHeadDim;
using qw38::reference::kHidden;
using qw38::reference::kKvHeads;
using qw38::reference::kQgWidth;
using qw38::reference::kQueryHeads;
using qw38::reference::kRopeFreqs;
using qw38::reference::kRotaryDim;
using qw38::reference::rope_inv_freq;
using qw38::reference::tol::kAttnPrepLargeAbs;
using qw38::reference::tol::kAttnPrepLargeRel;
using qw38::reference::tol::kAttnPrepSmallAbs;
using qw38::reference::tol::kAttnPrepSmallRel;
using qw38::reference::tol::kAttnProjAbs;
using qw38::reference::tol::kAttnProjRel;
using qw38::runtime::AttentionPrepBindViews;
using qw38::runtime::MemorySpace;
using qw38::runtime::TensorView;
using qw38::runtime::WorkspaceView;
using qw38::runtime::kAttentionWorkspaceBytesPerToken;

inline std::vector<std::uint16_t> pattern_h(std::uint32_t n, float seed) {
  std::vector<std::uint16_t> out(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    out[i] = fp32_to_bf16_rne(
        seed * (static_cast<float>(static_cast<int>(i % 9) - 4) / 4.0f));
  }
  return out;
}

inline std::vector<std::byte> bf16_bytes(std::span<std::uint16_t const> src) {
  return as_bytes(src);
}

inline WorkspaceView attention_workspace_view(
    void* pointer, std::uint64_t bytes = kAttentionWorkspaceBytesPerToken) {
  WorkspaceView workspace{};
  workspace.pointer = static_cast<std::byte*>(pointer);
  workspace.bytes = bytes;
  workspace.kind = qw38::format::ScratchKind::AttentionWorkspace;
  auto add = [&](std::uint64_t offset, std::uint64_t region_bytes,
                 ArithmeticDtype dtype, PhysicalLayoutId layout,
                 StorageClass storage, std::uint8_t rank, std::uint64_t e0,
                 std::uint64_t e1 = 0) {
    auto& region = workspace.region[workspace.region_count++];
    region.offset = offset;
    region.bytes = region_bytes;
    region.stride_bytes = region_bytes;
    region.tensor = make_view(workspace.pointer + offset, dtype, layout, storage,
                              true, rank, e0, e1);
  };
  add(qw38::runtime::kAttnOffQg, qw38::runtime::kAttnBytesQg,
      ArithmeticDtype::Bf16, PhysicalLayoutId::CudaBf16RowMajorV0,
      StorageClass::Bf16, 2, kQueryHeads, 2u * kHeadDim);
  add(qw38::runtime::kAttnOffK, qw38::runtime::kAttnBytesK,
      ArithmeticDtype::Bf16, PhysicalLayoutId::CudaBf16RowMajorV0,
      StorageClass::Bf16, 2, kKvHeads, kHeadDim);
  add(qw38::runtime::kAttnOffV, qw38::runtime::kAttnBytesV,
      ArithmeticDtype::Bf16, PhysicalLayoutId::CudaBf16RowMajorV0,
      StorageClass::Bf16, 2, kKvHeads, kHeadDim);
  add(qw38::runtime::kAttnOffQ, qw38::runtime::kAttnBytesQ,
      ArithmeticDtype::Bf16, PhysicalLayoutId::CudaBf16RowMajorV0,
      StorageClass::Bf16, 2, kQueryHeads, kHeadDim);
  add(qw38::runtime::kAttnOffG, qw38::runtime::kAttnBytesG,
      ArithmeticDtype::Bf16, PhysicalLayoutId::CudaBf16RowMajorV0,
      StorageClass::Bf16, 2, kQueryHeads, kHeadDim);
  auto const partial_bytes = bytes - qw38::runtime::kAttnOffPartials;
  add(qw38::runtime::kAttnOffPartials, partial_bytes, ArithmeticDtype::Fp32,
      PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, 1,
      partial_bytes / qw38::format::kFp32Size);
  return workspace;
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

struct HostAttn {
  qw38::format::PackedMatrix qg;
  qw38::format::PackedMatrix k;
  qw38::format::PackedMatrix v;
  qw38::format::PackedMatrix o;
  std::vector<std::uint16_t> w_qg;
  std::vector<std::uint16_t> w_k;
  std::vector<std::uint16_t> w_v;
  std::vector<std::uint16_t> w_o;
  std::vector<std::uint16_t> gamma;
  std::vector<std::uint16_t> gamma_q;
  std::vector<std::uint16_t> gamma_k;
  std::array<float, kRopeFreqs> inv_freq{};
  std::vector<float> residual;
};

struct DeviceAttn {
  DeviceBuffer qg_codes;
  DeviceBuffer qg_scales;
  DeviceBuffer k_codes;
  DeviceBuffer k_scales;
  DeviceBuffer v_codes;
  DeviceBuffer v_scales;
  DeviceBuffer gamma;
  DeviceBuffer gamma_q;
  DeviceBuffer gamma_k;
  DeviceBuffer inv_freq;
  DeviceBuffer residual;
  DeviceBuffer normalized;
  DeviceBuffer workspace;
  DeviceBuffer kv;
  std::uint64_t populated{0};
  std::uint64_t capacity{4};
  bool has_scales{true};
  PhysicalLayoutId layout{PhysicalLayoutId::CudaQ4G64V0};
  StorageClass storage{StorageClass::Int4Grouped};
};

inline bool upload_packed_named(DeviceBuffer& codes, DeviceBuffer& scales,
                                qw38::format::PackedMatrix const& packed,
                                Stream const& stream, std::string_view tag) {
  return qw38::mlp::test::upload_packed(codes, scales, packed, stream, tag);
}

inline bool upload_host_attn(HostAttn const& host, DeviceAttn& dev,
                             Stream const& stream, std::uint64_t capacity = 4) {
  if (!upload_packed_named(dev.qg_codes, dev.qg_scales, host.qg, stream, "qg") ||
      !upload_packed_named(dev.k_codes, dev.k_scales, host.k, stream, "k") ||
      !upload_packed_named(dev.v_codes, dev.v_scales, host.v, stream, "v")) {
    return false;
  }
  auto g = upload_vec(host.gamma, stream);
  auto gq = upload_vec(host.gamma_q, stream);
  auto gk = upload_vec(host.gamma_k, stream);
  std::vector<float> inv(host.inv_freq.begin(), host.inv_freq.end());
  auto inf = upload_vec(inv, stream);
  auto r = upload_vec(host.residual, stream);
  auto n = DeviceBuffer::allocate(kHidden * 2);
  auto ws = DeviceBuffer::allocate(kAttentionWorkspaceBytesPerToken);
  auto kv_n = 16u * 2u * kKvHeads * capacity * kHeadDim * 2u;
  auto kv = DeviceBuffer::allocate(kv_n);
  if (!g || !gq || !gk || !inf || !r || !n || !ws || !kv) {
    fail("attention activation upload");
    return false;
  }
  if (auto st = qw38::cuda::zero(*ws, stream); !st) {
    fail("zero workspace");
    return false;
  }
  if (auto st = qw38::cuda::zero(*kv, stream); !st) {
    fail("zero kv");
    return false;
  }
  dev.gamma = std::move(*g);
  dev.gamma_q = std::move(*gq);
  dev.gamma_k = std::move(*gk);
  dev.inv_freq = std::move(*inf);
  dev.residual = std::move(*r);
  dev.normalized = std::move(*n);
  dev.workspace = std::move(*ws);
  dev.kv = std::move(*kv);
  dev.populated = 0;
  dev.capacity = capacity;
  dev.has_scales = !host.qg.scales.empty();
  dev.layout = host.qg.layout;
  dev.storage = (host.qg.layout == PhysicalLayoutId::CudaQ4G64V0)
                    ? StorageClass::Int4Grouped
                    : StorageClass::Bf16;
  return true;
}

inline AttentionPrepBindViews bind_views(DeviceAttn& dev,
                                         std::uint32_t language_layer = 3) {
  AttentionPrepBindViews v;
  v.qg = weight_view(dev.qg_codes, dev.layout, dev.storage, kQgWidth, kHidden);
  v.k = weight_view(dev.k_codes, dev.layout, dev.storage, kAttnKvWidth, kHidden);
  v.v = weight_view(dev.v_codes, dev.layout, dev.storage, kAttnKvWidth, kHidden);
  if (dev.has_scales) {
    v.qg_scales = scale_view(dev.qg_scales, dev.layout, dev.storage,
                             dev.qg_scales.bytes() / 2);
    v.k_scales =
        scale_view(dev.k_scales, dev.layout, dev.storage, dev.k_scales.bytes() / 2);
    v.v_scales =
        scale_view(dev.v_scales, dev.layout, dev.storage, dev.v_scales.bytes() / 2);
  }
  v.gamma = vec_view(dev.gamma, ArithmeticDtype::Bf16,
                     PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16, false,
                     kHidden);
  v.gamma_q = vec_view(dev.gamma_q, ArithmeticDtype::Bf16,
                       PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16, false,
                       kHeadDim);
  v.gamma_k = vec_view(dev.gamma_k, ArithmeticDtype::Bf16,
                       PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16, false,
                       kHeadDim);
  v.inv_freq = vec_view(dev.inv_freq, ArithmeticDtype::Fp32,
                        PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, false,
                        kRopeFreqs);
  v.residual = vec_view(dev.residual, ArithmeticDtype::Fp32,
                        PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true,
                        kHidden);
  v.residual.rank = 2;
  v.residual.extent[0] = 1;
  v.residual.extent[1] = kHidden;
  v.normalized = vec_view(dev.normalized, ArithmeticDtype::Bf16,
                          PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16,
                          true, kHidden);
  v.workspace = attention_workspace_view(dev.workspace.data());
  v.kv = vec_view(dev.kv, ArithmeticDtype::Bf16,
                  PhysicalLayoutId::CudaBf16KvCacheV0, StorageClass::Bf16, true, 1);
  v.kv.rank = 5;
  v.kv.extent = {16, 2, kKvHeads, dev.capacity, kHeadDim};
  auto populated =
      qw38::runtime::KvPopulatedSlot::bind(&dev.populated, dev.capacity);
  if (populated) {
    v.populated = *populated;
  }
  v.kv_capacity = dev.capacity;
  v.language_layer = language_layer;
  return v;
}

inline bool make_host_q4(HostAttn& host, std::int8_t seed) {
  auto qg = make_logical(LogicalQuantizerId::Q4G64V0, kQgWidth, kHidden, 1, 0x3C00);
  auto k = make_logical(LogicalQuantizerId::Q4G64V0, kAttnKvWidth, kHidden, 2, 0x3C00);
  auto v = make_logical(LogicalQuantizerId::Q4G64V0, kAttnKvWidth, kHidden, -1, 0x3A00);
  auto o = make_logical(LogicalQuantizerId::Q4G64V0, kHidden, kAttnOutWidth, 1, 0x3A00);
  fill_logical_pattern(qg, seed);
  fill_logical_pattern(k, static_cast<std::int8_t>(seed + 1));
  fill_logical_pattern(v, static_cast<std::int8_t>(seed + 2));
  fill_logical_pattern(o, static_cast<std::int8_t>(seed + 3));
  if (!pack_q4_from_logical(qg, host.qg, "qg") ||
      !pack_q4_from_logical(k, host.k, "k") ||
      !pack_q4_from_logical(v, host.v, "v") ||
      !pack_q4_from_logical(o, host.o, "o")) {
    return false;
  }
  auto wqg = qw38::mlp::test::decoded_weights(host.qg);
  auto wk = qw38::mlp::test::decoded_weights(host.k);
  auto wv = qw38::mlp::test::decoded_weights(host.v);
  auto wo = qw38::mlp::test::decoded_weights(host.o);
  if (!wqg || !wk || !wv || !wo) {
    fail("decode packed attention weights");
    return false;
  }
  host.w_qg = std::move(*wqg);
  host.w_k = std::move(*wk);
  host.w_v = std::move(*wv);
  host.w_o = std::move(*wo);
  host.gamma = pattern_h(kHidden, 0.15f);
  host.gamma_q = pattern_h(kHeadDim, 0.08f);
  host.gamma_k = pattern_h(kHeadDim, -0.05f);
  host.inv_freq = rope_inv_freq();
  host.residual = residual_vec(kHidden, 0.35f);
  return true;
}

inline bool make_host_bf16(HostAttn& host, float seed) {
  host.w_qg = bf16_vec(kQgWidth * kHidden, seed);
  host.w_k = bf16_vec(kAttnKvWidth * kHidden, seed * 1.1f);
  host.w_v = bf16_vec(kAttnKvWidth * kHidden, seed * 0.9f);
  if (!pack_bf16_tile(host.w_qg, kQgWidth, kHidden, host.qg, "qg") ||
      !pack_bf16_tile(host.w_k, kAttnKvWidth, kHidden, host.k, "k") ||
      !pack_bf16_tile(host.w_v, kAttnKvWidth, kHidden, host.v, "v")) {
    return false;
  }
  host.gamma = pattern_h(kHidden, 0.12f);
  host.gamma_q = pattern_h(kHeadDim, 0.07f);
  host.gamma_k = pattern_h(kHeadDim, 0.04f);
  host.inv_freq = rope_inv_freq();
  host.residual = residual_vec(kHidden, 0.28f);
  return true;
}

inline std::size_t kv_index(std::uint32_t layer, std::uint32_t component,
                            std::uint32_t head, std::uint64_t token,
                            std::uint32_t dim, std::uint64_t capacity) {
  std::size_t const head_stride = static_cast<std::size_t>(capacity) * kHeadDim;
  std::size_t const comp_stride = static_cast<std::size_t>(kKvHeads) * head_stride;
  return (static_cast<std::size_t>(layer) * 2u + component) * comp_stride +
         static_cast<std::size_t>(head) * head_stride +
         static_cast<std::size_t>(token) * kHeadDim + dim;
}

}  // namespace qw38::attn::test
