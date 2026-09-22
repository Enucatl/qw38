#include "runtime/gdn.hpp"

#include "cuda/activation.hpp"
#include "cuda/copy.hpp"
#include "cuda/decode_mmv.hpp"
#include "cuda/event.hpp"
#include "cuda/gdn.hpp"

#include "format/constants.hpp"

#include <cmath>
#include <sstream>
#include <string_view>
#include <utility>

namespace qw38::runtime {
namespace {

using qw38::cuda::DecodeEpilogue;
using qw38::cuda::DecodeDtype;
using qw38::cuda::DecodeMmvDesc;
using qw38::cuda::DecodeMmvPairedDesc;
using qw38::cuda::decode_matrix_view;
using qw38::cuda::decode_vector_view;
using qw38::cuda::decode_code_bytes;
using qw38::cuda::decode_pad_k;
using qw38::cuda::decode_pad_n;
using qw38::cuda::decode_scale_bytes;
using qw38::cuda::kDecodeLayoutBf16DenseTileV0;
using qw38::cuda::kDecodeLayoutQ4G64V0;
using qw38::cuda::kDecodeQuantizerNone;
using qw38::cuda::kDecodeQuantizerQ4G64V0;
using qw38::format::ArithmeticDtype;
using qw38::format::PhysicalLayoutId;
using qw38::format::StorageClass;

Error arg_error(std::string_view field, std::string_view detail) {
  return make_error(ErrorCode::InvalidArgument, field, detail);
}

std::uint64_t element_count(TensorView const& v) noexcept {
  if (v.rank == 0) {
    return 0;
  }
  std::uint64_t n = 1;
  for (std::uint8_t i = 0; i < v.rank; ++i) {
    n *= v.extent[i];
  }
  return n;
}

std::uint64_t view_bytes(TensorView const& v) noexcept {
  auto const n = element_count(v);
  if (v.dtype == ArithmeticDtype::Fp32) {
    return n * qw38::format::kFp32Size;
  }
  return n * qw38::format::kBf16Size;
}

bool q4_or_bf16_tile(PhysicalLayoutId layout) noexcept {
  return layout == PhysicalLayoutId::CudaQ4G64V0 ||
         layout == PhysicalLayoutId::CudaBf16DenseTileV0;
}

std::uint16_t quantizer_for(PhysicalLayoutId layout) noexcept {
  if (layout == PhysicalLayoutId::CudaQ4G64V0) {
    return kDecodeQuantizerQ4G64V0;
  }
  return kDecodeQuantizerNone;
}

std::expected<TensorView, Error> as_vector(TensorView v, std::uint64_t elems,
                                           ArithmeticDtype dtype, bool writable,
                                           std::string_view field) {
  if (v.pointer == nullptr) {
    return std::unexpected(arg_error(field, "null view"));
  }
  if (v.space != MemorySpace::Device) {
    return std::unexpected(arg_error(field, "view must be device memory"));
  }
  if (v.dtype != dtype) {
    return std::unexpected(arg_error(field, "dtype mismatch"));
  }
  if (writable && !v.writable) {
    return std::unexpected(arg_error(field, "view must be writable"));
  }
  if (v.rank == 0 || element_count(v) < elems) {
    return std::unexpected(arg_error(field, "view is smaller than decode extent"));
  }
  v.rank = 1;
  v.extent = {};
  v.extent[0] = elems;
  return v;
}

std::expected<GdnWeightBinding, Error> bind_q4_or_bf16(TensorView codes,
                                                       TensorView scales,
                                                       std::uint32_t want_n,
                                                       std::uint32_t want_k,
                                                       std::string_view field) {
  if (codes.pointer == nullptr) {
    return std::unexpected(arg_error(field, "null codes"));
  }
  if (codes.space != MemorySpace::Device) {
    return std::unexpected(arg_error(field, "codes must be device memory"));
  }
  if (codes.rank != 2 || codes.extent[0] != want_n || codes.extent[1] != want_k) {
    return std::unexpected(arg_error(field, "logical shape must be V0 GDN geometry"));
  }
  if (!q4_or_bf16_tile(codes.layout)) {
    return std::unexpected(
        arg_error(field, "layout must be cuda_q4g64_v0 or cuda_bf16_dense_tile_v0"));
  }
  bool const q4 = codes.layout == PhysicalLayoutId::CudaQ4G64V0;
  if (q4) {
    if (codes.storage != StorageClass::Int4Grouped) {
      return std::unexpected(arg_error(field, "Q4 payload storage must be int4_grouped"));
    }
  } else if (codes.storage != StorageClass::Bf16) {
    return std::unexpected(arg_error(field, "BF16 payload storage must be bf16"));
  }

  GdnWeightBinding b;
  b.codes = codes;
  b.layout = static_cast<std::uint16_t>(codes.layout);
  b.quantizer = quantizer_for(codes.layout);
  b.n = want_n;
  b.k = want_k;
  b.padded_n = decode_pad_n(want_n);
  b.padded_k = decode_pad_k(want_k);
  b.codes_bytes = decode_code_bytes(b.layout, b.padded_n, b.padded_k);
  b.scales_bytes = decode_scale_bytes(b.layout, b.padded_n, b.padded_k);
  if (b.scales_bytes == 0) {
    if (scales.pointer != nullptr || element_count(scales) != 0) {
      return std::unexpected(arg_error(field, "BF16 dense tile must not supply scales"));
    }
  } else {
    if (scales.pointer == nullptr) {
      return std::unexpected(arg_error(field, "Q4 weight requires scales"));
    }
    if (scales.space != MemorySpace::Device) {
      return std::unexpected(arg_error(field, "scales must be device memory"));
    }
    if (element_count(scales) * qw38::format::kFp16Size != b.scales_bytes) {
      return std::unexpected(arg_error(field, "scale view length does not match layout"));
    }
    b.scales = scales;
  }
  return b;
}

std::expected<GdnWeightBinding, Error> bind_ab_bf16(TensorView codes,
                                                    std::string_view field) {
  if (codes.pointer == nullptr) {
    return std::unexpected(arg_error(field, "null codes"));
  }
  if (codes.space != MemorySpace::Device) {
    return std::unexpected(arg_error(field, "codes must be device memory"));
  }
  if (codes.rank != 2 || codes.extent[0] != kGdnValueHeads ||
      codes.extent[1] != kHidden) {
    return std::unexpected(arg_error(field, "a/b must be [48,5120]"));
  }
  if (codes.layout != PhysicalLayoutId::CudaBf16DenseTileV0 ||
      codes.storage != StorageClass::Bf16) {
    return std::unexpected(
        arg_error(field, "a/b must be cuda_bf16_dense_tile_v0 BF16"));
  }
  GdnWeightBinding b;
  b.codes = codes;
  b.layout = kDecodeLayoutBf16DenseTileV0;
  b.quantizer = kDecodeQuantizerNone;
  b.n = kGdnValueHeads;
  b.k = kHidden;
  b.padded_n = decode_pad_n(kGdnValueHeads);
  b.padded_k = decode_pad_k(kHidden);
  b.codes_bytes = decode_code_bytes(b.layout, b.padded_n, b.padded_k);
  return b;
}

DecodeMmvDesc mmv_from_weight(GdnWeightBinding const& w) {
  DecodeMmvDesc d;
  d.layout = w.layout;
  d.quantizer = w.quantizer;
  d.n = w.n;
  d.k = w.k;
  d.padded_n = w.padded_n;
  d.padded_k = w.padded_k;
  DecodeDtype const dtype =
      w.layout == qw38::cuda::kDecodeLayoutQ4G64V0 ? DecodeDtype::Q4
                                                   : DecodeDtype::Bf16;
  d.codes = decode_matrix_view(w.codes.pointer, dtype, w.layout, w.n, w.k,
                               w.padded_n, w.padded_k, w.codes_bytes, 16);
  if (w.scales_bytes != 0) {
    d.scales = decode_matrix_view(
        w.scales.pointer, DecodeDtype::Fp16, w.layout, w.padded_n,
        w.padded_k / 64u, w.padded_n, w.padded_k / 64u,
        w.scales_bytes, 2);
  }
  return d;
}

std::expected<TensorView, Error> require_payload(Model const& model,
                                                 std::string const& name) {
  auto v = model.payload(name);
  if (!v) {
    return std::unexpected(v.error());
  }
  return *v;
}

std::expected<TensorView, Error> optional_scales(Model const& model,
                                                 std::string const& name,
                                                 PhysicalLayoutId layout) {
  if (layout == PhysicalLayoutId::CudaBf16DenseTileV0) {
    return TensorView{};
  }
  auto v = model.scales(name);
  if (!v) {
    return std::unexpected(v.error());
  }
  return *v;
}

TensorView overlay(std::byte* base, std::uint64_t off, ArithmeticDtype dtype,
                   PhysicalLayoutId layout, StorageClass storage, bool writable,
                   std::uint8_t rank, std::uint64_t e0, std::uint64_t e1 = 0) {
  TensorView v{};
  v.pointer = base + off;
  v.dtype = dtype;
  v.layout = layout;
  v.storage = storage;
  v.space = MemorySpace::Device;
  v.writable = writable;
  v.rank = rank;
  v.extent[0] = e0;
  v.extent[1] = e1;
  return v;
}

std::expected<void, Error> region_rms(GdnFrontPlan const& plan) {
  auto* residual = static_cast<float*>(plan.residual.pointer);
  auto* gamma = static_cast<std::uint16_t const*>(plan.gamma.pointer);
  auto* normalized = static_cast<std::uint16_t*>(plan.normalized.pointer);
  if (residual == nullptr || gamma == nullptr || normalized == nullptr) {
    return std::unexpected(arg_error("gdn", "plan views are null"));
  }
  if (auto st = qw38::cuda::launch_hidden_rms(residual, gamma, plan.eps, 1,
                                              normalized, *plan.stream);
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return {};
}

std::expected<void, Error> region_qkvz(GdnFrontPlan const& plan) {
  auto* normalized = static_cast<std::uint16_t*>(plan.normalized.pointer);
  auto* qkv = static_cast<std::uint16_t*>(plan.scratch.qkv.pointer);
  auto* z = static_cast<std::uint16_t*>(plan.scratch.z.pointer);
  DecodeMmvDesc qkv_d = mmv_from_weight(plan.qkv);
  qkv_d.input = decode_vector_view(
      normalized, DecodeDtype::Bf16, qw38::cuda::kDecodeLayoutBf16VectorV0,
      qkv_d.k, static_cast<std::uint64_t>(qkv_d.k) * 2u, 2, false);
  qkv_d.output = decode_vector_view(
      qkv, DecodeDtype::Bf16, qw38::cuda::kDecodeLayoutBf16VectorV0,
      qkv_d.n, static_cast<std::uint64_t>(qkv_d.n) * 2u, 2, true);
  qkv_d.epilogue = DecodeEpilogue::StoreBf16;
  if (auto st = qw38::cuda::launch_decode_mmv(qkv_d, *plan.stream); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  DecodeMmvDesc z_d = mmv_from_weight(plan.z);
  z_d.input = decode_vector_view(
      normalized, DecodeDtype::Bf16, qw38::cuda::kDecodeLayoutBf16VectorV0,
      z_d.k, static_cast<std::uint64_t>(z_d.k) * 2u, 2, false);
  z_d.output = decode_vector_view(
      z, DecodeDtype::Bf16, qw38::cuda::kDecodeLayoutBf16VectorV0,
      z_d.n, static_cast<std::uint64_t>(z_d.n) * 2u, 2, true);
  z_d.epilogue = DecodeEpilogue::StoreBf16;
  if (auto st = qw38::cuda::launch_decode_mmv(z_d, *plan.stream); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return {};
}

std::expected<void, Error> region_ab(GdnFrontPlan const& plan) {
  auto* normalized = static_cast<std::uint16_t*>(plan.normalized.pointer);
  auto* a = static_cast<float*>(plan.scratch.a.pointer);
  auto* b = static_cast<float*>(plan.scratch.b.pointer);
  DecodeMmvPairedDesc ab;
  ab.a = mmv_from_weight(plan.a_proj);
  ab.a.input = decode_vector_view(
      normalized, DecodeDtype::Bf16, qw38::cuda::kDecodeLayoutBf16VectorV0,
      ab.a.k, static_cast<std::uint64_t>(ab.a.k) * 2u, 2, false);
  ab.a.output = decode_vector_view(
      a, DecodeDtype::Fp32, qw38::cuda::kDecodeLayoutFp32VectorV0,
      ab.a.n, static_cast<std::uint64_t>(ab.a.n) * 4u, 4, true);
  ab.a.epilogue = DecodeEpilogue::StoreFp32;
  ab.b = mmv_from_weight(plan.b_proj);
  ab.b.input = ab.a.input;
  ab.b.output = decode_vector_view(
      b, DecodeDtype::Fp32, qw38::cuda::kDecodeLayoutFp32VectorV0,
      ab.b.n, static_cast<std::uint64_t>(ab.b.n) * 4u, 4, true);
  ab.b.epilogue = ab.a.epilogue;
  if (auto st = qw38::cuda::launch_decode_ab_bf16(ab, *plan.stream); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return {};
}

std::expected<void, Error> region_conv(GdnFrontPlan const& plan) {
  if (plan.host_cursor == nullptr || *plan.host_cursor >= kConvTaps) {
    return std::unexpected(arg_error("cursor", "invalid host cursor"));
  }
  auto* qkv = static_cast<std::uint16_t*>(plan.scratch.qkv.pointer);
  auto* convolved = static_cast<std::uint16_t*>(plan.scratch.convolved.pointer);
  auto* history = static_cast<std::uint16_t*>(plan.history.pointer);
  auto* taps = static_cast<std::uint16_t const*>(plan.taps.pointer);
  std::uint32_t const cursor = *plan.host_cursor;
  if (auto st = qw38::cuda::launch_gdn_conv_silu(qkv, taps, history, cursor,
                                                 convolved, *plan.stream);
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  *plan.host_cursor = (cursor + 1u) % kConvTaps;
  return {};
}

std::expected<void, Error> region_prep(GdnFrontPlan const& plan) {
  auto* a = static_cast<float*>(plan.scratch.a.pointer);
  auto* b = static_cast<float*>(plan.scratch.b.pointer);
  auto* convolved = static_cast<std::uint16_t*>(plan.scratch.convolved.pointer);
  auto* q_hat = static_cast<float*>(plan.scratch.q_hat.pointer);
  auto* k_hat = static_cast<float*>(plan.scratch.k_hat.pointer);
  auto* alpha = static_cast<float*>(plan.scratch.alpha.pointer);
  auto* beta = static_cast<float*>(plan.scratch.beta.pointer);
  auto* a_log = static_cast<std::uint16_t const*>(plan.a_log.pointer);
  auto* dt_bias = static_cast<std::uint16_t const*>(plan.dt_bias.pointer);
  if (auto st = qw38::cuda::launch_gdn_prepare(convolved, a, b, a_log, dt_bias,
                                               plan.eps, q_hat, k_hat, alpha,
                                               beta, *plan.stream);
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return {};
}

}  // namespace

std::expected<std::uint32_t, Error> gdn_state_index(std::uint32_t language_layer) {
  if (!is_gdn_language_layer(language_layer)) {
    return std::unexpected(arg_error("layer", "language layer is not a GDN mixer"));
  }
  return language_layer - language_layer / 4u;
}

std::string gdn_norm_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer << ".input_layernorm.weight";
  return out.str();
}

std::string gdn_gated_norm_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer << ".linear_attn.norm.weight";
  return out.str();
}

std::string gdn_qkv_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer
      << ".linear_attn.in_proj_qkv.weight";
  return out.str();
}

std::string gdn_z_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer
      << ".linear_attn.in_proj_z.weight";
  return out.str();
}

std::string gdn_a_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer
      << ".linear_attn.in_proj_a.weight";
  return out.str();
}

std::string gdn_b_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer
      << ".linear_attn.in_proj_b.weight";
  return out.str();
}

std::string gdn_out_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer << ".linear_attn.out_proj.weight";
  return out.str();
}

std::string gdn_conv_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer << ".linear_attn.conv1d.weight";
  return out.str();
}

std::string gdn_alog_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer << ".linear_attn.A_log";
  return out.str();
}

std::string gdn_dt_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer << ".linear_attn.dt_bias";
  return out.str();
}

std::expected<GdnWorkspaceViews, Error> bind_gdn_workspace(TensorView workspace) {
  if (workspace.pointer == nullptr) {
    return std::unexpected(arg_error("workspace", "null view"));
  }
  if (workspace.space != MemorySpace::Device) {
    return std::unexpected(arg_error("workspace", "view must be device memory"));
  }
  if (!workspace.writable) {
    return std::unexpected(arg_error("workspace", "view must be writable"));
  }
  if (view_bytes(workspace) < kGdnWorkspaceBytesPerToken) {
    return std::unexpected(
        arg_error("workspace", "GdnWorkspace must be 107264 bytes per token"));
  }
  auto* base = static_cast<std::byte*>(workspace.pointer);
  GdnWorkspaceViews v;
  v.qkv = overlay(base, kGdnOffQkv, ArithmeticDtype::Bf16,
                  PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, true, 1,
                  kConvChannels);
  v.z = overlay(base, kGdnOffZ, ArithmeticDtype::Bf16,
                PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, true, 2,
                kGdnValueHeads, kGdnValueDim);
  v.convolved = overlay(base, kGdnOffConvolved, ArithmeticDtype::Bf16,
                        PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, true,
                        1, kConvChannels);
  v.q_hat = overlay(base, kGdnOffQHat, ArithmeticDtype::Fp32,
                    PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true, 2,
                    kGdnKeyHeads, kGdnKeyDim);
  v.k_hat = overlay(base, kGdnOffKHat, ArithmeticDtype::Fp32,
                    PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true, 2,
                    kGdnKeyHeads, kGdnKeyDim);
  v.a = overlay(base, kGdnOffA, ArithmeticDtype::Fp32,
                PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true, 1,
                kGdnValueHeads);
  v.b = overlay(base, kGdnOffB, ArithmeticDtype::Fp32,
                PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true, 1,
                kGdnValueHeads);
  v.alpha = overlay(base, kGdnOffAlpha, ArithmeticDtype::Fp32,
                    PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true, 1,
                    kGdnValueHeads);
  v.beta = overlay(base, kGdnOffBeta, ArithmeticDtype::Fp32,
                   PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true, 1,
                   kGdnValueHeads);
  v.v = overlay(base, kGdnOffConvolved + kGdnVOffset * qw38::format::kBf16Size,
                ArithmeticDtype::Bf16, PhysicalLayoutId::CudaBf16RowMajorV0,
                StorageClass::Bf16, true, 2, kGdnValueHeads, kGdnValueDim);
  v.o = overlay(base, kGdnOffO, ArithmeticDtype::Fp32,
                PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true, 2,
                kGdnValueHeads, kGdnValueDim);
  v.u = overlay(base, kGdnOffU, ArithmeticDtype::Bf16,
                PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, true, 2,
                kGdnValueHeads, kGdnValueDim);
  return v;
}

std::expected<GdnFrontPlan, Error> bind_gdn_front_plan(
    GdnFrontBindViews const& views, qw38::cuda::Stream const& stream, float eps) {
  if (stream.empty()) {
    return std::unexpected(arg_error("stream", "empty stream"));
  }
  if (!std::isfinite(eps) || eps <= 0.0f) {
    return std::unexpected(arg_error("eps", "epsilon must be finite and > 0"));
  }
  if (views.host_cursor == nullptr) {
    return std::unexpected(arg_error("cursor", "host cursor pointer is required"));
  }
  if (*views.host_cursor >= kConvTaps) {
    return std::unexpected(arg_error("cursor", "convolution cursor must be < 3"));
  }
  auto gdn_i = gdn_state_index(views.language_layer);
  if (!gdn_i) {
    return std::unexpected(gdn_i.error());
  }

  auto qkv = bind_q4_or_bf16(views.qkv, views.qkv_scales, kConvChannels, kHidden,
                             "qkv");
  if (!qkv) {
    return std::unexpected(qkv.error());
  }
  auto z = bind_q4_or_bf16(views.z, views.z_scales, kGdnZWidth, kHidden, "z");
  if (!z) {
    return std::unexpected(z.error());
  }
  if (qkv->layout != z->layout || qkv->quantizer != z->quantizer) {
    return std::unexpected(
        arg_error("layout", "qkv/z must share one Q4 or BF16-control family"));
  }
  auto a = bind_ab_bf16(views.a, "a");
  if (!a) {
    return std::unexpected(a.error());
  }
  auto b = bind_ab_bf16(views.b, "b");
  if (!b) {
    return std::unexpected(b.error());
  }

  auto gamma = as_vector(views.gamma, kHidden, ArithmeticDtype::Bf16, false, "gamma");
  if (!gamma) {
    return std::unexpected(gamma.error());
  }
  auto residual =
      as_vector(views.residual, kHidden, ArithmeticDtype::Fp32, true, "residual");
  if (!residual) {
    return std::unexpected(residual.error());
  }
  auto normalized = as_vector(views.normalized, kHidden, ArithmeticDtype::Bf16, true,
                              "normalized");
  if (!normalized) {
    return std::unexpected(normalized.error());
  }

  if (views.taps.pointer == nullptr || views.taps.space != MemorySpace::Device) {
    return std::unexpected(arg_error("taps", "tap-major conv weights required"));
  }
  if (views.taps.layout != PhysicalLayoutId::CudaBf16TapMajorV0 ||
      views.taps.storage != StorageClass::Bf16 ||
      views.taps.dtype != ArithmeticDtype::Bf16) {
    return std::unexpected(
        arg_error("taps", "convolution must be cuda_bf16_tap_major_v0"));
  }
  if (views.taps.rank != 2 || views.taps.extent[0] != kConvKernel ||
      views.taps.extent[1] != kConvChannels) {
    return std::unexpected(arg_error("taps", "taps must be [4,10240]"));
  }

  auto a_log =
      as_vector(views.a_log, kGdnValueHeads, ArithmeticDtype::Bf16, false, "A_log");
  if (!a_log) {
    return std::unexpected(a_log.error());
  }
  auto dt = as_vector(views.dt_bias, kGdnValueHeads, ArithmeticDtype::Bf16, false,
                      "dt_bias");
  if (!dt) {
    return std::unexpected(dt.error());
  }

  if (views.history.pointer == nullptr || views.history.space != MemorySpace::Device) {
    return std::unexpected(arg_error("history", "history view required"));
  }
  if (!views.history.writable || views.history.dtype != ArithmeticDtype::Bf16) {
    return std::unexpected(arg_error("history", "history must be writable BF16"));
  }
  if (element_count(views.history) <
      static_cast<std::uint64_t>(kConvTaps) * kConvChannels) {
    return std::unexpected(arg_error("history", "history must be [3,10240]"));
  }
  TensorView history = views.history;
  history.rank = 2;
  history.extent = {};
  history.extent[0] = kConvTaps;
  history.extent[1] = kConvChannels;
  history.layout = PhysicalLayoutId::CudaBf16ConvHistoryV0;
  history.storage = StorageClass::Bf16;

  auto scratch = bind_gdn_workspace(views.workspace);
  if (!scratch) {
    return std::unexpected(scratch.error());
  }
  if (gamma->pointer == residual->pointer ||
      residual->pointer == normalized->pointer ||
      scratch->qkv.pointer == scratch->convolved.pointer ||
      scratch->q_hat.pointer == scratch->k_hat.pointer) {
    return std::unexpected(
        arg_error("scratch", "residual, normalized, qkv, and prepared arrays must be distinct"));
  }

  GdnFrontPlan plan;
  plan.qkv = *qkv;
  plan.z = *z;
  plan.a_proj = *a;
  plan.b_proj = *b;
  plan.gamma = *gamma;
  plan.taps = views.taps;
  plan.a_log = *a_log;
  plan.dt_bias = *dt;
  plan.residual = *residual;
  plan.normalized = *normalized;
  plan.scratch = *scratch;
  plan.history = history;
  plan.host_cursor = views.host_cursor;
  plan.stream = &stream;
  plan.eps = eps;
  plan.language_layer = views.language_layer;
  plan.gdn_layer = *gdn_i;
  return plan;
}

std::expected<GdnFrontPlan, Error> bind_gdn_front_plan(
    Model const& model, Session& session, std::uint32_t layer,
    qw38::cuda::Stream const& stream, float eps) {
  auto gdn_i = gdn_state_index(layer);
  if (!gdn_i) {
    return std::unexpected(gdn_i.error());
  }
  auto const qkv_n = gdn_qkv_name(layer);
  auto const z_n = gdn_z_name(layer);
  auto const a_n = gdn_a_name(layer);
  auto const b_n = gdn_b_name(layer);
  auto const conv_n = gdn_conv_name(layer);
  auto const alog_n = gdn_alog_name(layer);
  auto const dt_n = gdn_dt_name(layer);
  auto const norm_n = gdn_norm_name(layer);

  auto qkv = require_payload(model, qkv_n);
  auto z = require_payload(model, z_n);
  auto a = require_payload(model, a_n);
  auto b = require_payload(model, b_n);
  auto taps = require_payload(model, conv_n);
  auto alog = require_payload(model, alog_n);
  auto dt = require_payload(model, dt_n);
  auto gamma = require_payload(model, norm_n);
  if (!qkv) {
    return std::unexpected(qkv.error());
  }
  if (!z) {
    return std::unexpected(z.error());
  }
  if (!a) {
    return std::unexpected(a.error());
  }
  if (!b) {
    return std::unexpected(b.error());
  }
  if (!taps) {
    return std::unexpected(taps.error());
  }
  if (!alog) {
    return std::unexpected(alog.error());
  }
  if (!dt) {
    return std::unexpected(dt.error());
  }
  if (!gamma) {
    return std::unexpected(gamma.error());
  }
  auto qkv_s = optional_scales(model, qkv_n, qkv->layout);
  auto z_s = optional_scales(model, z_n, z->layout);
  if (!qkv_s) {
    return std::unexpected(qkv_s.error());
  }
  if (!z_s) {
    return std::unexpected(z_s.error());
  }

  auto cursor = session.conv_cursor_slot(*gdn_i);
  if (!cursor) {
    return std::unexpected(cursor.error());
  }
  auto off = conv_history_byte_offset(*gdn_i, 0, 0);
  if (!off) {
    return std::unexpected(off.error());
  }
  TensorView history = session.conv_history();
  history.pointer = static_cast<std::byte*>(history.pointer) + *off;
  history.rank = 2;
  history.extent = {};
  history.extent[0] = kConvTaps;
  history.extent[1] = kConvChannels;

  auto normalized = session.scratch(qw38::format::ScratchKind::NormalizedHidden);
  auto workspace = session.scratch(qw38::format::ScratchKind::GdnWorkspace);
  if (!normalized) {
    return std::unexpected(normalized.error());
  }
  if (!workspace) {
    return std::unexpected(workspace.error());
  }

  GdnFrontBindViews views;
  views.qkv = *qkv;
  views.qkv_scales = *qkv_s;
  views.z = *z;
  views.z_scales = *z_s;
  views.a = *a;
  views.b = *b;
  views.gamma = *gamma;
  views.taps = *taps;
  views.a_log = *alog;
  views.dt_bias = *dt;
  views.residual = session.residual_h();
  views.normalized = *normalized;
  views.workspace = *workspace;
  views.history = history;
  views.host_cursor = *cursor;
  views.language_layer = layer;
  return bind_gdn_front_plan(views, stream, eps);
}

std::expected<void, Error> execute_gdn_front(GdnFrontPlan const& plan) {
  if (plan.stream == nullptr || plan.stream->empty()) {
    return std::unexpected(arg_error("stream", "empty stream"));
  }
  if (auto st = region_rms(plan); !st) {
    return st;
  }
  if (auto st = region_qkvz(plan); !st) {
    return st;
  }
  if (auto st = region_ab(plan); !st) {
    return st;
  }
  if (auto st = region_conv(plan); !st) {
    return st;
  }
  return region_prep(plan);
}

std::expected<GdnRecurrencePlan, Error> bind_gdn_recurrence_plan(
    GdnRecurrenceBindViews const& views, qw38::cuda::Stream const& stream) {
  if (stream.empty()) {
    return std::unexpected(arg_error("stream", "empty stream"));
  }
  if (views.s_layer >= kGdnLayers) {
    return std::unexpected(arg_error("s_layer", "GDN state layer must be < 48"));
  }
  auto q = as_vector(views.q_hat, static_cast<std::uint64_t>(kGdnKeyHeads) * kGdnKeyDim,
                     ArithmeticDtype::Fp32, false, "q_hat");
  if (!q) {
    return std::unexpected(q.error());
  }
  auto k = as_vector(views.k_hat, static_cast<std::uint64_t>(kGdnKeyHeads) * kGdnKeyDim,
                     ArithmeticDtype::Fp32, false, "k_hat");
  if (!k) {
    return std::unexpected(k.error());
  }
  auto alpha =
      as_vector(views.alpha, kGdnValueHeads, ArithmeticDtype::Fp32, false, "alpha");
  if (!alpha) {
    return std::unexpected(alpha.error());
  }
  auto beta =
      as_vector(views.beta, kGdnValueHeads, ArithmeticDtype::Fp32, false, "beta");
  if (!beta) {
    return std::unexpected(beta.error());
  }
  auto v = as_vector(views.v, static_cast<std::uint64_t>(kGdnValueHeads) * kGdnValueDim,
                     ArithmeticDtype::Bf16, false, "v");
  if (!v) {
    return std::unexpected(v.error());
  }
  auto o = as_vector(views.o, static_cast<std::uint64_t>(kGdnValueHeads) * kGdnValueDim,
                     ArithmeticDtype::Fp32, true, "o");
  if (!o) {
    return std::unexpected(o.error());
  }
  if (views.s.pointer == nullptr) {
    return std::unexpected(arg_error("s", "null view"));
  }
  if (views.s.space != MemorySpace::Device) {
    return std::unexpected(arg_error("s", "view must be device memory"));
  }
  if (views.s.dtype != ArithmeticDtype::Fp32) {
    return std::unexpected(arg_error("s", "S must be FP32"));
  }
  if (!views.s.writable) {
    return std::unexpected(arg_error("s", "view must be writable"));
  }
  std::uint64_t const need =
      (static_cast<std::uint64_t>(views.s_layer) + 1u) * kGdnSElemsPerLayer;
  if (views.s.rank == 0 || element_count(views.s) < need) {
    return std::unexpected(
        arg_error("s", "view is smaller than the addressed GDN layer"));
  }
  if (q->pointer == k->pointer) {
    return std::unexpected(arg_error("q_hat", "q_hat and k_hat must be distinct"));
  }
  if (views.s.pointer == o->pointer) {
    return std::unexpected(arg_error("s", "S and o must be distinct"));
  }

  auto idx = gdn_state_index(views.language_layer);
  if (!idx) {
    return std::unexpected(idx.error());
  }

  GdnRecurrencePlan plan;
  plan.q_hat = *q;
  plan.k_hat = *k;
  plan.alpha = *alpha;
  plan.beta = *beta;
  plan.v = *v;
  plan.s = views.s;
  plan.s.writable = true;
  plan.o = *o;
  plan.s_layer = views.s_layer;
  plan.language_layer = views.language_layer;
  plan.gdn_layer = *idx;
  plan.stream = &stream;
  return plan;
}

std::expected<GdnRecurrencePlan, Error> bind_gdn_recurrence_plan(
    Session& session, std::uint32_t layer, qw38::cuda::Stream const& stream) {
  auto gdn_i = gdn_state_index(layer);
  if (!gdn_i) {
    return std::unexpected(gdn_i.error());
  }
  auto workspace = session.scratch(qw38::format::ScratchKind::GdnWorkspace);
  if (!workspace) {
    return std::unexpected(workspace.error());
  }
  auto slices = bind_gdn_workspace(*workspace);
  if (!slices) {
    return std::unexpected(slices.error());
  }
  GdnRecurrenceBindViews views;
  views.q_hat = slices->q_hat;
  views.k_hat = slices->k_hat;
  views.alpha = slices->alpha;
  views.beta = slices->beta;
  views.v = slices->v;
  views.s = session.gdn_s();
  views.o = slices->o;
  views.s_layer = *gdn_i;
  views.language_layer = layer;
  return bind_gdn_recurrence_plan(views, stream);
}

std::expected<void, Error> execute_gdn_recurrence(GdnRecurrencePlan const& plan) {
  if (plan.stream == nullptr || plan.stream->empty()) {
    return std::unexpected(arg_error("stream", "empty stream"));
  }
  auto* q_hat = static_cast<float const*>(plan.q_hat.pointer);
  auto* k_hat = static_cast<float const*>(plan.k_hat.pointer);
  auto* alpha = static_cast<float const*>(plan.alpha.pointer);
  auto* beta = static_cast<float const*>(plan.beta.pointer);
  auto* v = static_cast<std::uint16_t const*>(plan.v.pointer);
  auto* s = static_cast<float*>(plan.s.pointer);
  auto* o = static_cast<float*>(plan.o.pointer);
  if (q_hat == nullptr || k_hat == nullptr || alpha == nullptr || beta == nullptr ||
      v == nullptr || s == nullptr || o == nullptr) {
    return std::unexpected(arg_error("gdn", "recurrence views are null"));
  }
  if (auto st = qw38::cuda::launch_gdn_recurrence(q_hat, k_hat, alpha, beta, v, s,
                                                  plan.s_layer, o, *plan.stream);
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return {};
}

std::expected<GdnPlan, Error> bind_gdn_plan(GdnBindViews const& views,
                                           qw38::cuda::Stream const& stream,
                                           float eps) {
  GdnFrontBindViews front;
  front.qkv = views.qkv;
  front.qkv_scales = views.qkv_scales;
  front.z = views.z;
  front.z_scales = views.z_scales;
  front.a = views.a;
  front.b = views.b;
  front.gamma = views.gamma;
  front.taps = views.taps;
  front.a_log = views.a_log;
  front.dt_bias = views.dt_bias;
  front.residual = views.residual;
  front.normalized = views.normalized;
  front.workspace = views.workspace;
  front.history = views.history;
  front.host_cursor = views.host_cursor;
  front.language_layer = views.language_layer;
  auto fp = bind_gdn_front_plan(front, stream, eps);
  if (!fp) {
    return std::unexpected(fp.error());
  }
  auto out = bind_q4_or_bf16(views.out, views.out_scales, kHidden, kGdnZWidth, "out");
  if (!out) {
    return std::unexpected(out.error());
  }
  if (out->layout != fp->qkv.layout || out->quantizer != fp->qkv.quantizer) {
    return std::unexpected(
        arg_error("layout", "qkv/z/out must share one Q4 or BF16-control family"));
  }
  auto gated = as_vector(views.gated_gamma, kGdnValueDim, ArithmeticDtype::Bf16,
                         false, "gated_gamma");
  if (!gated) {
    return std::unexpected(gated.error());
  }
  auto residual_out =
      as_vector(views.residual_out, kHidden, ArithmeticDtype::Fp32, true,
                "residual_out");
  if (!residual_out) {
    return std::unexpected(residual_out.error());
  }
  if (residual_out->pointer == fp->residual.pointer) {
    return std::unexpected(
        arg_error("residual_out", "h_mid must be distinct from input residual"));
  }
  if (gated->pointer == fp->gamma.pointer) {
    return std::unexpected(
        arg_error("gated_gamma", "gated gamma must be distinct from input RMS gamma"));
  }
  if (views.s.pointer == nullptr) {
    return std::unexpected(arg_error("s", "null view"));
  }
  if (views.s.space != MemorySpace::Device) {
    return std::unexpected(arg_error("s", "view must be device memory"));
  }
  if (views.s.dtype != ArithmeticDtype::Fp32) {
    return std::unexpected(arg_error("s", "S must be FP32"));
  }
  if (!views.s.writable) {
    return std::unexpected(arg_error("s", "view must be writable"));
  }
  std::uint64_t const need =
      (static_cast<std::uint64_t>(fp->gdn_layer) + 1u) * kGdnSElemsPerLayer;
  if (views.s.rank == 0 || element_count(views.s) < need) {
    return std::unexpected(
        arg_error("s", "view is smaller than the addressed GDN layer"));
  }
  if (fp->scratch.u.pointer == nullptr) {
    return std::unexpected(arg_error("workspace", "u overlay is required"));
  }

  GdnPlan plan;
  plan.front = std::move(*fp);
  plan.out = *out;
  plan.gated_gamma = *gated;
  plan.residual_out = *residual_out;
  plan.s = views.s;
  plan.s.writable = true;
  plan.s_layer = plan.front.gdn_layer;
  return plan;
}

std::expected<GdnPlan, Error> bind_gdn_plan(Model const& model, Session& session,
                                           std::uint32_t layer,
                                           qw38::cuda::Stream const& stream,
                                           float eps) {
  auto gdn_i = gdn_state_index(layer);
  if (!gdn_i) {
    return std::unexpected(gdn_i.error());
  }
  auto const out_n = gdn_out_name(layer);
  auto const gated_n = gdn_gated_norm_name(layer);
  auto out = require_payload(model, out_n);
  auto gated = require_payload(model, gated_n);
  if (!out) {
    return std::unexpected(out.error());
  }
  if (!gated) {
    return std::unexpected(gated.error());
  }
  auto out_s = optional_scales(model, out_n, out->layout);
  if (!out_s) {
    return std::unexpected(out_s.error());
  }

  auto front = bind_gdn_front_plan(model, session, layer, stream, eps);
  if (!front) {
    return std::unexpected(front.error());
  }

  GdnBindViews views;
  views.qkv = front->qkv.codes;
  views.qkv_scales = front->qkv.scales;
  views.z = front->z.codes;
  views.z_scales = front->z.scales;
  views.a = front->a_proj.codes;
  views.b = front->b_proj.codes;
  views.out = *out;
  views.out_scales = *out_s;
  views.gamma = front->gamma;
  views.gated_gamma = *gated;
  views.taps = front->taps;
  views.a_log = front->a_log;
  views.dt_bias = front->dt_bias;
  views.residual = front->residual;
  views.residual_out = session.residual_h_mid();
  views.normalized = front->normalized;
  auto workspace = session.scratch(qw38::format::ScratchKind::GdnWorkspace);
  if (!workspace) {
    return std::unexpected(workspace.error());
  }
  views.workspace = *workspace;
  views.history = front->history;
  views.s = session.gdn_s();
  views.host_cursor = front->host_cursor;
  views.language_layer = layer;
  return bind_gdn_plan(views, stream, eps);
}

std::expected<void, Error> region_recur(GdnPlan const& plan) {
  auto* q_hat = static_cast<float const*>(plan.front.scratch.q_hat.pointer);
  auto* k_hat = static_cast<float const*>(plan.front.scratch.k_hat.pointer);
  auto* alpha = static_cast<float const*>(plan.front.scratch.alpha.pointer);
  auto* beta = static_cast<float const*>(plan.front.scratch.beta.pointer);
  auto* v = static_cast<std::uint16_t const*>(plan.front.scratch.v.pointer);
  auto* s = static_cast<float*>(plan.s.pointer);
  auto* o = static_cast<float*>(plan.front.scratch.o.pointer);
  if (q_hat == nullptr || k_hat == nullptr || alpha == nullptr || beta == nullptr ||
      v == nullptr || s == nullptr || o == nullptr) {
    return std::unexpected(arg_error("gdn", "recurrence views are null"));
  }
  if (auto st = qw38::cuda::launch_gdn_recurrence(q_hat, k_hat, alpha, beta, v, s,
                                                  plan.s_layer, o, *plan.front.stream);
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return {};
}

std::expected<void, Error> region_gated(GdnPlan const& plan) {
  auto* o = static_cast<float const*>(plan.front.scratch.o.pointer);
  auto* z = static_cast<std::uint16_t const*>(plan.front.scratch.z.pointer);
  auto* gamma = static_cast<std::uint16_t const*>(plan.gated_gamma.pointer);
  auto* u = static_cast<std::uint16_t*>(plan.front.scratch.u.pointer);
  if (o == nullptr || z == nullptr || gamma == nullptr || u == nullptr) {
    return std::unexpected(arg_error("gdn", "gated-RMS views are null"));
  }
  if (auto st = qw38::cuda::launch_gdn_output_transform(o, z, gamma, plan.front.eps,
                                                        u, *plan.front.stream);
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return {};
}

std::expected<void, Error> region_out_residual(GdnPlan const& plan) {
  auto* residual = static_cast<float const*>(plan.front.residual.pointer);
  auto* residual_out = static_cast<float*>(plan.residual_out.pointer);
  auto* u = static_cast<std::uint16_t const*>(plan.front.scratch.u.pointer);
  if (residual == nullptr || residual_out == nullptr || u == nullptr) {
    return std::unexpected(arg_error("gdn", "residual views are null"));
  }
  if (auto st = qw38::cuda::copy_d2d(residual_out, residual,
                                     kHidden * qw38::format::kFp32Size,
                                     *plan.front.stream);
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  qw38::cuda::DecodeMmvDesc out = mmv_from_weight(plan.out);
  out.input = decode_vector_view(
      const_cast<std::uint16_t*>(u), DecodeDtype::Bf16,
      qw38::cuda::kDecodeLayoutBf16VectorV0,
      out.k, static_cast<std::uint64_t>(out.k) * 2u, 2, false);
  out.residual = decode_vector_view(
      residual_out, DecodeDtype::Fp32, qw38::cuda::kDecodeLayoutFp32VectorV0,
      out.n, static_cast<std::uint64_t>(out.n) * 4u, 4, true);
  out.epilogue = DecodeEpilogue::ResidualAddFp32;
  if (auto st = qw38::cuda::launch_decode_mmv(out, *plan.front.stream); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return {};
}

std::expected<TensorView, Error> execute_decode_gdn_impl(
    GdnPlan const& plan, GdnRegionTimings* timings) {
  if (plan.front.stream == nullptr || plan.front.stream->empty()) {
    return std::unexpected(arg_error("stream", "empty stream"));
  }

  using qw38::cuda::Event;
  using qw38::cuda::elapsed_ms;
  Event marks[kGdnMixerRegions + 1];
  if (timings != nullptr) {
    for (int i = 0; i <= kGdnMixerRegions; ++i) {
      auto e = Event::create_timing();
      if (!e) {
        return std::unexpected(from_cuda(e.error()));
      }
      marks[i] = std::move(*e);
    }
  }

  auto mark = [&](int i) -> std::expected<void, Error> {
    if (timings == nullptr) {
      return {};
    }
    if (auto st = marks[i].record(*plan.front.stream); !st) {
      return std::unexpected(from_cuda(st.error()));
    }
    return {};
  };

  if (auto st = mark(0); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = region_rms(plan.front); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = mark(1); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = region_qkvz(plan.front); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = mark(2); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = region_ab(plan.front); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = mark(3); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = region_conv(plan.front); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = mark(4); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = region_prep(plan.front); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = mark(5); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = region_recur(plan); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = mark(6); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = region_gated(plan); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = mark(7); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = region_out_residual(plan); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = mark(8); !st) {
    return std::unexpected(st.error());
  }

  if (timings != nullptr) {
    if (auto st = marks[kGdnMixerRegions].sync(); !st) {
      return std::unexpected(from_cuda(st.error()));
    }
    for (int i = 0; i < kGdnMixerRegions; ++i) {
      auto ms = elapsed_ms(marks[i], marks[i + 1]);
      if (!ms) {
        return std::unexpected(from_cuda(ms.error()));
      }
      timings->ms[i] = *ms;
    }
  }
  return plan.residual_out;
}

std::expected<TensorView, Error> execute_decode_gdn(GdnPlan const& plan) {
  return execute_decode_gdn_impl(plan, nullptr);
}

std::expected<TensorView, Error> execute_decode_gdn_timed(
    GdnPlan const& plan, GdnRegionTimings& timings) {
  return execute_decode_gdn_impl(plan, &timings);
}

}  // namespace qw38::runtime
