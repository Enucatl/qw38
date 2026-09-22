#include "runtime/mlp.hpp"

#include "cuda/activation.hpp"
#include "cuda/decode_mmv.hpp"

#include "format/constants.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <string_view>

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

bool layout_ok(PhysicalLayoutId layout) noexcept {
  return layout == PhysicalLayoutId::CudaQ4G64V0 ||
         layout == PhysicalLayoutId::CudaBf16DenseTileV0;
}

std::uint16_t quantizer_for(PhysicalLayoutId layout) noexcept {
  if (layout == PhysicalLayoutId::CudaQ4G64V0) {
    return kDecodeQuantizerQ4G64V0;
  }
  return kDecodeQuantizerNone;
}

std::expected<TensorView, Error> as_decode_vector(TensorView v,
                                                  std::uint64_t elems,
                                                  ArithmeticDtype dtype,
                                                  bool writable,
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

std::expected<MlpWeightBinding, Error> bind_weight(TensorView codes,
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
    return std::unexpected(arg_error(field, "logical shape must be V0 MLP geometry"));
  }
  if (!layout_ok(codes.layout)) {
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

  MlpWeightBinding b;
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

bool same_family(MlpWeightBinding const& a, MlpWeightBinding const& b) noexcept {
  return a.layout == b.layout && a.quantizer == b.quantizer;
}

DecodeMmvDesc mmv_from_weight(MlpWeightBinding const& w) {
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

}  // namespace

std::string mlp_gate_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer << ".mlp.gate_proj.weight";
  return out.str();
}

std::string mlp_up_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer << ".mlp.up_proj.weight";
  return out.str();
}

std::string mlp_down_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer << ".mlp.down_proj.weight";
  return out.str();
}

std::string mlp_norm_name(std::uint32_t layer) {
  std::ostringstream out;
  out << "model.language_model.layers." << layer
      << ".post_attention_layernorm.weight";
  return out.str();
}

std::expected<MlpPlan, Error> bind_mlp_plan(MlpBindViews const& views,
                                            qw38::cuda::Stream const& stream,
                                            float eps) {
  if (stream.empty()) {
    return std::unexpected(arg_error("stream", "empty stream"));
  }
  if (!std::isfinite(eps) || eps <= 0.0f) {
    return std::unexpected(arg_error("eps", "epsilon must be finite and > 0"));
  }

  auto gate = bind_weight(views.gate, views.gate_scales, kFfnWidth, kHidden,
                          "gate");
  if (!gate) {
    return std::unexpected(gate.error());
  }
  auto up = bind_weight(views.up, views.up_scales, kFfnWidth, kHidden, "up");
  if (!up) {
    return std::unexpected(up.error());
  }
  auto down =
      bind_weight(views.down, views.down_scales, kHidden, kFfnWidth, "down");
  if (!down) {
    return std::unexpected(down.error());
  }
  if (!same_family(*gate, *up) || !same_family(*gate, *down)) {
    return std::unexpected(
        arg_error("layout", "gate/up/down must share one Q4 or BF16-control family"));
  }

  auto gamma = as_decode_vector(views.gamma, kHidden, ArithmeticDtype::Bf16,
                                false, "gamma");
  if (!gamma) {
    return std::unexpected(gamma.error());
  }
  auto h_mid = as_decode_vector(views.h_mid, kHidden, ArithmeticDtype::Fp32,
                                false, "h_mid");
  if (!h_mid) {
    return std::unexpected(h_mid.error());
  }
  auto next_h = as_decode_vector(views.next_h, kHidden, ArithmeticDtype::Fp32,
                                 true, "next_h");
  if (!next_h) {
    return std::unexpected(next_h.error());
  }
  auto normalized = as_decode_vector(views.normalized, kHidden,
                                     ArithmeticDtype::Bf16, true, "normalized");
  if (!normalized) {
    return std::unexpected(normalized.error());
  }
  auto swiglu = as_decode_vector(views.swiglu, kFfnWidth, ArithmeticDtype::Bf16,
                                 true, "swiglu");
  if (!swiglu) {
    return std::unexpected(swiglu.error());
  }
  if (h_mid->pointer == next_h->pointer) {
    return std::unexpected(
        arg_error("next_h", "next residual must be distinct from h_mid"));
  }
  if (gamma->pointer == h_mid->pointer || gamma->pointer == next_h->pointer ||
      normalized->pointer == swiglu->pointer ||
      h_mid->pointer == normalized->pointer ||
      next_h->pointer == normalized->pointer) {
    return std::unexpected(arg_error("scratch",
                                     "gamma, residuals, normalized, and SwiGLU must be distinct"));
  }

  MlpPlan plan;
  plan.gate = *gate;
  plan.up = *up;
  plan.down = *down;
  plan.gamma = *gamma;
  plan.h_mid = *h_mid;
  plan.next_h = *next_h;
  plan.normalized = *normalized;
  plan.swiglu = *swiglu;
  plan.stream = &stream;
  plan.eps = eps;
  return plan;
}

std::expected<MlpPlan, Error> bind_mlp_plan(Model const& model,
                                            Session const& session,
                                            std::uint32_t layer,
                                            qw38::cuda::Stream const& stream,
                                            float eps) {
  if (layer >= kMlpLayers) {
    return std::unexpected(arg_error("layer", "layer index must be < 64"));
  }
  auto const gate_n = mlp_gate_name(layer);
  auto const up_n = mlp_up_name(layer);
  auto const down_n = mlp_down_name(layer);
  auto const norm_n = mlp_norm_name(layer);

  auto gate = require_payload(model, gate_n);
  auto up = require_payload(model, up_n);
  auto down = require_payload(model, down_n);
  auto gamma = require_payload(model, norm_n);
  if (!gate) {
    return std::unexpected(gate.error());
  }
  if (!up) {
    return std::unexpected(up.error());
  }
  if (!down) {
    return std::unexpected(down.error());
  }
  if (!gamma) {
    return std::unexpected(gamma.error());
  }
  auto gate_s = optional_scales(model, gate_n, gate->layout);
  auto up_s = optional_scales(model, up_n, up->layout);
  auto down_s = optional_scales(model, down_n, down->layout);
  if (!gate_s) {
    return std::unexpected(gate_s.error());
  }
  if (!up_s) {
    return std::unexpected(up_s.error());
  }
  if (!down_s) {
    return std::unexpected(down_s.error());
  }

  auto h_mid = session.residual_h_mid();
  auto next_h = session.residual_h();
  auto normalized = session.scratch(qw38::format::ScratchKind::NormalizedHidden);
  auto swiglu = session.scratch(qw38::format::ScratchKind::MlpSwiglu);
  if (!normalized) {
    return std::unexpected(normalized.error());
  }
  if (!swiglu) {
    return std::unexpected(swiglu.error());
  }

  MlpBindViews views;
  views.gate = *gate;
  views.gate_scales = *gate_s;
  views.up = *up;
  views.up_scales = *up_s;
  views.down = *down;
  views.down_scales = *down_s;
  views.gamma = *gamma;
  views.h_mid = h_mid;
  views.next_h = next_h;
  views.normalized = *normalized;
  views.swiglu = *swiglu;
  return bind_mlp_plan(views, stream, eps);
}

std::expected<void, Error> execute_decode_mlp(MlpPlan const& plan) {
  if (plan.stream == nullptr || plan.stream->empty()) {
    return std::unexpected(arg_error("stream", "empty stream"));
  }
  auto* h_mid = static_cast<float const*>(plan.h_mid.pointer);
  auto* next_h = static_cast<float*>(plan.next_h.pointer);
  auto* gamma = static_cast<std::uint16_t const*>(plan.gamma.pointer);
  auto* normalized = static_cast<std::uint16_t*>(plan.normalized.pointer);
  auto* swiglu = static_cast<std::uint16_t*>(plan.swiglu.pointer);
  if (h_mid == nullptr || next_h == nullptr || gamma == nullptr ||
      normalized == nullptr || swiglu == nullptr) {
    return std::unexpected(arg_error("mlp", "plan views are null"));
  }

  // Region 1: post-mixer zero-centered RMS → BF16 normalized h_mid.
  if (auto st = qw38::cuda::launch_hidden_rms(h_mid, gamma, plan.eps, 1,
                                              normalized, *plan.stream);
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }

  // Region 2: paired gate/up, FP32 SiLU(gate)×up, BF16 SwiGLU scratch only.
  DecodeMmvPairedDesc paired;
  paired.a = mmv_from_weight(plan.gate);
  paired.a.input = decode_vector_view(
      normalized, DecodeDtype::Bf16, qw38::cuda::kDecodeLayoutBf16VectorV0,
      paired.a.k, static_cast<std::uint64_t>(paired.a.k) * 2u, 2, false);
  paired.a.output = decode_vector_view(
      swiglu, DecodeDtype::Bf16, qw38::cuda::kDecodeLayoutBf16VectorV0,
      paired.a.n, static_cast<std::uint64_t>(paired.a.n) * 2u, 2, true);
  paired.a.epilogue = DecodeEpilogue::SwigluStoreBf16;
  paired.b = mmv_from_weight(plan.up);
  paired.b.input = paired.a.input;
  paired.b.epilogue = paired.a.epilogue;
  if (auto st = qw38::cuda::launch_decode_mmv_paired(paired, *plan.stream); !st) {
    return std::unexpected(from_cuda(st.error()));
  }

  // Region 3: Q4/BF16 down contraction, FP32 add to original h_mid.
  DecodeMmvDesc down = mmv_from_weight(plan.down);
  down.input = decode_vector_view(
      swiglu, DecodeDtype::Bf16, qw38::cuda::kDecodeLayoutBf16VectorV0,
      down.k, static_cast<std::uint64_t>(down.k) * 2u, 2, false);
  down.residual = decode_vector_view(
      const_cast<float*>(h_mid), DecodeDtype::Fp32,
      qw38::cuda::kDecodeLayoutFp32VectorV0, down.n,
      static_cast<std::uint64_t>(down.n) * 4u, 4, false);
  down.output = decode_vector_view(
      next_h, DecodeDtype::Fp32, qw38::cuda::kDecodeLayoutFp32VectorV0,
      down.n, static_cast<std::uint64_t>(down.n) * 4u, 4, true);
  down.epilogue = DecodeEpilogue::ResidualAddFp32;
  if (auto st = qw38::cuda::launch_decode_mmv(down, *plan.stream); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return {};
}

}  // namespace qw38::runtime
