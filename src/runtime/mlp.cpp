#include "runtime/mlp.hpp"

#include "cuda/activation.hpp"
#include "cuda/decode_mmv.hpp"

#include "format/constants.hpp"
#include "format/layout.hpp"

#include <array>
#include <cmath>
#include <cstdint>
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
using qw38::cuda::kDecodeLayoutQ4G64CandidateV1;
using qw38::cuda::kDecodeQuantizerNone;
using qw38::cuda::kDecodeQuantizerQ4G64V0;
using qw38::cuda::kDecodeQuantizerQ4G64CandidateV1;
using qw38::format::ArithmeticDtype;
using qw38::format::PhysicalLayoutId;
using qw38::format::StorageClass;

Error arg_error(std::string_view field, std::string_view detail) {
  return make_error(ErrorCode::InvalidArgument, field, detail);
}

template <typename Pointer>
std::expected<std::uint64_t, Error> view_bytes(
    BasicTensorView<Pointer> const& v, std::string_view field) {
  if (v.rank == 0 || v.rank > qw38::format::kMaxRank) {
    return std::unexpected(arg_error(field, "rank is invalid"));
  }
  std::uint64_t n = 1;
  for (std::uint8_t i = 0; i < v.rank; ++i) {
    auto product = qw38::format::checked_mul(n, v.extent[i], 0, field);
    if (!product) {
      return std::unexpected(make_error(ErrorCode::Overflow, product.error().field,
                                        product.error().detail));
    }
    n = *product;
  }
  auto bytes = qw38::format::checked_mul(
      n, qw38::format::element_size(v.dtype), 0, field);
  if (!bytes) {
    return std::unexpected(make_error(ErrorCode::Overflow, bytes.error().field,
                                      bytes.error().detail));
  }
  return *bytes;
}

template <typename Pointer>
std::expected<void, Error> require_alignment(
    BasicTensorView<Pointer> const& v, std::uint64_t alignment,
    std::string_view field) {
  auto const address = reinterpret_cast<std::uintptr_t>(v.pointer);
  if (alignment == 0 || address % alignment != 0) {
    return std::unexpected(arg_error(field, "view alignment is invalid"));
  }
  return {};
}

bool layout_ok(PhysicalLayoutId layout) noexcept {
  return layout == PhysicalLayoutId::CudaQ4G64V0 ||
         layout == PhysicalLayoutId::CudaQ4G64CandidateV1 ||
         layout == PhysicalLayoutId::CudaBf16DenseTileV0;
}

std::uint16_t quantizer_for(PhysicalLayoutId layout) noexcept {
  if (layout == PhysicalLayoutId::CudaQ4G64V0) {
    return kDecodeQuantizerQ4G64V0;
  }
  if (layout == PhysicalLayoutId::CudaQ4G64CandidateV1) {
    return kDecodeQuantizerQ4G64CandidateV1;
  }
  return kDecodeQuantizerNone;
}

template <typename Pointer>
std::expected<BasicTensorView<Pointer>, Error> as_decode_vector(
    BasicTensorView<Pointer> v, std::uint64_t elems, ArithmeticDtype dtype,
    PhysicalLayoutId layout, StorageClass storage, std::string_view field) {
  if (v.pointer == nullptr || v.space != MemorySpace::Device ||
      v.dtype != dtype || v.layout != layout || v.storage != storage) {
    return std::unexpected(arg_error(field, "typed view contract mismatch"));
  }
  auto bytes = view_bytes(v, field);
  if (!bytes) {
    return std::unexpected(bytes.error());
  }
  if (v.rank != 1 || v.extent[0] != elems) {
    return std::unexpected(arg_error(field, "typed view extent mismatch"));
  }
  auto expected_bytes =
      qw38::format::checked_mul(elems, qw38::format::element_size(dtype), 0, field);
  if (!expected_bytes) {
    return std::unexpected(make_error(ErrorCode::Overflow,
                                      expected_bytes.error().field,
                                      expected_bytes.error().detail));
  }
  if (*bytes != *expected_bytes) {
    return std::unexpected(arg_error(field, "view byte extent mismatch"));
  }
  if (auto st = require_alignment(v, qw38::format::element_size(dtype), field);
      !st) {
    return std::unexpected(st.error());
  }
  return v;
}

std::expected<MlpWeightBinding, Error> bind_weight(ConstTensorView codes,
                                                   ConstTensorView scales,
                                                   std::uint32_t want_n,
                                                   std::uint32_t want_k,
                                                   std::string_view field) {
  if (codes.pointer == nullptr) {
    return std::unexpected(arg_error(field, "null codes"));
  }
  if (codes.space != MemorySpace::Device) {
    return std::unexpected(arg_error(field, "codes must be device memory"));
  }
  if (codes.dtype != ArithmeticDtype::Bf16) {
    return std::unexpected(arg_error(field, "payload arithmetic dtype must be bf16"));
  }
  if (auto st = require_alignment(codes, 16, field); !st) {
    return std::unexpected(st.error());
  }
  if (codes.rank != 2 || codes.extent[0] != want_n || codes.extent[1] != want_k) {
    return std::unexpected(arg_error(field, "logical shape must be V0 MLP geometry"));
  }
  if (!layout_ok(codes.layout)) {
    return std::unexpected(
        arg_error(field, "layout must be cuda_q4g64_v0 or cuda_bf16_dense_tile_v0"));
  }
  bool const q4 = codes.layout == PhysicalLayoutId::CudaQ4G64V0 ||
                  codes.layout == PhysicalLayoutId::CudaQ4G64CandidateV1;
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
  if (b.codes_bytes == 0) {
    return std::unexpected(arg_error(field, "payload byte count is invalid"));
  }
  if (b.scales_bytes == 0) {
    if (scales != ConstTensorView{}) {
      return std::unexpected(arg_error(field, "BF16 dense tile must not supply scales"));
    }
  } else {
    if (scales.pointer == nullptr) {
      return std::unexpected(arg_error(field, "Q4 weight requires scales"));
    }
    if (scales.space != MemorySpace::Device) {
      return std::unexpected(arg_error(field, "scales must be device memory"));
    }
    if (scales.dtype != ArithmeticDtype::Fp16 || scales.layout != codes.layout ||
        scales.storage != codes.storage || scales.rank != 1 ||
        scales.extent[0] != b.scales_bytes / qw38::format::kFp16Size) {
      return std::unexpected(arg_error(field, "scale typed view contract mismatch"));
    }
    if (auto st = require_alignment(scales, qw38::format::kFp16Size, field);
        !st) {
      return std::unexpected(st.error());
    }
    auto scale_bytes = view_bytes(scales, field);
    if (!scale_bytes) {
      return std::unexpected(scale_bytes.error());
    }
    if (*scale_bytes != b.scales_bytes) {
      return std::unexpected(arg_error(field, "scale view length does not match layout"));
    }
    b.scales = scales;
  }
  return b;
}

bool same_family(MlpWeightBinding const& a, MlpWeightBinding const& b) noexcept {
  return a.layout == b.layout && a.quantizer == b.quantizer;
}

struct ByteInterval {
  std::string_view name;
  std::uintptr_t begin{};
  std::uintptr_t end{};
};

std::expected<ByteInterval, Error> byte_interval(void const* pointer,
                                                 std::uint64_t bytes,
                                                 std::string_view name) {
  auto const begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (pointer == nullptr || bytes == 0) {
    return std::unexpected(arg_error(name, "empty byte interval"));
  }
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
    return std::unexpected(
        arg_error(name, "byte interval overflows address space"));
  }
  return ByteInterval{.name = name,
                      .begin = begin,
                      .end = begin + static_cast<std::uintptr_t>(bytes)};
}

bool overlaps(ByteInterval const& a, ByteInterval const& b) noexcept {
  return a.begin < b.end && b.begin < a.end;
}

std::expected<void, Error> validate_non_overlapping_bindings(
    MlpWeightBinding const& gate, MlpWeightBinding const& up,
    MlpWeightBinding const& down, ConstTensorView const& gamma,
    ConstTensorView const& h_mid, TensorView const& next_h,
    TensorView const& normalized, TensorView const& swiglu) {
  std::array<ByteInterval, 11> intervals{};
  std::size_t count = 0;
  auto add = [&](void const* pointer, std::uint64_t bytes,
                 std::string_view name) -> std::expected<void, Error> {
    if (pointer == nullptr && bytes == 0) {
      return {};
    }
    auto interval = byte_interval(pointer, bytes, name);
    if (!interval) {
      return std::unexpected(interval.error());
    }
    intervals[count++] = *interval;
    return {};
  };

  auto add_weight = [&](MlpWeightBinding const& weight,
                        std::string_view codes_name,
                        std::string_view scales_name)
      -> std::expected<void, Error> {
    if (auto st = add(weight.codes.pointer, weight.codes_bytes, codes_name); !st) {
      return st;
    }
    return add(weight.scales.pointer, weight.scales_bytes, scales_name);
  };

  if (auto st = add_weight(gate, "gate", "gate_scales"); !st) {
    return st;
  }
  if (auto st = add_weight(up, "up", "up_scales"); !st) {
    return st;
  }
  if (auto st = add_weight(down, "down", "down_scales"); !st) {
    return st;
  }
  if (auto st = add(gamma.pointer, static_cast<std::uint64_t>(kHidden) * 2u,
                    "gamma");
      !st) {
    return st;
  }
  if (auto st = add(h_mid.pointer, static_cast<std::uint64_t>(kHidden) * 4u,
                    "h_mid");
      !st) {
    return st;
  }
  if (auto st = add(next_h.pointer, static_cast<std::uint64_t>(kHidden) * 4u,
                    "next_h");
      !st) {
    return st;
  }
  if (auto st = add(normalized.pointer,
                    static_cast<std::uint64_t>(kHidden) * 2u, "normalized");
      !st) {
    return st;
  }
  if (auto st = add(swiglu.pointer, static_cast<std::uint64_t>(kFfnWidth) * 2u,
                    "swiglu");
      !st) {
    return st;
  }

  for (std::size_t i = 0; i < count; ++i) {
    for (std::size_t j = i + 1; j < count; ++j) {
      if (overlaps(intervals[i], intervals[j])) {
        return std::unexpected(arg_error(
            "alias", std::string(intervals[i].name) + " overlaps " +
                         std::string(intervals[j].name)));
      }
    }
  }
  return {};
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
      (w.layout == kDecodeLayoutQ4G64V0 ||
       w.layout == kDecodeLayoutQ4G64CandidateV1) ? DecodeDtype::Q4
                                                   : DecodeDtype::Bf16;
  d.codes = decode_matrix_view(const_cast<void*>(w.codes.pointer), dtype,
                               w.layout, w.n, w.k,
                               w.padded_n, w.padded_k, w.codes_bytes, 16);
  if (w.scales_bytes != 0) {
    d.scales = decode_matrix_view(
        const_cast<void*>(w.scales.pointer), DecodeDtype::Fp16, w.layout,
        w.padded_n,
        w.padded_k / 64u, w.padded_n, w.padded_k / 64u,
        w.scales_bytes, 2);
  }
  return d;
}

std::expected<ConstTensorView, Error> require_payload(Model const& model,
    std::string const& name, qw38::format::TensorRole role,
    std::uint32_t layer) {
  auto id = model.resolve_tensor(name, qw38::format::SemanticNodeKind::Mlp,
                                 role, layer);
  if (!id) return std::unexpected(id.error());
  auto v = model.payload(*id);
  if (!v) {
    return std::unexpected(v.error());
  }
  return *v;
}

std::expected<ConstTensorView, Error> optional_scales(
    Model const& model, std::string const& name, PhysicalLayoutId layout,
    std::uint32_t layer) {
  if (layout == PhysicalLayoutId::CudaBf16DenseTileV0) {
    return ConstTensorView{};
  }
  auto id = model.resolve_tensor(name, qw38::format::SemanticNodeKind::Mlp,
                                 qw38::format::TensorRole::DenseWeight, layer);
  if (!id) return std::unexpected(id.error());
  auto v = model.scales(*id);
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

  auto gamma = as_decode_vector(
      views.gamma, kHidden, ArithmeticDtype::Bf16,
      PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16, "gamma");
  if (!gamma) {
    return std::unexpected(gamma.error());
  }
  auto h_mid = as_decode_vector(
      views.h_mid, kHidden, ArithmeticDtype::Fp32,
      PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, "h_mid");
  if (!h_mid) {
    return std::unexpected(h_mid.error());
  }
  auto next_h = as_decode_vector(
      views.next_h, kHidden, ArithmeticDtype::Fp32,
      PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, "next_h");
  if (!next_h) {
    return std::unexpected(next_h.error());
  }
  auto normalized = as_decode_vector(
      views.normalized, kHidden, ArithmeticDtype::Bf16,
      PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, "normalized");
  if (!normalized) {
    return std::unexpected(normalized.error());
  }
  auto swiglu = as_decode_vector(
      views.swiglu, kFfnWidth, ArithmeticDtype::Bf16,
      PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16, "swiglu");
  if (!swiglu) {
    return std::unexpected(swiglu.error());
  }
  if (auto st = validate_non_overlapping_bindings(
          *gate, *up, *down, *gamma, *h_mid, *next_h, *normalized, *swiglu);
      !st) {
    return std::unexpected(st.error());
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
  auto const* session_stream = detail::SessionPlanAccess::stream(session);
  if (session_stream == nullptr || session_stream->empty() || stream.empty()) {
    return std::unexpected(arg_error("stream", "session and plan streams must be open"));
  }
  if (model.device() != stream.device() ||
      session_stream->native() != stream.native()) {
    return std::unexpected(arg_error(
        "stream", "model and plan must use the session device and stream"));
  }
  auto const* session_state = detail::SessionPlanAccess::execution_state(session);
  if (session_state == nullptr || session_state->is_poisoned()) {
    return std::unexpected(arg_error("session", "session is poisoned or closed"));
  }
  if (layer >= kMlpLayers) {
    return std::unexpected(arg_error("layer", "layer index must be < 64"));
  }
  auto const gate_n = mlp_gate_name(layer);
  auto const up_n = mlp_up_name(layer);
  auto const down_n = mlp_down_name(layer);
  auto const norm_n = mlp_norm_name(layer);

  auto gate = require_payload(model, gate_n, qw38::format::TensorRole::DenseWeight, layer);
  auto up = require_payload(model, up_n, qw38::format::TensorRole::DenseWeight, layer);
  auto down = require_payload(model, down_n, qw38::format::TensorRole::DenseWeight, layer);
  auto gamma = require_payload(model, norm_n, qw38::format::TensorRole::AdditiveNorm, layer);
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
  auto gate_s = optional_scales(model, gate_n, gate->layout, layer);
  auto up_s = optional_scales(model, up_n, up->layout, layer);
  auto down_s = optional_scales(model, down_n, down->layout, layer);
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
  h_mid.rank = 1;
  h_mid.extent = {};
  h_mid.extent[0] = kHidden;
  auto next_h = session.residual_h();
  next_h.rank = 1;
  next_h.extent = {};
  next_h.extent[0] = kHidden;
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
  views.normalized = normalized->region[0].tensor;
  views.normalized.rank = 1;
  views.normalized.extent = {};
  views.normalized.extent[0] = kHidden;
  views.swiglu = swiglu->region[0].tensor;
  views.swiglu.rank = 1;
  views.swiglu.extent = {};
  views.swiglu.extent[0] = kFfnWidth;
  auto plan = bind_mlp_plan(views, stream, eps);
  if (!plan) return std::unexpected(plan.error());
  plan->session_state = session_state;
  return plan;
}

std::expected<void, Error> execute_decode_mlp(MlpPlan const& plan) {
  if (plan.session_state != nullptr && plan.session_state->is_poisoned()) {
    return std::unexpected(arg_error("session", "session is poisoned"));
  }
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
