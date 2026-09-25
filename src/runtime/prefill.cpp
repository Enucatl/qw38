#include "runtime/prefill.hpp"

#include "runtime/attention.hpp"
#include "runtime/gdn.hpp"
#include "runtime/language_layer.hpp"
#include "runtime/mlp.hpp"
#include "runtime/sizes.hpp"

#include "format/constants.hpp"

#include <string_view>
#include <string>

namespace qw38::runtime {
namespace {

using qw38::format::SemanticNodeKind;
using qw38::format::TensorRole;
using qw38::format::PhysicalLayoutId;
using qw38::format::StorageClass;

std::expected<qw38::cuda::PrefillWeight, Error> bind_weight(
    Model const& model, std::string_view name, SemanticNodeKind kind,
    TensorRole role, std::uint32_t layer, std::uint32_t n, std::uint32_t k) {
  auto id = model.resolve_tensor(name, kind, role, layer);
  if (!id) return std::unexpected(id.error());
  auto const* record = model.find_tensor(name);
  if (!record || record->shape.rank != 2 || record->shape.logical[0] != n ||
      record->shape.logical[1] != k ||
      record->shape.padded[0] != qw38::cuda::decode_pad_n(n) ||
      record->shape.padded[1] != qw38::cuda::decode_pad_k(k)) {
    return std::unexpected(make_error(ErrorCode::MalformedArtifact, name,
                                      "prefill projection shape mismatch"));
  }
  auto const layout = static_cast<std::uint16_t>(record->layout);
  auto const quantizer = static_cast<std::uint16_t>(record->quantizer);
  bool const q4k = layout == qw38::cuda::kDecodeLayoutQ4KCandidateV2 &&
                   quantizer == qw38::cuda::kDecodeQuantizerQ4KCandidateV2 &&
                   record->storage == StorageClass::Int4Grouped;
  bool const q8 =
      ((layout == qw38::cuda::kDecodeLayoutQ8G32CandidateV1 &&
        quantizer == qw38::cuda::kDecodeQuantizerQ8G32CandidateV1) ||
       (layout == qw38::cuda::kDecodeLayoutQ8G32V0 &&
        quantizer == qw38::cuda::kDecodeQuantizerQ8G32V0)) &&
      record->storage == StorageClass::Int8Grouped;
  bool const bf16 = layout == qw38::cuda::kDecodeLayoutBf16DenseTileV0 &&
                    quantizer == qw38::cuda::kDecodeQuantizerNone &&
                    record->storage == StorageClass::Bf16;
  if (!q4k && !q8 && !bf16) {
    return std::unexpected(make_error(ErrorCode::MalformedArtifact, name,
                                      "unsupported prefill weight representation"));
  }
  std::uint64_t const codes_bytes = qw38::cuda::decode_code_bytes(
      layout, static_cast<std::uint32_t>(record->shape.padded[0]),
      static_cast<std::uint32_t>(record->shape.padded[1]));
  std::uint64_t const scales_bytes = qw38::cuda::decode_scale_bytes(
      layout, static_cast<std::uint32_t>(record->shape.padded[0]),
      static_cast<std::uint32_t>(record->shape.padded[1]));
  if (record->payload.length != codes_bytes || record->scales.length != scales_bytes) {
    return std::unexpected(make_error(ErrorCode::MalformedArtifact, name,
                                      "prefill payload length mismatch"));
  }
  auto codes = model.payload(*id);
  if (!codes) return std::unexpected(codes.error());
  if (codes->space != MemorySpace::Device ||
      codes->dtype != qw38::format::ArithmeticDtype::Bf16) {
    return std::unexpected(make_error(ErrorCode::MalformedArtifact, name,
                                      "prefill payload type mismatch"));
  }
  ConstTensorView scales{};
  if (scales_bytes) {
    auto value = model.scales(*id);
    if (!value) return std::unexpected(value.error());
    scales = *value;
  }
  return qw38::cuda::PrefillWeight{
      .codes = codes->pointer, .scales = scales.pointer,
      .layout = layout, .quantizer = quantizer, .n = n, .k = k,
      .padded_n = static_cast<std::uint32_t>(record->shape.padded[0]),
      .padded_k = static_cast<std::uint32_t>(record->shape.padded[1]),
      .codes_bytes = codes_bytes, .scales_bytes = scales_bytes};
}

std::expected<std::uint16_t const*, Error> bind_gamma(Model const& model,
    std::string_view name, std::uint32_t layer) {
  auto id = model.resolve_tensor(name, SemanticNodeKind::Mlp,
                                 TensorRole::AdditiveNorm, layer);
  if (!id) return std::unexpected(id.error());
  auto view = model.payload(*id);
  if (!view) return std::unexpected(view.error());
  if (view->space != MemorySpace::Device ||
      view->dtype != qw38::format::ArithmeticDtype::Bf16 ||
      view->layout != PhysicalLayoutId::CudaBf16VectorV0 ||
      view->storage != StorageClass::Bf16 || view->rank != 1 ||
      view->extent[0] != kHidden) {
    return std::unexpected(make_error(ErrorCode::MalformedArtifact, name,
                                      "MLP gamma contract mismatch"));
  }
  return static_cast<std::uint16_t const*>(view->pointer);
}

}  // namespace

std::expected<PrefillLayerProjectionPlan, Error> bind_prefill_layer_projections(
    Model const& model, std::uint32_t layer,
    qw38::cuda::Stream const& stream) {
  if (layer >= kLanguageLayers || stream.empty() ||
      model.device() != stream.device()) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.layer",
                                      "layer or device mismatch"));
  }
  PrefillLayerProjectionPlan p;
  p.layer = layer;
  p.gdn = is_gdn_language_layer(layer);
  SemanticNodeKind const kind = p.gdn ? SemanticNodeKind::GatedDeltaNet
                                      : SemanticNodeKind::GatedAttention;
  auto get = [&](std::string const& name, SemanticNodeKind node,
                 std::uint32_t n, std::uint32_t k) {
    return bind_weight(model, name, node, TensorRole::DenseWeight, layer, n, k);
  };
  if (p.gdn) {
    auto first = get(gdn_qkv_name(layer), kind, kConvChannels, kHidden);
    auto second = get(gdn_z_name(layer), kind, kGdnZWidth, kHidden);
    auto third = get(gdn_a_name(layer), kind, kGdnValueHeads, kHidden);
    auto fourth = get(gdn_b_name(layer), kind, kGdnValueHeads, kHidden);
    auto out = get(gdn_out_name(layer), kind, kHidden, kGdnZWidth);
    if (!first) return std::unexpected(first.error());
    if (!second) return std::unexpected(second.error());
    if (!third) return std::unexpected(third.error());
    if (!fourth) return std::unexpected(fourth.error());
    if (!out) return std::unexpected(out.error());
    p.first = *first; p.second = *second; p.third = *third;
    p.fourth = *fourth; p.mixer_out = *out;
  } else {
    auto first = get(attn_q_name(layer), kind, kQgWidth, kHidden);
    auto second = get(attn_k_name(layer), kind, kAttnKvWidth, kHidden);
    auto third = get(attn_v_name(layer), kind, kAttnKvWidth, kHidden);
    auto out = get(attn_o_name(layer), kind, kHidden, kAttnOutWidth);
    if (!first) return std::unexpected(first.error());
    if (!second) return std::unexpected(second.error());
    if (!third) return std::unexpected(third.error());
    if (!out) return std::unexpected(out.error());
    p.first = *first; p.second = *second; p.third = *third;
    p.mixer_out = *out;
  }
  auto gate = get(mlp_gate_name(layer), SemanticNodeKind::Mlp, kFfnWidth, kHidden);
  auto up = get(mlp_up_name(layer), SemanticNodeKind::Mlp, kFfnWidth, kHidden);
  auto down = get(mlp_down_name(layer), SemanticNodeKind::Mlp, kHidden, kFfnWidth);
  auto gamma = bind_gamma(model, mlp_norm_name(layer), layer);
  if (!gate) return std::unexpected(gate.error());
  if (!up) return std::unexpected(up.error());
  if (!down) return std::unexpected(down.error());
  if (!gamma) return std::unexpected(gamma.error());
  p.mlp_gate = *gate; p.mlp_up = *up; p.mlp_down = *down;
  p.mlp_gamma = *gamma;
  return p;
}

std::expected<qw38::cuda::PrefillWeight, Error> bind_prefill_head(
    Model const& model, qw38::cuda::Stream const& stream) {
  if (stream.empty() || model.device() != stream.device()) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.head",
                                      "device mismatch"));
  }
  // The output head has no language-layer index; its graph instance is 129.
  auto const* record = model.find_tensor("lm_head.weight");
  if (!record) {
    return std::unexpected(make_error(ErrorCode::MalformedArtifact, "lm_head",
                                      "missing head weight"));
  }
  std::uint32_t count = 0;
  for (auto const& b : model.graph_bindings()) {
    if (b.tensor_id != record->tensor_id) continue;
    if (b.instance_id != 129 || b.kind != SemanticNodeKind::LmHead ||
        b.role != TensorRole::LmHeadWeight ||
        b.layer_index != qw38::format::kNoLayerIndex) {
      return std::unexpected(make_error(ErrorCode::MalformedArtifact, "lm_head",
                                        "head graph binding mismatch"));
    }
    ++count;
  }
  if (count != 1) {
    return std::unexpected(make_error(ErrorCode::MalformedArtifact, "lm_head",
                                      "head graph binding count mismatch"));
  }
  // A layer index is not meaningful for the head; bind_weight's semantic
  // resolver is layer-scoped, so validate the same payload contract here.
  auto data = model.payload(record->tensor_id);
  if (!data) return std::unexpected(data.error());
  auto scales = model.scales(record->tensor_id);
  if (!scales) return std::unexpected(scales.error());
  auto const layout = static_cast<std::uint16_t>(record->layout);
  auto const quantizer = static_cast<std::uint16_t>(record->quantizer);
  auto const padded_n = static_cast<std::uint32_t>(qw38::cuda::decode_pad_n(kVocab));
  auto const padded_k = qw38::cuda::decode_pad_k(kHidden);
  auto const codes_bytes = qw38::cuda::decode_code_bytes(layout, padded_n, padded_k);
  auto const scales_bytes = qw38::cuda::decode_scale_bytes(layout, padded_n, padded_k);
  if (record->shape.rank != 2 || record->shape.logical[0] != kVocab ||
      record->shape.logical[1] != kHidden ||
      record->shape.padded[0] != padded_n || record->shape.padded[1] != padded_k ||
      record->payload.length != codes_bytes || record->scales.length != scales_bytes ||
      record->storage != StorageClass::Int8Grouped ||
      layout != qw38::cuda::kDecodeLayoutQ8G32CandidateV1 ||
      quantizer != qw38::cuda::kDecodeQuantizerQ8G32CandidateV1) {
    return std::unexpected(make_error(ErrorCode::MalformedArtifact, "lm_head",
                                      "candidate head representation mismatch"));
  }
  return qw38::cuda::PrefillWeight{
      .codes = data->pointer, .scales = scales->pointer,
      .layout = layout, .quantizer = quantizer, .n = kVocab, .k = kHidden,
      .padded_n = padded_n, .padded_k = padded_k,
      .codes_bytes = codes_bytes, .scales_bytes = scales_bytes};
}

std::expected<void, Error> execute_prefill_mlp(
    PrefillLayerProjectionPlan const& plan, qw38::cuda::PrefillEngine& engine,
    float const* h_mid, float* next_h, std::uint32_t valid_tokens,
    std::uint64_t first_position, float eps) {
  auto st = engine.mlp(plan.mlp_gate, plan.mlp_up, plan.mlp_down,
                       h_mid, plan.mlp_gamma, eps, next_h,
                       valid_tokens, first_position);
  if (!st) return std::unexpected(from_cuda(st.error()));
  return {};
}

}  // namespace qw38::runtime
