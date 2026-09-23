#include "runtime/language_model.hpp"

#include "cuda/activation.hpp"
#include "cuda/copy.hpp"
#include "format/constants.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>

namespace qw38::runtime {
namespace {

using qw38::format::SemanticNodeKind;
using qw38::format::TensorRole;
using qw38::format::PhysicalLayoutId;
using qw38::format::StorageClass;

std::expected<qw38::format::TensorRecord const*, Error> required_tensor(
    Model const& model, std::string_view name, std::uint32_t instance,
    SemanticNodeKind kind, TensorRole role) {
  auto const* record = model.find_tensor(name);
  if (record == nullptr) {
    return std::unexpected(make_error(ErrorCode::MalformedArtifact, "tensor",
                                      "required language tensor is missing"));
  }
  std::uint32_t matches = 0;
  for (auto const& binding : model.graph_bindings()) {
    if (binding.tensor_id == record->tensor_id) {
      if (binding.instance_id != instance || binding.kind != kind ||
          binding.role != role ||
          binding.layer_index != qw38::format::kNoLayerIndex) {
        return std::unexpected(make_error(ErrorCode::MalformedArtifact,
                                          "graph_bindings",
                                          "language output tensor has a wrong binding"));
      }
      ++matches;
    }
  }
  if (matches != 1) {
    return std::unexpected(make_error(ErrorCode::MalformedArtifact,
                                      "graph_bindings",
                                      "language output binding is missing or duplicated"));
  }
  return record;
}

bool vector_bf16(qw38::format::TensorRecord const& record,
                 std::uint64_t extent, PhysicalLayoutId layout) {
  return record.storage == StorageClass::Bf16 && record.shape.rank == 1 &&
         record.shape.logical[0] == extent && record.layout == layout;
}

}  // namespace

std::expected<void, Error> validate_primary_language_graph(
    qw38::format::ArtifactSchema const& schema) {
  if (schema.scope !=
      qw38::format::SemanticScope::LanguagePlusMtpDescriptors) {
    return std::unexpected(make_error(ErrorCode::MalformedArtifact, "scope",
                                      "language artifact must retain disabled MTP descriptors"));
  }
  std::array<bool, 130> instances{};
  for (auto const& binding : schema.graph_bindings) {
    if (binding.instance_id >= instances.size()) continue;
    auto const id = binding.instance_id;
    auto const expected_kind = id == 0 ? SemanticNodeKind::Embed
        : id == 129 ? SemanticNodeKind::LmHead
        : (id % 2 == 0 ? SemanticNodeKind::Mlp
           : (is_gdn_language_layer((id - 1) / 2)
                  ? SemanticNodeKind::GatedDeltaNet
                  : SemanticNodeKind::GatedAttention));
    auto const expected_layer = id == 0 || id == 129
        ? qw38::format::kNoLayerIndex : (id - 1) / 2;
    if (binding.kind != expected_kind ||
        binding.layer_index != expected_layer) {
      return std::unexpected(make_error(ErrorCode::MalformedArtifact,
                                        "graph_bindings",
                                        "language graph instance is out of order"));
    }
    instances[id] = true;
  }
  for (bool present : instances) {
    if (!present) {
      return std::unexpected(make_error(ErrorCode::MalformedArtifact,
                                        "graph_bindings",
                                        "language graph instance is missing"));
    }
  }
  return {};
}

std::expected<LanguageModelPlan, Error> LanguageModelPlan::bind(
    Model const& model, Session& session, qw38::cuda::Stream const& stream) {
  try {
    auto const* session_stream = detail::SessionPlanAccess::stream(session);
    auto* state = detail::SessionPlanAccess::execution_state(session);
    if (session_stream == nullptr || session_stream->empty() || stream.empty() ||
        session_stream->native() != stream.native() ||
        model.device() != stream.device() || state == nullptr ||
        state->is_poisoned()) {
      return std::unexpected(make_error(ErrorCode::InvalidArgument, "session",
                                        "model, session and stream must be open on one device"));
    }
    if (auto st = validate_primary_language_graph(model.schema()); !st) {
      return std::unexpected(st.error());
    }

    auto embed = required_tensor(model, "model.language_model.embed_tokens.weight",
                                 0, SemanticNodeKind::Embed,
                                 TensorRole::EmbeddingTable);
    auto norm = required_tensor(model, "model.language_model.norm.weight",
                                129, SemanticNodeKind::LmHead,
                                TensorRole::FinalLanguageNorm);
    auto head = required_tensor(model, "lm_head.weight", 129,
                                SemanticNodeKind::LmHead,
                                TensorRole::LmHeadWeight);
    if (!embed) return std::unexpected(embed.error());
    if (!norm) return std::unexpected(norm.error());
    if (!head) return std::unexpected(head.error());
    if ((*embed)->storage != StorageClass::Bf16 ||
        (*embed)->layout != PhysicalLayoutId::CudaBf16RowMajorV0 ||
        (*embed)->shape.rank != 2 || (*embed)->shape.logical[0] != kVocab ||
        (*embed)->shape.logical[1] != kHidden ||
        !vector_bf16(**norm, kHidden, PhysicalLayoutId::CudaBf16VectorV0) ||
        (*head)->shape.rank != 2 || (*head)->shape.logical[0] != kVocab ||
        (*head)->shape.logical[1] != kHidden) {
      return std::unexpected(make_error(ErrorCode::MalformedArtifact, "shape",
                                        "language embedding, norm or head shape is invalid"));
    }
    auto embed_data = model.payload((*embed)->tensor_id);
    auto norm_data = model.payload((*norm)->tensor_id);
    auto head_data = model.payload((*head)->tensor_id);
    if (!embed_data) return std::unexpected(embed_data.error());
    if (!norm_data) return std::unexpected(norm_data.error());
    if (!head_data) return std::unexpected(head_data.error());
    auto normalized = session.scratch(qw38::format::ScratchKind::NormalizedHidden);
    auto logits = session.scratch(qw38::format::ScratchKind::Logits);
    if (!normalized) return std::unexpected(normalized.error());
    if (!logits) return std::unexpected(logits.error());
    if (normalized->bytes < kHidden * 2u ||
        logits->bytes < kLogitsBytesPerToken) {
      return std::unexpected(make_error(ErrorCode::InvalidCapacity, "scratch",
                                        "language output scratch is too small"));
    }

    LanguageModelPlan plan;
    plan.layers_.reserve(kLanguageLayers);
    for (std::uint32_t layer = 0; layer < kLanguageLayers; ++layer) {
      auto bound = LanguageLayerPlan::bind(model, session, layer, stream);
      if (!bound) return std::unexpected(bound.error());
      plan.layers_.push_back(std::move(*bound));
    }
    plan.logits_.resize(kVocab);
    plan.session_ = &session;
    plan.stream_ = &stream;
    plan.state_ = state;
    plan.embedding_ = static_cast<std::uint16_t const*>(embed_data->pointer);
    plan.final_gamma_ = static_cast<std::uint16_t const*>(norm_data->pointer);
    plan.normalized_ = static_cast<std::uint16_t*>(normalized->region[0].tensor.pointer);
    plan.residual_ = static_cast<float*>(session.residual_h().pointer);
    plan.device_logits_ = static_cast<float*>(logits->region[0].tensor.pointer);

    auto const layout = static_cast<std::uint16_t>((*head)->layout);
    bool const q8 = layout == qw38::cuda::kDecodeLayoutQ8G32V0 &&
                    (*head)->storage == StorageClass::Int8Grouped;
    bool const bf16 = layout == qw38::cuda::kDecodeLayoutBf16DenseTileV0 &&
                      (*head)->storage == StorageClass::Bf16;
    if (!q8 && !bf16) {
      return std::unexpected(make_error(ErrorCode::MalformedArtifact, "lm_head",
                                        "head must be Q8G32 or BF16 identity"));
    }
    auto& d = plan.head_;
    d.layout = layout;
    d.quantizer = static_cast<std::uint16_t>((*head)->quantizer);
    d.n = kVocab;
    d.k = kHidden;
    d.padded_n = static_cast<std::uint32_t>(qw38::cuda::decode_pad_n(d.n));
    d.padded_k = qw38::cuda::decode_pad_k(d.k);
    auto const codes_bytes = qw38::cuda::decode_code_bytes(layout, d.padded_n,
                                                            d.padded_k);
    auto const scales_bytes = qw38::cuda::decode_scale_bytes(layout, d.padded_n,
                                                              d.padded_k);
    if ((*head)->payload.length != codes_bytes ||
        (*head)->scales.length != scales_bytes) {
      return std::unexpected(make_error(ErrorCode::MalformedArtifact, "lm_head",
                                        "head payload size does not match layout"));
    }
    d.codes = qw38::cuda::decode_matrix_view(
        const_cast<void*>(head_data->pointer),
        q8 ? qw38::cuda::DecodeDtype::Q8 : qw38::cuda::DecodeDtype::Bf16,
        layout, d.n, d.k, d.padded_n, d.padded_k, codes_bytes, 16);
    if (q8) {
      auto scales = model.scales((*head)->tensor_id);
      if (!scales) return std::unexpected(scales.error());
      d.scales = qw38::cuda::decode_matrix_view(
          const_cast<void*>(scales->pointer), qw38::cuda::DecodeDtype::Fp16,
          layout, d.padded_n, d.padded_k / 32u, d.padded_n,
          d.padded_k / 32u, scales_bytes, 2);
    }
    d.input = qw38::cuda::decode_vector_view(
        plan.normalized_, qw38::cuda::DecodeDtype::Bf16,
        qw38::cuda::kDecodeLayoutBf16VectorV0, d.k,
        static_cast<std::uint64_t>(d.k) * 2u, 2, false);
    d.output = qw38::cuda::decode_vector_view(
        plan.device_logits_, qw38::cuda::DecodeDtype::Fp32,
        qw38::cuda::kDecodeLayoutFp32VectorV0, d.n,
        static_cast<std::uint64_t>(d.n) * 4u, 4, true);
    d.epilogue = qw38::cuda::DecodeEpilogue::StoreFp32;
    return plan;
  } catch (std::bad_alloc const&) {
    return std::unexpected(make_error(ErrorCode::AllocationFailed,
                                      "language_model.bind", "host allocation failed"));
  } catch (std::length_error const&) {
    return std::unexpected(make_error(ErrorCode::AllocationFailed,
                                      "language_model.bind", "host allocation size is invalid"));
  }
}

std::expected<DecodeResult, Error> LanguageModelPlan::decode_token(
    std::uint32_t token_id, std::uint64_t position) {
  if (state_ == nullptr || state_->is_poisoned()) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "session",
                                      "session is poisoned or closed"));
  }
  if (token_id >= kVocab) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "token_id",
                                      "token ID exceeds language vocabulary"));
  }
  if (position >= session_->kv_capacity() ||
      position > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
    return std::unexpected(make_error(ErrorCode::InvalidCapacity, "position",
                                      "position exceeds session or RoPE capacity"));
  }
  if (position != state_->token_position() || !state_->layers_at(position)) {
    return std::unexpected(make_error(ErrorCode::InvalidPopulatedLength,
                                      "position", "session is not at a complete token boundary"));
  }
  auto fail = [&](Error error) -> std::expected<DecodeResult, Error> {
    state_->poison();
    return std::unexpected(std::move(error));
  };
  if (auto st = qw38::cuda::launch_embed_gather(
          embedding_, kVocab, token_id, residual_, *stream_); !st) {
    return fail(from_cuda(st.error()));
  }
  for (auto const& layer : layers_) {
    auto result = execute_decode_language_layer(layer, position);
    if (!result) return fail(result.error());
  }
  if (!state_->layers_at(position + 1u)) {
    return fail(make_error(ErrorCode::Internal, "position",
                           "a language layer did not advance its state"));
  }
  if (auto st = qw38::cuda::launch_hidden_rms(
          residual_, final_gamma_, kMlpRmsEps, 1, normalized_, *stream_); !st) {
    return fail(from_cuda(st.error()));
  }
  if (auto st = qw38::cuda::launch_decode_mmv(head_, *stream_); !st) {
    return fail(from_cuda(st.error()));
  }
  if (auto st = qw38::cuda::copy_d2h(logits_.data(), device_logits_,
                                     kLogitsBytesPerToken, *stream_); !st) {
    return fail(from_cuda(st.error()));
  }
  if (auto st = stream_->sync(); !st) return fail(from_cuda(st.error()));
  std::uint32_t best = 0;
  for (std::uint32_t i = 0; i < kVocab; ++i) {
    if (!std::isfinite(logits_[i])) {
      return fail(make_error(ErrorCode::Internal, "logits",
                             "language head produced a non-finite logit"));
    }
    if (logits_[i] > logits_[best]) best = i;
  }
  state_->commit_token();
  return DecodeResult{.logits = logits_, .argmax = best};
}

std::expected<DecodeResult, Error> LanguageModelPlan::setup_prompt_slow(
    std::span<std::uint32_t const> token_ids) {
  if (token_ids.empty()) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "prompt",
                                      "prompt must contain a token"));
  }
  std::expected<DecodeResult, Error> result = std::unexpected(
      make_error(ErrorCode::Internal, "prompt"));
  for (auto id : token_ids) {
    result = decode_token(id, state_->token_position());
    if (!result) return result;
  }
  return result;
}

}  // namespace qw38::runtime
