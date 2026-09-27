#include "runtime/language_model.hpp"
#include "runtime/submission.hpp"
#include "cuda/attention.hpp"

#include "cuda/activation.hpp"
#include "cuda/copy.hpp"
#include "format/constants.hpp"
#include "runtime/profiling.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <limits>
#include <new>
#include <stdexcept>
#include <variant>

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
    plan.prefill_layers_.reserve(kLanguageLayers);
    for (std::uint32_t layer = 0; layer < kLanguageLayers; ++layer) {
      if (is_gdn_language_layer(layer)) {
        auto bound = bind_prefill_gdn_layer(model, session, layer, stream);
        if (!bound) return std::unexpected(bound.error());
        plan.prefill_layers_.emplace_back(std::move(*bound));
      } else {
        auto bound = bind_prefill_attention_layer(model, session, layer, stream);
        if (!bound) return std::unexpected(bound.error());
        plan.prefill_layers_.emplace_back(std::move(*bound));
      }
    }
    auto readback = qw38::cuda::HostBuffer::allocate(kLogitsBytesPerToken +
                                                    sizeof(qw38::cuda::DecodeControl) + sizeof(int));
    if (!readback) return std::unexpected(from_cuda(readback.error()));
    plan.readback_ = std::move(*readback);
    auto failure = qw38::cuda::DeviceBuffer::allocate(sizeof(int), stream.device());
    if (!failure) return std::unexpected(from_cuda(failure.error()));
    plan.unit_failure_ = std::move(*failure);
    plan.logits_ = {static_cast<float*>(plan.readback_.data()), kVocab};
    plan.model_ = &model;
    plan.stream_ = &stream;
    plan.state_ = state;
    plan.embedding_ = static_cast<std::uint16_t const*>(embed_data->pointer);
    plan.final_gamma_ = static_cast<std::uint16_t const*>(norm_data->pointer);
    plan.normalized_ = static_cast<std::uint16_t*>(normalized->region[0].tensor.pointer);
    plan.residual_ = static_cast<float*>(session.residual_h().pointer);
    plan.residual_mid_ = static_cast<float*>(session.residual_h_mid().pointer);
    plan.device_logits_ = static_cast<float*>(logits->region[0].tensor.pointer);
    plan.kv_capacity_ = session.kv_capacity();

    auto const layout = static_cast<std::uint16_t>((*head)->layout);
    bool const q8 = (layout == qw38::cuda::kDecodeLayoutQ8G32V0 ||
                     layout == qw38::cuda::kDecodeLayoutQ8G32CandidateV1) &&
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

std::expected<qw38::cuda::DecodeControl, Error> LanguageModelPlan::pending(
    std::uint32_t token, std::uint64_t position, std::uint32_t count) const {
  if (!state_ || state_->is_poisoned() || state_->busy_ || !stream_ || stream_->empty() ||
      layers_.size() != kLanguageLayers)
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "session",
                                      "session or plan is poisoned, moved or closed"));
  if (auto st = qw38::cuda::require_uncaptured(*stream_); !st)
    return std::unexpected(from_cuda(st.error()));
  if (token >= kVocab)
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "token_id",
                                      "token ID exceeds language vocabulary"));
  if (!count || position >= kv_capacity_ || count > kv_capacity_ - position ||
      position > std::numeric_limits<std::int32_t>::max() ||
      count - 1u > std::numeric_limits<std::int32_t>::max() - position)
    return std::unexpected(make_error(ErrorCode::InvalidCapacity, "position",
                                      "position exceeds session or RoPE capacity"));
  if (position != state_->token_position() || !state_->layers_at(position))
    return std::unexpected(make_error(ErrorCode::InvalidPopulatedLength, "position",
                                      "session is not at a complete token boundary"));
  qw38::cuda::DecodeControl result{.position = position,
                                   .populated = position + count, .token = token};
  for (std::size_t i = 0; i < state_->conv_cursor.size(); ++i) {
    auto c = state_->conv_cursor[i];
    if (c >= kConvTaps)
      return std::unexpected(make_error(ErrorCode::InvalidArgument, "cursor",
                                        "convolution cursor exceeds ring"));
    result.cursor[i] = c;
  }
  return result;
}

std::expected<void, Error> LanguageModelPlan::enqueue_decode(
    qw38::cuda::DecodeControl const& p, qw38::cuda::DecodeControl const* controls,
    std::uint64_t bucket) {
  if (auto st = qw38::cuda::zero(unit_failure_, *stream_); !st)
    return std::unexpected(from_cuda(st.error()));
  if (auto st = qw38::cuda::launch_embed_gather(
          embedding_, kVocab, p.token, residual_, *stream_, controls); !st)
    return std::unexpected(from_cuda(st.error()));
  for (auto const& layer : layers_) {
    auto result = detail::Submission::layer(layer, p, controls, bucket,
        static_cast<int*>(unit_failure_.data()));
    if (!result) return std::unexpected(result.error());
  }
  if (auto st = qw38::cuda::launch_hidden_rms(
          residual_, final_gamma_, kMlpRmsEps, 1, normalized_, *stream_); !st)
    return std::unexpected(from_cuda(st.error()));
  if (auto st = qw38::cuda::launch_decode_mmv(head_, *stream_); !st)
    return std::unexpected(from_cuda(st.error()));
  return {};
}

std::expected<void, Error> LanguageModelPlan::complete_unit() {
  auto* host = reinterpret_cast<int*>(static_cast<std::byte*>(readback_.data()) +
      kLogitsBytesPerToken + sizeof(qw38::cuda::DecodeControl));
  if (auto st = qw38::cuda::copy_d2h(host, unit_failure_.data(), sizeof(int), *stream_); !st)
    return std::unexpected(from_cuda(st.error()));
  if (auto st = stream_->sync(); !st) return std::unexpected(from_cuda(st.error()));
  if (*host)
    return std::unexpected(from_cuda(qw38::cuda::make_error(
        qw38::cuda::ErrorCode::InvalidArgument, "q8.producer",
        "nonfinite producer or invalid Q8 scale")));
  return {};
}

std::expected<void, Error> LanguageModelPlan::prepare_graph(
    qw38::cuda::DecodeControl const& p) {
  auto const bucket = std::min(kv_capacity_,
      std::bit_ceil(std::max<std::uint64_t>(256, p.populated)));
  if (!graph_.empty() && graph_bucket_ == bucket) return {};
  // Calls occur only at completed token boundaries. No warmup inference.
  graph_ = {};
  graph_bucket_ = 0;
  if (controls_.empty()) {
    auto buffer = qw38::cuda::DeviceBuffer::allocate(sizeof(p), stream_->device());
    if (!buffer) return std::unexpected(from_cuda(buffer.error()));
    controls_ = std::move(*buffer);
  }
  auto guard = stream_->activate();
  if (!guard) return std::unexpected(from_cuda(guard.error()));
  if (auto resources = qw38::cuda::attention_decode_resources(); !resources)
    return std::unexpected(from_cuda(resources.error()));
  auto captured = qw38::cuda::Graph::begin(*stream_);
  if (!captured) return std::unexpected(from_cuda(captured.error()));
  auto enqueued = enqueue_decode(p,
      static_cast<qw38::cuda::DecodeControl const*>(controls_.data()), bucket);
  auto finished = captured->finish(); // End even an invalidated capture.
  if (!enqueued) {
    // Capture recorded work only; no persistent bytes were mutated.
    state_->recover();
    return std::unexpected(enqueued.error());
  }
  if (!finished) return std::unexpected(from_cuda(finished.error()));
  graph_ = std::move(*captured);
  graph_bucket_ = bucket;
  ++graph_builds_;
  return {};
}

std::expected<DecodeResult, Error> LanguageModelPlan::decode_token(
    std::uint32_t token_id, std::uint64_t position) {
  profiling::ScopedRange token_range("decode_token");
  auto p = pending(token_id, position, 1);
  if (!p) return std::unexpected(p.error());
  char detail[96] = "decode_token_detail";
  if (profiling::enabled) {
    auto const bucket = std::min(kv_capacity_,
        std::bit_ceil(std::max<std::uint64_t>(256, p->populated)));
    auto const graph = submission_ != DecodeSubmission::Graph || graph_failure_
        ? "eager"
        : graph_.empty() || graph_bucket_ != bucket ? "build" : "replay";
    std::snprintf(detail, sizeof(detail), "decode_token position=%llu bucket=%llu graph=%s",
        static_cast<unsigned long long>(position),
        static_cast<unsigned long long>(bucket), graph);
  }
  profiling::ScopedRange detail_range(detail);
  auto fail = [&](Error error) -> std::expected<DecodeResult, Error> {
    state_->poison();
    (void)stream_->sync(); // Drain before any borrowed/owned staging may die.
    return std::unexpected(std::move(error));
  };
  SessionExecutionState::Unit unit(*state_);
  bool replay = submission_ == DecodeSubmission::Graph && !graph_failure_;
  if (replay) {
    if (auto st = prepare_graph(*p); !st) {
      graph_failure_ = st.error();
      replay = false; // Construction failed before execution; eager is safe.
    }
  }
  if (replay) {
    auto* host = reinterpret_cast<qw38::cuda::DecodeControl*>(
        static_cast<std::byte*>(readback_.data()) + kLogitsBytesPerToken);
    *host = *p;
    if (auto st = qw38::cuda::copy_h2d(controls_.data(), host, sizeof(*host), *stream_); !st)
      return fail(from_cuda(st.error()));
    if (auto st = graph_.launch(*stream_); !st) return fail(from_cuda(st.error()));
    ++graph_replays_;
  } else {
    if (auto st = enqueue_decode(*p, nullptr, 0); !st) return fail(st.error());
  }
  if (auto st = qw38::cuda::copy_d2h(logits_.data(), device_logits_,
                                     kLogitsBytesPerToken, *stream_); !st)
    return fail(from_cuda(st.error()));
  if (auto st = complete_unit(); !st) return fail(st.error());
  std::uint32_t best = 0;
  for (std::uint32_t i = 0; i < kVocab; ++i) {
    if (!std::isfinite(logits_[i]))
      return fail(make_error(ErrorCode::Internal, "logits",
                             "language head produced a non-finite logit"));
    if (logits_[i] > logits_[best]) best = i;
  }
  state_->commit_unit(1);
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

std::expected<void, Error> LanguageModelPlan::initialize_prefill() {
  if (prefill_) return {};
  try {
    PrefillState state;
    constexpr auto capacity = static_cast<std::uint32_t>(kArenaTokenCapacity);
    auto engine = qw38::cuda::PrefillEngine::create(*stream_, capacity,
        qw38::cuda::kPrefillDefaultWeightRows, qw38::cuda::PrefillDispatch::BoundedUnpackBf16Cublas,
        model_->schema().precision.id == qw38::format::PrecisionPolicyId::Fp8MixerV1 ||
        model_->schema().precision.id == qw38::format::PrecisionPolicyId::Fp8MixerQ8MlpV1);
    if (!engine) return std::unexpected(from_cuda(engine.error()));
    state.engine = std::move(*engine);
    auto gdn = PrefillGdnWorkspace::create(capacity, stream_->device());
    if (!gdn) return std::unexpected(gdn.error());
    state.gdn_workspace = std::move(*gdn);
    auto attention = PrefillAttentionWorkspace::create(capacity, stream_->device());
    if (!attention) return std::unexpected(attention.error());
    state.attention_workspace = std::move(*attention);
    auto ids = qw38::cuda::DeviceBuffer::allocate(
        static_cast<std::uint64_t>(capacity) * sizeof(std::uint32_t),
        stream_->device());
    if (!ids) return std::unexpected(from_cuda(ids.error()));
    state.token_ids = std::move(*ids);
    auto next = qw38::cuda::DeviceBuffer::allocate(
        static_cast<std::uint64_t>(capacity) * kHidden * sizeof(float),
        stream_->device());
    if (!next) return std::unexpected(from_cuda(next.error()));
    state.next_h = std::move(*next);
    auto head = bind_prefill_head(*model_, *stream_);
    if (!head) return std::unexpected(head.error());
    state.head = *head;
    prefill_.emplace(std::move(state));
    return {};
  } catch (std::bad_alloc const&) {
    return std::unexpected(make_error(ErrorCode::AllocationFailed,
                                      "language_model.prefill", "host allocation failed"));
  }
}

std::expected<DecodeResult, Error> LanguageModelPlan::prefill_tokens(
    std::span<std::uint32_t const> token_ids,
    std::span<std::uint64_t const> requested_rows,
    LogitRowSink const& sink) {
  if (!state_ || state_->is_poisoned() || state_->busy_ || !stream_ || stream_->empty()) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "session",
                                      "session is poisoned or closed"));
  }
  auto const first = state_->token_position();
  if (token_ids.empty() || first >= kv_capacity_ ||
      token_ids.size() > kv_capacity_ - first ||
      first > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
      token_ids.size() - 1u >
          static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) - first ||
      !state_->layers_at(first)) {
    return std::unexpected(make_error(ErrorCode::InvalidCapacity, "prefill",
                                      "invalid token range or session position"));
  }
  for (auto id : token_ids) {
    if (id >= kVocab)
      return std::unexpected(make_error(ErrorCode::InvalidArgument, "token_id",
                                        "token ID exceeds language vocabulary"));
  }
  if (!requested_rows.empty() && !sink)
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.rows",
                                      "requested rows require a sink"));
  for (std::size_t i = 0; i < requested_rows.size(); ++i) {
    if (requested_rows[i] < first ||
        requested_rows[i] - first >= token_ids.size() ||
        (i && requested_rows[i] <= requested_rows[i - 1]))
      return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.rows",
                                        "requested rows must be unique and increasing"));
  }
  auto initial_pending = pending(token_ids.front(), first,
      static_cast<std::uint32_t>(token_ids.size()));
  if (!initial_pending) return std::unexpected(initial_pending.error());
  if (auto st = initialize_prefill(); !st) return std::unexpected(st.error());
  auto fail = [&](Error error) -> std::expected<DecodeResult, Error> {
    state_->poison();
    (void)stream_->sync();
    return std::unexpected(std::move(error));
  };
  SessionExecutionState::Unit unit(*state_);
  auto& p = *prefill_;
  std::size_t row_index = 0;
  std::uint32_t best = 0;
  for (std::size_t offset = 0; offset < token_ids.size();) {
    auto const count = static_cast<std::uint32_t>(std::min<std::size_t>(
        kArenaTokenCapacity, token_ids.size() - offset));
    auto const position = first + offset;
    auto chunk_pending = *initial_pending;
    chunk_pending.position = position;
    chunk_pending.populated = position + count;
    for (auto& cursor : chunk_pending.cursor)
      cursor = (cursor + offset % kConvTaps) % kConvTaps;
    if (auto st = qw38::cuda::zero(unit_failure_, *stream_); !st)
      return fail(from_cuda(st.error()));
    if (auto st = qw38::cuda::copy_h2d(p.token_ids.data(), token_ids.data() + offset,
                                       count * sizeof(std::uint32_t), *stream_); !st)
      return fail(from_cuda(st.error()));
    auto* residual = residual_;
    auto* h_mid = residual_mid_;
    auto* next = static_cast<float*>(p.next_h.data());
    if (auto st = qw38::cuda::launch_embed_gather_chunk(
            embedding_, kVocab, static_cast<std::uint32_t const*>(p.token_ids.data()),
            count, residual, *stream_); !st)
      return fail(from_cuda(st.error()));
    for (auto& layer : prefill_layers_) {
      std::expected<void, Error> status;
      if (auto* gdn = std::get_if<PrefillGdnLayerPlan>(&layer))
        status = detail::Submission::prefill_gdn(*gdn, p.gdn_workspace, p.engine,
                                           residual, h_mid, next, count, position, 64, false, static_cast<int*>(unit_failure_.data()), &chunk_pending);
      else
        status = detail::Submission::prefill_attention(
            std::get<PrefillAttentionLayerPlan>(layer), p.attention_workspace,
            p.engine, residual, h_mid, next, count, position, false, static_cast<int*>(unit_failure_.data()));
      if (!status) return fail(status.error());
      auto* old = residual;
      residual = next;
      next = h_mid;
      h_mid = old;
    }
    bool completed = false;
    auto readout = [&](std::uint32_t row) -> std::expected<std::uint32_t, Error> {
      if (auto st = qw38::cuda::launch_hidden_rms(
              residual + static_cast<std::uint64_t>(row) * kHidden,
              final_gamma_, kMlpRmsEps, 1, normalized_, *stream_); !st)
        return std::unexpected(from_cuda(st.error()));
      if (auto st = p.engine.head_generation(p.head, normalized_, 1,
                                             position + row, device_logits_); !st)
        return std::unexpected(from_cuda(st.error()));
      if (auto st = qw38::cuda::copy_d2h(logits_.data(), device_logits_,
                                         kLogitsBytesPerToken, *stream_); !st)
        return std::unexpected(from_cuda(st.error()));
      if (auto st = complete_unit(); !st) return std::unexpected(st.error());
      completed = true;
      std::uint32_t argmax = 0;
      for (std::uint32_t i = 0; i < kVocab; ++i) {
        if (!std::isfinite(logits_[i]))
          return std::unexpected(make_error(ErrorCode::Internal, "logits",
                                            "language head produced a non-finite logit"));
        if (logits_[i] > logits_[argmax]) argmax = i;
      }
      return argmax;
    };
    bool final_readout = false;
    while (row_index < requested_rows.size() &&
           requested_rows[row_index] < position + count) {
      auto const row = static_cast<std::uint32_t>(requested_rows[row_index] - position);
      auto result = readout(row);
      if (!result) return fail(result.error());
      std::expected<void, Error> delivered;
      try {
        delivered = sink(requested_rows[row_index], logits_);
      } catch (std::bad_alloc const&) {
        return fail(make_error(ErrorCode::AllocationFailed, "prefill.rows",
                               "requested-row sink allocation failed"));
      } catch (std::exception const&) {
        return fail(make_error(ErrorCode::Internal, "prefill.rows",
                               "requested-row sink threw"));
      } catch (...) {
        return fail(make_error(ErrorCode::Internal, "prefill.rows",
                               "requested-row sink threw"));
      }
      if (!delivered) return fail(delivered.error());
      if (row == count - 1u) {
        best = *result;
        final_readout = true;
      }
      ++row_index;
    }
    if (offset + count == token_ids.size() && !final_readout) {
      auto result = readout(count - 1u);
      if (!result) return fail(result.error());
      best = *result;
    }
    if (!completed) {
      if (auto st = complete_unit(); !st) return fail(st.error());
    }
    state_->commit_unit(count);
    offset += count;
  }
  return DecodeResult{.logits = logits_, .argmax = best};
}

}  // namespace qw38::runtime
