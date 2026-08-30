#include "scalar_runtime.h"

#include <limits>
#include <utility>

namespace qw38::internal {
namespace {

template <typename T>
T* offset(std::vector<T>* values, std::size_t index) noexcept {
  return values->data() + index;
}

bool valid_parameter_storage(const ScalarModelParameters& parameters) noexcept {
  return parameters.prepared_layers == kModelLayerCount &&
         parameters.input_norms.size() ==
             kModelLayerCount * kResidualWidth &&
         parameters.ffn_norms.size() == kModelLayerCount * kResidualWidth &&
         parameters.gdn_convolution.size() ==
             kGdnLayerCount * kGdnConvolutionValues &&
         parameters.gdn_folded_a.size() ==
             kGdnLayerCount * kGdnGateCount &&
         parameters.gdn_dt_bias.size() ==
             kGdnLayerCount * kGdnGateCount &&
         parameters.gdn_recurrent_norm.size() ==
             kGdnLayerCount * kGdnHeadWidth &&
         parameters.attention_query_norm.size() ==
             kAttentionLayerCount * kAttentionHeadWidth &&
         parameters.attention_key_norm.size() ==
             kAttentionLayerCount * kAttentionHeadWidth &&
         parameters.output_norm.size() == kResidualWidth;
}

bool valid_state_storage(const ScalarSessionState& state) noexcept {
  if (state.capacity == 0 || state.frontier >= state.capacity ||
      state.capacity > std::numeric_limits<std::size_t>::max() /
                           (kAttentionLayerCount * kAttentionKvWidth)) {
    return false;
  }
  const std::size_t attention_values =
      kAttentionLayerCount * state.capacity * kAttentionKvWidth;
  return state.gdn_convolution.size() ==
             kGdnLayerCount * kGdnConvolutionValues &&
         state.gdn_recurrent.size() ==
             kGdnLayerCount * kGdnRecurrentStateValues &&
         state.attention_key.size() == attention_values &&
         state.attention_value.size() == attention_values;
}

bool valid_workspace_storage(const ScalarWorkspace& workspace,
                             std::size_t capacity) noexcept {
  return workspace.capacity == capacity &&
         workspace.activation_a.size() == kResidualWidth &&
         workspace.activation_b.size() == kResidualWidth &&
         workspace.post_mixer.size() == kResidualWidth &&
         workspace.gdn_normalized.size() == kResidualWidth &&
         workspace.gdn_packed.size() == kGdnPackedQkvWidth &&
         workspace.gdn_projected_gate.size() == kGdnValueWidth &&
         workspace.gdn_projected_alpha.size() == kGdnGateCount &&
         workspace.gdn_projected_beta.size() == kGdnGateCount &&
         workspace.gdn_convolved.size() == kGdnPackedQkvWidth &&
         workspace.gdn_query.size() == kGdnKeyWidth &&
         workspace.gdn_key.size() == kGdnKeyWidth &&
         workspace.gdn_value_tiled.size() == kGdnValueWidth &&
         workspace.gdn_value_grouped.size() == kGdnValueWidth &&
         workspace.gdn_gate_grouped.size() == kGdnValueWidth &&
         workspace.gdn_alpha_grouped.size() == kGdnGateCount &&
         workspace.gdn_beta_grouped.size() == kGdnGateCount &&
         workspace.gdn_folded_grouped.size() == kGdnGateCount &&
         workspace.gdn_dt_grouped.size() == kGdnGateCount &&
         workspace.gdn_log_decay.size() == kGdnGateCount &&
         workspace.gdn_update_gate.size() == kGdnGateCount &&
         workspace.gdn_recurrent_output.size() == kGdnValueWidth &&
         workspace.gdn_gated_grouped.size() == kGdnValueWidth &&
         workspace.gdn_gated_tiled.size() == kGdnValueWidth &&
         workspace.gdn_mixer_output.size() == kResidualWidth &&
         workspace.attention_normalized.size() == kResidualWidth &&
         workspace.attention_packed.size() ==
             kAttentionPackedQueryGateWidth &&
         workspace.attention_query.size() == kAttentionQueryWidth &&
         workspace.attention_gate.size() == kAttentionQueryWidth &&
         workspace.attention_key.size() == kAttentionKvWidth &&
         workspace.attention_value.size() == kAttentionKvWidth &&
         workspace.attention_output.size() == kAttentionQueryWidth &&
         workspace.attention_scores.size() == capacity &&
         workspace.attention_mixer_output.size() == kResidualWidth &&
#ifdef QW38_DIAGNOSTIC_TRACE
         workspace.attention_rope_query.size() == kAttentionQueryWidth &&
         workspace.attention_rope_key.size() == kAttentionKvWidth &&
#endif
         workspace.ffn_normalized.size() == kResidualWidth &&
         workspace.ffn_gate.size() == kFfnWidth &&
         workspace.ffn_up.size() == kFfnWidth &&
         workspace.ffn_activated.size() == kFfnWidth &&
         workspace.ffn_correction.size() == kResidualWidth &&
         workspace.final_normalized.size() == kResidualWidth;
}

#ifdef QW38_DIAGNOSTIC_TRACE
Status offer_trace(const TraceFilter* filter, TraceSink sink, void* context,
                   const char* name, std::size_t layer, const float* values,
                   std::size_t count, std::size_t first,
                   std::size_t second = 0,
                   std::size_t third = 0) noexcept {
  if (filter == nullptr) return Status::ok();
  const std::size_t rank = third != 0 ? 3 : (second != 0 ? 2 : 1);
  return emit_trace_tensor(*filter, sink, context,
                           {name, layer, values, count, {first, second, third},
                            rank});
}
#endif

}  // namespace

Status prepare_scalar_model_parameters(
    const ModelWeights& weights, ScalarModelParameters* parameters) noexcept {
  if (parameters == nullptr) {
    return {StatusCode::kInvalidArgument,
            "scalar model parameter output is required"};
  }
  ScalarModelParameters candidate;
  candidate.input_norms.resize(kModelLayerCount * kResidualWidth);
  candidate.ffn_norms.resize(kModelLayerCount * kResidualWidth);
  candidate.gdn_convolution.resize(kGdnLayerCount * kGdnConvolutionValues);
  candidate.gdn_folded_a.resize(kGdnLayerCount * kGdnGateCount);
  candidate.gdn_dt_bias.resize(kGdnLayerCount * kGdnGateCount);
  candidate.gdn_recurrent_norm.resize(kGdnLayerCount * kGdnHeadWidth);
  candidate.attention_query_norm.resize(kAttentionLayerCount *
                                        kAttentionHeadWidth);
  candidate.attention_key_norm.resize(kAttentionLayerCount *
                                      kAttentionHeadWidth);
  candidate.output_norm.resize(kResidualWidth);

  std::size_t gdn_slot = 0;
  std::size_t attention_slot = 0;
  Status status = Status::ok();
  for (std::size_t layer = 0; layer < kModelLayerCount; ++layer) {
    const FfnScalarParameters ffn{
        offset(&candidate.ffn_norms, layer * kResidualWidth), kResidualWidth};
    if (layer % 4 == 3) {
      if (weights.layers[layer].kind != LayerKind::kAttention ||
          attention_slot >= kAttentionLayerCount) {
        return {StatusCode::kInvalidArgument,
                "typed weights do not follow the attention layer schedule"};
      }
      const AttentionScalarParameters mixer{
          offset(&candidate.input_norms, layer * kResidualWidth),
          kResidualWidth,
          offset(&candidate.attention_query_norm,
                 attention_slot * kAttentionHeadWidth),
          kAttentionHeadWidth,
          offset(&candidate.attention_key_norm,
                 attention_slot * kAttentionHeadWidth),
          kAttentionHeadWidth};
      candidate.attention[layer] = {mixer, ffn};
      status = prepare_attention_layer_scalar_parameters(
          weights.layers[layer], candidate.attention[layer]);
      ++attention_slot;
    } else {
      if (weights.layers[layer].kind != LayerKind::kGdn ||
          gdn_slot >= kGdnLayerCount) {
        return {StatusCode::kInvalidArgument,
                "typed weights do not follow the GDN layer schedule"};
      }
      const GdnScalarParameters mixer{
          offset(&candidate.input_norms, layer * kResidualWidth),
          kResidualWidth,
          offset(&candidate.gdn_convolution,
                 gdn_slot * kGdnConvolutionValues),
          kGdnConvolutionValues,
          offset(&candidate.gdn_folded_a, gdn_slot * kGdnGateCount),
          kGdnGateCount,
          offset(&candidate.gdn_dt_bias, gdn_slot * kGdnGateCount),
          kGdnGateCount,
          offset(&candidate.gdn_recurrent_norm, gdn_slot * kGdnHeadWidth),
          kGdnHeadWidth};
      candidate.gdn[layer] = {mixer, ffn};
      status = prepare_gdn_layer_scalar_parameters(weights.layers[layer],
                                                   candidate.gdn[layer]);
      ++gdn_slot;
    }
    if (!status.is_ok()) return status;
  }
  if (gdn_slot != kGdnLayerCount || attention_slot != kAttentionLayerCount) {
    return {StatusCode::kInternal,
            "scalar parameter preparation did not consume every layer slot"};
  }
  candidate.output = {candidate.output_norm.data(),
                      candidate.output_norm.size()};
  status = prepare_output_scalar_parameters(weights, candidate.output);
  if (!status.is_ok()) return status;
  candidate.prepared_layers = kModelLayerCount;
  *parameters = std::move(candidate);
  return Status::ok();
}

Status create_scalar_session_state(std::size_t capacity,
                                   ScalarSessionState* state) noexcept {
  if (state == nullptr || capacity == 0 ||
      capacity > std::numeric_limits<std::size_t>::max() /
                     (kAttentionLayerCount * kAttentionKvWidth)) {
    return {StatusCode::kInvalidArgument,
            "scalar session capacity or output is invalid"};
  }
  ScalarSessionState candidate;
  candidate.gdn_convolution.resize(kGdnLayerCount * kGdnConvolutionValues);
  candidate.gdn_recurrent.resize(kGdnLayerCount * kGdnRecurrentStateValues);
  const std::size_t cache_stride = capacity * kAttentionKvWidth;
  candidate.attention_key.resize(kAttentionLayerCount * cache_stride);
  candidate.attention_value.resize(kAttentionLayerCount * cache_stride);
  std::size_t gdn_slot = 0;
  std::size_t attention_slot = 0;
  for (std::size_t layer = 0; layer < kModelLayerCount; ++layer) {
    if (layer % 4 == 3) {
      candidate.attention[layer] = {
          offset(&candidate.attention_key, attention_slot * cache_stride),
          cache_stride,
          offset(&candidate.attention_value, attention_slot * cache_stride),
          cache_stride,
          capacity};
      ++attention_slot;
    } else {
      candidate.gdn[layer] = {
          offset(&candidate.gdn_convolution,
                 gdn_slot * kGdnConvolutionValues),
          kGdnConvolutionValues,
          offset(&candidate.gdn_recurrent,
                 gdn_slot * kGdnRecurrentStateValues),
          kGdnRecurrentStateValues};
      ++gdn_slot;
    }
  }
  candidate.capacity = capacity;
  candidate.frontier = 0;
  *state = std::move(candidate);
  return Status::ok();
}

Status create_scalar_workspace(std::size_t capacity,
                               ScalarWorkspace* workspace) noexcept {
  if (workspace == nullptr || capacity == 0) {
    return {StatusCode::kInvalidArgument,
            "scalar workspace capacity or output is invalid"};
  }
  ScalarWorkspace candidate;
  candidate.activation_a.resize(kResidualWidth);
  candidate.activation_b.resize(kResidualWidth);
  candidate.post_mixer.resize(kResidualWidth);
  candidate.gdn_normalized.resize(kResidualWidth);
  candidate.gdn_packed.resize(kGdnPackedQkvWidth);
  candidate.gdn_projected_gate.resize(kGdnValueWidth);
  candidate.gdn_projected_alpha.resize(kGdnGateCount);
  candidate.gdn_projected_beta.resize(kGdnGateCount);
  candidate.gdn_convolved.resize(kGdnPackedQkvWidth);
  candidate.gdn_query.resize(kGdnKeyWidth);
  candidate.gdn_key.resize(kGdnKeyWidth);
  candidate.gdn_value_tiled.resize(kGdnValueWidth);
  candidate.gdn_value_grouped.resize(kGdnValueWidth);
  candidate.gdn_gate_grouped.resize(kGdnValueWidth);
  candidate.gdn_alpha_grouped.resize(kGdnGateCount);
  candidate.gdn_beta_grouped.resize(kGdnGateCount);
  candidate.gdn_folded_grouped.resize(kGdnGateCount);
  candidate.gdn_dt_grouped.resize(kGdnGateCount);
  candidate.gdn_log_decay.resize(kGdnGateCount);
  candidate.gdn_update_gate.resize(kGdnGateCount);
  candidate.gdn_recurrent_output.resize(kGdnValueWidth);
  candidate.gdn_gated_grouped.resize(kGdnValueWidth);
  candidate.gdn_gated_tiled.resize(kGdnValueWidth);
  candidate.gdn_mixer_output.resize(kResidualWidth);

  candidate.attention_normalized.resize(kResidualWidth);
  candidate.attention_packed.resize(kAttentionPackedQueryGateWidth);
  candidate.attention_query.resize(kAttentionQueryWidth);
  candidate.attention_gate.resize(kAttentionQueryWidth);
  candidate.attention_key.resize(kAttentionKvWidth);
  candidate.attention_value.resize(kAttentionKvWidth);
  candidate.attention_output.resize(kAttentionQueryWidth);
  candidate.attention_scores.resize(capacity);
  candidate.attention_mixer_output.resize(kResidualWidth);
#ifdef QW38_DIAGNOSTIC_TRACE
  candidate.attention_rope_query.resize(kAttentionQueryWidth);
  candidate.attention_rope_key.resize(kAttentionKvWidth);
#endif

  candidate.ffn_normalized.resize(kResidualWidth);
  candidate.ffn_gate.resize(kFfnWidth);
  candidate.ffn_up.resize(kFfnWidth);
  candidate.ffn_activated.resize(kFfnWidth);
  candidate.ffn_correction.resize(kResidualWidth);
  candidate.final_normalized.resize(kResidualWidth);

  const FfnStepWorkspace ffn{
      candidate.ffn_normalized.data(), candidate.ffn_normalized.size(),
      candidate.ffn_gate.data(),       candidate.ffn_gate.size(),
      candidate.ffn_up.data(),         candidate.ffn_up.size(),
      candidate.ffn_activated.data(),  candidate.ffn_activated.size(),
      candidate.ffn_correction.data(), candidate.ffn_correction.size()};
  const GdnProjectionWorkspace gdn_projection{
      candidate.gdn_packed.data(),          candidate.gdn_packed.size(),
      candidate.gdn_projected_gate.data(),  candidate.gdn_projected_gate.size(),
      candidate.gdn_projected_alpha.data(), candidate.gdn_projected_alpha.size(),
      candidate.gdn_projected_beta.data(),  candidate.gdn_projected_beta.size()};
  const GdnStepWorkspace gdn_mixer{
      candidate.gdn_normalized.data(),
      candidate.gdn_normalized.size(),
      gdn_projection,
      candidate.gdn_convolved.data(),
      candidate.gdn_convolved.size(),
      candidate.gdn_query.data(),
      candidate.gdn_query.size(),
      candidate.gdn_key.data(),
      candidate.gdn_key.size(),
      candidate.gdn_value_tiled.data(),
      candidate.gdn_value_tiled.size(),
      candidate.gdn_value_grouped.data(),
      candidate.gdn_value_grouped.size(),
      candidate.gdn_gate_grouped.data(),
      candidate.gdn_gate_grouped.size(),
      candidate.gdn_alpha_grouped.data(),
      candidate.gdn_beta_grouped.data(),
      candidate.gdn_folded_grouped.data(),
      candidate.gdn_dt_grouped.data(),
      candidate.gdn_log_decay.data(),
      candidate.gdn_update_gate.data(),
      candidate.gdn_update_gate.size(),
      candidate.gdn_recurrent_output.data(),
      candidate.gdn_recurrent_output.size(),
      candidate.gdn_gated_grouped.data(),
      candidate.gdn_gated_grouped.size(),
      candidate.gdn_gated_tiled.data(),
      candidate.gdn_gated_tiled.size(),
      candidate.gdn_mixer_output.data(),
      candidate.gdn_mixer_output.size()};
  candidate.gdn = {gdn_mixer, ffn, candidate.post_mixer.data(),
                   candidate.post_mixer.size()};

  const AttentionProjectionWorkspace attention_projection{
      candidate.attention_packed.data(), candidate.attention_packed.size(),
      candidate.attention_query.data(),  candidate.attention_query.size(),
      candidate.attention_gate.data(),   candidate.attention_gate.size(),
      candidate.attention_key.data(),    candidate.attention_key.size(),
      candidate.attention_value.data(),  candidate.attention_value.size()};
  const AttentionStepWorkspace attention_mixer{
      candidate.attention_normalized.data(),
      candidate.attention_normalized.size(),
      attention_projection,
      candidate.attention_output.data(),
      candidate.attention_output.size(),
      candidate.attention_scores.data(),
      candidate.attention_scores.size(),
      candidate.attention_mixer_output.data(),
      candidate.attention_mixer_output.size()
#ifdef QW38_DIAGNOSTIC_TRACE
      , candidate.attention_rope_query.data(),
      candidate.attention_rope_query.size(), candidate.attention_rope_key.data(),
      candidate.attention_rope_key.size()
#endif
  };
  candidate.attention = {attention_mixer, ffn, candidate.post_mixer.data(),
                         candidate.post_mixer.size()};
  candidate.output = {candidate.final_normalized.data(),
                      candidate.final_normalized.size()};
  candidate.capacity = capacity;
  candidate.layers_completed = 0;
  *workspace = std::move(candidate);
  return Status::ok();
}

namespace {

Status execute_scalar_token_impl(
    const ModelWeights& weights, const ScalarModelParameters& parameters,
    std::size_t token, ScalarSessionState* state, ScalarWorkspace* workspace,
    float* logits, std::size_t logits_count
#ifdef QW38_DIAGNOSTIC_TRACE
    , const TraceFilter* filter, TraceSink sink, void* sink_context
#endif
    ) noexcept {
  if (state == nullptr || workspace == nullptr || token >= kVocabularySize ||
      logits == nullptr || logits_count != kVocabularySize ||
      !valid_parameter_storage(parameters) || !valid_state_storage(*state) ||
      !valid_workspace_storage(*workspace, state->capacity)) {
    return {StatusCode::kInvalidArgument,
            "scalar token parameters, state, workspace, or output are invalid"};
  }
  for (std::size_t layer = 0; layer < kModelLayerCount; ++layer) {
    const LayerKind expected =
        layer % 4 == 3 ? LayerKind::kAttention : LayerKind::kGdn;
    if (weights.layers[layer].kind != expected) {
      return {StatusCode::kInvalidArgument,
              "typed weights do not follow the 64-layer schedule"};
    }
  }

  workspace->layers_completed = 0;
  Status status = embed_token(weights, token, workspace->activation_a.data(),
                              workspace->activation_a.size());
#ifdef QW38_DIAGNOSTIC_TRACE
  if (status.is_ok()) {
    status = offer_trace(filter, sink, sink_context, "embedding",
                         kTraceAllLayers, workspace->activation_a.data(),
                         kResidualWidth, kResidualWidth);
  }
#endif
  float* input = workspace->activation_a.data();
  float* output = workspace->activation_b.data();
  for (std::size_t layer = 0; status.is_ok() && layer < kModelLayerCount;
       ++layer) {
    if (weights.layers[layer].kind == LayerKind::kAttention) {
      status = execute_attention_layer_step(
          weights.layers[layer], parameters.attention[layer], state->frontier,
          input, kResidualWidth, state->attention[layer], workspace->attention,
          output, kResidualWidth);
    } else {
      status = execute_gdn_layer_step(
          weights.layers[layer], parameters.gdn[layer], input, kResidualWidth,
          state->gdn[layer], workspace->gdn, output, kResidualWidth);
    }
    if (status.is_ok()) {
#ifdef QW38_DIAGNOSTIC_TRACE
      if (weights.layers[layer].kind == LayerKind::kAttention) {
        const std::size_t row = state->frontier * kAttentionKvWidth;
        status = offer_trace(filter, sink, sink_context, "input_norm", layer,
                             workspace->attention_normalized.data(),
                             kResidualWidth, kResidualWidth);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "attention.packed_query_gate", layer,
            workspace->attention_packed.data(), kAttentionPackedQueryGateWidth,
            kAttentionPackedQueryGateWidth);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "attention.query", layer, workspace->attention_query.data(),
            kAttentionQueryWidth, 24, kAttentionHeadWidth);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "attention.key", layer, workspace->attention_key.data(),
            kAttentionKvWidth, 4, kAttentionHeadWidth);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "attention.rope_query", layer,
            workspace->attention_rope_query.data(), kAttentionQueryWidth, 24,
            kAttentionHeadWidth);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "attention.rope_key", layer, workspace->attention_rope_key.data(),
            kAttentionKvWidth, 4, kAttentionHeadWidth);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "attention.value", layer, workspace->attention_value.data(),
            kAttentionKvWidth, 4, kAttentionHeadWidth);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "attention.kv_key_row", layer,
            state->attention[layer].key_cache + row, kAttentionKvWidth, 4,
            kAttentionHeadWidth);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "attention.kv_value_row", layer,
            state->attention[layer].value_cache + row, kAttentionKvWidth, 4,
            kAttentionHeadWidth);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "attention.context", layer, workspace->attention_output.data(),
            kAttentionQueryWidth, 24, kAttentionHeadWidth);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "attention.output", layer, workspace->attention_mixer_output.data(),
            kResidualWidth, kResidualWidth);
      } else {
        status = offer_trace(filter, sink, sink_context, "input_norm", layer,
                             workspace->gdn_normalized.data(), kResidualWidth,
                             kResidualWidth);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "gdn.packed_qkv", layer, workspace->gdn_packed.data(),
            kGdnPackedQkvWidth, kGdnPackedQkvWidth);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "gdn.convolution", layer, workspace->gdn_convolved.data(),
            kGdnPackedQkvWidth, kGdnPackedQkvWidth);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "gdn.query", layer, workspace->gdn_query.data(), kGdnKeyWidth, 16,
            kGdnHeadWidth);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "gdn.key", layer, workspace->gdn_key.data(), kGdnKeyWidth, 16,
            kGdnHeadWidth);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "gdn.recurrent_output", layer,
            workspace->gdn_recurrent_output.data(), kGdnValueWidth, 48,
            kGdnHeadWidth);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "gdn.recurrent_state", layer, state->gdn[layer].recurrent,
            kGdnRecurrentStateValues, 48, kGdnHeadWidth, kGdnHeadWidth);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "gdn.convolution_state", layer, state->gdn[layer].convolution,
            kGdnConvolutionValues, 4, kGdnPackedQkvWidth);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "gdn.output", layer, workspace->gdn_mixer_output.data(),
            kResidualWidth, kResidualWidth);
      }
      if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
          "mixer_residual", layer, workspace->post_mixer.data(), kResidualWidth,
          kResidualWidth);
      if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
          "ffn.input_norm", layer, workspace->ffn_normalized.data(),
          kResidualWidth, kResidualWidth);
      if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
          "ffn.gate", layer, workspace->ffn_gate.data(), kFfnWidth, kFfnWidth);
      if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
          "ffn.up", layer, workspace->ffn_up.data(), kFfnWidth, kFfnWidth);
      if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
          "ffn.activated", layer, workspace->ffn_activated.data(), kFfnWidth,
          kFfnWidth);
      if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
          "ffn.correction", layer, workspace->ffn_correction.data(),
          kResidualWidth, kResidualWidth);
      if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
          "layer_residual", layer, output, kResidualWidth, kResidualWidth);
#endif
    }
    if (status.is_ok()) {
      ++workspace->layers_completed;
      std::swap(input, output);
    }
  }
  if (!status.is_ok()) return status;
  status = project_logits(weights, parameters.output, input, kResidualWidth,
                          workspace->output, logits, logits_count);
  if (!status.is_ok()) return status;
#ifdef QW38_DIAGNOSTIC_TRACE
  status = offer_trace(filter, sink, sink_context, "final_norm",
                       kTraceAllLayers, workspace->final_normalized.data(),
                       kResidualWidth, kResidualWidth);
  if (status.is_ok()) {
    status = offer_trace(filter, sink, sink_context, "logits", kTraceAllLayers,
                         logits, logits_count, logits_count);
  }
  if (!status.is_ok()) return status;
#endif
  ++state->frontier;
  return Status::ok();
}

}  // namespace

Status execute_scalar_token(const ModelWeights& weights,
                            const ScalarModelParameters& parameters,
                            std::size_t token, ScalarSessionState* state,
                            ScalarWorkspace* workspace, float* logits,
                            std::size_t logits_count) noexcept {
  return execute_scalar_token_impl(weights, parameters, token, state, workspace,
                                   logits, logits_count
#ifdef QW38_DIAGNOSTIC_TRACE
                                   , nullptr, nullptr, nullptr
#endif
  );
}

Status execute_scalar_chunk(
    const ModelWeights& weights, const ScalarModelParameters& parameters,
    const std::size_t* tokens, std::size_t token_count,
    ScalarSessionState* state, ScalarWorkspace* workspace, float* logits,
    std::size_t logits_count) noexcept {
  if (tokens == nullptr || token_count == 0 || state == nullptr ||
      workspace == nullptr || logits == nullptr ||
      !valid_parameter_storage(parameters) || !valid_state_storage(*state) ||
      !valid_workspace_storage(*workspace, state->capacity) ||
      token_count > state->capacity - state->frontier ||
      token_count > std::numeric_limits<std::size_t>::max() / kVocabularySize ||
      logits_count != token_count * kVocabularySize) {
    return {StatusCode::kInvalidArgument,
            "scalar chunk tokens, capacity, storage, or output are invalid"};
  }
  for (std::size_t index = 0; index < token_count; ++index) {
    if (tokens[index] >= kVocabularySize) {
      return {StatusCode::kInvalidArgument,
              "scalar chunk contains an invalid token ID"};
    }
  }
  for (std::size_t index = 0; index < token_count; ++index) {
    Status status = execute_scalar_token(
        weights, parameters, tokens[index], state, workspace,
        logits + index * kVocabularySize, kVocabularySize);
    if (!status.is_ok()) return status;
  }
  return Status::ok();
}

#ifdef QW38_DIAGNOSTIC_TRACE
Status execute_scalar_token_traced(
    const ModelWeights& weights, const ScalarModelParameters& parameters,
    std::size_t token, ScalarSessionState* state, ScalarWorkspace* workspace,
    float* logits, std::size_t logits_count, const TraceFilter& filter,
    TraceSink sink, void* sink_context) noexcept {
  Status status = validate_trace_filter(filter);
  if (!status.is_ok() || sink == nullptr) {
    return status.is_ok()
               ? Status{StatusCode::kInvalidArgument,
                        "diagnostic scalar trace sink is required"}
               : status;
  }
  return execute_scalar_token_impl(weights, parameters, token, state, workspace,
                                   logits, logits_count, &filter, sink,
                                   sink_context);
}
#endif

}  // namespace qw38::internal
