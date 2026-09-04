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
  const ModelGeometry& geometry = parameters.geometry;
  return geometry_is_valid(geometry) &&
         parameters.prepared_layers == geometry.layer_count &&
         parameters.input_norms.size() ==
             geometry.layer_count * geometry.residual_width &&
         parameters.ffn_norms.size() ==
             geometry.layer_count * geometry.residual_width &&
         parameters.gdn_convolution.size() ==
             geometry.gdn_layer_count * geometry.gdn_conv_values() &&
         parameters.gdn_folded_a.size() ==
             geometry.gdn_layer_count * geometry.gdn_value_heads &&
         parameters.gdn_dt_bias.size() ==
             geometry.gdn_layer_count * geometry.gdn_value_heads &&
         parameters.gdn_recurrent_norm.size() ==
             geometry.gdn_layer_count * geometry.gdn_head_width &&
         parameters.attention_query_norm.size() ==
             geometry.attention_layer_count * geometry.attention_head_width &&
         parameters.attention_key_norm.size() ==
             geometry.attention_layer_count * geometry.attention_head_width &&
         parameters.output_norm.size() == geometry.residual_width;
}

bool valid_state_storage(const ScalarSessionState& state) noexcept {
  const ModelGeometry& geometry = state.geometry;
  if (!geometry_is_valid(geometry) || state.capacity == 0 ||
      state.frontier >= state.capacity ||
      geometry.attention_kv_width() == 0 ||
      state.capacity > std::numeric_limits<std::size_t>::max() /
                           (geometry.attention_layer_count *
                            geometry.attention_kv_width())) {
    return false;
  }
  const std::size_t attention_values = geometry.attention_layer_count *
                                       state.capacity *
                                       geometry.attention_kv_width();
  return state.gdn_convolution.size() ==
             geometry.gdn_layer_count * geometry.gdn_conv_values() &&
         state.gdn_recurrent.size() ==
             geometry.gdn_layer_count * geometry.gdn_recurrent_values() &&
         state.attention_key.size() == attention_values &&
         state.attention_value.size() == attention_values;
}

bool valid_workspace_storage(const ScalarWorkspace& workspace,
                             std::size_t capacity) noexcept {
  const ModelGeometry& geometry = workspace.geometry;
  return geometry_is_valid(geometry) && workspace.capacity == capacity &&
         workspace.activation_a.size() == geometry.residual_width &&
         workspace.activation_b.size() == geometry.residual_width &&
         workspace.post_mixer.size() == geometry.residual_width &&
         workspace.gdn_normalized.size() == geometry.residual_width &&
         workspace.gdn_packed.size() == geometry.gdn_packed_qkv() &&
         workspace.gdn_projected_gate.size() == geometry.gdn_value_width() &&
         workspace.gdn_projected_alpha.size() == geometry.gdn_value_heads &&
         workspace.gdn_projected_beta.size() == geometry.gdn_value_heads &&
         workspace.gdn_convolved.size() == geometry.gdn_packed_qkv() &&
         workspace.gdn_query.size() == geometry.gdn_key_width() &&
         workspace.gdn_key.size() == geometry.gdn_key_width() &&
         workspace.gdn_value_tiled.size() == geometry.gdn_value_width() &&
         workspace.gdn_value_grouped.size() == geometry.gdn_value_width() &&
         workspace.gdn_gate_grouped.size() == geometry.gdn_value_width() &&
         workspace.gdn_alpha_grouped.size() == geometry.gdn_value_heads &&
         workspace.gdn_beta_grouped.size() == geometry.gdn_value_heads &&
         workspace.gdn_folded_grouped.size() == geometry.gdn_value_heads &&
         workspace.gdn_dt_grouped.size() == geometry.gdn_value_heads &&
         workspace.gdn_log_decay.size() == geometry.gdn_value_heads &&
         workspace.gdn_update_gate.size() == geometry.gdn_value_heads &&
         workspace.gdn_recurrent_output.size() == geometry.gdn_value_width() &&
         workspace.gdn_gated_grouped.size() == geometry.gdn_value_width() &&
         workspace.gdn_gated_tiled.size() == geometry.gdn_value_width() &&
         workspace.gdn_mixer_output.size() == geometry.residual_width &&
         workspace.attention_normalized.size() == geometry.residual_width &&
         workspace.attention_packed.size() ==
             geometry.attention_packed_query_gate() &&
         workspace.attention_query.size() == geometry.attention_query_width() &&
         workspace.attention_gate.size() == geometry.attention_query_width() &&
         workspace.attention_key.size() == geometry.attention_kv_width() &&
         workspace.attention_value.size() == geometry.attention_kv_width() &&
         workspace.attention_output.size() ==
             geometry.attention_query_width() &&
         workspace.attention_scores.size() == capacity &&
         workspace.attention_mixer_output.size() == geometry.residual_width &&
#ifdef QW38_DIAGNOSTIC_TRACE
         workspace.attention_rope_query.size() ==
             geometry.attention_query_width() &&
         workspace.attention_rope_key.size() == geometry.attention_kv_width() &&
#endif
         workspace.ffn_normalized.size() == geometry.residual_width &&
         workspace.ffn_gate.size() == geometry.ffn_width &&
         workspace.ffn_up.size() == geometry.ffn_width &&
         workspace.ffn_activated.size() == geometry.ffn_width &&
         workspace.ffn_correction.size() == geometry.residual_width &&
         workspace.final_normalized.size() == geometry.residual_width;
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
  const ModelGeometry& geometry = weights.geometry;
  if (!geometry_is_valid(geometry) ||
      weights.bound_layers != geometry.layer_count) {
    return {StatusCode::kInvalidArgument,
            "typed weights do not declare an admitted geometry"};
  }
  ScalarModelParameters candidate;
  candidate.geometry = geometry;
  candidate.input_norms.resize(geometry.layer_count * geometry.residual_width);
  candidate.ffn_norms.resize(geometry.layer_count * geometry.residual_width);
  candidate.gdn_convolution.resize(geometry.gdn_layer_count *
                                   geometry.gdn_conv_values());
  candidate.gdn_folded_a.resize(geometry.gdn_layer_count *
                                geometry.gdn_value_heads);
  candidate.gdn_dt_bias.resize(geometry.gdn_layer_count *
                               geometry.gdn_value_heads);
  candidate.gdn_recurrent_norm.resize(geometry.gdn_layer_count *
                                      geometry.gdn_head_width);
  candidate.attention_query_norm.resize(geometry.attention_layer_count *
                                        geometry.attention_head_width);
  candidate.attention_key_norm.resize(geometry.attention_layer_count *
                                      geometry.attention_head_width);
  candidate.output_norm.resize(geometry.residual_width);

  std::size_t gdn_slot = 0;
  std::size_t attention_slot = 0;
  Status status = Status::ok();
  for (std::size_t layer = 0; layer < geometry.layer_count; ++layer) {
    const FfnScalarParameters ffn{
        offset(&candidate.ffn_norms, layer * geometry.residual_width),
        geometry.residual_width};
    if (layer % 4 == 3) {
      if (weights.layers[layer].kind != LayerKind::kAttention ||
          attention_slot >= geometry.attention_layer_count) {
        return {StatusCode::kInvalidArgument,
                "typed weights do not follow the attention layer schedule"};
      }
      const AttentionScalarParameters mixer{
          offset(&candidate.input_norms, layer * geometry.residual_width),
          geometry.residual_width,
          offset(&candidate.attention_query_norm,
                 attention_slot * geometry.attention_head_width),
          geometry.attention_head_width,
          offset(&candidate.attention_key_norm,
                 attention_slot * geometry.attention_head_width),
          geometry.attention_head_width};
      candidate.attention[layer] = {mixer, ffn};
      status = prepare_attention_layer_scalar_parameters(
          weights.layers[layer], candidate.attention[layer]);
      ++attention_slot;
    } else {
      if (weights.layers[layer].kind != LayerKind::kGdn ||
          gdn_slot >= geometry.gdn_layer_count) {
        return {StatusCode::kInvalidArgument,
                "typed weights do not follow the GDN layer schedule"};
      }
      const GdnScalarParameters mixer{
          offset(&candidate.input_norms, layer * geometry.residual_width),
          geometry.residual_width,
          offset(&candidate.gdn_convolution,
                 gdn_slot * geometry.gdn_conv_values()),
          geometry.gdn_conv_values(),
          offset(&candidate.gdn_folded_a,
                 gdn_slot * geometry.gdn_value_heads),
          geometry.gdn_value_heads,
          offset(&candidate.gdn_dt_bias, gdn_slot * geometry.gdn_value_heads),
          geometry.gdn_value_heads,
          offset(&candidate.gdn_recurrent_norm,
                 gdn_slot * geometry.gdn_head_width),
          geometry.gdn_head_width};
      candidate.gdn[layer] = {mixer, ffn};
      status = prepare_gdn_layer_scalar_parameters(weights.layers[layer],
                                                   candidate.gdn[layer]);
      ++gdn_slot;
    }
    if (!status.is_ok()) return status;
  }
  if (gdn_slot != geometry.gdn_layer_count ||
      attention_slot != geometry.attention_layer_count) {
    return {StatusCode::kInternal,
            "scalar parameter preparation did not consume every layer slot"};
  }
  candidate.output = {candidate.output_norm.data(),
                      candidate.output_norm.size()};
  status = prepare_output_scalar_parameters(weights, candidate.output);
  if (!status.is_ok()) return status;
  candidate.prepared_layers = geometry.layer_count;
  *parameters = std::move(candidate);
  return Status::ok();
}

Status create_scalar_session_state(const ModelGeometry& geometry,
                                   std::size_t capacity,
                                   ScalarSessionState* state) noexcept {
  if (state == nullptr || capacity == 0 || !geometry_is_valid(geometry) ||
      geometry.attention_kv_width() == 0 ||
      capacity > std::numeric_limits<std::size_t>::max() /
                     (geometry.attention_layer_count *
                      geometry.attention_kv_width())) {
    return {StatusCode::kInvalidArgument,
            "scalar session capacity or output is invalid"};
  }
  ScalarSessionState candidate;
  candidate.geometry = geometry;
  candidate.gdn_convolution.resize(geometry.gdn_layer_count *
                                   geometry.gdn_conv_values());
  candidate.gdn_recurrent.resize(geometry.gdn_layer_count *
                                 geometry.gdn_recurrent_values());
  const std::size_t cache_stride = capacity * geometry.attention_kv_width();
  candidate.attention_key.resize(geometry.attention_layer_count * cache_stride);
  candidate.attention_value.resize(geometry.attention_layer_count *
                                   cache_stride);
  std::size_t gdn_slot = 0;
  std::size_t attention_slot = 0;
  for (std::size_t layer = 0; layer < geometry.layer_count; ++layer) {
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
                 gdn_slot * geometry.gdn_conv_values()),
          geometry.gdn_conv_values(),
          offset(&candidate.gdn_recurrent,
                 gdn_slot * geometry.gdn_recurrent_values()),
          geometry.gdn_recurrent_values()};
      ++gdn_slot;
    }
  }
  candidate.capacity = capacity;
  candidate.frontier = 0;
  *state = std::move(candidate);
  return Status::ok();
}

Status create_scalar_session_state(std::size_t capacity,
                                   ScalarSessionState* state) noexcept {
  return create_scalar_session_state(qwen38_27b_geometry(), capacity, state);
}

Status create_scalar_workspace(const ModelGeometry& geometry,
                               std::size_t capacity,
                               ScalarWorkspace* workspace) noexcept {
  if (workspace == nullptr || capacity == 0 || !geometry_is_valid(geometry)) {
    return {StatusCode::kInvalidArgument,
            "scalar workspace capacity or output is invalid"};
  }
  ScalarWorkspace candidate;
  candidate.geometry = geometry;
  candidate.activation_a.resize(geometry.residual_width);
  candidate.activation_b.resize(geometry.residual_width);
  candidate.post_mixer.resize(geometry.residual_width);
  candidate.gdn_normalized.resize(geometry.residual_width);
  candidate.gdn_packed.resize(geometry.gdn_packed_qkv());
  candidate.gdn_projected_gate.resize(geometry.gdn_value_width());
  candidate.gdn_projected_alpha.resize(geometry.gdn_value_heads);
  candidate.gdn_projected_beta.resize(geometry.gdn_value_heads);
  candidate.gdn_convolved.resize(geometry.gdn_packed_qkv());
  candidate.gdn_query.resize(geometry.gdn_key_width());
  candidate.gdn_key.resize(geometry.gdn_key_width());
  candidate.gdn_value_tiled.resize(geometry.gdn_value_width());
  candidate.gdn_value_grouped.resize(geometry.gdn_value_width());
  candidate.gdn_gate_grouped.resize(geometry.gdn_value_width());
  candidate.gdn_alpha_grouped.resize(geometry.gdn_value_heads);
  candidate.gdn_beta_grouped.resize(geometry.gdn_value_heads);
  candidate.gdn_folded_grouped.resize(geometry.gdn_value_heads);
  candidate.gdn_dt_grouped.resize(geometry.gdn_value_heads);
  candidate.gdn_log_decay.resize(geometry.gdn_value_heads);
  candidate.gdn_update_gate.resize(geometry.gdn_value_heads);
  candidate.gdn_recurrent_output.resize(geometry.gdn_value_width());
  candidate.gdn_gated_grouped.resize(geometry.gdn_value_width());
  candidate.gdn_gated_tiled.resize(geometry.gdn_value_width());
  candidate.gdn_mixer_output.resize(geometry.residual_width);

  candidate.attention_normalized.resize(geometry.residual_width);
  candidate.attention_packed.resize(geometry.attention_packed_query_gate());
  candidate.attention_query.resize(geometry.attention_query_width());
  candidate.attention_gate.resize(geometry.attention_query_width());
  candidate.attention_key.resize(geometry.attention_kv_width());
  candidate.attention_value.resize(geometry.attention_kv_width());
  candidate.attention_output.resize(geometry.attention_query_width());
  candidate.attention_scores.resize(capacity);
  candidate.attention_mixer_output.resize(geometry.residual_width);
#ifdef QW38_DIAGNOSTIC_TRACE
  candidate.attention_rope_query.resize(geometry.attention_query_width());
  candidate.attention_rope_key.resize(geometry.attention_kv_width());
#endif

  candidate.ffn_normalized.resize(geometry.residual_width);
  candidate.ffn_gate.resize(geometry.ffn_width);
  candidate.ffn_up.resize(geometry.ffn_width);
  candidate.ffn_activated.resize(geometry.ffn_width);
  candidate.ffn_correction.resize(geometry.residual_width);
  candidate.final_normalized.resize(geometry.residual_width);

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

Status create_scalar_workspace(std::size_t capacity,
                               ScalarWorkspace* workspace) noexcept {
  return create_scalar_workspace(qwen38_27b_geometry(), capacity, workspace);
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
  const ModelGeometry& geometry = weights.geometry;
  if (state == nullptr || workspace == nullptr ||
      token >= geometry.vocabulary || logits == nullptr ||
      logits_count != geometry.vocabulary ||
      !valid_parameter_storage(parameters) || !valid_state_storage(*state) ||
      !valid_workspace_storage(*workspace, state->capacity) ||
      parameters.geometry.identity != geometry.identity ||
      state->geometry.identity != geometry.identity ||
      workspace->geometry.identity != geometry.identity) {
    return {StatusCode::kInvalidArgument,
            "scalar token parameters, state, workspace, or output are invalid"};
  }
  for (std::size_t layer = 0; layer < geometry.layer_count; ++layer) {
    const LayerKind expected =
        layer % 4 == 3 ? LayerKind::kAttention : LayerKind::kGdn;
    if (weights.layers[layer].kind != expected) {
      return {StatusCode::kInvalidArgument,
              "typed weights do not follow the hybrid layer schedule"};
    }
  }

  workspace->layers_completed = 0;
  Status status = embed_token(weights, token, workspace->activation_a.data(),
                              workspace->activation_a.size());
#ifdef QW38_DIAGNOSTIC_TRACE
  if (status.is_ok()) {
    status = offer_trace(filter, sink, sink_context, "embedding",
                         kTraceAllLayers, workspace->activation_a.data(),
                         geometry.residual_width, geometry.residual_width);
  }
#endif
  float* input = workspace->activation_a.data();
  float* output = workspace->activation_b.data();
  for (std::size_t layer = 0; status.is_ok() && layer < geometry.layer_count;
       ++layer) {
    if (weights.layers[layer].kind == LayerKind::kAttention) {
      status = execute_attention_layer_step(
          weights.layers[layer], parameters.attention[layer], state->frontier,
          input, geometry.residual_width, state->attention[layer],
          workspace->attention, output, geometry.residual_width);
    } else {
      status = execute_gdn_layer_step(
          weights.layers[layer], parameters.gdn[layer], input,
          geometry.residual_width, state->gdn[layer], workspace->gdn, output,
          geometry.residual_width);
    }
    if (status.is_ok()) {
#ifdef QW38_DIAGNOSTIC_TRACE
      if (weights.layers[layer].kind == LayerKind::kAttention) {
        const std::size_t row = state->frontier * geometry.attention_kv_width();
        status = offer_trace(filter, sink, sink_context, "input_norm", layer,
                             workspace->attention_normalized.data(),
                             geometry.residual_width, geometry.residual_width);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "attention.packed_query_gate", layer,
            workspace->attention_packed.data(), geometry.attention_packed_query_gate(),
            geometry.attention_packed_query_gate());
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "attention.query", layer, workspace->attention_query.data(),
            geometry.attention_query_width(), geometry.attention_query_heads, geometry.attention_head_width);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "attention.key", layer, workspace->attention_key.data(),
            geometry.attention_kv_width(), geometry.attention_kv_heads, geometry.attention_head_width);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "attention.rope_query", layer,
            workspace->attention_rope_query.data(), geometry.attention_query_width(), geometry.attention_query_heads,
            geometry.attention_head_width);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "attention.rope_key", layer, workspace->attention_rope_key.data(),
            geometry.attention_kv_width(), geometry.attention_kv_heads, geometry.attention_head_width);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "attention.value", layer, workspace->attention_value.data(),
            geometry.attention_kv_width(), geometry.attention_kv_heads, geometry.attention_head_width);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "attention.kv_key_row", layer,
            state->attention[layer].key_cache + row, geometry.attention_kv_width(), geometry.attention_kv_heads,
            geometry.attention_head_width);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "attention.kv_value_row", layer,
            state->attention[layer].value_cache + row, geometry.attention_kv_width(), geometry.attention_kv_heads,
            geometry.attention_head_width);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "attention.context", layer, workspace->attention_output.data(),
            geometry.attention_query_width(), geometry.attention_query_heads, geometry.attention_head_width);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "attention.output", layer, workspace->attention_mixer_output.data(),
            geometry.residual_width, geometry.residual_width);
      } else {
        status = offer_trace(filter, sink, sink_context, "input_norm", layer,
                             workspace->gdn_normalized.data(), geometry.residual_width,
                             geometry.residual_width);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "gdn.packed_qkv", layer, workspace->gdn_packed.data(),
            geometry.gdn_packed_qkv(), geometry.gdn_packed_qkv());
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "gdn.convolution", layer, workspace->gdn_convolved.data(),
            geometry.gdn_packed_qkv(), geometry.gdn_packed_qkv());
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "gdn.query", layer, workspace->gdn_query.data(), geometry.gdn_key_width(), geometry.gdn_key_heads,
            geometry.gdn_head_width);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "gdn.key", layer, workspace->gdn_key.data(), geometry.gdn_key_width(), geometry.gdn_key_heads,
            geometry.gdn_head_width);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "gdn.value", layer, workspace->gdn_value_grouped.data(),
            geometry.gdn_value_width(), geometry.gdn_value_heads, geometry.gdn_head_width);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "gdn.decay", layer, workspace->gdn_log_decay.data(),
            geometry.gdn_value_heads, geometry.gdn_value_heads);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "gdn.beta", layer, workspace->gdn_update_gate.data(),
            geometry.gdn_value_heads, geometry.gdn_value_heads);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "gdn.recurrent_output", layer,
            workspace->gdn_recurrent_output.data(), geometry.gdn_value_width(), geometry.gdn_value_heads,
            geometry.gdn_head_width);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "gdn.recurrent_state", layer, state->gdn[layer].recurrent,
            geometry.gdn_recurrent_values(), geometry.gdn_value_heads, geometry.gdn_head_width, geometry.gdn_head_width);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "gdn.convolution_state", layer, state->gdn[layer].convolution,
            geometry.gdn_conv_values(), geometry.gdn_packed_qkv(), geometry.gdn_conv_width);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "gdn.gated_output", layer, workspace->gdn_gated_grouped.data(),
            geometry.gdn_value_width(), geometry.gdn_value_heads, geometry.gdn_head_width);
        if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
            "gdn.output", layer, workspace->gdn_mixer_output.data(),
            geometry.residual_width, geometry.residual_width);
      }
      if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
          "mixer_residual", layer, workspace->post_mixer.data(), geometry.residual_width,
          geometry.residual_width);
      if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
          "ffn.input_norm", layer, workspace->ffn_normalized.data(),
          geometry.residual_width, geometry.residual_width);
      if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
          "ffn.gate", layer, workspace->ffn_gate.data(), geometry.ffn_width, geometry.ffn_width);
      if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
          "ffn.up", layer, workspace->ffn_up.data(), geometry.ffn_width, geometry.ffn_width);
      if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
          "ffn.activated", layer, workspace->ffn_activated.data(), geometry.ffn_width,
          geometry.ffn_width);
      if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
          "ffn.correction", layer, workspace->ffn_correction.data(),
          geometry.residual_width, geometry.residual_width);
      if (status.is_ok()) status = offer_trace(filter, sink, sink_context,
          "layer_residual", layer, output, geometry.residual_width, geometry.residual_width);
#endif
    }
    if (status.is_ok()) {
      ++workspace->layers_completed;
      std::swap(input, output);
    }
  }
  if (!status.is_ok()) return status;
  status = project_logits(weights, parameters.output, input, geometry.residual_width,
                          workspace->output, logits, logits_count);
  if (!status.is_ok()) return status;
#ifdef QW38_DIAGNOSTIC_TRACE
  status = offer_trace(filter, sink, sink_context, "final_norm",
                       kTraceAllLayers, workspace->final_normalized.data(),
                       geometry.residual_width, geometry.residual_width);
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
  const ModelGeometry& geometry = weights.geometry;
  if (tokens == nullptr || token_count == 0 || state == nullptr ||
      workspace == nullptr || logits == nullptr ||
      !valid_parameter_storage(parameters) || !valid_state_storage(*state) ||
      !valid_workspace_storage(*workspace, state->capacity) ||
      token_count > state->capacity - state->frontier ||
      token_count > std::numeric_limits<std::size_t>::max() / geometry.vocabulary ||
      logits_count != token_count * geometry.vocabulary) {
    return {StatusCode::kInvalidArgument,
            "scalar chunk tokens, capacity, storage, or output are invalid"};
  }
  for (std::size_t index = 0; index < token_count; ++index) {
    if (tokens[index] >= geometry.vocabulary) {
      return {StatusCode::kInvalidArgument,
              "scalar chunk contains an invalid token ID"};
    }
  }
  for (std::size_t index = 0; index < token_count; ++index) {
    Status status = execute_scalar_token(
        weights, parameters, tokens[index], state, workspace,
        logits + index * geometry.vocabulary, geometry.vocabulary);
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
