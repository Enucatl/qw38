#include "mixer.h"

#include <cmath>
#include <limits>

#include "attention.h"
#include "conversion.h"
#include "gdn.h"
#include "projection.h"
#include "tensor.h"

namespace qw38::internal {

Status project_gdn_mixer(const GdnLayerWeights& weights,
                         const float* activation,
                         std::size_t activation_count,
                         const GdnProjectionWorkspace& workspace) noexcept {
  if (activation == nullptr ||
      activation_count != weights.packed_qkv.columns ||
      workspace.packed_qkv == nullptr ||
      workspace.packed_qkv_count != weights.packed_qkv.rows ||
      workspace.value_gate == nullptr ||
      workspace.value_gate_count != weights.value_gate.rows ||
      workspace.alpha == nullptr ||
      workspace.alpha_count != weights.alpha.rows ||
      workspace.beta == nullptr ||
      workspace.beta_count != weights.beta.rows) {
    return {StatusCode::kInvalidArgument,
            "GDN projection activation or workspace is invalid"};
  }
  Status status = tensor_matvec(weights.packed_qkv, activation, activation_count,
                                workspace.packed_qkv,
                                workspace.packed_qkv_count);
  if (status.is_ok()) {
    status = tensor_matvec(weights.value_gate, activation, activation_count,
                           workspace.value_gate, workspace.value_gate_count);
  }
  if (status.is_ok()) {
    status = tensor_matvec(weights.alpha, activation, activation_count,
                           workspace.alpha, workspace.alpha_count);
  }
  if (status.is_ok()) {
    status = tensor_matvec(weights.beta, activation, activation_count,
                           workspace.beta, workspace.beta_count);
  }
  return status;
}

Status project_attention_mixer(
    const AttentionLayerWeights& weights, const float* activation,
    std::size_t activation_count,
    const AttentionProjectionWorkspace& workspace) noexcept {
  if (activation == nullptr ||
      activation_count != weights.query_gate.columns ||
      workspace.packed_query_gate == nullptr ||
      workspace.packed_query_gate_count != weights.query_gate.rows ||
      workspace.query == nullptr ||
      workspace.query_count != weights.query_gate.rows / 2 ||
      workspace.gate == nullptr ||
      workspace.gate_count != workspace.query_count || workspace.key == nullptr ||
      workspace.key_count != weights.key.rows || workspace.value == nullptr ||
      workspace.value_count != weights.value.rows) {
    return {StatusCode::kInvalidArgument,
            "attention projection activation or workspace is invalid"};
  }
  Status status = tensor_matvec(
      weights.query_gate, activation, activation_count,
      workspace.packed_query_gate, workspace.packed_query_gate_count);
  if (status.is_ok()) {
    status = split_attention_query_gate(
        workspace.packed_query_gate, workspace.packed_query_gate_count,
        workspace.query_count / weights.query_norm.count,
        weights.query_norm.count,
        workspace.query, workspace.query_count, workspace.gate,
        workspace.gate_count);
  }
  if (status.is_ok()) {
    status = tensor_matvec(weights.key, activation, activation_count,
                           workspace.key, workspace.key_count);
  }
  if (status.is_ok()) {
    status = tensor_matvec(weights.value, activation, activation_count,
                           workspace.value, workspace.value_count);
  }
  return status;
}

Status prepare_gdn_scalar_parameters(
    const LayerWeights& weights,
    const GdnScalarParameters& parameters) noexcept {
  const ModelGeometry& geometry = weights.geometry;
  if (weights.kind != LayerKind::kGdn || parameters.input_norm == nullptr ||
      parameters.input_norm_count != geometry.residual_width ||
      parameters.convolution == nullptr ||
      parameters.convolution_count != geometry.gdn_conv_values() ||
      parameters.folded_a_tiled == nullptr ||
      parameters.folded_a_count != geometry.gdn_value_heads ||
      parameters.dt_bias_tiled == nullptr ||
      parameters.dt_bias_count != geometry.gdn_value_heads ||
      parameters.recurrent_norm == nullptr ||
      parameters.recurrent_norm_count != geometry.gdn_head_width) {
    return {StatusCode::kInvalidArgument,
            "GDN scalar parameter buffers are invalid"};
  }
  Status status = vector_decode(weights.common.input_norm, parameters.input_norm,
                                parameters.input_norm_count);
  for (std::size_t channel = 0;
       status.is_ok() && channel < geometry.gdn_packed_qkv(); ++channel) {
    status = tensor_row_decode(weights.gdn.convolution, channel,
                               parameters.convolution + channel * 4, 4);
  }
  if (status.is_ok()) {
    status = vector_decode(weights.gdn.folded_a, parameters.folded_a_tiled,
                           parameters.folded_a_count);
  }
  if (status.is_ok()) {
    status = vector_decode(weights.gdn.dt_bias, parameters.dt_bias_tiled,
                           parameters.dt_bias_count);
  }
  if (status.is_ok()) {
    status = vector_decode(weights.gdn.norm, parameters.recurrent_norm,
                           parameters.recurrent_norm_count);
  }
  return status;
}

namespace {

bool valid_gdn_step_buffers(const ModelGeometry& geometry,
                            const GdnScalarParameters& parameters,
                            const GdnLayerStateView& state,
                            const GdnStepWorkspace& workspace,
                            const float* residual, std::size_t residual_count,
                            float* output, std::size_t output_count) noexcept {
  return parameters.input_norm != nullptr &&
         parameters.input_norm_count == geometry.residual_width &&
         parameters.convolution != nullptr &&
         parameters.convolution_count == geometry.gdn_conv_values() &&
         parameters.folded_a_tiled != nullptr &&
         parameters.folded_a_count == geometry.gdn_value_heads &&
         parameters.dt_bias_tiled != nullptr &&
         parameters.dt_bias_count == geometry.gdn_value_heads &&
         parameters.recurrent_norm != nullptr &&
         parameters.recurrent_norm_count == geometry.gdn_head_width &&
         residual != nullptr && residual_count == geometry.residual_width &&
         output != nullptr && output_count == geometry.residual_width &&
         state.convolution != nullptr &&
         state.convolution_count == geometry.gdn_conv_values() &&
         state.recurrent != nullptr &&
         state.recurrent_count == geometry.gdn_recurrent_values() &&
         workspace.normalized != nullptr &&
         workspace.normalized_count == geometry.residual_width &&
         workspace.convolved_qkv != nullptr &&
         workspace.convolved_qkv_count == geometry.gdn_packed_qkv() &&
         workspace.query != nullptr &&
         workspace.query_count == geometry.gdn_key_width() &&
         workspace.key != nullptr &&
         workspace.key_count == geometry.gdn_key_width() &&
         workspace.value_tiled != nullptr &&
         workspace.value_tiled_count == geometry.gdn_value_width() &&
         workspace.value_grouped != nullptr &&
         workspace.value_grouped_count == geometry.gdn_value_width() &&
         workspace.gate_grouped != nullptr &&
         workspace.gate_grouped_count == geometry.gdn_value_width() &&
         workspace.alpha_grouped != nullptr && workspace.beta_grouped != nullptr &&
         workspace.folded_a_grouped != nullptr &&
         workspace.dt_bias_grouped != nullptr && workspace.log_decay != nullptr &&
         workspace.update_gate != nullptr &&
         workspace.gate_count == geometry.gdn_value_heads &&
         workspace.recurrent_output != nullptr &&
         workspace.recurrent_output_count == geometry.gdn_value_width() &&
         workspace.gated_grouped != nullptr &&
         workspace.gated_grouped_count == geometry.gdn_value_width() &&
         workspace.gated_tiled != nullptr &&
         workspace.gated_tiled_count == geometry.gdn_value_width() &&
         workspace.mixer_output != nullptr &&
         workspace.mixer_output_count == geometry.residual_width;
}

}  // namespace

Status execute_gdn_mixer_step(
    const LayerWeights& weights, const GdnScalarParameters& parameters,
    const float* residual, std::size_t residual_count,
    const GdnLayerStateView& state, const GdnStepWorkspace& workspace,
    float* output, std::size_t output_count) noexcept {
  const ModelGeometry& geometry = weights.geometry;
  if (weights.kind != LayerKind::kGdn ||
      !valid_gdn_step_buffers(geometry, parameters, state, workspace, residual,
                              residual_count, output, output_count)) {
    return {StatusCode::kInvalidArgument,
            "GDN step parameters, state, or workspace are invalid"};
  }
  Status status = rms_norm_scale(residual, parameters.input_norm,
                                 residual_count, workspace.normalized);
  if (status.is_ok()) {
    status = project_gdn_mixer(weights.gdn, workspace.normalized,
                               workspace.normalized_count,
                               workspace.projections);
  }
  if (status.is_ok()) {
    status = causal_depthwise_conv_step(
        geometry.gdn_packed_qkv(), 4, workspace.projections.packed_qkv,
        workspace.projections.packed_qkv_count, parameters.convolution,
        parameters.convolution_count, state.convolution,
        state.convolution_count, workspace.convolved_qkv,
        workspace.convolved_qkv_count);
  }
  if (status.is_ok()) {
    status = split_gdn_qkv(
        workspace.convolved_qkv, workspace.convolved_qkv_count,
        geometry.gdn_key_heads, geometry.gdn_head_width,
        geometry.gdn_value_heads, geometry.gdn_head_width, workspace.query,
        workspace.query_count, workspace.key, workspace.key_count,
        workspace.value_tiled, workspace.value_tiled_count);
  }
  if (status.is_ok()) {
    status = gdn_tiled_to_grouped(workspace.value_tiled, geometry.gdn_key_heads,
                                  geometry.gdn_replicas(),
                                  geometry.gdn_head_width,
                                  workspace.value_grouped,
                                  workspace.value_grouped_count);
  }
  if (status.is_ok()) {
    status = gdn_tiled_to_grouped(
        workspace.projections.value_gate, geometry.gdn_key_heads,
        geometry.gdn_replicas(), geometry.gdn_head_width,
        workspace.gate_grouped, workspace.gate_grouped_count);
  }
  if (status.is_ok()) {
    status = gdn_tiled_to_grouped(workspace.projections.alpha,
                                  geometry.gdn_key_heads,
                                  geometry.gdn_replicas(), 1,
                                  workspace.alpha_grouped,
                                  workspace.gate_count);
  }
  if (status.is_ok()) {
    status = gdn_tiled_to_grouped(workspace.projections.beta,
                                  geometry.gdn_key_heads,
                                  geometry.gdn_replicas(), 1,
                                  workspace.beta_grouped,
                                  workspace.gate_count);
  }
  if (status.is_ok()) {
    status = gdn_tiled_to_grouped(parameters.folded_a_tiled,
                                  geometry.gdn_key_heads,
                                  geometry.gdn_replicas(), 1,
                                  workspace.folded_a_grouped,
                                  workspace.gate_count);
  }
  if (status.is_ok()) {
    status = gdn_tiled_to_grouped(parameters.dt_bias_tiled,
                                  geometry.gdn_key_heads,
                                  geometry.gdn_replicas(), 1,
                                  workspace.dt_bias_grouped,
                                  workspace.gate_count);
  }
  if (status.is_ok()) {
    status = gdn_gates_from_gguf(
        workspace.alpha_grouped, workspace.beta_grouped,
        workspace.folded_a_grouped, workspace.dt_bias_grouped,
        workspace.gate_count, workspace.log_decay, workspace.update_gate);
  }
  const GdnShape shape{geometry.gdn_key_heads, geometry.gdn_value_heads,
                       geometry.gdn_head_width, geometry.gdn_head_width};
  if (status.is_ok()) {
    status = gdn_recurrent_step_precomputed(
        shape, workspace.query, workspace.query_count, workspace.key,
        workspace.key_count, workspace.value_grouped,
        workspace.value_grouped_count, workspace.log_decay,
        workspace.update_gate, workspace.gate_count, state.recurrent,
        state.recurrent_count, workspace.recurrent_output,
        workspace.recurrent_output_count);
  }
  if (status.is_ok()) {
    status = gdn_gated_rms_norm(
        workspace.recurrent_output, workspace.gate_grouped,
        geometry.gdn_value_heads, geometry.gdn_head_width,
        parameters.recurrent_norm, parameters.recurrent_norm_count,
        workspace.gated_grouped, workspace.gated_grouped_count);
  }
  if (status.is_ok()) {
    status = gdn_grouped_to_tiled(
        workspace.gated_grouped, geometry.gdn_key_heads,
        geometry.gdn_replicas(), geometry.gdn_head_width, workspace.gated_tiled,
        workspace.gated_tiled_count);
  }
  if (status.is_ok()) {
    status = tensor_matvec(weights.gdn.output, workspace.gated_tiled,
                           workspace.gated_tiled_count, workspace.mixer_output,
                           workspace.mixer_output_count);
  }
  if (!status.is_ok()) return status;
  for (std::size_t index = 0; index < output_count; ++index) {
    output[index] = residual[index] + workspace.mixer_output[index];
  }
  return Status::ok();
}

Status prepare_ffn_scalar_parameters(
    const CommonLayerWeights& weights,
    const FfnScalarParameters& parameters) noexcept {
  if (parameters.norm == nullptr || parameters.norm_count == 0) {
    return {StatusCode::kInvalidArgument,
            "FFN scalar parameter buffer is invalid"};
  }
  return vector_decode(weights.ffn_norm, parameters.norm,
                       parameters.norm_count);
}

Status validate_ffn_step(const FfnScalarParameters& parameters,
                         const float* residual,
                         std::size_t residual_count,
                         const FfnStepWorkspace& workspace, float* output,
                         std::size_t output_count) noexcept {
  if (parameters.norm == nullptr || parameters.norm_count == 0 ||
      residual == nullptr || residual_count != parameters.norm_count ||
      workspace.normalized == nullptr ||
      workspace.normalized_count != parameters.norm_count ||
      workspace.gate == nullptr || workspace.gate_count == 0 ||
      workspace.up == nullptr ||
      workspace.up_count != workspace.gate_count ||
      workspace.activated == nullptr ||
      workspace.activated_count != workspace.gate_count ||
      workspace.correction == nullptr ||
      workspace.correction_count != parameters.norm_count || output == nullptr ||
      output_count != parameters.norm_count) {
    return {StatusCode::kInvalidArgument,
            "FFN parameters, activation, or workspace are invalid"};
  }
  return Status::ok();
}

Status execute_ffn_step(
    const CommonLayerWeights& weights, const FfnScalarParameters& parameters,
    const float* residual, std::size_t residual_count,
    const FfnStepWorkspace& workspace, float* output,
    std::size_t output_count) noexcept {
  Status status = validate_ffn_step(parameters, residual, residual_count,
                                    workspace, output, output_count);
  if (!status.is_ok()) return status;
  status = rms_norm_scale(residual, parameters.norm, residual_count,
                          workspace.normalized);
  if (status.is_ok()) {
    status = tensor_matvec(weights.ffn_gate, workspace.normalized,
                           workspace.normalized_count, workspace.gate,
                           workspace.gate_count);
  }
  if (status.is_ok()) {
    status = tensor_matvec(weights.ffn_up, workspace.normalized,
                           workspace.normalized_count, workspace.up,
                           workspace.up_count);
  }
  if (!status.is_ok()) return status;
  for (std::size_t index = 0; index < workspace.gate_count; ++index) {
    const float gate = workspace.gate[index];
    const float silu = gate / (1.0F + std::exp(-gate));
    workspace.activated[index] = silu * workspace.up[index];
  }
  status = tensor_matvec(weights.ffn_down, workspace.activated,
                         workspace.activated_count, workspace.correction,
                         workspace.correction_count);
  if (!status.is_ok()) return status;
  for (std::size_t index = 0; index < output_count; ++index) {
    output[index] = residual[index] + workspace.correction[index];
  }
  return Status::ok();
}

Status prepare_attention_scalar_parameters(
    const LayerWeights& weights,
    const AttentionScalarParameters& parameters) noexcept {
  const ModelGeometry& geometry = weights.geometry;
  if (weights.kind != LayerKind::kAttention ||
      parameters.input_norm == nullptr ||
      parameters.input_norm_count != geometry.residual_width ||
      parameters.query_norm == nullptr ||
      parameters.query_norm_count != geometry.attention_head_width ||
      parameters.key_norm == nullptr ||
      parameters.key_norm_count != geometry.attention_head_width) {
    return {StatusCode::kInvalidArgument,
            "attention scalar parameter buffers are invalid"};
  }
  Status status = vector_decode(weights.common.input_norm,
                                parameters.input_norm,
                                parameters.input_norm_count);
  if (status.is_ok()) {
    status = vector_decode(weights.attention.query_norm,
                           parameters.query_norm,
                           parameters.query_norm_count);
  }
  if (status.is_ok()) {
    status = vector_decode(weights.attention.key_norm, parameters.key_norm,
                           parameters.key_norm_count);
  }
  return status;
}

Status execute_attention_mixer_step(
    const LayerWeights& weights, const AttentionScalarParameters& parameters,
    std::size_t position, const float* residual, std::size_t residual_count,
    const AttentionLayerStateView& state,
    const AttentionStepWorkspace& workspace, float* output,
    std::size_t output_count) noexcept {
  const ModelGeometry& geometry = weights.geometry;
  if (geometry.attention_kv_width() == 0 ||
      state.capacity > std::numeric_limits<std::size_t>::max() /
                           geometry.attention_kv_width()) {
    return {StatusCode::kInvalidArgument,
            "attention step capacity overflows its cache size"};
  }
  const std::size_t cache_count =
      state.capacity * geometry.attention_kv_width();
  if (weights.kind != LayerKind::kAttention ||
      parameters.input_norm == nullptr ||
      parameters.input_norm_count != geometry.residual_width ||
      parameters.query_norm == nullptr ||
      parameters.query_norm_count != geometry.attention_head_width ||
      parameters.key_norm == nullptr ||
      parameters.key_norm_count != geometry.attention_head_width ||
      residual == nullptr || residual_count != geometry.residual_width ||
      state.capacity == 0 || position >= state.capacity ||
      state.key_cache == nullptr || state.key_cache_count != cache_count ||
      state.value_cache == nullptr || state.value_cache_count != cache_count ||
      workspace.normalized == nullptr ||
      workspace.normalized_count != geometry.residual_width ||
      workspace.attention_output == nullptr ||
      workspace.attention_output_count != geometry.attention_query_width() ||
      workspace.scores == nullptr || workspace.score_count < state.capacity ||
      workspace.mixer_output == nullptr ||
      workspace.mixer_output_count != geometry.residual_width ||
#ifdef QW38_DIAGNOSTIC_TRACE
      workspace.rope_query == nullptr ||
      workspace.rope_query_count != geometry.attention_query_width() ||
      workspace.rope_key == nullptr ||
      workspace.rope_key_count != geometry.attention_kv_width() ||
#endif
      output == nullptr || output_count != geometry.residual_width) {
    return {StatusCode::kInvalidArgument,
            "attention step parameters, state, or workspace are invalid"};
  }
  Status status = rms_norm_scale(residual, parameters.input_norm,
                                 residual_count, workspace.normalized);
  if (status.is_ok()) {
    status = project_attention_mixer(
        weights.attention, workspace.normalized, workspace.normalized_count,
        workspace.projections);
  }
  if (status.is_ok()) {
    const AttentionShape shape{geometry.attention_query_heads,
                               geometry.attention_kv_heads,
                               geometry.attention_head_width,
                               geometry.rope_dimensions, state.capacity};
    status =
#ifdef QW38_DIAGNOSTIC_TRACE
        attention_decode_step_scale_traced(
#else
        attention_decode_step_scale(
#endif
        shape, position, workspace.projections.query,
        workspace.projections.query_count, workspace.projections.key,
        workspace.projections.key_count, workspace.projections.value,
        workspace.projections.value_count, parameters.query_norm,
        parameters.key_norm, workspace.projections.gate,
        workspace.projections.gate_count, state.key_cache,
        state.key_cache_count, state.value_cache, state.value_cache_count,
        workspace.scores, workspace.score_count, workspace.attention_output,
        workspace.attention_output_count
#ifdef QW38_DIAGNOSTIC_TRACE
        , workspace.rope_query, workspace.rope_query_count, workspace.rope_key,
        workspace.rope_key_count
#endif
        );
  }
  if (status.is_ok()) {
    status = tensor_matvec(weights.attention.output,
                           workspace.attention_output,
                           workspace.attention_output_count,
                           workspace.mixer_output,
                           workspace.mixer_output_count);
  }
  if (!status.is_ok()) return status;
  for (std::size_t index = 0; index < output_count; ++index) {
    output[index] = residual[index] + workspace.mixer_output[index];
  }
  return Status::ok();
}

}  // namespace qw38::internal
