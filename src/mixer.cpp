#include "mixer.h"

#include <cmath>

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
  if (activation == nullptr || activation_count != kResidualWidth ||
      workspace.packed_qkv == nullptr ||
      workspace.packed_qkv_count != kGdnPackedQkvWidth ||
      workspace.value_gate == nullptr ||
      workspace.value_gate_count != kGdnValueWidth ||
      workspace.alpha == nullptr || workspace.alpha_count != kGdnGateCount ||
      workspace.beta == nullptr || workspace.beta_count != kGdnGateCount) {
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
  if (activation == nullptr || activation_count != kResidualWidth ||
      workspace.packed_query_gate == nullptr ||
      workspace.packed_query_gate_count != kAttentionPackedQueryGateWidth ||
      workspace.query == nullptr ||
      workspace.query_count != kAttentionQueryWidth || workspace.gate == nullptr ||
      workspace.gate_count != kAttentionQueryWidth || workspace.key == nullptr ||
      workspace.key_count != kAttentionKvWidth || workspace.value == nullptr ||
      workspace.value_count != kAttentionKvWidth) {
    return {StatusCode::kInvalidArgument,
            "attention projection activation or workspace is invalid"};
  }
  Status status = tensor_matvec(
      weights.query_gate, activation, activation_count,
      workspace.packed_query_gate, workspace.packed_query_gate_count);
  if (status.is_ok()) {
    status = split_attention_query_gate(
        workspace.packed_query_gate, workspace.packed_query_gate_count, 24, 256,
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
  if (weights.kind != LayerKind::kGdn || parameters.input_norm == nullptr ||
      parameters.input_norm_count != kResidualWidth ||
      parameters.convolution == nullptr ||
      parameters.convolution_count != kGdnConvolutionValues ||
      parameters.folded_a_tiled == nullptr ||
      parameters.folded_a_count != kGdnGateCount ||
      parameters.dt_bias_tiled == nullptr ||
      parameters.dt_bias_count != kGdnGateCount ||
      parameters.recurrent_norm == nullptr ||
      parameters.recurrent_norm_count != kGdnHeadWidth) {
    return {StatusCode::kInvalidArgument,
            "GDN scalar parameter buffers are invalid"};
  }
  Status status = vector_decode(weights.common.input_norm, parameters.input_norm,
                                parameters.input_norm_count);
  for (std::size_t channel = 0;
       status.is_ok() && channel < kGdnPackedQkvWidth; ++channel) {
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

bool valid_gdn_step_buffers(const GdnScalarParameters& parameters,
                            const GdnLayerStateView& state,
                            const GdnStepWorkspace& workspace,
                            const float* residual, std::size_t residual_count,
                            float* output, std::size_t output_count) noexcept {
  return parameters.input_norm != nullptr &&
         parameters.input_norm_count == kResidualWidth &&
         parameters.convolution != nullptr &&
         parameters.convolution_count == kGdnConvolutionValues &&
         parameters.folded_a_tiled != nullptr &&
         parameters.folded_a_count == kGdnGateCount &&
         parameters.dt_bias_tiled != nullptr &&
         parameters.dt_bias_count == kGdnGateCount &&
         parameters.recurrent_norm != nullptr &&
         parameters.recurrent_norm_count == kGdnHeadWidth && residual != nullptr &&
         residual_count == kResidualWidth && output != nullptr &&
         output_count == kResidualWidth && state.convolution != nullptr &&
         state.convolution_count == kGdnConvolutionValues &&
         state.recurrent != nullptr &&
         state.recurrent_count == kGdnRecurrentStateValues &&
         workspace.normalized != nullptr &&
         workspace.normalized_count == kResidualWidth &&
         workspace.convolved_qkv != nullptr &&
         workspace.convolved_qkv_count == kGdnPackedQkvWidth &&
         workspace.query != nullptr && workspace.query_count == kGdnKeyWidth &&
         workspace.key != nullptr && workspace.key_count == kGdnKeyWidth &&
         workspace.value_tiled != nullptr &&
         workspace.value_tiled_count == kGdnValueWidth &&
         workspace.value_grouped != nullptr &&
         workspace.value_grouped_count == kGdnValueWidth &&
         workspace.gate_grouped != nullptr &&
         workspace.gate_grouped_count == kGdnValueWidth &&
         workspace.alpha_grouped != nullptr && workspace.beta_grouped != nullptr &&
         workspace.folded_a_grouped != nullptr &&
         workspace.dt_bias_grouped != nullptr && workspace.log_decay != nullptr &&
         workspace.update_gate != nullptr &&
         workspace.gate_count == kGdnGateCount &&
         workspace.recurrent_output != nullptr &&
         workspace.recurrent_output_count == kGdnValueWidth &&
         workspace.gated_grouped != nullptr &&
         workspace.gated_grouped_count == kGdnValueWidth &&
         workspace.gated_tiled != nullptr &&
         workspace.gated_tiled_count == kGdnValueWidth &&
         workspace.mixer_output != nullptr &&
         workspace.mixer_output_count == kResidualWidth;
}

}  // namespace

Status execute_gdn_mixer_step(
    const LayerWeights& weights, const GdnScalarParameters& parameters,
    const float* residual, std::size_t residual_count,
    const GdnLayerStateView& state, const GdnStepWorkspace& workspace,
    float* output, std::size_t output_count) noexcept {
  if (weights.kind != LayerKind::kGdn ||
      !valid_gdn_step_buffers(parameters, state, workspace, residual,
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
        kGdnPackedQkvWidth, 4, workspace.projections.packed_qkv,
        workspace.projections.packed_qkv_count, parameters.convolution,
        parameters.convolution_count, state.convolution,
        state.convolution_count, workspace.convolved_qkv,
        workspace.convolved_qkv_count);
  }
  if (status.is_ok()) {
    status = split_gdn_qkv(
        workspace.convolved_qkv, workspace.convolved_qkv_count, 16, 128, 48,
        128, workspace.query, workspace.query_count, workspace.key,
        workspace.key_count, workspace.value_tiled,
        workspace.value_tiled_count);
  }
  if (status.is_ok()) {
    status = gdn_tiled_to_grouped(workspace.value_tiled, 16, 3, 128,
                                  workspace.value_grouped,
                                  workspace.value_grouped_count);
  }
  if (status.is_ok()) {
    status = gdn_tiled_to_grouped(
        workspace.projections.value_gate, 16, 3, 128,
        workspace.gate_grouped, workspace.gate_grouped_count);
  }
  if (status.is_ok()) {
    status = gdn_tiled_to_grouped(workspace.projections.alpha, 16, 3, 1,
                                  workspace.alpha_grouped,
                                  workspace.gate_count);
  }
  if (status.is_ok()) {
    status = gdn_tiled_to_grouped(workspace.projections.beta, 16, 3, 1,
                                  workspace.beta_grouped,
                                  workspace.gate_count);
  }
  if (status.is_ok()) {
    status = gdn_tiled_to_grouped(parameters.folded_a_tiled, 16, 3, 1,
                                  workspace.folded_a_grouped,
                                  workspace.gate_count);
  }
  if (status.is_ok()) {
    status = gdn_tiled_to_grouped(parameters.dt_bias_tiled, 16, 3, 1,
                                  workspace.dt_bias_grouped,
                                  workspace.gate_count);
  }
  if (status.is_ok()) {
    status = gdn_gates_from_gguf(
        workspace.alpha_grouped, workspace.beta_grouped,
        workspace.folded_a_grouped, workspace.dt_bias_grouped,
        workspace.gate_count, workspace.log_decay, workspace.update_gate);
  }
  constexpr GdnShape kShape{16, 48, 128, 128};
  if (status.is_ok()) {
    status = gdn_recurrent_step_precomputed(
        kShape, workspace.query, workspace.query_count, workspace.key,
        workspace.key_count, workspace.value_grouped,
        workspace.value_grouped_count, workspace.log_decay,
        workspace.update_gate, workspace.gate_count, state.recurrent,
        state.recurrent_count, workspace.recurrent_output,
        workspace.recurrent_output_count);
  }
  if (status.is_ok()) {
    status = gdn_gated_rms_norm(
        workspace.recurrent_output, workspace.gate_grouped, 48, 128,
        parameters.recurrent_norm, parameters.recurrent_norm_count,
        workspace.gated_grouped, workspace.gated_grouped_count);
  }
  if (status.is_ok()) {
    status = gdn_grouped_to_tiled(
        workspace.gated_grouped, 16, 3, 128, workspace.gated_tiled,
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
  if (parameters.norm == nullptr || parameters.norm_count != kResidualWidth) {
    return {StatusCode::kInvalidArgument,
            "FFN scalar parameter buffer is invalid"};
  }
  return vector_decode(weights.ffn_norm, parameters.norm,
                       parameters.norm_count);
}

Status execute_ffn_step(
    const CommonLayerWeights& weights, const FfnScalarParameters& parameters,
    const float* residual, std::size_t residual_count,
    const FfnStepWorkspace& workspace, float* output,
    std::size_t output_count) noexcept {
  if (parameters.norm == nullptr || parameters.norm_count != kResidualWidth ||
      residual == nullptr || residual_count != kResidualWidth ||
      workspace.normalized == nullptr ||
      workspace.normalized_count != kResidualWidth || workspace.gate == nullptr ||
      workspace.gate_count != kFfnWidth || workspace.up == nullptr ||
      workspace.up_count != kFfnWidth || workspace.activated == nullptr ||
      workspace.activated_count != kFfnWidth ||
      workspace.correction == nullptr ||
      workspace.correction_count != kResidualWidth || output == nullptr ||
      output_count != kResidualWidth) {
    return {StatusCode::kInvalidArgument,
            "FFN parameters, activation, or workspace are invalid"};
  }
  Status status = rms_norm_scale(residual, parameters.norm, residual_count,
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
  for (std::size_t index = 0; index < kFfnWidth; ++index) {
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

}  // namespace qw38::internal
