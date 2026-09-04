#include "scheduler.h"

#include "attention.h"

namespace qw38::internal {
namespace {

Status validate_layer_ffn(const FfnScalarParameters& parameters,
                          const FfnStepWorkspace& workspace,
                          float* post_mixer,
                          std::size_t post_mixer_count, float* output,
                          std::size_t output_count) noexcept {
  if (post_mixer == nullptr ||
      post_mixer_count != parameters.norm_count) {
    return {StatusCode::kInvalidArgument,
            "layer post-mixer buffer is invalid"};
  }
  return validate_ffn_step(parameters, post_mixer, post_mixer_count, workspace,
                           output, output_count);
}

}  // namespace

Status embed_token(const ModelWeights& weights, std::size_t token,
                   float* output, std::size_t output_count) noexcept {
  if (token >= weights.geometry.vocabulary || output == nullptr ||
      output_count != weights.geometry.residual_width) {
    return {StatusCode::kInvalidArgument,
            "token ID or embedding output is invalid"};
  }
  return tensor_row_decode(weights.token_embedding, token, output,
                           output_count);
}

Status prepare_output_scalar_parameters(
    const ModelWeights& weights,
    const OutputScalarParameters& parameters) noexcept {
  if (parameters.norm == nullptr ||
      parameters.norm_count != weights.geometry.residual_width) {
    return {StatusCode::kInvalidArgument,
            "output scalar parameter buffer is invalid"};
  }
  return vector_decode(weights.output_norm, parameters.norm,
                       parameters.norm_count);
}

Status project_logits(const ModelWeights& weights,
                      const OutputScalarParameters& parameters,
                      const float* hidden, std::size_t hidden_count,
                      const OutputWorkspace& workspace, float* logits,
                      std::size_t logits_count) noexcept {
  if (parameters.norm == nullptr ||
      parameters.norm_count != weights.geometry.residual_width ||
      hidden == nullptr || hidden_count != weights.geometry.residual_width ||
      workspace.normalized == nullptr ||
      workspace.normalized_count != weights.geometry.residual_width ||
      logits == nullptr || logits_count != weights.geometry.vocabulary) {
    return {StatusCode::kInvalidArgument,
            "final hidden, output workspace, or logits are invalid"};
  }
  Status status = rms_norm_scale(hidden, parameters.norm, hidden_count,
                                 workspace.normalized);
  if (status.is_ok()) {
    status = tensor_matvec(weights.output, workspace.normalized,
                           workspace.normalized_count, logits, logits_count);
  }
  return status;
}

Status prepare_gdn_layer_scalar_parameters(
    const LayerWeights& weights,
    const GdnLayerScalarParameters& parameters) noexcept {
  Status status = prepare_gdn_scalar_parameters(weights, parameters.mixer);
  if (status.is_ok()) {
    status = prepare_ffn_scalar_parameters(weights.common, parameters.ffn);
  }
  return status;
}

Status prepare_attention_layer_scalar_parameters(
    const LayerWeights& weights,
    const AttentionLayerScalarParameters& parameters) noexcept {
  Status status =
      prepare_attention_scalar_parameters(weights, parameters.mixer);
  if (status.is_ok()) {
    status = prepare_ffn_scalar_parameters(weights.common, parameters.ffn);
  }
  return status;
}

Status execute_gdn_layer_step(
    const LayerWeights& weights, const GdnLayerScalarParameters& parameters,
    const float* input, std::size_t input_count,
    const GdnLayerStateView& state, const GdnLayerWorkspace& workspace,
    float* output, std::size_t output_count) noexcept {
  Status status = validate_layer_ffn(
      parameters.ffn, workspace.ffn, workspace.post_mixer,
      workspace.post_mixer_count, output, output_count);
  if (!status.is_ok()) return status;
  status = execute_gdn_mixer_step(
      weights, parameters.mixer, input, input_count, state, workspace.mixer,
      workspace.post_mixer, workspace.post_mixer_count);
  if (!status.is_ok()) return status;
  return execute_ffn_step(weights.common, parameters.ffn,
                          workspace.post_mixer, workspace.post_mixer_count,
                          workspace.ffn, output, output_count);
}

Status execute_attention_layer_step(
    const LayerWeights& weights,
    const AttentionLayerScalarParameters& parameters, std::size_t position,
    const float* input, std::size_t input_count,
    const AttentionLayerStateView& state,
    const AttentionLayerWorkspace& workspace, float* output,
    std::size_t output_count) noexcept {
  Status status = validate_layer_ffn(
      parameters.ffn, workspace.ffn, workspace.post_mixer,
      workspace.post_mixer_count, output, output_count);
  if (!status.is_ok()) return status;
  status = execute_attention_mixer_step(
      weights, parameters.mixer, position, input, input_count, state,
      workspace.mixer, workspace.post_mixer, workspace.post_mixer_count);
  if (!status.is_ok()) return status;
  return execute_ffn_step(weights.common, parameters.ffn,
                          workspace.post_mixer, workspace.post_mixer_count,
                          workspace.ffn, output, output_count);
}

}  // namespace qw38::internal
