#include "mixer.h"

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

}  // namespace qw38::internal
