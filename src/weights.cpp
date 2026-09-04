#include "weights.h"

#include <algorithm>
#include <string>

namespace qw38::internal {
namespace {

constexpr std::uint32_t kF32 = 0;
constexpr std::uint32_t kQ80 = 8;
constexpr std::uint32_t kQ4K = 12;
constexpr std::uint32_t kQ5K = 13;
constexpr std::uint32_t kQ6K = 14;

bool allowed_quant(const ModelGeometry& geometry, std::uint32_t type,
                   std::uint32_t exact) noexcept {
  if (exact == kF32) return type == kF32;
  if (geometry.identity == kGeometryQwen38_27B) return type == exact;
  return type == kQ80 || type == kQ4K || type == kQ5K || type == kQ6K;
}

Status geometry_from_info(const ModelInfo& info, ModelGeometry* geometry) noexcept {
  if (geometry == nullptr) {
    return {StatusCode::kInvalidArgument, "geometry output is required"};
  }
  if (info.block_count == 64 && info.embedding_length == 5120) {
    *geometry = qwen38_27b_geometry();
    return Status::ok();
  }
  if (info.block_count == 24 && info.embedding_length == 2048) {
    *geometry = qwen35_2b_geometry();
    if (info.tensors.size() == geometry->expected_tensor_count + 1) {
      geometry->expected_tensor_count += 1;
      geometry->tied_embeddings = false;
    }
    return Status::ok();
  }
  return {StatusCode::kIncompatibleArtifact,
          "GGUF does not match an admitted host geometry"};
}

const TensorInfo* find_tensor(const ModelInfo& info,
                              const std::string& name) noexcept {
  const auto match = std::find_if(
      info.tensors.begin(), info.tensors.end(), [&name](const TensorInfo& tensor) {
        return tensor.name == name;
      });
  return match == info.tensors.end() ? nullptr : &*match;
}

Status bind_matrix(const ModelInfo& info, const MappedFile& mapping,
                   const ModelGeometry& geometry, const std::string& name,
                   const char* role, std::size_t columns, std::size_t rows,
                   std::uint32_t exact_type, TensorView* view,
                   std::size_t* bound_count) noexcept {
  const TensorInfo* tensor = find_tensor(info, name);
  if (tensor == nullptr || tensor->semantic_role != role ||
      tensor->dimensions.size() != 2 || tensor->dimensions[0] != columns ||
      tensor->dimensions[1] != rows ||
      !allowed_quant(geometry, tensor->type, exact_type)) {
    return {StatusCode::kInvalidArgument,
            "matrix does not match its admitted typed weight role"};
  }
  Status status = bind_tensor_view(info, mapping, name, view);
  if (status.is_ok()) ++*bound_count;
  return status;
}

Status bind_vector(const ModelInfo& info, const MappedFile& mapping,
                   const std::string& name, const char* role,
                   std::size_t count, VectorView* view,
                   std::size_t* bound_count) noexcept {
  const TensorInfo* tensor = find_tensor(info, name);
  if (tensor == nullptr || tensor->semantic_role != role ||
      tensor->dimensions.size() != 1 || tensor->dimensions[0] != count ||
      tensor->type != kF32) {
    return {StatusCode::kInvalidArgument,
            "vector does not match its admitted typed weight role"};
  }
  Status status = bind_vector_view(info, mapping, name, view);
  if (status.is_ok()) ++*bound_count;
  return status;
}

Status bind_common(const ModelInfo& info, const MappedFile& mapping,
                   const ModelGeometry& geometry, const std::string& prefix,
                   CommonLayerWeights* common, std::size_t* count) noexcept {
  Status status =
      bind_vector(info, mapping, prefix + "attn_norm.weight", "input_norm",
                  geometry.residual_width, &common->input_norm, count);
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, geometry, prefix + "ffn_gate.weight",
                         "ffn_gate", geometry.residual_width, geometry.ffn_width,
                         kQ4K, &common->ffn_gate, count);
  }
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, geometry, prefix + "ffn_up.weight",
                         "ffn_up", geometry.residual_width, geometry.ffn_width,
                         kQ4K, &common->ffn_up, count);
  }
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, geometry, prefix + "ffn_down.weight",
                         "ffn_down", geometry.ffn_width, geometry.residual_width,
                         kQ4K, &common->ffn_down, count);
  }
  if (status.is_ok()) {
    status =
        bind_vector(info, mapping, prefix + "post_attention_norm.weight",
                    "ffn_norm", geometry.residual_width, &common->ffn_norm,
                    count);
  }
  return status;
}

Status bind_gdn(const ModelInfo& info, const MappedFile& mapping,
                const ModelGeometry& geometry, const std::string& prefix,
                GdnLayerWeights* gdn, std::size_t* count) noexcept {
  Status status = bind_matrix(
      info, mapping, geometry, prefix + "attn_qkv.weight", "gdn_packed_qkv",
      geometry.residual_width, geometry.gdn_packed_qkv(), kQ80,
      &gdn->packed_qkv, count);
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, geometry, prefix + "attn_gate.weight",
                         "gdn_value_gate", geometry.residual_width,
                         geometry.gdn_value_width(), kQ80, &gdn->value_gate,
                         count);
  }
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, geometry, prefix + "ssm_alpha.weight",
                         "gdn_alpha", geometry.residual_width,
                         geometry.gdn_value_heads, kQ80, &gdn->alpha, count);
  }
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, geometry, prefix + "ssm_beta.weight",
                         "gdn_beta", geometry.residual_width,
                         geometry.gdn_value_heads, kQ80, &gdn->beta, count);
  }
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, geometry, prefix + "ssm_conv1d.weight",
                         "gdn_convolution", geometry.gdn_conv_width,
                         geometry.gdn_packed_qkv(), kF32, &gdn->convolution,
                         count);
  }
  if (status.is_ok()) {
    status = bind_vector(info, mapping, prefix + "ssm_a", "gdn_decay",
                         geometry.gdn_value_heads, &gdn->folded_a, count);
  }
  if (status.is_ok()) {
    status = bind_vector(info, mapping, prefix + "ssm_dt.bias", "gdn_dt_bias",
                         geometry.gdn_value_heads, &gdn->dt_bias, count);
  }
  if (status.is_ok()) {
    status = bind_vector(info, mapping, prefix + "ssm_norm.weight", "gdn_norm",
                         geometry.gdn_head_width, &gdn->norm, count);
  }
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, geometry, prefix + "ssm_out.weight",
                         "gdn_output", geometry.gdn_value_width(),
                         geometry.residual_width, kQ80, &gdn->output, count);
  }
  return status;
}

Status bind_attention(const ModelInfo& info, const MappedFile& mapping,
                      const ModelGeometry& geometry, const std::string& prefix,
                      AttentionLayerWeights* attention,
                      std::size_t* count) noexcept {
  Status status = bind_matrix(
      info, mapping, geometry, prefix + "attn_q.weight", "attention_q_gate",
      geometry.residual_width, geometry.attention_packed_query_gate(), kQ80,
      &attention->query_gate, count);
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, geometry, prefix + "attn_k.weight",
                         "attention_k", geometry.residual_width,
                         geometry.attention_kv_width(), kQ80, &attention->key,
                         count);
  }
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, geometry, prefix + "attn_v.weight",
                         "attention_v", geometry.residual_width,
                         geometry.attention_kv_width(), kQ80, &attention->value,
                         count);
  }
  if (status.is_ok()) {
    status = bind_vector(info, mapping, prefix + "attn_q_norm.weight",
                         "attention_q_norm", geometry.attention_head_width,
                         &attention->query_norm, count);
  }
  if (status.is_ok()) {
    status = bind_vector(info, mapping, prefix + "attn_k_norm.weight",
                         "attention_k_norm", geometry.attention_head_width,
                         &attention->key_norm, count);
  }
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, geometry, prefix + "attn_output.weight",
                         "attention_output", geometry.attention_query_width(),
                         geometry.residual_width, kQ6K, &attention->output,
                         count);
  }
  return status;
}

}  // namespace

Status bind_model_weights(const ModelInfo& info, const MappedFile& mapping,
                          ModelWeights* weights) noexcept {
  if (weights == nullptr) {
    return {StatusCode::kInvalidArgument, "model weights output is required"};
  }
  ModelGeometry geometry;
  Status status = geometry_from_info(info, &geometry);
  if (!status.is_ok()) return status;
  if (info.block_count != geometry.layer_count ||
      info.tensors.size() != geometry.expected_tensor_count) {
    return {StatusCode::kInvalidArgument,
            "model does not contain the exact typed weight schema"};
  }

  ModelWeights candidate;
  candidate.geometry = geometry;
  std::size_t count = 0;
  status = bind_matrix(info, mapping, geometry, "token_embd.weight",
                       "token_embedding", geometry.residual_width,
                       geometry.vocabulary, kQ4K, &candidate.token_embedding,
                       &count);
  if (status.is_ok()) {
    status = bind_vector(info, mapping, "output_norm.weight", "final_norm",
                         geometry.residual_width, &candidate.output_norm,
                         &count);
  }
  if (status.is_ok()) {
    if (geometry.tied_embeddings &&
        find_tensor(info, "output.weight") == nullptr) {
      candidate.output = candidate.token_embedding;
    } else {
      status = bind_matrix(info, mapping, geometry, "output.weight",
                           "output_projection", geometry.residual_width,
                           geometry.vocabulary, kQ6K, &candidate.output,
                           &count);
    }
  }
  for (std::size_t layer = 0; status.is_ok() && layer < geometry.layer_count;
       ++layer) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    LayerWeights& destination = candidate.layers[layer];
    destination.geometry = geometry;
    status =
        bind_common(info, mapping, geometry, prefix, &destination.common, &count);
    if (!status.is_ok()) break;
    if (layer % 4 == 3) {
      destination.kind = LayerKind::kAttention;
      status = bind_attention(info, mapping, geometry, prefix,
                              &destination.attention, &count);
    } else {
      destination.kind = LayerKind::kGdn;
      status =
          bind_gdn(info, mapping, geometry, prefix, &destination.gdn, &count);
    }
  }
  if (!status.is_ok()) return status;
  if (count != geometry.expected_tensor_count) {
    return {StatusCode::kInternal,
            "typed model binding did not consume every admitted tensor"};
  }
  candidate.bound_tensor_count = count;
  candidate.bound_layers = geometry.layer_count;
  *weights = candidate;
  return Status::ok();
}

}  // namespace qw38::internal
