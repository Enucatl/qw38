#include "weights.h"

#include <algorithm>
#include <string>

namespace qw38::internal {
namespace {

constexpr std::size_t kExpectedTensorCount = 851;
constexpr std::uint32_t kF32 = 0;
constexpr std::uint32_t kQ80 = 8;
constexpr std::uint32_t kQ4K = 12;
constexpr std::uint32_t kQ6K = 14;

const TensorInfo* find_tensor(const ModelInfo& info,
                              const std::string& name) noexcept {
  const auto match = std::find_if(
      info.tensors.begin(), info.tensors.end(), [&name](const TensorInfo& tensor) {
        return tensor.name == name;
      });
  return match == info.tensors.end() ? nullptr : &*match;
}

Status bind_matrix(const ModelInfo& info, const MappedFile& mapping,
                   const std::string& name, const char* role,
                   std::size_t columns, std::size_t rows, std::uint32_t type,
                   TensorView* view, std::size_t* bound_count) noexcept {
  const TensorInfo* tensor = find_tensor(info, name);
  if (tensor == nullptr || tensor->semantic_role != role ||
      tensor->dimensions.size() != 2 || tensor->dimensions[0] != columns ||
      tensor->dimensions[1] != rows || tensor->type != type) {
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
                   const std::string& prefix, CommonLayerWeights* common,
                   std::size_t* count) noexcept {
  Status status = bind_vector(info, mapping, prefix + "attn_norm.weight",
                              "input_norm", 5120, &common->input_norm, count);
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, prefix + "ffn_gate.weight", "ffn_gate",
                         5120, 17408, kQ4K, &common->ffn_gate, count);
  }
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, prefix + "ffn_up.weight", "ffn_up",
                         5120, 17408, kQ4K, &common->ffn_up, count);
  }
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, prefix + "ffn_down.weight", "ffn_down",
                         17408, 5120, kQ4K, &common->ffn_down, count);
  }
  if (status.is_ok()) {
    status = bind_vector(info, mapping, prefix + "post_attention_norm.weight",
                         "ffn_norm", 5120, &common->ffn_norm, count);
  }
  return status;
}

Status bind_gdn(const ModelInfo& info, const MappedFile& mapping,
                const std::string& prefix, GdnLayerWeights* gdn,
                std::size_t* count) noexcept {
  Status status = bind_matrix(info, mapping, prefix + "attn_qkv.weight",
                              "gdn_packed_qkv", 5120, 10240, kQ80,
                              &gdn->packed_qkv, count);
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, prefix + "attn_gate.weight",
                         "gdn_value_gate", 5120, 6144, kQ80,
                         &gdn->value_gate, count);
  }
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, prefix + "ssm_alpha.weight",
                         "gdn_alpha", 5120, 48, kQ80, &gdn->alpha, count);
  }
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, prefix + "ssm_beta.weight", "gdn_beta",
                         5120, 48, kQ80, &gdn->beta, count);
  }
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, prefix + "ssm_conv1d.weight",
                         "gdn_convolution", 4, 10240, kF32,
                         &gdn->convolution, count);
  }
  if (status.is_ok()) {
    status = bind_vector(info, mapping, prefix + "ssm_a", "gdn_decay", 48,
                         &gdn->folded_a, count);
  }
  if (status.is_ok()) {
    status = bind_vector(info, mapping, prefix + "ssm_dt.bias", "gdn_dt_bias",
                         48, &gdn->dt_bias, count);
  }
  if (status.is_ok()) {
    status = bind_vector(info, mapping, prefix + "ssm_norm.weight", "gdn_norm",
                         128, &gdn->norm, count);
  }
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, prefix + "ssm_out.weight", "gdn_output",
                         6144, 5120, kQ80, &gdn->output, count);
  }
  return status;
}

Status bind_attention(const ModelInfo& info, const MappedFile& mapping,
                      const std::string& prefix,
                      AttentionLayerWeights* attention,
                      std::size_t* count) noexcept {
  Status status = bind_matrix(info, mapping, prefix + "attn_q.weight",
                              "attention_q_gate", 5120, 12288, kQ80,
                              &attention->query_gate, count);
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, prefix + "attn_k.weight", "attention_k",
                         5120, 1024, kQ80, &attention->key, count);
  }
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, prefix + "attn_v.weight", "attention_v",
                         5120, 1024, kQ80, &attention->value, count);
  }
  if (status.is_ok()) {
    status = bind_vector(info, mapping, prefix + "attn_q_norm.weight",
                         "attention_q_norm", 256, &attention->query_norm,
                         count);
  }
  if (status.is_ok()) {
    status = bind_vector(info, mapping, prefix + "attn_k_norm.weight",
                         "attention_k_norm", 256, &attention->key_norm, count);
  }
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, prefix + "attn_output.weight",
                         "attention_output", 6144, 5120, kQ6K,
                         &attention->output, count);
  }
  return status;
}

}  // namespace

Status bind_model_weights(const ModelInfo& info, const MappedFile& mapping,
                          ModelWeights* weights) noexcept {
  if (weights == nullptr) {
    return {StatusCode::kInvalidArgument, "model weights output is required"};
  }
  if (info.block_count != kModelLayerCount ||
      info.tensors.size() != kExpectedTensorCount) {
    return {StatusCode::kInvalidArgument,
            "model does not contain the exact typed weight schema"};
  }

  ModelWeights candidate;
  std::size_t count = 0;
  Status status = bind_matrix(info, mapping, "token_embd.weight",
                              "token_embedding", 5120, 248320, kQ4K,
                              &candidate.token_embedding, &count);
  if (status.is_ok()) {
    status = bind_vector(info, mapping, "output_norm.weight", "final_norm", 5120,
                         &candidate.output_norm, &count);
  }
  if (status.is_ok()) {
    status = bind_matrix(info, mapping, "output.weight", "output_projection",
                         5120, 248320, kQ6K, &candidate.output, &count);
  }
  for (std::size_t layer = 0; status.is_ok() && layer < kModelLayerCount;
       ++layer) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    LayerWeights& destination = candidate.layers[layer];
    status = bind_common(info, mapping, prefix, &destination.common, &count);
    if (!status.is_ok()) break;
    if (layer % 4 == 3) {
      destination.kind = LayerKind::kAttention;
      status = bind_attention(info, mapping, prefix, &destination.attention,
                              &count);
    } else {
      destination.kind = LayerKind::kGdn;
      status = bind_gdn(info, mapping, prefix, &destination.gdn, &count);
    }
  }
  if (!status.is_ok()) return status;
  if (count != kExpectedTensorCount) {
    return {StatusCode::kInternal,
            "typed model binding did not consume every admitted tensor"};
  }
  candidate.bound_tensor_count = count;
  *weights = candidate;
  return Status::ok();
}

}  // namespace qw38::internal
