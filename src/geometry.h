#ifndef QW38_GEOMETRY_H_
#define QW38_GEOMETRY_H_

#include <cstddef>
#include <cstdint>

namespace qw38::internal {

constexpr std::size_t kMaximumLayerCount = 64;
constexpr std::size_t kHostDefaultContext = 4096;
constexpr std::size_t kHostMaximumContext = 8192;

struct ModelGeometry final {
  std::size_t layer_count = 0;
  std::size_t gdn_layer_count = 0;
  std::size_t attention_layer_count = 0;
  std::size_t residual_width = 0;
  std::size_t ffn_width = 0;
  std::size_t vocabulary = 0;
  std::size_t gdn_key_heads = 0;
  std::size_t gdn_value_heads = 0;
  std::size_t gdn_head_width = 0;
  std::size_t gdn_conv_width = 0;
  std::size_t attention_query_heads = 0;
  std::size_t attention_kv_heads = 0;
  std::size_t attention_head_width = 0;
  std::size_t rope_dimensions = 0;
  std::size_t expected_tensor_count = 0;
  std::size_t default_context = 0;
  bool tied_embeddings = false;
  std::uint32_t identity = 0;

  std::size_t gdn_key_width() const noexcept {
    return gdn_key_heads * gdn_head_width;
  }
  std::size_t gdn_value_width() const noexcept {
    return gdn_value_heads * gdn_head_width;
  }
  std::size_t gdn_packed_qkv() const noexcept {
    return gdn_key_width() * 2 + gdn_value_width();
  }
  std::size_t gdn_replicas() const noexcept {
    return gdn_key_heads == 0 ? 0 : gdn_value_heads / gdn_key_heads;
  }
  std::size_t gdn_conv_values() const noexcept {
    return gdn_conv_width * gdn_packed_qkv();
  }
  std::size_t gdn_recurrent_values() const noexcept {
    return gdn_value_heads * gdn_head_width * gdn_head_width;
  }
  std::size_t attention_query_width() const noexcept {
    return attention_query_heads * attention_head_width;
  }
  std::size_t attention_kv_width() const noexcept {
    return attention_kv_heads * attention_head_width;
  }
  std::size_t attention_packed_query_gate() const noexcept {
    return attention_query_width() * 2;
  }
};

constexpr std::uint32_t kGeometryQwen38_27B = 1;
constexpr std::uint32_t kGeometryQwen35_2B = 2;

inline ModelGeometry qwen38_27b_geometry() noexcept {
  ModelGeometry geometry;
  geometry.layer_count = 64;
  geometry.gdn_layer_count = 48;
  geometry.attention_layer_count = 16;
  geometry.residual_width = 5120;
  geometry.ffn_width = 17408;
  geometry.vocabulary = 248320;
  geometry.gdn_key_heads = 16;
  geometry.gdn_value_heads = 48;
  geometry.gdn_head_width = 128;
  geometry.gdn_conv_width = 4;
  geometry.attention_query_heads = 24;
  geometry.attention_kv_heads = 4;
  geometry.attention_head_width = 256;
  geometry.rope_dimensions = 64;
  geometry.expected_tensor_count = 851;
  geometry.default_context = 131072;
  geometry.tied_embeddings = false;
  geometry.identity = kGeometryQwen38_27B;
  return geometry;
}

inline ModelGeometry qwen35_2b_geometry() noexcept {
  ModelGeometry geometry;
  geometry.layer_count = 24;
  geometry.gdn_layer_count = 18;
  geometry.attention_layer_count = 6;
  geometry.residual_width = 2048;
  geometry.ffn_width = 6144;
  geometry.vocabulary = 248320;
  geometry.gdn_key_heads = 16;
  geometry.gdn_value_heads = 16;
  geometry.gdn_head_width = 128;
  geometry.gdn_conv_width = 4;
  geometry.attention_query_heads = 8;
  geometry.attention_kv_heads = 2;
  geometry.attention_head_width = 256;
  geometry.rope_dimensions = 64;
  geometry.expected_tensor_count = 320;
  geometry.default_context = kHostDefaultContext;
  geometry.tied_embeddings = true;
  geometry.identity = kGeometryQwen35_2B;
  return geometry;
}

inline bool geometry_is_valid(const ModelGeometry& geometry) noexcept {
  return geometry.layer_count > 0 &&
         geometry.layer_count <= kMaximumLayerCount &&
         geometry.gdn_layer_count + geometry.attention_layer_count ==
             geometry.layer_count &&
         geometry.residual_width > 0 && geometry.ffn_width > 0 &&
         geometry.vocabulary > 0 && geometry.gdn_key_heads > 0 &&
         geometry.gdn_value_heads > 0 &&
         geometry.gdn_value_heads % geometry.gdn_key_heads == 0 &&
         geometry.gdn_head_width > 0 && geometry.gdn_conv_width > 0 &&
         geometry.attention_query_heads > 0 &&
         geometry.attention_kv_heads > 0 &&
         geometry.attention_query_heads % geometry.attention_kv_heads == 0 &&
         geometry.attention_head_width > 0 && geometry.rope_dimensions > 0 &&
         geometry.rope_dimensions % 2 == 0 &&
         geometry.default_context > 0;
}

}  // namespace qw38::internal

#endif  // QW38_GEOMETRY_H_
