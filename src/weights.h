#ifndef QW38_WEIGHTS_H_
#define QW38_WEIGHTS_H_

#include <array>
#include <cstddef>

#include "geometry.h"
#include "model.h"
#include "qw38/status.h"
#include "tensor.h"

namespace qw38::internal {

constexpr std::size_t kModelLayerCount = kMaximumLayerCount;

enum class LayerKind { kGdn, kAttention };

struct CommonLayerWeights final {
  VectorView input_norm;
  TensorView ffn_gate;
  TensorView ffn_up;
  TensorView ffn_down;
  VectorView ffn_norm;
};

struct GdnLayerWeights final {
  TensorView packed_qkv;
  TensorView value_gate;
  TensorView alpha;
  TensorView beta;
  TensorView convolution;
  VectorView folded_a;
  VectorView dt_bias;
  VectorView norm;
  TensorView output;
};

struct AttentionLayerWeights final {
  TensorView query_gate;
  TensorView key;
  TensorView value;
  VectorView query_norm;
  VectorView key_norm;
  TensorView output;
};

struct LayerWeights final {
  LayerKind kind = LayerKind::kGdn;
  ModelGeometry geometry{};
  CommonLayerWeights common;
  GdnLayerWeights gdn;
  AttentionLayerWeights attention;
};

struct ModelWeights final {
  ModelGeometry geometry{};
  TensorView token_embedding;
  VectorView output_norm;
  TensorView output;
  std::array<LayerWeights, kModelLayerCount> layers;
  std::size_t bound_tensor_count = 0;
  std::size_t bound_layers = 0;
};

Status bind_model_weights(const ModelInfo& info, const MappedFile& mapping,
                          ModelWeights* weights) noexcept;

}  // namespace qw38::internal

#endif  // QW38_WEIGHTS_H_
