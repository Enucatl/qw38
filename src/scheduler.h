#ifndef QW38_SCHEDULER_H_
#define QW38_SCHEDULER_H_

#include <cstddef>

#include "mixer.h"
#include "qw38/status.h"
#include "weights.h"

namespace qw38::internal {

constexpr std::size_t kVocabularySize = 248320;

struct GdnLayerScalarParameters final {
  GdnScalarParameters mixer;
  FfnScalarParameters ffn;
};

struct AttentionLayerScalarParameters final {
  AttentionScalarParameters mixer;
  FfnScalarParameters ffn;
};

struct GdnLayerWorkspace final {
  GdnStepWorkspace mixer;
  FfnStepWorkspace ffn;
  float* post_mixer;
  std::size_t post_mixer_count;
};

struct AttentionLayerWorkspace final {
  AttentionStepWorkspace mixer;
  FfnStepWorkspace ffn;
  float* post_mixer;
  std::size_t post_mixer_count;
};

struct OutputScalarParameters final {
  float* norm;
  std::size_t norm_count;
};

struct OutputWorkspace final {
  float* normalized;
  std::size_t normalized_count;
};

Status embed_token(const ModelWeights& weights, std::size_t token,
                   float* output, std::size_t output_count) noexcept;

Status prepare_output_scalar_parameters(
    const ModelWeights& weights,
    const OutputScalarParameters& parameters) noexcept;

Status project_logits(const ModelWeights& weights,
                      const OutputScalarParameters& parameters,
                      const float* hidden, std::size_t hidden_count,
                      const OutputWorkspace& workspace, float* logits,
                      std::size_t logits_count) noexcept;

Status prepare_gdn_layer_scalar_parameters(
    const LayerWeights& weights,
    const GdnLayerScalarParameters& parameters) noexcept;

Status prepare_attention_layer_scalar_parameters(
    const LayerWeights& weights,
    const AttentionLayerScalarParameters& parameters) noexcept;

Status execute_gdn_layer_step(
    const LayerWeights& weights, const GdnLayerScalarParameters& parameters,
    const float* input, std::size_t input_count,
    const GdnLayerStateView& state, const GdnLayerWorkspace& workspace,
    float* output, std::size_t output_count) noexcept;

Status execute_attention_layer_step(
    const LayerWeights& weights,
    const AttentionLayerScalarParameters& parameters, std::size_t position,
    const float* input, std::size_t input_count,
    const AttentionLayerStateView& state,
    const AttentionLayerWorkspace& workspace, float* output,
    std::size_t output_count) noexcept;

}  // namespace qw38::internal

#endif  // QW38_SCHEDULER_H_
