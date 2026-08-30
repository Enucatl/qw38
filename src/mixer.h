#ifndef QW38_MIXER_H_
#define QW38_MIXER_H_

#include <cstddef>

#include "qw38/status.h"
#include "weights.h"

namespace qw38::internal {

constexpr std::size_t kResidualWidth = 5120;
constexpr std::size_t kGdnPackedQkvWidth = 10240;
constexpr std::size_t kGdnValueWidth = 6144;
constexpr std::size_t kGdnGateCount = 48;
constexpr std::size_t kGdnKeyWidth = 2048;
constexpr std::size_t kGdnConvolutionValues = 40960;
constexpr std::size_t kGdnRecurrentStateValues = 786432;
constexpr std::size_t kGdnHeadWidth = 128;
constexpr std::size_t kAttentionPackedQueryGateWidth = 12288;
constexpr std::size_t kAttentionQueryWidth = 6144;
constexpr std::size_t kAttentionKvWidth = 1024;
constexpr std::size_t kAttentionHeadWidth = 256;
constexpr std::size_t kFfnWidth = 17408;

struct GdnProjectionWorkspace final {
  float* packed_qkv;
  std::size_t packed_qkv_count;
  float* value_gate;
  std::size_t value_gate_count;
  float* alpha;
  std::size_t alpha_count;
  float* beta;
  std::size_t beta_count;
};

struct AttentionProjectionWorkspace final {
  float* packed_query_gate;
  std::size_t packed_query_gate_count;
  float* query;
  std::size_t query_count;
  float* gate;
  std::size_t gate_count;
  float* key;
  std::size_t key_count;
  float* value;
  std::size_t value_count;
};

struct GdnScalarParameters final {
  float* input_norm;
  std::size_t input_norm_count;
  float* convolution;
  std::size_t convolution_count;
  float* folded_a_tiled;
  std::size_t folded_a_count;
  float* dt_bias_tiled;
  std::size_t dt_bias_count;
  float* recurrent_norm;
  std::size_t recurrent_norm_count;
};

struct GdnLayerStateView final {
  float* convolution;
  std::size_t convolution_count;
  float* recurrent;
  std::size_t recurrent_count;
};

struct GdnStepWorkspace final {
  float* normalized;
  std::size_t normalized_count;
  GdnProjectionWorkspace projections;
  float* convolved_qkv;
  std::size_t convolved_qkv_count;
  float* query;
  std::size_t query_count;
  float* key;
  std::size_t key_count;
  float* value_tiled;
  std::size_t value_tiled_count;
  float* value_grouped;
  std::size_t value_grouped_count;
  float* gate_grouped;
  std::size_t gate_grouped_count;
  float* alpha_grouped;
  float* beta_grouped;
  float* folded_a_grouped;
  float* dt_bias_grouped;
  float* log_decay;
  float* update_gate;
  std::size_t gate_count;
  float* recurrent_output;
  std::size_t recurrent_output_count;
  float* gated_grouped;
  std::size_t gated_grouped_count;
  float* gated_tiled;
  std::size_t gated_tiled_count;
  float* mixer_output;
  std::size_t mixer_output_count;
};

struct FfnScalarParameters final {
  float* norm;
  std::size_t norm_count;
};

struct FfnStepWorkspace final {
  float* normalized;
  std::size_t normalized_count;
  float* gate;
  std::size_t gate_count;
  float* up;
  std::size_t up_count;
  float* activated;
  std::size_t activated_count;
  float* correction;
  std::size_t correction_count;
};

struct AttentionScalarParameters final {
  float* input_norm;
  std::size_t input_norm_count;
  float* query_norm;
  std::size_t query_norm_count;
  float* key_norm;
  std::size_t key_norm_count;
};

struct AttentionLayerStateView final {
  float* key_cache;
  std::size_t key_cache_count;
  float* value_cache;
  std::size_t value_cache_count;
  std::size_t capacity;
};

struct AttentionStepWorkspace final {
  float* normalized;
  std::size_t normalized_count;
  AttentionProjectionWorkspace projections;
  float* attention_output;
  std::size_t attention_output_count;
  float* scores;
  std::size_t score_count;
  float* mixer_output;
  std::size_t mixer_output_count;
};

Status project_gdn_mixer(const GdnLayerWeights& weights,
                         const float* activation,
                         std::size_t activation_count,
                         const GdnProjectionWorkspace& workspace) noexcept;

Status project_attention_mixer(
    const AttentionLayerWeights& weights, const float* activation,
    std::size_t activation_count,
    const AttentionProjectionWorkspace& workspace) noexcept;

Status prepare_gdn_scalar_parameters(
    const LayerWeights& weights,
    const GdnScalarParameters& parameters) noexcept;

Status execute_gdn_mixer_step(
    const LayerWeights& weights, const GdnScalarParameters& parameters,
    const float* residual, std::size_t residual_count,
    const GdnLayerStateView& state, const GdnStepWorkspace& workspace,
    float* output, std::size_t output_count) noexcept;

Status prepare_ffn_scalar_parameters(
    const CommonLayerWeights& weights,
    const FfnScalarParameters& parameters) noexcept;

Status validate_ffn_step(const FfnScalarParameters& parameters,
                         const float* residual,
                         std::size_t residual_count,
                         const FfnStepWorkspace& workspace, float* output,
                         std::size_t output_count) noexcept;

Status execute_ffn_step(
    const CommonLayerWeights& weights, const FfnScalarParameters& parameters,
    const float* residual, std::size_t residual_count,
    const FfnStepWorkspace& workspace, float* output,
    std::size_t output_count) noexcept;

Status prepare_attention_scalar_parameters(
    const LayerWeights& weights,
    const AttentionScalarParameters& parameters) noexcept;

Status execute_attention_mixer_step(
    const LayerWeights& weights, const AttentionScalarParameters& parameters,
    std::size_t position, const float* residual, std::size_t residual_count,
    const AttentionLayerStateView& state,
    const AttentionStepWorkspace& workspace, float* output,
    std::size_t output_count) noexcept;

}  // namespace qw38::internal

#endif  // QW38_MIXER_H_
