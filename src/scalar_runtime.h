#ifndef QW38_SCALAR_RUNTIME_H_
#define QW38_SCALAR_RUNTIME_H_

#include <array>
#include <cstddef>
#include <vector>

#include "qw38/status.h"
#include "scheduler.h"
#include "weights.h"

#ifdef QW38_DIAGNOSTIC_TRACE
#include "diagnostic_trace.h"
#endif

namespace qw38::internal {

constexpr std::size_t kGdnLayerCount = 48;
constexpr std::size_t kAttentionLayerCount = 16;

struct ScalarModelParameters final {
  ScalarModelParameters() = default;
  ScalarModelParameters(const ScalarModelParameters&) = delete;
  ScalarModelParameters& operator=(const ScalarModelParameters&) = delete;
  ScalarModelParameters(ScalarModelParameters&&) noexcept = default;
  ScalarModelParameters& operator=(ScalarModelParameters&&) noexcept = default;

  std::vector<float> input_norms;
  std::vector<float> ffn_norms;
  std::vector<float> gdn_convolution;
  std::vector<float> gdn_folded_a;
  std::vector<float> gdn_dt_bias;
  std::vector<float> gdn_recurrent_norm;
  std::vector<float> attention_query_norm;
  std::vector<float> attention_key_norm;
  std::vector<float> output_norm;
  std::array<GdnLayerScalarParameters, kModelLayerCount> gdn{};
  std::array<AttentionLayerScalarParameters, kModelLayerCount> attention{};
  OutputScalarParameters output{};
  std::size_t prepared_layers = 0;
};

struct ScalarSessionState final {
  ScalarSessionState() = default;
  ScalarSessionState(const ScalarSessionState&) = delete;
  ScalarSessionState& operator=(const ScalarSessionState&) = delete;
  ScalarSessionState(ScalarSessionState&&) noexcept = default;
  ScalarSessionState& operator=(ScalarSessionState&&) noexcept = default;

  std::vector<float> gdn_convolution;
  std::vector<float> gdn_recurrent;
  std::vector<float> attention_key;
  std::vector<float> attention_value;
  std::array<GdnLayerStateView, kModelLayerCount> gdn{};
  std::array<AttentionLayerStateView, kModelLayerCount> attention{};
  std::size_t capacity = 0;
  std::size_t frontier = 0;
};

struct ScalarWorkspace final {
  ScalarWorkspace() = default;
  ScalarWorkspace(const ScalarWorkspace&) = delete;
  ScalarWorkspace& operator=(const ScalarWorkspace&) = delete;
  ScalarWorkspace(ScalarWorkspace&&) noexcept = default;
  ScalarWorkspace& operator=(ScalarWorkspace&&) noexcept = default;

  std::vector<float> activation_a;
  std::vector<float> activation_b;
  std::vector<float> post_mixer;

  std::vector<float> gdn_normalized;
  std::vector<float> gdn_packed;
  std::vector<float> gdn_projected_gate;
  std::vector<float> gdn_projected_alpha;
  std::vector<float> gdn_projected_beta;
  std::vector<float> gdn_convolved;
  std::vector<float> gdn_query;
  std::vector<float> gdn_key;
  std::vector<float> gdn_value_tiled;
  std::vector<float> gdn_value_grouped;
  std::vector<float> gdn_gate_grouped;
  std::vector<float> gdn_alpha_grouped;
  std::vector<float> gdn_beta_grouped;
  std::vector<float> gdn_folded_grouped;
  std::vector<float> gdn_dt_grouped;
  std::vector<float> gdn_log_decay;
  std::vector<float> gdn_update_gate;
  std::vector<float> gdn_recurrent_output;
  std::vector<float> gdn_gated_grouped;
  std::vector<float> gdn_gated_tiled;
  std::vector<float> gdn_mixer_output;

  std::vector<float> attention_normalized;
  std::vector<float> attention_packed;
  std::vector<float> attention_query;
  std::vector<float> attention_gate;
  std::vector<float> attention_key;
  std::vector<float> attention_value;
  std::vector<float> attention_output;
  std::vector<float> attention_scores;
  std::vector<float> attention_mixer_output;
#ifdef QW38_DIAGNOSTIC_TRACE
  std::vector<float> attention_rope_query;
  std::vector<float> attention_rope_key;
#endif

  std::vector<float> ffn_normalized;
  std::vector<float> ffn_gate;
  std::vector<float> ffn_up;
  std::vector<float> ffn_activated;
  std::vector<float> ffn_correction;
  std::vector<float> final_normalized;

  GdnLayerWorkspace gdn{};
  AttentionLayerWorkspace attention{};
  OutputWorkspace output{};
  std::size_t capacity = 0;
  std::size_t layers_completed = 0;
};

Status prepare_scalar_model_parameters(
    const ModelWeights& weights, ScalarModelParameters* parameters) noexcept;

Status create_scalar_session_state(std::size_t capacity,
                                   ScalarSessionState* state) noexcept;

Status create_scalar_workspace(std::size_t capacity,
                               ScalarWorkspace* workspace) noexcept;

Status execute_scalar_token(const ModelWeights& weights,
                            const ScalarModelParameters& parameters,
                            std::size_t token, ScalarSessionState* state,
                            ScalarWorkspace* workspace, float* logits,
                            std::size_t logits_count) noexcept;

#ifdef QW38_DIAGNOSTIC_TRACE
Status execute_scalar_token_traced(
    const ModelWeights& weights, const ScalarModelParameters& parameters,
    std::size_t token, ScalarSessionState* state, ScalarWorkspace* workspace,
    float* logits, std::size_t logits_count, const TraceFilter& filter,
    TraceSink sink, void* sink_context) noexcept;
#endif

}  // namespace qw38::internal

#endif  // QW38_SCALAR_RUNTIME_H_
