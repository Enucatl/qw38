#pragma once

#include "runtime/language_layer.hpp"
#include "runtime/prefill.hpp"
#include "cuda/decode_mmv.hpp"

#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace qw38::runtime {

[[nodiscard]] std::expected<void, Error> validate_primary_language_graph(
    qw38::format::ArtifactSchema const& schema);

struct DecodeResult {
  // Valid until the next decode or prefill on this plan, or its destruction.
  std::span<float const> logits;
  std::uint32_t argmax{};
};

// The model and session allocations must outlive this borrowed plan;
// reset/restore and Session movement may be used between calls.
class LanguageModelPlan {
 public:
  [[nodiscard]] static std::expected<LanguageModelPlan, Error> bind(
      Model const& model, Session& session, qw38::cuda::Stream const& stream);

  [[nodiscard]] std::expected<DecodeResult, Error> decode_token(
      std::uint32_t token_id, std::uint64_t position);
  [[nodiscard]] std::expected<DecodeResult, Error> setup_prompt_slow(
      std::span<std::uint32_t const> token_ids);
  // Logits passed to the sink are valid only during the callback. A returned
  // error or thrown exception poisons the session until reset/restore.
  using LogitRowSink = std::function<std::expected<void, Error>(
      std::uint64_t, std::span<float const>)>;
  [[nodiscard]] std::expected<DecodeResult, Error> prefill_tokens(
      std::span<std::uint32_t const> token_ids,
      std::span<std::uint64_t const> requested_rows = {},
      LogitRowSink const& sink = {});

 private:
  struct PrefillState {
    qw38::cuda::PrefillEngine engine;
    PrefillGdnWorkspace gdn_workspace;
    PrefillAttentionWorkspace attention_workspace;
    qw38::cuda::DeviceBuffer token_ids;
    qw38::cuda::DeviceBuffer next_h;
    qw38::cuda::PrefillWeight head;
  };
  [[nodiscard]] std::expected<void, Error> initialize_prefill();
  friend struct LanguageModelPlanTestAccess;
  std::vector<LanguageLayerPlan> layers_;
  std::vector<std::variant<PrefillGdnLayerPlan, PrefillAttentionLayerPlan>> prefill_layers_;
  std::vector<float> logits_;
  Model const* model_{};
  qw38::cuda::Stream const* stream_{};
  SessionExecutionState* state_{};
  std::uint16_t const* embedding_{};
  std::uint16_t const* final_gamma_{};
  std::uint16_t* normalized_{};
  float* residual_{};
  float* residual_mid_{};
  float* device_logits_{};
  std::uint64_t kv_capacity_{};
  qw38::cuda::DecodeMmvDesc head_{};
  std::optional<PrefillState> prefill_;
};

}  // namespace qw38::runtime
