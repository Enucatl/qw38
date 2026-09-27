#pragma once

#include "runtime/language_layer.hpp"
#include "runtime/prefill.hpp"
#include "cuda/decode_mmv.hpp"
#include "cuda/decode_control.hpp"
#include "cuda/graph.hpp"

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

enum class DecodeSubmission { Eager, Graph };

// The model and session allocations must outlive this borrowed plan;
// reset/restore and Session movement may be used between calls.
class LanguageModelPlan {
 public:
  [[nodiscard]] static std::expected<LanguageModelPlan, Error> bind(
      Model const& model, Session& session, qw38::cuda::Stream const& stream);

  [[nodiscard]] std::expected<DecodeResult, Error> decode_token(
      std::uint32_t token_id, std::uint64_t position);
  // Explicit diagnostic selection; changing mode does not execute or commit.
  void set_decode_submission(DecodeSubmission mode) noexcept { submission_ = mode; }
  [[nodiscard]] std::optional<Error> const& graph_fallback() const noexcept {
    return graph_failure_;
  }
  [[nodiscard]] std::expected<DecodeResult, Error> setup_prompt_slow(
      std::span<std::uint32_t const> token_ids);
  // Logits passed to the sink are valid only during the callback. A returned
  // error or thrown exception poisons the session until reset/restore.
  // For partitioned replay, logical_prompt_tokens is the same incoming total
  // on every partition; otherwise the complete input span defines the regime.
  // It excludes any preexisting prefix and does not latch session state.
  using LogitRowSink = std::function<std::expected<void, Error>(
      std::uint64_t, std::span<float const>)>;
  [[nodiscard]] std::expected<DecodeResult, Error> prefill_tokens(
      std::span<std::uint32_t const> token_ids,
      std::span<std::uint64_t const> requested_rows = {},
      LogitRowSink const& sink = {},
      std::optional<std::uint64_t> logical_prompt_tokens = std::nullopt);

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
  [[nodiscard]] std::expected<qw38::cuda::DecodeControl, Error> pending(
      std::uint32_t token, std::uint64_t position, std::uint32_t count) const;
  [[nodiscard]] std::expected<void, Error> enqueue_decode(
      qw38::cuda::DecodeControl const&, qw38::cuda::DecodeControl const*, std::uint64_t);
  [[nodiscard]] std::expected<void, Error> prepare_graph(qw38::cuda::DecodeControl const&);
  friend struct LanguageModelPlanTestAccess;
  std::vector<LanguageLayerPlan> layers_;
  std::vector<std::variant<PrefillGdnLayerPlan, PrefillAttentionLayerPlan>> prefill_layers_;
  qw38::cuda::HostBuffer readback_;
  std::span<float> logits_;
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
  qw38::cuda::DeviceBuffer unit_failure_;
  [[nodiscard]] std::expected<void, Error> complete_unit();
  DecodeSubmission submission_{DecodeSubmission::Graph};
  qw38::cuda::DeviceBuffer controls_;
  qw38::cuda::Graph graph_;
  std::uint64_t graph_bucket_{};
  std::uint64_t graph_builds_{};
  std::uint64_t graph_replays_{};
  std::optional<Error> graph_failure_;
};

}  // namespace qw38::runtime
