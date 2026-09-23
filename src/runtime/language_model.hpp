#pragma once

#include "runtime/language_layer.hpp"
#include "cuda/decode_mmv.hpp"

#include <cstdint>
#include <expected>
#include <span>
#include <vector>

namespace qw38::runtime {

[[nodiscard]] std::expected<void, Error> validate_primary_language_graph(
    qw38::format::ArtifactSchema const& schema);

struct DecodeResult {
  // Valid until the next decode on this plan or its destruction.
  std::span<float const> logits;
  std::uint32_t argmax{};
};

// One primary-language token at a time. The model and session must outlive
// this borrowed plan; reset/restore may be used between calls.
class LanguageModelPlan {
 public:
  [[nodiscard]] static std::expected<LanguageModelPlan, Error> bind(
      Model const& model, Session& session, qw38::cuda::Stream const& stream);

  [[nodiscard]] std::expected<DecodeResult, Error> decode_token(
      std::uint32_t token_id, std::uint64_t position);
  [[nodiscard]] std::expected<DecodeResult, Error> setup_prompt_slow(
      std::span<std::uint32_t const> token_ids);

 private:
  friend struct LanguageModelPlanTestAccess;
  std::vector<LanguageLayerPlan> layers_;
  std::vector<float> logits_;
  Session* session_{};
  qw38::cuda::Stream const* stream_{};
  SessionExecutionState* state_{};
  std::uint16_t const* embedding_{};
  std::uint16_t const* final_gamma_{};
  std::uint16_t* normalized_{};
  float* residual_{};
  float* device_logits_{};
  qw38::cuda::DecodeMmvDesc head_{};
};

}  // namespace qw38::runtime
