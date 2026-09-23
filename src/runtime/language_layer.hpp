#pragma once

#include "runtime/attention.hpp"
#include "runtime/gdn.hpp"
#include "runtime/mlp.hpp"

#include <cstdint>
#include <expected>

namespace qw38::runtime {

enum class LanguageMixerKind : std::uint8_t { Gdn, Attention };

// A fully bound, immutable-by-interface decode layer. Construction resolves
// artifact tensor names and session views once; execution performs no plan or
// storage allocation.
class LanguageLayerPlan {
 public:
  [[nodiscard]] static std::expected<LanguageLayerPlan, Error> bind(
      Model const& model, Session& session, std::uint32_t layer,
      qw38::cuda::Stream const& stream);

  [[nodiscard]] std::uint32_t layer() const noexcept { return layer_; }
  [[nodiscard]] LanguageMixerKind mixer_kind() const noexcept { return kind_; }

 private:
  friend std::expected<TensorView, Error> execute_decode_language_layer(
      LanguageLayerPlan const&, std::uint64_t);
  friend struct LanguageLayerPlanTestAccess;

  std::uint32_t layer_{};
  LanguageMixerKind kind_{LanguageMixerKind::Gdn};
  GdnPlan gdn_{};
  AttentionMixerPlan attention_{};
  MlpPlan mlp_{};
};

// Executes mixer residual transition followed by the shared post-mixer MLP
// transition, in deterministic order on the bound Session stream.
[[nodiscard]] std::expected<TensorView, Error> execute_decode_language_layer(
    LanguageLayerPlan const& plan, std::uint64_t position);

}  // namespace qw38::runtime
