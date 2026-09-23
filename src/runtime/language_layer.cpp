#include "runtime/language_layer.hpp"

namespace qw38::runtime {
namespace {

Error invalid_layer_error() {
  return make_error(ErrorCode::InvalidArgument, "layer",
                    "language layer must be < 64");
}

}  // namespace

std::expected<LanguageLayerPlan, Error> LanguageLayerPlan::bind(
    Model const& model, Session& session, std::uint32_t layer,
    qw38::cuda::Stream const& stream) {
  if (layer >= kLanguageLayers) {
    return std::unexpected(invalid_layer_error());
  }

  LanguageLayerPlan plan;
  plan.layer_ = layer;
  if (is_gdn_language_layer(layer)) {
    auto mixer = bind_gdn_plan(model, session, layer, stream);
    if (!mixer) return std::unexpected(mixer.error());
    plan.kind_ = LanguageMixerKind::Gdn;
    plan.gdn_ = std::move(*mixer);
  } else {
    auto mixer = bind_attention_mixer_plan(model, session, layer, stream);
    if (!mixer) return std::unexpected(mixer.error());
    plan.kind_ = LanguageMixerKind::Attention;
    plan.attention_ = std::move(*mixer);
  }
  auto mlp = bind_mlp_plan(model, session, layer, stream);
  if (!mlp) return std::unexpected(mlp.error());
  plan.mlp_ = std::move(*mlp);
  return plan;
}

std::expected<TensorView, Error> execute_decode_language_layer(
    LanguageLayerPlan const& plan, std::uint64_t position) {
  std::expected<TensorView, Error> mixed =
      plan.kind_ == LanguageMixerKind::Gdn
          ? execute_decode_gdn(plan.gdn_, position)
          : execute_decode_attention(plan.attention_, position);
  if (!mixed) return std::unexpected(mixed.error());
  auto mlp = execute_decode_mlp(plan.mlp_);
  if (!mlp) {
    // The mixer has already committed persistent state. If the following MLP
    // fails, only reset/restore can re-establish a known token boundary.
    auto* session_state = plan.kind_ == LanguageMixerKind::Gdn
                              ? plan.gdn_.session_state
                              : plan.attention_.prep.session_state;
    if (session_state != nullptr) session_state->poison();
    return std::unexpected(mlp.error());
  }
  return plan.mlp_.next_h;
}

}  // namespace qw38::runtime
