#pragma once

#include "runtime/language_layer.hpp"
#include "runtime/prefill.hpp"
#include "cuda/decode_control.hpp"
#include "cuda/attention.hpp"

namespace qw38::runtime::detail {
// Private full-model route. The caller validates the complete boundary, owns
// completion/error draining, and commits all host metadata only after success.
struct Submission {
  static std::expected<void, Error> mlp(MlpPlan const&, int*);
  static std::expected<void, Error> prefill_mlp(PrefillLayerProjectionPlan const&,
      qw38::cuda::PrefillEngine&, float const*, float*, std::uint32_t, std::uint64_t, float, int*);
  static std::expected<TensorView, Error> layer(LanguageLayerPlan const&,
      qw38::cuda::DecodeControl const&, qw38::cuda::DecodeControl const*, std::uint64_t, int*);
  static std::expected<TensorView, Error> gdn(GdnPlan const&,
      qw38::cuda::DecodeControl const&, qw38::cuda::DecodeControl const*);
  static std::expected<void, Error> attention_prep(AttentionPrepPlan const&,
      std::uint64_t, bool, qw38::cuda::DecodeControl const* = nullptr);
  static std::expected<TensorView, Error> attention_core(AttentionCorePlan const&,
      std::uint64_t, qw38::cuda::DecodeControl const* = nullptr, std::uint64_t = 0);
  static std::expected<void, Error> prefill_gdn(PrefillGdnLayerPlan const&,
      PrefillGdnWorkspace&, qw38::cuda::PrefillEngine&, float const*, float*, float*,
      std::uint32_t, std::uint64_t, std::uint32_t, bool, int* = nullptr,
      qw38::cuda::DecodeControl const* = nullptr);
  static std::expected<void, Error> prefill_attention(PrefillAttentionLayerPlan const&,
      PrefillAttentionWorkspace&, qw38::cuda::PrefillEngine&, float const*, float*, float*,
      std::uint32_t, std::uint64_t, bool, int* = nullptr,
      qw38::cuda::ContextRegime = qw38::cuda::ContextRegime::Large);
};
}  // namespace qw38::runtime::detail
