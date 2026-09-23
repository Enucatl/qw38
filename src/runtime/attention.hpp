#pragma once

#include "runtime/error.hpp"
#include "runtime/model.hpp"
#include "runtime/session.hpp"
#include "runtime/sizes.hpp"
#include "runtime/view.hpp"

#include "cuda/stream.hpp"

#include <cstdint>
#include <expected>
#include <string>

namespace qw38::runtime {

inline constexpr float kAttnRmsEps = 1.0e-6f;

[[nodiscard]] constexpr bool is_attention_language_layer(
    std::uint32_t layer) noexcept {
  return layer < kLanguageLayers && (layer % 4u) == 3u;
}

[[nodiscard]] constexpr std::uint32_t kv_head_for_query(
    std::uint32_t query_head) noexcept {
  return query_head / kGqaGroup;
}

[[nodiscard]] std::expected<std::uint32_t, Error> attn_state_index(
    std::uint32_t language_layer);

struct AttnWeightBinding {
  ConstTensorView codes{};
  ConstTensorView scales{};
  std::uint16_t layout{};
  std::uint16_t quantizer{};
  std::uint32_t n{};
  std::uint32_t k{};
  std::uint32_t padded_n{};
  std::uint32_t padded_k{};
  std::uint64_t codes_bytes{};
  std::uint64_t scales_bytes{};
};

// Typed aliases of the single AttentionWorkspace allocation (M-01).
struct AttnWorkspaceViews {
  TensorView qg{};  // BF16 [24, 512] per-head q then g
  TensorView k{};   // BF16 [4, 256] projected
  TensorView v{};   // BF16 [4, 256] projected
  TensorView q{};   // BF16 [24, 256] prepared
  TensorView g{};   // BF16 [24, 256] per-query-head scratch
};

struct AttentionPrepPlan {
  AttnWeightBinding qg{};
  AttnWeightBinding k{};
  AttnWeightBinding v{};
  ConstTensorView gamma{};      // input RMS, BF16 [5120]
  ConstTensorView gamma_q{};    // QK RMS, BF16 [256]
  ConstTensorView gamma_k{};    // QK RMS, BF16 [256]
  ConstTensorView inv_freq{};   // FP32 [32]
  TensorView residual{};   // FP32 h, decode token 0
  TensorView normalized{}; // BF16 [5120]
  AttnWorkspaceViews scratch{};
  TensorView kv{};         // BF16 [layer, 2, 4, capacity, 256]
  KvPopulatedSlot populated{};
  std::uint64_t kv_capacity{};
  qw38::cuda::Stream const* stream{nullptr};
  float eps{kAttnRmsEps};
  std::uint32_t language_layer{};
  std::uint32_t attn_layer{};
  SessionExecutionState* session_state{};
};

struct AttentionPrepBindViews {
  ConstTensorView qg{};
  ConstTensorView qg_scales{};
  ConstTensorView k{};
  ConstTensorView k_scales{};
  ConstTensorView v{};
  ConstTensorView v_scales{};
  ConstTensorView gamma{};
  ConstTensorView gamma_q{};
  ConstTensorView gamma_k{};
  ConstTensorView inv_freq{};
  TensorView residual{};
  TensorView normalized{};
  WorkspaceView workspace{};
  TensorView kv{};
  KvPopulatedSlot populated{};
  std::uint64_t kv_capacity{};
  std::uint32_t language_layer{};
};

// Attention core binds the prepared Q/g vectors and persistent cache to the
// sequence-length partial workspace and the output projection.  Partials are
// laid out [query_head, segment, (max,sum,numerator[256])].
struct AttentionCoreBindViews {
  TensorView q{};
  TensorView g{};
  TensorView kv{};
  TensorView partials{};      // FP32, enough for all segments used by capacity
  TensorView y{};             // BF16 [24,256], gated attention output
  TensorView residual{};      // FP32 input residual
  TensorView residual_out{};  // FP32 output residual
  ConstTensorView out{};           // [5120,6144] Q4/BF16 projection
  ConstTensorView out_scales{};
  KvPopulatedSlot populated{};
  std::uint64_t kv_capacity{};
  std::uint32_t language_layer{};
};

struct AttentionCorePlan {
  AttnWeightBinding out{};
  TensorView q{};
  TensorView g{};
  TensorView kv{};
  TensorView partials{};
  TensorView y{};
  TensorView residual{};
  TensorView residual_out{};
  KvPopulatedSlot populated{};
  std::uint64_t kv_capacity{};
  std::uint32_t attn_layer{};
  qw38::cuda::Stream const* stream{nullptr};
  SessionExecutionState* session_state{};
};

struct AttentionMixerBindViews {
  AttentionPrepBindViews prep{};
  ConstTensorView out{};
  ConstTensorView out_scales{};
  TensorView residual_out{};
};

struct AttentionMixerPlan {
  AttentionPrepPlan prep{};
  AttentionCorePlan core{};
};

[[nodiscard]] std::string attn_norm_name(std::uint32_t layer);
[[nodiscard]] std::string attn_q_name(std::uint32_t layer);
[[nodiscard]] std::string attn_k_name(std::uint32_t layer);
[[nodiscard]] std::string attn_v_name(std::uint32_t layer);
[[nodiscard]] std::string attn_o_name(std::uint32_t layer);
[[nodiscard]] std::string attn_q_norm_name(std::uint32_t layer);
[[nodiscard]] std::string attn_k_norm_name(std::uint32_t layer);

[[nodiscard]] std::expected<AttnWorkspaceViews, Error> bind_attention_workspace(
    WorkspaceView workspace);

[[nodiscard]] std::expected<AttentionPrepPlan, Error> bind_attention_prep_plan(
    AttentionPrepBindViews const& views, qw38::cuda::Stream const& stream,
    float eps = kAttnRmsEps);

[[nodiscard]] std::expected<AttentionPrepPlan, Error> bind_attention_prep_plan(
    Model const& model, Session& session, std::uint32_t layer,
    qw38::cuda::Stream const& stream, float eps = kAttnRmsEps);

[[nodiscard]] std::expected<AttentionCorePlan, Error> bind_attention_core_plan(
    AttentionCoreBindViews const& views, qw38::cuda::Stream const& stream);

[[nodiscard]] std::expected<AttentionMixerPlan, Error> bind_attention_mixer_plan(
    AttentionMixerBindViews const& views, qw38::cuda::Stream const& stream,
    float eps = kAttnRmsEps);

[[nodiscard]] std::expected<AttentionMixerPlan, Error> bind_attention_mixer_plan(
    Model const& model, Session& session, std::uint32_t layer,
    qw38::cuda::Stream const& stream, float eps = kAttnRmsEps);

// Steps 1–3: RMS, one ranged Q4 q/g+k+v launch, fused QK-norm/RoPE + cache
// append. Advances populated length only after successful work. No allocation.
[[nodiscard]] std::expected<void, Error> execute_decode_attention_prep(
    AttentionPrepPlan const& plan, std::uint64_t position);

// Preparation followed by ordered segmented scan, fixed-order merge, and
// Q4/BF16 output projection plus direct FP32 residual add.
[[nodiscard]] std::expected<TensorView, Error> execute_decode_attention(
    AttentionMixerPlan const& plan, std::uint64_t position);

[[nodiscard]] std::expected<TensorView, Error> execute_attention_core(
    AttentionCorePlan const& plan);

}  // namespace qw38::runtime
