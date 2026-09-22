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

inline constexpr float kGdnRmsEps = 1.0e-6f;

[[nodiscard]] constexpr bool is_gdn_language_layer(std::uint32_t layer) noexcept {
  return layer < kLanguageLayers && (layer % 4u) != 3u;
}

[[nodiscard]] constexpr std::uint32_t gdn_key_head(
    std::uint32_t value_head) noexcept {
  return value_head / kGdnRepeat;
}

[[nodiscard]] std::expected<std::uint32_t, Error> gdn_state_index(
    std::uint32_t language_layer);

struct GdnWeightBinding {
  TensorView codes{};
  TensorView scales{};
  std::uint16_t layout{};
  std::uint16_t quantizer{};
  std::uint32_t n{};
  std::uint32_t k{};
  std::uint32_t padded_n{};
  std::uint32_t padded_k{};
  std::uint64_t codes_bytes{};
  std::uint64_t scales_bytes{};
};

// Typed aliases of the single GdnWorkspace allocation (M-01). q/k are 16 heads.
struct GdnWorkspaceViews {
  TensorView qkv{};        // BF16 [10240]
  TensorView z{};          // BF16 [48,128]
  TensorView convolved{};  // BF16 [10240]
  TensorView q_hat{};      // FP32 [16,128]
  TensorView k_hat{};      // FP32 [16,128]
  TensorView a{};          // FP32 [48]
  TensorView b{};          // FP32 [48]
  TensorView alpha{};      // FP32 [48]
  TensorView beta{};       // FP32 [48]
  TensorView v{};          // BF16 [48,128] alias of convolved[4096:]
  TensorView o{};          // FP32 [48,128]
  TensorView u{};          // BF16 [48,128]
};

struct GdnFrontPlan {
  GdnWeightBinding qkv{};
  GdnWeightBinding z{};
  GdnWeightBinding a_proj{};
  GdnWeightBinding b_proj{};
  TensorView gamma{};
  TensorView taps{};       // BF16 tap-major [4,10240]
  TensorView a_log{};      // BF16 [48]
  TensorView dt_bias{};    // BF16 [48]
  TensorView residual{};   // FP32 h, decode token 0
  TensorView normalized{}; // BF16 [5120]
  GdnWorkspaceViews scratch{};
  TensorView history{};    // BF16 [3,10240] for this GDN layer
  std::uint32_t* host_cursor{nullptr};
  qw38::cuda::Stream const* stream{nullptr};
  float eps{kGdnRmsEps};
  std::uint32_t language_layer{};
  std::uint32_t gdn_layer{};
};

struct GdnFrontBindViews {
  TensorView qkv{};
  TensorView qkv_scales{};
  TensorView z{};
  TensorView z_scales{};
  TensorView a{};
  TensorView b{};
  TensorView gamma{};
  TensorView taps{};
  TensorView a_log{};
  TensorView dt_bias{};
  TensorView residual{};
  TensorView normalized{};
  TensorView workspace{};
  TensorView history{};
  std::uint32_t* host_cursor{nullptr};
  std::uint32_t language_layer{};
};

[[nodiscard]] std::string gdn_norm_name(std::uint32_t layer);
[[nodiscard]] std::string gdn_gated_norm_name(std::uint32_t layer);
[[nodiscard]] std::string gdn_qkv_name(std::uint32_t layer);
[[nodiscard]] std::string gdn_z_name(std::uint32_t layer);
[[nodiscard]] std::string gdn_a_name(std::uint32_t layer);
[[nodiscard]] std::string gdn_b_name(std::uint32_t layer);
[[nodiscard]] std::string gdn_out_name(std::uint32_t layer);
[[nodiscard]] std::string gdn_conv_name(std::uint32_t layer);
[[nodiscard]] std::string gdn_alog_name(std::uint32_t layer);
[[nodiscard]] std::string gdn_dt_name(std::uint32_t layer);

[[nodiscard]] std::expected<GdnWorkspaceViews, Error> bind_gdn_workspace(
    TensorView workspace);

[[nodiscard]] std::expected<GdnFrontPlan, Error> bind_gdn_front_plan(
    GdnFrontBindViews const& views, qw38::cuda::Stream const& stream,
    float eps = kGdnRmsEps);

[[nodiscard]] std::expected<GdnFrontPlan, Error> bind_gdn_front_plan(
    Model const& model, Session& session, std::uint32_t layer,
    qw38::cuda::Stream const& stream, float eps = kGdnRmsEps);

// GDN steps 1–5: RMS, qkv, z, a/b, conv+history, prepare. No allocation.
[[nodiscard]] std::expected<void, Error> execute_gdn_front(GdnFrontPlan const& plan);

struct GdnRecurrenceBindViews {
  TensorView q_hat{};
  TensorView k_hat{};
  TensorView alpha{};
  TensorView beta{};
  TensorView v{};
  TensorView s{};  // FP32 session S [layer,value_head,value,key]
  TensorView o{};
  std::uint32_t s_layer{};  // Must equal the layer derived from language_layer.
  std::uint32_t language_layer{};
};

// Decode recurrence over prepared q/k/α/β/v and persistent FP32 S. No allocation.
struct GdnRecurrencePlan {
  TensorView q_hat{};   // FP32 [16,128]
  TensorView k_hat{};   // FP32 [16,128]
  TensorView alpha{};   // FP32 [48]
  TensorView beta{};    // FP32 [48]
  TensorView v{};       // BF16 [48,128]
  TensorView s{};       // FP32 [layer,value_head,value,key]
  TensorView o{};       // FP32 [48,128]
  std::uint32_t s_layer{};
  std::uint32_t language_layer{};
  std::uint32_t gdn_layer{};
  qw38::cuda::Stream const* stream{nullptr};
};

[[nodiscard]] std::expected<GdnRecurrencePlan, Error> bind_gdn_recurrence_plan(
    GdnRecurrenceBindViews const& views, qw38::cuda::Stream const& stream);

[[nodiscard]] std::expected<GdnRecurrencePlan, Error> bind_gdn_recurrence_plan(
    Session& session, std::uint32_t layer, qw38::cuda::Stream const& stream);

[[nodiscard]] std::expected<void, Error> execute_gdn_recurrence(
    GdnRecurrencePlan const& plan);

// Complete decode GDN mixer: eight V0 regions, MLP-compatible ping-pong residual.
// Input residual stays live until region 8; output is the other FP32 buffer.
inline constexpr int kGdnMixerRegions = 8;
inline constexpr char const* kGdnMixerRegionNames[kGdnMixerRegions] = {
    "rms", "qkvz", "ab", "conv", "prep", "recur", "gated", "out-residual"};

struct GdnRegionTimings {
  float ms[kGdnMixerRegions]{};
};

struct GdnBindViews {
  TensorView qkv{};
  TensorView qkv_scales{};
  TensorView z{};
  TensorView z_scales{};
  TensorView a{};
  TensorView b{};
  TensorView out{};
  TensorView out_scales{};
  TensorView gamma{};
  TensorView gated_gamma{};
  TensorView taps{};
  TensorView a_log{};
  TensorView dt_bias{};
  TensorView residual{};
  TensorView residual_out{};
  TensorView normalized{};
  TensorView workspace{};
  TensorView history{};
  TensorView s{};
  std::uint32_t* host_cursor{nullptr};
  std::uint32_t language_layer{};
};

struct GdnPlan {
  GdnFrontPlan front{};
  GdnWeightBinding out{};
  TensorView gated_gamma{};   // BF16 [128], multiplicative
  TensorView residual_out{};  // FP32 h_mid; original residual stays live
  TensorView s{};             // FP32 session S
  std::uint32_t s_layer{};
};

[[nodiscard]] std::expected<GdnPlan, Error> bind_gdn_plan(
    GdnBindViews const& views, qw38::cuda::Stream const& stream,
    float eps = kGdnRmsEps);

[[nodiscard]] std::expected<GdnPlan, Error> bind_gdn_plan(
    Model const& model, Session& session, std::uint32_t layer,
    qw38::cuda::Stream const& stream, float eps = kGdnRmsEps);

// Eight launches, no allocation. Returns residual_out (h_mid).
[[nodiscard]] std::expected<TensorView, Error> execute_decode_gdn(
    GdnPlan const& plan);

// Same map; fills diagnostic per-region milliseconds. Not a fusion license.
[[nodiscard]] std::expected<TensorView, Error> execute_decode_gdn_timed(
    GdnPlan const& plan, GdnRegionTimings& timings);

}  // namespace qw38::runtime
