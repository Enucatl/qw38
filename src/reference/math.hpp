#pragma once

#include "reference/error.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

namespace qw38::reference {

// Qwen3.8 language shapes (OBSERVED / DERIVED). Wrappers are shape-specific.
inline constexpr std::uint32_t kHidden = 5120;
inline constexpr std::uint32_t kFfn = 17408;
inline constexpr std::uint32_t kHeadDim = 256;
inline constexpr std::uint32_t kRotaryDim = 64;
inline constexpr std::uint32_t kRopeFreqs = 32;
inline constexpr std::uint32_t kGdnHeadDim = 128;
inline constexpr std::uint32_t kGdnKeyHeads = 16;
inline constexpr std::uint32_t kGdnValueHeads = 48;
inline constexpr std::uint32_t kGdnRepeat = 3;
inline constexpr std::uint32_t kQkvWidth = 10240;
inline constexpr std::uint32_t kGdnZWidth = 6144;
inline constexpr std::uint32_t kQueryHeads = 24;
inline constexpr std::uint32_t kKvHeads = 4;
inline constexpr std::uint32_t kGqaGroup = 6;
inline constexpr std::uint32_t kQgWidth = 12288;
inline constexpr std::uint32_t kAttnKvWidth = 1024;
inline constexpr std::uint32_t kAttnOutWidth = 6144;
inline constexpr std::uint32_t kAttnSegmentKeys = 256;
inline constexpr float kAttnScale = 1.0f / 16.0f;
inline constexpr std::uint32_t kKvComponentK = 0;
inline constexpr std::uint32_t kKvComponentV = 1;
inline constexpr std::uint32_t kConvKernel = 4;
inline constexpr std::uint32_t kConvHistoryTaps = 3;
inline constexpr float kDefaultRmsEps = 1.0e-6f;
inline constexpr double kRopeTheta = 10000000.0;

// Documented CUDA-vs-reference tolerances (declared BF16 store compared in
// FP32). Embed widening is bit-exact. RMS/RoPE allow one BF16 ULP at 1 plus
// FP32 reduction/trig noise; large-position RoPE uses a looser phase budget.
namespace tol {
inline constexpr float kEmbedAbs = 0.0f;
inline constexpr float kRmsBf16Abs = 8.0e-3f;
inline constexpr float kGatedRmsBf16Abs = 1.6e-2f;
inline constexpr float kSiluAbs = 2.0e-5f;
inline constexpr float kSigmoidAbs = 2.0e-5f;
inline constexpr float kRopeSmallAbs = 8.0e-3f;
inline constexpr float kRopeLargeAbs = 5.0e-2f;
inline constexpr float kGemvRel = 1.0e-5f;
inline constexpr float kGemvAbs = 1.0e-5f;
// Composed decode MLP: RMS store, SwiGLU BF16 product, down+residual.
// GEMV reduction-order noise (decode_mmv_tol) plus one BF16 store per stage.
inline constexpr float kMlpNormAbs = kRmsBf16Abs;
inline constexpr float kMlpSwigluAbs = 2.0e-2f;
inline constexpr float kMlpSwigluRel = 1.0e-3f;
inline constexpr float kMlpResidualAbs = 5.0e-2f;
inline constexpr float kMlpResidualRel = 1.0e-3f;
// GDN front: FP32 FIR+SiLU then one BF16 store; FP32 q/k L2 and gates.
inline constexpr float kGdnConvBf16Abs = 8.0e-3f;
inline constexpr float kGdnQkFp32Abs = 2.0e-5f;
inline constexpr float kGdnQkFp32Rel = 1.0e-5f;
inline constexpr float kGdnGateFp32Abs = 2.0e-5f;
inline constexpr float kGdnGateFp32Rel = 1.0e-5f;
// Recurrence: FP32 prediction/update/readout (P-01). Warp tree vs sequential
// 128-key reduction is the one-step budget; multi-step lets that feed S.
inline constexpr float kGdnRecurSAbs = 2.0e-5f;
inline constexpr float kGdnRecurSRel = 1.0e-5f;
inline constexpr float kGdnRecurOAbs = 2.0e-5f;
inline constexpr float kGdnRecurORel = 1.0e-5f;
inline constexpr float kGdnRecurMultiSAbs = 5.0e-4f;
inline constexpr float kGdnRecurMultiSRel = 1.0e-4f;
inline constexpr float kGdnRecurMultiOAbs = 5.0e-4f;
inline constexpr float kGdnRecurMultiORel = 1.0e-4f;
// Complete mixer: gated-RMS BF16 u plus Q4/BF16 out GEMV residual-add.
inline constexpr float kGdnUAbs = 2.0e-2f;
inline constexpr float kGdnURel = 1.0e-3f;
inline constexpr float kGdnMixerResidualAbs = 5.0e-2f;
inline constexpr float kGdnMixerResidualRel = 1.0e-3f;
// Attention prep: GEMV BF16 store plus fused QK-RMS/RoPE store. Large-position
// RoPE uses the existing phase budget.
inline constexpr float kAttnProjAbs = 8.0e-3f;
inline constexpr float kAttnProjRel = 1.0e-4f;
inline constexpr float kAttnPrepSmallAbs = 2.0e-2f;
inline constexpr float kAttnPrepSmallRel = 1.0e-3f;
inline constexpr float kAttnPrepLargeAbs = 5.0e-2f;
inline constexpr float kAttnPrepLargeRel = 1.0e-3f;
// Segmented online softmax: sequential vs CUDA subtile/warp reduction, then
// one BF16 gated store and Q4/BF16 o_proj residual-add.
inline constexpr float kAttnOnlineFp32Abs = 2.0e-3f;
inline constexpr float kAttnOnlineFp32Rel = 1.0e-4f;
inline constexpr float kAttnGateBf16Abs = 2.0e-2f;
inline constexpr float kAttnGateBf16Rel = 1.0e-3f;
inline constexpr float kAttnMixerResidualAbs = 5.0e-2f;
inline constexpr float kAttnMixerResidualRel = 1.0e-3f;
}  // namespace tol

[[nodiscard]] bool eps_ok(float eps) noexcept;

// ω_j = θ^{-2j/64}, j=0..31, matching compiler generate_rope_inv_freq.
[[nodiscard]] std::array<float, kRopeFreqs> rope_inv_freq();

[[nodiscard]] float sigmoid_fp32(float u) noexcept;
[[nodiscard]] float silu_fp32(float z) noexcept;
[[nodiscard]] double sigmoid_f64(double u) noexcept;
[[nodiscard]] double silu_f64(double z) noexcept;

// BF16 row gather → FP32 residual. table is row-major [vocab, 5120].
[[nodiscard]] std::expected<void, Error> embed_gather_bf16_to_fp32(
    std::span<std::uint16_t const> table, std::uint32_t vocab,
    std::uint32_t token_id, std::span<float> residual);

// Zero-centered RMS: y = (1+γ) ⊙ x / RMS(x), FP32 math, BF16 store.
[[nodiscard]] std::expected<void, Error> hidden_rms_norm_1p_gamma(
    std::span<float const> residual, std::span<std::uint16_t const> gamma,
    float eps, std::span<std::uint16_t> out_bf16);
[[nodiscard]] std::expected<void, Error> hidden_rms_norm_1p_gamma_f64(
    std::span<float const> residual, std::span<std::uint16_t const> gamma,
    float eps, std::span<std::uint16_t> out_bf16);

// Authoritative attention preparation boundary: BF16 projection staging,
// FP32 QK RMS and partial RoPE, then one BF16 store.
[[nodiscard]] std::expected<void, Error> qk_rms_rope_1p_gamma(
    std::span<std::uint16_t const> projected_head_bf16,
    std::span<std::uint16_t const> gamma, float eps,
    std::span<float const> inv_freq, std::int32_t position,
    std::span<std::uint16_t> out_bf16);
[[nodiscard]] std::expected<void, Error> qk_rms_rope_1p_gamma_f64(
    std::span<std::uint16_t const> projected_head_bf16,
    std::span<std::uint16_t const> gamma, float eps,
    std::span<float const> inv_freq, std::int32_t position,
    std::span<std::uint16_t> out_bf16);

// Diagnostic standalone QK RMS. It rounds its FP32 input to BF16 output and is
// non-authoritative when followed by RoPE.
[[nodiscard]] std::expected<void, Error> qk_rms_norm_1p_gamma(
    std::span<float const> head, std::span<std::uint16_t const> gamma,
    float eps, std::span<std::uint16_t> out_bf16);
[[nodiscard]] std::expected<void, Error> qk_rms_norm_1p_gamma_f64(
    std::span<float const> head, std::span<std::uint16_t const> gamma,
    float eps, std::span<std::uint16_t> out_bf16);

// GDN gated RMS: y = (γ ⊙ o/RMS(o)) ⊙ SiLU(z). Multiplicative γ, not 1+γ.
[[nodiscard]] std::expected<void, Error> gdn_gated_rms_norm(
    std::span<float const> o, std::span<std::uint16_t const> z_bf16,
    std::span<std::uint16_t const> gamma, float eps,
    std::span<std::uint16_t> out_bf16);
[[nodiscard]] std::expected<void, Error> gdn_gated_rms_norm_f64(
    std::span<float const> o, std::span<std::uint16_t const> z_bf16,
    std::span<std::uint16_t const> gamma, float eps,
    std::span<std::uint16_t> out_bf16);

[[nodiscard]] std::expected<void, Error> sigmoid_fp32(
    std::span<float const> in, std::span<float> out);
[[nodiscard]] std::expected<void, Error> silu_fp32(std::span<float const> in,
                                                   std::span<float> out);
[[nodiscard]] std::expected<void, Error> sigmoid_f64(std::span<float const> in,
                                                     std::span<double> out);
[[nodiscard]] std::expected<void, Error> silu_f64(std::span<float const> in,
                                                  std::span<double> out);

// Diagnostic standalone partial RoPE over BF16 input. This rounded primitive
// is non-authoritative after QK RMS; use qk_rms_rope_1p_gamma for that path.
// Integer position is converted only for FP32 (or f64) phase evaluation.
[[nodiscard]] std::expected<void, Error> partial_rope(
    std::span<std::uint16_t const> head_bf16,
    std::span<float const> inv_freq, std::int32_t position,
    std::span<std::uint16_t> out_bf16);
[[nodiscard]] std::expected<void, Error> partial_rope_f64(
    std::span<std::uint16_t const> head_bf16,
    std::span<float const> inv_freq, std::int32_t position,
    std::span<std::uint16_t> out_bf16);

// Decode MLP on decoded BF16 operands: RMS → gate/up GEMV → FP32 SiLU×up
// BF16 store → down GEMV → FP32 residual add. Does not unpack Q4.
struct DecodeMlpReference {
  std::vector<std::uint16_t> normalized;
  std::vector<std::uint16_t> swiglu;
  std::vector<float> residual;
};

[[nodiscard]] constexpr std::uint32_t gdn_key_head(
    std::uint32_t value_head) noexcept {
  return value_head / kGdnRepeat;
}

[[nodiscard]] constexpr std::uint32_t kv_head_for_query(
    std::uint32_t query_head) noexcept {
  return query_head / kGqaGroup;
}

[[nodiscard]] float softplus_fp32(float x) noexcept;

// Four-tap causal FIR + SiLU. History is circular [3,10240] oldest at cursor;
// current raw qkv (not SiLU) overwrites that slot. taps is tap-major [4,10240].
[[nodiscard]] std::expected<void, Error> gdn_conv_history_step(
    std::span<std::uint16_t const> qkv, std::span<std::uint16_t const> taps,
    std::span<std::uint16_t> history, std::uint32_t& cursor,
    std::span<std::uint16_t> convolved);

// Eq. (16): per-head L2, ε added inside the sqrt. Writes 16 q/k heads, not 48.
[[nodiscard]] std::expected<void, Error> gdn_qk_normalize(
    std::span<std::uint16_t const> convolved, float eps,
    std::span<float> q_hat, std::span<float> k_hat);

// Eq. (15): β=σ(b), α=exp(-exp(A_log)⊙softplus(a+dt_bias)). a/b FP32, params BF16.
[[nodiscard]] std::expected<void, Error> gdn_alpha_beta(
    std::span<float const> a, std::span<float const> b,
    std::span<std::uint16_t const> a_log, std::span<std::uint16_t const> dt_bias,
    std::span<float> alpha, std::span<float> beta);

[[nodiscard]] std::expected<void, Error> gdn_prepare(
    std::span<std::uint16_t const> convolved, std::span<float const> a,
    std::span<float const> b, std::span<std::uint16_t const> a_log,
    std::span<std::uint16_t const> dt_bias, float eps, std::span<float> q_hat,
    std::span<float> k_hat, std::span<float> alpha, std::span<float> beta);

// Decode GDN steps 1–5 on decoded BF16 operands. History/cursor in/out.
struct GdnFrontReference {
  std::vector<std::uint16_t> normalized;
  std::vector<std::uint16_t> qkv;
  std::vector<std::uint16_t> z;
  std::vector<float> a;
  std::vector<float> b;
  std::vector<std::uint16_t> convolved;
  std::vector<float> q_hat;
  std::vector<float> k_hat;
  std::vector<float> alpha;
  std::vector<float> beta;
  std::vector<std::uint16_t> history;
  std::uint32_t cursor{};
};

// One decode recurrence step. Physical S is FP32 [value_head, value, key];
// q/k are 16 heads indexed by value_head/3; v is BF16 [48,128]; o FP32 [48,128].
// Update precedes readout: D=α S_old, p=Σ D k̂, e=β(v-p), S=D+k̂ e, o=Σ S q̂/√128.
[[nodiscard]] std::expected<void, Error> gdn_recurrence_step(
    std::span<float const> q_hat, std::span<float const> k_hat,
    std::span<float const> alpha, std::span<float const> beta,
    std::span<std::uint16_t const> v, std::span<float> s, std::span<float> o);

[[nodiscard]] std::expected<GdnFrontReference, Error> gdn_front_reference(
    std::span<float const> residual, std::span<std::uint16_t const> gamma,
    float eps, std::span<std::uint16_t const> w_qkv,
    std::span<std::uint16_t const> w_z, std::span<std::uint16_t const> w_a,
    std::span<std::uint16_t const> w_b, std::span<std::uint16_t const> taps,
    std::span<std::uint16_t const> a_log, std::span<std::uint16_t const> dt_bias,
    std::span<std::uint16_t const> history, std::uint32_t cursor);

// Decode GDN steps 1–8 on decoded BF16 operands. S/history/cursor in/out.
// residual_out = residual + W_out vec(u); input residual is not mutated.
struct GdnMixerReference {
  GdnFrontReference front;
  std::vector<float> o;
  std::vector<std::uint16_t> u;
  std::vector<float> residual;
  std::vector<float> s;
  std::vector<std::uint16_t> history;
  std::uint32_t cursor{};
};

[[nodiscard]] std::expected<GdnMixerReference, Error> gdn_mixer_reference(
    std::span<float const> residual, std::span<std::uint16_t const> gamma,
    std::span<std::uint16_t const> gated_gamma, float eps,
    std::span<std::uint16_t const> w_qkv, std::span<std::uint16_t const> w_z,
    std::span<std::uint16_t const> w_a, std::span<std::uint16_t const> w_b,
    std::span<std::uint16_t const> w_out, std::span<std::uint16_t const> taps,
    std::span<std::uint16_t const> a_log, std::span<std::uint16_t const> dt_bias,
    std::span<std::uint16_t const> history, std::uint32_t cursor,
    std::span<float const> s);

[[nodiscard]] std::expected<DecodeMlpReference, Error> decode_mlp_reference(
    std::span<float const> h_mid, std::span<std::uint16_t const> gamma,
    float eps, std::span<std::uint16_t const> w_gate,
    std::span<std::uint16_t const> w_up, std::span<std::uint16_t const> w_down);

// Split projected q/g [24,512] into per-head q' and g, each [24,256].
[[nodiscard]] std::expected<void, Error> attn_split_qg(
    std::span<std::uint16_t const> qg, std::span<std::uint16_t> q_raw,
    std::span<std::uint16_t> g);

// Zero-centered QK RMS in FP32, then partial RoPE, one BF16 store.
[[nodiscard]] std::expected<void, Error> attn_qk_norm_rope(
    std::span<std::uint16_t const> head_bf16,
    std::span<std::uint16_t const> gamma, std::span<float const> inv_freq,
    std::int32_t position, float eps, std::span<std::uint16_t> out_bf16);

// Append one prepared K and V into BF16 [layers,2,4,capacity,256].
[[nodiscard]] std::expected<void, Error> attn_cache_append(
    std::span<std::uint16_t> kv, std::uint32_t attn_layer,
    std::uint64_t capacity, std::uint64_t token,
    std::span<std::uint16_t const> k, std::span<std::uint16_t const> v);

// Decode attention steps 1–3 on decoded BF16 operands. Does not unpack Q4.
struct AttnPrepReference {
  std::vector<std::uint16_t> normalized;
  std::vector<std::uint16_t> qg;
  std::vector<std::uint16_t> k_raw;
  std::vector<std::uint16_t> v_raw;
  std::vector<std::uint16_t> q;
  std::vector<std::uint16_t> g;
  std::vector<std::uint16_t> k;
  std::vector<std::uint16_t> v;
};

[[nodiscard]] std::expected<AttnPrepReference, Error> attn_prep_reference(
    std::span<float const> residual, std::span<std::uint16_t const> gamma,
    float eps, std::span<std::uint16_t const> w_qg,
    std::span<std::uint16_t const> w_k, std::span<std::uint16_t const> w_v,
    std::span<std::uint16_t const> gamma_q, std::span<std::uint16_t const> gamma_k,
    std::span<float const> inv_freq, std::int32_t position);

// Independent FP32 online GQA attention. Segmented uses 256-key chunks and
// merges in increasing segment index; unsegmented is one online pass over
// [0, populated). No quadratic score matrix. Sigmoid gate then BF16 y.
struct AttnCoreReference {
  std::vector<float> attn;       // FP32 [24,256] after normalize, before gate
  std::vector<std::uint16_t> y;  // BF16 gated [24,256]
};

[[nodiscard]] std::expected<AttnCoreReference, Error> attn_online_core(
    std::span<std::uint16_t const> q, std::span<std::uint16_t const> g,
    std::span<std::uint16_t const> kv, std::uint32_t attn_layer,
    std::uint64_t capacity, std::uint64_t populated, bool segmented);

struct AttnMixerReference {
  AttnPrepReference prep;
  AttnCoreReference core;
  std::vector<std::uint16_t> kv;
  std::vector<float> residual;  // h + o_proj(y); input residual is not mutated
};

[[nodiscard]] std::expected<AttnMixerReference, Error> attn_mixer_reference(
    std::span<float const> residual, std::span<std::uint16_t const> gamma,
    float eps, std::span<std::uint16_t const> w_qg,
    std::span<std::uint16_t const> w_k, std::span<std::uint16_t const> w_v,
    std::span<std::uint16_t const> w_o, std::span<std::uint16_t const> gamma_q,
    std::span<std::uint16_t const> gamma_k, std::span<float const> inv_freq,
    std::span<std::uint16_t const> kv_in, std::uint32_t attn_layer,
    std::uint64_t capacity, std::uint64_t token, bool segmented);

// Simple row-major BF16 GEMV: y_n = Σ_k BF16(W_nk)*BF16(x_k) in FP32.
[[nodiscard]] std::expected<void, Error> dense_gemv_bf16(
    std::span<std::uint16_t const> weight, std::span<std::uint16_t const> input,
    std::uint32_t n, std::uint32_t k, std::span<float> out);
[[nodiscard]] std::expected<void, Error> dense_gemv_bf16_f64(
    std::span<std::uint16_t const> weight, std::span<std::uint16_t const> input,
    std::uint32_t n, std::uint32_t k, std::span<double> out);

// Lowest index of the maximum finite-comparable value. Ties keep the first.
[[nodiscard]] std::expected<std::uint32_t, Error> argmax_fp32(
    std::span<float const> logits);

}  // namespace qw38::reference
