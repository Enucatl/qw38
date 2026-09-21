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

// QK / head RMS: same 1+γ role on a contiguous 256-vector.
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

// Partial RoPE on the first 64 of 256 coordinates; 192-suffix unchanged.
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

[[nodiscard]] std::expected<DecodeMlpReference, Error> decode_mlp_reference(
    std::span<float const> h_mid, std::span<std::uint16_t const> gamma,
    float eps, std::span<std::uint16_t const> w_gate,
    std::span<std::uint16_t const> w_up, std::span<std::uint16_t const> w_down);

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
