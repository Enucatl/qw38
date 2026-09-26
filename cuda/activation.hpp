#pragma once

#include "cuda/error.hpp"
#include "cuda/stream.hpp"

#include <cstdint>
#include <expected>

namespace qw38::cuda {

// Shape-specific language dimensions. Tuning thread counts are T-01 / T-03.
inline constexpr std::uint32_t kHidden = 5120;
inline constexpr std::uint32_t kHeadDim = 256;
inline constexpr std::uint32_t kRotaryDim = 64;
inline constexpr std::uint32_t kRopeFreqs = 32;
inline constexpr std::uint32_t kGdnHeadDim = 128;
inline constexpr int kHiddenRmsThreads = 256;  // T-01: one 256-thread block/token
inline constexpr int kHeadNormThreads = 128;   // T-03: one block/head

// Gather one BF16 embedding row into an FP32 residual [5120].
[[nodiscard]] std::expected<void, Error> launch_embed_gather(
    std::uint16_t const* table, std::uint32_t vocab, std::uint32_t token_id,
    float* residual, Stream const& stream);

// Gather valid token rows from device IDs into token-major FP32 residuals.
[[nodiscard]] std::expected<void, Error> launch_embed_gather_chunk(
    std::uint16_t const* table, std::uint32_t vocab,
    std::uint32_t const* token_ids, std::uint32_t valid_tokens,
    float* residual, Stream const& stream);

// Hidden zero-centered RMS (1+gamma) → BF16. residual/out are [n_tokens, 5120].
[[nodiscard]] std::expected<void, Error> launch_hidden_rms(
    float const* residual, std::uint16_t const* gamma, float eps,
    std::uint32_t n_tokens, std::uint16_t* out_bf16, Stream const& stream);

// Authoritative attention preparation boundary. Consume BF16 projection
// staging, keep QK RMS + partial RoPE in FP32, then store BF16 once.
[[nodiscard]] std::expected<void, Error> launch_qk_rms_rope(
    std::uint16_t const* projected_heads_bf16, std::uint16_t const* gamma,
    float eps, float const* inv_freq, std::int32_t position,
    std::uint32_t n_heads, std::uint16_t* out_bf16, Stream const& stream);

// Diagnostic standalone QK RMS → BF16. Non-authoritative before RoPE because
// composing the two rounded primitives introduces an extra BF16 store.
[[nodiscard]] std::expected<void, Error> launch_qk_rms(
    float const* heads, std::uint16_t const* gamma, float eps,
    std::uint32_t n_heads, std::uint16_t* out_bf16, Stream const& stream);

// Per-head GDN multiplicative gated RMS → BF16. o is FP32 [n_heads, 128].
[[nodiscard]] std::expected<void, Error> launch_gdn_gated_rms(
    float const* o, std::uint16_t const* z_bf16, std::uint16_t const* gamma,
    float eps, std::uint32_t n_heads, std::uint16_t* out_bf16,
    Stream const& stream);

[[nodiscard]] std::expected<void, Error> launch_sigmoid_fp32(
    float const* in, float* out, std::uint32_t n, Stream const& stream);
[[nodiscard]] std::expected<void, Error> launch_silu_fp32(
    float const* in, float* out, std::uint32_t n, Stream const& stream);

// Diagnostic standalone partial RoPE over BF16 input. Non-authoritative after
// QK RMS; launch_qk_rms_rope owns the production precision boundary.
[[nodiscard]] std::expected<void, Error> launch_partial_rope(
    std::uint16_t const* heads_bf16, float const* inv_freq,
    std::int32_t position, std::uint32_t n_heads, std::uint16_t* out_bf16,
    Stream const& stream);

// Deterministic argmax: lowest index of the maximum. Rejects any non-finite
// input value and writes one uint32 only for valid input.
[[nodiscard]] std::expected<void, Error> launch_argmax_fp32(
    float const* logits, std::uint32_t n, std::uint32_t* out_index,
    Stream const& stream);

}  // namespace qw38::cuda
