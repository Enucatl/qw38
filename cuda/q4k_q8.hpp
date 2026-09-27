#pragma once

#include "cuda/error.hpp"
#include "cuda/stream.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

namespace qw38::cuda {
struct PrefillWeight;

// Row-major K32 groups; padding is excluded from the producer reduction.
// Scales are FP32 and sums are exact sums of the signed codes.
struct Q8Input {
  std::span<std::int8_t> codes;
  std::span<float> scales;
  std::span<std::int32_t> sums;
  unsigned m{}, k{}, padded_k{};
};

inline constexpr int kQ4KQ8DotBound = 32 * 15 * 127;
inline constexpr int kQ8SumBound = 32 * 127;
inline constexpr unsigned kQ8MlpMaxTokens = 256;
// One reusable pack (max K=17408), two full-width FP32 gate/up slabs,
// and one producer failure flag. No allocation occurs during execution.
[[nodiscard]] constexpr std::uint64_t q8_mlp_workspace_bytes(unsigned m) {
  return std::uint64_t(m) * (17408 + 17408 / 32 * 8 + 2 * 17408 * 4) + 4;
}

[[nodiscard]] std::expected<void, Error> pack_q8(
    std::span<std::uint16_t const> input, Q8Input output,
    std::span<int> failure, Stream const& stream);
[[nodiscard]] std::expected<void, Error> q4k_q8_project(
    PrefillWeight const& weight, Q8Input input, std::span<float> output,
    Stream const& stream, std::span<float const> residual = {});

// Borrowed workspace, weights and activations remain live until return.
// Completes the entire MLP and returns a typed error for nonfinite producers.
// Internal submission may supply a sticky unit failure flag, cleared/read back
// by its owner; in that case this operation only enqueues and never waits.
// submitted is false for prelaunch rejection, true once device work is attempted.
[[nodiscard]] std::expected<void, Error> q4k_q8_mlp(
    PrefillWeight const& gate, PrefillWeight const& up,
    PrefillWeight const& down, float const* residual,
    std::uint16_t const* gamma, float eps, float* output, unsigned m,
    std::span<std::byte> workspace, Stream const& stream, bool& submitted,
    int* pending_failure = nullptr);
}  // namespace qw38::cuda
