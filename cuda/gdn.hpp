#pragma once

#include "cuda/error.hpp"
#include "cuda/stream.hpp"

#include <cstdint>
#include <expected>

namespace qw38::cuda {

inline constexpr std::uint32_t kGdnQkvWidth = 10240;
inline constexpr std::uint32_t kGdnZWidth = 6144;
inline constexpr std::uint32_t kGdnKeyHeads = 16;
inline constexpr std::uint32_t kGdnValueHeads = 48;
inline constexpr std::uint32_t kGdnRepeat = 3;
inline constexpr std::uint32_t kGdnConvHistoryTaps = 3;
inline constexpr std::uint32_t kGdnConvKernel = 4;
inline constexpr int kGdnConvThreads = 256;     // channel-parallel FIR
inline constexpr int kGdnPrepThreads = 128;     // T-03: one block/head

// Four-tap causal FIR + SiLU, then overwrite the oldest raw-qkv history slot.
// taps is tap-major [4, 10240]. cursor is the oldest slot in {0,1,2}.
[[nodiscard]] std::expected<void, Error> launch_gdn_conv_silu(
    std::uint16_t const* qkv, std::uint16_t const* taps, std::uint16_t* history,
    std::uint32_t cursor, std::uint16_t* convolved, Stream const& stream);

// One block per key head: L2-normalize q and k in FP32 (Eq. 16) and compute
// three alpha/beta gates (Eq. 15). v is not written; it aliases convolved v.
[[nodiscard]] std::expected<void, Error> launch_gdn_prepare(
    std::uint16_t const* convolved, float const* a, float const* b,
    std::uint16_t const* a_log, std::uint16_t const* dt_bias, float eps,
    float* q_hat, float* k_hat, float* alpha, float* beta, Stream const& stream);

}  // namespace qw38::cuda
