#pragma once

#include "cuda/error.hpp"
#include "cuda/stream.hpp"

#include <cstddef>
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
inline constexpr std::uint32_t kGdnLayers = 48;
inline constexpr int kGdnConvThreads = 256;     // channel-parallel FIR
inline constexpr int kGdnPrepThreads = 128;     // T-03: one block/head
inline constexpr int kGdnRecurThreads = 128;    // T-03: four warps, four value rows
inline constexpr int kGdnRecurWarps = 4;
inline constexpr int kGdnRecurValuesPerBlock = 4;
inline constexpr int kGdnRecurBlocksPerHead = 32;  // 128 values / 4
inline constexpr int kGdnRecurBlocksPerLayer =
    static_cast<int>(kGdnValueHeads) * kGdnRecurBlocksPerHead;  // 1536
inline constexpr std::uint32_t kGdnSElemsPerLayer =
    kGdnValueHeads * 128u * 128u;

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

// Decode recurrence: 1536 blocks/layer, 128 threads. s_layer indexes the
// FP32 [layer,value_head,value,key] session buffer. No allocation.
[[nodiscard]] std::expected<void, Error> launch_gdn_recurrence(
    float const* q_hat, float const* k_hat, float const* alpha, float const* beta,
    std::uint16_t const* v, float* s, std::uint32_t s_layer, float* o,
    Stream const& stream);

// One block per value head: multiplicative gamma, FP32 SiLU(z), BF16 u [48,128].
[[nodiscard]] std::expected<void, Error> launch_gdn_output_transform(
    float const* o, std::uint16_t const* z, std::uint16_t const* gamma, float eps,
    std::uint16_t* u, Stream const& stream);

struct GdnRecurrenceResources {
  int registers{0};
  std::size_t shared_bytes{0};
  std::size_t local_bytes{0};
  std::size_t const_bytes{0};
  int max_threads_per_block{0};
  int occupancy_blocks_per_sm{0};
};

[[nodiscard]] std::expected<GdnRecurrenceResources, Error>
gdn_recurrence_resources();

}  // namespace qw38::cuda
