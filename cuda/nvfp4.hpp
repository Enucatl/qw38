#pragma once
#include "cuda/error.hpp"
#include "cuda/stream.hpp"
#include <cstdint>
#include <expected>
namespace qw38::cuda {
// Internal native projection boundary. Caller validates device extents/overlap;
// Prepare the fixed 128x128x128 kernel and maximum workspace at engine binding.
[[nodiscard]] std::expected<int, Error> prepare_nvfp4_gemm(
    std::uint32_t max_m, std::uint64_t workspace_bytes, Stream const& stream);
// Rows are bounded by the caller's preallocated full-width accumulators.
[[nodiscard]] std::expected<void, Error> nvfp4_gemm_tile(
    void const* codes, void const* scales, std::uint8_t const* activation,
    std::uint8_t const* activation_scales, std::uint32_t m, std::uint32_t n,
    std::uint32_t k, std::uint32_t row_start, float* output,
    void* workspace, std::uint64_t workspace_bytes, int sm_count, Stream const& stream);
[[nodiscard]] std::expected<void, Error> launch_pack_nvfp4(
    std::uint16_t const* input, std::uint8_t* codes, std::uint8_t* scales,
    std::uint32_t m, std::uint32_t k, Stream const& stream);
}
