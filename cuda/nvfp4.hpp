#pragma once
#include "cuda/error.hpp"
#include "cuda/stream.hpp"
#include <cstdint>
#include <expected>
namespace qw38::cuda {
// Internal native projection boundary. Caller validates device extents/overlap;
// rows are bounded by its preallocated accumulators and row_start is a 128 multiple.
[[nodiscard]] std::expected<void, Error> nvfp4_gemm_tile(
    void const* codes, void const* scales, std::uint8_t const* activation,
    std::uint8_t const* activation_scales, std::uint32_t m, std::uint32_t n,
    std::uint32_t k, std::uint32_t row_start, float* output,
    void* workspace, std::uint64_t workspace_bytes, Stream const& stream);
[[nodiscard]] std::expected<void, Error> launch_pack_nvfp4(
    std::uint16_t const* input, std::uint8_t* codes, std::uint8_t* scales,
    std::uint32_t m, std::uint32_t k, Stream const& stream);
}
