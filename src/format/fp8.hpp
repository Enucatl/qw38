#pragma once
#include "format/error.hpp"
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
namespace qw38::format {
// cuda_fp8_v1: N/K padded to 128, row-major E4M3 codes; little-endian
// FP32 scales at (K/128)*(padded_N/128)+N/128, CUTLASS ScaleConfig MN-major.
[[nodiscard]] std::expected<void,FormatError> validate_fp8(std::uint64_t n,
    std::uint64_t k,std::span<std::byte const> codes,std::span<std::byte const> scales);
}
