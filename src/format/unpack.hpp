#pragma once

#include "format/error.hpp"
#include "format/logical.hpp"
#include "format/pack.hpp"

#include <cstdint>
#include <expected>
#include <span>
#include <vector>

namespace qw38::format {

// Validates the complete packed domain consumed by CUDA: code ranges, padded
// coordinates, and FP16 scale encodings. This performs no allocation.
[[nodiscard]] std::expected<void, FormatError> validate_cuda_v0(
    LogicalQuantizerId quantizer, PhysicalLayoutId layout, std::uint64_t n,
    std::uint64_t k, std::span<std::byte const> codes,
    std::span<std::byte const> scales, std::uint64_t codes_file_offset = 0,
    std::uint64_t scales_file_offset = 0);

// Independent decoder for cuda_q4g64_v0 / cuda_q8g32_v0. Does not call pack.
[[nodiscard]] std::expected<LogicalWeightCodes, FormatError> unpack_cuda_v0(
    LogicalQuantizerId quantizer, PhysicalLayoutId layout, std::uint64_t n,
    std::uint64_t k, std::span<std::byte const> codes,
    std::span<std::byte const> scales);

[[nodiscard]] std::expected<std::vector<std::byte>, FormatError>
unpack_bf16_dense_tile_v0(std::span<std::byte const> tiled, std::uint64_t n,
                          std::uint64_t k);

}  // namespace qw38::format
