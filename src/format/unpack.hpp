#pragma once

#include "format/error.hpp"
#include "format/logical.hpp"
#include "format/pack.hpp"

#include <cstdint>
#include <expected>
#include <span>
#include <vector>

namespace qw38::format {

// Independent decoder for cuda_q4g64_v0 / cuda_q8g32_v0. Does not call pack.
[[nodiscard]] std::expected<LogicalWeightCodes, FormatError> unpack_cuda_v0(
    LogicalQuantizerId quantizer, PhysicalLayoutId layout, std::uint64_t n,
    std::uint64_t k, std::span<std::byte const> codes,
    std::span<std::byte const> scales);

[[nodiscard]] std::expected<std::vector<std::byte>, FormatError>
unpack_bf16_dense_tile_v0(std::span<std::byte const> tiled, std::uint64_t n,
                          std::uint64_t k);

}  // namespace qw38::format
