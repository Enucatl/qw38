#pragma once

#include "format/constants.hpp"
#include "format/error.hpp"
#include "format/logical.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

namespace qw38::format {

struct PackedMatrix {
  PhysicalLayoutId layout{PhysicalLayoutId::CudaQ4G64V0};
  LogicalQuantizerId quantizer{LogicalQuantizerId::Q4G64V0};
  std::uint64_t logical_n{};
  std::uint64_t logical_k{};
  std::uint64_t padded_n{};
  std::uint64_t padded_k{};
  std::vector<std::byte> codes;
  std::vector<std::byte> scales;
};

[[nodiscard]] bool quantizer_layout_pair_ok(LogicalQuantizerId quantizer,
                                            PhysicalLayoutId layout) noexcept;

[[nodiscard]] std::uint64_t dense_pad_n(std::uint64_t n) noexcept;
[[nodiscard]] std::uint64_t dense_pad_k(std::uint64_t k) noexcept;

// Physical packer for cuda_q4g64_v0 / cuda_q8g32_v0. Independent of how codes
// and scales were produced; unit tests supply golden logical values.
[[nodiscard]] std::expected<PackedMatrix, FormatError> pack_cuda_v0(
    LogicalQuantizerId quantizer, PhysicalLayoutId layout,
    LogicalWeightCodes const& logical);

[[nodiscard]] std::expected<PackedMatrix, FormatError> pack_bf16_dense_tile_v0(
    std::span<std::byte const> row_major, std::uint64_t n, std::uint64_t k);

}  // namespace qw38::format
