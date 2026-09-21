#pragma once

#include "compiler/error.hpp"
#include "format/constants.hpp"
#include "format/logical.hpp"

#include <cstdint>
#include <expected>
#include <span>
#include <vector>

namespace qw38::compiler {

using qw38::format::LogicalWeightCodes;

struct QuantizedGroup {
  std::uint16_t scale_bits{};
  std::vector<std::int8_t> codes;
};

[[nodiscard]] std::uint32_t quantizer_group_size(
    qw38::format::LogicalQuantizerId id) noexcept;
[[nodiscard]] int quantizer_qmax(qw38::format::LogicalQuantizerId id) noexcept;

// Logical group absmax quantizer. Independent of physical packing (A-02).
[[nodiscard]] std::expected<QuantizedGroup, CompilerError> quantize_group(
    qw38::format::LogicalQuantizerId quantizer, std::span<float const> weights);

[[nodiscard]] std::expected<LogicalWeightCodes, CompilerError> quantize_fp32(
    qw38::format::LogicalQuantizerId quantizer, std::uint64_t n, std::uint64_t k,
    std::span<float const> weights);

[[nodiscard]] std::expected<LogicalWeightCodes, CompilerError> quantize_bf16(
    qw38::format::LogicalQuantizerId quantizer, std::uint64_t n, std::uint64_t k,
    std::span<std::byte const> weights);

}  // namespace qw38::compiler
