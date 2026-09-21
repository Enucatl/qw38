#pragma once

#include "format/constants.hpp"
#include "format/error.hpp"

#include <cstdint>
#include <expected>
#include <span>
#include <string_view>

namespace qw38::format {

[[nodiscard]] std::expected<std::uint64_t, FormatError> checked_add(
    std::uint64_t a, std::uint64_t b, std::uint64_t offset,
    std::string_view field);

[[nodiscard]] std::expected<std::uint64_t, FormatError> checked_mul(
    std::uint64_t a, std::uint64_t b, std::uint64_t offset,
    std::string_view field);

[[nodiscard]] std::expected<std::uint64_t, FormatError> align_up(
    std::uint64_t value, std::uint64_t alignment, std::uint64_t offset,
    std::string_view field);

[[nodiscard]] bool is_aligned(std::uint64_t value,
                              std::uint64_t alignment) noexcept;

[[nodiscard]] std::expected<std::uint64_t, FormatError> checked_product(
    std::span<std::uint64_t const> dims, std::uint64_t offset,
    std::string_view field);

[[nodiscard]] std::expected<void, FormatError> check_range(
    std::uint64_t offset, std::uint64_t length, std::uint64_t limit,
    std::string_view field);

[[nodiscard]] std::expected<void, FormatError> require_span_alignment(
    std::uint64_t offset, std::uint64_t length, std::string_view field);

}  // namespace qw38::format
