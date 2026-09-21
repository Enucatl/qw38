#include "format/layout.hpp"

#include <limits>

namespace qw38::format {

std::expected<std::uint64_t, FormatError> checked_add(std::uint64_t a,
                                                      std::uint64_t b,
                                                      std::uint64_t offset,
                                                      std::string_view field) {
  if (a > std::numeric_limits<std::uint64_t>::max() - b) {
    return std::unexpected(make_error(FormatErrorCode::Overflow, offset, field,
                                      "unsigned add overflow"));
  }
  return a + b;
}

std::expected<std::uint64_t, FormatError> checked_mul(std::uint64_t a,
                                                      std::uint64_t b,
                                                      std::uint64_t offset,
                                                      std::string_view field) {
  if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b) {
    return std::unexpected(make_error(FormatErrorCode::Overflow, offset, field,
                                      "unsigned multiply overflow"));
  }
  return a * b;
}

std::expected<std::uint64_t, FormatError> align_up(std::uint64_t value,
                                                   std::uint64_t alignment,
                                                   std::uint64_t offset,
                                                   std::string_view field) {
  if (alignment == 0) {
    return std::unexpected(make_error(FormatErrorCode::Misaligned, offset, field,
                                      "alignment must be nonzero"));
  }
  std::uint64_t const rem = value % alignment;
  if (rem == 0) {
    return value;
  }
  return checked_add(value, alignment - rem, offset, field);
}

bool is_aligned(std::uint64_t value, std::uint64_t alignment) noexcept {
  return alignment != 0 && (value % alignment) == 0;
}

std::expected<std::uint64_t, FormatError> checked_product(
    std::span<std::uint64_t const> dims, std::uint64_t offset,
    std::string_view field) {
  std::uint64_t acc = 1;
  for (std::uint64_t dim : dims) {
    auto next = checked_mul(acc, dim, offset, field);
    if (!next) {
      return next;
    }
    acc = *next;
  }
  return acc;
}

std::expected<void, FormatError> check_range(std::uint64_t offset,
                                             std::uint64_t length,
                                             std::uint64_t limit,
                                             std::string_view field) {
  auto const end = checked_add(offset, length, offset, field);
  if (!end) {
    return std::unexpected(end.error());
  }
  if (*end > limit) {
    return std::unexpected(make_error(FormatErrorCode::InvalidSpan, offset, field,
                                      "range exceeds limit"));
  }
  return {};
}

std::expected<void, FormatError> require_span_alignment(
    std::uint64_t offset, std::uint64_t length, std::string_view field) {
  if (length == 0) {
    if (offset != 0) {
      return std::unexpected(make_error(FormatErrorCode::InvalidSpan, offset,
                                        field,
                                        "empty span must use offset 0"));
    }
    return {};
  }
  if (!is_aligned(offset, kSpanAlignment)) {
    return std::unexpected(make_error(FormatErrorCode::Misaligned, offset, field,
                                      "span base must be 256-byte aligned"));
  }
  return {};
}

}  // namespace qw38::format
