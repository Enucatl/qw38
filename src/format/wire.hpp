#pragma once

#include "format/error.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>

namespace qw38::format {

inline constexpr std::endian kWireEndian = std::endian::little;

class ByteWriter {
 public:
  explicit ByteWriter(std::span<std::byte> out) noexcept;

  [[nodiscard]] std::size_t offset() const noexcept { return pos_; }
  [[nodiscard]] std::size_t remaining() const noexcept {
    return buf_.size() - pos_;
  }

  [[nodiscard]] std::expected<void, FormatError> u8(std::uint8_t value,
                                                    std::string_view field);
  [[nodiscard]] std::expected<void, FormatError> u16(std::uint16_t value,
                                                     std::string_view field);
  [[nodiscard]] std::expected<void, FormatError> u32(std::uint32_t value,
                                                     std::string_view field);
  [[nodiscard]] std::expected<void, FormatError> u64(std::uint64_t value,
                                                     std::string_view field);
  [[nodiscard]] std::expected<void, FormatError> bytes(
      std::span<std::byte const> value, std::string_view field);
  [[nodiscard]] std::expected<void, FormatError> zeros(std::size_t count,
                                                       std::string_view field);
  [[nodiscard]] std::expected<void, FormatError> name(std::string_view value,
                                                      std::string_view field);

 private:
  [[nodiscard]] std::expected<void, FormatError> ensure(
      std::size_t count, std::string_view field) const;

  std::span<std::byte> buf_{};
  std::size_t pos_{0};
};

class ByteReader {
 public:
  explicit ByteReader(std::span<std::byte const> in,
                      std::uint64_t base_offset = 0) noexcept;

  [[nodiscard]] std::uint64_t offset() const noexcept {
    return base_offset_ + pos_;
  }
  [[nodiscard]] std::size_t remaining() const noexcept {
    return buf_.size() - pos_;
  }
  [[nodiscard]] bool empty() const noexcept { return remaining() == 0; }

  [[nodiscard]] std::expected<std::uint8_t, FormatError> u8(
      std::string_view field);
  [[nodiscard]] std::expected<std::uint16_t, FormatError> u16(
      std::string_view field);
  [[nodiscard]] std::expected<std::uint32_t, FormatError> u32(
      std::string_view field);
  [[nodiscard]] std::expected<std::uint64_t, FormatError> u64(
      std::string_view field);
  [[nodiscard]] std::expected<void, FormatError> bytes(std::span<std::byte> out,
                                                       std::string_view field);
  [[nodiscard]] std::expected<void, FormatError> expect_zeros(
      std::size_t count, std::string_view field);
  [[nodiscard]] std::expected<std::string, FormatError> name(
      std::string_view field);
  [[nodiscard]] std::expected<void, FormatError> expect_consumed() const;

 private:
  [[nodiscard]] std::expected<void, FormatError> ensure(
      std::size_t count, std::string_view field) const;

  std::span<std::byte const> buf_{};
  std::size_t pos_{0};
  std::uint64_t base_offset_{0};
};

[[nodiscard]] std::uint16_t native_to_le_u16(std::uint16_t value) noexcept;
[[nodiscard]] std::uint32_t native_to_le_u32(std::uint32_t value) noexcept;
[[nodiscard]] std::uint64_t native_to_le_u64(std::uint64_t value) noexcept;
[[nodiscard]] std::uint16_t le_to_native_u16(std::uint16_t value) noexcept;
[[nodiscard]] std::uint32_t le_to_native_u32(std::uint32_t value) noexcept;
[[nodiscard]] std::uint64_t le_to_native_u64(std::uint64_t value) noexcept;

}  // namespace qw38::format
