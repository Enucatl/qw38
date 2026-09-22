#include "format/wire.hpp"

#include "format/constants.hpp"

#include <cstring>

namespace qw38::format {
namespace {

template <typename T>
T native_to_le(T value) noexcept {
  if constexpr (std::endian::native == std::endian::little) {
    return value;
  } else {
    return std::byteswap(value);
  }
}

template <typename T>
T le_to_native(T value) noexcept {
  if constexpr (std::endian::native == std::endian::little) {
    return value;
  } else {
    return std::byteswap(value);
  }
}

template <typename T>
void store_le_bytes(std::span<std::byte> dest, T native) noexcept {
  T const le = native_to_le(native);
  unsigned char raw[sizeof(T)]{};
  std::memcpy(raw, &le, sizeof(T));
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    dest[i] = static_cast<std::byte>(raw[i]);
  }
}

template <typename T>
T load_le_bytes(std::span<std::byte const> src) noexcept {
  unsigned char raw[sizeof(T)]{};
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    raw[i] = static_cast<unsigned char>(src[i]);
  }
  T le{};
  std::memcpy(&le, raw, sizeof(T));
  return le_to_native(le);
}

}  // namespace

ByteWriter::ByteWriter(std::span<std::byte> out) noexcept : buf_(out) {}

std::expected<void, FormatError> ByteWriter::ensure(
    std::size_t count, std::string_view field) const {
  if (count > remaining()) {
    return std::unexpected(make_error(FormatErrorCode::BufferTooSmall, pos_,
                                      field, "encode buffer exhausted"));
  }
  return {};
}

std::expected<void, FormatError> ByteWriter::u8(std::uint8_t value,
                                                std::string_view field) {
  if (auto st = ensure(1, field); !st) {
    return st;
  }
  buf_[pos_] = static_cast<std::byte>(value);
  ++pos_;
  return {};
}

std::expected<void, FormatError> ByteWriter::u16(std::uint16_t value,
                                                 std::string_view field) {
  if (auto st = ensure(2, field); !st) {
    return st;
  }
  store_le_bytes<std::uint16_t>(buf_.subspan(pos_, 2), value);
  pos_ += 2;
  return {};
}

std::expected<void, FormatError> ByteWriter::u32(std::uint32_t value,
                                                 std::string_view field) {
  if (auto st = ensure(4, field); !st) {
    return st;
  }
  store_le_bytes<std::uint32_t>(buf_.subspan(pos_, 4), value);
  pos_ += 4;
  return {};
}

std::expected<void, FormatError> ByteWriter::u64(std::uint64_t value,
                                                 std::string_view field) {
  if (auto st = ensure(8, field); !st) {
    return st;
  }
  store_le_bytes<std::uint64_t>(buf_.subspan(pos_, 8), value);
  pos_ += 8;
  return {};
}

std::expected<void, FormatError> ByteWriter::bytes(
    std::span<std::byte const> value, std::string_view field) {
  if (auto st = ensure(value.size(), field); !st) {
    return st;
  }
  for (std::size_t i = 0; i < value.size(); ++i) {
    buf_[pos_ + i] = value[i];
  }
  pos_ += value.size();
  return {};
}

std::expected<void, FormatError> ByteWriter::zeros(std::size_t count,
                                                   std::string_view field) {
  if (auto st = ensure(count, field); !st) {
    return st;
  }
  for (std::size_t i = 0; i < count; ++i) {
    buf_[pos_ + i] = std::byte{0};
  }
  pos_ += count;
  return {};
}

std::expected<void, FormatError> ByteWriter::name(std::string_view value,
                                                  std::string_view field) {
  if (value.size() > kMaxNameBytes) {
    return std::unexpected(make_error(FormatErrorCode::NameTooLong, pos_, field,
                                      "logical name exceeds 1024 bytes"));
  }
  if (auto st = u16(static_cast<std::uint16_t>(value.size()), field); !st) {
    return st;
  }
  auto const* p = reinterpret_cast<std::byte const*>(value.data());
  return bytes(std::span<std::byte const>{p, value.size()}, field);
}

ByteReader::ByteReader(std::span<std::byte const> in,
                       std::uint64_t base_offset) noexcept
    : buf_(in), base_offset_(base_offset) {}

std::expected<void, FormatError> ByteReader::ensure(
    std::size_t count, std::string_view field) const {
  if (count > remaining()) {
    return std::unexpected(make_error(FormatErrorCode::Truncated, offset(), field,
                                      "input ends before field"));
  }
  return {};
}

std::expected<std::uint8_t, FormatError> ByteReader::u8(std::string_view field) {
  if (auto st = ensure(1, field); !st) {
    return std::unexpected(st.error());
  }
  auto const value = static_cast<std::uint8_t>(buf_[pos_]);
  ++pos_;
  return value;
}

std::expected<std::uint16_t, FormatError> ByteReader::u16(
    std::string_view field) {
  if (auto st = ensure(2, field); !st) {
    return std::unexpected(st.error());
  }
  auto const value = load_le_bytes<std::uint16_t>(buf_.subspan(pos_, 2));
  pos_ += 2;
  return value;
}

std::expected<std::uint32_t, FormatError> ByteReader::u32(
    std::string_view field) {
  if (auto st = ensure(4, field); !st) {
    return std::unexpected(st.error());
  }
  auto const value = load_le_bytes<std::uint32_t>(buf_.subspan(pos_, 4));
  pos_ += 4;
  return value;
}

std::expected<std::uint64_t, FormatError> ByteReader::u64(
    std::string_view field) {
  if (auto st = ensure(8, field); !st) {
    return std::unexpected(st.error());
  }
  auto const value = load_le_bytes<std::uint64_t>(buf_.subspan(pos_, 8));
  pos_ += 8;
  return value;
}

std::expected<void, FormatError> ByteReader::bytes(std::span<std::byte> out,
                                                   std::string_view field) {
  if (auto st = ensure(out.size(), field); !st) {
    return st;
  }
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = buf_[pos_ + i];
  }
  pos_ += out.size();
  return {};
}

std::expected<void, FormatError> ByteReader::expect_zeros(
    std::size_t count, std::string_view field) {
  if (auto st = ensure(count, field); !st) {
    return st;
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (buf_[pos_ + i] != std::byte{0}) {
      return std::unexpected(make_error(FormatErrorCode::ReservedNonzero,
                                        offset() + i, field,
                                        "reserved bytes must be zero"));
    }
  }
  pos_ += count;
  return {};
}

std::expected<std::string, FormatError> ByteReader::name(
    std::string_view field) {
  auto const len = u16(field);
  if (!len) {
    return std::unexpected(len.error());
  }
  if (*len > kMaxNameBytes) {
    return std::unexpected(make_error(FormatErrorCode::NameTooLong, offset() - 2,
                                      field, "logical name exceeds 1024 bytes"));
  }
  if (auto st = ensure(*len, field); !st) {
    return std::unexpected(st.error());
  }
  std::string out(*len, '\0');
  for (std::uint16_t i = 0; i < *len; ++i) {
    out[i] = static_cast<char>(buf_[pos_ + i]);
  }
  pos_ += *len;
  return out;
}

std::expected<void, FormatError> ByteReader::expect_consumed() const {
  if (!empty()) {
    return std::unexpected(make_error(FormatErrorCode::LeftoverBytes, offset(),
                                      "trailing", "unconsumed manifest bytes"));
  }
  return {};
}

std::uint16_t native_to_le_u16(std::uint16_t value) noexcept {
  return native_to_le(value);
}
std::uint32_t native_to_le_u32(std::uint32_t value) noexcept {
  return native_to_le(value);
}
std::uint64_t native_to_le_u64(std::uint64_t value) noexcept {
  return native_to_le(value);
}
std::uint16_t le_to_native_u16(std::uint16_t value) noexcept {
  return le_to_native(value);
}
std::uint32_t le_to_native_u32(std::uint32_t value) noexcept {
  return le_to_native(value);
}
std::uint64_t le_to_native_u64(std::uint64_t value) noexcept {
  return le_to_native(value);
}

}  // namespace qw38::format
