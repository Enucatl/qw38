#pragma once

#include "format/schema.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace qw38::format {

// Host-only SHA-256 (FIPS 180-4). Implemented locally so qw38_format does not
// depend on libcrypto: the digest is small, public test vectors exist, and the
// V0 dependency policy forbids pulling in a larger cryptographic or ML stack
// for a single hash.
class Sha256 {
 public:
  Sha256() noexcept { reset(); }

  void reset() noexcept;
  void update(std::span<std::byte const> data) noexcept;
  [[nodiscard]] Hash256 finish() noexcept;
  [[nodiscard]] Hash256 digest() const noexcept;

 private:
  void compress(std::span<std::uint8_t const, 64> block) noexcept;

  std::uint32_t state_[8]{};
  std::uint64_t nbytes_{0};
  std::uint8_t buffer_[64]{};
  std::size_t buffer_len_{0};
};

[[nodiscard]] Hash256 sha256(std::span<std::byte const> data) noexcept;

}  // namespace qw38::format
