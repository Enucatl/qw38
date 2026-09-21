#pragma once

#include "cuda/error.hpp"
#include "cuda/stream.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

namespace qw38::cuda {

[[nodiscard]] std::expected<void, Error> copy_h2d(void* dst, void const* src,
                                                  std::uint64_t bytes,
                                                  Stream const& stream);
[[nodiscard]] std::expected<void, Error> copy_d2h(void* dst, void const* src,
                                                  std::uint64_t bytes,
                                                  Stream const& stream);
[[nodiscard]] std::expected<void, Error> copy_d2d(void* dst, void const* src,
                                                  std::uint64_t bytes,
                                                  Stream const& stream);

[[nodiscard]] std::expected<void, Error> copy_h2d(void* dst,
                                                  std::span<std::byte const> src,
                                                  Stream const& stream);
[[nodiscard]] std::expected<void, Error> copy_d2h(std::span<std::byte> dst,
                                                  void const* src,
                                                  Stream const& stream);

// Deterministic byte pattern: dst[i] = seed + i (uint8 wrap). Used by indexing
// tests, not by inference.
[[nodiscard]] std::expected<void, Error> fill_pattern(void* dst,
                                                      std::uint64_t bytes,
                                                      std::uint8_t seed,
                                                      Stream const& stream);

}  // namespace qw38::cuda
