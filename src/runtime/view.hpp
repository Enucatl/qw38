#pragma once

#include "format/constants.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace qw38::runtime {

enum class MemorySpace : std::uint8_t {
  Device = 1,
  Host = 2,
};

// Borrowed view of known memory. Does not own storage and does not dispatch
// computation. Model views are immutable (writable=false).
struct TensorView {
  void* pointer{nullptr};
  qw38::format::ArithmeticDtype dtype{};
  qw38::format::PhysicalLayoutId layout{};
  qw38::format::StorageClass storage{};
  MemorySpace space{MemorySpace::Device};
  bool writable{false};
  std::uint8_t rank{0};
  std::array<std::uint64_t, qw38::format::kMaxRank> extent{};

  [[nodiscard]] std::span<std::uint64_t const> extents() const noexcept {
    return std::span<std::uint64_t const>{extent.data(), rank};
  }

  friend bool operator==(TensorView const&, TensorView const&) = default;
};

}  // namespace qw38::runtime
