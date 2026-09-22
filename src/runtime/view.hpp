#pragma once

#include "format/constants.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace qw38::runtime {

enum class MemorySpace : std::uint8_t {
  Device = 1,
  Host = 2,
};

// Borrowed view of known memory. Does not own storage and does not dispatch
// computation. Mutable views implicitly convert to const views, never back.
template <typename Pointer>
struct BasicTensorView {
  Pointer pointer{nullptr};
  qw38::format::ArithmeticDtype dtype{};
  qw38::format::PhysicalLayoutId layout{};
  qw38::format::StorageClass storage{};
  MemorySpace space{MemorySpace::Device};
  std::uint8_t rank{0};
  std::array<std::uint64_t, qw38::format::kMaxRank> extent{};

  constexpr BasicTensorView() = default;

  template <typename OtherPointer>
    requires std::convertible_to<OtherPointer, Pointer>
  constexpr BasicTensorView(BasicTensorView<OtherPointer> const& other) noexcept
      : pointer(other.pointer),
        dtype(other.dtype),
        layout(other.layout),
        storage(other.storage),
        space(other.space),
        rank(other.rank),
        extent(other.extent) {}

  [[nodiscard]] std::span<std::uint64_t const> extents() const noexcept {
    return std::span<std::uint64_t const>{extent.data(), rank};
  }

  friend bool operator==(BasicTensorView const&, BasicTensorView const&) = default;
};

using TensorView = BasicTensorView<void*>;
using ConstTensorView = BasicTensorView<void const*>;

static_assert(std::convertible_to<TensorView, ConstTensorView>);
static_assert(!std::convertible_to<ConstTensorView, TensorView>);
static_assert(std::is_same_v<decltype(ConstTensorView::pointer), void const*>);

// A scratch allocation is a byte arena whose regions carry their own tensor
// types. Homogeneous arenas have one region; mixed GDN/attention arenas have
// one entry per typed subregion.
struct WorkspaceRegion {
  std::uint64_t offset{};
  std::uint64_t bytes{};
  std::uint64_t stride_bytes{};
  std::uint64_t repetitions{1};
  TensorView tensor{};
};

struct WorkspaceView {
  std::byte* pointer{nullptr};
  std::uint64_t bytes{};
  MemorySpace space{MemorySpace::Device};
  qw38::format::ScratchKind kind{qw38::format::ScratchKind::ResidualH};
  std::uint8_t region_count{};
  std::array<WorkspaceRegion, 12> region{};

  WorkspaceView() = default;

  // Keeps synthetic workspace construction convenient for low-level tests.
  WorkspaceView(TensorView tensor) noexcept
      : pointer(static_cast<std::byte*>(tensor.pointer)) {
    std::uint64_t elements = tensor.rank == 0 ? 0 : 1;
    for (std::uint8_t i = 0; i < tensor.rank; ++i) {
      elements *= tensor.extent[i];
    }
    bytes = elements * qw38::format::element_size(tensor.dtype);
    space = tensor.space;
    region_count = 1;
    region[0] = WorkspaceRegion{.offset = 0,
                                .bytes = bytes,
                                .stride_bytes = bytes,
                                .repetitions = 1,
                                .tensor = tensor};
  }

  [[nodiscard]] std::span<WorkspaceRegion const> regions() const noexcept {
    return {region.data(), region_count};
  }
};

}  // namespace qw38::runtime
