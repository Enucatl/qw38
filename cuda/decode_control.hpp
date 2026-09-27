#pragma once
#include <cstdint>

namespace qw38::cuda {
// Updated before replay on the owning session stream; never captured by value.
struct DecodeControl {
  std::uint64_t position{};
  std::uint64_t populated{};
  std::uint32_t token{};
  std::uint32_t cursor[48]{};
};
}  // namespace qw38::cuda
