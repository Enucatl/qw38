#pragma once

#include "format/constants.hpp"

#include <cstdint>
#include <vector>

namespace qw38::format {

// Logical codes and FP16 scales in row-major group order. Not a physical layout.
struct LogicalWeightCodes {
  LogicalQuantizerId quantizer{LogicalQuantizerId::None};
  std::uint64_t n{};
  std::uint64_t k{};
  std::uint32_t group_size{};
  int qmax{};
  std::vector<std::int8_t> codes;
  std::vector<std::uint16_t> scales;
};

}  // namespace qw38::format
