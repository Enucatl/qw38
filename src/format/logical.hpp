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
  // Q4_K only: row-major 256-weight blocks, each with d and dmin as
  // little-endian FP16, followed by llama.cpp's 12 packed scale/minimum bytes.
  // Codes are unsigned 0..15; the symmetric quantizer scales vector is empty.
  std::vector<std::uint8_t> q4k_metadata;
};

}  // namespace qw38::format
