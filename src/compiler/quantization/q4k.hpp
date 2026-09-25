#pragma once

#include "compiler/error.hpp"
#include <cstdint>
#include <expected>
#include <span>

namespace qw38::compiler {

// Pinned llama.cpp e6ab7c1a4 reference fit, without an importance matrix.
// Logical codes are unsigned 0..15 in row order. Metadata is little-endian
// FP16 d, FP16 dmin, then llama.cpp's twelve packed scale/minimum bytes.
[[nodiscard]] std::expected<void, CompilerError> quantize_q4k_block(
    std::span<float const, 256> weights, std::span<std::int8_t, 256> codes,
    std::span<std::uint8_t, 16> metadata);

}  // namespace qw38::compiler
