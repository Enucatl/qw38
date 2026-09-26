#pragma once
#include "compiler/error.hpp"
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

namespace qw38::compiler {
// FAST-01 development recipe/layout v1: E4M3 RNE, FP32 absmax/448 per
// N128/K128 block, TN row-major codes, ScaleConfig's MN-major scales.
// This is a compiler result, not a serialized C++ structure.
struct Fp8Matrix {
  std::uint32_t n{}, k{}, padded_n{}, padded_k{};
  std::vector<std::uint8_t> codes;
  std::vector<float> scales;
};
[[nodiscard]] std::expected<Fp8Matrix, CompilerError> quantize_fp8(
    std::span<std::byte const> bf16, std::uint32_t n, std::uint32_t k);
}
