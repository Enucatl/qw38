#pragma once
#include "compiler/error.hpp"
#include "format/pack.hpp"
namespace qw38::compiler {
[[nodiscard]] std::expected<qw38::format::PackedMatrix, CompilerError>
quantize_nvfp4(std::span<std::byte const> bf16, std::uint64_t n, std::uint64_t k);
}
