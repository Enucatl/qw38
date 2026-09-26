#pragma once
#include "cuda/error.hpp"
#include "cuda/decode_mmv.hpp"
#include "cuda/stream.hpp"
#include <cstdint>
#include <expected>
#include <span>

namespace qw38::cuda {
// Explicit borrowed device spans; owners and stream outlive queued work.
// Padded extents are multiples of 128. The caller owns the packed operand
// until every sibling completes; consumers never pack it again.
struct Fp8Input {
  std::span<std::uint8_t const> codes;
  std::span<float const> scales;
  unsigned m{}, k{};
};
struct Fp8Weight {
  std::span<std::uint8_t const> codes;
  std::span<float const> scales;
  unsigned n{}, k{};
};
// Internal: launch_decode_mmv has already validated every descriptor span.
[[nodiscard]] std::expected<void, Error> fp8_decode(DecodeMmvDesc const&, Stream const&);
[[nodiscard]] std::expected<int, Error> prepare_fp8_gemm(Stream const& stream);
[[nodiscard]] std::expected<void, Error> fp8_store_bf16(
    std::span<float const> input, std::span<std::uint16_t> output, Stream const& stream);
[[nodiscard]] std::expected<void, Error> pack_fp8(
    std::span<std::uint16_t const> input, unsigned m, unsigned k,
    std::span<std::uint8_t> codes, std::span<float> scales, Stream const& stream);
[[nodiscard]] std::expected<void, Error> fp8_gemm(
    Fp8Weight weight, Fp8Input input, std::span<float> output,
    std::span<std::byte> workspace, int sm_count, Stream const& stream);
[[nodiscard]] std::expected<void, Error> fp8_gemv(
    Fp8Weight weight, std::span<std::uint16_t const> input,
    std::span<float> output, Stream const& stream);
}
