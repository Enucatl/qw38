#pragma once

#include "compiler/error.hpp"
#include "compiler/identity.hpp"
#include "format/constants.hpp"
#include "format/logical.hpp"
#include "format/pack.hpp"

#include <cstdint>
#include <expected>
#include <span>
#include <vector>

namespace qw38::compiler {

struct WeightFormat {
  qw38::format::StorageClass storage{qw38::format::StorageClass::Bf16};
  qw38::format::LogicalQuantizerId quantizer{
      qw38::format::LogicalQuantizerId::None};
  qw38::format::PhysicalLayoutId layout{
      qw38::format::PhysicalLayoutId::CudaBf16VectorV0};
};

enum class WeightFormatPolicy : std::uint8_t {
  IdentityBf16 = 1,
  ProductionV0 = 2,
  CandidateV1 = 3,
  CandidateV2 = 4,
};

[[nodiscard]] bool family_uses_q4g64(TensorFamily family) noexcept;
[[nodiscard]] bool family_uses_q8g32(TensorFamily family) noexcept;

[[nodiscard]] WeightFormat select_weight_format(
    TensorFamily family, qw38::format::PhysicalLayoutId identity_layout,
    WeightFormatPolicy policy) noexcept;

[[nodiscard]] std::expected<std::vector<std::uint16_t>, CompilerError>
dequantize_to_bf16(qw38::format::LogicalWeightCodes const& logical);

[[nodiscard]] std::expected<std::vector<std::uint16_t>, CompilerError>
decode_bf16_payload(qw38::format::PhysicalLayoutId layout,
                    std::span<std::byte const> payload, std::uint64_t n,
                    std::uint64_t k);

// CPU FP32 GEMV. Decoded weight operands are rounded to BF16 before the
// multiply-accumulate; activations are BF16 bits of length K.
[[nodiscard]] std::expected<std::vector<float>, CompilerError>
reference_gemv_bf16(std::span<std::uint16_t const> w_bf16,
                    std::span<std::uint16_t const> x_bf16, std::uint64_t n,
                    std::uint64_t k);

[[nodiscard]] std::expected<std::vector<float>, CompilerError>
reference_gemv_packed(qw38::format::PackedMatrix const& packed,
                      std::span<std::uint16_t const> x_bf16);

}  // namespace qw38::compiler
