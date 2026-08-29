#ifndef QW38_QUANT_H_
#define QW38_QUANT_H_

#include <cstddef>
#include <cstdint>

#include "qw38/status.h"

namespace qw38::internal {

constexpr std::size_t kQuantBlockValues = 256;
constexpr std::size_t kQ4KBlockBytes = 144;
constexpr std::size_t kQ6KBlockBytes = 210;
constexpr std::size_t kQ80BlockValues = 32;
constexpr std::size_t kQ80BlockBytes = 34;

Status decode_q4_k(const std::uint8_t* block, std::size_t block_bytes,
                   float* output, std::size_t output_count) noexcept;
Status decode_q6_k(const std::uint8_t* block, std::size_t block_bytes,
                   float* output, std::size_t output_count) noexcept;
Status decode_q8_0(const std::uint8_t* block, std::size_t block_bytes,
                   float* output, std::size_t output_count) noexcept;
Status dot_q4_k(const std::uint8_t* block, std::size_t block_bytes,
                const float* activation, std::size_t activation_count,
                float* output) noexcept;
Status dot_q6_k(const std::uint8_t* block, std::size_t block_bytes,
                const float* activation, std::size_t activation_count,
                float* output) noexcept;
Status dot_q8_0(const std::uint8_t* block, std::size_t block_bytes,
                const float* activation, std::size_t activation_count,
                float* output) noexcept;

}  // namespace qw38::internal

#endif  // QW38_QUANT_H_
