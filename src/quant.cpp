#include "quant.h"

#include <array>
#include <cstring>

namespace qw38::internal {
namespace {

std::uint16_t read_u16_le(const std::uint8_t* bytes) noexcept {
  return static_cast<std::uint16_t>(bytes[0]) |
         (static_cast<std::uint16_t>(bytes[1]) << 8U);
}

float half_to_float(std::uint16_t half) noexcept {
  const std::uint32_t sign =
      static_cast<std::uint32_t>(half & 0x8000U) << 16U;
  std::uint32_t exponent = (half >> 10U) & 0x1FU;
  std::uint32_t fraction = half & 0x03FFU;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (fraction == 0) {
      bits = sign;
    } else {
      std::uint32_t shifts = 0;
      while ((fraction & 0x0400U) == 0) {
        fraction <<= 1U;
        ++shifts;
      }
      fraction &= 0x03FFU;
      bits = sign | ((113U - shifts) << 23U) | (fraction << 13U);
    }
  } else if (exponent == 0x1FU) {
    bits = sign | 0x7F800000U | (fraction << 13U);
  } else {
    exponent += 112U;
    bits = sign | (exponent << 23U) | (fraction << 13U);
  }
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

Status validate(const std::uint8_t* block, std::size_t block_bytes,
                std::size_t expected_bytes, float* output,
                std::size_t output_count, std::size_t expected_count) noexcept {
  if (block == nullptr || output == nullptr) {
    return {StatusCode::kInvalidArgument,
            "quant block and output pointers must not be null"};
  }
  if (block_bytes != expected_bytes) {
    return {StatusCode::kInvalidArgument, "quant block has the wrong byte size"};
  }
  if (output_count != expected_count) {
    return {StatusCode::kInvalidArgument,
            "quant output has the wrong value count"};
  }
  return Status::ok();
}

void q4_scale_min(const std::uint8_t* packed, std::size_t index,
                  std::uint8_t* scale, std::uint8_t* minimum) noexcept {
  if (index < 4) {
    *scale = packed[index] & 63U;
    *minimum = packed[index + 4] & 63U;
    return;
  }
  *scale = static_cast<std::uint8_t>(
      (packed[index + 4] & 15U) | ((packed[index - 4] >> 6U) << 4U));
  *minimum = static_cast<std::uint8_t>(
      (packed[index + 4] >> 4U) | ((packed[index] >> 6U) << 4U));
}

Status dot_decoded(const float* values, const float* activation,
                   std::size_t activation_count, std::size_t expected_count,
                   float* output) noexcept {
  if (activation == nullptr || output == nullptr) {
    return {StatusCode::kInvalidArgument,
            "activation and dot output pointers must not be null"};
  }
  if (activation_count != expected_count) {
    return {StatusCode::kInvalidArgument,
            "quant dot activation has the wrong value count"};
  }
  float total = 0.0F;
  for (std::size_t index = 0; index < expected_count; ++index) {
    const float product = values[index] * activation[index];
    total += product;
  }
  *output = total;
  return Status::ok();
}

}  // namespace

Status decode_q4_k(const std::uint8_t* block, std::size_t block_bytes,
                   float* output, std::size_t output_count) noexcept {
  Status status =
      validate(block, block_bytes, kQ4KBlockBytes, output, output_count,
               kQuantBlockValues);
  if (!status.is_ok()) return status;

  const float d = half_to_float(read_u16_le(block));
  const float dmin = half_to_float(read_u16_le(block + 2));
  const std::uint8_t* packed = block + 4;
  const std::uint8_t* quants = block + 16;
  std::size_t output_offset = 0;
  for (std::size_t group = 0; group < 4; ++group) {
    std::uint8_t low_scale = 0;
    std::uint8_t low_minimum = 0;
    std::uint8_t high_scale = 0;
    std::uint8_t high_minimum = 0;
    q4_scale_min(packed, group * 2, &low_scale, &low_minimum);
    q4_scale_min(packed, group * 2 + 1, &high_scale, &high_minimum);
    const float low_d = d * static_cast<float>(low_scale);
    const float low_min = dmin * static_cast<float>(low_minimum);
    const float high_d = d * static_cast<float>(high_scale);
    const float high_min = dmin * static_cast<float>(high_minimum);
    for (std::size_t lane = 0; lane < 32; ++lane) {
      const std::uint8_t quant = quants[group * 32 + lane];
      output[output_offset + lane] =
          low_d * static_cast<float>(quant & 15U) - low_min;
      output[output_offset + 32 + lane] =
          high_d * static_cast<float>(quant >> 4U) - high_min;
    }
    output_offset += 64;
  }
  return Status::ok();
}

Status decode_q5_k(const std::uint8_t* block, std::size_t block_bytes,
                   float* output, std::size_t output_count) noexcept {
  Status status =
      validate(block, block_bytes, kQ5KBlockBytes, output, output_count,
               kQuantBlockValues);
  if (!status.is_ok()) return status;
  const float d = half_to_float(read_u16_le(block));
  const float dmin = half_to_float(read_u16_le(block + 2));
  const std::uint8_t* packed = block + 4;
  const std::uint8_t* high = block + 16;
  const std::uint8_t* quants = block + 48;
  std::uint8_t low_bit = 1;
  std::uint8_t high_bit = 2;
  std::size_t output_offset = 0;
  for (std::size_t group = 0; group < 4; ++group) {
    std::uint8_t low_scale = 0;
    std::uint8_t low_minimum = 0;
    std::uint8_t high_scale = 0;
    std::uint8_t high_minimum = 0;
    q4_scale_min(packed, group * 2, &low_scale, &low_minimum);
    q4_scale_min(packed, group * 2 + 1, &high_scale, &high_minimum);
    const float low_d = d * static_cast<float>(low_scale);
    const float low_min = dmin * static_cast<float>(low_minimum);
    const float high_d = d * static_cast<float>(high_scale);
    const float high_min = dmin * static_cast<float>(high_minimum);
    for (std::size_t lane = 0; lane < 32; ++lane) {
      const int low_quant = static_cast<int>(quants[lane] & 15U) +
                            ((high[lane] & low_bit) != 0 ? 16 : 0);
      const int high_quant = static_cast<int>(quants[lane] >> 4U) +
                             ((high[lane] & high_bit) != 0 ? 16 : 0);
      output[output_offset + lane] =
          low_d * static_cast<float>(low_quant) - low_min;
      output[output_offset + 32 + lane] =
          high_d * static_cast<float>(high_quant) - high_min;
    }
    quants += 32;
    output_offset += 64;
    low_bit = static_cast<std::uint8_t>(low_bit << 2U);
    high_bit = static_cast<std::uint8_t>(high_bit << 2U);
  }
  return Status::ok();
}

Status decode_q6_k(const std::uint8_t* block, std::size_t block_bytes,
                   float* output, std::size_t output_count) noexcept {
  Status status =
      validate(block, block_bytes, kQ6KBlockBytes, output, output_count,
               kQuantBlockValues);
  if (!status.is_ok()) return status;

  const std::uint8_t* low = block;
  const std::uint8_t* high = block + 128;
  const std::uint8_t* scales = block + 192;
  const float d = half_to_float(read_u16_le(block + 208));
  for (std::size_t half = 0; half < 2; ++half) {
    const std::size_t low_offset = half * 64;
    const std::size_t high_offset = half * 32;
    const std::size_t scale_offset = half * 8;
    const std::size_t output_offset = half * 128;
    for (std::size_t lane = 0; lane < 32; ++lane) {
      const std::size_t scale_pair = lane / 16;
      const int q1 =
          static_cast<int>((low[low_offset + lane] & 15U) |
                           (((high[high_offset + lane] >> 0U) & 3U) << 4U)) -
          32;
      const int q2 = static_cast<int>(
                         (low[low_offset + lane + 32] & 15U) |
                         (((high[high_offset + lane] >> 2U) & 3U) << 4U)) -
                     32;
      const int q3 =
          static_cast<int>((low[low_offset + lane] >> 4U) |
                           (((high[high_offset + lane] >> 4U) & 3U) << 4U)) -
          32;
      const int q4 =
          static_cast<int>((low[low_offset + lane + 32] >> 4U) |
                           (((high[high_offset + lane] >> 6U) & 3U) << 4U)) -
          32;
      const int quants[4] = {q1, q2, q3, q4};
      for (std::size_t group = 0; group < 4; ++group) {
        const std::uint8_t scale_byte =
            scales[scale_offset + scale_pair + group * 2];
        const int scale = scale_byte < 128U ? static_cast<int>(scale_byte)
                                            : static_cast<int>(scale_byte) - 256;
        output[output_offset + lane + group * 32] =
            d * static_cast<float>(scale) * static_cast<float>(quants[group]);
      }
    }
  }
  return Status::ok();
}

Status decode_q8_0(const std::uint8_t* block, std::size_t block_bytes,
                   float* output, std::size_t output_count) noexcept {
  Status status = validate(block, block_bytes, kQ80BlockBytes, output,
                           output_count, kQ80BlockValues);
  if (!status.is_ok()) return status;
  const float scale = half_to_float(read_u16_le(block));
  for (std::size_t index = 0; index < kQ80BlockValues; ++index) {
    const std::uint8_t byte = block[index + 2];
    const int quant =
        byte < 128U ? static_cast<int>(byte) : static_cast<int>(byte) - 256;
    output[index] = scale * static_cast<float>(quant);
  }
  return Status::ok();
}

Status dot_q4_k(const std::uint8_t* block, std::size_t block_bytes,
                const float* activation, std::size_t activation_count,
                float* output) noexcept {
  std::array<float, kQuantBlockValues> values{};
  Status status = decode_q4_k(block, block_bytes, values.data(), values.size());
  if (!status.is_ok()) return status;
  return dot_decoded(values.data(), activation, activation_count,
                     kQuantBlockValues, output);
}

Status dot_q5_k(const std::uint8_t* block, std::size_t block_bytes,
                const float* activation, std::size_t activation_count,
                float* output) noexcept {
  std::array<float, kQuantBlockValues> values{};
  Status status = decode_q5_k(block, block_bytes, values.data(), values.size());
  if (!status.is_ok()) return status;
  return dot_decoded(values.data(), activation, activation_count,
                     kQuantBlockValues, output);
}

Status dot_q6_k(const std::uint8_t* block, std::size_t block_bytes,
                const float* activation, std::size_t activation_count,
                float* output) noexcept {
  std::array<float, kQuantBlockValues> values{};
  Status status = decode_q6_k(block, block_bytes, values.data(), values.size());
  if (!status.is_ok()) return status;
  return dot_decoded(values.data(), activation, activation_count,
                     kQuantBlockValues, output);
}

Status dot_q8_0(const std::uint8_t* block, std::size_t block_bytes,
                const float* activation, std::size_t activation_count,
                float* output) noexcept {
  std::array<float, kQ80BlockValues> values{};
  Status status = decode_q8_0(block, block_bytes, values.data(), values.size());
  if (!status.is_ok()) return status;
  return dot_decoded(values.data(), activation, activation_count,
                     kQ80BlockValues, output);
}

}  // namespace qw38::internal
