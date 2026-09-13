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

std::uint16_t float_to_half(float value) noexcept {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint16_t sign =
      static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
  const std::int32_t exponent =
      static_cast<std::int32_t>((bits >> 23U) & 0xFFU) - 127;
  std::uint32_t fraction = bits & 0x7FFFFFU;
  if ((bits & 0x7FFFFFFFU) == 0U) {
    return sign;
  }
  if (exponent == 128) {
    if (fraction != 0U) {
      return static_cast<std::uint16_t>(sign | 0x7E00U);
    }
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  if (exponent > 15) {
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  if (exponent >= -14) {
    return static_cast<std::uint16_t>(
        sign | static_cast<std::uint16_t>((exponent + 15) << 10) |
        static_cast<std::uint16_t>(fraction >> 13U));
  }
  if (exponent < -24) {
    return sign;
  }
  fraction |= 0x800000U;
  const std::uint32_t shift =
      static_cast<std::uint32_t>(-14 - exponent + 13);
  return static_cast<std::uint16_t>(sign | (fraction >> shift));
}

int nearest_nonneg_int(float value, int max_value) noexcept {
  if (!(value > 0.0F)) return 0;
  const int rounded = static_cast<int>(value + 0.5F);
  if (rounded < 0) return 0;
  if (rounded > max_value) return max_value;
  return rounded;
}

void write_u16_le(std::uint8_t* bytes, std::uint16_t value) noexcept {
  bytes[0] = static_cast<std::uint8_t>(value & 0xFFU);
  bytes[1] = static_cast<std::uint8_t>(value >> 8U);
}

void pack_q4_scale_min(std::uint8_t* packed, const std::uint8_t* scales,
                       const std::uint8_t* mins) noexcept {
  for (std::size_t index = 0; index < 4; ++index) {
    packed[index] = static_cast<std::uint8_t>(
        (scales[index] & 63U) | ((scales[index + 4] & 48U) << 2U));
    packed[index + 4] = static_cast<std::uint8_t>(
        (mins[index] & 63U) | ((mins[index + 4] & 48U) << 2U));
    packed[index + 8] = static_cast<std::uint8_t>(
        (scales[index + 4] & 15U) | ((mins[index + 4] & 15U) << 4U));
  }
}

Status encode_q4_k(const float* input, std::size_t input_count,
                   std::uint8_t* block, std::size_t block_bytes) noexcept {
  if (input == nullptr || block == nullptr) {
    return {StatusCode::kInvalidArgument,
            "quant block and input pointers must not be null"};
  }
  if (block_bytes != kQ4KBlockBytes || input_count != kQuantBlockValues) {
    return {StatusCode::kInvalidArgument, "q4_k encode size mismatch"};
  }
  float group_scale[8];
  float group_min[8];
  std::uint8_t codes[256];
  float max_scale = 0.0F;
  float max_min = 0.0F;
  for (std::size_t group = 0; group < 8; ++group) {
    float minv = input[group * 32];
    float maxv = minv;
    for (std::size_t lane = 1; lane < 32; ++lane) {
      const float value = input[group * 32 + lane];
      if (value < minv) minv = value;
      if (value > maxv) maxv = value;
    }
    if (minv > 0.0F) minv = 0.0F;
    const float range = maxv - minv;
    group_scale[group] = range > 0.0F ? range / 15.0F : 0.0F;
    group_min[group] = -minv;
    if (group_scale[group] > max_scale) max_scale = group_scale[group];
    if (group_min[group] > max_min) max_min = group_min[group];
  }
  const float d = max_scale > 0.0F ? max_scale / 63.0F : 0.0F;
  const float dmin = max_min > 0.0F ? max_min / 63.0F : 0.0F;
  std::uint8_t packed_scales[8];
  std::uint8_t packed_mins[8];
  for (std::size_t group = 0; group < 8; ++group) {
    packed_scales[group] = static_cast<std::uint8_t>(
        nearest_nonneg_int(d > 0.0F ? group_scale[group] / d : 0.0F, 63));
    packed_mins[group] = static_cast<std::uint8_t>(
        nearest_nonneg_int(dmin > 0.0F ? group_min[group] / dmin : 0.0F, 63));
    const float used_d = d * static_cast<float>(packed_scales[group]);
    const float used_min = dmin * static_cast<float>(packed_mins[group]);
    for (std::size_t lane = 0; lane < 32; ++lane) {
      float quant = 0.0F;
      if (used_d > 0.0F) {
        quant = (input[group * 32 + lane] + used_min) / used_d;
      }
      int code = nearest_nonneg_int(quant, 15);
      if (code < 0) code = 0;
      codes[group * 32 + lane] = static_cast<std::uint8_t>(code);
    }
  }
  std::memset(block, 0, kQ4KBlockBytes);
  write_u16_le(block, float_to_half(d));
  write_u16_le(block + 2, float_to_half(dmin));
  pack_q4_scale_min(block + 4, packed_scales, packed_mins);
  std::uint8_t* quants = block + 16;
  for (std::size_t pair = 0; pair < 4; ++pair) {
    for (std::size_t lane = 0; lane < 32; ++lane) {
      const std::uint8_t low = codes[pair * 64 + lane] & 15U;
      const std::uint8_t high = codes[pair * 64 + 32 + lane] & 15U;
      quants[pair * 32 + lane] = static_cast<std::uint8_t>(low | (high << 4U));
    }
  }
  return Status::ok();
}

Status encode_q4_k_row(const float* input, std::size_t columns,
                       std::uint8_t* output,
                       std::size_t output_bytes) noexcept {
  if (input == nullptr || output == nullptr || columns == 0 ||
      columns % kQuantBlockValues != 0) {
    return {StatusCode::kInvalidArgument, "q4_k row encode is not block aligned"};
  }
  const std::size_t blocks = columns / kQuantBlockValues;
  if (output_bytes != blocks * kQ4KBlockBytes) {
    return {StatusCode::kInvalidArgument, "q4_k row output size mismatch"};
  }
  for (std::size_t index = 0; index < blocks; ++index) {
    const Status status =
        encode_q4_k(input + index * kQuantBlockValues, kQuantBlockValues,
                    output + index * kQ4KBlockBytes, kQ4KBlockBytes);
    if (!status.is_ok()) return status;
  }
  return Status::ok();
}

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
