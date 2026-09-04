#include "tensor.h"

#if defined(__APPLE__) && defined(__x86_64__) && !defined(__AVX2__)
#error "Darwin/x86_64 host inference requires AVX2 (-mavx2)"
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <pthread.h>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "quant.h"

namespace qw38::internal {
namespace {

Status decode_quant_block(std::uint32_t type, const std::uint8_t* block,
                          std::size_t block_bytes, float* output,
                          std::size_t output_count) noexcept {
  if (type == 8) {
    return decode_q8_0(block, block_bytes, output, output_count);
  }
  if (type == 12) {
    return decode_q4_k(block, block_bytes, output, output_count);
  }
  if (type == 13) {
    return decode_q5_k(block, block_bytes, output, output_count);
  }
  if (type == 14) {
    return decode_q6_k(block, block_bytes, output, output_count);
  }
  return {StatusCode::kInvalidArgument, "unsupported quantized tensor type"};
}

bool format_layout(std::uint32_t type, std::size_t* block_values,
                   std::size_t* block_bytes) noexcept {
  switch (type) {
    case 0:
      *block_values = 1;
      *block_bytes = 4;
      return true;
    case 8:
      *block_values = kQ80BlockValues;
      *block_bytes = kQ80BlockBytes;
      return true;
    case 12:
      *block_values = kQuantBlockValues;
      *block_bytes = kQ4KBlockBytes;
      return true;
    case 13:
      *block_values = kQuantBlockValues;
      *block_bytes = kQ5KBlockBytes;
      return true;
    case 14:
      *block_values = kQuantBlockValues;
      *block_bytes = kQ6KBlockBytes;
      return true;
    default:
      return false;
  }
}

float read_f32_le(const std::uint8_t* bytes) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(bytes[0]) |
                             (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                             (static_cast<std::uint32_t>(bytes[2]) << 16U) |
                             (static_cast<std::uint32_t>(bytes[3]) << 24U);
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

Status validate_view(const TensorView& view) noexcept {
  if (view.data == nullptr || view.columns == 0 || view.rows == 0 ||
      view.row_bytes == 0) {
    return {StatusCode::kInvalidArgument, "tensor view is empty"};
  }
  if (view.rows > std::numeric_limits<std::size_t>::max() / view.row_bytes ||
      view.rows * view.row_bytes != view.storage_bytes) {
    return {StatusCode::kInvalidArgument,
            "tensor view storage does not equal its complete rows"};
  }
  return Status::ok();
}

#if defined(__AVX2__)
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

float avx2_dot(const float* values, const float* activation,
               std::size_t count) noexcept {
  __m256 acc = _mm256_setzero_ps();
  std::size_t index = 0;
  for (; index + 8 <= count; index += 8) {
    const __m256 left = _mm256_loadu_ps(values + index);
    const __m256 right = _mm256_loadu_ps(activation + index);
#if defined(__FMA__)
    acc = _mm256_fmadd_ps(left, right, acc);
#else
    acc = _mm256_add_ps(acc, _mm256_mul_ps(left, right));
#endif
  }
  const __m128 low = _mm256_castps256_ps128(acc);
  const __m128 high = _mm256_extractf128_ps(acc, 1);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
  sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
  float total = _mm_cvtss_f32(sum);
  for (; index < count; ++index) {
    total += values[index] * activation[index];
  }
  return total;
}

Status avx2_quant_dot(const TensorView& view, const std::uint8_t* row_data,
                      const float* activation, float* output) noexcept {
  float total = 0.0F;
  std::size_t block_values = 0;
  std::size_t block_bytes = 0;
  if (!format_layout(view.type, &block_values, &block_bytes)) {
    return {StatusCode::kInvalidArgument, "unsupported quantized tensor type"};
  }
  alignas(32) float decoded[kQuantBlockValues];
  for (std::size_t column = 0; column < view.columns;
       column += block_values) {
    const std::uint8_t* block =
        row_data + column / block_values * block_bytes;
    Status status = decode_quant_block(view.type, block, block_bytes, decoded,
                                       block_values);
    if (!status.is_ok()) return status;
    total += avx2_dot(decoded, activation + column, block_values);
  }
  *output = total;
  return Status::ok();
}

struct ActQ8Block final {
  float scale = 0.0F;
  std::int32_t sum = 0;
  std::int8_t qs[kQ80BlockValues]{};
};

void quantize_activation_q8(const float* activation, std::size_t count,
                            ActQ8Block* blocks) noexcept {
  const std::size_t block_count = count / kQ80BlockValues;
  for (std::size_t block = 0; block < block_count; ++block) {
    const float* row = activation + block * kQ80BlockValues;
    float amax = 0.0F;
    for (std::size_t index = 0; index < kQ80BlockValues; ++index) {
      amax = std::max(amax, std::fabs(row[index]));
    }
    const float scale = amax / 127.0F;
    const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
    std::int32_t sum = 0;
    for (std::size_t index = 0; index < kQ80BlockValues; ++index) {
      int quantized = static_cast<int>(std::lrintf(row[index] * inverse));
      if (quantized > 127) quantized = 127;
      if (quantized < -127) quantized = -127;
      blocks[block].qs[index] = static_cast<std::int8_t>(quantized);
      sum += quantized;
    }
    blocks[block].scale = scale;
    blocks[block].sum = sum;
  }
}

int hsum256_epi32(__m256i value) noexcept {
  const __m128i low = _mm256_castsi256_si128(value);
  const __m128i high = _mm256_extracti128_si256(value, 1);
  __m128i sum = _mm_add_epi32(low, high);
  sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E));
  sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1));
  return _mm_cvtsi128_si32(sum);
}

int dot_u8_i8_32(__m256i unsigned_left, const std::int8_t* right) noexcept {
  const __m256i signed_right =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(right));
  const __m256i products16 = _mm256_maddubs_epi16(unsigned_left, signed_right);
  const __m256i products32 =
      _mm256_madd_epi16(products16, _mm256_set1_epi16(1));
  return hsum256_epi32(products32);
}

int dot_i8_i8_32(const std::int8_t* left, const std::int8_t* right) noexcept {
  const __m256i left_v =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(left));
  const __m256i right_v =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(right));
  const __m256i left_low =
      _mm256_cvtepi8_epi16(_mm256_castsi256_si128(left_v));
  const __m256i right_low =
      _mm256_cvtepi8_epi16(_mm256_castsi256_si128(right_v));
  const __m256i left_high =
      _mm256_cvtepi8_epi16(_mm256_extracti128_si256(left_v, 1));
  const __m256i right_high =
      _mm256_cvtepi8_epi16(_mm256_extracti128_si256(right_v, 1));
  const __m256i products = _mm256_add_epi32(
      _mm256_madd_epi16(left_low, right_low),
      _mm256_madd_epi16(left_high, right_high));
  return hsum256_epi32(products);
}

int signed_scale(std::uint8_t byte) noexcept {
  return byte < 128U ? static_cast<int>(byte) : static_cast<int>(byte) - 256;
}

float row_dot_q8_weights(const TensorView& view, const std::uint8_t* row_data,
                         const ActQ8Block* activation) noexcept {
  float total = 0.0F;
  const std::size_t blocks = view.columns / kQ80BlockValues;
  for (std::size_t block = 0; block < blocks; ++block) {
    const std::uint8_t* packed = row_data + block * kQ80BlockBytes;
    const float scale =
        half_to_float(read_u16_le(packed)) * activation[block].scale;
    const int sum = dot_i8_i8_32(
        reinterpret_cast<const std::int8_t*>(packed + 2), activation[block].qs);
    total += scale * static_cast<float>(sum);
  }
  return total;
}

float row_dot_q4_weights(const TensorView& view, const std::uint8_t* row_data,
                         const ActQ8Block* activation) noexcept {
  float total = 0.0F;
  const std::size_t superblocks = view.columns / kQuantBlockValues;
  const __m256i nibble_mask = _mm256_set1_epi8(15);
  const __m256i high_mask = _mm256_set1_epi8(-16);
  for (std::size_t super = 0; super < superblocks; ++super) {
    const std::uint8_t* block = row_data + super * kQ4KBlockBytes;
    const float d = half_to_float(read_u16_le(block));
    const float dmin = half_to_float(read_u16_le(block + 2));
    const std::uint8_t* packed = block + 4;
    const std::uint8_t* quants = block + 16;
    const ActQ8Block* act = activation + super * 8;
    for (std::size_t group = 0; group < 4; ++group) {
      std::uint8_t low_scale = 0;
      std::uint8_t low_minimum = 0;
      std::uint8_t high_scale = 0;
      std::uint8_t high_minimum = 0;
      q4_scale_min(packed, group * 2, &low_scale, &low_minimum);
      q4_scale_min(packed, group * 2 + 1, &high_scale, &high_minimum);
      const __m256i packed_q =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(quants + group * 32));
      const __m256i low_q = _mm256_and_si256(packed_q, nibble_mask);
      const __m256i high_q = _mm256_and_si256(
          _mm256_srli_epi16(_mm256_and_si256(packed_q, high_mask), 4),
          nibble_mask);
      const int low_dot = dot_u8_i8_32(low_q, act[group * 2].qs);
      const int high_dot = dot_u8_i8_32(high_q, act[group * 2 + 1].qs);
      total += d * static_cast<float>(low_scale) * act[group * 2].scale *
                   static_cast<float>(low_dot) -
               dmin * static_cast<float>(low_minimum) * act[group * 2].scale *
                   static_cast<float>(act[group * 2].sum);
      total += d * static_cast<float>(high_scale) * act[group * 2 + 1].scale *
                   static_cast<float>(high_dot) -
               dmin * static_cast<float>(high_minimum) *
                   act[group * 2 + 1].scale *
                   static_cast<float>(act[group * 2 + 1].sum);
    }
  }
  return total;
}

float row_dot_q5_weights(const TensorView& view, const std::uint8_t* row_data,
                         const ActQ8Block* activation) noexcept {
  float total = 0.0F;
  const std::size_t superblocks = view.columns / kQuantBlockValues;
  const __m256i nibble_mask = _mm256_set1_epi8(15);
  const __m256i sixteen = _mm256_set1_epi8(16);
  for (std::size_t super = 0; super < superblocks; ++super) {
    const std::uint8_t* block = row_data + super * kQ5KBlockBytes;
    const float d = half_to_float(read_u16_le(block));
    const float dmin = half_to_float(read_u16_le(block + 2));
    const std::uint8_t* packed = block + 4;
    const std::uint8_t* high = block + 16;
    const std::uint8_t* quants = block + 48;
    const ActQ8Block* act = activation + super * 8;
    const __m256i high_v =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(high));
    std::uint8_t low_bit = 1;
    std::uint8_t high_bit = 2;
    for (std::size_t group = 0; group < 4; ++group) {
      std::uint8_t low_scale = 0;
      std::uint8_t low_minimum = 0;
      std::uint8_t high_scale = 0;
      std::uint8_t high_minimum = 0;
      q4_scale_min(packed, group * 2, &low_scale, &low_minimum);
      q4_scale_min(packed, group * 2 + 1, &high_scale, &high_minimum);
      const __m256i packed_q =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(quants + group * 32));
      const __m256i low_n = _mm256_and_si256(packed_q, nibble_mask);
      const __m256i high_n = _mm256_and_si256(
          _mm256_srli_epi16(_mm256_and_si256(packed_q, _mm256_set1_epi8(-16)), 4),
          nibble_mask);
      const __m256i low_extra = _mm256_and_si256(
          _mm256_cmpeq_epi8(_mm256_and_si256(high_v, _mm256_set1_epi8(
                                  static_cast<char>(low_bit))),
                            _mm256_set1_epi8(static_cast<char>(low_bit))),
          sixteen);
      const __m256i high_extra = _mm256_and_si256(
          _mm256_cmpeq_epi8(_mm256_and_si256(high_v, _mm256_set1_epi8(
                                   static_cast<char>(high_bit))),
                            _mm256_set1_epi8(static_cast<char>(high_bit))),
          sixteen);
      const int low_dot =
          dot_u8_i8_32(_mm256_add_epi8(low_n, low_extra), act[group * 2].qs);
      const int high_dot = dot_u8_i8_32(_mm256_add_epi8(high_n, high_extra),
                                        act[group * 2 + 1].qs);
      total += d * static_cast<float>(low_scale) * act[group * 2].scale *
                   static_cast<float>(low_dot) -
               dmin * static_cast<float>(low_minimum) * act[group * 2].scale *
                   static_cast<float>(act[group * 2].sum);
      total += d * static_cast<float>(high_scale) * act[group * 2 + 1].scale *
                   static_cast<float>(high_dot) -
               dmin * static_cast<float>(high_minimum) *
                   act[group * 2 + 1].scale *
                   static_cast<float>(act[group * 2 + 1].sum);
      low_bit = static_cast<std::uint8_t>(low_bit << 2U);
      high_bit = static_cast<std::uint8_t>(high_bit << 2U);
    }
  }
  return total;
}

__m256i unpack_q6_group(const std::uint8_t* low, const std::uint8_t* high,
                        int low_high_nibble, int high_shift) noexcept {
  const __m256i low_v =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(low));
  const __m256i high_v =
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(high));
  __m256i nibbles = low_v;
  if (low_high_nibble != 0) {
    nibbles = _mm256_and_si256(
        _mm256_srli_epi16(_mm256_and_si256(low_v, _mm256_set1_epi8(-16)), 4),
        _mm256_set1_epi8(15));
  } else {
    nibbles = _mm256_and_si256(low_v, _mm256_set1_epi8(15));
  }
  __m256i high_bits =
      _mm256_and_si256(high_v, _mm256_set1_epi8(static_cast<char>(3 << high_shift)));
  if (high_shift != 0) {
    high_bits = _mm256_and_si256(_mm256_srli_epi16(high_bits, high_shift),
                                 _mm256_set1_epi8(3));
  }
  const __m256i combined = _mm256_or_si256(
      nibbles, _mm256_slli_epi16(high_bits, 4));
  return _mm256_sub_epi8(_mm256_and_si256(combined, _mm256_set1_epi8(63)),
                         _mm256_set1_epi8(32));
}

int dot_i8_i8_16_reg(__m128i left, __m128i right) noexcept {
  return hsum256_epi32(
      _mm256_madd_epi16(_mm256_cvtepi8_epi16(left), _mm256_cvtepi8_epi16(right)));
}

float row_dot_q6_weights(const TensorView& view, const std::uint8_t* row_data,
                         const ActQ8Block* activation) noexcept {
  float total = 0.0F;
  const std::size_t superblocks = view.columns / kQuantBlockValues;
  for (std::size_t super = 0; super < superblocks; ++super) {
    const std::uint8_t* block = row_data + super * kQ6KBlockBytes;
    const std::uint8_t* low = block;
    const std::uint8_t* high = block + 128;
    const std::uint8_t* scales = block + 192;
    const float d = half_to_float(read_u16_le(block + 208));
    const ActQ8Block* act = activation + super * 8;
    for (std::size_t half = 0; half < 2; ++half) {
      const std::size_t low_offset = half * 64;
      const std::size_t high_offset = half * 32;
      const std::size_t scale_offset = half * 8;
      const int shifts[4] = {0, 2, 4, 6};
      const int low_offs[4] = {0, 32, 0, 32};
      const int nibbles[4] = {0, 0, 1, 1};
      for (std::size_t group = 0; group < 4; ++group) {
        const __m256i quants = unpack_q6_group(
            low + low_offset + static_cast<std::size_t>(low_offs[group]),
            high + high_offset, nibbles[group], shifts[group]);
        const __m256i acts = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
            act[half * 4 + group].qs));
        const int scale0 = signed_scale(scales[scale_offset + group * 2]);
        const int scale1 = signed_scale(scales[scale_offset + 1 + group * 2]);
        const int sum0 = dot_i8_i8_16_reg(_mm256_castsi256_si128(quants),
                                          _mm256_castsi256_si128(acts));
        const int sum1 = dot_i8_i8_16_reg(_mm256_extracti128_si256(quants, 1),
                                          _mm256_extracti128_si256(acts, 1));
        const float act_scale = act[half * 4 + group].scale;
        total += d * static_cast<float>(scale0) * act_scale *
                 static_cast<float>(sum0);
        total += d * static_cast<float>(scale1) * act_scale *
                 static_cast<float>(sum1);
      }
    }
  }
  return total;
}

void q6_dot_four_rows(const TensorView& view, std::size_t row,
                      const ActQ8Block* activation, float* output) noexcept {
  float totals[4] = {0.0F, 0.0F, 0.0F, 0.0F};
  const std::uint8_t* rows[4] = {
      view.data + row * view.row_bytes, view.data + (row + 1) * view.row_bytes,
      view.data + (row + 2) * view.row_bytes,
      view.data + (row + 3) * view.row_bytes};
  const std::size_t superblocks = view.columns / kQuantBlockValues;
  for (std::size_t super = 0; super < superblocks; ++super) {
    const ActQ8Block* act = activation + super * 8;
    float d[4];
    const std::uint8_t* low[4];
    const std::uint8_t* high[4];
    const std::uint8_t* scales[4];
    for (int r = 0; r < 4; ++r) {
      const std::uint8_t* block = rows[r] + super * kQ6KBlockBytes;
      low[r] = block;
      high[r] = block + 128;
      scales[r] = block + 192;
      d[r] = half_to_float(read_u16_le(block + 208));
    }
    for (std::size_t half = 0; half < 2; ++half) {
      const std::size_t low_offset = half * 64;
      const std::size_t high_offset = half * 32;
      const std::size_t scale_offset = half * 8;
      const int shifts[4] = {0, 2, 4, 6};
      const int low_offs[4] = {0, 32, 0, 32};
      const int nibbles[4] = {0, 0, 1, 1};
      for (std::size_t group = 0; group < 4; ++group) {
        const __m256i acts = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
            act[half * 4 + group].qs));
        const float act_scale = act[half * 4 + group].scale;
        const __m128i act_lo = _mm256_castsi256_si128(acts);
        const __m128i act_hi = _mm256_extracti128_si256(acts, 1);
        for (int r = 0; r < 4; ++r) {
          const __m256i quants = unpack_q6_group(
              low[r] + low_offset + static_cast<std::size_t>(low_offs[group]),
              high[r] + high_offset, nibbles[group], shifts[group]);
          const int scale0 =
              signed_scale(scales[r][scale_offset + group * 2]);
          const int scale1 =
              signed_scale(scales[r][scale_offset + 1 + group * 2]);
          const int sum0 = dot_i8_i8_16_reg(_mm256_castsi256_si128(quants), act_lo);
          const int sum1 =
              dot_i8_i8_16_reg(_mm256_extracti128_si256(quants, 1), act_hi);
          totals[r] += d[r] * static_cast<float>(scale0) * act_scale *
                       static_cast<float>(sum0);
          totals[r] += d[r] * static_cast<float>(scale1) * act_scale *
                       static_cast<float>(sum1);
        }
      }
    }
  }
  output[row] = totals[0];
  output[row + 1] = totals[1];
  output[row + 2] = totals[2];
  output[row + 3] = totals[3];
}

Status quantized_row_dot(const TensorView& view, std::size_t row,
                         const ActQ8Block* activation, float* output) noexcept {
  const std::uint8_t* row_data = view.data + row * view.row_bytes;
  if (view.type == 8) {
    *output = row_dot_q8_weights(view, row_data, activation);
    return Status::ok();
  }
  if (view.type == 12) {
    *output = row_dot_q4_weights(view, row_data, activation);
    return Status::ok();
  }
  if (view.type == 13) {
    *output = row_dot_q5_weights(view, row_data, activation);
    return Status::ok();
  }
  if (view.type == 14) {
    *output = row_dot_q6_weights(view, row_data, activation);
    return Status::ok();
  }
  return {StatusCode::kInvalidArgument, "unsupported quantized tensor type"};
}
#endif

constexpr std::size_t kMatvecThreadCap = 8;
constexpr std::size_t kMatvecThreadThreshold = 256;

struct MatvecSlice final {
  const TensorView* view = nullptr;
  const float* activation = nullptr;
  const void* quantized = nullptr;
  float* output = nullptr;
  std::size_t begin = 0;
  std::size_t end = 0;
};

struct MatvecPool final {
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t wake = PTHREAD_COND_INITIALIZER;
  pthread_cond_t done = PTHREAD_COND_INITIALIZER;
  pthread_t threads[kMatvecThreadCap]{};
  std::size_t worker_index[kMatvecThreadCap]{};
  MatvecSlice slices[kMatvecThreadCap]{};
  std::size_t workers = 0;
  int generation = 0;
  int busy = 0;
  int errors = 0;
  bool stop = false;
};

MatvecPool g_matvec_pool;
pthread_once_t g_matvec_once = PTHREAD_ONCE_INIT;

std::size_t logical_cpu_count() noexcept {
#if defined(__APPLE__)
  int value = 0;
  std::size_t size = sizeof(value);
  if (sysctlbyname("hw.logicalcpu", &value, &size, nullptr, 0) == 0 &&
      value > 0) {
    return static_cast<std::size_t>(value);
  }
#endif
  const unsigned hardware = std::thread::hardware_concurrency();
  if (hardware == 0) {
    return 1;
  }
  return static_cast<std::size_t>(hardware);
}

std::size_t choose_matvec_workers() noexcept {
  const std::size_t logical = logical_cpu_count();
  if (logical == 0) {
    return 1;
  }
  return logical > kMatvecThreadCap ? kMatvecThreadCap : logical;
}

void* matvec_pool_worker(void* argument) {
  const std::size_t index = *static_cast<const std::size_t*>(argument);
  int generation = 0;
  for (;;) {
    pthread_mutex_lock(&g_matvec_pool.mutex);
    while (!g_matvec_pool.stop && g_matvec_pool.generation == generation) {
      pthread_cond_wait(&g_matvec_pool.wake, &g_matvec_pool.mutex);
    }
    if (g_matvec_pool.stop) {
      pthread_mutex_unlock(&g_matvec_pool.mutex);
      return nullptr;
    }
    generation = g_matvec_pool.generation;
    const MatvecSlice slice = g_matvec_pool.slices[index];
    pthread_mutex_unlock(&g_matvec_pool.mutex);
    bool failed = false;
    std::size_t row = slice.begin;
#if defined(__AVX2__)
    if (slice.quantized != nullptr && slice.view->type == 14) {
      const auto* quantized =
          static_cast<const ActQ8Block*>(slice.quantized);
      for (; row + 3 < slice.end; row += 4) {
        q6_dot_four_rows(*slice.view, row, quantized, slice.output);
      }
    }
#endif
    for (; row < slice.end; ++row) {
      Status status = Status::ok();
      if (slice.quantized != nullptr) {
#if defined(__AVX2__)
        status = quantized_row_dot(
            *slice.view, row, static_cast<const ActQ8Block*>(slice.quantized),
            slice.output + row);
#else
        status = {StatusCode::kInternal, "quantized matvec requires AVX2"};
#endif
      } else {
        status = tensor_row_dot(*slice.view, row, slice.activation,
                                slice.view->columns, slice.output + row);
      }
      if (!status.is_ok()) {
        failed = true;
        break;
      }
    }
    pthread_mutex_lock(&g_matvec_pool.mutex);
    if (failed) g_matvec_pool.errors = 1;
    --g_matvec_pool.busy;
    if (g_matvec_pool.busy == 0) pthread_cond_signal(&g_matvec_pool.done);
    pthread_mutex_unlock(&g_matvec_pool.mutex);
  }
}

void start_matvec_pool() {
  g_matvec_pool.workers = choose_matvec_workers();
  for (std::size_t index = 0; index < g_matvec_pool.workers; ++index) {
    g_matvec_pool.worker_index[index] = index;
    pthread_create(&g_matvec_pool.threads[index], nullptr, matvec_pool_worker,
                   &g_matvec_pool.worker_index[index]);
  }
}

Status parallel_matvec(const TensorView& view, const float* activation,
                       std::size_t activation_count, float* output) noexcept {
  (void)activation_count;
  pthread_once(&g_matvec_once, start_matvec_pool);
  const std::size_t workers = g_matvec_pool.workers;
  const void* quantized_pointer = nullptr;
#if defined(__AVX2__)
  std::vector<ActQ8Block> quantized;
  if (view.type != 0 && view.columns % kQ80BlockValues == 0) {
    quantized.resize(view.columns / kQ80BlockValues);
    quantize_activation_q8(activation, view.columns, quantized.data());
    quantized_pointer = quantized.data();
  }
#endif
  pthread_mutex_lock(&g_matvec_pool.mutex);
  g_matvec_pool.errors = 0;
  g_matvec_pool.busy = static_cast<int>(workers);
  for (std::size_t index = 0; index < workers; ++index) {
    g_matvec_pool.slices[index].view = &view;
    g_matvec_pool.slices[index].activation = activation;
    g_matvec_pool.slices[index].quantized = quantized_pointer;
    g_matvec_pool.slices[index].output = output;
    g_matvec_pool.slices[index].begin = view.rows * index / workers;
    g_matvec_pool.slices[index].end = view.rows * (index + 1) / workers;
  }
  ++g_matvec_pool.generation;
  pthread_cond_broadcast(&g_matvec_pool.wake);
  while (g_matvec_pool.busy != 0) {
    pthread_cond_wait(&g_matvec_pool.done, &g_matvec_pool.mutex);
  }
  const int errors = g_matvec_pool.errors;
  pthread_mutex_unlock(&g_matvec_pool.mutex);
  if (errors != 0) {
    return {StatusCode::kInternal, "threaded matvec worker failed"};
  }
  return Status::ok();
}

}  // namespace

std::size_t matvec_worker_count() noexcept { return choose_matvec_workers(); }

Status make_tensor_view(const std::uint8_t* data, std::size_t storage_bytes,
                        std::uint32_t type, std::size_t columns,
                        std::size_t rows, TensorView* view) noexcept {
  if (view == nullptr) {
    return {StatusCode::kInvalidArgument, "tensor view output is required"};
  }
  std::size_t block_values = 0;
  std::size_t block_bytes = 0;
  if (data == nullptr || columns == 0 || rows == 0 ||
      !format_layout(type, &block_values, &block_bytes) ||
      columns % block_values != 0 ||
      columns / block_values >
          std::numeric_limits<std::size_t>::max() / block_bytes) {
    return {StatusCode::kInvalidArgument,
            "invalid tensor format, dimensions, or block-aligned row width"};
  }
  const std::size_t row_bytes = columns / block_values * block_bytes;
  if (rows > std::numeric_limits<std::size_t>::max() / row_bytes ||
      rows * row_bytes != storage_bytes) {
    return {StatusCode::kInvalidArgument,
            "tensor storage does not contain exactly the declared rows"};
  }
  *view = {data, storage_bytes, type, columns, rows, row_bytes};
  return Status::ok();
}

Status bind_tensor_view(const ModelInfo& info, const MappedFile& mapping,
                        const std::string& name, TensorView* view) noexcept {
  const auto match = std::find_if(
      info.tensors.begin(), info.tensors.end(), [&name](const TensorInfo& tensor) {
        return tensor.name == name;
      });
  if (match == info.tensors.end()) {
    return {StatusCode::kInvalidArgument, "tensor name is not admitted"};
  }
  if (match->dimensions.size() != 2 ||
      match->dimensions[0] > std::numeric_limits<std::size_t>::max() ||
      match->dimensions[1] > std::numeric_limits<std::size_t>::max() ||
      match->storage_bytes > std::numeric_limits<std::size_t>::max() ||
      match->offset > std::numeric_limits<std::size_t>::max() ||
      info.data_offset > std::numeric_limits<std::size_t>::max()) {
    return {StatusCode::kInvalidArgument,
            "tensor is not a supported two-dimensional host matrix"};
  }
  const std::size_t relative = static_cast<std::size_t>(match->offset);
  const std::size_t data_offset = static_cast<std::size_t>(info.data_offset);
  const std::size_t bytes = static_cast<std::size_t>(match->storage_bytes);
  if (relative > std::numeric_limits<std::size_t>::max() - data_offset) {
    return {StatusCode::kInvalidArgument, "tensor absolute offset overflows"};
  }
  const std::size_t absolute = data_offset + relative;
  if (absolute > mapping.size() || bytes > mapping.size() - absolute) {
    return {StatusCode::kInvalidArgument,
            "tensor payload exceeds the mapped artifact"};
  }
  return make_tensor_view(mapping.data() + absolute, bytes, match->type,
                          static_cast<std::size_t>(match->dimensions[0]),
                          static_cast<std::size_t>(match->dimensions[1]), view);
}

Status make_vector_view(const std::uint8_t* data, std::size_t storage_bytes,
                        std::uint32_t type, std::size_t count,
                        VectorView* view) noexcept {
  if (view == nullptr) {
    return {StatusCode::kInvalidArgument, "vector view output is required"};
  }
  if (data == nullptr || type != 0 || count == 0 ||
      count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      count * sizeof(float) != storage_bytes) {
    return {StatusCode::kInvalidArgument,
            "vector must be a complete nonempty F32 payload"};
  }
  *view = {data, storage_bytes, type, count};
  return Status::ok();
}

Status bind_vector_view(const ModelInfo& info, const MappedFile& mapping,
                        const std::string& name, VectorView* view) noexcept {
  const auto match = std::find_if(
      info.tensors.begin(), info.tensors.end(), [&name](const TensorInfo& tensor) {
        return tensor.name == name;
      });
  if (match == info.tensors.end()) {
    return {StatusCode::kInvalidArgument, "vector name is not admitted"};
  }
  if (match->dimensions.size() != 1 ||
      match->dimensions[0] > std::numeric_limits<std::size_t>::max() ||
      match->storage_bytes > std::numeric_limits<std::size_t>::max() ||
      match->offset > std::numeric_limits<std::size_t>::max() ||
      info.data_offset > std::numeric_limits<std::size_t>::max()) {
    return {StatusCode::kInvalidArgument,
            "tensor is not a supported one-dimensional host vector"};
  }
  const std::size_t relative = static_cast<std::size_t>(match->offset);
  const std::size_t data_offset = static_cast<std::size_t>(info.data_offset);
  const std::size_t bytes = static_cast<std::size_t>(match->storage_bytes);
  if (relative > std::numeric_limits<std::size_t>::max() - data_offset) {
    return {StatusCode::kInvalidArgument, "vector absolute offset overflows"};
  }
  const std::size_t absolute = data_offset + relative;
  if (absolute > mapping.size() || bytes > mapping.size() - absolute) {
    return {StatusCode::kInvalidArgument,
            "vector payload exceeds the mapped artifact"};
  }
  return make_vector_view(mapping.data() + absolute, bytes, match->type,
                          static_cast<std::size_t>(match->dimensions[0]), view);
}

Status vector_decode(const VectorView& view, float* output,
                     std::size_t output_count) noexcept {
  if (view.data == nullptr || view.type != 0 || view.count == 0 ||
      view.count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      view.count * sizeof(float) != view.storage_bytes || output == nullptr ||
      output_count != view.count) {
    return {StatusCode::kInvalidArgument,
            "vector view or output count is invalid"};
  }
  for (std::size_t index = 0; index < view.count; ++index) {
    output[index] = read_f32_le(view.data + index * sizeof(float));
  }
  return Status::ok();
}

Status tensor_row_decode(const TensorView& view, std::size_t row, float* output,
                         std::size_t output_count) noexcept {
  Status status = validate_view(view);
  if (!status.is_ok()) return status;
  if (output == nullptr || output_count != view.columns || row >= view.rows) {
    return {StatusCode::kInvalidArgument,
            "tensor decode row or output count is out of range"};
  }
  const std::uint8_t* row_data = view.data + row * view.row_bytes;
  if (view.type == 0) {
    for (std::size_t column = 0; column < view.columns; ++column) {
      output[column] = read_f32_le(row_data + column * 4);
    }
    return Status::ok();
  }
  std::size_t block_values = 0;
  std::size_t block_bytes = 0;
  if (!format_layout(view.type, &block_values, &block_bytes)) {
    return {StatusCode::kInvalidArgument, "unsupported quantized tensor type"};
  }
  for (std::size_t column = 0; column < view.columns;
       column += block_values) {
    const std::uint8_t* block =
        row_data + column / block_values * block_bytes;
    status = decode_quant_block(view.type, block, block_bytes, output + column,
                                block_values);
    if (!status.is_ok()) return status;
  }
  return Status::ok();
}

Status tensor_row_dot(const TensorView& view, std::size_t row,
                      const float* activation, std::size_t activation_count,
                      float* output) noexcept {
  Status status = validate_view(view);
  if (!status.is_ok()) return status;
  if (activation == nullptr || output == nullptr ||
      activation_count != view.columns || row >= view.rows) {
    return {StatusCode::kInvalidArgument,
            "tensor dot row or activation count is out of range"};
  }
  const std::uint8_t* row_data = view.data + row * view.row_bytes;
  if (view.type == 0) {
    float total = 0.0F;
    for (std::size_t column = 0; column < view.columns; ++column) {
      total += read_f32_le(row_data + column * 4) * activation[column];
    }
    *output = total;
    return Status::ok();
  }
#if defined(__AVX2__)
  return avx2_quant_dot(view, row_data, activation, output);
#else
  float total = 0.0F;
  const std::size_t block_values =
      view.type == 8 ? kQ80BlockValues : kQuantBlockValues;
  const std::size_t block_bytes = view.type == 8   ? kQ80BlockBytes
                                  : view.type == 12 ? kQ4KBlockBytes
                                                  : kQ6KBlockBytes;
  for (std::size_t column = 0; column < view.columns;
       column += block_values) {
    const std::uint8_t* block =
        row_data + column / block_values * block_bytes;
    float partial = 0.0F;
    if (view.type == 8) {
      status = dot_q8_0(block, block_bytes, activation + column, block_values,
                        &partial);
    } else if (view.type == 12) {
      status = dot_q4_k(block, block_bytes, activation + column, block_values,
                        &partial);
    } else {
      status = dot_q6_k(block, block_bytes, activation + column, block_values,
                        &partial);
    }
    if (!status.is_ok()) return status;
    total += partial;
  }
  *output = total;
  return Status::ok();
#endif
}

Status tensor_matvec(const TensorView& view, const float* activation,
                     std::size_t activation_count, float* output,
                     std::size_t output_count) noexcept {
  Status status = validate_view(view);
  if (!status.is_ok()) return status;
  if (output == nullptr || output_count != view.rows) {
    return {StatusCode::kInvalidArgument,
            "matvec output count does not equal tensor rows"};
  }
  if (view.rows >= kMatvecThreadThreshold) {
    const char* scalar = std::getenv("QW38_SCALAR_MATVEC");
    if (scalar == nullptr || std::strcmp(scalar, "1") != 0) {
      return parallel_matvec(view, activation, activation_count, output);
    }
  }
  for (std::size_t row = 0; row < view.rows; ++row) {
    status = tensor_row_dot(view, row, activation, activation_count,
                            output + row);
    if (!status.is_ok()) return status;
  }
  return Status::ok();
}

}  // namespace qw38::internal
