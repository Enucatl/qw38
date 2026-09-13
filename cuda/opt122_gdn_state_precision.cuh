#pragma once

// OPT-122 lower-precision persistent GDN state. Production pin stays FP32
// recurrence. Candidates are BF16 matrices with FP32 update/accumulation, and
// block-scaled integer 8-bit (group 32, FP16 scales). Convolution rings,
// gates/decay, reductions and update arithmetic stay FP32. Packed loads go
// into registers; there is no persistent FP32 shadow of packed state.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace qw38::cuda {

enum class GdnStateFormat : int {
  kFp32 = 0,
  kBf16 = 1,
  kQ8Block32 = 2,
};

constexpr char kLegalGdnStateFp32[] = "fp32";
constexpr char kLegalGdnStateBf16[] = "bf16";
constexpr char kLegalGdnStateQ8[] = "q8_block32";

constexpr GdnStateFormat kSelectedGdnStateFormat = GdnStateFormat::kFp32;
constexpr int kGdnStateQ8Group = 32;
constexpr int kGdnStateQ8Max = 127;
constexpr std::uint32_t kGdnStateCheckpointVersion = 1;
constexpr unsigned kGdnStateLayers = 48;
constexpr unsigned kGdnStateValueHeads = 48;
constexpr unsigned kGdnStateKeyHeads = 16;
constexpr unsigned kGdnStateHeadWidth = 128;
constexpr std::size_t kGdnStateRecurrentValues = 786432;
constexpr std::size_t kGdnStateConvolutionValues = 40960;
constexpr float kGdnStatePeakBandwidthGbS = 1792.0F;

inline thread_local int g_gdn_state_format_override = -1;

inline bool legal_gdn_state_format(GdnStateFormat format) noexcept {
  return format == GdnStateFormat::kFp32 || format == GdnStateFormat::kBf16 ||
         format == GdnStateFormat::kQ8Block32;
}

inline GdnStateFormat gdn_state_format_from_ident(const char* ident) noexcept {
  if (ident == nullptr) return GdnStateFormat::kFp32;
  if (std::strcmp(ident, kLegalGdnStateBf16) == 0) return GdnStateFormat::kBf16;
  if (std::strcmp(ident, kLegalGdnStateQ8) == 0)
    return GdnStateFormat::kQ8Block32;
  return GdnStateFormat::kFp32;
}

inline const char* gdn_state_format_ident(GdnStateFormat format) noexcept {
  switch (format) {
    case GdnStateFormat::kBf16:
      return kLegalGdnStateBf16;
    case GdnStateFormat::kQ8Block32:
      return kLegalGdnStateQ8;
    case GdnStateFormat::kFp32:
    default:
      return kLegalGdnStateFp32;
  }
}

inline GdnStateFormat effective_gdn_state_format() noexcept {
  if (g_gdn_state_format_override >= 0) {
    return static_cast<GdnStateFormat>(g_gdn_state_format_override);
  }
  return kSelectedGdnStateFormat;
}

inline void set_gdn_state_format_override(GdnStateFormat format) noexcept {
  g_gdn_state_format_override = static_cast<int>(format);
}

inline void clear_gdn_state_format_override() noexcept {
  g_gdn_state_format_override = -1;
}

inline std::size_t gdn_state_q8_groups(std::size_t values) noexcept {
  return (values + static_cast<std::size_t>(kGdnStateQ8Group) - 1) /
         static_cast<std::size_t>(kGdnStateQ8Group);
}

inline std::size_t gdn_state_recurrent_payload_bytes(
    GdnStateFormat format, std::size_t layers = kGdnStateLayers) noexcept {
  const std::size_t values = layers * kGdnStateRecurrentValues;
  switch (format) {
    case GdnStateFormat::kBf16:
      return values * sizeof(__nv_bfloat16);
    case GdnStateFormat::kQ8Block32:
      return values * sizeof(std::int8_t) +
             gdn_state_q8_groups(values) * sizeof(__half);
    case GdnStateFormat::kFp32:
    default:
      return values * sizeof(float);
  }
}

inline std::size_t gdn_state_convolution_payload_bytes(
    std::size_t layers = kGdnStateLayers) noexcept {
  return layers * kGdnStateConvolutionValues * sizeof(float);
}

inline std::size_t gdn_state_session_bytes(
    GdnStateFormat format, std::size_t layers = kGdnStateLayers) noexcept {
  return gdn_state_recurrent_payload_bytes(format, layers) +
         gdn_state_convolution_payload_bytes(layers);
}

// Sequential decode touches each recurrent element twice (prediction + update)
// and writes the candidate once. The 3.1 MiB per-layer working set fits in
// RTX 5090 L2, so the second read is not charged as extra DRAM. Convolution
// rings stay FP32: one DRAM read of committed plus one write of candidate.
// Decode commits by pointer swap: no extra full-state copy. Prompt carry D2D
// is unused at the shipping 4096 chunk/microbatch pin (OPT-118).
inline std::size_t gdn_state_decode_dram_bytes(
    GdnStateFormat format, std::size_t layers = kGdnStateLayers) noexcept {
  const std::size_t rec = gdn_state_recurrent_payload_bytes(format, layers);
  const std::size_t conv = gdn_state_convolution_payload_bytes(layers);
  return rec * 2U + conv * 2U;
}

inline std::size_t gdn_state_decode_register_bytes(
    GdnStateFormat format, std::size_t layers = kGdnStateLayers) noexcept {
  const std::size_t rec = gdn_state_recurrent_payload_bytes(format, layers);
  const std::size_t conv = gdn_state_convolution_payload_bytes(layers);
  return rec * 3U + conv * 2U;
}

inline float gdn_state_bytes_to_peak_ms(std::size_t bytes) noexcept {
  return (static_cast<float>(bytes) / (kGdnStatePeakBandwidthGbS * 1.0e9F)) *
         1000.0F;
}

inline __host__ __device__ float gdn_state_bf16_to_fp32(
    __nv_bfloat16 value) noexcept {
  return __bfloat162float(value);
}

inline __host__ __device__ __nv_bfloat16 gdn_state_fp32_to_bf16(
    float value) noexcept {
  return __float2bfloat16(value);
}

inline __host__ __device__ void gdn_state_pack_q8_group(
    const float* source, std::int8_t* packed, __half* scale) noexcept {
  float peak = 0.0F;
  for (int index = 0; index < kGdnStateQ8Group; ++index) {
    const float magnitude = fabsf(source[index]);
    peak = fmaxf(peak, magnitude);
  }
  const float inv = peak > 0.0F
                        ? static_cast<float>(kGdnStateQ8Max) / peak
                        : 0.0F;
  *scale = __float2half(peak > 0.0F ? peak / static_cast<float>(kGdnStateQ8Max)
                                    : 0.0F);
  for (int index = 0; index < kGdnStateQ8Group; ++index) {
    const float quantized = nearbyintf(source[index] * inv);
    const float clipped = fminf(fmaxf(quantized, -127.0F), 127.0F);
    packed[index] = static_cast<std::int8_t>(clipped);
  }
}

inline __host__ __device__ void gdn_state_unpack_q8_group(
    const std::int8_t* packed, __half scale, float* dest) noexcept {
  const float restored = __half2float(scale);
  for (int index = 0; index < kGdnStateQ8Group; ++index) {
    dest[index] = static_cast<float>(packed[index]) * restored;
  }
}

inline void gdn_state_pack_bf16_host(const float* source, __nv_bfloat16* dest,
                                    std::size_t count) noexcept {
  for (std::size_t index = 0; index < count; ++index) {
    dest[index] = gdn_state_fp32_to_bf16(source[index]);
  }
}

inline void gdn_state_unpack_bf16_host(const __nv_bfloat16* source, float* dest,
                                      std::size_t count) noexcept {
  for (std::size_t index = 0; index < count; ++index) {
    dest[index] = gdn_state_bf16_to_fp32(source[index]);
  }
}

inline void gdn_state_pack_q8_host(const float* source, std::int8_t* packed,
                                  __half* scales, std::size_t count) noexcept {
  const std::size_t groups = gdn_state_q8_groups(count);
  for (std::size_t group = 0; group < groups; ++group) {
    float block[kGdnStateQ8Group];
    for (int lane = 0; lane < kGdnStateQ8Group; ++lane) {
      const std::size_t index =
          group * static_cast<std::size_t>(kGdnStateQ8Group) +
          static_cast<std::size_t>(lane);
      block[lane] = index < count ? source[index] : 0.0F;
    }
    gdn_state_pack_q8_group(block,
                            packed + group * static_cast<std::size_t>(
                                                 kGdnStateQ8Group),
                            scales + group);
  }
}

inline void gdn_state_unpack_q8_host(const std::int8_t* packed,
                                    const __half* scales, float* dest,
                                    std::size_t count) noexcept {
  const std::size_t groups = gdn_state_q8_groups(count);
  for (std::size_t group = 0; group < groups; ++group) {
    float block[kGdnStateQ8Group];
    gdn_state_unpack_q8_group(packed + group * static_cast<std::size_t>(
                                                   kGdnStateQ8Group),
                              scales[group], block);
    for (int lane = 0; lane < kGdnStateQ8Group; ++lane) {
      const std::size_t index =
          group * static_cast<std::size_t>(kGdnStateQ8Group) +
          static_cast<std::size_t>(lane);
      if (index < count) dest[index] = block[lane];
    }
  }
}

}  // namespace qw38::cuda
