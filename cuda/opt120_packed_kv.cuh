#pragma once

// OPT-120 physically packed integer KV cache. Integer 8-bit is chosen once
// (llama fattn-vec Q8_0 K/V consumers exist; ds4 FP8 writes floats into a
// sizeof(float) raw-cache API). Group size 32, FP16 scales, clip, no recent
// committed BF16 window (in-flight candidate row stays BF16). Packed payload
// plus scales are the authoritative cache; attention unpacks into
// registers/shared memory. No dense shadow.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace qw38::cuda {

inline __host__ __device__ std::size_t packed_kv_physical_index(
    std::size_t token, std::uint32_t kv_head, std::uint32_t lane,
    std::uint32_t capacity, std::uint32_t head_width) noexcept {
  return (static_cast<std::size_t>(kv_head) * capacity + token) * head_width +
         lane;
}

enum class PackedKvFormat : int {
  kDenseBf16 = 0,
  kQ8Q8 = 1,
  kQ8Q4 = 2,
  kQ4Q4 = 3,
};

enum class PackedKvTensor : int {
  kBf16 = 0,
  kQ8 = 1,
  kQ4 = 2,
};

constexpr char kLegalPackedKvDense[] = "dense_bf16";
constexpr char kLegalPackedKvQ8Q8[] = "q8q8";
constexpr char kLegalPackedKvQ8Q4[] = "q8q4";
constexpr char kLegalPackedKvQ4Q4[] = "q4q4";

// Production pin stays dense unless OPT-120 keeps a candidate.
constexpr PackedKvFormat kSelectedPackedKvFormat = PackedKvFormat::kDenseBf16;
constexpr int kPackedKvGroup = 32;
constexpr int kPackedKvQ8Max = 127;
constexpr int kPackedKvQ4Max = 7;
constexpr int kPackedKvRecentWindow = 0;
constexpr std::uint32_t kPackedKvCheckpointVersion = 2;

inline thread_local int g_packed_kv_format_override = -1;

inline bool legal_packed_kv_format(PackedKvFormat format) noexcept {
  return format == PackedKvFormat::kDenseBf16 ||
         format == PackedKvFormat::kQ8Q8 || format == PackedKvFormat::kQ8Q4 ||
         format == PackedKvFormat::kQ4Q4;
}

inline PackedKvFormat packed_kv_format_from_ident(const char* ident) noexcept {
  if (ident == nullptr) return PackedKvFormat::kDenseBf16;
  if (std::strcmp(ident, kLegalPackedKvQ8Q8) == 0) return PackedKvFormat::kQ8Q8;
  if (std::strcmp(ident, kLegalPackedKvQ8Q4) == 0) return PackedKvFormat::kQ8Q4;
  if (std::strcmp(ident, kLegalPackedKvQ4Q4) == 0) return PackedKvFormat::kQ4Q4;
  return PackedKvFormat::kDenseBf16;
}

inline const char* packed_kv_format_ident(PackedKvFormat format) noexcept {
  switch (format) {
    case PackedKvFormat::kQ8Q8:
      return kLegalPackedKvQ8Q8;
    case PackedKvFormat::kQ8Q4:
      return kLegalPackedKvQ8Q4;
    case PackedKvFormat::kQ4Q4:
      return kLegalPackedKvQ4Q4;
    case PackedKvFormat::kDenseBf16:
    default:
      return kLegalPackedKvDense;
  }
}

inline PackedKvFormat effective_packed_kv_format() noexcept {
  if (g_packed_kv_format_override >= 0 &&
      legal_packed_kv_format(
          static_cast<PackedKvFormat>(g_packed_kv_format_override))) {
    return static_cast<PackedKvFormat>(g_packed_kv_format_override);
  }
  return kSelectedPackedKvFormat;
}

inline bool packed_kv_enabled() noexcept {
  return effective_packed_kv_format() != PackedKvFormat::kDenseBf16;
}

inline __host__ __device__ PackedKvTensor packed_kv_key_tensor(
    PackedKvFormat format) noexcept {
  switch (format) {
    case PackedKvFormat::kQ8Q8:
    case PackedKvFormat::kQ8Q4:
      return PackedKvTensor::kQ8;
    case PackedKvFormat::kQ4Q4:
      return PackedKvTensor::kQ4;
    case PackedKvFormat::kDenseBf16:
    default:
      return PackedKvTensor::kBf16;
  }
}

inline __host__ __device__ PackedKvTensor packed_kv_value_tensor(
    PackedKvFormat format) noexcept {
  switch (format) {
    case PackedKvFormat::kQ8Q8:
      return PackedKvTensor::kQ8;
    case PackedKvFormat::kQ8Q4:
    case PackedKvFormat::kQ4Q4:
      return PackedKvTensor::kQ4;
    case PackedKvFormat::kDenseBf16:
    default:
      return PackedKvTensor::kBf16;
  }
}

inline __host__ __device__ std::size_t packed_kv_group_count(
    std::uint32_t head_width) noexcept {
  return (static_cast<std::size_t>(head_width) +
          static_cast<std::size_t>(kPackedKvGroup) - 1U) /
         static_cast<std::size_t>(kPackedKvGroup);
}

inline __host__ __device__ std::size_t packed_kv_payload_bytes(
    PackedKvTensor tensor, std::size_t capacity, std::uint32_t kv_heads,
    std::uint32_t head_width) noexcept {
  const std::size_t values = capacity * static_cast<std::size_t>(kv_heads) *
                             static_cast<std::size_t>(head_width);
  if (tensor == PackedKvTensor::kQ4) return (values + 1U) / 2U;
  if (tensor == PackedKvTensor::kQ8) return values;
  return values * sizeof(__nv_bfloat16);
}

inline __host__ __device__ std::size_t packed_kv_scale_bytes(
    PackedKvTensor tensor, std::size_t capacity, std::uint32_t kv_heads,
    std::uint32_t head_width) noexcept {
  if (tensor == PackedKvTensor::kBf16) return 0;
  return capacity * static_cast<std::size_t>(kv_heads) *
         packed_kv_group_count(head_width) * sizeof(__half);
}

inline __host__ __device__ std::size_t packed_kv_layer_bytes(
    PackedKvTensor tensor, std::size_t capacity, std::uint32_t kv_heads = 4,
    std::uint32_t head_width = 256) noexcept {
  return packed_kv_payload_bytes(tensor, capacity, kv_heads, head_width) +
         packed_kv_scale_bytes(tensor, capacity, kv_heads, head_width);
}

inline std::size_t packed_kv_cache_bytes(PackedKvFormat format,
                                        std::size_t layers,
                                        std::size_t capacity,
                                        std::uint32_t kv_heads = 4,
                                        std::uint32_t head_width = 256) noexcept {
  return layers * (packed_kv_layer_bytes(packed_kv_key_tensor(format), capacity,
                                         kv_heads, head_width) +
                    packed_kv_layer_bytes(packed_kv_value_tensor(format),
                                          capacity, kv_heads, head_width));
}

inline __host__ __device__ __nv_bfloat16* packed_kv_layer_ptr(
    __nv_bfloat16* base, std::size_t layer, PackedKvTensor tensor,
    std::size_t capacity, std::uint32_t kv_heads = 4,
    std::uint32_t head_width = 256) noexcept {
  return reinterpret_cast<__nv_bfloat16*>(
      reinterpret_cast<std::uint8_t*>(base) +
      layer * packed_kv_layer_bytes(tensor, capacity, kv_heads, head_width));
}

inline const __nv_bfloat16* packed_kv_layer_ptr(
    const __nv_bfloat16* base, std::size_t layer, PackedKvTensor tensor,
    std::size_t capacity, std::uint32_t kv_heads = 4,
    std::uint32_t head_width = 256) noexcept {
  return packed_kv_layer_ptr(const_cast<__nv_bfloat16*>(base), layer, tensor,
                             capacity, kv_heads, head_width);
}

inline __nv_bfloat16* packed_kv_committed_key(__nv_bfloat16* base,
                                              std::size_t layer,
                                              std::size_t capacity) noexcept {
  return packed_kv_layer_ptr(base, layer,
                             packed_kv_key_tensor(effective_packed_kv_format()),
                             capacity);
}

inline __nv_bfloat16* packed_kv_committed_value(__nv_bfloat16* base,
                                                std::size_t layer,
                                                std::size_t capacity) noexcept {
  return packed_kv_layer_ptr(
      base, layer, packed_kv_value_tensor(effective_packed_kv_format()),
      capacity);
}

inline const __nv_bfloat16* packed_kv_committed_key(
    const __nv_bfloat16* base, std::size_t layer,
    std::size_t capacity) noexcept {
  return packed_kv_committed_key(const_cast<__nv_bfloat16*>(base), layer,
                                 capacity);
}

inline const __nv_bfloat16* packed_kv_committed_value(
    const __nv_bfloat16* base, std::size_t layer,
    std::size_t capacity) noexcept {
  return packed_kv_committed_value(const_cast<__nv_bfloat16*>(base), layer,
                                   capacity);
}

inline std::size_t packed_kv_frontier_payload_bytes(
    PackedKvTensor tensor, std::size_t frontier, std::uint32_t kv_heads,
    std::uint32_t head_width) noexcept {
  return packed_kv_payload_bytes(tensor, frontier, kv_heads, head_width);
}

inline std::size_t packed_kv_frontier_scale_bytes(
    PackedKvTensor tensor, std::size_t frontier, std::uint32_t kv_heads,
    std::uint32_t head_width) noexcept {
  return packed_kv_scale_bytes(tensor, frontier, kv_heads, head_width);
}

inline std::size_t packed_kv_checkpoint_tensor_bytes(
    PackedKvTensor tensor, std::size_t layers, std::size_t frontier,
    std::uint32_t kv_heads = 4, std::uint32_t head_width = 256) noexcept {
  return layers * (packed_kv_frontier_payload_bytes(tensor, frontier, kv_heads,
                                                    head_width) +
                    packed_kv_frontier_scale_bytes(tensor, frontier, kv_heads,
                                                   head_width));
}

inline __host__ __device__ int packed_kv_qmax(PackedKvTensor tensor) noexcept {
  return tensor == PackedKvTensor::kQ4 ? kPackedKvQ4Max : kPackedKvQ8Max;
}

inline void packed_kv_quantize_group(const float* values, std::size_t count,
                                     PackedKvTensor tensor, std::int8_t* codes,
                                     float* scale) noexcept {
  float amax = 0.0F;
  for (std::size_t index = 0; index < count; ++index) {
    const float magnitude = values[index] < 0.0F ? -values[index] : values[index];
    if (magnitude > amax) amax = magnitude;
  }
  const int qmax = packed_kv_qmax(tensor);
  *scale = amax / static_cast<float>(qmax);
  if (!(*scale > 0.0F)) {
    for (std::size_t index = 0; index < count; ++index) codes[index] = 0;
    *scale = 0.0F;
    return;
  }
  const float inv = 1.0F / *scale;
  for (std::size_t index = 0; index < count; ++index) {
    float quantized = values[index] * inv;
    if (quantized > static_cast<float>(qmax)) {
      quantized = static_cast<float>(qmax);
    } else if (quantized < static_cast<float>(-qmax)) {
      quantized = static_cast<float>(-qmax);
    } else {
      quantized = quantized >= 0.0F ? quantized + 0.5F : quantized - 0.5F;
      quantized = static_cast<float>(static_cast<int>(quantized));
    }
    codes[index] = static_cast<std::int8_t>(quantized);
  }
}

inline void packed_kv_dequantize_group(const std::int8_t* codes, std::size_t count,
                                       float scale, float* values) noexcept {
  for (std::size_t index = 0; index < count; ++index) {
    values[index] = scale * static_cast<float>(codes[index]);
  }
}

inline void packed_kv_pack_q4(const std::int8_t* codes, std::size_t count,
                              std::uint8_t* packed) noexcept {
  for (std::size_t index = 0; index < count; index += 2) {
    const std::uint8_t lo =
        static_cast<std::uint8_t>(codes[index] + kPackedKvQ4Max + 1);
    const std::uint8_t hi =
        index + 1 < count
            ? static_cast<std::uint8_t>(codes[index + 1] + kPackedKvQ4Max + 1)
            : static_cast<std::uint8_t>(kPackedKvQ4Max + 1);
    packed[index / 2] = static_cast<std::uint8_t>(lo | (hi << 4));
  }
}

inline void packed_kv_unpack_q4(const std::uint8_t* packed, std::size_t count,
                                std::int8_t* codes) noexcept {
  for (std::size_t index = 0; index < count; index += 2) {
    const std::uint8_t byte = packed[index / 2];
    codes[index] = static_cast<std::int8_t>((byte & 0x0F) - (kPackedKvQ4Max + 1));
    if (index + 1 < count) {
      codes[index + 1] =
          static_cast<std::int8_t>(((byte >> 4) & 0x0F) - (kPackedKvQ4Max + 1));
    }
  }
}

cudaError_t publish_packed_kv_device_format(PackedKvFormat format) noexcept;

inline void set_packed_kv_format_override(PackedKvFormat format) noexcept {
  g_packed_kv_format_override = static_cast<int>(format);
  publish_packed_kv_device_format(format);
}

inline void clear_packed_kv_format_override() noexcept {
  g_packed_kv_format_override = -1;
  publish_packed_kv_device_format(kSelectedPackedKvFormat);
}

inline bool apply_packed_kv_format_ident(const char* ident) noexcept {
  if (ident == nullptr) return false;
  if (std::strcmp(ident, kLegalPackedKvDense) == 0 ||
      std::strcmp(ident, kLegalPackedKvQ8Q8) == 0 ||
      std::strcmp(ident, kLegalPackedKvQ8Q4) == 0 ||
      std::strcmp(ident, kLegalPackedKvQ4Q4) == 0) {
    set_packed_kv_format_override(packed_kv_format_from_ident(ident));
    return true;
  }
  return false;
}

struct PackedKvFormatScope final {
  explicit PackedKvFormatScope(PackedKvFormat format) noexcept {
    set_packed_kv_format_override(format);
  }
  ~PackedKvFormatScope() { clear_packed_kv_format_override(); }
  PackedKvFormatScope(const PackedKvFormatScope&) = delete;
  PackedKvFormatScope& operator=(const PackedKvFormatScope&) = delete;
};

#ifdef __CUDACC__
__device__ PackedKvFormat packed_kv_device_format();

__device__ __forceinline__ PackedKvTensor packed_kv_device_key_tensor() {
  return packed_kv_key_tensor(packed_kv_device_format());
}

__device__ __forceinline__ PackedKvTensor packed_kv_device_value_tensor() {
  return packed_kv_value_tensor(packed_kv_device_format());
}

__device__ __forceinline__ const std::uint8_t* packed_kv_payload_base(
    const __nv_bfloat16* layer) {
  return reinterpret_cast<const std::uint8_t*>(layer);
}

__device__ __forceinline__ const __half* packed_kv_scale_base(
    const __nv_bfloat16* layer, PackedKvTensor tensor, std::uint32_t capacity,
    std::uint32_t kv_heads, std::uint32_t head_width) {
  const std::uint8_t* payload = packed_kv_payload_base(layer);
  return reinterpret_cast<const __half*>(
      payload + packed_kv_payload_bytes(tensor, capacity, kv_heads, head_width));
}

__device__ __forceinline__ std::uint8_t* packed_kv_payload_base_mut(
    __nv_bfloat16* layer) {
  return reinterpret_cast<std::uint8_t*>(layer);
}

__device__ __forceinline__ __half* packed_kv_scale_base_mut(
    __nv_bfloat16* layer, PackedKvTensor tensor, std::uint32_t capacity,
    std::uint32_t kv_heads, std::uint32_t head_width) {
  std::uint8_t* payload = packed_kv_payload_base_mut(layer);
  return reinterpret_cast<__half*>(
      payload + packed_kv_payload_bytes(tensor, capacity, kv_heads, head_width));
}

__device__ __forceinline__ float packed_kv_load_tensor(
    const __nv_bfloat16* layer, PackedKvTensor tensor, std::size_t token,
    std::uint32_t kv_head, std::uint32_t dim, std::uint32_t capacity,
    std::uint32_t kv_heads, std::uint32_t head_width) {
  if (tensor == PackedKvTensor::kBf16) {
        return __bfloat162float(layer[packed_kv_physical_index(
        token, kv_head, dim, capacity, head_width)]);
  }
  const std::size_t phys =
      packed_kv_physical_index(token, kv_head, dim, capacity, head_width);
  const std::size_t groups = packed_kv_group_count(head_width);
  const std::size_t scale_index =
      (static_cast<std::size_t>(kv_head) * capacity + token) * groups +
      static_cast<std::size_t>(dim) / static_cast<std::size_t>(kPackedKvGroup);
  const float scale = __half2float(
      packed_kv_scale_base(layer, tensor, capacity, kv_heads, head_width)
          [scale_index]);
  if (tensor == PackedKvTensor::kQ8) {
    const std::int8_t code =
        static_cast<std::int8_t>(packed_kv_payload_base(layer)[phys]);
    return scale * static_cast<float>(code);
  }
  const std::uint8_t byte = packed_kv_payload_base(layer)[phys / 2U];
  const int nibble =
      (phys & 1U) != 0U ? static_cast<int>((byte >> 4) & 0x0F)
                        : static_cast<int>(byte & 0x0F);
  return scale * static_cast<float>(nibble - (kPackedKvQ4Max + 1));
}

__device__ __forceinline__ float packed_kv_load_key(
    const __nv_bfloat16* layer, std::size_t token, std::uint32_t kv_head,
    std::uint32_t dim, std::uint32_t capacity, std::uint32_t kv_heads,
    std::uint32_t head_width) {
  return packed_kv_load_tensor(layer, packed_kv_device_key_tensor(), token,
                               kv_head, dim, capacity, kv_heads, head_width);
}

__device__ __forceinline__ float packed_kv_load_value(
    const __nv_bfloat16* layer, std::size_t token, std::uint32_t kv_head,
    std::uint32_t dim, std::uint32_t capacity, std::uint32_t kv_heads,
    std::uint32_t head_width) {
  return packed_kv_load_tensor(layer, packed_kv_device_value_tensor(), token,
                               kv_head, dim, capacity, kv_heads, head_width);
}

__device__ __forceinline__ __nv_bfloat16 packed_kv_load_key_bf16(
    const __nv_bfloat16* layer, std::size_t token, std::uint32_t kv_head,
    std::uint32_t dim, std::uint32_t capacity, std::uint32_t kv_heads,
    std::uint32_t head_width) {
  if (packed_kv_device_format() == PackedKvFormat::kDenseBf16) {
    return layer[packed_kv_physical_index(token, kv_head, dim, capacity,
                                          head_width)];
  }
  return __float2bfloat16_rn(packed_kv_load_key(
      layer, token, kv_head, dim, capacity, kv_heads, head_width));
}

__device__ __forceinline__ __nv_bfloat16 packed_kv_load_value_bf16(
    const __nv_bfloat16* layer, std::size_t token, std::uint32_t kv_head,
    std::uint32_t dim, std::uint32_t capacity, std::uint32_t kv_heads,
    std::uint32_t head_width) {
  if (packed_kv_device_format() == PackedKvFormat::kDenseBf16) {
    return layer[packed_kv_physical_index(token, kv_head, dim, capacity,
                                          head_width)];
  }
  return __float2bfloat16_rn(packed_kv_load_value(
      layer, token, kv_head, dim, capacity, kv_heads, head_width));
}

__device__ __forceinline__ void packed_kv_store_group(
    __nv_bfloat16* layer, PackedKvTensor tensor, std::size_t token,
    std::uint32_t kv_head, std::uint32_t group, const float* values,
    std::uint32_t count, std::uint32_t capacity, std::uint32_t kv_heads,
    std::uint32_t head_width) {
  float amax = 0.0F;
  for (std::uint32_t index = 0; index < count; ++index) {
    const float magnitude = fabsf(values[index]);
    if (magnitude > amax) amax = magnitude;
  }
  const int qmax = packed_kv_qmax(tensor);
  const float scale = amax / static_cast<float>(qmax);
  const std::size_t groups = packed_kv_group_count(head_width);
  packed_kv_scale_base_mut(layer, tensor, capacity, kv_heads,
                           head_width)[(static_cast<std::size_t>(kv_head) *
                                            capacity +
                                        token) *
                                           groups +
                                       group] = __float2half_rn(scale);
  const float inv = scale > 0.0F ? 1.0F / scale : 0.0F;
  for (std::uint32_t index = 0; index < count; ++index) {
    float quantized = values[index] * inv;
    if (quantized > static_cast<float>(qmax))
      quantized = static_cast<float>(qmax);
    else if (quantized < static_cast<float>(-qmax))
      quantized = static_cast<float>(-qmax);
    else
      quantized = quantized >= 0.0F ? quantized + 0.5F : quantized - 0.5F;
    const std::int8_t code = static_cast<std::int8_t>(quantized);
    const std::uint32_t dim = group * kPackedKvGroup + index;
    const std::size_t phys =
        packed_kv_physical_index(token, kv_head, dim, capacity, head_width);
    if (tensor == PackedKvTensor::kQ8) {
      packed_kv_payload_base_mut(layer)[phys] = static_cast<std::uint8_t>(code);
    } else {
      std::uint8_t* byte = packed_kv_payload_base_mut(layer) + phys / 2U;
      const std::uint8_t nibble =
          static_cast<std::uint8_t>(code + kPackedKvQ4Max + 1);
      if ((phys & 1U) == 0U) {
        *byte = static_cast<std::uint8_t>((*byte & 0xF0) | nibble);
      } else {
        *byte = static_cast<std::uint8_t>((*byte & 0x0F) | (nibble << 4));
      }
    }
  }
}

#endif  // __CUDACC__

}  // namespace qw38::cuda
