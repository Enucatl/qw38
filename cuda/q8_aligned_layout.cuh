#pragma once

// OPT-100 lossless 64-byte-aligned Q8_0 SoA device layout.
// Scale plane (IEEE-754 binary16, one per 32-value GGUF block) followed by a
// 64-byte pad and a row-major int8 code plane. Versioned descriptor records
// exact block and tensor geometry. Inverse reconstruction is byte-exact with
// the public GGUF Q8_0 payload. No public GGUF rewrite.

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define QW38_Q8_HD __host__ __device__
#else
#define QW38_Q8_HD
#endif

namespace qw38::cuda {

constexpr std::uint32_t kQ8AlignedLayoutVersion = 1;
constexpr std::size_t kQ8GgufBlockBytes = 34;
constexpr std::size_t kQ8GgufBlockValues = 32;
constexpr std::size_t kQ8AlignedAlignBytes = 64;
constexpr char kLegalQ8DeviceLayoutRawGguf[] = "raw_gguf";
constexpr char kLegalQ8DeviceLayoutAlignedSoa[] = "aligned_soa";

struct Q8AlignedLayoutDesc final {
  std::uint32_t version = kQ8AlignedLayoutVersion;
  std::uint32_t reserved = 0;
  std::size_t rows = 0;
  std::size_t columns = 0;
  std::size_t n_blocks_per_row = 0;
  std::size_t n_blocks = 0;
  std::size_t scale_bytes = 0;
  std::size_t pad_bytes = 0;
  std::size_t code_offset = 0;
  std::size_t code_bytes = 0;
  std::size_t total_bytes = 0;
  std::size_t gguf_bytes = 0;
};

QW38_Q8_HD inline bool q8_aligned_geometry_ok(std::size_t rows,
                                             std::size_t columns) noexcept {
  return rows > 0 && columns > 0 && columns % kQ8GgufBlockValues == 0;
}

QW38_Q8_HD inline std::size_t q8_gguf_payload_bytes(std::size_t rows,
                                                   std::size_t columns) noexcept {
  if (!q8_aligned_geometry_ok(rows, columns)) return 0;
  return rows * (columns / kQ8GgufBlockValues) * kQ8GgufBlockBytes;
}

QW38_Q8_HD inline std::size_t q8_aligned_scale_bytes(
    std::size_t rows, std::size_t columns) noexcept {
  if (!q8_aligned_geometry_ok(rows, columns)) return 0;
  return rows * (columns / kQ8GgufBlockValues) * sizeof(std::uint16_t);
}

QW38_Q8_HD inline std::size_t q8_aligned_pad_bytes(
    std::size_t scale_bytes) noexcept {
  return (kQ8AlignedAlignBytes - (scale_bytes % kQ8AlignedAlignBytes)) %
         kQ8AlignedAlignBytes;
}

QW38_Q8_HD inline Q8AlignedLayoutDesc make_q8_aligned_layout_desc(
    std::size_t rows, std::size_t columns) noexcept {
  Q8AlignedLayoutDesc desc{};
  desc.version = kQ8AlignedLayoutVersion;
  if (!q8_aligned_geometry_ok(rows, columns)) return desc;
  desc.rows = rows;
  desc.columns = columns;
  desc.n_blocks_per_row = columns / kQ8GgufBlockValues;
  desc.n_blocks = rows * desc.n_blocks_per_row;
  desc.scale_bytes = desc.n_blocks * sizeof(std::uint16_t);
  desc.pad_bytes = q8_aligned_pad_bytes(desc.scale_bytes);
  desc.code_offset = desc.scale_bytes + desc.pad_bytes;
  desc.code_bytes = rows * columns;
  desc.total_bytes = desc.code_offset + desc.code_bytes;
  desc.gguf_bytes = desc.n_blocks * kQ8GgufBlockBytes;
  return desc;
}

QW38_Q8_HD inline bool q8_aligned_desc_valid(
    const Q8AlignedLayoutDesc& desc) noexcept {
  if (desc.version != kQ8AlignedLayoutVersion || desc.rows == 0 ||
      desc.columns == 0 || desc.columns % kQ8GgufBlockValues != 0) {
    return false;
  }
  const Q8AlignedLayoutDesc expected =
      make_q8_aligned_layout_desc(desc.rows, desc.columns);
  return desc.n_blocks_per_row == expected.n_blocks_per_row &&
         desc.n_blocks == expected.n_blocks &&
         desc.scale_bytes == expected.scale_bytes &&
         desc.pad_bytes == expected.pad_bytes &&
         desc.code_offset == expected.code_offset &&
         desc.code_bytes == expected.code_bytes &&
         desc.total_bytes == expected.total_bytes &&
         desc.gguf_bytes == expected.gguf_bytes;
}

QW38_Q8_HD inline bool q8_aligned_is_byte_neutral(
    const Q8AlignedLayoutDesc& desc) noexcept {
  return q8_aligned_desc_valid(desc) && desc.pad_bytes == 0 &&
         desc.total_bytes == desc.gguf_bytes;
}

QW38_Q8_HD inline const std::uint8_t* q8_aligned_code_plane(
    const std::uint8_t* soa, const Q8AlignedLayoutDesc& desc) noexcept {
  return soa + desc.code_offset;
}

QW38_Q8_HD inline std::uint8_t* q8_aligned_code_plane(
    std::uint8_t* soa, const Q8AlignedLayoutDesc& desc) noexcept {
  return soa + desc.code_offset;
}

inline void pack_q8_aligned_host(const std::uint8_t* gguf,
                                 const Q8AlignedLayoutDesc& desc,
                                 std::uint8_t* soa) noexcept {
  if (gguf == nullptr || soa == nullptr || !q8_aligned_desc_valid(desc)) return;
  std::memset(soa, 0, desc.total_bytes);
  std::uint8_t* scales = soa;
  std::uint8_t* codes = q8_aligned_code_plane(soa, desc);
  for (std::size_t row = 0; row < desc.rows; ++row) {
    for (std::size_t kb = 0; kb < desc.n_blocks_per_row; ++kb) {
      const std::size_t block = row * desc.n_blocks_per_row + kb;
      const std::uint8_t* src = gguf + block * kQ8GgufBlockBytes;
      std::memcpy(scales + block * sizeof(std::uint16_t), src, 2);
      std::memcpy(codes + row * desc.columns + kb * kQ8GgufBlockValues, src + 2,
                  kQ8GgufBlockValues);
    }
  }
}

inline void unpack_q8_aligned_host(const std::uint8_t* soa,
                                   const Q8AlignedLayoutDesc& desc,
                                   std::uint8_t* gguf) noexcept {
  if (gguf == nullptr || soa == nullptr || !q8_aligned_desc_valid(desc)) return;
  const std::uint8_t* scales = soa;
  const std::uint8_t* codes = q8_aligned_code_plane(soa, desc);
  for (std::size_t row = 0; row < desc.rows; ++row) {
    for (std::size_t kb = 0; kb < desc.n_blocks_per_row; ++kb) {
      const std::size_t block = row * desc.n_blocks_per_row + kb;
      std::uint8_t* dst = gguf + block * kQ8GgufBlockBytes;
      std::memcpy(dst, scales + block * sizeof(std::uint16_t), 2);
      std::memcpy(dst + 2, codes + row * desc.columns + kb * kQ8GgufBlockValues,
                  kQ8GgufBlockValues);
    }
  }
}

inline bool q8_aligned_inverse_matches(const std::uint8_t* gguf,
                                       const std::uint8_t* reconstructed,
                                       const Q8AlignedLayoutDesc& desc) noexcept {
  if (gguf == nullptr || reconstructed == nullptr ||
      !q8_aligned_desc_valid(desc)) {
    return false;
  }
  return std::memcmp(gguf, reconstructed, desc.gguf_bytes) == 0;
}

}  // namespace qw38::cuda
