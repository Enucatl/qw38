#pragma once

// OPT-104 lossless 64-byte-aligned Q6_K device layout.
// Superblock d (2-byte half) plane, signed int8 subscale plane (16 bytes),
// low-nibble ql plane (128 bytes), high-bit qh plane (64 bytes). Inverse
// reconstruction is byte-exact with the public GGUF Q6_K payload. No public
// GGUF rewrite.

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define QW38_Q6_HD __host__ __device__
#else
#define QW38_Q6_HD
#endif

namespace qw38::cuda {

constexpr std::uint32_t kQ6KAlignedLayoutVersion = 1;
constexpr std::size_t kQ6KGgufBlockBytes = 210;
constexpr std::size_t kQ6KGgufBlockValues = 256;
constexpr std::size_t kQ6KAlignedAlignBytes = 64;
constexpr std::size_t kQ6KDBytes = 2;
constexpr std::size_t kQ6KScaleBytes = 16;
constexpr std::size_t kQ6KQlBytes = 128;
constexpr std::size_t kQ6KQhBytes = 64;
constexpr char kLegalQ6DeviceLayoutRawGguf[] = "raw_gguf";
constexpr char kLegalQ6DeviceLayoutAlignedSoa[] = "aligned_soa";

struct Q6KAlignedLayoutDesc final {
  std::uint32_t version = kQ6KAlignedLayoutVersion;
  std::uint32_t reserved = 0;
  std::size_t rows = 0;
  std::size_t columns = 0;
  std::size_t n_blocks_per_row = 0;
  std::size_t n_blocks = 0;
  std::size_t d_bytes = 0;
  std::size_t d_pad_bytes = 0;
  std::size_t scale_offset = 0;
  std::size_t scale_bytes = 0;
  std::size_t scale_pad_bytes = 0;
  std::size_t ql_offset = 0;
  std::size_t ql_bytes = 0;
  std::size_t qh_offset = 0;
  std::size_t qh_bytes = 0;
  std::size_t total_bytes = 0;
  std::size_t gguf_bytes = 0;
};

QW38_Q6_HD inline bool q6k_aligned_geometry_ok(std::size_t rows,
                                               std::size_t columns) noexcept {
  return rows > 0 && columns > 0 && columns % kQ6KGgufBlockValues == 0;
}

QW38_Q6_HD inline std::size_t q6k_aligned_pad_bytes(std::size_t bytes) noexcept {
  return (kQ6KAlignedAlignBytes - (bytes % kQ6KAlignedAlignBytes)) %
         kQ6KAlignedAlignBytes;
}

QW38_Q6_HD inline Q6KAlignedLayoutDesc make_q6k_aligned_layout_desc(
    std::size_t rows, std::size_t columns) noexcept {
  Q6KAlignedLayoutDesc desc{};
  desc.version = kQ6KAlignedLayoutVersion;
  if (!q6k_aligned_geometry_ok(rows, columns)) return desc;
  desc.rows = rows;
  desc.columns = columns;
  desc.n_blocks_per_row = columns / kQ6KGgufBlockValues;
  desc.n_blocks = rows * desc.n_blocks_per_row;
  desc.d_bytes = desc.n_blocks * kQ6KDBytes;
  desc.d_pad_bytes = q6k_aligned_pad_bytes(desc.d_bytes);
  desc.scale_offset = desc.d_bytes + desc.d_pad_bytes;
  desc.scale_bytes = desc.n_blocks * kQ6KScaleBytes;
  desc.scale_pad_bytes = q6k_aligned_pad_bytes(desc.scale_bytes);
  desc.ql_offset = desc.scale_offset + desc.scale_bytes + desc.scale_pad_bytes;
  desc.ql_bytes = desc.n_blocks * kQ6KQlBytes;
  desc.qh_offset = desc.ql_offset + desc.ql_bytes;
  desc.qh_bytes = desc.n_blocks * kQ6KQhBytes;
  desc.total_bytes = desc.qh_offset + desc.qh_bytes;
  desc.gguf_bytes = desc.n_blocks * kQ6KGgufBlockBytes;
  return desc;
}

QW38_Q6_HD inline bool q6k_aligned_desc_valid(
    const Q6KAlignedLayoutDesc& desc) noexcept {
  if (desc.version != kQ6KAlignedLayoutVersion || desc.rows == 0 ||
      desc.columns == 0 || desc.columns % kQ6KGgufBlockValues != 0) {
    return false;
  }
  const Q6KAlignedLayoutDesc expected =
      make_q6k_aligned_layout_desc(desc.rows, desc.columns);
  return desc.n_blocks_per_row == expected.n_blocks_per_row &&
         desc.n_blocks == expected.n_blocks && desc.d_bytes == expected.d_bytes &&
         desc.d_pad_bytes == expected.d_pad_bytes &&
         desc.scale_offset == expected.scale_offset &&
         desc.scale_bytes == expected.scale_bytes &&
         desc.scale_pad_bytes == expected.scale_pad_bytes &&
         desc.ql_offset == expected.ql_offset &&
         desc.ql_bytes == expected.ql_bytes &&
         desc.qh_offset == expected.qh_offset &&
         desc.qh_bytes == expected.qh_bytes &&
         desc.total_bytes == expected.total_bytes &&
         desc.gguf_bytes == expected.gguf_bytes;
}

QW38_Q6_HD inline bool q6k_aligned_is_byte_neutral(
    const Q6KAlignedLayoutDesc& desc) noexcept {
  return q6k_aligned_desc_valid(desc) && desc.d_pad_bytes == 0 &&
         desc.scale_pad_bytes == 0 && desc.total_bytes == desc.gguf_bytes;
}

QW38_Q6_HD inline const std::uint8_t* q6k_aligned_d_plane(
    const std::uint8_t* soa) noexcept {
  return soa;
}

QW38_Q6_HD inline const std::uint8_t* q6k_aligned_scale_plane(
    const std::uint8_t* soa, const Q6KAlignedLayoutDesc& desc) noexcept {
  return soa + desc.scale_offset;
}

QW38_Q6_HD inline const std::uint8_t* q6k_aligned_ql_plane(
    const std::uint8_t* soa, const Q6KAlignedLayoutDesc& desc) noexcept {
  return soa + desc.ql_offset;
}

QW38_Q6_HD inline const std::uint8_t* q6k_aligned_qh_plane(
    const std::uint8_t* soa, const Q6KAlignedLayoutDesc& desc) noexcept {
  return soa + desc.qh_offset;
}

inline void pack_q6k_aligned_host(const std::uint8_t* gguf,
                                  const Q6KAlignedLayoutDesc& desc,
                                  std::uint8_t* soa) noexcept {
  if (gguf == nullptr || soa == nullptr || !q6k_aligned_desc_valid(desc)) return;
  std::memset(soa, 0, desc.total_bytes);
  std::uint8_t* d = soa;
  std::uint8_t* scales = soa + desc.scale_offset;
  std::uint8_t* ql = soa + desc.ql_offset;
  std::uint8_t* qh = soa + desc.qh_offset;
  for (std::size_t block = 0; block < desc.n_blocks; ++block) {
    const std::uint8_t* src = gguf + block * kQ6KGgufBlockBytes;
    std::memcpy(ql + block * kQ6KQlBytes, src, kQ6KQlBytes);
    std::memcpy(qh + block * kQ6KQhBytes, src + kQ6KQlBytes, kQ6KQhBytes);
    std::memcpy(scales + block * kQ6KScaleBytes, src + 192, kQ6KScaleBytes);
    std::memcpy(d + block * kQ6KDBytes, src + 208, kQ6KDBytes);
  }
}

inline void unpack_q6k_aligned_host(const std::uint8_t* soa,
                                    const Q6KAlignedLayoutDesc& desc,
                                    std::uint8_t* gguf) noexcept {
  if (gguf == nullptr || soa == nullptr || !q6k_aligned_desc_valid(desc)) return;
  const std::uint8_t* d = soa;
  const std::uint8_t* scales = soa + desc.scale_offset;
  const std::uint8_t* ql = soa + desc.ql_offset;
  const std::uint8_t* qh = soa + desc.qh_offset;
  for (std::size_t block = 0; block < desc.n_blocks; ++block) {
    std::uint8_t* dst = gguf + block * kQ6KGgufBlockBytes;
    std::memcpy(dst, ql + block * kQ6KQlBytes, kQ6KQlBytes);
    std::memcpy(dst + kQ6KQlBytes, qh + block * kQ6KQhBytes, kQ6KQhBytes);
    std::memcpy(dst + 192, scales + block * kQ6KScaleBytes, kQ6KScaleBytes);
    std::memcpy(dst + 208, d + block * kQ6KDBytes, kQ6KDBytes);
  }
}

inline bool q6k_aligned_inverse_matches(
    const std::uint8_t* gguf, const std::uint8_t* reconstructed,
    const Q6KAlignedLayoutDesc& desc) noexcept {
  if (gguf == nullptr || reconstructed == nullptr ||
      !q6k_aligned_desc_valid(desc)) {
    return false;
  }
  return std::memcmp(gguf, reconstructed, desc.gguf_bytes) == 0;
}

}  // namespace qw38::cuda
