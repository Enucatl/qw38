#pragma once

// OPT-102 lossless 64-byte-aligned Q4_K metadata/code-plane device layout.
// Superblock d/dmin (4 bytes) plane, packed 12-byte 6-bit scale/min plane,
// then 128-byte qs plane. Inverse reconstruction is byte-exact with the
// public GGUF Q4_K payload. No public GGUF rewrite. OPT-093 factored
// association is not used.

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define QW38_Q4_HD __host__ __device__
#else
#define QW38_Q4_HD
#endif

namespace qw38::cuda {

constexpr std::uint32_t kQ4KAlignedLayoutVersion = 1;
constexpr std::size_t kQ4KGgufBlockBytes = 144;
constexpr std::size_t kQ4KGgufBlockValues = 256;
constexpr std::size_t kQ4KAlignedAlignBytes = 64;
constexpr std::size_t kQ4KDmBytes = 4;
constexpr std::size_t kQ4KScaleBytes = 12;
constexpr std::size_t kQ4KQsBytes = 128;
constexpr char kLegalQ4DeviceLayoutRawGguf[] = "raw_gguf";
constexpr char kLegalQ4DeviceLayoutAlignedMeta[] = "aligned_meta";

struct Q4KAlignedLayoutDesc final {
  std::uint32_t version = kQ4KAlignedLayoutVersion;
  std::uint32_t reserved = 0;
  std::size_t rows = 0;
  std::size_t columns = 0;
  std::size_t n_blocks_per_row = 0;
  std::size_t n_blocks = 0;
  std::size_t dm_bytes = 0;
  std::size_t dm_pad_bytes = 0;
  std::size_t scale_offset = 0;
  std::size_t scale_bytes = 0;
  std::size_t scale_pad_bytes = 0;
  std::size_t qs_offset = 0;
  std::size_t qs_bytes = 0;
  std::size_t total_bytes = 0;
  std::size_t gguf_bytes = 0;
};

QW38_Q4_HD inline bool q4k_aligned_geometry_ok(std::size_t rows,
                                               std::size_t columns) noexcept {
  return rows > 0 && columns > 0 && columns % kQ4KGgufBlockValues == 0;
}

QW38_Q4_HD inline std::size_t q4k_aligned_pad_bytes(std::size_t bytes) noexcept {
  return (kQ4KAlignedAlignBytes - (bytes % kQ4KAlignedAlignBytes)) %
         kQ4KAlignedAlignBytes;
}

QW38_Q4_HD inline Q4KAlignedLayoutDesc make_q4k_aligned_layout_desc(
    std::size_t rows, std::size_t columns) noexcept {
  Q4KAlignedLayoutDesc desc{};
  desc.version = kQ4KAlignedLayoutVersion;
  if (!q4k_aligned_geometry_ok(rows, columns)) return desc;
  desc.rows = rows;
  desc.columns = columns;
  desc.n_blocks_per_row = columns / kQ4KGgufBlockValues;
  desc.n_blocks = rows * desc.n_blocks_per_row;
  desc.dm_bytes = desc.n_blocks * kQ4KDmBytes;
  desc.dm_pad_bytes = q4k_aligned_pad_bytes(desc.dm_bytes);
  desc.scale_offset = desc.dm_bytes + desc.dm_pad_bytes;
  desc.scale_bytes = desc.n_blocks * kQ4KScaleBytes;
  desc.scale_pad_bytes = q4k_aligned_pad_bytes(desc.scale_bytes);
  desc.qs_offset = desc.scale_offset + desc.scale_bytes + desc.scale_pad_bytes;
  desc.qs_bytes = desc.n_blocks * kQ4KQsBytes;
  desc.total_bytes = desc.qs_offset + desc.qs_bytes;
  desc.gguf_bytes = desc.n_blocks * kQ4KGgufBlockBytes;
  return desc;
}

QW38_Q4_HD inline bool q4k_aligned_desc_valid(
    const Q4KAlignedLayoutDesc& desc) noexcept {
  if (desc.version != kQ4KAlignedLayoutVersion || desc.rows == 0 ||
      desc.columns == 0 || desc.columns % kQ4KGgufBlockValues != 0) {
    return false;
  }
  const Q4KAlignedLayoutDesc expected =
      make_q4k_aligned_layout_desc(desc.rows, desc.columns);
  return desc.n_blocks_per_row == expected.n_blocks_per_row &&
         desc.n_blocks == expected.n_blocks &&
         desc.dm_bytes == expected.dm_bytes &&
         desc.dm_pad_bytes == expected.dm_pad_bytes &&
         desc.scale_offset == expected.scale_offset &&
         desc.scale_bytes == expected.scale_bytes &&
         desc.scale_pad_bytes == expected.scale_pad_bytes &&
         desc.qs_offset == expected.qs_offset &&
         desc.qs_bytes == expected.qs_bytes &&
         desc.total_bytes == expected.total_bytes &&
         desc.gguf_bytes == expected.gguf_bytes;
}

QW38_Q4_HD inline bool q4k_aligned_is_byte_neutral(
    const Q4KAlignedLayoutDesc& desc) noexcept {
  return q4k_aligned_desc_valid(desc) && desc.dm_pad_bytes == 0 &&
         desc.scale_pad_bytes == 0 && desc.total_bytes == desc.gguf_bytes;
}

// Branchless 6-bit scale/min unpack. Technique from llama.cpp commit
// 73ab7599b (ggml-cuda Q4_K scale unpack). Index with j=index&3 so upper
// groups never read q[j-4]. Select is arithmetic; no control-flow on index.
QW38_Q4_HD inline void q4k_scale_min_branchless(const std::uint8_t* packed,
                                                int index, int* scale,
                                                int* minimum) noexcept {
  const int upper = index >> 2;
  const int j = index & 3;
  const int d_lo = packed[j] & 63;
  const int m_lo = packed[j + 4] & 63;
  const int d_hi = (packed[j + 8] & 15) | ((packed[j] >> 6) << 4);
  const int m_hi = (packed[j + 8] >> 4) | ((packed[j + 4] >> 6) << 4);
  const int mask = -upper;
  *scale = (d_lo & ~mask) | (d_hi & mask);
  *minimum = (m_lo & ~mask) | (m_hi & mask);
}

inline void pack_q4k_aligned_host(const std::uint8_t* gguf,
                                  const Q4KAlignedLayoutDesc& desc,
                                  std::uint8_t* soa) noexcept {
  if (gguf == nullptr || soa == nullptr || !q4k_aligned_desc_valid(desc)) return;
  std::memset(soa, 0, desc.total_bytes);
  std::uint8_t* dm = soa;
  std::uint8_t* scales = soa + desc.scale_offset;
  std::uint8_t* qs = soa + desc.qs_offset;
  for (std::size_t block = 0; block < desc.n_blocks; ++block) {
    const std::uint8_t* src = gguf + block * kQ4KGgufBlockBytes;
    std::memcpy(dm + block * kQ4KDmBytes, src, kQ4KDmBytes);
    std::memcpy(scales + block * kQ4KScaleBytes, src + kQ4KDmBytes,
                kQ4KScaleBytes);
    std::memcpy(qs + block * kQ4KQsBytes, src + 16, kQ4KQsBytes);
  }
}

inline void unpack_q4k_aligned_host(const std::uint8_t* soa,
                                    const Q4KAlignedLayoutDesc& desc,
                                    std::uint8_t* gguf) noexcept {
  if (gguf == nullptr || soa == nullptr || !q4k_aligned_desc_valid(desc)) return;
  const std::uint8_t* dm = soa;
  const std::uint8_t* scales = soa + desc.scale_offset;
  const std::uint8_t* qs = soa + desc.qs_offset;
  for (std::size_t block = 0; block < desc.n_blocks; ++block) {
    std::uint8_t* dst = gguf + block * kQ4KGgufBlockBytes;
    std::memcpy(dst, dm + block * kQ4KDmBytes, kQ4KDmBytes);
    std::memcpy(dst + kQ4KDmBytes, scales + block * kQ4KScaleBytes,
                kQ4KScaleBytes);
    std::memcpy(dst + 16, qs + block * kQ4KQsBytes, kQ4KQsBytes);
  }
}

inline bool q4k_aligned_inverse_matches(
    const std::uint8_t* gguf, const std::uint8_t* reconstructed,
    const Q4KAlignedLayoutDesc& desc) noexcept {
  if (gguf == nullptr || reconstructed == nullptr ||
      !q4k_aligned_desc_valid(desc)) {
    return false;
  }
  return std::memcmp(gguf, reconstructed, desc.gguf_bytes) == 0;
}

}  // namespace qw38::cuda
