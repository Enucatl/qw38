#pragma once

// OPT-047 production Q8_0 mixer decode path pin. Host-only; do not include
// the DP4A device header from packed MMV TUs.

#include <cstddef>
#include <cstring>

namespace qw38::cuda {

constexpr char kLegalQ8DecodePathDirectBf16[] = "direct_bf16";
constexpr char kLegalQ8DecodePathDp4aQ81[] = "dp4a_q8_1";

// Production pin. Keep sitting may switch away from direct BF16; reject
// restores it. Warps 0 on a role means that role stays on direct BF16.
constexpr char kSelectedQ8DecodePath[] = "dp4a_q8_1";
constexpr unsigned int kSelectedQ8DecodeWarpsSkinny = 4;
constexpr unsigned int kSelectedQ8DecodeWarpsMedium = 4;
constexpr unsigned int kSelectedQ8DecodeWarpsWide = 4;

inline thread_local const char* g_q8_decode_path_override = nullptr;
inline thread_local unsigned int g_q8_decode_warps_skinny_override = 0;
inline thread_local unsigned int g_q8_decode_warps_medium_override = 0;
inline thread_local unsigned int g_q8_decode_warps_wide_override = 0;

inline bool legal_q8_decode_path(const char* path) noexcept {
  return path != nullptr &&
         (std::strcmp(path, kLegalQ8DecodePathDirectBf16) == 0 ||
          std::strcmp(path, kLegalQ8DecodePathDp4aQ81) == 0);
}

inline bool legal_q8_decode_warps_per_row(unsigned int warps) noexcept {
  return warps == 0 || warps == 1 || warps == 2 || warps == 4 || warps == 8;
}

inline const char* selected_q8_decode_path() noexcept {
  return kSelectedQ8DecodePath;
}

inline unsigned int selected_q8_decode_warps_skinny() noexcept {
  return kSelectedQ8DecodeWarpsSkinny;
}

inline unsigned int selected_q8_decode_warps_medium() noexcept {
  return kSelectedQ8DecodeWarpsMedium;
}

inline unsigned int selected_q8_decode_warps_wide() noexcept {
  return kSelectedQ8DecodeWarpsWide;
}

inline const char* effective_q8_decode_path() noexcept {
  return g_q8_decode_path_override != nullptr ? g_q8_decode_path_override
                                              : kSelectedQ8DecodePath;
}

inline unsigned int effective_q8_decode_warps_skinny() noexcept {
  return g_q8_decode_warps_skinny_override > 0
             ? g_q8_decode_warps_skinny_override
             : kSelectedQ8DecodeWarpsSkinny;
}

inline unsigned int effective_q8_decode_warps_medium() noexcept {
  return g_q8_decode_warps_medium_override > 0
             ? g_q8_decode_warps_medium_override
             : kSelectedQ8DecodeWarpsMedium;
}

inline unsigned int effective_q8_decode_warps_wide() noexcept {
  return g_q8_decode_warps_wide_override > 0 ? g_q8_decode_warps_wide_override
                                             : kSelectedQ8DecodeWarpsWide;
}

inline bool q8_decode_uses_dp4a() noexcept {
  return std::strcmp(effective_q8_decode_path(), kLegalQ8DecodePathDp4aQ81) ==
         0;
}

inline unsigned int q8_decode_warps_for_rows(std::size_t rows) noexcept {
  if (rows > 0 && rows < 128) return effective_q8_decode_warps_skinny();
  if (rows < 4096) return effective_q8_decode_warps_medium();
  return effective_q8_decode_warps_wide();
}

inline bool q8_decode_uses_dp4a_for_rows(std::size_t rows) noexcept {
  return q8_decode_uses_dp4a() && q8_decode_warps_for_rows(rows) > 0;
}

inline void set_q8_decode_path_override(const char* path,
                                        unsigned int warps_skinny,
                                        unsigned int warps_medium,
                                        unsigned int warps_wide) noexcept {
  g_q8_decode_path_override = path;
  g_q8_decode_warps_skinny_override = warps_skinny;
  g_q8_decode_warps_medium_override = warps_medium;
  g_q8_decode_warps_wide_override = warps_wide;
}

inline void clear_q8_decode_path_override() noexcept {
  g_q8_decode_path_override = nullptr;
  g_q8_decode_warps_skinny_override = 0;
  g_q8_decode_warps_medium_override = 0;
  g_q8_decode_warps_wide_override = 0;
}

struct Q8DecodePathScope final {
  Q8DecodePathScope(const char* path, unsigned int warps_skinny,
                    unsigned int warps_medium,
                    unsigned int warps_wide) noexcept {
    set_q8_decode_path_override(path, warps_skinny, warps_medium, warps_wide);
  }
  ~Q8DecodePathScope() { clear_q8_decode_path_override(); }
  Q8DecodePathScope(const Q8DecodePathScope&) = delete;
  Q8DecodePathScope& operator=(const Q8DecodePathScope&) = delete;
};

}  // namespace qw38::cuda
