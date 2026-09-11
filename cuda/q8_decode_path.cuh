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
// OPT-047 historical warps-per-row stay 4/4/4. OPT-064 row grouping uses
// kSelectedQ8DecodeRows* plus kSelectedQ8DecodeLayoutWarps* so a later
// layout keep does not rewrite the OPT-047 warp constants.
constexpr char kSelectedQ8DecodePath[] = "dp4a_q8_1";
constexpr unsigned int kSelectedQ8DecodeWarpsSkinny = 4;
constexpr unsigned int kSelectedQ8DecodeWarpsMedium = 4;
constexpr unsigned int kSelectedQ8DecodeWarpsWide = 4;
constexpr unsigned int kSelectedQ8DecodeRowsSkinny = 1;
constexpr unsigned int kSelectedQ8DecodeRowsMedium = 1;
constexpr unsigned int kSelectedQ8DecodeRowsWide = 1;
constexpr unsigned int kSelectedQ8DecodeLayoutWarpsSkinny = 4;
constexpr unsigned int kSelectedQ8DecodeLayoutWarpsMedium = 4;
constexpr unsigned int kSelectedQ8DecodeLayoutWarpsWide = 4;

struct Q8DecodeLayout final {
  unsigned int rows_per_cta = 1;
  unsigned int warps_per_row = 4;
};

inline thread_local const char* g_q8_decode_path_override = nullptr;
inline thread_local unsigned int g_q8_decode_warps_skinny_override = 0;
inline thread_local unsigned int g_q8_decode_warps_medium_override = 0;
inline thread_local unsigned int g_q8_decode_warps_wide_override = 0;
inline thread_local unsigned int g_q8_decode_rows_skinny_override = 0;
inline thread_local unsigned int g_q8_decode_rows_medium_override = 0;
inline thread_local unsigned int g_q8_decode_rows_wide_override = 0;
inline thread_local unsigned int g_q8_decode_layout_warps_skinny_override = 0;
inline thread_local unsigned int g_q8_decode_layout_warps_medium_override = 0;
inline thread_local unsigned int g_q8_decode_layout_warps_wide_override = 0;

inline bool legal_q8_decode_path(const char* path) noexcept {
  return path != nullptr &&
         (std::strcmp(path, kLegalQ8DecodePathDirectBf16) == 0 ||
          std::strcmp(path, kLegalQ8DecodePathDp4aQ81) == 0);
}

inline bool legal_q8_decode_warps_per_row(unsigned int warps) noexcept {
  return warps == 0 || warps == 1 || warps == 2 || warps == 4 || warps == 8;
}

inline bool legal_q8_decode_rows_per_cta(unsigned int rows) noexcept {
  return rows == 1 || rows == 2 || rows == 4 || rows == 8;
}

inline bool legal_q8_decode_layout(unsigned int rows_per_cta,
                                   unsigned int warps_per_row) noexcept {
  if (!legal_q8_decode_rows_per_cta(rows_per_cta) ||
      !legal_q8_decode_warps_per_row(warps_per_row) || warps_per_row == 0) {
    return false;
  }
  // Instantiated study + OPT-047 one-row occupancy/launch set.
  if (rows_per_cta == 1) return true;
  if (rows_per_cta == 2 && warps_per_row == 2) return true;
  if (rows_per_cta == 4 && warps_per_row == 1) return true;
  if (rows_per_cta == 8 && warps_per_row == 1) return true;
  return false;
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

inline unsigned int selected_q8_decode_rows_skinny() noexcept {
  return kSelectedQ8DecodeRowsSkinny;
}

inline unsigned int selected_q8_decode_rows_medium() noexcept {
  return kSelectedQ8DecodeRowsMedium;
}

inline unsigned int selected_q8_decode_rows_wide() noexcept {
  return kSelectedQ8DecodeRowsWide;
}

inline unsigned int selected_q8_decode_layout_warps_skinny() noexcept {
  return kSelectedQ8DecodeLayoutWarpsSkinny;
}

inline unsigned int selected_q8_decode_layout_warps_medium() noexcept {
  return kSelectedQ8DecodeLayoutWarpsMedium;
}

inline unsigned int selected_q8_decode_layout_warps_wide() noexcept {
  return kSelectedQ8DecodeLayoutWarpsWide;
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

inline unsigned int effective_q8_decode_rows_skinny() noexcept {
  return g_q8_decode_rows_skinny_override > 0
             ? g_q8_decode_rows_skinny_override
             : kSelectedQ8DecodeRowsSkinny;
}

inline unsigned int effective_q8_decode_rows_medium() noexcept {
  return g_q8_decode_rows_medium_override > 0
             ? g_q8_decode_rows_medium_override
             : kSelectedQ8DecodeRowsMedium;
}

inline unsigned int effective_q8_decode_rows_wide() noexcept {
  return g_q8_decode_rows_wide_override > 0 ? g_q8_decode_rows_wide_override
                                            : kSelectedQ8DecodeRowsWide;
}

inline unsigned int effective_q8_decode_layout_warps_skinny() noexcept {
  return g_q8_decode_layout_warps_skinny_override > 0
             ? g_q8_decode_layout_warps_skinny_override
             : kSelectedQ8DecodeLayoutWarpsSkinny;
}

inline unsigned int effective_q8_decode_layout_warps_medium() noexcept {
  return g_q8_decode_layout_warps_medium_override > 0
             ? g_q8_decode_layout_warps_medium_override
             : kSelectedQ8DecodeLayoutWarpsMedium;
}

inline unsigned int effective_q8_decode_layout_warps_wide() noexcept {
  return g_q8_decode_layout_warps_wide_override > 0
             ? g_q8_decode_layout_warps_wide_override
             : kSelectedQ8DecodeLayoutWarpsWide;
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

inline Q8DecodeLayout q8_decode_layout_for_rows(std::size_t rows) noexcept {
  Q8DecodeLayout layout;
  if (rows > 0 && rows < 128) {
    layout.rows_per_cta = effective_q8_decode_rows_skinny();
    layout.warps_per_row = effective_q8_decode_layout_warps_skinny();
  } else if (rows < 4096) {
    layout.rows_per_cta = effective_q8_decode_rows_medium();
    layout.warps_per_row = effective_q8_decode_layout_warps_medium();
  } else {
    layout.rows_per_cta = effective_q8_decode_rows_wide();
    layout.warps_per_row = effective_q8_decode_layout_warps_wide();
  }
  return layout;
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

inline void set_q8_decode_layout_override(
    const char* path, unsigned int rows_skinny, unsigned int warps_skinny,
    unsigned int rows_medium, unsigned int warps_medium,
    unsigned int rows_wide, unsigned int warps_wide) noexcept {
  set_q8_decode_path_override(path, warps_skinny, warps_medium, warps_wide);
  g_q8_decode_rows_skinny_override = rows_skinny;
  g_q8_decode_rows_medium_override = rows_medium;
  g_q8_decode_rows_wide_override = rows_wide;
  g_q8_decode_layout_warps_skinny_override = warps_skinny;
  g_q8_decode_layout_warps_medium_override = warps_medium;
  g_q8_decode_layout_warps_wide_override = warps_wide;
}

inline void clear_q8_decode_path_override() noexcept {
  g_q8_decode_path_override = nullptr;
  g_q8_decode_warps_skinny_override = 0;
  g_q8_decode_warps_medium_override = 0;
  g_q8_decode_warps_wide_override = 0;
  g_q8_decode_rows_skinny_override = 0;
  g_q8_decode_rows_medium_override = 0;
  g_q8_decode_rows_wide_override = 0;
  g_q8_decode_layout_warps_skinny_override = 0;
  g_q8_decode_layout_warps_medium_override = 0;
  g_q8_decode_layout_warps_wide_override = 0;
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

struct Q8DecodeLayoutScope final {
  Q8DecodeLayoutScope(const char* path, unsigned int rows_skinny,
                      unsigned int warps_skinny, unsigned int rows_medium,
                      unsigned int warps_medium, unsigned int rows_wide,
                      unsigned int warps_wide) noexcept {
    set_q8_decode_layout_override(path, rows_skinny, warps_skinny, rows_medium,
                                  warps_medium, rows_wide, warps_wide);
  }
  ~Q8DecodeLayoutScope() { clear_q8_decode_path_override(); }
  Q8DecodeLayoutScope(const Q8DecodeLayoutScope&) = delete;
  Q8DecodeLayoutScope& operator=(const Q8DecodeLayoutScope&) = delete;
};

struct Q8DecodeDispatch final {
  unsigned int rows_per_cta = 0;
  unsigned int warps_per_row = 0;
  std::size_t rows = 0;
  std::size_t columns = 0;
  const char* layout = "";
  bool graph_capture = false;
};

inline thread_local Q8DecodeDispatch g_last_q8_decode_dispatch{};

inline const char* q8_layout_ident(unsigned int rows_per_cta,
                                   unsigned int warps_per_row) noexcept {
  if (rows_per_cta == 1 && warps_per_row == 4) return "r1_w4";
  if (rows_per_cta == 2 && warps_per_row == 2) return "r2_w2";
  if (rows_per_cta == 4 && warps_per_row == 1) return "r4_w1";
  if (rows_per_cta == 8 && warps_per_row == 1) return "r8_w1";
  return "unknown";
}

inline bool apply_q8_layout_ident(const char* ident) noexcept {
  if (ident == nullptr) return false;
  if (std::strcmp(ident, "r1_w4") == 0) {
    set_q8_decode_layout_override(kSelectedQ8DecodePath, 1, 4, 1, 4, 1, 4);
    return true;
  }
  if (std::strcmp(ident, "r2_w2") == 0) {
    set_q8_decode_layout_override(kSelectedQ8DecodePath, 2, 2, 2, 2, 2, 2);
    return true;
  }
  if (std::strcmp(ident, "r4_w1") == 0) {
    set_q8_decode_layout_override(kSelectedQ8DecodePath, 4, 1, 4, 1, 4, 1);
    return true;
  }
  if (std::strcmp(ident, "r8_w1") == 0) {
    set_q8_decode_layout_override(kSelectedQ8DecodePath, 8, 1, 8, 1, 8, 1);
    return true;
  }
  return false;
}

inline void record_q8_decode_dispatch(std::size_t rows, std::size_t columns,
                                      unsigned int rows_per_cta,
                                      unsigned int warps_per_row) noexcept {
  g_last_q8_decode_dispatch.rows = rows;
  g_last_q8_decode_dispatch.columns = columns;
  g_last_q8_decode_dispatch.rows_per_cta = rows_per_cta;
  g_last_q8_decode_dispatch.warps_per_row = warps_per_row;
  g_last_q8_decode_dispatch.layout =
      q8_layout_ident(rows_per_cta, warps_per_row);
}

inline const Q8DecodeDispatch& last_q8_decode_dispatch() noexcept {
  return g_last_q8_decode_dispatch;
}

inline void clear_q8_decode_dispatch() noexcept {
  g_last_q8_decode_dispatch = {};
}

}  // namespace qw38::cuda
