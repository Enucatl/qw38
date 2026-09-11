#pragma once

// OPT-077 one-token decode GDN recurrence selector. Host-includable.

#include <cstring>

namespace qw38::cuda {

constexpr char kLegalGdnDecodePathSequential[] = "sequential";
constexpr char kLegalGdnDecodePathTile16[] = "tile16";
constexpr char kLegalGdnDecodePathTile32[] = "tile32";

constexpr char kGdnDecodeLaunchVariantSequential[] =
    "prepare_recurrence_window";
constexpr char kGdnDecodeLaunchVariantTile16[] =
    "prepare_recurrence_decode_tiled16";
constexpr char kGdnDecodeLaunchVariantTile32[] =
    "prepare_recurrence_decode_tiled32";

// Production pin. Keep sitting may switch; reject restores sequential.
constexpr char kSelectedGdnDecodePath[] = "sequential";
constexpr unsigned int kGdnDecodeWarpsPerCta = 4;
constexpr unsigned int kGdnDecodeThreads = 128;
constexpr unsigned int kGdnDecodeValueTile16 = 16;
constexpr unsigned int kGdnDecodeValueTile32 = 32;

inline thread_local const char* g_gdn_decode_path_override = nullptr;
inline thread_local const char* g_last_gdn_decode_launch_variant = "";
inline thread_local unsigned int g_last_gdn_decode_value_tile = 0;
inline thread_local unsigned int g_last_gdn_decode_grid_x = 0;
inline thread_local unsigned int g_last_gdn_decode_grid_y = 0;

inline bool legal_gdn_decode_path(const char* path) noexcept {
  return path != nullptr &&
         (std::strcmp(path, kLegalGdnDecodePathSequential) == 0 ||
          std::strcmp(path, kLegalGdnDecodePathTile16) == 0 ||
          std::strcmp(path, kLegalGdnDecodePathTile32) == 0);
}

inline const char* selected_gdn_decode_path() noexcept {
  return kSelectedGdnDecodePath;
}

inline const char* effective_gdn_decode_path() noexcept {
  return g_gdn_decode_path_override != nullptr ? g_gdn_decode_path_override
                                               : kSelectedGdnDecodePath;
}

inline bool gdn_decode_uses_tiled() noexcept {
  const char* path = effective_gdn_decode_path();
  return std::strcmp(path, kLegalGdnDecodePathTile16) == 0 ||
         std::strcmp(path, kLegalGdnDecodePathTile32) == 0;
}

inline unsigned int gdn_decode_value_tile() noexcept {
  const char* path = effective_gdn_decode_path();
  if (std::strcmp(path, kLegalGdnDecodePathTile32) == 0) {
    return kGdnDecodeValueTile32;
  }
  if (std::strcmp(path, kLegalGdnDecodePathTile16) == 0) {
    return kGdnDecodeValueTile16;
  }
  return 0;
}

inline void record_gdn_decode_launch(const char* variant, unsigned int tile,
                                     unsigned int grid_x,
                                     unsigned int grid_y) noexcept {
  g_last_gdn_decode_launch_variant = variant != nullptr ? variant : "";
  g_last_gdn_decode_value_tile = tile;
  g_last_gdn_decode_grid_x = grid_x;
  g_last_gdn_decode_grid_y = grid_y;
}

inline const char* last_gdn_decode_launch_variant() noexcept {
  return g_last_gdn_decode_launch_variant != nullptr
             ? g_last_gdn_decode_launch_variant
             : "";
}

inline unsigned int last_gdn_decode_value_tile() noexcept {
  return g_last_gdn_decode_value_tile;
}

inline unsigned int last_gdn_decode_grid_x() noexcept {
  return g_last_gdn_decode_grid_x;
}

inline unsigned int last_gdn_decode_grid_y() noexcept {
  return g_last_gdn_decode_grid_y;
}

inline void set_gdn_decode_path_override(const char* path) noexcept {
  g_gdn_decode_path_override = path;
}

inline bool apply_gdn_decode_ident(const char* path) noexcept {
  if (!legal_gdn_decode_path(path)) return false;
  set_gdn_decode_path_override(path);
  return true;
}

inline void clear_gdn_decode_path_override() noexcept {
  g_gdn_decode_path_override = nullptr;
}

struct GdnDecodePathScope final {
  explicit GdnDecodePathScope(const char* path) noexcept {
    set_gdn_decode_path_override(path);
  }
  ~GdnDecodePathScope() { clear_gdn_decode_path_override(); }
  GdnDecodePathScope(const GdnDecodePathScope&) = delete;
  GdnDecodePathScope& operator=(const GdnDecodePathScope&) = delete;
};

}  // namespace qw38::cuda
