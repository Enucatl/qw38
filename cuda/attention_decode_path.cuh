#pragma once

// OPT-078 one-token decode query-prep selector. Host-includable.
// Production pin stays in-kernel warp_query RMS+RoPE unless KEEP succeeds.

#include <cstddef>
#include <cstring>

namespace qw38::cuda {

constexpr char kLegalDecodeQueryPrepWarpQuery[] = "warp_query";
constexpr char kLegalDecodeQueryPrepPreparedQ[] = "prepared_q";
constexpr char kLegalDecodeQueryPrepPreparedQVecKv[] = "prepared_q_veckv";

constexpr char kDecodeQueryPrepLaunchWarpQuery[] =
    "warp_query_decode_attention";
constexpr char kDecodeQueryPrepLaunchPreparedQ[] =
    "prepare_decode_query+warp_query_prepared_q";
constexpr char kDecodeQueryPrepLaunchPreparedQVecKv[] =
    "prepare_decode_query+warp_query_prepared_q_veckv";
constexpr char kDecodeQueryPrepKernel[] = "prepare_decode_query";

// Production pin. Keep sitting may switch; reject restores warp_query.
constexpr char kSelectedDecodeQueryPrepPath[] = "warp_query";

inline thread_local const char* g_decode_query_prep_path_override = nullptr;
inline thread_local const char* g_last_decode_query_prep_launch_variant = "";
inline thread_local unsigned int g_last_decode_query_prep_grid = 0;
inline thread_local unsigned int g_last_decode_query_prep_block = 0;
inline thread_local unsigned int g_last_decode_attention_n_parts = 0;
inline thread_local unsigned int g_last_decode_attention_prep_launches = 0;
inline thread_local bool g_last_decode_query_prep_used = false;
inline thread_local bool g_last_decode_vec_kv_used = false;

inline bool legal_decode_query_prep_path(const char* path) noexcept {
  return path != nullptr &&
         (std::strcmp(path, kLegalDecodeQueryPrepWarpQuery) == 0 ||
          std::strcmp(path, kLegalDecodeQueryPrepPreparedQ) == 0 ||
          std::strcmp(path, kLegalDecodeQueryPrepPreparedQVecKv) == 0);
}

inline const char* selected_decode_query_prep_path() noexcept {
  return kSelectedDecodeQueryPrepPath;
}

inline const char* effective_decode_query_prep_path() noexcept {
  return g_decode_query_prep_path_override != nullptr
             ? g_decode_query_prep_path_override
             : kSelectedDecodeQueryPrepPath;
}

inline bool decode_query_prep_uses_prepared(const char* path) noexcept {
  return path != nullptr &&
         (std::strcmp(path, kLegalDecodeQueryPrepPreparedQ) == 0 ||
          std::strcmp(path, kLegalDecodeQueryPrepPreparedQVecKv) == 0);
}

inline bool decode_query_prep_uses_prepared() noexcept {
  return decode_query_prep_uses_prepared(effective_decode_query_prep_path());
}

inline bool decode_query_prep_uses_veckv(const char* path) noexcept {
  return path != nullptr &&
         std::strcmp(path, kLegalDecodeQueryPrepPreparedQVecKv) == 0;
}

inline bool decode_query_prep_uses_veckv() noexcept {
  return decode_query_prep_uses_veckv(effective_decode_query_prep_path());
}

inline const char* decode_query_prep_launch_variant(const char* path) noexcept {
  if (decode_query_prep_uses_veckv(path)) {
    return kDecodeQueryPrepLaunchPreparedQVecKv;
  }
  if (decode_query_prep_uses_prepared(path)) {
    return kDecodeQueryPrepLaunchPreparedQ;
  }
  return kDecodeQueryPrepLaunchWarpQuery;
}

inline void record_decode_query_prep_launch(const char* variant,
                                            unsigned int prep_grid,
                                            unsigned int prep_block,
                                            unsigned int n_parts,
                                            bool prepared, bool vec_kv,
                                            unsigned int prep_launches) noexcept {
  g_last_decode_query_prep_launch_variant = variant != nullptr ? variant : "";
  g_last_decode_query_prep_grid = prep_grid;
  g_last_decode_query_prep_block = prep_block;
  g_last_decode_attention_n_parts = n_parts;
  g_last_decode_query_prep_used = prepared;
  g_last_decode_vec_kv_used = vec_kv;
  g_last_decode_attention_prep_launches = prep_launches;
}

inline const char* last_decode_query_prep_launch_variant() noexcept {
  return g_last_decode_query_prep_launch_variant != nullptr
             ? g_last_decode_query_prep_launch_variant
             : "";
}

inline unsigned int last_decode_query_prep_grid() noexcept {
  return g_last_decode_query_prep_grid;
}

inline unsigned int last_decode_query_prep_block() noexcept {
  return g_last_decode_query_prep_block;
}

inline unsigned int last_decode_attention_n_parts() noexcept {
  return g_last_decode_attention_n_parts;
}

inline unsigned int last_decode_attention_prep_launches() noexcept {
  return g_last_decode_attention_prep_launches;
}

inline bool last_decode_query_prep_used() noexcept {
  return g_last_decode_query_prep_used;
}

inline bool last_decode_vec_kv_used() noexcept {
  return g_last_decode_vec_kv_used;
}

inline void set_decode_query_prep_path_override(const char* path) noexcept {
  g_decode_query_prep_path_override = path;
}

inline bool apply_decode_query_prep_ident(const char* path) noexcept {
  if (!legal_decode_query_prep_path(path)) return false;
  set_decode_query_prep_path_override(path);
  return true;
}

inline void clear_decode_query_prep_path_override() noexcept {
  g_decode_query_prep_path_override = nullptr;
}

struct DecodeQueryPrepPathScope final {
  explicit DecodeQueryPrepPathScope(const char* path) noexcept {
    set_decode_query_prep_path_override(path);
  }
  ~DecodeQueryPrepPathScope() { clear_decode_query_prep_path_override(); }
  DecodeQueryPrepPathScope(const DecodeQueryPrepPathScope&) = delete;
  DecodeQueryPrepPathScope& operator=(const DecodeQueryPrepPathScope&) = delete;
};

// OPT-095 decode attention GQA KV-sharing selector. Production pin stays
// warp_query unless acceptance KEEP succeeds.
constexpr char kLegalDecodeAttentionGqaWarpQuery[] = "warp_query";
constexpr char kLegalDecodeAttentionGqaWarpQueryGqa6[] = "warp_query_gqa6";

constexpr char kDecodeAttentionGqaLaunchWarpQuery[] =
    "warp_query_decode_attention";
constexpr char kDecodeAttentionGqaLaunchWarpQueryGqa6[] =
    "warp_query_gqa6_decode_attention";

constexpr char kSelectedDecodeAttentionGqaPath[] = "warp_query";

inline thread_local const char* g_decode_attention_gqa_path_override =
    nullptr;
inline thread_local unsigned int g_last_decode_attention_gqa_grid_x = 0;
inline thread_local unsigned int g_last_decode_attention_gqa_block_x = 0;
inline thread_local unsigned int g_last_decode_attention_gqa_block_y = 0;

inline bool legal_decode_attention_gqa_path(const char* path) noexcept {
  return path != nullptr &&
         (std::strcmp(path, kLegalDecodeAttentionGqaWarpQuery) == 0 ||
          std::strcmp(path, kLegalDecodeAttentionGqaWarpQueryGqa6) == 0);
}

inline const char* selected_decode_attention_gqa_path() noexcept {
  return kSelectedDecodeAttentionGqaPath;
}

inline const char* effective_decode_attention_gqa_path() noexcept {
  return g_decode_attention_gqa_path_override != nullptr
             ? g_decode_attention_gqa_path_override
             : kSelectedDecodeAttentionGqaPath;
}

inline bool decode_attention_gqa_uses_gqa6(const char* path) noexcept {
  return path != nullptr &&
         std::strcmp(path, kLegalDecodeAttentionGqaWarpQueryGqa6) == 0;
}

inline bool decode_attention_gqa_uses_gqa6() noexcept {
  return decode_attention_gqa_uses_gqa6(effective_decode_attention_gqa_path());
}

inline const char* decode_attention_gqa_launch_variant(const char* path) noexcept {
  if (decode_attention_gqa_uses_gqa6(path)) {
    return kDecodeAttentionGqaLaunchWarpQueryGqa6;
  }
  return kDecodeAttentionGqaLaunchWarpQuery;
}

inline void record_decode_attention_gqa_launch(const char* variant,
                                               unsigned int grid_x,
                                               unsigned int block_x,
                                               unsigned int block_y,
                                               unsigned int n_parts) noexcept {
  g_last_decode_query_prep_launch_variant = variant != nullptr ? variant : "";
  g_last_decode_attention_gqa_grid_x = grid_x;
  g_last_decode_attention_gqa_block_x = block_x;
  g_last_decode_attention_gqa_block_y = block_y;
  g_last_decode_query_prep_grid = grid_x;
  g_last_decode_query_prep_block = block_x;
  g_last_decode_attention_n_parts = n_parts;
  g_last_decode_attention_prep_launches = 0;
  g_last_decode_query_prep_used = false;
  g_last_decode_vec_kv_used = false;
}

inline unsigned int last_decode_attention_gqa_grid_x() noexcept {
  return g_last_decode_attention_gqa_grid_x;
}

inline unsigned int last_decode_attention_gqa_block_x() noexcept {
  return g_last_decode_attention_gqa_block_x;
}

inline unsigned int last_decode_attention_gqa_block_y() noexcept {
  return g_last_decode_attention_gqa_block_y;
}

inline void set_decode_attention_gqa_path_override(const char* path) noexcept {
  g_decode_attention_gqa_path_override = path;
}

inline bool apply_decode_attention_gqa_ident(const char* path) noexcept {
  if (!legal_decode_attention_gqa_path(path)) return false;
  set_decode_attention_gqa_path_override(path);
  return true;
}

inline void clear_decode_attention_gqa_path_override() noexcept {
  g_decode_attention_gqa_path_override = nullptr;
}

struct DecodeAttentionGqaPathScope final {
  explicit DecodeAttentionGqaPathScope(const char* path) noexcept {
    set_decode_attention_gqa_path_override(path);
  }
  ~DecodeAttentionGqaPathScope() { clear_decode_attention_gqa_path_override(); }
  DecodeAttentionGqaPathScope(const DecodeAttentionGqaPathScope&) = delete;
  DecodeAttentionGqaPathScope& operator=(
      const DecodeAttentionGqaPathScope&) = delete;
};

// OPT-103 one-query 128-thread online-softmax vector attention. Production pin
// stays warp_query unless acceptance KEEP succeeds. warp_query_gqa6 stays
// rejected. Partition count is chosen from {4,8,16} by the OPT-103 screen.
constexpr char kLegalDecodeAttentionVec128WarpQuery[] = "warp_query";
constexpr char kLegalDecodeAttentionVec128Online[] = "vec128_online";

constexpr char kDecodeAttentionVec128LaunchWarpQuery[] =
    "warp_query_decode_attention";
constexpr char kDecodeAttentionVec128LaunchOnline[] =
    "vec128_online_decode_attention";

constexpr char kSelectedDecodeAttentionVec128Path[] = "warp_query";
constexpr int kSelectedVec128NParts = 16;
constexpr int kVec128Threads = 128;
constexpr int kVec128Warps = 4;

// OPT-107 prefix-aware hybrid. Threshold 0 disables vec128_online everywhere
// and is byte-for-byte the OPT-103 warp_query production path. A positive
// threshold admits vec128_online only on [threshold, verified_max]; shorter
// prefixes and anything past the measured range stay warp_query.
constexpr int kSelectedDecodeAttentionCrossoverThreshold = 1024;
constexpr int kSelectedDecodeAttentionVerifiedMax = 4096;
constexpr int kDecodeAttentionFallback128K = 131072;

inline thread_local const char* g_decode_attention_vec128_path_override =
    nullptr;
inline thread_local int g_vec128_n_parts_override = 0;
inline thread_local int g_decode_attention_crossover_threshold_override = -1;

inline bool legal_vec128_n_parts(int n_parts) noexcept {
  return n_parts == 4 || n_parts == 8 || n_parts == 16;
}

inline bool legal_decode_attention_vec128_path(const char* path) noexcept {
  return path != nullptr &&
         (std::strcmp(path, kLegalDecodeAttentionVec128WarpQuery) == 0 ||
          std::strcmp(path, kLegalDecodeAttentionVec128Online) == 0);
}

inline bool legal_decode_attention_crossover_threshold(int threshold) noexcept {
  return threshold == 0 || threshold == 512 || threshold == 1024 ||
         threshold == 1536 || threshold == 2048;
}

inline const char* selected_decode_attention_vec128_path() noexcept {
  return kSelectedDecodeAttentionVec128Path;
}

inline int selected_decode_attention_crossover_threshold() noexcept {
  return kSelectedDecodeAttentionCrossoverThreshold;
}

inline int selected_decode_attention_verified_max() noexcept {
  return kSelectedDecodeAttentionVerifiedMax;
}

inline int effective_decode_attention_crossover_threshold() noexcept {
  if (g_decode_attention_crossover_threshold_override >= 0 &&
      legal_decode_attention_crossover_threshold(
          g_decode_attention_crossover_threshold_override)) {
    return g_decode_attention_crossover_threshold_override;
  }
  return kSelectedDecodeAttentionCrossoverThreshold;
}

inline const char* effective_decode_attention_vec128_path() noexcept {
  return g_decode_attention_vec128_path_override != nullptr
             ? g_decode_attention_vec128_path_override
             : kSelectedDecodeAttentionVec128Path;
}

inline const char* decode_attention_vec128_path_for_position(
    std::size_t position) noexcept {
  if (g_decode_attention_vec128_path_override != nullptr) {
    return g_decode_attention_vec128_path_override;
  }
  const int threshold = effective_decode_attention_crossover_threshold();
  if (threshold > 0 &&
      position >= static_cast<std::size_t>(threshold) &&
      position <= static_cast<std::size_t>(kSelectedDecodeAttentionVerifiedMax)) {
    return kLegalDecodeAttentionVec128Online;
  }
  return kSelectedDecodeAttentionVec128Path;
}

inline bool decode_attention_vec128_uses_online(const char* path) noexcept {
  return path != nullptr &&
         std::strcmp(path, kLegalDecodeAttentionVec128Online) == 0;
}

inline bool decode_attention_vec128_uses_online() noexcept {
  return decode_attention_vec128_uses_online(
      effective_decode_attention_vec128_path());
}

inline bool decode_attention_vec128_uses_online_at(
    std::size_t position) noexcept {
  return decode_attention_vec128_uses_online(
      decode_attention_vec128_path_for_position(position));
}

inline int selected_vec128_n_parts() noexcept { return kSelectedVec128NParts; }

inline int effective_vec128_n_parts() noexcept {
  if (legal_vec128_n_parts(g_vec128_n_parts_override)) {
    return g_vec128_n_parts_override;
  }
  return kSelectedVec128NParts;
}

inline const char* decode_attention_vec128_launch_variant(
    const char* path) noexcept {
  if (decode_attention_vec128_uses_online(path)) {
    return kDecodeAttentionVec128LaunchOnline;
  }
  return kDecodeAttentionVec128LaunchWarpQuery;
}

inline void set_decode_attention_vec128_path_override(const char* path) noexcept {
  g_decode_attention_vec128_path_override = path;
}

inline void set_vec128_n_parts_override(int n_parts) noexcept {
  g_vec128_n_parts_override = n_parts;
}

inline bool apply_decode_attention_vec128_ident(const char* path) noexcept {
  if (!legal_decode_attention_vec128_path(path)) return false;
  set_decode_attention_vec128_path_override(path);
  return true;
}

inline bool apply_vec128_n_parts(int n_parts) noexcept {
  if (!legal_vec128_n_parts(n_parts)) return false;
  set_vec128_n_parts_override(n_parts);
  return true;
}

inline void clear_decode_attention_vec128_path_override() noexcept {
  g_decode_attention_vec128_path_override = nullptr;
}

inline void clear_vec128_n_parts_override() noexcept {
  g_vec128_n_parts_override = 0;
}

inline bool apply_decode_attention_crossover_threshold(int threshold) noexcept {
  if (!legal_decode_attention_crossover_threshold(threshold)) return false;
  g_decode_attention_crossover_threshold_override = threshold;
  return true;
}

inline void clear_decode_attention_crossover_threshold_override() noexcept {
  g_decode_attention_crossover_threshold_override = -1;
}

inline const char* effective_decode_attention_dispatch_path() noexcept {
  if (decode_attention_vec128_uses_online()) {
    return effective_decode_attention_vec128_path();
  }
  return effective_decode_attention_gqa_path();
}

inline const char* effective_decode_attention_dispatch_path(
    std::size_t position) noexcept {
  if (decode_attention_vec128_uses_online_at(position)) {
    return decode_attention_vec128_path_for_position(position);
  }
  return effective_decode_attention_gqa_path();
}

struct DecodeAttentionVec128PathScope final {
  explicit DecodeAttentionVec128PathScope(const char* path) noexcept {
    set_decode_attention_vec128_path_override(path);
  }
  ~DecodeAttentionVec128PathScope() {
    clear_decode_attention_vec128_path_override();
  }
  DecodeAttentionVec128PathScope(const DecodeAttentionVec128PathScope&) = delete;
  DecodeAttentionVec128PathScope& operator=(
      const DecodeAttentionVec128PathScope&) = delete;
};

struct Vec128NPartsScope final {
  explicit Vec128NPartsScope(int n_parts) noexcept {
    set_vec128_n_parts_override(n_parts);
  }
  ~Vec128NPartsScope() { clear_vec128_n_parts_override(); }
  Vec128NPartsScope(const Vec128NPartsScope&) = delete;
  Vec128NPartsScope& operator=(const Vec128NPartsScope&) = delete;
};

struct DecodeAttentionCrossoverScope final {
  explicit DecodeAttentionCrossoverScope(int threshold) noexcept {
    apply_decode_attention_crossover_threshold(threshold);
  }
  ~DecodeAttentionCrossoverScope() {
    clear_decode_attention_crossover_threshold_override();
  }
  DecodeAttentionCrossoverScope(const DecodeAttentionCrossoverScope&) = delete;
  DecodeAttentionCrossoverScope& operator=(
      const DecodeAttentionCrossoverScope&) = delete;
};

}  // namespace qw38::cuda
