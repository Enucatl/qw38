#pragma once

// OPT-078 one-token decode query-prep selector. Host-includable.
// Production pin stays in-kernel warp_query RMS+RoPE unless KEEP succeeds.

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

}  // namespace qw38::cuda
