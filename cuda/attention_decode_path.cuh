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
// OPT-130 occupancy freeze (llama vec n_parts~9 at D2048 → legal 8) and
// vec128_open past verified_max. Production pins stay 16 / 4096 until KEEP.
constexpr int kOpt130CandidateVec128NParts = 8;
constexpr int kOpt130CandidateVerifiedMax = 131072;
constexpr char kLegalDecodeAttentionOccupancyVec128Open[] =
    "occupancy_vec128_open";
constexpr char kLegalDecodeAttentionHybridCrossover[] = "hybrid_crossover";
constexpr char kLegalDecodeAttentionDenseMma[] =
    "dense_bf16_tile_f16_mma_decode_v1";
constexpr char kDecodeAttentionMmaLaunch[] =
    "dense_bf16_tile_f16_mma_decode_v1";
constexpr int kOpt137MmaThreshold = 8192;
constexpr int kOpt137MmaNcols1 = 1;
constexpr int kOpt137MmaNcols2 = 8;
constexpr int kOpt137MmaGqaRatio = 6;
constexpr int kOpt137MmaNbatchFa = 64;
constexpr int kOpt137MmaNthreads = 64;
constexpr int kOpt137MmaOccupancyPinned = 4;
constexpr int kOpt137MmaNstages = 2;
constexpr int kOpt137TopologyCount = 5;
constexpr int kOpt137Bucket32768 = 32768;
constexpr int kOpt137Bucket65536 = 65536;
constexpr std::size_t kOpt137RepVisible8448 = 8448;
constexpr std::size_t kOpt137RepVisible33024 = 33024;
constexpr std::size_t kOpt137RepVisible131072 = 131072;
constexpr bool kSelectedOpt137DenseMma = true;
// OPT-151 QK+PV MMA. Production stays OPT-137 scalar-QK/PV MMA until KEEP.
constexpr char kLegalDecodeAttentionQkPvMmaV2[] =
    "decode_attention_qk_pv_mma_v2";
constexpr char kDecodeAttentionQkPvMmaV2Launch[] =
    "decode_attention_qk_pv_mma_v2";
constexpr int kOpt151MmaOccupancyPinned = 4;
constexpr int kOpt151MmaNthreads = 64;
constexpr bool kSelectedOpt151QkPvMma = false;
// OPT-148 flash-style vector decode. Production stays vec128_online in the
// hybrid crossover window until KEEP flips this pin. D128 warp_query and
// OPT-137 MMA at >=8192 are unchanged. Do not revive OPT-130 n_parts=8.
constexpr char kLegalDecodeAttentionFlashVec[] =
    "decode_attention_flash_vec_v1";
constexpr char kDecodeAttentionFlashVecLaunch[] =
    "flash_vec_decode_attention";
constexpr bool kSelectedDecodeAttentionFlashVec = true;

inline thread_local bool g_opt137_mma_override = false;
inline thread_local bool g_opt151_qk_pv_mma_override = false;
inline thread_local unsigned int g_opt151_mma_launches = 0;
inline thread_local unsigned int g_opt151_mma_grid_x = 0;
inline thread_local unsigned int g_opt151_mma_grid_y = 0;
inline thread_local unsigned int g_opt151_mma_block = 0;
inline thread_local int g_opt151_last_n_parts = 0;
inline thread_local bool g_decode_attention_flash_vec_override = false;
inline thread_local int g_opt137_frozen_n_parts[3] = {0, 0, 0};
inline thread_local unsigned int g_opt137_mma_launches = 0;
inline thread_local unsigned int g_opt137_mma_grid_x = 0;
inline thread_local unsigned int g_opt137_mma_grid_y = 0;
inline thread_local unsigned int g_opt137_mma_block = 0;
inline thread_local int g_opt137_last_n_parts = 0;

inline thread_local const char* g_decode_attention_vec128_path_override =
    nullptr;
inline thread_local int g_vec128_n_parts_override = 0;
inline thread_local int g_decode_attention_crossover_threshold_override = -1;
inline thread_local int g_decode_attention_verified_max_override = -1;

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

inline bool legal_decode_attention_verified_max(int verified_max) noexcept {
  return verified_max == 4096 || verified_max == 8192 ||
         verified_max == 32768 || verified_max == 131072;
}

inline int effective_decode_attention_verified_max() noexcept {
  if (g_decode_attention_verified_max_override >= 0 &&
      legal_decode_attention_verified_max(
          g_decode_attention_verified_max_override)) {
    return g_decode_attention_verified_max_override;
  }
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

inline bool opt137_mma_enabled() noexcept {
  return g_opt137_mma_override || kSelectedOpt137DenseMma;
}

inline bool opt137_uses_mma_at(std::size_t position) noexcept {
  return opt137_mma_enabled() &&
         position >= static_cast<std::size_t>(kOpt137MmaThreshold);
}

inline bool opt151_qk_pv_mma_enabled() noexcept {
  return g_opt151_qk_pv_mma_override || kSelectedOpt151QkPvMma;
}

inline bool opt151_uses_qk_pv_mma_at(std::size_t position) noexcept {
  return opt151_qk_pv_mma_enabled() &&
         position >= static_cast<std::size_t>(kOpt137MmaThreshold);
}

inline bool decode_attention_flash_vec_enabled() noexcept {
  return g_decode_attention_flash_vec_override ||
         kSelectedDecodeAttentionFlashVec;
}

inline int opt137_bucket_index(std::size_t position) noexcept {
  if (position >= static_cast<std::size_t>(kOpt137Bucket65536)) return 2;
  if (position >= static_cast<std::size_t>(kOpt137Bucket32768)) return 1;
  return 0;
}

inline int opt137_parallel_blocks(std::size_t visible, int occupancy,
                                  int nsm) noexcept {
  const int ntiles_kv =
      static_cast<int>((visible + static_cast<std::size_t>(kOpt137MmaNbatchFa) -
                        1U) /
                       static_cast<std::size_t>(kOpt137MmaNbatchFa));
  const int ntiles_dst = 4;
  const int occ_use = occupancy > 0 ? occupancy : kOpt137MmaOccupancyPinned;
  const int sm = nsm > 0 ? nsm : 1;
  int parallel = occ_use < ntiles_kv ? occ_use : ntiles_kv;
  if (parallel < 1) parallel = 1;
  const int blocks_per_wave = sm * occ_use;
  int nwaves_best = 0;
  int efficiency_best = 0;
  for (int test = parallel; test <= ntiles_kv; ++test) {
    const int nblocks_total = ntiles_dst * test;
    const int nwaves =
        (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
    const int efficiency =
        blocks_per_wave > 0 ? (100 * nblocks_total) / (nwaves * blocks_per_wave)
                            : 0;
    if (efficiency_best >= 95 && nwaves > nwaves_best) break;
    if (efficiency > efficiency_best) {
      nwaves_best = nwaves;
      efficiency_best = efficiency;
      parallel = test;
    }
  }
  if (parallel > 256) parallel = 256;
  if (parallel < 1) parallel = 1;
  return parallel;
}

inline void opt137_set_frozen_n_parts(int p8448, int p33024,
                                      int p131072) noexcept {
  g_opt137_frozen_n_parts[0] = p8448;
  g_opt137_frozen_n_parts[1] = p33024;
  g_opt137_frozen_n_parts[2] = p131072;
}

inline int opt137_n_parts_for_position(std::size_t position) noexcept {
  const int bucket = opt137_bucket_index(position);
  if (g_opt137_frozen_n_parts[bucket] > 0) {
    return g_opt137_frozen_n_parts[bucket];
  }
  const std::size_t visible = bucket == 2 ? kOpt137RepVisible131072
                            : bucket == 1 ? kOpt137RepVisible33024
                                          : kOpt137RepVisible8448;
  return opt137_parallel_blocks(visible, kOpt137MmaOccupancyPinned, 148);
}

inline bool legal_opt137_n_parts(int n_parts) noexcept {
  return n_parts >= 1 && n_parts <= 256;
}

inline void record_opt137_mma_launch(unsigned int grid_x, unsigned int grid_y,
                                     unsigned int block, int n_parts) noexcept {
  ++g_opt137_mma_launches;
  g_opt137_mma_grid_x = grid_x;
  g_opt137_mma_grid_y = grid_y;
  g_opt137_mma_block = block;
  g_opt137_last_n_parts = n_parts;
}

inline unsigned int last_opt137_mma_launches() noexcept {
  return g_opt137_mma_launches;
}

inline void reset_opt137_mma_launches() noexcept { g_opt137_mma_launches = 0; }

inline void record_opt151_mma_launch(unsigned int grid_x, unsigned int grid_y,
                                     unsigned int block, int n_parts) noexcept {
  ++g_opt151_mma_launches;
  g_opt151_mma_grid_x = grid_x;
  g_opt151_mma_grid_y = grid_y;
  g_opt151_mma_block = block;
  g_opt151_last_n_parts = n_parts;
}

inline unsigned int last_opt151_mma_launches() noexcept {
  return g_opt151_mma_launches;
}

inline void reset_opt151_mma_launches() noexcept { g_opt151_mma_launches = 0; }

inline const char* decode_attention_vec128_path_for_position(
    std::size_t position) noexcept {
  if (opt151_uses_qk_pv_mma_at(position)) {
    return kLegalDecodeAttentionQkPvMmaV2;
  }
  if (opt137_uses_mma_at(position)) {
    return kLegalDecodeAttentionDenseMma;
  }
  if (g_decode_attention_vec128_path_override != nullptr) {
    return g_decode_attention_vec128_path_override;
  }
  const int threshold = effective_decode_attention_crossover_threshold();
  const int verified = effective_decode_attention_verified_max();
  if (threshold > 0 &&
      position >= static_cast<std::size_t>(threshold) &&
      position <= static_cast<std::size_t>(verified)) {
    if (decode_attention_flash_vec_enabled()) {
      return kLegalDecodeAttentionFlashVec;
    }
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

inline bool decode_attention_flash_vec_uses(const char* path) noexcept {
  return path != nullptr &&
         std::strcmp(path, kLegalDecodeAttentionFlashVec) == 0;
}

inline bool decode_attention_flash_vec_at(std::size_t position) noexcept {
  return decode_attention_flash_vec_uses(
      decode_attention_vec128_path_for_position(position));
}

inline bool decode_attention_uses_short_vector_at(
    std::size_t position) noexcept {
  return decode_attention_vec128_uses_online_at(position) ||
         decode_attention_flash_vec_at(position);
}

// Bounded decode-graph topology: warp_query vs vec128_online/flash_vec vs
// OPT-137 MMA buckets. Flash-vec reuses topology 1; it does not add a graph.
inline int decode_graph_topology_index(std::size_t position) noexcept {
  if (opt137_uses_mma_at(position)) {
    return 2 + opt137_bucket_index(position);
  }
  return decode_attention_uses_short_vector_at(position) ? 1 : 0;
}

inline std::size_t decode_graph_topology_capture_position(int topology) noexcept {
  if (topology == 4) return static_cast<std::size_t>(kOpt137Bucket65536);
  if (topology == 3) return static_cast<std::size_t>(kOpt137Bucket32768);
  if (topology == 2) return static_cast<std::size_t>(kOpt137MmaThreshold);
  if (topology == 1) {
    return static_cast<std::size_t>(kSelectedDecodeAttentionCrossoverThreshold);
  }
  return 0;
}

inline int effective_decode_graph_topology_count() noexcept {
  return opt137_mma_enabled() ? kOpt137TopologyCount : 2;
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
  if (decode_attention_flash_vec_uses(path)) {
    return kDecodeAttentionFlashVecLaunch;
  }
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

inline bool apply_decode_attention_verified_max(int verified_max) noexcept {
  if (!legal_decode_attention_verified_max(verified_max)) return false;
  g_decode_attention_verified_max_override = verified_max;
  return true;
}

inline void clear_decode_attention_verified_max_override() noexcept {
  g_decode_attention_verified_max_override = -1;
}

inline bool apply_opt137_dense_mma_ident(const char* path) noexcept {
  if (path != nullptr &&
      std::strcmp(path, kLegalDecodeAttentionDenseMma) == 0) {
    g_opt137_mma_override = true;
    g_opt151_qk_pv_mma_override = false;
    return true;
  }
  if (path != nullptr &&
      std::strcmp(path, kLegalDecodeAttentionHybridCrossover) == 0) {
    g_opt137_mma_override = false;
    g_opt151_qk_pv_mma_override = false;
    return true;
  }
  return false;
}

inline bool apply_opt151_qk_pv_mma_ident(const char* path) noexcept {
  if (path != nullptr &&
      std::strcmp(path, kLegalDecodeAttentionQkPvMmaV2) == 0) {
    g_opt151_qk_pv_mma_override = true;
    return true;
  }
  if (path != nullptr &&
      (std::strcmp(path, kLegalDecodeAttentionDenseMma) == 0 ||
       std::strcmp(path, kLegalDecodeAttentionHybridCrossover) == 0)) {
    g_opt151_qk_pv_mma_override = false;
    return true;
  }
  return false;
}

inline void clear_opt151_qk_pv_mma_ident() noexcept {
  g_opt151_qk_pv_mma_override = false;
}

inline void set_decode_attention_flash_vec_override(bool enabled) noexcept {
  g_decode_attention_flash_vec_override = enabled;
}

inline bool apply_decode_attention_flash_vec_ident(const char* path) noexcept {
  if (path != nullptr &&
      std::strcmp(path, kLegalDecodeAttentionFlashVec) == 0) {
    g_decode_attention_flash_vec_override = true;
    return true;
  }
  if (path != nullptr &&
      std::strcmp(path, kLegalDecodeAttentionHybridCrossover) == 0) {
    g_decode_attention_flash_vec_override = false;
    return true;
  }
  if (path != nullptr &&
      std::strcmp(path, kLegalDecodeAttentionVec128Online) == 0) {
    g_decode_attention_flash_vec_override = false;
    return true;
  }
  return false;
}

inline void clear_decode_attention_flash_vec_override() noexcept {
  g_decode_attention_flash_vec_override = false;
}

inline void clear_opt137_dense_mma_ident() noexcept {
  g_opt137_mma_override = false;
  g_opt151_qk_pv_mma_override = false;
}

inline bool apply_opt130_dense_attention_ident(const char* path) noexcept {
  if (path != nullptr &&
      std::strcmp(path, kLegalDecodeAttentionOccupancyVec128Open) == 0) {
    g_opt137_mma_override = false;
    return apply_decode_attention_verified_max(kOpt130CandidateVerifiedMax) &&
           apply_vec128_n_parts(kOpt130CandidateVec128NParts);
  }
  if (path != nullptr &&
      std::strcmp(path, kLegalDecodeAttentionHybridCrossover) == 0) {
    clear_decode_attention_verified_max_override();
    clear_vec128_n_parts_override();
    g_opt137_mma_override = false;
    g_opt151_qk_pv_mma_override = false;
    g_decode_attention_flash_vec_override = false;
    return true;
  }
  if (apply_opt137_dense_mma_ident(path)) {
    return true;
  }
  if (apply_opt151_qk_pv_mma_ident(path)) {
    return true;
  }
  return false;
}

inline void clear_opt130_dense_attention_ident() noexcept {
  clear_decode_attention_verified_max_override();
  clear_vec128_n_parts_override();
  clear_opt137_dense_mma_ident();
}

inline const char* effective_decode_attention_dispatch_path() noexcept {
  if (opt151_qk_pv_mma_enabled()) {
    return kLegalDecodeAttentionQkPvMmaV2;
  }
  if (opt137_mma_enabled()) {
    return kLegalDecodeAttentionDenseMma;
  }
  if (decode_attention_flash_vec_enabled()) {
    return kLegalDecodeAttentionFlashVec;
  }
  if (decode_attention_vec128_uses_online()) {
    return effective_decode_attention_vec128_path();
  }
  return effective_decode_attention_gqa_path();
}

inline const char* effective_decode_attention_dispatch_path(
    std::size_t position) noexcept {
  if (opt151_uses_qk_pv_mma_at(position)) {
    return kLegalDecodeAttentionQkPvMmaV2;
  }
  if (opt137_uses_mma_at(position)) {
    return kLegalDecodeAttentionDenseMma;
  }
  if (decode_attention_flash_vec_at(position) ||
      decode_attention_vec128_uses_online_at(position)) {
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

struct DecodeAttentionVerifiedMaxScope final {
  explicit DecodeAttentionVerifiedMaxScope(int verified_max) noexcept {
    apply_decode_attention_verified_max(verified_max);
  }
  ~DecodeAttentionVerifiedMaxScope() {
    clear_decode_attention_verified_max_override();
  }
  DecodeAttentionVerifiedMaxScope(const DecodeAttentionVerifiedMaxScope&) =
      delete;
  DecodeAttentionVerifiedMaxScope& operator=(
      const DecodeAttentionVerifiedMaxScope&) = delete;
};

struct Opt130DenseAttentionScope final {
  explicit Opt130DenseAttentionScope(const char* path) noexcept {
    apply_opt130_dense_attention_ident(path);
  }
  ~Opt130DenseAttentionScope() { clear_opt130_dense_attention_ident(); }
  Opt130DenseAttentionScope(const Opt130DenseAttentionScope&) = delete;
  Opt130DenseAttentionScope& operator=(const Opt130DenseAttentionScope&) =
      delete;
};

struct Opt137DenseMmaScope final {
  explicit Opt137DenseMmaScope(const char* path) noexcept {
    apply_opt137_dense_mma_ident(path);
  }
  ~Opt137DenseMmaScope() { clear_opt137_dense_mma_ident(); }
  Opt137DenseMmaScope(const Opt137DenseMmaScope&) = delete;
  Opt137DenseMmaScope& operator=(const Opt137DenseMmaScope&) = delete;
};

struct Opt151QkPvMmaScope final {
  explicit Opt151QkPvMmaScope(const char* path) noexcept
      : previous_(g_opt151_qk_pv_mma_override) {
    apply_opt151_qk_pv_mma_ident(path);
  }
  ~Opt151QkPvMmaScope() { g_opt151_qk_pv_mma_override = previous_; }
  Opt151QkPvMmaScope(const Opt151QkPvMmaScope&) = delete;
  Opt151QkPvMmaScope& operator=(const Opt151QkPvMmaScope&) = delete;

 private:
  bool previous_;
};

struct DecodeAttentionFlashVecScope final {
  explicit DecodeAttentionFlashVecScope(const char* path) noexcept
      : previous_(g_decode_attention_flash_vec_override) {
    apply_decode_attention_flash_vec_ident(path);
  }
  ~DecodeAttentionFlashVecScope() {
    g_decode_attention_flash_vec_override = previous_;
  }
  DecodeAttentionFlashVecScope(const DecodeAttentionFlashVecScope&) = delete;
  DecodeAttentionFlashVecScope& operator=(
      const DecodeAttentionFlashVecScope&) = delete;

 private:
  bool previous_;
};

}  // namespace qw38::cuda
