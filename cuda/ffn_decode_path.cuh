#pragma once

// OPT-049 production decode-FFN gate/up path pin. Host-only.

#include <cstring>

namespace qw38::cuda {

constexpr char kLegalFfnDecodePathSeparate[] = "separate";
constexpr char kLegalFfnDecodePathSharedStage[] = "shared_stage";
constexpr char kLegalFfnDecodePathPaired[] = "paired";
constexpr char kLegalFfnDecodePathPairedStaged[] = "paired_staged";

// Production pin. Keep sitting may switch away from separate; reject restores it.
constexpr char kSelectedFfnDecodePath[] = "paired_staged";

inline thread_local const char* g_ffn_decode_path_override = nullptr;

inline bool legal_ffn_decode_path(const char* path) noexcept {
  return path != nullptr &&
         (std::strcmp(path, kLegalFfnDecodePathSeparate) == 0 ||
          std::strcmp(path, kLegalFfnDecodePathSharedStage) == 0 ||
          std::strcmp(path, kLegalFfnDecodePathPaired) == 0 ||
          std::strcmp(path, kLegalFfnDecodePathPairedStaged) == 0);
}

inline const char* selected_ffn_decode_path() noexcept {
  return kSelectedFfnDecodePath;
}

inline const char* effective_ffn_decode_path() noexcept {
  return g_ffn_decode_path_override != nullptr ? g_ffn_decode_path_override
                                               : kSelectedFfnDecodePath;
}

inline bool ffn_decode_shares_stage() noexcept {
  const char* path = effective_ffn_decode_path();
  return std::strcmp(path, kLegalFfnDecodePathSharedStage) == 0 ||
         std::strcmp(path, kLegalFfnDecodePathPairedStaged) == 0;
}

inline bool ffn_decode_uses_paired() noexcept {
  const char* path = effective_ffn_decode_path();
  return std::strcmp(path, kLegalFfnDecodePathPaired) == 0 ||
         std::strcmp(path, kLegalFfnDecodePathPairedStaged) == 0;
}

inline bool ffn_decode_uses_bf16_activations() noexcept {
  return std::strcmp(effective_ffn_decode_path(), kLegalFfnDecodePathPaired) ==
         0;
}

// OPT-062: actual gate/up/down launch variants and staging counts. Graph
// capture and eager work both write this record; a selector constant is not
// proof that integer Q4 reached every FFN leg.
struct FfnDecodeDispatchRecord final {
  const char* gate_variant = "";
  const char* up_variant = "";
  const char* down_variant = "";
  int gate_up_stage_count = 0;
  int down_stage_count = 0;
  bool captured_in_graph = false;
  const char* q4_path = "";
  const char* ffn_path = "";
  const char* staging = "";
};

inline thread_local FfnDecodeDispatchRecord g_last_ffn_decode_dispatch{};

inline const FfnDecodeDispatchRecord& last_ffn_decode_dispatch() noexcept {
  return g_last_ffn_decode_dispatch;
}

inline void clear_ffn_decode_dispatch() noexcept {
  g_last_ffn_decode_dispatch = FfnDecodeDispatchRecord{};
}

inline void record_ffn_decode_dispatch(const char* gate_variant,
                                       const char* up_variant,
                                       const char* down_variant,
                                       int gate_up_stage_count,
                                       int down_stage_count,
                                       bool captured_in_graph,
                                       const char* q4_path, const char* ffn_path,
                                       const char* staging) noexcept {
  g_last_ffn_decode_dispatch.gate_variant =
      gate_variant != nullptr ? gate_variant : "";
  g_last_ffn_decode_dispatch.up_variant =
      up_variant != nullptr ? up_variant : "";
  g_last_ffn_decode_dispatch.down_variant =
      down_variant != nullptr ? down_variant : "";
  g_last_ffn_decode_dispatch.gate_up_stage_count = gate_up_stage_count;
  g_last_ffn_decode_dispatch.down_stage_count = down_stage_count;
  g_last_ffn_decode_dispatch.captured_in_graph = captured_in_graph;
  g_last_ffn_decode_dispatch.q4_path = q4_path != nullptr ? q4_path : "";
  g_last_ffn_decode_dispatch.ffn_path = ffn_path != nullptr ? ffn_path : "";
  g_last_ffn_decode_dispatch.staging = staging != nullptr ? staging : "";
}

inline void set_ffn_decode_path_override(const char* path) noexcept {
  g_ffn_decode_path_override = path;
}

inline void clear_ffn_decode_path_override() noexcept {
  g_ffn_decode_path_override = nullptr;
}

struct FfnDecodePathScope final {
  explicit FfnDecodePathScope(const char* path) noexcept {
    set_ffn_decode_path_override(path);
  }
  ~FfnDecodePathScope() { clear_ffn_decode_path_override(); }
  FfnDecodePathScope(const FfnDecodePathScope&) = delete;
  FfnDecodePathScope& operator=(const FfnDecodePathScope&) = delete;
};

}  // namespace qw38::cuda
